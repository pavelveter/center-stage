"""Camera enumeration, model management and face/zoom geometry math."""

import contextlib
import math
import os
import urllib.request

import cv2


# Face detector model is downloaded once and cached in the user's cache dir,
# so the repo stays free of binary assets and subsequent starts are offline.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)
MODEL_DIR = os.path.expanduser("~/.cache/hyper-stage")
MODEL_PATH = os.path.join(MODEL_DIR, "blaze_face_short_range.tflite")

# Detection runs on a small frame: faces are still found reliably while keeping
# preprocessing cost low and stable regardless of the capture resolution.
DETECT_W, DETECT_H = 640, 360


@contextlib.contextmanager
def quiet_native_stderr():
    """Silence MediaPipe's native C++ stderr logs at detector setup and close.

    MediaPipe 1.2+ initializes LiteRT, whose absl logging writes INFO lines
    straight to file descriptor 2; no environment variable suppresses them
    (GLOG_minloglevel and MEDIAPIPE_CPP_LOG_LEVEL are both ignored). Inside
    the Textual TUI they trash the screen. Only detector creation and close
    are covered - milliseconds of native init - so real warnings from the
    rest of the pipeline still reach the terminal.
    """
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


def ensure_model():
    """Return the local model path, downloading it on first use.

    Raises RuntimeError if the download leaves no file behind.
    """
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Downloading face-detector model to {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("model download failed")
    return MODEL_PATH


def list_cameras(max_index=8):
    """Probe camera indices and print which ones open and deliver a frame.

    Returns a list of (index, readable) pairs for opened devices.
    """
    found = []
    for i in range(max_index):
        # Always release each probe so the device is available to the next one.
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        ok = cap.isOpened()
        readable = False
        if ok:
            ret, _ = cap.read()
            readable = bool(ret)
        cap.release()
        if ok:
            found.append((i, readable))
            print(f"  {i}: opened={ok} readable={readable}")
    if not found:
        print("  (no cameras found)")
    return found


DEFAULT_CENTER = (0.5, 0.45)
# BlazeFace keypoints 0 and 1 are the eyes; their distance is a more stable
# distance proxy than bbox height, which changes with head pitch/rotation.
# Typical inter-ocular distance is ~0.36 of face height.
EYE_TO_FACE_HEIGHT = 2.8


def subject_score(detection):
    """Largest-area detection wins; confidence only breaks near-ties."""
    bb = detection.bounding_box
    area = max(0.0, bb.width * bb.height)
    categories = getattr(detection, "categories", None) or []
    score = max((getattr(c, "score", 0.0) or 0.0 for c in categories), default=0.0)
    return area * (0.9 + 0.1 * score)


def pick_subject(detections):
    """One consistent subject for both controllers, or None."""
    return max(detections, key=subject_score) if detections else None


def subject_center(detection, det_w, det_h):
    """Normalized (x, y) center of one detection (chin-biased vertically)."""
    bb = detection.bounding_box
    return (
        (bb.origin_x + bb.width / 2.0) / det_w,
        (bb.origin_y + bb.height * 0.6) / det_h,
    )


def face_size(detection, det_w, det_h):
    """Robust face size as a fraction of frame height, from landmarks if present.

    Inter-ocular distance is largely insensitive to the bbox padding changes
    caused by head rotation and facial expressions. Keypoints are normalized
    to width/height separately, so x is rescaled by the detection aspect
    ratio. Falls back to bbox height when keypoints are absent (e.g. tests).
    Limitation: strong yaw still foreshortens the eyes; filtering and the zoom
    deadband absorb the remainder.
    """
    bb = detection.bounding_box
    fallback = bb.height / det_h
    keypoints = getattr(detection, "keypoints", None) or []
    if len(keypoints) >= 2:
        first, second = keypoints[0], keypoints[1]
        dx = (getattr(first, "x", 0.0) - getattr(second, "x", 0.0)) * det_w / det_h
        dy = getattr(first, "y", 0.0) - getattr(second, "y", 0.0)
        eye = math.hypot(dx, dy)
        if 0.02 < eye < 0.5:
            return max(0.01, min(1.0, eye * EYE_TO_FACE_HEIGHT))
    return fallback


def clamp(value, low, high):
    return max(low, min(high, value))


class PositionController:
    """Center-stage pan: hysteresis on measurements, time-based damped spring.

    Independent of zoom. The deadband is measured from the last *accepted*
    target, never from the moving camera position, so a slowly drifting view
    cannot keep re-arming itself on small measurement noise.
    """

    def __init__(self, x=DEFAULT_CENTER[0], y=DEFAULT_CENTER[1]):
        self.x, self.y = x, y
        self.target_x, self.target_y = x, y
        self.vx = self.vy = 0.0
        self.last_detection = -math.inf
        self.confidence = 0.0
        self.raw_x = self.raw_y = 0.5

    def observe(self, cx, cy, now, deadband, confidence=0.0):
        self.raw_x, self.raw_y = cx, cy
        self.confidence = confidence
        self.last_detection = now
        if abs(cx - self.target_x) > deadband:
            self.target_x = clamp(cx, 0.0, 1.0)
        if abs(cy - self.target_y) > deadband:
            self.target_y = clamp(cy, 0.0, 1.0)

    def relax(self):
        """Forget measurements; ease back to default framing (tracking off)."""
        self.last_detection = -math.inf

    def step(self, dt, now, response_time, damping, timeout):
        if now - self.last_detection > timeout:
            # No fresh detection: drift home slowly instead of snapping.
            k = 1.0 - math.exp(-dt / max(0.2, timeout * 4.0))
            self.target_x += (DEFAULT_CENTER[0] - self.target_x) * k
            self.target_y += (DEFAULT_CENTER[1] - self.target_y) * k
        omega = 2.0 * math.pi / max(0.2, response_time)
        zeta = max(1.0, damping)  # never overshoot
        self.x, self.vx = spring_step(self.x, self.vx, self.target_x, omega, zeta, dt)
        self.y, self.vy = spring_step(self.y, self.vy, self.target_y, omega, zeta, dt)
        return self.x, self.y


class ZoomController:
    """Independent zoom: filtered size, asymmetric time-based response.

    Raw face sizes are rate-limited and low-pass filtered before they can
    influence zoom; the accepted target has a relative deadband. Zoom-in is
    slow and deliberate, zoom-out responsive, both exponential in real time
    so frame-rate changes cannot alter the speed.
    """

    def __init__(self, value=1.0):
        self.value = self.target = value
        self.filtered_size = None
        self.raw_size = 0.0
        self.desired = value
        self.last_measurement = self.last_detection = -math.inf
        self.confidence = 0.0

    def observe(self, size, now, smoothing_time, max_change, confidence=0.0):
        self.raw_size = size
        self.confidence = confidence
        self.last_detection = now
        if self.filtered_size is None:
            self.filtered_size = size
            self.last_measurement = now
            return
        elapsed = max(1e-3, now - self.last_measurement)
        self.last_measurement = now
        # Reject implausible jumps: limit the change per second, not per frame.
        limit = max(0.01, max_change) * elapsed
        delta = size - self.filtered_size
        size = self.filtered_size + clamp(delta, -limit, limit)
        alpha = 1.0 - math.exp(-elapsed / max(0.05, smoothing_time))
        self.filtered_size += (size - self.filtered_size) * alpha

    def relax(self):
        """Forget measurements; ease back to the base zoom (tracking off)."""
        self.last_detection = -math.inf

    def step(self, dt, now, base, max_head, head_ratio, deadband,
             zoom_in_time, zoom_out_time, timeout):
        base = max(1.0, base)
        if self.filtered_size is None or now - self.last_detection > timeout:
            self.desired = base
        else:
            self.desired = fit_zoom(self.filtered_size, base, max_head, head_ratio)
        if abs(self.desired - self.target) > self.target * deadband:
            self.target = self.desired
        self.target = clamp(self.target, 1.0, base)
        tau = zoom_in_time if self.target > self.value else zoom_out_time
        alpha = 1.0 - math.exp(-dt / max(0.05, tau))
        self.value += (self.target - self.value) * alpha
        # Only the lower bound is hard: if the base zoom is lowered live,
        # ease down to it instead of snapping the published frame.
        self.value = max(1.0, self.value)
        return self.value


def face_centers(detections, det_w, det_h):
    """Normalized (x, y) centers for each detection in the detection frame.

    The vertical center is biased toward the chin so the crop leaves headroom
    for hair instead of centering on the eyes.
    """
    centers = []
    for d in detections:
        bb = d.bounding_box
        cx = (bb.origin_x + bb.width / 2.0) / det_w
        cy = (bb.origin_y + bb.height * 0.6) / det_h
        centers.append((cx, cy))
    return centers


def largest_face_height(detections, det_h):
    """Height of the largest (nearest) face as a fraction of frame height; 0 if none."""
    return max((d.bounding_box.height / det_h for d in detections), default=0.0)


def fit_zoom(face_h, base_zoom, max_head, head_ratio):
    """Max zoom such that the head is no taller than max_head of the output frame height.

    face_h is the face bbox height as a fraction of the input frame height. In the
    output the head occupies head_ratio * face_h * zoom, so zoom <= max_head /
    (head_ratio * face_h). Never zoom out below a full frame (zoom 1.0).
    """
    if face_h <= 0:
        return base_zoom
    limit = max_head / (head_ratio * face_h)
    return max(min(1.0, base_zoom), min(base_zoom, limit))


def pick_target_zoom(target, wanted, deadzone):
    """Apply zoom-out immediately; apply zoom-in only when it is noticeable (anti-jitter)."""
    if wanted < target or wanted > target * (1.0 + deadzone):
        return wanted
    return target


def spring_step(pos, vel, target, omega, zeta, dt):
    """Damped spring step, stable across variable/large frame intervals.

    Semi-implicit Euler is only stable while omega*dt < 2, so long frames are
    split into bounded substeps. With zeta >= 1 the motion is critically
    damped (no overshoot), and the result depends on elapsed time, not FPS.
    """
    if dt <= 0.0:
        return pos, vel
    dt = min(dt, 0.5)  # a long stall must not fast-forward the camera
    max_step = min(1.0 / 120.0, 0.3 / max(0.1, omega))
    steps = max(1, math.ceil(dt / max_step))
    sub = dt / steps
    for _ in range(steps):
        accel = omega * omega * (target - pos) - 2.0 * zeta * omega * vel
        vel += accel * sub
        pos += vel * sub
    return pos, vel


def crop_origin(curr_x, curr_y, zoom, in_w, in_h, out_w=None, out_h=None):
    """Top-left corner and size of the crop window, clamped to frame bounds.

    curr_x/curr_y are normalized (0..1) centers of the desired view. When an
    output size is given, the crop keeps that aspect ratio so the final
    resize never distorts the picture (portrait input -> landscape output is
    the common case). With no output size the input aspect is kept, as before.
    """
    zoom = max(1.0, zoom)
    aspect = (out_w / out_h) if out_w and out_h else (in_w / in_h)
    base_h = min(in_h, in_w / aspect)
    crop_h = max(1, min(in_h, round(base_h / zoom)))
    crop_w = max(1, min(in_w, round(crop_h * aspect)))
    x1 = int(curr_x * in_w - crop_w / 2)
    y1 = int(curr_y * in_h - crop_h / 2)
    x1 = max(0, min(in_w - crop_w, x1))
    y1 = max(0, min(in_h - crop_h, y1))
    return x1, y1, crop_w, crop_h
