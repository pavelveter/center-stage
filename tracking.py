"""Camera enumeration, model management and face/zoom geometry math."""

import contextlib
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
    """One semi-implicit step of a damped spring toward target.

    zeta=1.0 is critically damped and does not overshoot.
    """
    accel = omega * omega * (target - pos) - 2.0 * zeta * omega * vel
    vel += accel * dt
    pos += vel * dt
    return pos, vel


def crop_origin(curr_x, curr_y, zoom, in_w, in_h):
    """Top-left corner and size of the crop window, clamped to frame bounds.

    curr_x/curr_y are normalized (0..1) centers of the desired view. Clamping
    guarantees the returned window is never larger than the frame, which also
    keeps cv2.resize from receiving an empty slice.
    """
    crop_w = int(in_w / zoom)
    crop_h = int(in_h / zoom)
    x1 = int(curr_x * in_w - crop_w / 2)
    y1 = int(curr_y * in_h - crop_h / 2)
    x1 = max(0, min(in_w - crop_w, x1))
    y1 = max(0, min(in_h - crop_h, y1))
    return x1, y1, crop_w, crop_h
