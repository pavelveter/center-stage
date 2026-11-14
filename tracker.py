"""Apple-style Center Stage for a regular webcam.

Captures a high-res frame, tracks face(s) with MediaPipe, and publishes a
cropped/zoomed view to a virtual camera so video-call apps see a framed shot.

The video pipeline runs in a daemon thread while tui.py renders a Textual
control panel. Settings come from an INI file (see center-stage.ini), can be
changed live in the TUI, and are written back when they change.
"""

import argparse
import collections
import configparser
import glob
import math
import os
import sys
import tempfile
import threading
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Defaults used when the INI file omits a key or no config is found. Keeping
# them in one dict makes config loading and CLI overrides uniform: every value
# flows through the same validate/override path.
DEFAULTS = {
    "camera_index": 1,          # AVFoundation device index (0 = FaceTime on macOS)
    "in_w": 2560,               # requested capture width
    "in_h": 1440,               # requested capture height
    "out_w": 1920,              # published virtual-camera width
    "out_h": 1080,              # published virtual-camera height
    "fps": 30,                  # capture and virtual-camera frame rate
    "tracking_enabled": True,   # face tracking on/off (off = base zoom, centered)
    "zoom": 1.5,                # base crop factor (higher = tighter framing)
    "max_head_height": 0.667,   # max head height as a fraction of frame height
    "head_ratio": 1.35,         # full head / face detector bbox
    "zoom_response_time": 0.9,  # spring response time (s); higher = smoother zoom
    "zoom_damping_ratio": 1.0,  # 1.0 = critically damped (no overshoot)
    "zoom_deadzone": 0.05,      # hysteresis on zoom-in, fraction
    "smoothing": 0.08,          # EMA alpha for pan (lower = smoother, more lag)
    "deadzone": 0.04,           # ignore pan motion smaller than this fraction
    "min_det_conf": 0.6,        # MediaPipe face-detection confidence floor
    "detect_every": 3,          # run detection every N frames (CPU vs latency)
    "lut_enabled": True,        # toggle color grading at runtime
    "lut_file": "",             # path to a .cube 3D LUT (empty = no grading)
    "lut_scale": 8,             # LUT downscale factor (higher = faster, softer)
    "lut_detail": True,         # upscale only the color delta to keep sharpness
    "sharpen_enabled": False,   # toggle edge-aware sharpening
    "sharpen_strength": 0.3,    # unsharp-mask amount (0..1 looks natural)
    "sharpen_threshold": 10,    # min local contrast before sharpening kicks in
    "sharpen_sigma": 2.0,       # Gaussian blur radius for the high-pass filter
    "time_loop_enabled": False, # replay the last second of processed frames
    "deep_fry_enabled": False,  # high-contrast saturation + RGB channel roll
    "magic_loop_enabled": False,  # AFK: play a seamless loop of recent video
}

# Maps INI sections/keys (human-facing) to config keys (code-facing). Both
# load_config and save_config use this single table, so reading and writing
# settings can never drift apart.
INI_MAP = {
    "camera": {"index": "camera_index"},
    "capture": {
        "in_width": "in_w",
        "in_height": "in_h",
        "out_width": "out_w",
        "out_height": "out_h",
        "fps": "fps",
    },
    "tracking": {
        "enabled": "tracking_enabled",
        "zoom": "zoom",
        "smoothing": "smoothing",
        "max_head_height": "max_head_height",
        "head_ratio": "head_ratio",
        "zoom_response_time": "zoom_response_time",
        "zoom_damping_ratio": "zoom_damping_ratio",
        "zoom_deadzone": "zoom_deadzone",
        "deadzone": "deadzone",
        "min_detection_confidence": "min_det_conf",
        "detect_every": "detect_every",
    },
    "color": {
        "lut_enabled": "lut_enabled",
        "lut_file": "lut_file",
        "lut_scale": "lut_scale",
        "lut_detail": "lut_detail",
    },
    "sharpen": {
        "enabled": "sharpen_enabled",
        "strength": "sharpen_strength",
        "threshold": "sharpen_threshold",
        "sigma": "sharpen_sigma",
    },
    "effects": {
        "time_loop": "time_loop_enabled",
        "deep_fry": "deep_fry_enabled",
        "magic_loop": "magic_loop_enabled",
    },
}

# One second of processed frames at the nominal 30 FPS.
TIME_LOOP_FRAMES = 30

# Magic Loop (AFK) history: ten seconds of encoded processed frames. Frames
# are stored JPEG-compressed at reduced size; raw 1080p would need ~1.9 GB for
# 300 frames, this stays around 100 MB and the encode fits the frame budget.
MAGIC_LOOP_FRAMES = 300
MAGIC_LOOP_W, MAGIC_LOOP_H = 960, 540
MAGIC_LOOP_JPEG_QUALITY = 85
# Crossfade length in frames at the splice between loop end and start.
MAGIC_LOOP_FADE = 12
# Need three seconds of history before a loop can be built at all.
MAGIC_LOOP_MIN_FRAMES = 90
# The loop itself must be 2-5 seconds: short enough segments look like a
# nervous stutter, long ones stop feeling "frozen".
MAGIC_LOOP_MIN_LOOP_FRAMES = 60
MAGIC_LOOP_MAX_LOOP_FRAMES = 150
MAGIC_LOOP_LENGTH_STEP = 10
# Seconds of dissolve when switching from the live feed into the loop.
MAGIC_LOOP_ENTRY_FADE_SECONDS = 0.5

# Face detector model is downloaded once and cached in the user's cache dir,
# so the repo stays free of binary assets and subsequent starts are offline.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)
MODEL_DIR = os.path.expanduser("~/.cache/center-stage")
MODEL_PATH = os.path.join(MODEL_DIR, "blaze_face_short_range.tflite")

# Detection runs on a small frame: faces are still found reliably while keeping
# preprocessing cost low and stable regardless of the capture resolution.
DETECT_W, DETECT_H = 640, 360


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


def load_config(path):
    """Load an INI file into a flat dict of settings.

    Unknown sections/keys are ignored and missing keys fall back to DEFAULTS.
    Values are coerced to the type of their default, so a malformed entry logs a
    warning and keeps the safe default instead of crashing at runtime.
    """
    cfg = dict(DEFAULTS)
    if not path or not os.path.exists(path):
        return cfg
    parser = configparser.ConfigParser(interpolation=None)
    # A missing/unreadable file simply leaves every default in place.
    parser.read(path)
    for section, keys in INI_MAP.items():
        if not parser.has_section(section):
            continue
        for ini_key, cfg_key in keys.items():
            if parser.has_option(section, ini_key):
                raw = parser.get(section, ini_key).strip()
                default = DEFAULTS[cfg_key]
                # Type is inferred from the default: a string default stays a
                # string, bools use getboolean, ints/floats parse numerically.
                if isinstance(default, str):
                    cfg[cfg_key] = raw
                else:
                    try:
                        if isinstance(default, bool):
                            cfg[cfg_key] = parser.getboolean(section, ini_key)
                        elif isinstance(default, int):
                            cfg[cfg_key] = int(float(raw))
                        else:
                            cfg[cfg_key] = float(raw)
                    except ValueError:
                        print(
                            f"warning: invalid value for {section}.{ini_key}={raw!r}; using default {default}"
                        )
    return cfg


def ini_value(value):
    """Format a Python value the way load_config expects to read it back."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def portable_path(value, base_dir):
    """Return a path relative to base_dir when it lives below it, else absolute.

    Keeps the saved INI usable after the project folder moves, without turning
    an intentionally external LUT path into something relative and broken.
    """
    if not value:
        return value
    absolute = os.path.abspath(value)
    try:
        relative = os.path.relpath(absolute, base_dir)
    except ValueError:
        return absolute
    if relative == ".." or relative.startswith(".." + os.sep):
        return absolute
    return relative


def save_config(path, cfg):
    """Atomically write runtime settings back to the INI file.

    Sections and keys already in the file but unknown to this program are kept;
    every key listed in INI_MAP is overwritten. Writing to a temporary file in
    the same directory and moving it into place with os.replace guarantees a
    crash mid-write can never truncate the existing config.
    """
    directory = os.path.dirname(os.path.abspath(path))
    parser = configparser.ConfigParser(interpolation=None)
    if os.path.exists(path):
        parser.read(path)
    for section, keys in INI_MAP.items():
        if not parser.has_section(section):
            parser.add_section(section)
        for ini_key, cfg_key in keys.items():
            value = cfg[cfg_key]
            if cfg_key == "lut_file":
                value = portable_path(value, directory)
            parser.set(section, ini_key, ini_value(value))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".center-stage-", suffix=".ini", dir=directory
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.write("; Center Stage configuration.\n")
            f.write("; Updated automatically when settings change in the TUI.\n\n")
            parser.write(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def find_config_path(cli_path):
    """Resolve the config path: explicit CLI value, then local, then user config."""
    if cli_path:
        return cli_path
    for candidate in (
        "center-stage.ini",
        os.path.expanduser("~/.config/center-stage.ini"),
    ):
        if os.path.exists(candidate):
            return candidate
    return None


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


def load_lut(path):
    """Parse a .cube 3D LUT into a (N,N,N,3) float32 array (axes r,g,b)."""
    if not path:
        return None
    if not os.path.exists(path):
        print(f"warning: LUT file not found: {path}")
        return None
    size = None
    data = []
    # .cube files are text: comments (#), optional TITLE/DOMAIN headers, a size
    # declaration, then size**3 RGB triples. Only the triples are consumed.
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.upper().startswith("TITLE") or line.upper().startswith("DOMAIN_"):
                continue
            if line.upper().startswith("LUT_3D_SIZE"):
                size = int(line.split()[-1])
                continue
            if line.upper().startswith("LUT_1D_SIZE"):
                print("warning: 1D LUTs are not supported; ignoring")
                continue
            parts = line.split()
            if len(parts) == 3:
                try:
                    data.append((float(parts[0]), float(parts[1]), float(parts[2])))
                except ValueError:
                    pass
    if size is None or len(data) != size * size * size:
        print(f"warning: invalid LUT {path}: size={size} entries={len(data)}")
        return None
    # file order is R fastest -> C-order (b,g,r); transpose to (r,g,b)
    lut = np.array(data, dtype=np.float32).reshape(size, size, size, 3)
    lut = np.transpose(lut, (2, 1, 0, 3))
    return lut


def bake_lut(lut, grid=128):
    """Bake once at startup: (N,N,N,3) float (r,g,b) -> flat uint8 BGR table.

    Trilinear interpolation happens here, not per frame. At runtime only one
    take() on the precomputed 128^3 grid (7 bits per channel) with no float math.
    """
    if lut is None:
        return None
    assert grid == 128, "apply_lut indexing assumes 7 bits per channel"
    N = lut.shape[0]
    pos = np.linspace(0, N - 1, grid, dtype=np.float32)
    i0 = np.floor(pos).astype(np.int32)
    i1 = np.minimum(i0 + 1, N - 1)
    w = (pos - i0).astype(np.float32)
    o = lut.astype(np.float32)
    # separate per-axis linear interpolation = trilinear
    o = o[i0] * (1 - w)[:, None, None, None] + o[i1] * w[:, None, None, None]
    o = o[:, i0] * (1 - w)[None, :, None, None] + o[:, i1] * w[None, :, None, None]
    o = (
        o[:, :, i0] * (1 - w)[None, None, :, None]
        + o[:, :, i1] * w[None, None, :, None]
    )
    table = np.clip(o * 255.0 + 0.5, 0, 255).astype(np.uint8)
    table = table[..., ::-1]  # LUT yields RGB, frame and virtual camera are BGR
    return np.ascontiguousarray(table.reshape(-1, 3))


def apply_lut(frame_bgr, table, scale=8, detail=True):
    """Fast LUT application.

    Color is computed on a downscaled copy (1/scale), then one of two modes:
      detail=True  - only the correction (graded - orig) is upscaled and added to
                     the original: frame sharpness is preserved;
      detail=False - the graded frame itself is upscaled (faster, but softer).
    """
    if table is None:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    scale = max(2, int(scale))
    scale = min(scale, max(2, w // 4), max(2, h // 4))

    # cheap downscale: 2x2-box + bilinear (much faster than INTER_AREA)
    half = cv2.resize(frame_bgr, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
    small = cv2.resize(half, (w // scale, h // scale), interpolation=cv2.INTER_LINEAR)

    # integer index into the 128^3 table: (r<<14)|(g<<7)|b, frame is BGR
    c = (small >> 1).astype(np.int32)
    idx = (c[..., 2] << 14) | (c[..., 1] << 7) | c[..., 0]
    graded = table.take(idx, axis=0)

    if not detail:
        return cv2.resize(graded, (w, h), interpolation=cv2.INTER_LINEAR)

    # signed correction stored as uint8 with offset 128 (range +-127)
    delta = cv2.subtract(graded, small, dtype=cv2.CV_16S)
    delta8 = np.clip(delta + 128, 0, 255).astype(np.uint8)
    delta8 = cv2.resize(delta8, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(frame_bgr, 1.0, delta8, 1.0, -128.0)


def apply_smart_sharpen(frame_bgr, strength=0.3, threshold=10, sigma=2.0):
    """Edge-aware unsharp mask.

    Boosts high-frequency detail (contours, texture) while leaving flat areas
    such as skin untouched: only pixels whose local contrast exceeds threshold
    are sharpened. Working in signed int16 preserves both light and dark edges.
    """
    if strength <= 0:
        return frame_bgr

    blurred = cv2.GaussianBlur(frame_bgr, (0, 0), sigmaX=float(sigma))
    frame_i16 = frame_bgr.astype(np.int16)
    detail = cv2.subtract(frame_i16, blurred.astype(np.int16))
    sharpened = cv2.addWeighted(frame_i16, 1.0, detail, float(strength), 0.0)

    # Gate on local contrast so digital noise on smooth skin is not amplified.
    mask = cv2.absdiff(frame_bgr, blurred) > threshold
    out = np.where(mask, sharpened, frame_i16)
    return np.clip(out, 0, 255).astype(np.uint8)


def resolve_lut_path(path, base_dir=None):
    """Resolve a LUT path relative to the config dir, cwd, or as-is.

    Returns None for an empty path; otherwise the first existing candidate or
    the original path so the caller can report a clear "not found" warning.
    """
    if not path:
        return None
    if os.path.isabs(path) and os.path.exists(path):
        return path
    candidates = []
    if base_dir:
        candidates.append(os.path.join(base_dir, path))
    candidates.append(path)
    candidates.append(os.path.join(os.getcwd(), path))
    for c in candidates:
        if os.path.exists(c):
            return c
    return path


def list_lut_files(primary_path):
    """Return sorted, de-duplicated .cube files usable as LUT choices.

    Looks next to the configured LUT, in the current directory, and in LUT/,
    so files dropped in either conventional place show up in the TUI list.
    """
    folders = []
    if primary_path:
        folders.append(os.path.dirname(os.path.abspath(primary_path)))
    folders.append(os.getcwd())
    folders.append(os.path.join(os.getcwd(), "LUT"))
    found = {}
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for path in glob.glob(os.path.join(folder, "*.cube")):
            found.setdefault(os.path.abspath(path), None)
    return sorted(found)


def cycle_index(current, total, step):
    """Wrap a list index forward/backward, safe for an empty list."""
    if total <= 0:
        return 0
    return (current + step) % total


class SharedState:
    """Thread-safe runtime state shared by the TUI and the video thread.

    Tunable values and telemetry live behind one lock. The baked LUT table is
    kept apart: the UI's background worker swaps it in while the video thread
    picks it up on the next frame. A revision counter tracks unsaved changes so
    the periodic autosave can never drop an edit made while it was writing.
    """

    TUNABLE = (
        "tracking_enabled",
        "zoom",
        "smoothing",
        "lut_enabled",
        "lut_file",
        "lut_scale",
        "sharpen_enabled",
        "sharpen_strength",
        "time_loop_enabled",
        "deep_fry_enabled",
        "magic_loop_enabled",
    )

    def __init__(self, cfg):
        self._lock = threading.Lock()
        self._values = {key: cfg[key] for key in self.TUNABLE}
        self._telemetry = {
            "fps": 0.0,
            "zoom": float(cfg["zoom"]),
            "tracking": False,
            "device": "starting",
            "status": "starting",
        }
        self._lut_table = None
        self._revision = 0
        self._saved_revision = 0

    def snapshot(self):
        """Copy of the tunable values for one frame of the video thread."""
        with self._lock:
            return dict(self._values)

    def update(self, **values):
        """Set tunable values and mark the config dirty."""
        unknown = set(values) - set(self.TUNABLE)
        if unknown:
            raise KeyError(f"unknown tunable setting: {sorted(unknown)[0]}")
        with self._lock:
            for key, value in values.items():
                if self._values[key] != value:
                    self._values[key] = value
                    self._revision += 1

    def config(self, base):
        """Copy of base config with the current tunable values applied."""
        merged = dict(base)
        with self._lock:
            merged.update(self._values)
        return merged

    def telemetry(self):
        """Copy of the read-only counters published by the video thread."""
        with self._lock:
            return dict(self._telemetry)

    def set_telemetry(self, **values):
        """Publish counters for the UI; never marks the config dirty."""
        with self._lock:
            self._telemetry.update(values)

    @property
    def lut_table(self):
        """Currently active baked LUT table, or None."""
        with self._lock:
            return self._lut_table

    def set_lut_table(self, table):
        """Replace the active baked LUT table (cheap; consumes a reference)."""
        with self._lock:
            self._lut_table = table

    def pending_revision(self):
        """Revision to save, or None when everything is already persisted."""
        with self._lock:
            if self._revision == self._saved_revision:
                return None
            return self._revision

    def mark_saved(self, revision):
        """Record a successful save without hiding edits made during the write."""
        with self._lock:
            if revision == self._revision:
                self._saved_revision = revision


class TimeLoop:
    """One-second replay buffer of already processed frames.

    While disabled the buffer records live output. While enabled the recorded
    frames are replayed in order and wrap around, so the published stream
    freezes into a seamless loop without re-running the image pipeline.
    """

    def __init__(self, length=TIME_LOOP_FRAMES):
        self._frames = collections.deque(maxlen=length)
        self._index = 0

    @property
    def ready(self):
        """True once at least one processed frame has been recorded."""
        return bool(self._frames)

    def record(self, frame):
        """Append a live frame to the buffer and return it for publishing."""
        self._index = 0
        self._frames.append(frame)
        return frame

    def next_frame(self):
        """Return the next buffered frame; the caller must check ready first."""
        frame = self._frames[self._index % len(self._frames)]
        self._index += 1
        return frame


class MagicLoopBuffer:
    """Rolling history of encoded processed frames for the AFK Magic Loop.

    Every processed output frame is stored JPEG-compressed at reduced size
    together with a tiny grayscale thumbnail. The thumbnail makes the calm
    window search nearly free; the JPEGs keep ten seconds of history around
    100 MB instead of the ~1.9 GB raw frames would need.
    """

    THUMB_W, THUMB_H = 160, 90

    def __init__(self, length=MAGIC_LOOP_FRAMES):
        self._frames = collections.deque(maxlen=length)

    def record(self, frame):
        """Encode one processed frame into the history (best effort)."""
        small = cv2.resize(frame, (MAGIC_LOOP_W, MAGIC_LOOP_H))
        ok, encoded = cv2.imencode(
            ".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, MAGIC_LOOP_JPEG_QUALITY]
        )
        if not ok:
            return
        thumb = cv2.cvtColor(
            cv2.resize(small, (self.THUMB_W, self.THUMB_H)), cv2.COLOR_BGR2GRAY
        )
        self._frames.append((encoded.tobytes(), thumb))

    def build_loop(self):
        """Find the calmest 2-5 s window and return a MagicLoopPlayer.

        Each candidate window is scored by its internal motion (mean
        difference between consecutive thumbnails, so gestures, blinks and
        other transitions raise the score) plus the wrap difference between
        its last and first frame, which is the visible jump at the splice.
        The quietest window wins, so the loop plays a moment where the user
        sits still and breathes naturally. Returns None while there is not
        enough history yet.
        """
        frames = list(self._frames)
        n = len(frames)
        if n < MAGIC_LOOP_MIN_FRAMES:
            return None
        thumbs = [thumb for _, thumb in frames]
        # motion[i] = difference between frame i and i+1; prefix sums turn
        # any window's mean motion into an O(1) lookup.
        motion = [
            float(np.mean(cv2.absdiff(thumbs[i], thumbs[i + 1])))
            for i in range(n - 1)
        ]
        prefix = [0.0]
        for value in motion:
            prefix.append(prefix[-1] + value)

        def window_motion(start, length):
            return (prefix[start + length - 1] - prefix[start]) / (length - 1)

        max_len = min(MAGIC_LOOP_MAX_LOOP_FRAMES, n)
        best = None  # (score, start, length)
        # ponytail: ~2000 thumbnail diffs per activation only (~50 ms); use
        # a stride over starts if that ever becomes visible.
        for length in range(
            MAGIC_LOOP_MIN_LOOP_FRAMES, max_len + 1, MAGIC_LOOP_LENGTH_STEP
        ):
            for start in range(0, n - length + 1):
                wrap = float(
                    np.mean(
                        cv2.absdiff(thumbs[start], thumbs[start + length - 1])
                    )
                )
                score = window_motion(start, length) + wrap
                if best is None or score < best[0]:
                    best = (score, start, length)
        if best is None:
            return None
        _, start, length = best
        return MagicLoopPlayer([jpeg for jpeg, _ in frames[start : start + length]])


class MagicLoopPlayer:
    """Plays a fixed seamless loop with a short crossfade at the splice.

    The last MAGIC_LOOP_FADE frames of the loop are alpha-blended with the
    frames right after the start, so wrapping from the end back to the head
    never produces a visible jump. The blended head frames are skipped on
    later passes because they were already shown through the fade.
    """

    def __init__(self, encoded, fade=MAGIC_LOOP_FADE):
        self._frames = encoded
        # A loop too short to fade would blend a frame with itself.
        self._fade = max(0, min(fade, len(encoded) - 2))
        self._index = 0

    def _decode(self, i, out_size):
        jpeg = cv2.imdecode(
            np.frombuffer(self._frames[i], np.uint8), cv2.IMREAD_COLOR
        )
        return cv2.resize(jpeg, out_size, interpolation=cv2.INTER_LINEAR)

    def next_frame(self, out_size):
        """Return the next loop frame resized to out_size (w, h)."""
        n = len(self._frames)
        fade = self._fade
        i = self._index
        if fade and i >= n - fade:
            k = i - (n - fade)
            alpha = (k + 1) / fade
            frame = cv2.addWeighted(
                self._decode(i, out_size),
                1.0 - alpha,
                self._decode(k + 1, out_size),
                alpha,
                0.0,
            )
        else:
            frame = self._decode(i, out_size)
        if i + 1 >= n:
            # The crossfade already revealed frames 1..fade, so later passes
            # resume right after them; without a fade, wrap to the start.
            self._index = fade + 1 if fade else 0
        else:
            self._index = i + 1
        return np.ascontiguousarray(frame)


def apply_deep_fry(frame_bgr, saturation=2.0, contrast=1.3, shift=10):
    """Exaggerated saturation/contrast plus an RGB channel roll.

    The roll happens on the red and blue planes in opposite directions, which
    produces the offset fringes of the classic over-processed "deep fried" look.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(
        hsv[..., 1].astype(np.float32) * float(saturation), 0, 255
    ).astype(np.uint8)
    graded = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    graded = np.clip(
        (graded.astype(np.float32) - 128.0) * float(contrast) + 128.0, 0, 255
    ).astype(np.uint8)
    blue, green, red = cv2.split(graded)
    red = np.roll(red, int(shift), axis=1)
    blue = np.roll(blue, -int(shift), axis=1)
    return cv2.merge((blue, green, red))


def build_lut_table(path):
    """Load and bake a .cube LUT into the runtime table; None when unusable."""
    lut = load_lut(path)
    return bake_lut(lut) if lut is not None else None


class VideoPipeline(threading.Thread):
    """Capture, detect, crop, grade and publish, isolated from the TUI thread.

    The detector, capture device and virtual camera are created and used only
    inside this daemon thread, preserving their thread affinity. All shared
    values are read from SharedState, so the TUI can change settings at any time
    without blocking or dropping frames.
    """

    def __init__(self, cfg, state, preview=False):
        super().__init__(name="center-stage-video", daemon=True)
        self.cfg = cfg
        self.state = state
        self.preview = preview
        self._stop_event = threading.Event()

    def stop(self):
        """Ask the loop to exit; the thread finishes the current frame first."""
        self._stop_event.set()

    def run(self):
        """Open the devices, run the frame loop, and always release resources."""
        cfg = self.cfg
        cap = None
        try:
            model_path = ensure_model()
            options = vision.FaceDetectorOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                min_detection_confidence=cfg["min_det_conf"],
            )
            # Single-threaded OpenCV avoids occasional macOS crashes when its
            # worker threads compete with MediaPipe's thread pool.
            cv2.setNumThreads(1)

            cap = cv2.VideoCapture(cfg["camera_index"], cv2.CAP_AVFOUNDATION)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["in_w"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["in_h"])
            cap.set(cv2.CAP_PROP_FPS, cfg["fps"])
            if not cap.isOpened():
                self.state.set_telemetry(
                    device="none",
                    status=f"cannot open camera {cfg['camera_index']}; "
                    "run --list-cameras to see indices",
                )
                return

            with (
                vision.FaceDetector.create_from_options(options) as detector,
                pyvirtualcam.Camera(
                    width=cfg["out_w"],
                    height=cfg["out_h"],
                    fps=int(cfg["fps"]),
                    fmt=pyvirtualcam.PixelFormat.BGR,
                ) as cam,
            ):
                self.state.set_telemetry(device=cam.device, status="running")
                self._loop(cap, detector, cam)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            self.state.set_telemetry(status=f"stopped: {exc}")
        finally:
            if cap is not None:
                cap.release()
            if self.preview:
                cv2.destroyAllWindows()
            self.state.set_telemetry(fps=0.0, tracking=False)

    def _loop(self, cap, detector, cam):
        """Frame loop; returns when stopped, when the camera dies, or on a fault."""
        cfg = self.cfg
        out_w, out_h = cfg["out_w"], cfg["out_h"]
        dt = 1.0 / max(1.0, float(cfg["fps"]))
        omega = 2.0 * math.pi / max(0.05, float(cfg["zoom_response_time"]))
        zeta = max(0.1, float(cfg["zoom_damping_ratio"]))
        detect_every = max(1, int(cfg["detect_every"]))
        deadzone = float(cfg["deadzone"])
        zoom_deadzone = float(cfg["zoom_deadzone"])
        max_head = float(cfg["max_head_height"])
        head_ratio = float(cfg["head_ratio"])

        curr_x, curr_y = 0.5, 0.45
        target_x, target_y = curr_x, curr_y
        curr_zoom = target_zoom = max(1.0, float(cfg["zoom"]))
        zoom_vel = 0.0
        frame_idx = 0
        consecutive_errors = 0
        fps_ema = 0.0
        last_time = time.monotonic()
        time_loop = TimeLoop()
        magic_loop = MagicLoopBuffer()
        magic_player = None
        magic_on_prev = False
        magic_entry_fade = max(1, int(cfg["fps"] * MAGIC_LOOP_ENTRY_FADE_SECONDS))
        magic_entry_left = 0

        while not self._stop_event.is_set():
            try:
                ret, frame = cap.read()

                # A camera can briefly hand back None or a malformed buffer.
                # Skip it rather than letting downstream OpenCV calls crash.
                if (
                    not ret
                    or frame is None
                    or not isinstance(frame, np.ndarray)
                    or frame.ndim != 3
                    or frame.shape[2] != 3
                ):
                    consecutive_errors += 1
                    if consecutive_errors > 30:
                        self.state.set_telemetry(
                            status="camera stopped delivering frames"
                        )
                        return
                    continue
                frame = np.ascontiguousarray(frame)
                consecutive_errors = 0
                state = self.state.snapshot()
                base_zoom = max(1.0, float(state["zoom"]))

                magic_on = bool(state["magic_loop_enabled"])
                if magic_on != magic_on_prev:
                    magic_on_prev = magic_on
                    # Activation freezes the current framing: replaying skips
                    # the whole pipeline, so tracking and zoom simply stop.
                    magic_player = magic_loop.build_loop() if magic_on else None
                    magic_entry_left = magic_entry_fade if magic_on else 0
                if magic_on and magic_player is None:
                    # History still building up; retry until there is enough.
                    player = magic_loop.build_loop()
                    if player is not None:
                        magic_player = player
                        magic_entry_left = magic_entry_fade

                magic_active = magic_on and magic_player is not None
                looping = bool(state["time_loop_enabled"]) and time_loop.ready
                if magic_active and magic_entry_left <= 0:
                    # Steady state: replay the frozen loop.
                    out_frame = magic_player.next_frame((out_w, out_h))
                elif looping and not magic_active:
                    # Replaying the buffer skips the whole image pipeline.
                    out_frame = time_loop.next_frame()
                    # Replaying the buffer skips the whole image pipeline.
                    out_frame = time_loop.next_frame()
                else:
                    if not state["tracking_enabled"]:
                        # Disabled: ease the view back to the base framing.
                        target_x, target_y, target_zoom = 0.5, 0.45, base_zoom
                        self.state.set_telemetry(tracking=False)
                    elif frame_idx % detect_every == 0:
                        small = cv2.resize(frame, (DETECT_W, DETECT_H))
                        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        result = detector.detect(mp_image)
                        centers = face_centers(result.detections, DETECT_W, DETECT_H)
                        if centers:
                            tx = sum(c[0] for c in centers) / len(centers)
                            ty = sum(c[1] for c in centers) / len(centers)
                            if abs(tx - curr_x) > deadzone:
                                target_x = tx
                            if abs(ty - curr_y) > deadzone:
                                target_y = ty
                            wanted = fit_zoom(
                                largest_face_height(result.detections, DETECT_H),
                                base_zoom,
                                max_head,
                                head_ratio,
                            )
                            target_zoom = pick_target_zoom(
                                target_zoom, wanted, zoom_deadzone
                            )
                        self.state.set_telemetry(tracking=bool(centers))

                    # Exponential moving average smooths panning even though
                    # detection only updates every detect_every frames.
                    smoothing = float(state["smoothing"])
                    curr_x += (target_x - curr_x) * smoothing
                    curr_y += (target_y - curr_y) * smoothing

                    # Dolly zoom: damped spring, then clamped to the valid range.
                    curr_zoom, zoom_vel = spring_step(
                        curr_zoom,
                        zoom_vel,
                        max(1.0, min(base_zoom, target_zoom)),
                        omega,
                        zeta,
                        dt,
                    )
                    curr_zoom = max(1.0, min(base_zoom, curr_zoom))

                    # Use actual frame dimensions: the camera may negotiate a
                    # different resolution than requested.
                    fh, fw = frame.shape[:2]
                    x1, y1, crop_w, crop_h = crop_origin(
                        curr_x, curr_y, curr_zoom, fw, fh
                    )
                    cropped = frame[y1 : y1 + crop_h, x1 : x1 + crop_w]
                    if crop_w < 1 or crop_h < 1 or cropped.size == 0:
                        continue
                    out_frame = cv2.resize(
                        np.ascontiguousarray(cropped),
                        (out_w, out_h),
                        interpolation=cv2.INTER_LINEAR,
                    )

                    table = self.state.lut_table
                    if state["lut_enabled"] and table is not None:
                        out_frame = apply_lut(
                            out_frame,
                            table,
                            scale=int(state["lut_scale"]),
                            detail=bool(cfg["lut_detail"]),
                        )
                    strength = float(state["sharpen_strength"])
                    if state["sharpen_enabled"] and strength > 0:
                        out_frame = apply_smart_sharpen(
                            out_frame,
                            strength=strength,
                            threshold=int(cfg["sharpen_threshold"]),
                            sigma=float(cfg["sharpen_sigma"]),
                        )
                    if state["deep_fry_enabled"]:
                        out_frame = apply_deep_fry(out_frame)

                    if magic_active:
                        # Entry crossfade: dissolve the live picture into the
                        # loop over ~0.5 s so engaging Magic Loop never pops.
                        alpha = 1.0 - magic_entry_left / magic_entry_fade
                        loop_frame = magic_player.next_frame((out_w, out_h))
                        out_frame = cv2.addWeighted(
                            out_frame, 1.0 - alpha, loop_frame, alpha, 0.0
                        )
                        magic_entry_left -= 1

                    out_frame = time_loop.record(out_frame)
                    # Always feed the AFK history so Magic Loop can start
                    # from the moment it is switched on. The ~2 ms encode
                    # fits the frame budget at 30 fps.
                    magic_loop.record(out_frame)

                cam.send(out_frame)
                cam.sleep_until_next_frame()

                now = time.monotonic()
                elapsed = now - last_time
                if elapsed > 0:
                    measured = 1.0 / elapsed
                    if fps_ema == 0.0:
                        fps_ema = measured
                    else:
                        fps_ema = fps_ema * 0.9 + measured * 0.1
                last_time = now
                self.state.set_telemetry(fps=fps_ema, zoom=curr_zoom)

                if self.preview:
                    cv2.imshow("center-stage", out_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        return

                frame_idx += 1
            except Exception as exc:
                # Transient per-frame errors should not kill a call; only give
                # up after repeated failures, printing the stack for diagnosis.
                consecutive_errors += 1
                if consecutive_errors > 10:
                    import traceback

                    traceback.print_exc()
                    raise
                print(f"warning: frame processing error ({consecutive_errors}): {exc}")
                time.sleep(0.01)


def parse_args(argv=None):
    """Parse CLI flags; all values override the corresponding INI settings."""
    p = argparse.ArgumentParser(description="Center Stage for a webcam")
    p.add_argument("--config", help="path to INI config file")
    p.add_argument("--camera", type=int, help="camera index override")
    p.add_argument("--lut", help="path to a .cube 3D LUT (overrides ini)")
    p.add_argument("--lut-scale", type=int, help="LUT downscale factor (default 8)")
    p.add_argument("--detect-every", type=int, help="run face detection every N frames")
    p.add_argument(
        "--sharpen",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable/disable edge-aware sharpening",
    )
    p.add_argument(
        "--sharpen-strength", type=float, help="sharpening amount (default 0.3)"
    )
    p.add_argument(
        "--tracking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable/disable face tracking (off = base zoom, centered)",
    )
    p.add_argument(
        "--time-loop",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="replay the last second of processed video in a loop",
    )
    p.add_argument(
        "--deep-fry",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="apply the deep-fry saturation and RGB channel-roll effect",
    )
    p.add_argument(
        "--magic-loop",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="AFK mode: replay a seamless crossfaded loop of recent video",
    )
    p.add_argument(
        "--preview", action="store_true", help="show a debug preview window (costs CPU)"
    )
    p.add_argument(
        "--list-cameras",
        action="store_true",
        help="list available camera indices and exit",
    )
    return p.parse_args(argv)



def main(argv=None):
    """Start the video pipeline and the Textual control panel."""
    args = parse_args(argv)

    # Diagnostics mode: enumerate devices and exit without touching the camera.
    if args.list_cameras:
        print("Available cameras:")
        list_cameras()
        return

    # Config precedence: defaults < INI file < CLI flags.
    config_path = find_config_path(args.config)
    cfg = load_config(config_path)
    if args.camera is not None:
        cfg["camera_index"] = args.camera
    if args.lut is not None:
        cfg["lut_file"] = args.lut
    if args.lut_scale is not None:
        cfg["lut_scale"] = args.lut_scale
    if args.detect_every is not None:
        cfg["detect_every"] = args.detect_every
    if args.sharpen is not None:
        cfg["sharpen_enabled"] = args.sharpen
    if args.sharpen_strength is not None:
        cfg["sharpen_strength"] = args.sharpen_strength
    if args.tracking is not None:
        cfg["tracking_enabled"] = args.tracking
    if args.time_loop is not None:
        cfg["time_loop_enabled"] = args.time_loop
    if args.deep_fry is not None:
        cfg["deep_fry_enabled"] = args.deep_fry
    if args.magic_loop is not None:
        cfg["magic_loop_enabled"] = args.magic_loop

    base_dir = (
        os.path.dirname(os.path.abspath(config_path)) if config_path else os.getcwd()
    )
    # Keep the configured LUT resolvable from the config folder or the working
    # directory; the TUI lists it and the background worker bakes from it.
    cfg["lut_file"] = resolve_lut_path(cfg["lut_file"], base_dir) or ""

    # Without an existing config, autosave creates one next to the user.
    if config_path is None:
        config_path = os.path.abspath("center-stage.ini")

    # Imported here so --list-cameras works even if the TUI stack is broken.
    from tui import CenterStageApp

    CenterStageApp(cfg, config_path, preview=args.preview).run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
