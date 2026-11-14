"""Apple-style Center Stage for a regular webcam.

Captures a high-res frame, tracks face(s) with MediaPipe, and publishes a
cropped/zoomed view to a virtual camera so video-call apps see a framed shot.

The video pipeline runs in a daemon thread while tui.py renders a Textual
control panel. Settings come from an INI file (see hyper-stage.ini), can be
changed live in the TUI, and are written back when they change.
"""

import argparse
import collections
import configparser
import contextlib
import glob
import math
import os
import re
import sys
import tempfile
import threading
import time
import urllib.request

import cv2
import mediapipe as mp
import moderngl
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
    "shader_enabled": False,    # toggle GLSL shader processing on the GPU
    "shader_file": "",          # path to a .glsl fragment shader (empty = none)
    "sharpen_enabled": False,   # toggle edge-aware sharpening
    "sharpen_strength": 0.3,    # unsharp-mask amount (0..1 looks natural)
    "sharpen_threshold": 10,    # min local contrast before sharpening kicks in
    "sharpen_sigma": 2.0,       # Gaussian blur radius for the high-pass filter
    "time_loop_enabled": False, # replay the last second of processed frames
    "deep_fry_enabled": False,  # high-contrast saturation + RGB channel roll
    "magic_loop_enabled": False,  # AFK: play a seamless loop of recent video
    "theme": "textual-dark",    # Textual TUI theme (change via the command palette)
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
        "shader_enabled": "shader_enabled",
        "shader_file": "shader_file",
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
    "appearance": {"theme": "theme"},
}

# One second of processed frames at the nominal 30 FPS.
TIME_LOOP_FRAMES = 30

# Magic Loop (AFK) history: ten seconds of encoded processed frames. Frames
# are stored JPEG-compressed at reduced size; raw 1080p would need ~1.9 GB for
# 300 frames, this stays around 100 MB and the encode fits the frame budget.
MAGIC_LOOP_FRAMES = 300
MAGIC_LOOP_W, MAGIC_LOOP_H = 960, 540
MAGIC_LOOP_JPEG_QUALITY = 85
# Frames right before the key press are never used for the loop: the head is
# already turning towards the keyboard and the hand is on its way. Everything
# recorded after the press is off limits too (the person is getting up).
MAGIC_LOOP_GUARD_SECONDS = 0.8
# Need three seconds of history before a loop can be built at all.
MAGIC_LOOP_MIN_FRAMES = 90
# Seconds of dissolve from the live feed into the loop and back out of it.
MAGIC_LOOP_ENTRY_FADE_SECONDS = 0.4
MAGIC_LOOP_EXIT_FADE_SECONDS = 0.5
# The loop itself must be 2-5 seconds: short enough segments look like a
# nervous stutter, long ones stop feeling "frozen".
MAGIC_LOOP_MIN_LOOP_FRAMES = 60
MAGIC_LOOP_MAX_LOOP_FRAMES = 150
MAGIC_LOOP_LENGTH_STEP = 10
# The seam is hidden by a crossfade this many frames long. The tail of the
# loop runs on past its nominal end into the footage that originally followed
# it, and that footage is faded into the head, so motion never reverses or
# jumps at the restart.
MAGIC_LOOP_BLEND_FRAMES = 12
# Score weights: a ghosting seam and a crop that differs from the frozen
# framing (cost per percent of crop width) versus plain motion.
MAGIC_LOOP_SEAM_WEIGHT = 2.0
MAGIC_LOOP_CROP_WEIGHT = 3.0
# Shorter loops repeat more noticeably; the shortest gets this much penalty.
MAGIC_LOOP_SHORT_PENALTY = 0.25

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
        prefix=".hyper-stage-", suffix=".ini", dir=directory
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.write("; Hyper Stage configuration.\n")
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
        "hyper-stage.ini",
        os.path.expanduser("~/.config/hyper-stage.ini"),
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
      detail=True  - the graded frame is upscaled and the luma-only detail
                     of the original is added back: sharpness is preserved
                     without leaking the original colors through the detail;
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

    # Sharpness detail with the chroma stripped. The naive form
    # (frame + upscaled correction) keeps the full-res chroma, so a B&W
    # grade leaks the original colors back at edges and in sensor noise.
    # Adding only the luma detail to the upscaled graded image preserves
    # the sharpness this mode exists for without re-tinting the grade.
    # cvtColor's gray conversion uses the same Rec.601 weights as a manual
    # 0.299/0.587/0.114 transform but is several times faster.
    luma = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    luma_small_up = cv2.resize(
        cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (w, h), interpolation=cv2.INTER_LINEAR
    )
    # Signed luma correction stored as uint8 with offset 128 (range +-127);
    # one saturating op replaces the int16 subtract/offset/clip/astype chain.
    delta8 = cv2.addWeighted(luma, 1.0, luma_small_up, -1.0, 128.0)
    graded_up = cv2.resize(graded, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(
        graded_up, 1.0, cv2.cvtColor(delta8, cv2.COLOR_GRAY2BGR), 1.0, -128.0
    )


def apply_smart_sharpen(frame_bgr, strength=0.3, threshold=10, sigma=2.0):
    """Edge-aware unsharp mask.

    Boosts high-frequency detail (contours, texture) while leaving flat areas
    such as skin untouched: only pixels whose local contrast exceeds threshold
    are sharpened.

    This runs on every output frame on a single OpenCV thread, so it is built
    from as few full-frame passes as possible:
      - frame + strength * (frame - blurred) is one saturating uint8
        addWeighted (no int16 round trip, same result within +-1 level);
      - the gate is a uint8 threshold plus copyTo, not np.where;
      - for sigma >= 1.5 the blur is computed at half resolution and scaled
        back up. A Gaussian is low-pass, so little is lost, and the blur is
        the most expensive pass. Smaller sigmas target fine detail and keep
        the full-resolution blur.
    About 3x faster than the previous int16 version at 1080p.
    """
    if strength <= 0:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    if sigma >= 1.5 and w >= 4 and h >= 4:
        small = cv2.resize(
            frame_bgr, (w // 2, h // 2), interpolation=cv2.INTER_AREA
        )
        small = cv2.GaussianBlur(small, (0, 0), sigmaX=float(sigma) / 2.0)
        blurred = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        blurred = cv2.GaussianBlur(frame_bgr, (0, 0), sigmaX=float(sigma))

    sharpened = cv2.addWeighted(
        frame_bgr, 1.0 + float(strength), blurred, -float(strength), 0.0
    )
    # Gate on local contrast so digital noise on smooth skin is not amplified.
    mask = cv2.threshold(
        cv2.absdiff(frame_bgr, blurred), float(threshold), 255, cv2.THRESH_BINARY
    )[1]
    out = frame_bgr.copy()
    cv2.copyTo(sharpened, mask, out)
    return out


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
        "shader_enabled",
        "shader_file",
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


def _grid_diff(a, b):
    """Visible difference between stacks of grid thumbnails (..., H, W).

    The plain mean hides small local changes: a blink or a lip twitch moves a
    handful of cells and barely shifts the average of a whole frame. Adding the
    worst single cell makes those events stand out, while the area averaging
    that produced the grid already suppresses sensor noise.
    """
    d = np.abs(a - b)
    return d.mean(axis=(-2, -1)) + d.max(axis=(-2, -1))


class MagicLoopBuffer:
    """Rolling history of encoded processed frames for the AFK Magic Loop.

    Every processed output frame is stored JPEG-compressed at reduced size
    together with a small grayscale grid (an area-average thumbnail) and the
    crop (x, y, zoom) it was rendered with. The grids make the window search
    nearly free; the crops let the search prefer footage framed exactly like
    the frozen live view; the JPEGs keep ten seconds of history around
    100 MB instead of the ~1.9 GB raw frames would need.
    """

    # Scoring grid: each cell averages a 15x15 block of the stored frame.
    GRID_W, GRID_H = 64, 36

    def __init__(self, length=MAGIC_LOOP_FRAMES):
        self._frames = collections.deque(maxlen=length)
        # Total frames ever recorded. It never resets, so a mark taken at the
        # key press stays meaningful while old frames fall out of the deque.
        self.count = 0

    def clear(self):
        """Forget the history (frames from before an absence are stale)."""
        self._frames.clear()

    def record(self, frame, crop):
        """Encode one processed frame into the history (best effort)."""
        small = cv2.resize(frame, (MAGIC_LOOP_W, MAGIC_LOOP_H))
        ok, encoded = cv2.imencode(
            ".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, MAGIC_LOOP_JPEG_QUALITY]
        )
        if not ok:
            return
        grid = cv2.cvtColor(
            cv2.resize(
                small, (self.GRID_W, self.GRID_H), interpolation=cv2.INTER_AREA
            ),
            cv2.COLOR_BGR2GRAY,
        )
        self._frames.append((encoded.tobytes(), grid, crop))
        self.count += 1

    def build_loop(self, frozen_crop, upto=None):
        """Pick the best loop from history and return a player, or None.

        Only frames recorded before the absolute frame number `upto` are
        considered (None = everything). A candidate is a window of 2-5 s plus
        the crossfade tail that follows it, scored by:
          - internal motion (blinks, gestures, head turns raise it),
          - seam ghosting: how much the tail differs from the head it is
            blended into, which is what the audience sees at the restart,
          - how far its crop is from the frozen live framing, because the
            loop is dissolved in from and out to the live view.
        Every candidate of one length is scored in a single vectorized pass,
        so the whole search takes a few milliseconds. Returns None while there
        is not enough usable history yet.
        """
        frames = list(self._frames)
        if upto is not None:
            first = self.count - len(frames)
            frames = frames[: max(0, upto - first)]
        m = len(frames)
        if m < MAGIC_LOOP_MIN_FRAMES:
            return None

        grids = np.stack([g for _, g, _ in frames]).astype(np.int16)
        crops = np.array([c for _, _, c in frames], dtype=np.float64)
        fx, fy, fz = frozen_crop
        # Crop deviation in percent of the crop width: a pan of dx (fraction
        # of the frame) moves the picture by dx * zoom crop widths.
        dev = 100.0 * np.maximum(
            np.maximum(np.abs(crops[:, 0] - fx), np.abs(crops[:, 1] - fy)) * fz,
            np.abs(crops[:, 2] - fz) / fz,
        )
        motion = _grid_diff(grids[:-1], grids[1:])
        prefix = np.concatenate(([0.0], np.cumsum(motion)))

        blend = MAGIC_LOOP_BLEND_FRAMES
        probes = sorted({0, blend // 3, (2 * blend) // 3, blend - 1})
        min_len, max_len = MAGIC_LOOP_MIN_LOOP_FRAMES, MAGIC_LOOP_MAX_LOOP_FRAMES
        best = None  # (score, start, length)
        for length in range(min_len, max_len + 1, MAGIC_LOOP_LENGTH_STEP):
            span = length + blend
            if span > m:
                break
            starts = np.arange(m - span + 1)
            internal = (prefix[starts + span - 1] - prefix[starts]) / (span - 1)
            seam = np.mean(
                [_grid_diff(grids[starts + k], grids[starts + length + k]) for k in probes],
                axis=0,
            )
            crop_dev = np.lib.stride_tricks.sliding_window_view(dev, span).max(axis=1)
            score = (
                internal
                + MAGIC_LOOP_SEAM_WEIGHT * seam
                + MAGIC_LOOP_CROP_WEIGHT * crop_dev
            )
            score = score * (
                1.0
                + MAGIC_LOOP_SHORT_PENALTY * (max_len - length) / (max_len - min_len)
            )
            k = int(np.argmin(score))
            if best is None or score[k] < best[0]:
                best = (float(score[k]), int(starts[k]), length)
        if best is None:
            return None
        _, start, length = best
        return MagicLoopPlayer(
            [jpeg for jpeg, _, _ in frames[start : start + length + blend]],
            length,
            blend,
        )


class MagicLoopPlayer:
    """Plays a fixed loop with a crossfaded restart.

    The stored clip is `length + blend` frames. Playback frame t of the loop
    is clip[t] for t >= blend; for the first `blend` frames it is a mix that
    slides from clip[length + t] (the footage that originally followed the
    loop's last frame) to clip[t]. So the last frame of one pass continues
    into the first frame of the next the way the real video would have.
    """

    def __init__(self, encoded, length, blend):
        self._frames = encoded
        self._length = length
        self._blend = blend
        self._index = 0

    def _decode(self, i):
        return cv2.imdecode(
            np.frombuffer(self._frames[i], np.uint8), cv2.IMREAD_COLOR
        )

    def next_frame(self, out_size):
        """Return the next loop frame resized to out_size (w, h)."""
        t = self._index % self._length
        self._index += 1
        frame = self._decode(t)
        if t < self._blend:
            weight = (t + 1) / (self._blend + 1)
            tail = self._decode(self._length + t)
            frame = cv2.addWeighted(tail, 1.0 - weight, frame, weight, 0.0)
        return np.ascontiguousarray(
            cv2.resize(frame, out_size, interpolation=cv2.INTER_LINEAR)
        )


def apply_deep_fry(frame_bgr, saturation=2.0, contrast=1.3, shift=10):
    """Exaggerated saturation/contrast plus an RGB channel roll.

    The roll happens on the red and blue planes in opposite directions, which
    produces the offset fringes of the classic over-processed "deep fried" look.

    Saturation and contrast are per-value maps, so each is a 256-entry lookup
    table applied in one pass (the saturation table only touches the S plane
    of the HSV image) instead of float32 arithmetic over the whole frame; the
    output is bit-identical to the float version. The roll is written straight
    into the result with slices rather than through np.roll copies.
    """
    values = np.arange(256, dtype=np.float32)
    hsv_lut = np.empty((256, 1, 3), np.uint8)
    hsv_lut[:, 0, 0] = np.arange(256)
    hsv_lut[:, 0, 1] = np.clip(values * float(saturation), 0, 255).astype(np.uint8)
    hsv_lut[:, 0, 2] = np.arange(256)
    contrast_lut = np.clip(
        (values - 128.0) * float(contrast) + 128.0, 0, 255
    ).astype(np.uint8)

    hsv = cv2.LUT(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV), hsv_lut)
    graded = cv2.LUT(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), contrast_lut)

    s = int(shift) % graded.shape[1]
    if s == 0:
        return graded
    out = graded.copy()
    # Blue rolled left by s, red rolled right by s (np.roll semantics).
    out[:, :-s, 0] = graded[:, s:, 0]
    out[:, -s:, 0] = graded[:, :s, 0]
    out[:, s:, 2] = graded[:, :-s, 2]
    out[:, :s, 2] = graded[:, -s:, 2]
    return out


# Vertex source for the shader pass: a full-screen quad passing pixel
# coordinates through. gl_FragCoord in the fragment shader is Shadertoy's
# fragCoord (pixels, origin bottom-left).
SHADER_VERTEX_SOURCE = """
#version 330
in vec2 in_pos;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

# The user file provides mainImage(out vec4 fragColor, in vec2 fragCoord),
# Shadertoy style; this wrapper supplies the entry point, the input texture
# (texture0, holding the BGR frame), and the iResolution/iTime uniforms. The
# frame data and the rendered result are both BGR, so shaders that preserve
# channel order (all the brightness/pattern effects this is for) need no
# reordering.
SHADER_FRAGMENT_TEMPLATE = """
#version 330
uniform sampler2D texture0;
uniform sampler2D texture1;
uniform sampler2D texture2;
uniform sampler2D texture3;
uniform vec2 iResolution;
uniform float iTime;
uniform vec4 iMouse;
#define iChannel0 texture0
#define iChannel1 texture1
#define iChannel2 texture2
#define iChannel3 texture3
uniform vec3 iChannelResolution[4];
{user_source}
out vec4 out_Color;
void main() {{
    vec4 color;
    mainImage(color, gl_FragCoord.xy);
    out_Color = color;
}}
"""


def set_channel_resolution(program, textures):
    """Upload Shadertoy's iChannelResolution[4] from the bound textures.

    Defensive like the other uniforms: a program that never reads it has it
    optimized out. The Apple GL driver reports array_length as the highest
    referenced element index + 1, so upload exactly that many rows (packed
    arrays cannot skip elements).
    """
    if "iChannelResolution" not in program:
        return
    uniform = program["iChannelResolution"]
    length = uniform.array_length
    if length <= 0:
        return
    rows = [
        (float(tex.width), float(tex.height), 1.0)
        for tex in textures[:length]
        if tex is not None
    ]
    rows.extend([(0.0, 0.0, 1.0)] * (length - len(rows)))
    if length == 1:
        uniform.value = rows[0]
    else:
        uniform.value = rows


def list_shader_files(primary_path):
    """Return sorted, de-duplicated shader entries usable as TUI choices.

    Mirrors list_lut_files: looks next to the configured shader, in the
    current directory, and in shaders/. Entries are single .glsl files or
    multibuffer pack folders named <name>.glsl (Shadertoy exports with
    Common/Buffer/Image passes).
    """
    folders = []
    if primary_path:
        folders.append(os.path.dirname(os.path.abspath(primary_path)))
    folders.append(os.getcwd())
    folders.append(os.path.join(os.getcwd(), "shaders"))
    found = {}
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for path in glob.glob(os.path.join(folder, "*.glsl")):
            found.setdefault(os.path.abspath(path), None)
    return sorted(found)


def load_shader(path):
    """Read a .glsl fragment shader; None when the file is unusable."""
    if not path:
        return None
    try:
        with open(path) as f:
            source = f.read()
    except OSError as exc:
        print(f"warning: shader file not readable: {path} ({exc})")
        return None
    if "void mainImage" not in source:
        print(f"warning: shader {path} has no mainImage(); ignoring")
        return None
    return source


class MultiShader:
    """A Shadertoy multibuffer export: a <name>.glsl folder of passes.

    The export format stores pass wiring outside the files, so this uses a
    fixed convention plus per-pass directives. Without directives the
    standard layout (Common.glsl, Buffer A/B/... .glsl, Image.glsl) works
    like the common export: Common is prepended to every pass, buffers run
    in file-name order, the first buffer's iChannel0 is the camera frame
    and each following pass reads the previous pass output, every buffer's
    iChannel1 is its own previous frame (ping-pong feedback, so temporal
    denoising like Lucas-Kanade's Buffer A works), and Image.glsl reads the
    last buffer on iChannel0. Any channel a pass samples but that has no
    directive defaults to the noise texture (unbound samplers read garbage
    on macOS). Exports with other wiring (say Image.glsl reading Buffer B
    on iChannel2) add a directive comment to the pass file, e.g.

        // hyper-stage: 2=Buffer B

    Names: camera, noise, font, feedback (the pass's own previous frame),
    or a buffer pass name like "Buffer B" (its latest output, which may be
    a previous frame when read before that buffer runs).

    Unlike single-file shaders (BGR in/out, matching the camera), a pack is
    a verbatim Shadertoy import, so the camera frame is converted to RGB
    before the first pass and the final render back to BGR.
    """

    # Directive names that resolve to auxiliary textures or feedback.
    AUX_NAMES = ("camera", "noise", "font", "feedback")

    def __init__(self, ctx, quad_buffer, folder, start, aux_texture):
        self._ctx = ctx
        self._folder = folder
        self._start = start
        self._buffer = quad_buffer
        self._aux_texture = aux_texture
        self._camera = None
        # Per buffer pass: [ping texture, pong texture] and matching FBOs;
        # rendering goes to the front entry, which then rotates to the back
        # so the next frame reads it as feedback.
        self._textures = []
        self._fbos = []
        self._passes = []  # (program, quad VAO, channels) per buffer pass
        self._final = None  # (program, quad VAO, channels) for Image.glsl
        self._size = (0, 0)

    @staticmethod
    def parse_channels(source, buffer_names):
        """Map unit number -> channel name from // hyper-stage: directives.

        Accepts the aux names (camera/noise/font/feedback) and references
        to a buffer pass file like "Buffer B" (spaces optional, case
        ignored). Unknown names are ignored, leaving the default wiring.
        """
        channels = {}
        for line in source.splitlines():
            marker = line.find("hyper-stage:")
            if marker < 0:
                continue
            matches = re.findall(
                r"(\d+)\s*=\s*(\w+(?:\s+\w+)*?)(?=\s+\d+\s*=|$)",
                line[marker:],
            )
            for unit, name in matches:
                key = name.strip().lower().replace(" ", "")
                if key in MultiShader.AUX_NAMES:
                    channels[int(unit)] = key
                else:
                    for i, buf in enumerate(buffer_names):
                        stem = buf[: -len(".glsl")].lower().replace(" ", "")
                        if key == stem:
                            channels[int(unit)] = ("buffer", i)
                            break
        return channels

    def close(self):
        """Release everything, including the compiled pass programs."""
        self._release_targets()
        for entry in self._passes:
            self._release_pass(entry)
        if self._final is not None:
            self._release_pass(self._final)
        self._passes = []
        self._final = None
        self._size = (0, 0)

    @staticmethod
    def _release_pass(entry):
        program, quad = entry[0], entry[1]
        if quad is not None:
            quad.release()
        program.release()

    def _release_targets(self):
        """Release render targets; programs and quads stay usable."""
        for texs in self._textures:
            for tex in texs:
                tex.release()
        for fbos in self._fbos:
            for fbo in fbos:
                fbo.release()
        self._textures = []
        self._fbos = []
        if self._camera is not None:
            self._camera.release()
            self._camera = None

    def _compile(self, source):
        program = self._ctx.program(
            vertex_shader=SHADER_VERTEX_SOURCE,
            fragment_shader=SHADER_FRAGMENT_TEMPLATE.format(user_source=source),
        )
        # The quad VAO must belong to the program it renders with.
        quad = self._ctx.vertex_array(program, [(self._buffer, "2f", "in_pos")])
        return program, quad

    def _set_clock(self, program):
        # Defensive upload: unused uniforms are optimized out of a program.
        if "iResolution" in program:
            program["iResolution"].value = (
                float(self._size[0]),
                float(self._size[1]),
            )
        if "iTime" in program:
            program["iTime"].value = time.monotonic() - self._start

    def _default_channel(self, pass_index, unit):
        """The fixed convention for a pass channel without a directive."""
        if pass_index == len(self._passes):
            # Image.glsl: read the last buffer on iChannel0.
            return ("buffer", len(self._passes) - 1) if unit == 0 else "noise"
        if unit == 0:
            # First buffer reads the camera, the rest the previous pass.
            return "camera" if pass_index == 0 else ("buffer", pass_index - 1)
        if unit == 1:
            return ("feedback", pass_index)
        return "noise"

    def _resolve_channel(self, name, pass_index):
        """Texture for a channel name at pass runtime; None keeps it unbound."""
        if name == "camera":
            return self._camera
        if name in ("noise", "font"):
            return self._aux_texture(name)
        if isinstance(name, tuple):
            kind, index = name
            if kind == "feedback":
                # At render time the back entry is the previous frame; the
                # pass about to run has not rotated yet.
                return self._textures[index][1]
            if kind == "buffer":
                # That buffer's latest output (after its rotation this
                # frame, or its previous frame if it has not run yet).
                return self._textures[index][1]
        return None

    def _bind_channels(self, program, channels, pass_index):
        """Bind all four samplers, defaults included, for one pass."""
        bound = []
        for unit in range(4):
            name = channels.get(
                unit, self._default_channel(pass_index, unit)
            )
            tex = self._resolve_channel(name, pass_index)
            if tex is None:
                # Unknown/unresolvable names (or an aux that could not be
                # created) fall back to noise; unbound reads garbage.
                tex = self._aux_texture("noise")
            bound.append(tex)
            tex.use(unit)
            sampler = (
                program[f"texture{unit}"]
                if f"texture{unit}" in program
                else None
            )
            if sampler is not None:
                sampler.value = unit
        set_channel_resolution(program, bound)

    def build(self):
        """Compile every pass; None (after a warning) on any failure."""
        try:
            names = sorted(os.listdir(self._folder))
        except OSError as exc:
            print(f"warning: shader pack {self._folder} unreadable: {exc}")
            return None
        names = [n for n in names if n.endswith(".glsl")]
        common = ""
        if "Common.glsl" in names:
            with open(os.path.join(self._folder, "Common.glsl")) as f:
                common = f.read() + "\n"
        buffers = [n for n in names if n not in ("Common.glsl", "Image.glsl")]
        if "Image.glsl" not in names or not buffers:
            print(
                f"warning: shader pack {self._folder} needs buffer passes "
                "and Image.glsl"
            )
            return None
        try:
            for name in buffers:
                with open(os.path.join(self._folder, name)) as f:
                    source = common + f.read()
                channels = self.parse_channels(source, buffers)
                self._passes.append(
                    (*self._compile(source), channels)
                )
            with open(os.path.join(self._folder, "Image.glsl")) as f:
                source = common + f.read()
            channels = self.parse_channels(source, buffers)
            self._final = (*self._compile(source), channels)
        except Exception as exc:
            print(f"warning: shader pack {self._folder} failed: {exc}")
            self.close()
            return None
        return self

    def _ensure_size(self, width, height):
        if self._size == (width, height):
            return True
        self._release_targets()
        try:
            self._camera = self._ctx.texture((width, height), 3, dtype="f1")
            # NEAREST, not LINEAR: imported shaders sample at uv = k/N,
            # which lands exactly on texel boundaries; linear would average
            # a 2x2 texel block per sample (a blur that shifts luma across
            # dither thresholds - 1bit.glsl turned speckled). Nearest keeps
            # the same-resolution mapping a crisp texel-per-pixel copy,
            # which is what Shadertoy's larger webcam texture effectively
            # gives those uv's too.
            self._camera.filter = (moderngl.NEAREST, moderngl.NEAREST)
            for _ in self._passes:
                # Two ping-pong targets per buffer: the framebuffer wraps the
                # texture, so what is rendered is exactly what the next pass
                # (and the next frame's feedback) samples. Buffers are float
                # (RGBA16F) like Shadertoy's, so passes can store negative or
                # fractional data (Lucas-Kanade's motion vectors) between
                # passes and frames.
                texs = [
                    self._ctx.texture((width, height), 4, dtype="f2")
                    for _ in range(2)
                ]
                for tex in texs:
                    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                    tex.repeat_x = False
                    tex.repeat_y = False
                self._textures.append(texs)
                self._fbos.append([self._ctx.framebuffer(tex) for tex in texs])
        except Exception as exc:
            print(f"warning: shader pack {self._folder} size error: {exc}")
            self._release_targets()
            return False
        self._size = (width, height)
        return True

    def run(self, output_fbo, frame_bgr):
        """Process one BGR camera frame through the pass chain."""
        height, width = frame_bgr.shape[:2]
        if not self._ensure_size(width, height):
            return frame_bgr
        # Same boundary flips as the single-file path: upload flipped so the
        # camera texture reads like a Shadertoy asset (t=0 at image bottom),
        # final read flipped back to top-down. Internal buffer passes render
        # and sample with the same uv mapping, so they need no flips.
        rgb = cv2.cvtColor(cv2.flip(frame_bgr, 0), cv2.COLOR_BGR2RGB)
        self._camera.write(rgb)
        for i, (program, quad, channels) in enumerate(self._passes):
            textures = self._textures[i]
            fbos = self._fbos[i]
            self._bind_channels(program, channels, i)
            self._set_clock(program)
            fbos[0].use()
            quad.render(moderngl.TRIANGLE_STRIP)
            # Rotate so the just-written texture becomes next frame's
            # feedback and the old feedback becomes the render target.
            textures.append(textures.pop(0))
            fbos.append(fbos.pop(0))
            # Shadertoy keeps mipmaps on its buffer textures; passes that
            # texelFetch an explicit LOD (Ben-Day-Dots downsampling) read
            # black from a level that does not exist without this.
            textures[1].build_mipmaps()
        program, quad, channels = self._final
        self._bind_channels(program, channels, len(self._passes))
        self._set_clock(program)
        output_fbo.use()
        quad.render(moderngl.TRIANGLE_STRIP)
        data = output_fbo.read(components=3, alignment=1)
        # Undo the bottom-up GL readback (see the upload flip above).
        rgb_out = cv2.flip(
            np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3), 0
        )
        return cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)



class ShaderEngine:
    """Compiles and runs the active .glsl shader on the GPU.

    One moderngl standalone context lives inside the video daemon thread and
    is created lazily on first use, keeping GL affinity in one place. Programs
    are cached per file so switching shaders or re-selecting the same file
    never recompiles; a shader that fails to compile disables the effect
    (the live frame passes through) with a single warning. A multibuffer
    pack (a <name>.glsl folder) is compiled into a MultiShader instead.
    """

    VERTEX = SHADER_VERTEX_SOURCE

    # Auxiliary textures a shader can bind instead of the camera, selectable
    # per channel with a directive comment in the shader file, e.g.
    #   // hyper-stage: 0=font 1=camera
    # Channels not mentioned default to: 0=camera, others noise (unbound
    # samplers read garbage on macOS, never black).
    AUX_KEYS = ("camera", "noise", "font")

    def __init__(self):
        self._ctx = None
        self._buffer = None
        self._fbo = None
        self._texture = None
        self._size = (0, 0)
        self._start = time.monotonic()
        self._aux = {}  # name -> texture, created lazily per context
        # path -> (program, quad VAO) / MultiShader, or None (broken)
        self._programs = {}
        self._broken = set()

    @staticmethod
    def parse_channels(source):
        """Map unit number -> aux name from // hyper-stage: directives."""
        channels = {}
        for line in source.splitlines():
            marker = line.find("hyper-stage:")
            if marker < 0:
                continue
            for unit, name in re.findall(
                r"(\d+)\s*=\s*(\w+)", line[marker:]
            ):
                if name in ShaderEngine.AUX_KEYS:
                    channels[int(unit)] = name
        return channels

    def _aux_texture(self, name):
        """Return the named auxiliary texture, creating it on first use."""
        if name in self._aux:
            return self._aux[name]
        if name == "noise":
            rng = np.random.default_rng(1234)
            data = rng.integers(0, 256, (256, 256, 4), dtype=np.uint8)
            tex = self._ctx.texture((256, 256), 4, data.tobytes())
            # Noise lookups often go outside [0,1] on purpose.
            tex.repeat_x = True
            tex.repeat_y = True
        elif name == "font":
            # A 16x16 grid of ASCII glyphs standing in for Shadertoy's font
            # texture (built with OpenCV's vector font, not pixel-perfect).
            cell = 16
            atlas = np.zeros((256, 256, 4), dtype=np.uint8)
            for code in range(32, 127):
                x = (code % 16) * cell
                y = (code // 16) * cell
                glyph = np.zeros((cell, cell), dtype=np.uint8)
                cv2.putText(
                    glyph, chr(code), (1, cell - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1, cv2.LINE_AA,
                )
                atlas[y : y + cell, x : x + cell, 0] = glyph
                # Approximate the SDF-ish alpha some shaders read for
                # outlines: a slightly dilated glyph.
                atlas[y : y + cell, x : x + cell, 3] = cv2.dilate(
                    glyph, np.ones((3, 3), np.uint8)
                )
            tex = self._ctx.texture((256, 256), 4, atlas.tobytes())
        else:
            return None
        self._aux[name] = tex
        return tex

    def _ensure_size(self, width, height):
        """Create the GL objects for a frame size; recreate if it changed."""
        if self._ctx is None:
            try:
                self._ctx = moderngl.create_standalone_context()
            except Exception as exc:
                print(f"warning: GPU shader context unavailable ({exc})")
                return False
        if self._size != (width, height):
            self._buffer = self._ctx.buffer(
                np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32).tobytes()
            )
            self._texture = self._ctx.texture((width, height), 3, dtype="f1")
            # NEAREST, not LINEAR: imported shaders sample at uv = k/N,
            # which lands exactly on texel boundaries; linear would average
            # a 2x2 texel block per sample (a blur that shifts luma across
            # dither thresholds - 1bit.glsl turned speckled). Nearest keeps
            # the same-resolution mapping a crisp texel-per-pixel copy.
            self._texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self._texture.repeat_x = False
            self._texture.repeat_y = False
            self._fbo = self._ctx.simple_framebuffer((width, height), 3)
            self._size = (width, height)
            # The compiled VAOs reference the old-size-independent buffer, so
            # they stay valid; drop nothing.
        return True

    def apply(self, frame_bgr, path):
        """Run the shader on frame_bgr; return the result or the input frame."""
        height, width = frame_bgr.shape[:2]
        if not self._ensure_size(width, height):
            return frame_bgr
        if os.path.isdir(path):
            return self._apply_multi(frame_bgr, path)
        entry = self._programs.get(path, False)
        if entry is False:
            source = load_shader(path)
            entry = None
            if source is not None:
                try:
                    program = self._ctx.program(
                        vertex_shader=self.VERTEX,
                        fragment_shader=SHADER_FRAGMENT_TEMPLATE.format(
                            user_source=source
                        ),
                    )
                    # The quad VAO must belong to the program it renders with.
                    entry = (
                        program,
                        self._ctx.vertex_array(
                            program, [(self._buffer, "2f", "in_pos")]
                        ),
                        self.parse_channels(source),
                    )
                except Exception as exc:
                    if path not in self._broken:
                        self._broken.add(path)
                        print(f"warning: shader {path} failed to compile: {exc}")
            self._programs[path] = entry
        if entry is None:
            return frame_bgr
        program, quad, channels = entry

        # Verbatim Shadertoy import: the camera frame goes in as RGB (BGR
        # -> RGB on upload, back on readback), so shaders hardcoding RGB
        # palette colors (1bit.glsl, smiley.glsl) render their intended
        # hues. Pass-through shaders are unaffected - both swaps cancel.
        rgb_in = cv2.cvtColor(cv2.flip(frame_bgr, 0), cv2.COLOR_BGR2RGB)
        self._texture.write(rgb_in)
        bound = []
        for unit in range(4):
            name = channels.get(unit, "camera" if unit == 0 else "noise")
            # The camera texture on the requested unit; auxiliary stand-ins
            # (noise/font) for decorative channels. Unbound samplers read
            # garbage on macOS, so every declared unit gets a real texture.
            tex = self._texture if name == "camera" else self._aux_texture(name)
            bound.append(tex)
            if tex is not None:
                tex.use(unit)
            sampler = program[f"texture{unit}"] if f"texture{unit}" in program else None
            if sampler is not None:
                sampler.value = unit
        set_channel_resolution(program, bound)
        if "iResolution" in program:
            program["iResolution"].value = (float(width), float(height))
        if "iTime" in program:
            # Seconds since the engine started; shader effects use it as
            # their animation clock (monotonic, no wall-clock jumps).
            program["iTime"].value = time.monotonic() - self._start
        self._fbo.use()
        quad.render(moderngl.TRIANGLE_STRIP)
        data = self._fbo.read(components=3, alignment=1)
        # Undo the bottom-up GL readback and the BGR->RGB upload swap (see
        # the upload comment above).
        return cv2.cvtColor(
            cv2.flip(
                np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3), 0
            ),
            cv2.COLOR_RGB2BGR,
        )

    def _apply_multi(self, frame_bgr, path):
        """Run a multibuffer pack folder; input frame passes through if broken."""
        entry = self._programs.get(path, False)
        if entry is False:
            entry = MultiShader(
                self._ctx, self._buffer, path, self._start, self._aux_texture
            ).build()
            if entry is None:
                self._broken.add(path)
            self._programs[path] = entry
        if entry is None:
            return frame_bgr
        return entry.run(self._fbo, frame_bgr)


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
        super().__init__(name="hyper-stage-video", daemon=True)
        self.cfg = cfg
        self.state = state
        self.preview = preview
        self._stop_event = threading.Event()
        # Created lazily inside the video thread: the GL context must live
        # and be used on one thread only.
        self._shader_engine = None

    def stop(self):
        """Ask the loop to exit; the thread finishes the current frame first."""
        self._stop_event.set()

    def reload_shader(self):
        """Drop cached shader programs; the next frame recompiles from disk."""
        engine = self._shader_engine
        if engine is not None:
            # dict.clear() is atomic enough in CPython; the video thread only
            # reads entries and recompiles missing ones. Multibuffer packs
            # release their GL objects explicitly.
            for entry in engine._programs.values():
                if isinstance(entry, MultiShader):
                    entry.close()
            engine._programs.clear()

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

            # Native LiteRT logs land on the terminal during detector
            # create/close; scope the silencing to exactly those two points
            # instead of the whole pipeline so real errors stay visible.
            with quiet_native_stderr():
                detector_ctx = vision.FaceDetector.create_from_options(options)
                detector = detector_ctx.__enter__()
            try:
                with pyvirtualcam.Camera(
                    width=cfg["out_w"],
                    height=cfg["out_h"],
                    fps=int(cfg["fps"]),
                    fmt=pyvirtualcam.PixelFormat.BGR,
                ) as cam:
                    self.state.set_telemetry(device=cam.device, status="running")
                    self._loop(cap, detector, cam)
            finally:
                with quiet_native_stderr():
                    detector_ctx.__exit__(None, None, None)
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
        # Crop (x, y, zoom) held while Magic Loop is on, and the history
        # position of the key press (minus the guard) that bounds the loop.
        magic_frozen = None
        magic_mark = 0
        magic_guard = max(1, int(cfg["fps"] * MAGIC_LOOP_GUARD_SECONDS))
        magic_entry_fade = max(1, int(cfg["fps"] * MAGIC_LOOP_ENTRY_FADE_SECONDS))
        magic_entry_left = 0
        magic_exit_fade = max(1, int(cfg["fps"] * MAGIC_LOOP_EXIT_FADE_SECONDS))
        magic_exit_left = 0
        magic_exit_player = None

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
                    if magic_on:
                        # Freeze: tracking stops and the crop stays exactly
                        # where it is. The history was rendered with this
                        # framing, so any piece of it can be spliced into the
                        # live view without a pan or zoom jump.
                        if state["tracking_enabled"]:
                            self.state.update(tracking_enabled=False)
                            state["tracking_enabled"] = False
                        magic_frozen = (curr_x, curr_y, curr_zoom)
                        zoom_vel = 0.0
                        # Only footage from before the person reached for the
                        # key may be used; what comes after is not trusted.
                        magic_mark = magic_loop.count - magic_guard
                        magic_player = None
                        magic_entry_left = 0
                        magic_exit_left = 0
                        magic_exit_player = None
                    else:
                        # Dissolve back to live and drop the old history: it
                        # now shows the person leaving or an empty chair.
                        magic_exit_player = magic_player
                        magic_exit_left = (
                            magic_exit_fade if magic_player is not None else 0
                        )
                        magic_player = None
                        magic_entry_left = 0
                        magic_frozen = None
                        magic_loop.clear()

                if magic_on and magic_player is None:
                    # The loop is picked the moment the key is pressed, so the
                    # audience never sees the person leaving. Without enough
                    # history before the press (e.g. started with
                    # --magic-loop) keep the live feed and retry every frame
                    # with whatever has been recorded, minus the guard.
                    player = magic_loop.build_loop(magic_frozen, upto=magic_mark)
                    if player is None:
                        player = magic_loop.build_loop(
                            magic_frozen, upto=magic_loop.count - magic_guard
                        )
                    if player is not None:
                        magic_player = player
                        magic_entry_left = magic_entry_fade

                # The plain Time Loop replay skips the whole image pipeline.
                # Magic Loop keeps it running (crop frozen) until its loop is
                # chosen, and the live feed still dissolves in and out of it.
                looping = (
                    bool(state["time_loop_enabled"])
                    and time_loop.ready
                    and not magic_on
                )
                # Once the entry dissolve is over, the loop replaces the live
                # view completely. Detection, crop, grading, sharpening and
                # recording would all be thrown away, and an AFK call can run
                # for hours: skip them and just play the loop.
                loop_covers_live = magic_player is not None and magic_entry_left <= 0
                if looping:
                    out_frame = time_loop.next_frame()
                elif loop_covers_live:
                    out_frame = None
                else:
                    if magic_on:
                        # Magic Loop holds the crop exactly as frozen at the
                        # key press: no detection, no pan, no zoom spring.
                        self.state.set_telemetry(tracking=False)
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
                    if state["shader_enabled"] and state["shader_file"]:
                        if self._shader_engine is None:
                            self._shader_engine = ShaderEngine()
                        out_frame = self._shader_engine.apply(
                            out_frame, state["shader_file"]
                        )
                    if state["deep_fry_enabled"]:
                        out_frame = apply_deep_fry(out_frame)

                    # Feed both replay buffers with the pure processed frame;
                    # blending below must never leak into recorded history.
                    out_frame = time_loop.record(out_frame)
                    # The ~2 ms encode fits the frame budget at 30 fps. Once a
                    # loop is playing, recording stops: it would only capture
                    # the empty chair, and the history is cleared on exit.
                    if magic_player is None:
                        magic_loop.record(out_frame, (curr_x, curr_y, curr_zoom))

                if magic_player is not None:
                    # Seamless continuous playback: the crossfade seam is
                    # inside the player; only the entry dissolves from live.
                    loop_frame = magic_player.next_frame((out_w, out_h))
                    if magic_entry_left > 0:
                        alpha = 1.0 - magic_entry_left / magic_entry_fade
                        out_frame = cv2.addWeighted(
                            out_frame, 1.0 - alpha, loop_frame, alpha, 0.0
                        )
                        magic_entry_left -= 1
                    else:
                        out_frame = loop_frame
                elif magic_exit_left > 0:
                    # Magic Loop was just switched off: the loop keeps
                    # running underneath while it dissolves into the live feed.
                    loop_frame = magic_exit_player.next_frame((out_w, out_h))
                    alpha = magic_exit_left / magic_exit_fade
                    out_frame = cv2.addWeighted(
                        out_frame, 1.0 - alpha, loop_frame, alpha, 0.0
                    )
                    magic_exit_left -= 1
                    if magic_exit_left == 0:
                        magic_exit_player = None

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
                    cv2.imshow("hyper-stage", out_frame)
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
        "--shader",
        help="path to a .glsl fragment shader (overrides ini)",
    )
    p.add_argument(
        "--shader-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable/disable GPU shader processing",
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
    if args.shader is not None:
        cfg["shader_file"] = args.shader
    if args.shader_enabled is not None:
        cfg["shader_enabled"] = args.shader_enabled

    base_dir = (
        os.path.dirname(os.path.abspath(config_path)) if config_path else os.getcwd()
    )
    # Keep the configured LUT resolvable from the config folder or the working
    # directory; the TUI lists it and the background worker bakes from it.
    cfg["lut_file"] = resolve_lut_path(cfg["lut_file"], base_dir) or ""

    # Without an existing config, autosave creates one next to the user.
    if config_path is None:
        config_path = os.path.abspath("hyper-stage.ini")

    # Imported here so --list-cameras works even if the TUI stack is broken.
    from tui import HyperStageApp

    HyperStageApp(cfg, config_path, preview=args.preview).run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)