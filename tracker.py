"""Apple-style Center Stage for a regular webcam.

Captures a high-res frame, tracks face(s) with MediaPipe, and publishes a
cropped/zoomed view to a virtual camera so video-call apps see a framed shot.

Camera and tuning parameters can be set in an INI file (see center-stage.ini)
and overridden on the command line.
"""

import argparse
import atexit
import configparser
import glob
import math
import os
import sys
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
}

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
    parser = configparser.ConfigParser()
    # A missing/unreadable file simply leaves every default in place.
    parser.read(path)
    # Maps INI names (human-facing) to internal config keys (code-facing).
    section_map = {
        "camera": {"index": "camera_index"},
        "capture": {
            "in_width": "in_w",
            "in_height": "in_h",
            "out_width": "out_w",
            "out_height": "out_h",
            "fps": "fps",
        },
        "tracking": {
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
    }
    for section, keys in section_map.items():
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
    """Return sorted .cube files in the folder of the given LUT (or LUT/)."""
    folder = None
    if primary_path:
        folder = os.path.dirname(os.path.abspath(primary_path))
    if not folder or not os.path.isdir(folder):
        folder = os.path.abspath("LUT")
    if not os.path.isdir(folder):
        return []
    return sorted(glob.glob(os.path.join(folder, "*.cube")))


def cycle_index(current, total, step):
    """Wrap a list index forward/backward, safe for an empty list."""
    if total <= 0:
        return 0
    return (current + step) % total


def setup_keyboard():
    """Put the terminal in cbreak mode for single-key input.

    Returns the saved terminal state, or None when stdin is not a TTY (e.g. the
    process runs in the background).
    """
    if not sys.stdin.isatty():
        return None
    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        return old
    except (ImportError, OSError):
        return None


def restore_keyboard(old):
    """Restore the terminal state saved by setup_keyboard()."""
    if old is None:
        return
    try:
        import termios

        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old)
    except (ImportError, OSError):
        pass


def read_keys():
    """Return any pending keypresses as a decoded string (empty if none)."""
    import select

    if not select.select([sys.stdin], [], [], 0)[0]:
        return ""
    try:
        # Read a few bytes so multi-byte UTF-8 (e.g. Russian х/ъ) stays intact.
        return os.read(sys.stdin.fileno(), 8).decode("utf-8", "ignore")
    except OSError:
        return ""


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
        "--preview", action="store_true", help="show a debug preview window (costs CPU)"
    )
    p.add_argument(
        "--list-cameras",
        action="store_true",
        help="list available camera indices and exit",
    )
    return p.parse_args(argv)


def main(argv=None):
    """Run the capture -> detect -> crop -> grade -> virtual-camera pipeline."""
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

    base_dir = (
        os.path.dirname(os.path.abspath(config_path)) if config_path else os.getcwd()
    )
    # Color grading is optional. The expensive interpolation is baked into a
    # flat lookup table once at startup; per-frame work is a single take().
    lut_table = None
    lut_path = resolve_lut_path(cfg["lut_file"], base_dir)
    lut_files = list_lut_files(lut_path) if cfg.get("lut_enabled", True) else []
    lut_index = 0
    if lut_files:
        try:
            lut_index = lut_files.index(lut_path)
        except ValueError:
            lut_index = 0
    if cfg.get("lut_enabled", True):
        lut = load_lut(lut_path)
        if lut is not None:
            print(f"LUT loaded: {lut_path} (size {lut.shape[0]})")
        lut_table = bake_lut(lut)  # bake once at startup

    camera_index = cfg["camera_index"]
    in_w, in_h = cfg["in_w"], cfg["in_h"]
    out_w, out_h = cfg["out_w"], cfg["out_h"]
    fps = cfg["fps"]
    zoom = cfg["zoom"]
    max_head = cfg["max_head_height"]
    head_ratio = cfg["head_ratio"]
    zoom_response_time = max(0.05, float(cfg["zoom_response_time"]))
    zoom_damping_ratio = max(0.1, float(cfg["zoom_damping_ratio"]))
    zoom_deadzone = cfg["zoom_deadzone"]
    smoothing = cfg["smoothing"]
    deadzone = cfg["deadzone"]
    min_det_conf = cfg["min_det_conf"]
    detect_every = max(1, int(cfg["detect_every"]))
    sharpen_enabled = bool(cfg["sharpen_enabled"])
    sharpen_strength = float(cfg["sharpen_strength"])
    sharpen_threshold = int(cfg["sharpen_threshold"])
    sharpen_sigma = float(cfg["sharpen_sigma"])
    preview = bool(args.preview)

    model_path = ensure_model()
    base_opts = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceDetectorOptions(
        base_options=base_opts, min_detection_confidence=min_det_conf
    )

    # Single-threaded OpenCV avoids occasional macOS crashes when its worker
    # threads compete with MediaPipe's thread pool.
    cv2.setNumThreads(1)

    # Request the desired capture format; the actual delivered size may differ.
    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, in_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, in_h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        print(f"error: cannot open camera {camera_index}")
        print("run with --list-cameras to see available indices")
        sys.exit(1)

    # Smoothed view state: pan center, zoom, and a frame counter that drives
    # the detection cadence.
    curr_x, curr_y = 0.5, 0.45
    target_x, target_y = curr_x, curr_y
    curr_zoom = target_zoom = zoom
    frame_idx = 0
    center_stage_enabled = True

    # Single-key terminal controls. atexit restores echo/canonical mode even on
    # Ctrl+C or an unhandled error.
    old_term = setup_keyboard()
    atexit.register(restore_keyboard, old_term)

    # Context managers guarantee the detector and virtual camera are closed
    # even when the loop exits via an error.
    with (
        vision.FaceDetector.create_from_options(options) as detector,
        pyvirtualcam.Camera(
            width=out_w, height=out_h, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR
        ) as cam,
    ):
        print(f"Virtual cam started: {cam.device}")
        if preview:
            print("Preview enabled. Press 'q' in the window to quit.")
        else:
            print("Running headless. Ctrl+C to stop.")
        consecutive_errors = 0

        # Critically damped spring for the dolly zoom (frame-rate independent).
        zoom_vel = 0.0
        dt = 1.0 / max(1.0, float(fps))
        omega = 2.0 * math.pi / zoom_response_time

        while True:
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
                        print("error: camera stopped delivering frames; exiting")
                        break
                    continue
                frame = np.ascontiguousarray(frame)
                consecutive_errors = 0

                # Keyboard controls: [ / ] (or Russian х / ъ) cycle LUTs,
                # Enter toggles center stage on/off.
                keys = read_keys() if old_term is not None else ""
                if keys:
                    if ("]" in keys or "ъ" in keys) and lut_files:
                        lut_index = cycle_index(lut_index, len(lut_files), 1)
                        new_lut = load_lut(lut_files[lut_index])
                        if new_lut is not None:
                            lut_table = bake_lut(new_lut)
                            print(f"LUT: {os.path.basename(lut_files[lut_index])}")
                    elif "[" in keys or "х" in keys:
                        if lut_files:
                            lut_index = cycle_index(lut_index, len(lut_files), -1)
                            new_lut = load_lut(lut_files[lut_index])
                            if new_lut is not None:
                                lut_table = bake_lut(new_lut)
                                print(f"LUT: {os.path.basename(lut_files[lut_index])}")
                    if "\r" in keys or "\n" in keys:
                        center_stage_enabled = not center_stage_enabled
                        print(
                            f"Center stage: {'ON' if center_stage_enabled else 'OFF'}"
                        )

                # Run face detection only every N frames; reuse last target otherwise.
                # Skip tracking entirely when center stage is toggled off.
                if center_stage_enabled and frame_idx % detect_every == 0:
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
                            zoom,
                            max_head,
                            head_ratio,
                        )
                        target_zoom = pick_target_zoom(
                            target_zoom, wanted, zoom_deadzone
                        )

                if center_stage_enabled:
                    # Exponential moving average gives smooth panning even
                    # though detection only updates every detect_every frames.
                    curr_x += (target_x - curr_x) * smoothing
                    curr_y += (target_y - curr_y) * smoothing

                    # --- Dolly zoom: critically damped spring (no overshoot) ---
                    curr_zoom, zoom_vel = spring_step(
                        curr_zoom, zoom_vel, target_zoom,
                        omega, zoom_damping_ratio, dt
                    )
                    # keep within the valid zoom range even if velocity drifts
                    curr_zoom = max(1.0, min(zoom, curr_zoom))
                    # ----------------------------------------------------------

                    # Use actual frame dimensions: the camera may negotiate a
                    # different resolution than requested.
                    fh, fw = frame.shape[:2]
                    x1, y1, crop_w, crop_h = crop_origin(
                        curr_x, curr_y, curr_zoom, fw, fh
                    )
                    cropped = frame[y1 : y1 + crop_h, x1 : x1 + crop_w]
                    if crop_w < 1 or crop_h < 1 or cropped.size == 0:
                        continue
                    cropped = np.ascontiguousarray(cropped)
                    out_frame = cv2.resize(
                        cropped, (out_w, out_h), interpolation=cv2.INTER_LINEAR
                    )
                else:
                    # Center stage off: publish the whole frame un-cropped.
                    out_frame = cv2.resize(
                        frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR
                    )

                # Color grade and sharpen the final image in both modes.
                if lut_table is not None:
                    out_frame = apply_lut(
                        out_frame,
                        lut_table,
                        scale=cfg["lut_scale"],
                        detail=cfg["lut_detail"],
                    )
                if sharpen_enabled:
                    out_frame = apply_smart_sharpen(
                        out_frame,
                        strength=sharpen_strength,
                        threshold=sharpen_threshold,
                        sigma=sharpen_sigma,
                    )

                # Publish and pace the loop to the virtual camera's frame rate.
                cam.send(out_frame)
                cam.sleep_until_next_frame()

                if preview:
                    cv2.imshow("center-stage", out_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_idx += 1
            except KeyboardInterrupt:
                raise
            except Exception as e:
                # Transient per-frame errors should not kill a call; only give
                # up after repeated failures, printing the stack for diagnosis.
                consecutive_errors += 1
                print(f"warning: frame processing error ({consecutive_errors}): {e}")
                if consecutive_errors > 10:
                    import traceback

                    traceback.print_exc()
                    break
                continue

    cap.release()
    if preview:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
