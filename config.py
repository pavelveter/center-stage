"""Configuration: defaults, INI mapping, load/save and path resolution."""

import configparser
import math
import os
import tempfile


# Defaults used when the INI file omits a key or no config is found. Keeping
# them in one dict makes config loading and CLI overrides uniform: every value
# flows through the same validate/override path.
DEFAULTS = {
    "camera_index": 1,          # AVFoundation device index (0 = FaceTime on macOS)
    "cam_flip": False,          # mirror the camera feed horizontally (selfie view)
    "text_enabled": True,       # draw the centered text overlay when text is set
    "text_string": "",          # active overlay text (Enter in the TUI input)
    "text_font_face": "Nothing Font C",  # TTF name/path resolved by textoverlay
    "text_font_scale": 1.5,     # base font scale for the overlay
    "text_thickness": 3,        # text outline/stroke width in pixels
    "text_padding": 50,         # minimum gap from the frame's left/right edges
    "text_color_start": "#FF5F6D",       # gradient start (left)
    "text_color_end": "#FFC371",         # gradient end (right)
    "in_w": 2560,               # requested capture width
    "in_h": 1440,               # requested capture height
    "out_w": 1920,              # published virtual-camera width
    "out_h": 1080,              # published virtual-camera height
    "fps": 30,                  # capture and virtual-camera frame rate
    "tracking_enabled": True,   # face tracking on/off (off = base zoom, centered)
    "zoom": 1.5,                # base crop factor (higher = tighter framing)
    "max_head_height": 0.667,   # max head height as a fraction of frame height
    "head_ratio": 1.35,         # full head / face detector bbox
    "zoom_response_time": 0.9,  # legacy: migrated to zoom_in/out response times
    "zoom_damping_ratio": 1.0,  # legacy: kept for config compatibility
    "zoom_deadzone": 0.05,      # relative hysteresis on the zoom target
    "position_response_time": 0.7,  # spring response time for pan (seconds)
    "position_damping": 1.0,    # >=1 prevents pan overshoot
    "position_deadband": 0.01,  # ignore center changes below 1% of the frame
    "zoom_in_response_time": 4.0,   # slow, deliberate zoom-in (seconds)
    "zoom_out_response_time": 1.0,  # responsive zoom-out (seconds)
    "zoom_measurement_smoothing": 0.6,  # time constant for face-size EMA (s)
    "detection_timeout": 0.5,   # seconds before easing to default framing
    "max_measurement_change": 0.6,  # max face-size change per second
    "tracking_debug": False,    # publish extra tracking telemetry
    "smoothing": 0.08,          # legacy EMA alpha (migrated on load)
    "deadzone": 0.04,           # legacy pan deadzone (migrated on load)
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
    "pip_enabled": False,       # PiP slide mode: image background + small webcam
    "pip_image": "",            # path to the background image (empty = none)
    "pip_flip": False,          # mirror the background image horizontally
    "pip_scale": 0.25,          # webcam window size as a fraction of the frame
    "pip_padding": 96,          # webcam window offset from the screen edges, pixels
    "pip_animation": 3.0,       # PiP on/off zoom duration in seconds (0 = instant)
    "pip_corner": 0,            # default webcam corner: 0 TL, 1 TR, 2 BL, 3 BR
    "theme": "textual-dark",    # Textual TUI theme (change via the command palette)
}

# Maps INI sections/keys (human-facing) to config keys (code-facing). Both
# load_config and save_config use this single table, so reading and writing
# settings can never drift apart.
INI_MAP = {
    "camera": {"index": "camera_index", "flip": "cam_flip"},
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
        "position_response_time": "position_response_time",
        "position_damping": "position_damping",
        "position_deadband": "position_deadband",
        "zoom_in_response_time": "zoom_in_response_time",
        "zoom_out_response_time": "zoom_out_response_time",
        "zoom_measurement_smoothing": "zoom_measurement_smoothing",
        "detection_timeout": "detection_timeout",
        "max_measurement_change": "max_measurement_change",
        "debug": "tracking_debug",
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
    "pip": {
        "enabled": "pip_enabled",
        "image": "pip_image",
        "flip": "pip_flip",
        "scale": "pip_scale",
        "padding": "pip_padding",
        "animation": "pip_animation",
        "corner": "pip_corner",
    },
    "appearance": {"theme": "theme"},
    "text": {
        "enabled": "text_enabled",
        "text": "text_string",
        "font_face": "text_font_face",
        "font_scale": "text_font_scale",
        "thickness": "text_thickness",
        "padding": "text_padding",
        "color_start": "text_color_start",
        "color_end": "text_color_end",
    },
}


# Explanations written above every key by save_config, so the live ini stays
# self-documenting through autosaves instead of losing comments on each write.
SECTION_HELP = {
    "camera": "Webcam device and the selfie-mirror of the published stream.",
    "capture": "Camera capture and the resolution/fps of the virtual camera.",
    "tracking": (
        "Center Stage: pan and zoom are independent controllers. Times are in "
        "seconds, deadbands are fractions of the frame."
    ),
    "color": "LUT grading and GLSL shaders (applied to the camera, not the PiP image).",
    "sharpen": "Edge-gated unsharp mask applied to the camera feed.",
    "effects": "Time Loop / Deep Fry / Magic Loop (AFK) toggles.",
    "pip": "Picture-in-Picture slide/meme background and its webcam window.",
    "appearance": "TUI theme (change it with Ctrl+P -> theme).",
    "text": "Centered gradient caption published in the TUI text field.",
}

KEY_HELP = {
    ("camera", "index"): "AVFoundation camera index; run --list-cameras to see options.",
    ("camera", "flip"): "Mirror the whole feed horizontally (selfie view).",
    ("capture", "in_width"): "Requested capture width; the camera may negotiate a different one.",
    ("capture", "in_height"): "Requested capture height; detection always runs on a 640x360 copy.",
    ("capture", "out_width"): "Virtual camera output width.",
    ("capture", "out_height"): "Virtual camera output height; crop keeps this aspect ratio.",
    ("capture", "fps"): "Capture and virtual-camera frame rate.",
    ("tracking", "enabled"): "Master switch for face tracking (hotkey t / е).",
    ("tracking", "zoom"): "Base/max zoom factor; the controller never exceeds it. TUI: Zoom tuner.",
    ("tracking", "smoothing"): "LEGACY EMA alpha, migrated to position_response_time on load.",
    ("tracking", "deadzone"): "LEGACY pan deadzone, migrated to position_deadband on load.",
    ("tracking", "min_detection_confidence"): "MediaPipe detector confidence floor (0..1).",
    ("tracking", "detect_every"): "Run inference every Nth frame (6 = ~5 detections/s at 30 fps).",
    ("tracking", "max_head_height"): "Head may occupy at most this fraction of the output height.",
    ("tracking", "head_ratio"): "Full head / detected face bbox height; maps face size to framing.",
    ("tracking", "zoom_response_time"): "LEGACY spring time, migrated to zoom_in/out_response_time.",
    ("tracking", "zoom_damping_ratio"): "LEGACY spring damping; kept for old configs, unused now.",
    ("tracking", "position_response_time"): "Pan spring response time: larger = slower, calmer pan. TUI: Position Response.",
    ("tracking", "position_damping"): "Pan damping (>=1 means no overshoot / oscillation).",
    ("tracking", "position_deadband"): "Ignore face-center moves below this fraction of the frame (0.01 = 1%).",
    ("tracking", "zoom_in_response_time"): "Time constant for zoom-in: deliberately slow to avoid breathing.",
    ("tracking", "zoom_out_response_time"): "Time constant for zoom-out: faster, keeps subject in frame.",
    ("tracking", "zoom_measurement_smoothing"): "Low-pass time constant for the filtered face-size measurement.",
    ("tracking", "zoom_deadzone"): "Relative hysteresis on the zoom target (0.05 = 5%).",
    ("tracking", "detection_timeout"): "After this long without a detection, ease back to default framing.",
    ("tracking", "max_measurement_change"): "Reject face-size changes faster than this fraction per second.",
    ("tracking", "debug"): "Publish raw/filtered center/size, zoom, age, velocity in telemetry.",
    ("color", "lut_enabled"): "Apply the .cube LUT (hotkey c / с).",
    ("color", "lut_file"): "Path to the .cube file (relative to this ini is fine).",
    ("color", "lut_scale"): "LUT downscale factor: higher = faster but softer grading.",
    ("color", "lut_detail"): "Upscale only the color delta to keep fine detail.",
    ("color", "shader_enabled"): "Apply the GLSL shader pack (hotkey h / р).",
    ("color", "shader_file"): "Path to a single .glsl or a shader pack directory.",
    ("sharpen", "enabled"): "Edge-aware sharpening (hotkey s / ы).",
    ("sharpen", "strength"): "Unsharp-mask amount (0..1 looks natural).",
    ("sharpen", "threshold"): "Minimum local contrast before sharpening kicks in.",
    ("sharpen", "sigma"): "Gaussian radius of the high-pass filter.",
    ("effects", "time_loop"): "Replay the last second of processed video (hotkey l / д).",
    ("effects", "deep_fry"): "High-contrast saturation + RGB channel roll (hotkey f / а).",
    ("effects", "magic_loop"): "AFK: freeze framing and loop pre-recorded video (hotkey m / ь).",
    ("pip", "enabled"): "PiP slide/meme mode (hotkey p / з).",
    ("pip", "image"): "Background image path; Pick button or dropdown in the TUI.",
    ("pip", "flip"): "Mirror the background image horizontally.",
    ("pip", "scale"): "Webcam window size as a fraction of the output frame (0.25 = 25%).",
    ("pip", "padding"): "Equal pixel gap from every screen edge to the webcam window.",
    ("pip", "animation"): "Zoom transition duration in seconds when PiP toggles (0 = instant).",
    ("pip", "corner"): "Default corner: 0 TL, 1 TR, 2 BL, 3 BR; arrows move it live.",
    ("pip", "margin"): "IGNORED legacy key (replaced by padding); safe to delete.",
    ("appearance", "theme"): "Textual theme name.",
    ("text", "enabled"): "Draw the centered text overlay; Enter in the input turns it on.",
    ("text", "text"): "Published caption (persisted from the TUI input).",
    ("text", "font_face"): "TTF name or absolute path; 'Nothing Font C' resolves from ~/Library/Fonts.",
    ("text", "font_scale"): "Base font scale (1.0 = 32px); long lines shrink to fit.",
    ("text", "thickness"): "Text stroke width in pixels.",
    ("text", "padding"): "Minimum gap in pixels from the frame's left/right edges.",
    ("text", "color_start"): "Gradient start color at the left edge, #RRGGBB.",
    ("text", "color_end"): "Gradient end color at the right edge, #RRGGBB.",
}


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
    _migrate_tracking_settings(parser, cfg)
    return cfg


def _migrate_tracking_settings(parser, cfg):
    """Map old tracking keys onto the new controllers when no new key is set.

    Old configs keep working: an EMA alpha is converted to an equivalent
    response time at the configured FPS, and the old single zoom spring time
    seeds the new asymmetric in/out times. Explicit new values always win.
    """
    if not parser.has_section("tracking"):
        return
    has = lambda key: parser.has_option("tracking", key)

    def number(key):
        raw = parser.get("tracking", key).strip()
        try:
            return float(raw)
        except ValueError:
            return None

    if has("smoothing") and not has("position_response_time"):
        alpha = number("smoothing")
        if alpha is not None and 0.0 < alpha < 1.0:
            fps = max(1.0, cfg["fps"])
            cfg["position_response_time"] = min(
                4.0, max(0.2, -1.0 / fps / math.log(1.0 - alpha))
            )
    if has("deadzone") and not has("position_deadband"):
        deadband = number("deadzone")
        if deadband is not None:
            cfg["position_deadband"] = min(0.1, max(0.002, deadband))
    if has("zoom_response_time") and not has("zoom_in_response_time"):
        old = number("zoom_response_time")
        if old is not None:
            cfg["zoom_out_response_time"] = min(4.0, max(0.3, old * 2.0))
            cfg["zoom_in_response_time"] = min(12.0, max(1.0, old * 8.0))


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
            if cfg_key in ("lut_file", "pip_image"):
                value = portable_path(value, directory)
            parser.set(section, ini_key, ini_value(value))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".hyper-stage-", suffix=".ini", dir=directory
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.write("; Hyper Stage configuration.\n")
            f.write("; Comments are written on every autosave; edit values freely.\n")
            for section in parser.sections():
                f.write("\n")
                if section in SECTION_HELP:
                    f.write(f"; {SECTION_HELP[section]}\n")
                f.write(f"[{section}]\n")
                for key, value in parser.items(section, raw=True):
                    help_text = KEY_HELP.get((section, key))
                    if help_text:
                        f.write(f"; {help_text}\n")
                    f.write(f"{key} = {value}\n")
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
