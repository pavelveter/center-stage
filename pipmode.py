"""PiP (Picture-in-Picture) slide mode.

A static image (slide or meme) fills the output frame while the live webcam
feed shrinks to a small window snapped into one of the four corners. Images
are discovered when dropped into the app directory and composed per frame by
the video pipeline.
"""

import os
import subprocess

import cv2
import numpy as np

# Case-insensitive: checked through .lower().
IMAGE_EXTENSIONS = (".gif", ".jpg", ".jpeg", ".png")

# Corner order used by config (pip_corner), state and the arrow-key bindings.
CORNER_TL, CORNER_TR, CORNER_BL, CORNER_BR = 0, 1, 2, 3
CORNER_NAMES = ("top left", "top right", "bottom left", "bottom right")

# Each arrow points at a unique corner, clockwise from left: <- TL, ^ TR,
# v BL, -> BR.
ARROW_CORNERS = {
    "left": CORNER_TL,
    "up": CORNER_TR,
    "down": CORNER_BL,
    "right": CORNER_BR,
}


def is_image_file(name):
    """Case-insensitive extension check for supported image formats."""
    return str(name).lower().endswith(IMAGE_EXTENSIONS)


def list_pip_files(primary_path):
    """Sorted, de-duplicated image files usable as PiP choices.

    Looks next to the configured image and in the app directory, so a file
    dropped into either place shows up in the TUI list.
    """
    folders = []
    if primary_path:
        folders.append(os.path.dirname(os.path.abspath(primary_path)))
    folders.append(os.getcwd())
    found = {}
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if is_image_file(name):
                found.setdefault(os.path.abspath(os.path.join(folder, name)), None)
    return sorted(found, key=str.lower)


def load_pip_image(path, size):
    """Decode an image and letterbox-fit it to (width, height) BGR uint8.

    Aspect ratio is preserved with black bars, so slides and memes are never
    cropped or distorted. Animated .gif files use their first frame.
    Returns None when the file cannot be read.
    """
    if not path or not os.path.exists(path):
        return None
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        return None
    out_w, out_h = size
    src_h, src_w = image.shape[:2]
    scale = min(out_w / src_w, out_h / src_h)
    scaled_w = max(1, min(out_w, round(src_w * scale)))
    scaled_h = max(1, min(out_h, round(src_h * scale)))
    resized = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    x = (out_w - scaled_w) // 2
    y = (out_h - scaled_h) // 2
    canvas[y : y + scaled_h, x : x + scaled_w] = resized
    return canvas


def pip_corner_rect(corner, scale, margin, out_w, out_h):
    """(x, y, w, h) of the webcam window for one corner.

    scale is a fraction of the whole output frame; margin is a fraction of
    the frame width applied to every side, so the visual padding is equal
    on the horizontal and vertical edges. corner is one of the CORNER_*
    constants. Results are rounded and always inside the frame.
    """
    w = max(1, min(out_w, round(out_w * scale)))
    h = max(1, min(out_h, round(out_h * scale)))
    m = max(0, min(min(out_w - w, out_h - h), round(out_w * margin)))
    x = m if corner in (CORNER_TL, CORNER_BL) else out_w - w - m
    y = m if corner in (CORNER_TL, CORNER_TR) else out_h - h - m
    return x, y, w, h


def pip_transition_rect(progress, corner, scale, margin, out_w, out_h):
    """(x, y, w, h) of the webcam window mid-animation.

    progress 0.0 is the full output frame (PiP off), 1.0 the corner
    window (PiP on); in between the rect is eased with a smoothstep and
    always stays inside the frame.
    """
    x, y, w, h = pip_corner_rect(corner, scale, margin, out_w, out_h)
    eased = progress * progress * (3.0 - 2.0 * progress)
    rx = round(x * eased)
    ry = round(y * eased)
    rw = max(1, min(out_w, round(out_w + (w - out_w) * eased)))
    rh = max(1, min(out_h, round(out_h + (h - out_h) * eased)))
    return rx, ry, rw, rh


def compose_pip_rect(canvas, background, webcam, rect):
    """Background image with the webcam pasted at an explicit (x, y, w, h)."""
    bg = background
    if bg.shape[:2] != canvas.shape[:2]:
        bg = cv2.resize(
            bg, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_LINEAR
        )
    out = bg.copy()
    x, y, pw, ph = rect
    if webcam.shape[:2] != (ph, pw):
        webcam = cv2.resize(webcam, (pw, ph), interpolation=cv2.INTER_AREA)
    out[y : y + ph, x : x + pw] = webcam
    return out


def compose_pip(canvas, background, webcam, corner, scale, margin):
    """Background image plus the webcam window snapped into one corner.

    background is a pre-fitted, display-ready frame (see load_pip_image); it
    is copied, never mutated, because SharedState shares it with the video
    thread. The webcam frame is resized to the corner window size.
    """
    h, w = canvas.shape[:2]
    rect = pip_corner_rect(corner, scale, margin, w, h)
    return compose_pip_rect(canvas, background, webcam, rect)


def pick_image_dialog(title="Select a slide or meme image"):
    """Native macOS file dialog via osascript; returns a path or None.

    Blocks its calling thread only, so the TUI must invoke it from a worker.
    Cancel (or any osascript failure) returns None; the extension is verified
    again here because the dialog filter is a convenience, not a guarantee.
    """
    script = (
        f'POSIX path of (choose file with prompt "{title}" '
        'of type {"gif", "jpg", "jpeg", "png"})'
    )
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,  # the return code itself means "user cancelled"
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    path = result.stdout.strip()
    return path if path and is_image_file(path) else None
