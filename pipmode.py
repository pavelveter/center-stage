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
    return sorted(found)


def load_pip_image(path, size):
    """Decode an image and letterbox-fit it to (width, height) BGR uint8.

    Aspect ratio is preserved with black bars, so slides and memes are never
    cropped or distorted. Animated .gif files use their first frame.
    Returns (frame, content_rect) where content_rect is (x, y, w, h) of the
    actual picture inside the frame, or (None, None) when unreadable.
    """
    if not path or not os.path.exists(path):
        return None, None
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        return None, None
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
    return canvas, (x, y, scaled_w, scaled_h)


def pip_corner_rect(corner, scale, margin, out_w, out_h, bounds=None):
    """(x, y, w, h) of the webcam window for one corner.

    scale and margin are fractions of the full output frame; corner is one of
    the CORNER_* constants. bounds limits placement: when the background is
    letterboxed, the window snaps inside the picture with its margin measured
    from the picture's edges, not the frame's. Results stay inside bounds.
    """
    if bounds is None:
        bounds = (0, 0, out_w, out_h)
    bx, by, bw, bh = bounds
    # Window size is a fraction of the full frame but never larger than the
    # picture it must sit in.
    w = max(1, min(bw, round(out_w * scale)))
    h = max(1, min(bh, round(out_h * scale)))
    mx = max(0, min(bw - w, round(bw * margin)))
    my = max(0, min(bh - h, round(bh * margin)))
    x = bx + (mx if corner in (CORNER_TL, CORNER_BL) else bw - w - mx)
    y = by + (my if corner in (CORNER_TL, CORNER_TR) else bh - h - my)
    return x, y, w, h


def compose_pip(canvas, background, webcam, corner, scale, margin, content=None):
    """Background image plus the webcam window snapped into one corner.

    background is a pre-fitted frame (see load_pip_image); it is copied, never
    mutated, because SharedState shares it with the video thread. The webcam
    frame is resized to the corner window size. content is the background's
    picture rectangle (from load_pip_image); the window keeps its margin from
    the picture's edges on every side.
    """
    bg = background
    if bg.shape[:2] != canvas.shape[:2]:
        bg = cv2.resize(
            bg, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_LINEAR
        )
    out = bg.copy()
    h, w = canvas.shape[:2]
    x, y, pw, ph = pip_corner_rect(corner, scale, margin, w, h, bounds=content)
    if webcam.shape[:2] != (ph, pw):
        webcam = cv2.resize(webcam, (pw, ph), interpolation=cv2.INTER_AREA)
    out[y : y + ph, x : x + pw] = webcam
    return out


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
