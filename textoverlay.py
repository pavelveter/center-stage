"""Centered, gradient-colored text overlay for the published video frame.

OpenCV's Python wheel in this environment cannot render system TTF fonts
(no Qt / FreeType text API), so the text is rasterized with Pillow and
blended into the frame with numpy. The rendered mask and gradient are
cached and rebuilt only when the text or its settings change, so the
per-frame cost is a single masked copy over the text bounding box.
"""

import os
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# font_scale 1.0 == 32px TTF, matching the familiar cv2.putText scale.
BASE_PX = 32
HEX_COLOR = re.compile(r"#[0-9a-fA-F]{6}\Z")

_cache_key = None
_cache_mask = None
_cache_gradient = None
_cache_origin = (0, 0)


def parse_color(value, fallback=(255, 255, 255)):
    """Parse a #RRGGBB string into RGB; malformed values use the fallback."""
    if isinstance(value, str) and HEX_COLOR.fullmatch(value.strip()):
        parts = value.strip().lstrip("#")
        return tuple(int(parts[i : i + 2], 16) for i in (0, 2, 4))
    return fallback


def resolve_font(name):
    """Find a TTF/OTF by name or path; None means the Pillow default font."""
    if not name or not isinstance(name, str):
        return None
    if os.path.isfile(name):
        return name
    wanted = name.strip().lower().replace(" ", "-").replace("_", "-")
    for folder in (
        os.path.expanduser("~/Library/Fonts"),
        "/Library/Fonts",
        "/System/Library/Fonts",
    ):
        if not os.path.isdir(folder):
            continue
        for entry in os.listdir(folder):
            stem, ext = os.path.splitext(entry)
            if ext.lower() in (".ttf", ".otf") and stem.lower() in (
                wanted,
                wanted + "-regular",
            ):
                return os.path.join(folder, entry)
    return None


def _font(path, size):
    return ImageFont.truetype(path, size) if path else ImageFont.load_default(size)


def _render(text, values, frame_w, frame_h, start, end):
    """Build the alpha mask, BGR gradient and top-left origin for the text."""
    path = resolve_font(values["text_font_face"])
    stroke = max(0, int(values["text_thickness"]) - 1)
    size = max(4, round(BASE_PX * float(values["text_font_scale"])))
    max_w = max(1, frame_w - 2 * max(0, int(values["text_padding"])))

    font = _font(path, size)
    bbox = font.getbbox(text, stroke_width=stroke)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Fit the safe zone: one proportional step plus a few 1px corrections
    # for rounding in the font metrics.
    for _ in range(4):
        if w <= max_w:
            break
        size = max(4, min(size - 1, int(size * max_w / w)))
        font = _font(path, size)
        bbox = font.getbbox(text, stroke_width=stroke)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    canvas = Image.new("L", (w, h), 0)
    ImageDraw.Draw(canvas).text((-bbox[0], -bbox[1]), text, font=font, fill=255,
                                stroke_width=stroke, stroke_fill=255)
    mask = np.asarray(canvas, dtype=np.uint8)

    # Smooth horizontal gradient from the start color to the end color.
    t = np.linspace(0.0, 1.0, w, dtype=np.float32)
    rgb = np.stack([a + (b - a) * t for a, b in zip(start, end)], axis=1)
    gradient = np.repeat(rgb[None, :, ::-1].astype(np.uint8), h, axis=0)  # RGB -> BGR

    x = max(0, (frame_w - w) // 2)
    y = max(0, (frame_h - h) // 2)
    return mask, gradient, (x, y)


def apply_text_overlay(frame, values):
    """Draw the cached centered gradient text onto a BGR frame, in place."""
    global _cache_key, _cache_mask, _cache_gradient, _cache_origin

    if not values["text_enabled"]:
        return
    text = str(values.get("text_string") or "").strip()
    if not text:
        return

    start = parse_color(values["text_color_start"])
    end = parse_color(values["text_color_end"])
    key = (
        text,
        frame.shape[1],
        frame.shape[0],
        values["text_font_face"],
        float(values["text_font_scale"]),
        int(values["text_thickness"]),
        int(values["text_padding"]),
        start,
        end,
    )
    if key != _cache_key:
        _cache_mask, _cache_gradient, _cache_origin = _render(
            text, values, frame.shape[1], frame.shape[0], start, end
        )
        _cache_key = key

    x, y = _cache_origin
    h, w = _cache_mask.shape
    roi = frame[y : y + h, x : x + w]
    if roi.shape[:2] == (h, w):
        np.copyto(roi, _cache_gradient, where=(_cache_mask > 0)[:, :, None])
