"""Single-frame effects (smart sharpen, deep fry) and the Time Loop."""

import collections

import cv2
import numpy as np


# One second of processed frames at the nominal 30 FPS.
TIME_LOOP_FRAMES = 30


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
