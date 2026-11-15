"""Magic Loop (AFK): rolling history, loop search and crossfaded player."""

import collections

import cv2
import numpy as np


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
