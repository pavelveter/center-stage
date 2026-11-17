"""Deterministic, hardware-free benchmark: synchronous vs three-stage pipeline.

The mock camera produces real-resolution (2560x1440) frames at the configured
capture rate and the mock detector takes roughly as long as the measured
BlazeFace inference (~28 ms). The same FrameProcessor runs the effect chain in
both modes, so the only difference is where inference happens.

Run: .venv/bin/python benchmark.py --frames 180
"""

import argparse
import time
import types
from unittest import mock

import cv2
import numpy as np
import pyvirtualcam
from mediapipe.tasks.python import vision

from hyper_stage.config import DEFAULTS
from hyper_stage.detector import DetectorWorker
from hyper_stage.pipeline import VideoPipeline
from hyper_stage.renderer import FrameProcessor
from hyper_stage.state import SharedState

INFERENCE_SECONDS = 0.028  # measured on this M1 with the cached BlazeFace model


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    import math

    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


class SyntheticCamera:
    def __init__(self, cfg):
        self.cfg = cfg
        self.released = False
        self._seq = 0
        self._interval = 1.0 / float(cfg["fps"])
        self._next = time.monotonic()

    def isOpened(self):
        return True

    def set(self, *args):
        return True

    def read(self):
        self._next += self._interval
        delay = self._next - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            self._next = time.monotonic()
        self._seq += 1
        frame = np.full((self.cfg["in_h"], self.cfg["in_w"], 3), 128, np.uint8)
        shift = (self._seq * 8) % (self.cfg["in_w"] - 200)
        frame[:, shift : shift + 200] = 220
        return True, frame

    def release(self):
        self.released = True


class SlowDetector:
    def __init__(self):
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        time.sleep(INFERENCE_SECONDS)
        return types.SimpleNamespace(detections=[])


class DetectorContext:
    def __init__(self, detector):
        self.detector = detector

    def __enter__(self):
        return self.detector

    def __exit__(self, *args):
        return False


class RecordingVcam:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = "mock-virtual-cam"
        self.timestamps = []
        self.durations = []
        self._sent_at = None

    def send(self, frame):
        now = time.monotonic()
        self.timestamps.append(now)
        if self._sent_at is not None:
            self.durations.append(now - self._sent_at)
        self._sent_at = now

    def sleep_until_next_frame(self):
        # The real pyvirtualcam paces the stream; the mock must do the same or
        # the synchronous baseline would unfairly run unthrottled.
        interval = 1.0 / float(self.cfg["fps"])
        target = self._sent_at + interval if self._sent_at else time.monotonic()
        delay = target - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def close(self):
        pass


def make_cfg(fps, detect_every):
    cfg = dict(DEFAULTS)
    cfg.update(
        in_w=2560, in_h=1440, out_w=1920, out_h=1080,
        fps=fps, detect_every=detect_every,
        tracking_enabled=True,  # both modes pay for the same inference work
        lut_enabled=False, sharpen_enabled=True, shader_enabled=False,
    )
    return cfg


def summarize(name, vcam, render_durations, latency, detections):
    intervals = [d for d in vcam.durations if d > 0]
    wall = (
        vcam.timestamps[-1] - vcam.timestamps[0]
        if len(vcam.timestamps) > 1 else 0.0
    )
    fps = (len(vcam.timestamps) - 1) / wall if wall else 0.0
    return {
        "mode": name,
        "frames": len(vcam.timestamps),
        "average_fps": fps,
        "frame_interval_p50_ms": percentile(intervals, 0.5) * 1000,
        "frame_interval_p95_ms": percentile(intervals, 0.95) * 1000,
        "frame_interval_p99_ms": percentile(intervals, 0.99) * 1000,
        "render_p50_ms": percentile(render_durations, 0.5) * 1000,
        "render_p95_ms": percentile(render_durations, 0.95) * 1000,
        "render_p99_ms": percentile(render_durations, 0.99) * 1000,
        "latency_p50_ms": percentile(latency, 0.5) * 1000,
        "detections": detections,
    }


def run_baseline(cfg, frames):
    """The original design: capture, inline inference and rendering in one loop."""
    state = SharedState(cfg)
    processor = FrameProcessor(cfg, state)
    camera = SyntheticCamera(cfg)
    vcam = RecordingVcam(cfg)
    detector = SlowDetector()
    detect_every = max(1, int(cfg["detect_every"]))
    durations, latency, detections = [], [], 0
    for index in range(frames):
        ret, frame = camera.read()
        if not ret:
            break
        captured = time.monotonic()
        values = state.snapshot()
        if index % detect_every == 0:
            # The blocking call is what stalls this design's output cadence.
            detector.detect(np.zeros((360, 640, 3), np.uint8))
            detections += 1
        x, y, zoom = processor.step(1.0 / cfg["fps"], captured, values)
        render_start = time.monotonic()
        out = processor.render(frame, values, render_start, x, y, zoom)
        durations.append(time.monotonic() - render_start)
        latency.append(time.monotonic() - captured)
        vcam.send(out)
        vcam.sleep_until_next_frame()
    processor.close()
    camera.release()
    return summarize("synchronous", vcam, durations, latency, detections)


def run_concurrent(cfg, frames):
    state = SharedState(cfg)
    vcam = RecordingVcam(cfg)
    detector = SlowDetector()
    camera = SyntheticCamera(cfg)
    pipeline = VideoPipeline(cfg, state)
    with (
        mock.patch.object(cv2, "VideoCapture", return_value=camera),
        mock.patch.object(pyvirtualcam, "Camera", return_value=vcam),
        mock.patch("hyper_stage.detector.ensure_model", return_value="mock"),
        mock.patch.object(
            vision.FaceDetector, "create_from_options",
            return_value=DetectorContext(detector),
        ),
    ):
        pipeline.start()
        deadline = time.monotonic() + 60
        while len(vcam.timestamps) < frames and time.monotonic() < deadline:
            time.sleep(0.01)
        pipeline.stop()
        pipeline.join(3.0)
    durations = list(pipeline.renderer._duration.values()) if pipeline.renderer else []
    latency = list(pipeline.renderer._latency.values()) if pipeline.renderer else []
    return summarize("three-stage", vcam, durations, latency, detector.calls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--detect-every", type=int, default=6)
    args = parser.parse_args()
    cfg = make_cfg(args.fps, args.detect_every)

    results = [run_baseline(cfg, args.frames), run_concurrent(cfg, args.frames)]
    fields = [
        "mode", "frames", "average_fps", "frame_interval_p50_ms",
        "frame_interval_p95_ms", "frame_interval_p99_ms", "render_p50_ms",
        "render_p95_ms", "render_p99_ms", "latency_p50_ms", "detections",
    ]
    print(f"{'metric':<26}{'synchronous':>16}{'three-stage':>16}")
    for field in fields:
        left, right = results[0][field], results[1][field]
        if isinstance(left, float):
            left, right = f"{left:.2f}", f"{right:.2f}"
        print(f"{field:<26}{str(left):>16}{str(right):>16}")


if __name__ == "__main__":
    main()
