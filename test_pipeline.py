"""Deterministic tests for the three-stage video pipeline (no real devices)."""

import threading
import time
import types
import unittest
from unittest import mock

import numpy as np

from capture import CameraWorker
from config import DEFAULTS
from detector import DetectorWorker
from pipeline import VideoPipeline
from renderer import RendererWorker
from state import SharedState
from framebus import FrameEnvelope, LatestFrame, LatestTracking, TrackingSnapshot
from metrics import RollingStats


class FakeCamera:
    """Minimal VideoCapture stand-in driven by a queued list of results."""

    def __init__(self, results, delay=0.001):
        self.results = list(results)
        self.delay = delay
        self.released = False
        self.index = 0

    def read(self):
        if self.delay:
            time.sleep(self.delay)
        if self.index >= len(self.results):
            return False, None
        result = self.results[self.index]
        self.index += 1
        return result

    def release(self):
        self.released = True


def good_frames(count, size=(8, 12)):
    return [(True, np.full((size[1], size[0], 3), i % 250, np.uint8))
            for i in range(count)]


class TestCameraWorker(unittest.TestCase):
    @staticmethod
    def cfg():
        return {"camera_index": 0, "in_w": 8, "in_h": 12, "fps": 30}

    def test_publishes_immutable_frames_with_monotonic_metadata(self):
        cam = FakeCamera(good_frames(5))
        frames = LatestFrame()
        stop = threading.Event()
        worker = CameraWorker(self.cfg(), frames, stop, cap_factory=lambda: cam)
        worker.start()
        deadline = time.monotonic() + 3
        while worker.stats()["capture_reads"] < 5 and time.monotonic() < deadline:
            time.sleep(0.005)
        stop.set()
        worker.join(timeout=3)
        item = frames.latest()
        self.assertIsNotNone(item)
        self.assertEqual(worker.stats()["capture_reads"], 5)
        self.assertTrue(cam.released)
        self.assertFalse(item.frame.flags.writeable)
        self.assertEqual(item.sequence, 5)
        self.assertGreater(item.captured_at, 0)

    def test_malformed_frames_are_rejected_and_counted(self):
        results = [(False, None), (True, None), (True, np.zeros((4, 4), np.uint8))]
        results += good_frames(2)
        cam = FakeCamera(results)
        frames = LatestFrame()
        stop = threading.Event()
        worker = CameraWorker(self.cfg(), frames, stop, cap_factory=lambda: cam)
        worker.start()
        deadline = time.monotonic() + 3
        while worker.stats()["capture_reads"] < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertGreaterEqual(worker.stats()["capture_failures"], 3)
        stop.set()
        worker.join(timeout=3)
        self.assertGreaterEqual(worker.stats()["capture_failures"], 3)
        self.assertEqual(worker.stats()["capture_reads"], 2)
        self.assertTrue(cam.released)

    def test_open_failure_surfaces_through_error_callback(self):
        errors = []
        frames = LatestFrame()
        stop = threading.Event()
        worker = CameraWorker(
            self.cfg(), frames, stop, on_error=errors.append,
            cap_factory=lambda: (_ for _ in ()).throw(RuntimeError("no camera")),
        )
        worker.start()
        worker.join(timeout=3)
        self.assertTrue(worker.wait_ready(0.1))
        self.assertEqual(len(errors), 1)
        self.assertIn("no camera", str(errors[0]))


class TestLatestFrame(unittest.TestCase):
    @staticmethod
    def envelope(seq, ts=None):
        frame = np.full((4, 6, 3), seq % 251, dtype=np.uint8)
        frame.setflags(write=False)
        return FrameEnvelope(frame, seq, ts if ts is not None else float(seq))

    def test_overwrite_and_newest_retrieval(self):
        bus = LatestFrame()
        self.assertIsNone(bus.latest())
        first = self.envelope(1)
        self.assertTrue(bus.publish(first))
        second = self.envelope(2)
        self.assertTrue(bus.publish(second))
        self.assertIs(bus.latest(), second)
        self.assertEqual(bus.stats()["published"], 2)
        self.assertEqual(bus.stats()["overwritten"], 1)

    def test_no_unbounded_accumulation(self):
        bus = LatestFrame()
        for seq in range(1000):
            bus.publish(self.envelope(seq))
        self.assertEqual(bus.latest().sequence, 999)
        self.assertEqual(bus.stats()["published"], 1000)

    def test_timestamps_and_sequences_preserved(self):
        bus = LatestFrame()
        env = self.envelope(7, 123.5)
        bus.publish(env)
        got = bus.latest()
        self.assertEqual(got.sequence, 7)
        self.assertEqual(got.captured_at, 123.5)
        self.assertEqual(int(got.frame[0, 0, 0]), 7)
        self.assertFalse(got.frame.flags.writeable)

    def test_older_sequence_is_rejected(self):
        bus = LatestFrame()
        bus.publish(self.envelope(5))
        self.assertFalse(bus.publish(self.envelope(4)))
        self.assertEqual(bus.latest().sequence, 5)

    def test_concurrent_publish_and_consume_is_thread_safe(self):
        bus = LatestFrame()
        stop = threading.Event()

        def producer():
            for seq in range(5000):
                bus.publish(self.envelope(seq))

        seen = []

        def consumer():
            while not stop.is_set():
                item = bus.latest()
                if item is not None:
                    seen.append(item.sequence)

        reader = threading.Thread(target=consumer)
        reader.start()
        writer = threading.Thread(target=producer)
        writer.start()
        writer.join()
        stop.set()
        reader.join()
        self.assertTrue(seen)
        self.assertEqual(seen, sorted(seen))


class TestLatestTracking(unittest.TestCase):
    @staticmethod
    def snapshot(seq, valid=True):
        return TrackingSnapshot(
            center_x=0.5, center_y=0.45, size=0.3, confidence=0.9,
            captured_at=float(seq), sequence=seq,
            completed_at=float(seq) + 0.01, valid=valid,
        )

    def test_newest_snapshot_wins_and_older_is_discarded(self):
        bus = LatestTracking()
        self.assertTrue(bus.publish(self.snapshot(2)))
        self.assertFalse(bus.publish(self.snapshot(1)))
        self.assertEqual(bus.latest().sequence, 2)
        self.assertEqual(bus.stats()["stale_discarded"], 1)

    def test_snapshot_is_immutable(self):
        snap = self.snapshot(1)
        with self.assertRaises(Exception):
            snap.center_x = 0.2


def fake_detection(cx=0.5, cy=0.45, height=0.3, score=0.9):
    det_h = 360
    bb = types.SimpleNamespace(
        origin_x=(cx - 0.2) * 640, origin_y=(cy - height * 0.6) * det_h,
        width=0.4 * 640, height=height * det_h,
    )
    return types.SimpleNamespace(
        bounding_box=bb, keypoints=None,
        categories=[types.SimpleNamespace(score=score)],
    )


class FakeDetector:
    def __init__(self, results=None, delay=0.0, fail_times=0):
        self.results = results
        self.delay = delay
        self.fail_times = fail_times
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.calls <= self.fail_times:
            raise RuntimeError("bad frame")
        detections = self.results
        if detections is None:
            detections = [fake_detection()]
        return types.SimpleNamespace(detections=detections)


class FakeDetectorContext:
    def __init__(self, detector):
        self.detector = detector
        self.closed = False

    def __enter__(self):
        return self.detector

    def __exit__(self, *args):
        self.closed = True


def publish_frames(bus, count, start=1, delay=0.0):
    for seq in range(start, start + count):
        frame = np.full((8, 12, 3), seq % 250, np.uint8)
        frame.setflags(write=False)
        bus.publish(FrameEnvelope(frame, seq, 100.0 + seq * 0.01))
        if delay:
            time.sleep(delay)


class TestDetectorWorker(unittest.TestCase):
    @staticmethod
    def cfg(detect_every=6):
        return {**DEFAULTS, "detect_every": detect_every, "min_det_conf": 0.6}

    def run_detector(self, cfg, detector, frames, snapshots, state=None):
        stop = threading.Event()
        ctx = FakeDetectorContext(detector)
        state = state or SharedState(cfg)
        worker = DetectorWorker(
            cfg, state, frames, snapshots, stop,
            detector_factory=lambda: (ctx, detector),
        )
        worker.start()
        return stop, ctx, worker

    def test_detect_every_is_respected(self):
        cfg = self.cfg(6)
        frames, snapshots = LatestFrame(), LatestTracking()
        detector = FakeDetector()
        stop, ctx, worker = self.run_detector(cfg, detector, frames, snapshots)
        self.assertTrue(worker.wait_ready(3))
        publish_frames(frames, 20, delay=0.03)
        deadline = time.monotonic() + 5
        while worker.stats()["detections"] < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        stop.set()
        worker.join(timeout=3)
        self.assertGreaterEqual(worker.stats()["detections"], 3)
        self.assertLessEqual(detector.calls, 5)

    def test_slow_inference_skips_intermediate_frames(self):
        cfg = self.cfg(1)
        frames, snapshots = LatestFrame(), LatestTracking()
        detector = FakeDetector(delay=0.15)
        stop, ctx, worker = self.run_detector(cfg, detector, frames, snapshots)
        self.assertTrue(worker.wait_ready(3))
        publish_frames(frames, 1)
        deadline = time.monotonic() + 3
        while detector.calls < 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        # Everything published while inference is busy must be collapsed to
        # the newest frame; no backlog is built.
        publish_frames(frames, 9, start=2)
        deadline = time.monotonic() + 5
        while worker.stats()["detections"] < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        stop.set()
        worker.join(timeout=3)
        second = snapshots.latest()
        # The snapshot bus keeps the newest; confirm it is not a stale frame.
        self.assertGreaterEqual(second.sequence, 10)
        self.assertEqual(worker.stats()["detections"], 2)

    def test_timestamps_come_from_capture_not_completion(self):
        cfg = self.cfg(1)
        frames, snapshots = LatestFrame(), LatestTracking()
        detector = FakeDetector()
        stop, ctx, worker = self.run_detector(cfg, detector, frames, snapshots)
        self.assertTrue(worker.wait_ready(3))
        publish_frames(frames, 1)
        deadline = time.monotonic() + 3
        while snapshots.latest() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        stop.set()
        worker.join(timeout=3)
        snap = snapshots.latest()
        self.assertEqual(snap.sequence, 1)
        self.assertEqual(snap.captured_at, 100.01)
        self.assertGreaterEqual(snap.completed_at, snap.captured_at)
        self.assertTrue(snap.valid)
        self.assertAlmostEqual(snap.center_x, 0.5)
        self.assertAlmostEqual(snap.center_y, 0.45)

    def test_exceptions_do_not_kill_worker_and_are_counted(self):
        cfg = self.cfg(1)
        frames, snapshots = LatestFrame(), LatestTracking()
        detector = FakeDetector(fail_times=1)
        stop, ctx, worker = self.run_detector(cfg, detector, frames, snapshots)
        self.assertTrue(worker.wait_ready(3))
        publish_frames(frames, 1)
        deadline = time.monotonic() + 3
        while worker.stats()["detection_errors"] < 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        publish_frames(frames, 3, start=2)
        deadline = time.monotonic() + 5
        while worker.stats()["detections"] < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        stop.set()
        worker.join(timeout=3)
        self.assertEqual(worker.stats()["detection_errors"], 1)
        self.assertGreaterEqual(worker.stats()["detections"], 1)
        self.assertTrue(ctx.closed)

    def test_tracking_disabled_skips_inference(self):
        cfg = self.cfg(1)
        frames, snapshots = LatestFrame(), LatestTracking()
        detector = FakeDetector()
        state = SharedState(cfg)
        state.update(tracking_enabled=False)
        stop, ctx, worker = self.run_detector(
            cfg, detector, frames, snapshots, state=state
        )
        self.assertTrue(worker.wait_ready(3))
        publish_frames(frames, 5)
        time.sleep(0.15)
        stop.set()
        worker.join(timeout=3)
        self.assertEqual(detector.calls, 0)


class FakeVcam:
    def __init__(self, send_delay=0.0):
        self.device = "Fake"
        self.sent = []
        self.send_delay = send_delay
        self.closed = False

    def send(self, frame):
        if self.send_delay:
            time.sleep(self.send_delay)
        self.sent.append(frame)

    def close(self):
        self.closed = True


class RecordingProcessor:
    """Processor stand-in that records controller interactions."""

    def __init__(self):
        self.preview = False
        self.observes = []
        self.steps = []
        self.renders = 0
        self.closed = False

    def observe(self, snapshot, values):
        self.observes.append(snapshot)

    def step(self, dt, now, values):
        self.steps.append((dt, now))
        return 0.5, 0.45, 1.5

    def render(self, frame, values, now, x, y, z):
        self.renders += 1
        return np.full(values.get("_size", (8, 12, 3)), 7, np.uint8)

    def reload_shader(self):
        pass

    def close(self):
        self.closed = True


def publish_one(bus, seq=1):
    frame = np.full((12, 8, 3), seq % 250, np.uint8)
    frame.setflags(write=False)
    bus.publish(FrameEnvelope(frame, seq, time.monotonic()))


class TestRendererWorker(unittest.TestCase):
    @staticmethod
    def cfg():
        return {**DEFAULTS, "out_w": 8, "out_h": 12, "fps": 30}

    def setUp(self):
        self.cfg = self.cfg()
        self.state = SharedState(self.cfg)
        self.frames = LatestFrame()
        self.snapshots = LatestTracking()
        self.stop = threading.Event()

    def start_renderer(self, processor, cam, cfg=None):
        worker = RendererWorker(
            cfg or self.cfg, self.state, self.frames, self.snapshots,
            processor, self.stop, cam_factory=lambda: cam,
        )
        worker.start()
        self.assertTrue(worker.wait_ready(3))
        return worker

    def test_renders_without_any_detector_snapshot(self):
        cam = FakeVcam()
        processor = RecordingProcessor()
        worker = self.start_renderer(processor, cam)
        publish_one(self.frames)
        deadline = time.monotonic() + 0.4
        while len(cam.sent) < 5 and time.monotonic() < deadline:
            time.sleep(0.005)
        self.stop.set(); worker.join(timeout=3)
        self.assertGreaterEqual(len(cam.sent), 5)
        self.assertEqual(processor.observes, [])
        self.assertGreater(len(processor.steps), 0)
        self.assertTrue(cam.closed)
        self.assertTrue(processor.closed)

    def test_snapshot_is_observed_once_and_step_uses_real_delta(self):
        cam = FakeVcam()
        processor = RecordingProcessor()
        worker = self.start_renderer(processor, cam)
        snap = TrackingSnapshot(0.5, 0.45, 0.3, 0.9, time.monotonic(), 1,
                                time.monotonic(), True)
        self.snapshots.publish(snap)
        publish_one(self.frames)
        deadline = time.monotonic() + 0.3
        while len(processor.observes) < 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        # No new frame: the last output is re-sent, no extra observe/step.
        time.sleep(0.15)
        self.stop.set(); worker.join(timeout=3)
        self.assertEqual(len(processor.observes), 1)
        self.assertIs(processor.observes[0], snap)
        self.assertTrue(all(0 < dt <= 0.5 for dt, _ in processor.steps))
        self.assertGreater(worker.stats()["render_repeats"], 0)

    def test_slow_render_drops_ticks_instead_of_bursting(self):
        cam = FakeVcam(send_delay=0.06)  # slower than the 1/30 s interval
        processor = RecordingProcessor()
        worker = self.start_renderer(processor, cam)
        for seq in range(1, 40):
            publish_one(self.frames, seq)
            time.sleep(0.02)
        time.sleep(0.2)
        self.stop.set(); worker.join(timeout=3)
        self.assertGreater(worker.stats()["render_drops"], 0)
        self.assertLess(worker.stats()["render_fps"], 30)
        self.assertLess(len(cam.sent), 40)


class TestPipelineLifecycle(unittest.TestCase):
    def test_partial_startup_failure_releases_started_workers(self):
        cfg = dict(DEFAULTS)
        state = SharedState(cfg)
        pipeline = VideoPipeline(cfg, state)
        with mock.patch.object(CameraWorker, "wait_ready", return_value=False):
            pipeline.start()
        self.assertTrue(pipeline.failed)
        self.assertIn("camera did not become ready", state.telemetry()["status"])
        self.assertIsNone(pipeline.renderer)
        pipeline.stop()
        pipeline.join(2.0)
        self.assertFalse(pipeline.is_alive())

    def test_workers_do_not_survive_shutdown(self):
        cfg = dict(DEFAULTS)
        state = SharedState(cfg)
        pipeline = VideoPipeline(cfg, state)
        pipeline.camera = CameraWorker(cfg, pipeline.frames, pipeline._stop_event)
        pipeline.detector = DetectorWorker(
            cfg, state, pipeline.frames, pipeline.snapshots, pipeline._stop_event
        )
        # Only exercise the join path; workers were never started.
        pipeline.join(1.0)
        self.assertFalse(pipeline.is_alive())


class TestRollingStats(unittest.TestCase):
    def test_percentiles_and_fps(self):
        stats = RollingStats()
        now = 100.0
        for i, value in enumerate([1, 2, 3, 4, 100]):
            stats.add(value, now + i * 0.1)
        self.assertEqual(stats.count(), 5)
        self.assertEqual(stats.p50(), 3)
        self.assertEqual(stats.p95(), 100)
        self.assertAlmostEqual(stats.fps(), 12.5, places=3)

    def test_window_bounds_memory(self):
        stats = RollingStats(window=0.5)
        for i in range(1000):
            stats.add(1, i * 0.01)
        self.assertLessEqual(stats.count(), 51)


if __name__ == "__main__":
    unittest.main()
