import math
import threading
import time
import types
from unittest import mock

import cv2
import numpy as np
import pyvirtualcam
import pytest
from mediapipe.tasks.python import vision

from hyper_stage.capture import CameraWorker
from hyper_stage.config import DEFAULTS
from hyper_stage.detector import DetectorWorker
from hyper_stage.framebus import FrameEnvelope, LatestFrame, LatestTracking, TrackingSnapshot
from hyper_stage.pipeline import VideoPipeline
from hyper_stage.pipmode import pip_corner_rect
from hyper_stage.renderer import RendererWorker
from hyper_stage.state import SharedState
from hyper_stage.tracking import quiet_native_stderr


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


class TestCameraWorker:
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
        assert item is not None
        assert worker.stats()['capture_reads'] == 5
        assert cam.released
        assert not (item.frame.flags.writeable)
        assert item.sequence == 5
        assert item.captured_at > 0

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
        assert worker.stats()['capture_failures'] >= 3
        stop.set()
        worker.join(timeout=3)
        assert worker.stats()['capture_failures'] >= 3
        assert worker.stats()['capture_reads'] == 2
        assert cam.released

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
        assert worker.wait_ready(0.1)
        assert len(errors) == 1
        assert 'no camera' in str(errors[0])


class TestDetectorWorker:
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
        assert worker.wait_ready(3)
        publish_frames(frames, 20, delay=0.03)
        deadline = time.monotonic() + 5
        while worker.stats()["detections"] < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        stop.set()
        worker.join(timeout=3)
        assert worker.stats()['detections'] >= 3
        assert detector.calls <= 5

    def test_slow_inference_skips_intermediate_frames(self):
        cfg = self.cfg(1)
        frames, snapshots = LatestFrame(), LatestTracking()
        detector = FakeDetector(delay=0.15)
        stop, ctx, worker = self.run_detector(cfg, detector, frames, snapshots)
        assert worker.wait_ready(3)
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
        assert second.sequence >= 10
        assert worker.stats()['detections'] == 2

    def test_timestamps_come_from_capture_not_completion(self):
        cfg = self.cfg(1)
        frames, snapshots = LatestFrame(), LatestTracking()
        detector = FakeDetector()
        stop, ctx, worker = self.run_detector(cfg, detector, frames, snapshots)
        assert worker.wait_ready(3)
        publish_frames(frames, 1)
        deadline = time.monotonic() + 3
        while snapshots.latest() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        stop.set()
        worker.join(timeout=3)
        snap = snapshots.latest()
        assert snap.sequence == 1
        assert snap.captured_at == 100.01
        assert snap.completed_at >= snap.captured_at
        assert snap.valid
        assert snap.center_x == pytest.approx(0.5)
        assert snap.center_y == pytest.approx(0.45)

    def test_exceptions_do_not_kill_worker_and_are_counted(self):
        cfg = self.cfg(1)
        frames, snapshots = LatestFrame(), LatestTracking()
        detector = FakeDetector(fail_times=1)
        stop, ctx, worker = self.run_detector(cfg, detector, frames, snapshots)
        assert worker.wait_ready(3)
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
        assert worker.stats()['detection_errors'] == 1
        assert worker.stats()['detections'] >= 1
        assert ctx.closed

    def test_tracking_disabled_skips_inference(self):
        cfg = self.cfg(1)
        frames, snapshots = LatestFrame(), LatestTracking()
        detector = FakeDetector()
        state = SharedState(cfg)
        state.update(tracking_enabled=False)
        stop, ctx, worker = self.run_detector(
            cfg, detector, frames, snapshots, state=state
        )
        assert worker.wait_ready(3)
        publish_frames(frames, 5)
        time.sleep(0.15)
        stop.set()
        worker.join(timeout=3)
        assert detector.calls == 0


class TestRendererWorker:
    @staticmethod
    def default_cfg():
        return {**DEFAULTS, "out_w": 8, "out_h": 12, "fps": 30}

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cfg = self.default_cfg()
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
        assert worker.wait_ready(3)
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
        assert len(cam.sent) >= 5
        assert processor.observes == []
        assert len(processor.steps) > 0
        assert cam.closed
        assert processor.closed

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
        assert len(processor.observes) == 1
        assert processor.observes[0] is snap
        assert all((0 < dt <= 0.5 for dt, _ in processor.steps))
        assert worker.stats()['render_repeats'] > 0

    def test_slow_render_drops_ticks_instead_of_bursting(self):
        cam = FakeVcam(send_delay=0.06)  # slower than the 1/30 s interval
        processor = RecordingProcessor()
        worker = self.start_renderer(processor, cam)
        for seq in range(1, 40):
            publish_one(self.frames, seq)
            time.sleep(0.02)
        time.sleep(0.2)
        self.stop.set(); worker.join(timeout=3)
        assert worker.stats()['render_drops'] > 0
        assert worker.stats()['render_fps'] < 30
        assert len(cam.sent) < 40


class TestPipelineLifecycle:
    def test_partial_startup_failure_releases_started_workers(self):
        cfg = dict(DEFAULTS)
        state = SharedState(cfg)
        pipeline = VideoPipeline(cfg, state)
        with mock.patch.object(CameraWorker, "wait_ready", return_value=False):
            pipeline.start()
        assert pipeline.failed
        assert 'camera did not become ready' in state.telemetry()['status']
        assert pipeline.renderer is None
        pipeline.stop()
        pipeline.join(2.0)
        assert not (pipeline.is_alive())

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
        assert not (pipeline.is_alive())



def make_detection(cx, cy, height, eye=None, score=0.9):
    """Small MediaPipe-like detection used by the integration fakes."""
    from hyper_stage.tracking import DETECT_H, DETECT_W

    aspect = DETECT_W / DETECT_H
    w = height * DETECT_H / aspect
    bb = types.SimpleNamespace(
        origin_x=(cx - w / 2) * DETECT_W,
        origin_y=(cy - height * 0.6) * DETECT_H,
        width=w * DETECT_W,
        height=height * DETECT_H,
    )
    kps = None
    if eye is not None:
        dx = eye / 2.8 / aspect
        kps = [
            types.SimpleNamespace(x=cx - dx, y=cy),
            types.SimpleNamespace(x=cx + dx, y=cy),
        ]
    return types.SimpleNamespace(
        bounding_box=bb, keypoints=kps,
        categories=[types.SimpleNamespace(score=score)],
    )


class TestVideoPipeline:
    def test_processes_frames_and_reports_telemetry(self):
        import time
        import types
        from unittest import mock

        import numpy as np

        sent = []

        class FakeCapture:
            def __init__(self, *args, **kwargs):
                pass

            def isOpened(self):
                return True

            def set(self, *args):
                return True

            def read(self):
                frame = np.zeros((120, 160, 3), dtype=np.uint8)
                frame[..., 1] = 128
                return True, frame

            def release(self):
                pass

        class FakeCamera:
            device = "Fake Virtual Camera"

            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def send(self, frame):
                sent.append(frame)

            def sleep_until_next_frame(self):
                pass

            def close(self):
                pass

        class FakeDetector:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def detect(self, image):
                return types.SimpleNamespace(detections=[])

        cfg = dict(DEFAULTS)
        cfg.update(out_w=160, out_h=90, tracking_enabled=False, sharpen_enabled=True)
        state = SharedState(cfg)
        pipeline = VideoPipeline(cfg, state)
        with (
            mock.patch("hyper_stage.detector.ensure_model", return_value="fake"),
            mock.patch.object(cv2, "VideoCapture", FakeCapture),
            mock.patch.object(pyvirtualcam, "Camera", FakeCamera),
            mock.patch.object(
                vision.FaceDetector,
                "create_from_options",
                return_value=FakeDetector(),
            ),
        ):
            pipeline.start()
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if sent and state.telemetry()["fps"] > 0:
                    break
                time.sleep(0.005)
            telemetry = state.telemetry()
            pipeline.stop()
            pipeline.join(2.0)

        assert not (pipeline.is_alive())
        assert len(sent) >= 1
        assert sent[0].shape == (90, 160, 3)
        assert telemetry['device'] == 'Fake Virtual Camera'
        assert telemetry['status'] == 'running'
        assert telemetry['fps'] > 0.0
        assert not (telemetry['tracking'])

    def test_tracking_debug_telemetry_with_detection(self):
        import time
        import types
        from unittest import mock

        import numpy as np

        sent = []

        class FakeCapture:
            def __init__(self, *args, **kwargs):
                pass

            def isOpened(self):
                return True

            def set(self, *args):
                return True

            def read(self):
                return True, np.zeros((120, 160, 3), dtype=np.uint8)

            def release(self):
                pass

        class FakeCamera:
            device = "Fake"

            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def send(self, frame):
                sent.append(frame)

            def sleep_until_next_frame(self):
                pass

            def close(self):
                pass

        class FakeDetector:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def detect(self, image):
                d = make_detection(0.5, 0.45, 0.3, eye=0.3 / 2.8)
                return types.SimpleNamespace(detections=[d])

        cfg = dict(DEFAULTS)
        cfg.update(out_w=160, out_h=90, tracking_enabled=True, tracking_debug=True)
        state = SharedState(cfg)
        pipeline = VideoPipeline(cfg, state)
        with (
            mock.patch("hyper_stage.detector.ensure_model", return_value="fake"),
            mock.patch.object(cv2, "VideoCapture", FakeCapture),
            mock.patch.object(pyvirtualcam, "Camera", FakeCamera),
            mock.patch.object(
                vision.FaceDetector,
                "create_from_options",
                return_value=FakeDetector(),
            ),
        ):
            pipeline.start()
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                t = state.telemetry()
                if t.get("detection_age", math.inf) < 1.0 and sent:
                    break
                time.sleep(0.005)
            telemetry = state.telemetry()
            pipeline.stop()
            pipeline.join(2.0)

        assert telemetry['tracking']
        assert telemetry['detection_age'] < 1.0
        for key in ("raw_center", "filtered_center", "raw_height", "filtered_height",
                    "desired_zoom", "actual_zoom", "pos_velocity", "tracking_state"):
            assert key in telemetry

    def test_pip_composites_slide_background_with_corner_webcam(self):
        import time
        import types
        from unittest import mock

        import numpy as np

        sent = []

        class FakeCapture:
            def __init__(self, *args, **kwargs):
                pass

            def isOpened(self):
                return True

            def set(self, *args):
                return True

            def read(self):
                # Distinct green live frame: easy to tell from the background.
                frame = np.zeros((120, 160, 3), dtype=np.uint8)
                frame[..., 1] = 128
                return True, frame

            def release(self):
                pass

        class FakeCamera:
            device = "Fake Virtual Camera"

            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def send(self, frame):
                sent.append(frame)

            def sleep_until_next_frame(self):
                pass

            def close(self):
                pass

        class FakeDetector:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def detect(self, image):
                return types.SimpleNamespace(detections=[])

        cfg = dict(DEFAULTS)
        cfg.update(out_w=160, out_h=90, tracking_enabled=False, pip_enabled=True)
        state = SharedState(cfg)
        # An asymmetric blue/white background, pre-fitted to the output size
        # like the TUI worker would deliver it: blue right half, white left.
        background = np.zeros((90, 160, 3), dtype=np.uint8)
        background[:, :80] = (255, 255, 255)
        background[:, 80:] = (255, 10, 10)
        state.set_pip_frame(background)
        pipeline = VideoPipeline(cfg, state)
        with (
            mock.patch("hyper_stage.detector.ensure_model", return_value="fake"),
            mock.patch.object(cv2, "VideoCapture", FakeCapture),
            mock.patch.object(pyvirtualcam, "Camera", FakeCamera),
            mock.patch.object(
                vision.FaceDetector,
                "create_from_options",
                return_value=FakeDetector(),
            ),
        ):
            pipeline.start()
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if sent and state.telemetry()["fps"] > 0:
                    break
                time.sleep(0.005)
            unflipped = sent[0]
            # Enabling PiP Flip mirrors the cached display background: the
            # white half moves right on the next frame, without touching the
            # stored one.
            state.update(pip_flip=True)
            sent.clear()
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and len(sent) < 3:
                time.sleep(0.005)
            pipeline.stop()
            pipeline.join(2.0)

        assert not (pipeline.is_alive())
        assert len(sent) >= 1
        x, y, w, h = pip_corner_rect(
            state.snapshot()["pip_corner"],
            cfg["pip_scale"],
            cfg["pip_padding"],
            160,
            90,
        )
        # Flip off: the white half is on the left, as stored...
        assert tuple(unflipped[45, 40]) == (255, 255, 255)
        assert tuple(unflipped[45, 120]) == (255, 10, 10)
        # ...and the corner window holds the live webcam.
        assert tuple(unflipped[y + h // 2, x + w // 2]) == (0, 128, 0)
        # Flip on: the published frame is mirrored...
        flipped = sent[-1]
        assert tuple(flipped[45, 120]) == (255, 255, 255)
        assert tuple(flipped[45, 40]) == (255, 10, 10)
        # ...and the stored background itself was never mutated.
        assert tuple(background[45, 120]) == (255, 10, 10)

    def test_pip_transition_rect_eases_from_full_frame_to_corner(self):
        from hyper_stage.pipmode import (
            CORNER_BR,
            compose_pip_rect,
            pip_corner_rect,
            pip_transition_rect,
        )

        w, h = 160, 90
        corner = pip_corner_rect(CORNER_BR, 0.25, 8, w, h)
        # progress 0: the full frame (PiP off)...
        assert pip_transition_rect(0.0, CORNER_BR, 0.25, 8, w, h) == (0, 0, w, h)
        # ...progress 1: the exact corner window (PiP on).
        assert pip_transition_rect(1.0, CORNER_BR, 0.25, 8, w, h) == corner
        # Mid-flight the rect sits between the two and stays in frame.
        for progress in (0.25, 0.5, 0.75):
            x, y, rw, rh = pip_transition_rect(
                progress, CORNER_BR, 0.25, 8, w, h
            )
            assert x > 0
            assert x + rw < w
            assert y > 0
            assert y + rh < h
            assert rw > corner[2]
            assert rw < w

        # Composing at the corner rect matches the settled compose_pip,
        # and at the full-frame rect just pastes the webcam over the bg.
        background = np.full((h, w, 3), 10, dtype=np.uint8)
        webcam = np.full((30, 30, 3), 200, dtype=np.uint8)
        settled = compose_pip_rect(
            np.zeros((h, w, 3), dtype=np.uint8), background, webcam, corner
        )
        from hyper_stage.pipmode import compose_pip

        assert (settled == compose_pip(np.zeros((h, w, 3), dtype=np.uint8), background, webcam, CORNER_BR, 0.25, 8)).all()
        full = compose_pip_rect(
            np.zeros((h, w, 3), dtype=np.uint8), background, webcam, (0, 0, w, h)
        )
        assert tuple(full[45, 45]) == (200, 200, 200)
        # The background was never mutated.
        assert tuple(background[45, 45]) == (10, 10, 10)

    def test_cam_flip_mirrors_the_published_stream(self):
        import time
        import types
        from unittest import mock

        import numpy as np

        sent = []

        class FakeCapture:
            def __init__(self, *args, **kwargs):
                pass

            def isOpened(self):
                return True

            def set(self, *args):
                return True

            def read(self):
                # Asymmetric raw frame: red left half, blue right half.
                frame = np.zeros((120, 160, 3), dtype=np.uint8)
                frame[:, :80] = (10, 10, 255)
                frame[:, 80:] = (255, 10, 10)
                return True, frame

            def release(self):
                pass

        class FakeCamera:
            device = "Fake Virtual Camera"

            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def send(self, frame):
                sent.append(frame)

            def sleep_until_next_frame(self):
                pass

            def close(self):
                pass

        class FakeDetector:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def detect(self, image):
                return types.SimpleNamespace(detections=[])

        cfg = dict(DEFAULTS)
        cfg.update(out_w=160, out_h=90, tracking_enabled=False, cam_flip=True)
        state = SharedState(cfg)
        pipeline = VideoPipeline(cfg, state)
        with (
            mock.patch("hyper_stage.detector.ensure_model", return_value="fake"),
            mock.patch.object(cv2, "VideoCapture", FakeCapture),
            mock.patch.object(pyvirtualcam, "Camera", FakeCamera),
            mock.patch.object(
                vision.FaceDetector,
                "create_from_options",
                return_value=FakeDetector(),
            ),
        ):
            pipeline.start()
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and not sent:
                time.sleep(0.005)
            pipeline.stop()
            pipeline.join(2.0)

        assert len(sent) >= 1
        # The published frame is the mirror of the raw capture: red right,
        # blue left (the raw frame's halves are swapped).
        assert int(sent[0][45, 120][2]) == 255   # red half on the right
        assert int(sent[0][45, 40][0]) == 255    # blue half on the left
        assert tuple(sent[0][45, 40]) != tuple(sent[0][45, 120])

    def test_magic_loop_replays_a_frozen_seamless_loop(self):
        import time
        import types
        from unittest import mock

        import numpy as np

        sent = []
        counter = {"value": 0}

        class FakeCapture:
            def __init__(self, *args, **kwargs):
                pass

            def isOpened(self):
                return True

            def set(self, *args):
                return True

            def read(self):
                # Mock hardware pacing: without it the producer outruns the
                # renderer and overwrites the early moving frames in the bus.
                time.sleep(1.0 / 120.0)
                counter["value"] += 1
                value = counter["value"]
                frame = np.full((120, 160, 3), 90, dtype=np.uint8)
                # A moving column gives the first era distinct structure that
                # survives JPEG; afterwards the scene is perfectly still, so
                # a fresh loop pulled from the newest history has no column.
                if value <= 120:
                    frame[:, value % 160] = 255
                return True, frame

            def release(self):
                pass

        class FakeCamera:
            device = "Fake Virtual Camera"

            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def send(self, frame):
                sent.append(frame)

            def sleep_until_next_frame(self):
                pass

            def close(self):
                pass

        class FakeDetector:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def detect(self, image):
                return types.SimpleNamespace(detections=[])

        cfg = dict(DEFAULTS)
        # Zoom 1.0 keeps the whole frame so the moving column stays visible
        # and every live frame remains distinct.
        cfg.update(
            out_w=160,
            out_h=90,
            magic_loop_enabled=True,
            tracking_enabled=False,
            zoom=1.0,
        )
        state = SharedState(cfg)
        pipeline = VideoPipeline(cfg, state)
        with (
            mock.patch("hyper_stage.detector.ensure_model", return_value="fake"),
            mock.patch.object(cv2, "VideoCapture", FakeCapture),
            mock.patch.object(pyvirtualcam, "Camera", FakeCamera),
            mock.patch.object(
                vision.FaceDetector,
                "create_from_options",
                return_value=FakeDetector(),
            ),
        ):
            pipeline.start()
            # Enough frames for the settling period, the loop build, the
            # entry dissolve, and several full passes of the frozen loop.
            deadline = time.monotonic() + 12.0
            while time.monotonic() < deadline and len(sent) < 700:
                time.sleep(0.01)
            pipeline.stop()
            pipeline.join(2.0)

        assert not (pipeline.is_alive())
        # The renderer now paces at the configured 30 fps, not as fast as a
        # mock camera can be drained: 12 s yields ~360 output frames.
        assert len(sent) > 300
        # Settling grace: the first frames after activation are still the
        # live feed - distinct, moving-column frames, not a loop.
        early = {frame.tobytes() for frame in sent[:40]}
        assert len(early) > 25
        assert any((int(frame.max()) >= 200 for frame in sent[:40]))
        # Then the frozen loop repeats exactly: find its period in the tail
        # and verify every cycle is identical (clean restart, no crossfades).
        tail = sent[-120:]
        period = next(
            k
            for k in range(55, 121)
            if np.array_equal(tail[k], tail[0])
        )
        # A paced renderer occasionally re-sends the last output when capture
        # has not published a newer frame; tolerate those repeats while still
        # requiring the loop itself to repeat exactly.
        matches = sum(
            np.array_equal(tail[k], tail[k - period])
            for k in range(period, len(tail))
        )
        assert matches / (len(tail) - period) >= 0.85

class TestQuietNativeStderr:
    def test_detector_setup_logs_are_silenced(self):
        """Native LiteRT logs at detector create/close never reach stderr."""
        import subprocess
        import sys

        code = (
            "import sys\n"
            "from mediapipe.tasks import python as mp_python\n"
            "from mediapipe.tasks.python import vision\n"
            "from hyper_stage.tracking import ensure_model, quiet_native_stderr\n"
            "options = vision.FaceDetectorOptions(\n"
            "    base_options=mp_python.BaseOptions(\n"
            "        model_asset_path=ensure_model()))\n"
            "with quiet_native_stderr():\n"
            "    ctx = vision.FaceDetector.create_from_options(options)\n"
            "    detector = ctx.__enter__()\n"
            "detector.detect  # attribute access proves it is live\n"
            "with quiet_native_stderr():\n"
            "    ctx.__exit__(None, None, None)\n"
            "sys.stderr.write('AFTER-MARK\\n')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert 'INFO:' not in result.stderr
        assert 'TensorFlow Lite' not in result.stderr
        # stderr outside the silenced windows still reaches the terminal.
        assert 'AFTER-MARK' in result.stderr

    def test_quiet_native_stderr_swallows_fd2_writes_only(self):
        import os
        import subprocess
        import sys

        code = (
            "import os, sys\n"
            "from hyper_stage.tracking import quiet_native_stderr\n"
            "with quiet_native_stderr():\n"
            "    os.write(2, b'HIDDEN\\n')\n"
            "os.write(2, b'VISIBLE\\n')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert 'HIDDEN' not in result.stderr
        assert 'VISIBLE' in result.stderr

