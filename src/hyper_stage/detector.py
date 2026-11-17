"""Face detection stage: owns MediaPipe, publishes immutable tracking snapshots."""

import threading
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from hyper_stage.framebus import TrackingSnapshot
from hyper_stage.metrics import RollingStats
from hyper_stage.tracking import (
    DETECT_H,
    DETECT_W,
    ensure_model,
    face_size,
    pick_subject,
    quiet_native_stderr,
    subject_center,
)


class DetectorWorker(threading.Thread):
    """Independent inference worker.

    It reads the newest capture frame only when at least ``detect_every`` new
    frames have arrived since the last inference. Because inference runs in
    this thread, intermediate frames are never queued - they are simply
    replaced in the mailbox while a detection is in flight.
    """

    def __init__(self, cfg, state, frames, snapshots, stop_event,
                 on_ready=None, on_error=None, detector_factory=None):
        super().__init__(name="hyper-stage-detector", daemon=True)
        self.cfg = cfg
        self.state = state
        self.frames = frames
        self.snapshots = snapshots
        self.stop_event = stop_event
        self._on_ready = on_ready
        self._on_error = on_error
        self._detector_factory = detector_factory
        self._ready = threading.Event()
        self._inference = RollingStats()
        self._rate = RollingStats()
        self._last_inference_seq = 0
        self._last_considered_seq = 0
        self._new_frames = 0
        self._detections = 0
        self._errors = 0
        self._skipped = 0
        self._detector = None
        self._ctx = None

    def _create_detector(self):
        options = vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=ensure_model()),
            min_detection_confidence=self.cfg["min_det_conf"],
        )
        with quiet_native_stderr():
            ctx = vision.FaceDetector.create_from_options(options)
            detector = ctx.__enter__()
        return ctx, detector

    def wait_ready(self, timeout):
        return self._ready.wait(timeout)

    def _inference_allowed(self, envelope):
        """Count newly observed capture frames; never queue intermediate ones."""
        if envelope.sequence == self._last_considered_seq:
            return False
        self._last_considered_seq = envelope.sequence
        values = self.state.snapshot()
        if not values["tracking_enabled"] or values["magic_loop_enabled"]:
            self._skipped += 1
            return False
        self._new_frames += 1
        if self._new_frames < max(1, int(self.cfg["detect_every"])):
            self._skipped += 1
            return False
        self._new_frames = 0
        return True

    def _detect(self, detector, envelope):
        small = cv2.resize(envelope.frame, (DETECT_W, DETECT_H))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(image)
        subject = pick_subject(result.detections)
        completed = time.monotonic()
        if subject is None:
            return None
        cx, cy = subject_center(subject, DETECT_W, DETECT_H)
        categories = getattr(subject, "categories", None) or []
        confidence = max(
            (getattr(c, "score", 0.0) or 0.0 for c in categories), default=0.0
        )
        return TrackingSnapshot(
            center_x=cx,
            center_y=cy,
            size=face_size(subject, DETECT_W, DETECT_H),
            confidence=float(confidence),
            captured_at=envelope.captured_at,
            sequence=envelope.sequence,
            completed_at=completed,
            valid=True,
        )

    def run(self):
        try:
            if self._detector_factory:
                self._ctx, self._detector = self._detector_factory()
            else:
                self._ctx, self._detector = self._create_detector()
            self._ready.set()
            if self._on_ready:
                self._on_ready()

            while not self.stop_event.is_set():
                envelope = self.frames.latest()
                if envelope is None:
                    self.frames.wake.wait(0.05)
                    self.frames.wake.clear()
                    continue
                if not self._inference_allowed(envelope):
                    # Check again shortly; the producer sets wake on new frames.
                    self.frames.wake.wait(0.02)
                    self.frames.wake.clear()
                    continue

                # Take the newest frame at the moment inference actually starts,
                # not the one that first satisfied the interval check.
                self._last_inference_seq = envelope.sequence
                self.frames.wake.clear()
                started = time.monotonic()
                try:
                    snapshot = self._detect(self._detector, envelope)
                    if snapshot is not None:
                        if self.snapshots.publish(snapshot):
                            self._detections += 1
                            self._rate.add(started)
                except Exception as exc:
                    # One bad frame must not kill the worker; the controller's
                    # detection timeout handles the resulting measurement gap.
                    self._errors += 1
                    if self._errors <= 3:
                        print(f"warning: detection error ({self._errors}): {exc}")
                finally:
                    self._inference.add(time.monotonic() - started)
        except Exception as exc:
            if self._on_error:
                self._on_error(exc)
            else:
                print(f"warning: detector worker stopped: {exc}")
        finally:
            if self._ctx is not None:
                try:
                    with quiet_native_stderr():
                        self._ctx.__exit__(None, None, None)
                except Exception:
                    pass
                self._ctx = None
                self._detector = None
            self._ready.set()

    def stats(self):
        bus_stats = self.snapshots.stats()
        return {
            "detector_fps": self._rate.fps(),
            "inference_p50": self._inference.p50(),
            "inference_p95": self._inference.p95(),
            "detections": self._detections,
            "detection_errors": self._errors,
            "detection_skipped": self._skipped,
            "stale_discarded": bus_stats["stale_discarded"],
        }
