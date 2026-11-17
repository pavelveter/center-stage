"""Camera capture stage: owns the VideoCapture and publishes latest frames."""

import threading
import time

import cv2
import numpy as np

from framebus import FrameEnvelope
from metrics import RollingStats


class CameraWorker(threading.Thread):
    """Producer thread. Blocks only on the camera, never on a consumer.

    The ndarray returned by ``cap.read()`` is made read-only before it is
    published, so handing out references is a zero-copy ownership transfer:
    consumers may read or downscale it but must create a new array to modify it.
    """

    MAX_CONSECUTIVE_ERRORS = 30

    def __init__(self, cfg, frames, stop_event, on_ready=None, on_error=None,
                 cap_factory=None):
        super().__init__(name="hyper-stage-capture", daemon=True)
        self.cfg = cfg
        self.frames = frames
        self.stop_event = stop_event
        self._on_ready = on_ready
        self._on_error = on_error
        self._cap_factory = cap_factory
        self._ready = threading.Event()
        self._fps = RollingStats()
        self._reads = 0
        self._failures = 0
        self._consecutive_errors = 0
        self._sequence = 0
        self._cap = None

    def _open_camera(self):
        cap = cv2.VideoCapture(self.cfg["camera_index"], cv2.CAP_AVFOUNDATION)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg["in_w"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg["in_h"])
        cap.set(cv2.CAP_PROP_FPS, self.cfg["fps"])
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"cannot open camera {self.cfg['camera_index']}; "
                "run --list-cameras to see indices"
            )
        return cap

    def wait_ready(self, timeout):
        return self._ready.wait(timeout)

    def run(self):
        try:
            self._cap = (
                self._cap_factory() if self._cap_factory else self._open_camera()
            )
            while not self.stop_event.is_set():
                ret, frame = self._cap.read()
                if (
                    not ret
                    or frame is None
                    or not isinstance(frame, np.ndarray)
                    or frame.ndim != 3
                    or frame.shape[2] != 3
                ):
                    self._failures += 1
                    self._consecutive_errors += 1
                    if self._consecutive_errors > self.MAX_CONSECUTIVE_ERRORS:
                        raise RuntimeError("camera stopped delivering frames")
                    time.sleep(0.005)
                    continue

                # A fresh buffer each read; freeze it so no consumer can mutate
                # what another consumer is still reading.
                frame = np.ascontiguousarray(frame)
                frame.setflags(write=False)
                captured = time.monotonic()
                self._sequence += 1
                self.frames.publish(
                    FrameEnvelope(frame, self._sequence, captured)
                )
                self._fps.add(captured)
                self._reads += 1
                self._consecutive_errors = 0
                if not self._ready.is_set():
                    self._ready.set()
                    if self._on_ready:
                        self._on_ready()
        except Exception as exc:
            if self._on_error:
                self._on_error(exc)
            else:
                print(f"warning: capture worker stopped: {exc}")
        finally:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            # Wake startup waiters even when opening the camera failed.
            self._ready.set()

    def stats(self):
        return {
            "capture_fps": self._fps.fps(),
            "capture_reads": self._reads,
            "capture_failures": self._failures,
            "capture_sequence": self._sequence,
        }
