"""Pipeline lifecycle: start, supervise and stop the three media workers.

Capture, detection and rendering each own their native resources in their own
thread. This orchestrator only holds shared state, the two mailboxes and the
worker references; it never touches a camera, a detector or an OpenGL context.
The public API (start/stop/join/reload_shader/ident) is unchanged for the TUI.
"""

import threading
import time
import traceback

import cv2

from capture import CameraWorker
from detector import DetectorWorker
from framebus import LatestFrame, LatestTracking
from renderer import FrameProcessor, RendererWorker


class VideoPipeline:
    """Owns the three stages and their deterministic startup/shutdown."""

    STARTUP_TIMEOUT = 20.0
    JOIN_TIMEOUT = 5.0

    def __init__(self, cfg, state, preview=False):
        self.cfg = cfg
        self.state = state
        self.preview = preview
        self._stop_event = threading.Event()
        self._reload_event = threading.Event()
        self.frames = LatestFrame()
        self.snapshots = LatestTracking()
        self.processor = FrameProcessor(cfg, state, preview)
        self.camera = None
        self.detector = None
        self.renderer = None
        self.failed = False

    def _on_error(self, exc):
        """A worker fault stops the other stages and surfaces in the TUI."""
        traceback.print_exc()
        self.failed = True
        self.state.set_telemetry(status=f"stopped: {exc}")
        self.stop()

    def start(self):
        """Start capture, then detection, then rendering; clean up on failure."""
        # OpenCV's worker threads compete with MediaPipe's pool on macOS.
        cv2.setNumThreads(1)
        self.state.set_telemetry(status="starting")
        try:
            self.camera = CameraWorker(
                self.cfg, self.frames, self._stop_event, on_error=self._on_error
            )
            self.camera.start()
            if not self.camera.wait_ready(self.STARTUP_TIMEOUT):
                raise RuntimeError("camera did not become ready")

            self.detector = DetectorWorker(
                self.cfg, self.state, self.frames, self.snapshots,
                self._stop_event, on_error=self._on_error,
            )
            self.detector.start()
            if not self.detector.wait_ready(self.STARTUP_TIMEOUT):
                raise RuntimeError("face detector did not become ready")

            self.renderer = RendererWorker(
                self.cfg, self.state, self.frames, self.snapshots,
                self.processor, self._stop_event, on_error=self._on_error,
                reload_event=self._reload_event,
            )
            self.renderer.start()
            if not self.renderer.wait_ready(self.STARTUP_TIMEOUT):
                raise RuntimeError("virtual camera did not become ready")

            self.state.set_telemetry(status="running")
        except Exception as exc:
            self._on_error(exc)

    def stop(self):
        """Signal every worker to stop and wake any that are waiting."""
        self._stop_event.set()
        self.frames.wake.set()

    def join(self, timeout=None):
        """Join workers with a bounded total budget; never block the TUI."""
        deadline = time.monotonic() + (
            self.JOIN_TIMEOUT if timeout is None else timeout
        )
        for worker in (self.renderer, self.detector, self.camera):
            if worker is None or not worker.is_alive():
                continue
            remaining = max(0.0, deadline - time.monotonic())
            worker.join(timeout=remaining)
        survivors = [
            w.name for w in (self.camera, self.detector, self.renderer)
            if w is not None and w.is_alive()
        ]
        if survivors:
            self.state.set_telemetry(status=f"workers did not stop: {survivors}")
        self.state.set_telemetry(fps=0.0, capture_fps=0.0, tracking=False)

    @property
    def ident(self):
        """Kept for the TUI's 'was ever started' check (any worker counts)."""
        first = self.camera or self.detector or self.renderer
        return first.ident if first is not None else None

    def is_alive(self):
        return any(
            w is not None and w.is_alive()
            for w in (self.camera, self.detector, self.renderer)
        )

    def reload_shader(self):
        """Marshal a hot-reload request to the rendering thread."""
        self._reload_event.set()

    def stats(self):
        """Aggregated per-stage telemetry for the TUI's periodic refresh."""
        stats = {}
        if self.camera is not None:
            stats.update(self.camera.stats())
        if self.detector is not None:
            stats.update(self.detector.stats())
        if self.renderer is not None:
            stats.update(self.renderer.stats())
        stats.update(self.frames.stats())
        return stats
