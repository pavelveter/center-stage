"""Single-slot, thread-safe transport between the three pipeline stages.

The frame bus holds at most one frame: a new publication overwrites the old
one, so producers never wait and consumers always see the newest frame. Frames
are handed over as immutable envelopes (the producer marks the ndarray
read-only) - no full-resolution copy happens at the boundary. Consumers create
a fresh array whenever they need to transform a frame.

The snapshot bus works the same way but additionally rejects late results from
older capture frames, so a slow inference cannot roll tracking backwards.
"""

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class FrameEnvelope:
    """One captured frame plus the metadata needed for pacing and latency."""

    frame: object  # read-only numpy.ndarray, owned by nobody in particular
    sequence: int
    captured_at: float


@dataclass(frozen=True)
class TrackingSnapshot:
    """Immutable result of one inference, stamped with the source frame."""

    center_x: float
    center_y: float
    size: float
    confidence: float
    captured_at: float
    sequence: int
    completed_at: float
    valid: bool


class LatestFrame:
    """One-slot mailbox for captured frames (mailbox / latest-value semantics)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._item = None
        self._published = 0
        self._overwritten = 0
        # Non-blocking wake-up for consumers; the producer only sets it.
        self.wake = threading.Event()

    def publish(self, envelope):
        """Replace the retained frame. Never blocks on a consumer."""
        with self._lock:
            if self._item is not None and envelope.sequence <= self._item.sequence:
                return False
            if self._item is not None:
                self._overwritten += 1
            self._item = envelope
            self._published += 1
        self.wake.set()
        return True

    def latest(self):
        """Newest available frame, or None. Safe to call from any thread."""
        with self._lock:
            return self._item

    def stats(self):
        with self._lock:
            return {
                "published": self._published,
                "overwritten": self._overwritten,
            }


class LatestTracking:
    """One-slot mailbox for tracking snapshots, rejecting older results."""

    def __init__(self):
        self._lock = threading.Lock()
        self._item = None
        self._published = 0
        self._stale_discarded = 0

    def publish(self, snapshot):
        """Accept a result only when it describes a newer capture frame."""
        with self._lock:
            if self._item is not None and snapshot.sequence <= self._item.sequence:
                self._stale_discarded += 1
                return False
            self._item = snapshot
            self._published += 1
        return True

    def latest(self):
        with self._lock:
            return self._item

    def stats(self):
        with self._lock:
            return {
                "published": self._published,
                "stale_discarded": self._stale_discarded,
            }
