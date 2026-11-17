"""Inexpensive rolling timing statistics for pipeline telemetry.

One bounded window serves both percentiles and rate measurement, so telemetry
never needs a per-frame log line or an unbounded sample list.
"""

import math
import threading
import time
from collections import deque


class RollingStats:
    """Rolling window of (monotonic timestamp, duration) samples."""

    def __init__(self, window=2.0, max_samples=600):
        self._window = float(window)
        self._max = int(max_samples)
        self._samples = deque()
        self._lock = threading.Lock()

    def add(self, value, now=None):
        if now is None:
            now = time.monotonic()
        with self._lock:
            self._samples.append((now, float(value)))
            while self._samples and now - self._samples[0][0] > self._window:
                self._samples.popleft()
            while len(self._samples) > self._max:
                self._samples.popleft()

    def _values(self):
        with self._lock:
            return [value for _, value in self._samples]

    def values(self):
        """Snapshot of the window's values (for offline percentile analysis)."""
        return self._values()

    def count(self):
        with self._lock:
            return len(self._samples)

    def last(self):
        with self._lock:
            return self._samples[-1][1] if self._samples else None

    def mean(self):
        values = self._values()
        return sum(values) / len(values) if values else 0.0

    def _percentile(self, fraction):
        values = sorted(self._values())
        if not values:
            return 0.0
        index = min(len(values) - 1, max(0, math.ceil(fraction * len(values)) - 1))
        return values[index]

    def p50(self):
        return self._percentile(0.5)

    def p95(self):
        return self._percentile(0.95)

    def fps(self):
        """Observed rate over the window (never the configured target)."""
        with self._lock:
            if len(self._samples) < 2:
                return 0.0
            span = self._samples[-1][0] - self._samples[0][0]
        return len(self._samples) / span if span > 0 else 0.0

    def clear(self):
        with self._lock:
            self._samples.clear()
