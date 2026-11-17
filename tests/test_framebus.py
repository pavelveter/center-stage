import threading
import time

import numpy as np
import pytest

from hyper_stage.framebus import FrameEnvelope, LatestFrame, LatestTracking, TrackingSnapshot
from hyper_stage.metrics import RollingStats


class TestLatestFrame:
    @staticmethod
    def envelope(seq, ts=None):
        frame = np.full((4, 6, 3), seq % 251, dtype=np.uint8)
        frame.setflags(write=False)
        return FrameEnvelope(frame, seq, ts if ts is not None else float(seq))

    def test_overwrite_and_newest_retrieval(self):
        bus = LatestFrame()
        assert bus.latest() is None
        first = self.envelope(1)
        assert bus.publish(first)
        second = self.envelope(2)
        assert bus.publish(second)
        assert bus.latest() is second
        assert bus.stats()['published'] == 2
        assert bus.stats()['overwritten'] == 1

    def test_no_unbounded_accumulation(self):
        bus = LatestFrame()
        for seq in range(1000):
            bus.publish(self.envelope(seq))
        assert bus.latest().sequence == 999
        assert bus.stats()['published'] == 1000

    def test_timestamps_and_sequences_preserved(self):
        bus = LatestFrame()
        env = self.envelope(7, 123.5)
        bus.publish(env)
        got = bus.latest()
        assert got.sequence == 7
        assert got.captured_at == 123.5
        assert int(got.frame[0, 0, 0]) == 7
        assert not (got.frame.flags.writeable)

    def test_older_sequence_is_rejected(self):
        bus = LatestFrame()
        bus.publish(self.envelope(5))
        assert not (bus.publish(self.envelope(4)))
        assert bus.latest().sequence == 5

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
        assert seen
        assert seen == sorted(seen)


class TestLatestTracking:
    @staticmethod
    def snapshot(seq, valid=True):
        return TrackingSnapshot(
            center_x=0.5, center_y=0.45, size=0.3, confidence=0.9,
            captured_at=float(seq), sequence=seq,
            completed_at=float(seq) + 0.01, valid=valid,
        )

    def test_newest_snapshot_wins_and_older_is_discarded(self):
        bus = LatestTracking()
        assert bus.publish(self.snapshot(2))
        assert not (bus.publish(self.snapshot(1)))
        assert bus.latest().sequence == 2
        assert bus.stats()['stale_discarded'] == 1

    def test_snapshot_is_immutable(self):
        snap = self.snapshot(1)
        with pytest.raises(Exception):
            snap.center_x = 0.2


class TestRollingStats:
    def test_percentiles_and_fps(self):
        stats = RollingStats()
        now = 100.0
        for i, value in enumerate([1, 2, 3, 4, 100]):
            stats.add(value, now + i * 0.1)
        assert stats.count() == 5
        assert stats.p50() == 3
        assert stats.p95() == 100
        assert stats.fps() == pytest.approx(12.5, abs=10 ** -(3))

    def test_window_bounds_memory(self):
        stats = RollingStats(window=0.5)
        for i in range(1000):
            stats.add(1, i * 0.01)
        assert stats.count() <= 51

