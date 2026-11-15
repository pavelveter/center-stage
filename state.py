"""Thread-safe runtime state shared by the TUI and the video thread."""

import threading


class SharedState:
    """Thread-safe runtime state shared by the TUI and the video thread.

    Tunable values and telemetry live behind one lock. The baked LUT table is
    kept apart: the UI's background worker swaps it in while the video thread
    picks it up on the next frame. A revision counter tracks unsaved changes so
    the periodic autosave can never drop an edit made while it was writing.
    """

    TUNABLE = (
        "tracking_enabled",
        "zoom",
        "smoothing",
        "lut_enabled",
        "lut_file",
        "lut_scale",
        "sharpen_enabled",
        "sharpen_strength",
        "shader_enabled",
        "shader_file",
        "time_loop_enabled",
        "deep_fry_enabled",
        "magic_loop_enabled",
    )

    def __init__(self, cfg):
        self._lock = threading.Lock()
        self._values = {key: cfg[key] for key in self.TUNABLE}
        self._telemetry = {
            "fps": 0.0,
            "zoom": float(cfg["zoom"]),
            "tracking": False,
            "device": "starting",
            "status": "starting",
        }
        self._lut_table = None
        self._revision = 0
        self._saved_revision = 0

    def snapshot(self):
        """Copy of the tunable values for one frame of the video thread."""
        with self._lock:
            return dict(self._values)

    def update(self, **values):
        """Set tunable values and mark the config dirty."""
        unknown = set(values) - set(self.TUNABLE)
        if unknown:
            raise KeyError(f"unknown tunable setting: {sorted(unknown)[0]}")
        with self._lock:
            for key, value in values.items():
                if self._values[key] != value:
                    self._values[key] = value
                    self._revision += 1

    def config(self, base):
        """Copy of base config with the current tunable values applied."""
        merged = dict(base)
        with self._lock:
            merged.update(self._values)
        return merged

    def telemetry(self):
        """Copy of the read-only counters published by the video thread."""
        with self._lock:
            return dict(self._telemetry)

    def set_telemetry(self, **values):
        """Publish counters for the UI; never marks the config dirty."""
        with self._lock:
            self._telemetry.update(values)

    @property
    def lut_table(self):
        """Currently active baked LUT table, or None."""
        with self._lock:
            return self._lut_table

    def set_lut_table(self, table):
        """Replace the active baked LUT table (cheap; consumes a reference)."""
        with self._lock:
            self._lut_table = table

    def pending_revision(self):
        """Revision to save, or None when everything is already persisted."""
        with self._lock:
            if self._revision == self._saved_revision:
                return None
            return self._revision

    def mark_saved(self, revision):
        """Record a successful save without hiding edits made during the write."""
        with self._lock:
            if revision == self._revision:
                self._saved_revision = revision
