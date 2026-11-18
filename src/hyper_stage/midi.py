"""MIDI input/output subsystem: device lifecycle, queues and worker threads.

All native MIDI I/O happens here, off the Textual event loop and off the video
threads. The mido callback only copies a small immutable message into a bounded
queue; decoding, coalescing and cross-thread delivery happen on one dispatcher
thread. Feedback is queued from the UI thread and sent by a separate output
thread. Missing backends/ports are reported through a status callback and never
raise into the application.
"""

import queue
import threading
import time

from hyper_stage.midi_mapping import (
    Feedback,
    Incoming,
    MidiRouter,
    normalize_value,
)

CONTINUOUS_OPS = frozenset({"set_value", "step", "select_index"})
POLL_SECONDS = 1.0
QUEUE_LIMIT = 512


class MidiUnavailable(Exception):
    """Raised only by explicit device queries when no backend is installed."""


class MidiManager:
    """Owns MIDI ports and threads. Safe to construct with no hardware."""

    def __init__(self, cfg, mappings, on_event=None, on_status=None, on_learn=None):
        self.cfg = cfg
        self.name_in = cfg.get("midi_input_device", "")
        self.name_out = cfg.get("midi_output_device", "")
        self.feedback_enabled = bool(cfg.get("midi_feedback", True))
        self.auto_reconnect = bool(cfg.get("midi_auto_reconnect", True))
        self.interval_ms = max(0, int(cfg.get("midi_feedback_interval", 20)))
        self.on_event = on_event
        self.on_status = on_status
        self.on_learn = on_learn

        self.router = MidiRouter(
            mappings,
            int(cfg.get("midi_channel", 1)),
            debounce_ms=int(cfg.get("midi_button_debounce", 80)),
        )
        self._mido = None
        self._backend_error = ""
        self._load_backend()

        self._in_q = queue.Queue(maxsize=QUEUE_LIMIT)
        self._out_q = queue.Queue(maxsize=128)
        self._stop = threading.Event()
        self._started = False
        self._in_port = None
        self._out_port = None
        self._learning = False
        self._status = "disabled"
        self._ports_lock = threading.Lock()

        self._dispatcher = threading.Thread(
            target=self._dispatch_loop, name="midi-dispatch", daemon=True
        )
        self._output = threading.Thread(
            target=self._output_loop, name="midi-output", daemon=True
        )
        self._monitor = threading.Thread(
            target=self._monitor_loop, name="midi-monitor", daemon=True
        )

    # --- backend / discovery --------------------------------------------

    def _load_backend(self):
        try:
            import mido

            self._mido = mido
        except ImportError as exc:  # pragma: no cover - depends on environment
            self._backend_error = f"MIDI backend unavailable: {exc}"

    @property
    def status(self):
        return self._status

    @property
    def connected(self):
        return self._in_port is not None or self._out_port is not None

    @property
    def learning(self):
        return self._learning

    def list_devices(self):
        """Return available port names. Call from a worker thread."""
        if self._mido is None:
            return {"inputs": [], "outputs": [], "error": self._backend_error}
        try:
            return {
                "inputs": list(self._mido.get_input_names()),
                "outputs": list(self._mido.get_output_names()),
            }
        except Exception as exc:
            return {"inputs": [], "outputs": [], "error": str(exc)}

    def _set_status(self, message):
        self._status = message
        if self.on_status is not None:
            self._emit(self.on_status, message)

    @staticmethod
    def _emit(callback, *args):
        if callback is None:
            return
        try:
            callback(*args)
        except Exception as exc:  # a dead TUI must never kill a worker
            print(f"warning: MIDI callback failed: {exc}")

    def _open_input(self):
        devices = self.list_devices()
        if devices.get("error"):
            return False, devices["error"]
        names = devices["inputs"]
        name = self.name_in
        if name:
            if name not in names:
                # Explicit config wins: never silently pick a different device.
                return False, f"input device not found: {name}"
        elif names:
            name = names[0]
        else:
            return False, "no MIDI inputs found"
        try:
            port = self._mido.open_input(name, callback=self._on_raw)
            with self._ports_lock:
                self._in_port = port
            return True, f"input: {name}"
        except Exception as exc:
            return False, f"input open failed: {exc}"

    def _open_output(self):
        devices = self.list_devices()
        if devices.get("error"):
            return False, devices["error"]
        names = devices["outputs"]
        name = self.name_out
        if name:
            if name not in names:
                return False, f"output device not found: {name}"
        elif names:
            name = names[0]
        else:
            return False, "no MIDI outputs found"
        try:
            port = self._mido.open_output(name)
            with self._ports_lock:
                self._out_port = port
            return True, f"output: {name}"
        except Exception as exc:
            return False, f"output open failed: {exc}"

    def _close_ports(self):
        with self._ports_lock:
            ports = [self._in_port, self._out_port]
            self._in_port = self._out_port = None
        for port in ports:
            if port is None:
                continue
            try:
                port.close()
            except Exception:
                pass

    # --- lifecycle -------------------------------------------------------

    def start(self):
        """Start workers and try to connect. Call from a non-UI thread."""
        if self._started:
            return
        self._started = True
        self._stop.clear()
        self._dispatcher.start()
        if self.feedback_enabled:
            self._output.start()
        if self._mido is None:
            self._set_status(self._backend_error)
            return
        self._connect()
        self._monitor.start()

    def stop(self, notify=True):
        """Signal every worker and release ports with a bounded wait."""
        if not self._started:
            return
        self._stop.set()
        self._close_ports()
        for thread in (self._dispatcher, self._output, self._monitor):
            if thread.is_alive():
                thread.join(timeout=1.0)
        self._started = False
        if notify:
            self._set_status("stopped")

    def set_mappings(self, mappings):
        """Replace the active routing table (used by MIDI Learn)."""
        self.router.set_mappings(mappings)

    # --- input path ------------------------------------------------------

    def _on_raw(self, msg):
        """mido callback (foreign thread): copy and enqueue, nothing else."""
        if self._stop.is_set():
            return
        captured = None
        if self._learning:
            captured = self.router.capture(msg)
            if captured is not None:
                self._learning = False
                self._enqueue(("learn", captured))
                return
        try:
            msg_type = "cc" if msg.type == "control_change" else msg.type
            number = msg.control if msg_type == "cc" else msg.note
            value = normalize_value(msg)
            incoming = Incoming(msg_type, msg.channel, number, value)
        except (AttributeError, TypeError, ValueError):
            return
        if self.router.is_loopback(
            Feedback(msg_type, msg.channel + 1, number, value)
        ):
            return
        self._enqueue(("raw", incoming))

    def _enqueue(self, item):
        """Bounded input: drop redundant continuous items, keep button edges."""
        if not self._in_q.full():
            self._in_q.put_nowait(item)
            return
        kept = []
        while True:
            try:
                kept.append(self._in_q.get_nowait())
            except queue.Empty:
                break
        for index, old in enumerate(kept):
            events = old[1] if old[0] == "raw" else None
            if isinstance(events, Incoming) and events.type == "cc":
                del kept[index]
                break
        else:
            for old in kept:
                self._in_q.put_nowait(old)
            return  # saturated with discrete events; drop this one item
        for old in kept:
            self._in_q.put_nowait(old)
        self._in_q.put_nowait(item)

    def _dispatch_loop(self):
        while not self._stop.is_set():
            try:
                item = self._in_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if item[0] == "learn":
                self._emit(self.on_learn, item[1])
                continue
            if item[0] != "raw":
                continue
            events = list(self.router.route(item[1]))
            # Coalesce one burst: button edges stay ordered, redundant knob
            # values collapse to the newest / accumulated movement.
            discrete, continuous = [], {}
            for event in events + self._burst():
                if event.op in CONTINUOUS_OPS:
                    if event.op == "step":
                        continuous[event.target] = continuous.get(
                            event.target, 0
                        ) + event.value
                    else:
                        continuous[event.target] = event
                else:
                    discrete.append(event)
            from hyper_stage.midi_mapping import ControlEvent

            for event in discrete:
                self._emit(self.on_event, event)
            for target, event in continuous.items():
                if isinstance(event, ControlEvent):
                    self._emit(self.on_event, event)
                else:
                    self._emit(
                        self.on_event,
                        ControlEvent(target, "step", event),
                    )

    def _burst(self):
        """Pull a few more queued raw messages from this dispatch tick."""
        events = []
        for _ in range(64):
            try:
                item = self._in_q.get_nowait()
            except queue.Empty:
                break
            if item[0] == "learn":
                self._emit(self.on_learn, item[1])
            elif item[0] == "raw":
                events.extend(self.router.route(item[1]))
        return events

    def _connect(self):
        if self._in_port is None:
            ok_in, msg_in = self._open_input()
        else:
            ok_in, msg_in = True, "input connected"
        if self.feedback_enabled and self._out_port is None:
            ok_out, msg_out = self._open_output()
        elif not self.feedback_enabled:
            ok_out, msg_out = True, "feedback disabled"
        else:
            ok_out, msg_out = True, "output connected"
        if ok_in and ok_out:
            self._set_status(f"connected: {msg_in}, {msg_out}")
        else:
            self._set_status(
                "; ".join(m for ok, m in ((ok_in, msg_in), (ok_out, msg_out)) if not ok)
            )

    def _monitor_loop(self):
        while not self._stop.is_set():
            self._stop.wait(POLL_SECONDS)
            if self._stop.is_set():
                return
            if self._mido is None:
                return
            stale = self._in_port is not None and getattr(
                self._in_port, "closed", False
            ) or self._out_port is not None and getattr(
                self._out_port, "closed", False
            )
            if stale:
                self._close_ports()
                self.router.clear_pending()
            if self._in_port is None:
                self._open_input()
            if self.feedback_enabled and self._out_port is None:
                self._open_output()
            if self.auto_reconnect and (self._in_port is None or (
                self.feedback_enabled and self._out_port is None
            )):
                self._connect()

    # --- output path -----------------------------------------------------

    def send_state(self, target, value):
        """Queue feedback for a state change. Called on the Textual thread."""
        if not self.feedback_enabled or not self._started:
            return
        for mapping in self.router.mappings:
            if mapping.target != target:
                continue
            feedback = self.router.feedback(mapping, value)
            if self.router.should_send(feedback, self.interval_ms):
                self._enqueue_out(feedback)

    def send_pulse(self, target):
        """Queue a one-shot LED pulse for an action target (selection/reload)."""
        if not self.feedback_enabled or not self._started:
            return
        for mapping in self.router.mappings:
            if mapping.target != target:
                continue
            feedback = self.router.feedback(mapping, 1)
            if self.router.should_send_pulse(feedback, self.interval_ms):
                self._enqueue_out(feedback)

    def _enqueue_out(self, feedback):
        item = (feedback, time.monotonic())
        try:
            self._out_q.put_nowait(item)
        except queue.Full:
            try:
                self._out_q.get_nowait()
            except queue.Empty:
                return
            self._out_q.put_nowait(item)

    def _output_loop(self):
        while not self._stop.is_set():
            try:
                feedback, created = self._out_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if time.monotonic() - created > 1.0:  # stale LED update
                continue
            with self._ports_lock:
                port = self._out_port
            if port is None:
                ok, _ = self._open_output()
                if not ok:
                    continue
                port = self._out_port
            try:
                # mido's message type is "control_change", not the short "cc"
                # used in mapping strings.
                mido_type = (
                    "control_change" if feedback.type == "cc" else feedback.type
                )
                message = self._mido.Message(
                    mido_type,
                    channel=max(0, feedback.channel - 1),
                    **(
                        {"control": feedback.number, "value": feedback.value}
                        if feedback.type == "cc"
                        else {"note": feedback.number, "velocity": feedback.value}
                    ),
                )
                port.send(message)
            except Exception as exc:
                self._set_status(f"feedback failed: {exc}")

    # --- MIDI Learn ------------------------------------------------------

    def start_learn(self):
        self._learning = True

    def cancel_learn(self):
        self._learning = False
