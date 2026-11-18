"""Declarative MIDI target registry and hardware-independent mapping engine.

This module knows nothing about mido, ports or Textual. It is the single
authoritative list of controllable targets (also used by the TUI for its
switches and tuners), parses INI mapping strings and turns raw MIDI messages
into small application-level events. Keeping it pure makes the whole mapping
layer testable without hardware.
"""

import time
from dataclasses import dataclass

MODES = frozenset(
    {
        "absolute",
        "toggle",
        "momentary",
        "trigger",
        "next",
        "prev",
        "relative_twos",
        "relative_binary",
        "relative_0_127",
    }
)
OPTIONS = frozenset({"invert", "accel"})
MESSAGE_TYPES = frozenset({"cc", "note"})

# Which mapping modes make sense for each kind of target. Validating this here
# prevents a typo in the INI from silently doing nothing useful at runtime.
MODES_BY_KIND = {
    "switch": frozenset({"toggle", "momentary", "trigger"}),
    "tuning": frozenset(
        {"absolute", "relative_twos", "relative_binary", "relative_0_127"}
    ),
    "action": frozenset({"trigger", "next", "prev"}),
    "select_index": frozenset({"absolute"}),
    "corner": frozenset({"trigger"}),
}


@dataclass(frozen=True)
class Target:
    """One controllable application action and its TUI/state coordinates."""

    name: str
    kind: str
    label: str
    widget_id: str = ""
    state_key: str = ""
    suffix: str = ""
    low: float = 0
    high: float = 1
    step: float = 1
    action: str = ""
    args: tuple = ()
    danger: bool = False


def _switch(name, label, widget_id, state_key):
    return Target(name, "switch", label, widget_id=widget_id, state_key=state_key)


def _tuning(name, label, suffix, state_key, low, high, step):
    return Target(
        name,
        "tuning",
        label,
        state_key=state_key,
        suffix=suffix,
        low=low,
        high=high,
        step=step,
    )


def _action(name, label, action, args=(), danger=False):
    return Target(name, "action", label, action=action, args=args, danger=danger)


def _corner(name, label, corner):
    return Target(name, "corner", label, action="set_pip_corner", args=(corner,))


TARGETS = {
    # --- boolean switches (same order as the TUI rows) -------------------
    "tracking_enabled": _switch(
        "tracking_enabled", "Face Tracking", "sw-tracking", "tracking_enabled"
    ),
    "cam_flip": _switch("cam_flip", "Flip Camera", "sw-cam-flip", "cam_flip"),
    "text_enabled": _switch("text_enabled", "Text Overlay", "sw-text", "text_enabled"),
    "lut_enabled": _switch("lut_enabled", "LUT Grading", "sw-lut", "lut_enabled"),
    "shader_enabled": _switch(
        "shader_enabled", "GLSL Shader", "sw-shader", "shader_enabled"
    ),
    "sharpen_enabled": _switch(
        "sharpen_enabled", "Smart Sharpen", "sw-sharpen", "sharpen_enabled"
    ),
    "time_loop_enabled": _switch(
        "time_loop_enabled", "Time Loop (1s)", "sw-loop", "time_loop_enabled"
    ),
    "magic_loop_enabled": _switch(
        "magic_loop_enabled", "Magic Loop (AFK)", "sw-magic", "magic_loop_enabled"
    ),
    "deep_fry_enabled": _switch(
        "deep_fry_enabled", "Deep Fry", "sw-fry", "deep_fry_enabled"
    ),
    "pip_enabled": _switch("pip_enabled", "PiP Slide", "sw-pip", "pip_enabled"),
    "pip_flip": _switch("pip_flip", "PiP Flip", "sw-pip-flip", "pip_flip"),
    # --- continuous tuners ----------------------------------------------
    "zoom": _tuning("zoom", "Zoom", "zoom", "zoom", 1.0, 4.0, 0.05),
    "position_response_time": _tuning(
        "position_response_time", "Position Response", "response",
        "position_response_time", 0.2, 3.0, 0.1,
    ),
    "sharpen_strength": _tuning(
        "sharpen_strength", "Strength", "sharpen", "sharpen_strength", 0.0, 1.0, 0.05
    ),
    "lut_scale": _tuning("lut_scale", "Scale", "lutscale", "lut_scale", 2, 16, 1),
    # --- selection / actions --------------------------------------------
    "lut_next": _action("lut_next", "Next LUT", "next_lut"),
    "lut_prev": _action("lut_prev", "Previous LUT", "prev_lut"),
    "shader_next": _action("shader_next", "Next shader", "next_shader"),
    "shader_prev": _action("shader_prev", "Previous shader", "prev_shader"),
    "pip_next": _action("pip_next", "Next PiP image", "pip_next"),
    "pip_prev": _action("pip_prev", "Previous PiP image", "pip_prev"),
    "pip_index": Target("pip_index", "select_index", "Select PiP image"),
    "pip_corner_tl": _corner("pip_corner_tl", "PiP top-left", 0),
    "pip_corner_tr": _corner("pip_corner_tr", "PiP top-right", 1),
    "pip_corner_bl": _corner("pip_corner_bl", "PiP bottom-left", 2),
    "pip_corner_br": _corner("pip_corner_br", "PiP bottom-right", 3),
    "reload_files": _action("reload_files", "Reload files", "reload_files"),
    "pip_pick": _action("pip_pick", "Pick PiP image", "pick_pip_image"),
    # Never enabled by default; requires explicit config + a second press.
    "shutdown": _action("shutdown", "Quit", "shutdown", danger=True),
}
TARGET_NAMES = tuple(TARGETS)

# The TUI's existing SWITCHES/TUNING tables are derived from the registry, so
# adding a control in one place cannot leave the MIDI list behind.
SWITCHES = {
    target.widget_id: (target.label, target.state_key)
    for target in TARGETS.values()
    if target.kind == "switch"
}
TUNING = {
    target.suffix: (
        target.label,
        target.state_key,
        target.low,
        target.high,
        target.step,
    )
    for target in TARGETS.values()
    if target.kind == "tuning"
}

# Stable TUI control ids -> MIDI targets, derived from the same registry.
# Dropdowns resolve to their natural selection action (picking a LUT and
# cycling to a LUT are the same user intent). The text field and other
# non-mappable controls are deliberately absent: they never steal the target.
CONTROL_TARGETS = {
    **{widget_id: spec[1] for widget_id, spec in SWITCHES.items()},
    **{f"in-{suffix}": spec[1] for suffix, spec in TUNING.items()},
    **{f"inc-{suffix}": spec[1] for suffix, spec in TUNING.items()},
    **{f"dec-{suffix}": spec[1] for suffix, spec in TUNING.items()},
    "lut-select": "lut_next",
    "btn-lut-prev": "lut_prev",
    "btn-lut-next": "lut_next",
    "shader-select": "shader_next",
    "btn-shader-prev": "shader_prev",
    "btn-shader-next": "shader_next",
    "pip-select": "pip_next",
    "btn-pip-prev": "pip_prev",
    "btn-pip-next": "pip_next",
    "btn-pip-pick": "pip_pick",
}


def target_for_control(control_id):
    """Resolve a stable TUI control id to its MIDI target, or None."""
    return CONTROL_TARGETS.get(control_id) if control_id else None


@dataclass(frozen=True)
class Mapping:
    """A single parsed ``type:channel:number:mode[:option...]`` INI value."""

    target: str
    msg_type: str
    channel: int
    number: int
    mode: str
    invert: bool = False
    accel: bool = False

    @property
    def signature(self):
        return (self.msg_type, self.channel, self.number, self.mode, self.invert,
                self.accel)


def parse_mapping(target, spec):
    """Parse and validate one mapping string; raise ValueError when invalid."""
    if target not in TARGETS:
        raise ValueError(f"unknown MIDI target {target!r}")
    if not isinstance(spec, str):
        raise ValueError(f"mapping for {target} must be a string")
    parts = [part.strip() for part in spec.split(":")]
    if len(parts) < 4:
        raise ValueError(f"expected type:channel:number:mode, got {spec!r}")
    msg_type, channel_raw, number_raw, mode = parts[:4]
    options = parts[4:]
    if msg_type not in MESSAGE_TYPES:
        raise ValueError(f"unknown MIDI message type {msg_type!r}")
    if mode not in MODES:
        raise ValueError(f"unknown MIDI mapping mode {mode!r}")
    try:
        channel, number = int(channel_raw), int(number_raw)
    except ValueError as exc:
        raise ValueError(f"non-numeric channel/number in {spec!r}") from exc
    if not 0 <= channel <= 16:
        raise ValueError(f"MIDI channel must be 0-16, got {channel}")
    if not 0 <= number <= 127:
        raise ValueError(f"MIDI controller/note number must be 0-127, got {number}")
    unexpected = [opt for opt in options if opt not in OPTIONS]
    if unexpected:
        raise ValueError(f"unknown MIDI mapping option {unexpected[0]!r}")
    allowed = MODES_BY_KIND[TARGETS[target].kind]
    if mode not in allowed:
        raise ValueError(f"mode {mode!r} is not valid for target {target}")
    return Mapping(
        target=target,
        msg_type=msg_type,
        channel=channel,
        number=number,
        mode=mode,
        invert="invert" in options,
        accel="accel" in options,
    )


def parse_specs(target, raw):
    """Parse one or more ``|``-separated specs for a single target."""
    if not isinstance(raw, str):
        raise ValueError(f"mapping for {target} must be a string")
    specs = [part.strip() for part in raw.split("|") if part.strip()]
    if not specs:
        raise ValueError(f"empty mapping for {target}")
    return [parse_mapping(target, spec) for spec in specs]


def build_mappings(cfg):
    """Build valid mappings from config; return (mappings, warnings)."""
    mappings, warnings = [], []
    for target, spec in (cfg.get("midi_mappings") or {}).items():
        try:
            mappings.extend(parse_specs(target, spec))
        except ValueError as exc:
            warnings.append(f"warning: invalid MIDI mapping {target}={spec!r}: {exc}")
    return mappings, warnings


@dataclass(frozen=True)
class Incoming:
    """Raw MIDI message normalized for the mapping engine (channel 0-based)."""

    type: str
    channel: int
    number: int
    value: int

    @property
    def kind(self):
        """Note On velocity 0 is a Note Off."""
        if self.type == "note_on" and self.value == 0:
            return "note_off"
        return self.type


@dataclass(frozen=True)
class ControlEvent:
    """An application-level command produced by one incoming message."""

    target: str
    op: str  # toggle | set_bool | set_value | step | trigger | select_index
    value: object = None


@dataclass(frozen=True)
class Captured:
    """Result of MIDI Learn: identity of a physical control."""

    type: str
    channel: int  # 1-based
    number: int

    def __str__(self):
        return f"{self.type}:{self.channel}:{self.number}"


@dataclass(frozen=True)
class Feedback:
    """Outgoing message generated from an application state change."""

    type: str
    channel: int  # 1-based
    number: int
    value: int

    @property
    def key(self):
        """Identity for state de-duplication (value is intentionally excluded)."""
        return (self.type, self.channel, self.number)

    @property
    def loopback_key(self):
        """Full message identity: a release (value 0) is a different message."""
        return (self.type, self.channel, self.number, self.value)


def normalize_value(msg):
    """CC-style 0..127 value; note messages carry velocity, CCs carry value."""
    raw = getattr(msg, "value", None)
    if raw is None:
        raw = getattr(msg, "velocity", 0)
    return max(0, min(127, int(raw)))


def _relative_delta(mode, value):
    """Decode a relative encoder value into -127..127 movement."""
    if mode == "relative_twos":
        if value == 0 or value == 64:
            return 0
        return value if value < 64 else value - 128
    if mode == "relative_binary":
        if value == 0 or value == 64:
            return 0
        return value if value < 64 else -(value - 64)
    if mode == "relative_0_127":
        # Controller-native convention: 0 = clockwise, 127 = counter-clockwise.
        # Every other value is not movement and must be ignored.
        if value == 0:
            return 1
        if value == 127:
            return -1
        return 0
    return 0


class MidiRouter:
    """Stateful edge detection + mapping translation. Still hardware-free."""

    def __init__(self, mappings, global_channel=1, debounce_ms=80):
        self.global_channel = global_channel
        # Refractory window for edge-triggered controls: one physical button
        # press can emit several press/release pairs (contact bounce).
        self.debounce_seconds = max(0, int(debounce_ms)) / 1000.0
        self.mappings = tuple(mappings)
        self.groups = {}
        for mapping in self.mappings:
            self.groups.setdefault(
                (mapping.msg_type, mapping.channel), []
            ).append(mapping)
        self._active = {}
        self._last_edge = {}
        self._last_sent = {}
        self._pulse_sent = {}
        self._recent_out = {}

    def set_mappings(self, mappings):
        """Replace mappings at runtime (MIDI Learn), preserving button state."""
        self.mappings = tuple(mappings)
        self.groups = {}
        for mapping in self.mappings:
            self.groups.setdefault(
                (mapping.msg_type, mapping.channel), []
            ).append(mapping)

    def capture(self, msg):
        """Return a Learn candidate for a recognizable message, else None."""
        msg_type = getattr(msg, "type", "")
        if msg_type == "control_change":
            msg_type = "cc"
        elif msg_type in ("note_on", "note_off"):
            msg_type = "note"
        number = getattr(msg, "number", None)
        if number is None:
            number = getattr(msg, "control", getattr(msg, "note", -1))
        value = getattr(msg, "velocity", getattr(msg, "value", 0))
        incoming = Incoming(msg_type, getattr(msg, "channel", -1), number, value)
        kind = incoming.kind
        if kind not in ("cc", "note", "note_on", "note_off"):
            return None
        if not 0 <= incoming.channel <= 15 or not 0 <= number <= 127:
            return None
        return Captured(msg_type, incoming.channel + 1, number)

    @staticmethod
    def learn_mapping(captured, target):
        """Pick a sensible mode for a captured control and target kind."""
        kind = TARGETS[target].kind
        if kind in ("tuning", "select_index"):
            mode = "absolute"
        else:
            mode = "toggle" if kind == "switch" else "trigger"
        msg_type = "cc" if captured.type == "cc" else "note"
        return f"{msg_type}:{captured.channel}:{captured.number}:{mode}"

    def is_loopback(self, feedback):
        """True when an incoming message mirrors one we just sent."""
        import time

        key = feedback.loopback_key
        sent = self._recent_out.get(key)
        return sent is not None and time.monotonic() - sent < 0.05

    def note_sent(self, feedback):
        import time

        self._recent_out[feedback.loopback_key] = time.monotonic()

    def route(self, msg):
        """Translate one incoming message into zero or more control events."""
        if not isinstance(msg, Incoming):
            msg = Incoming(
                getattr(msg, "type", ""),
                getattr(msg, "channel", -1),
                getattr(msg, "control", getattr(msg, "note", -1)),
                getattr(msg, "value", getattr(msg, "velocity", 0)),
            )
        kind = msg.kind
        if kind not in ("cc", "note_on", "note_off"):
            return []
        if not 0 <= msg.channel <= 15:
            return []
        if self.global_channel and msg.channel + 1 != self.global_channel:
            return []

        msg_type = "cc" if kind == "cc" else "note"
        candidates = list(self.groups.get((msg_type, 0), ()))
        candidates.extend(self.groups.get((msg_type, msg.channel + 1), ()))
        events = []
        for mapping in candidates:
            if mapping.number != msg.number:
                continue
            if mapping.channel and mapping.channel != msg.channel + 1:
                continue
            events.extend(self._decode(mapping, msg_type, kind, msg.value))
        return events

    def _button(self, mapping, msg_type, kind, value):
        """Return (held, rising_edge) with release handling and debounce."""
        key = (mapping.signature, msg_type)
        if kind == "note_on":
            rising = not self._active.get(key, False)
            self._active[key] = True
            return True, self._accepted(key, rising)
        if kind == "note_off":
            self._active[key] = False
            return False, False
        held = value >= 64
        rising = held and not self._active.get(key, False)
        self._active[key] = held
        return held, self._accepted(key, rising)

    def _accepted(self, key, rising):
        """Reject a fresh edge that arrives inside the calm window."""
        if not rising:
            return False
        now = time.monotonic()
        if now - self._last_edge.get(key, float("-inf")) < self.debounce_seconds:
            return False
        self._last_edge[key] = now
        return True

    def _decode(self, mapping, msg_type, kind, value):
        mode = mapping.mode
        target = TARGETS[mapping.target]
        if mode in ("toggle", "momentary", "trigger", "next", "prev"):
            held, rising = self._button(mapping, msg_type, kind, value)
            if mode == "momentary":
                state = (not held) if mapping.invert else held
                return [ControlEvent(mapping.target, "set_bool", state)]
            if not rising:
                return []
            if mode == "toggle":
                return [ControlEvent(mapping.target, "toggle")]
            op = "trigger"
            extra = None
            if mode == "next":
                extra = 1
            elif mode == "prev":
                extra = -1
            if target.kind == "switch":
                return [ControlEvent(mapping.target, "toggle")]
            return [ControlEvent(mapping.target, op, extra)]

        if mode == "absolute":
            normalized = normalize_value(Incoming("", 0, 0, value)) / 127.0
            if mapping.invert:
                normalized = 1.0 - normalized
            if target.kind == "select_index":
                return [ControlEvent(mapping.target, "select_index", normalized)]
            return [ControlEvent(mapping.target, "set_value", normalized)]

        delta = _relative_delta(mode, value)
        if not delta:
            return []
        if mapping.invert:
            delta = -delta
        if mapping.accel:
            delta *= 2
        amount = delta * target.step
        if isinstance(target.low, int) and isinstance(target.high, int):
            amount = int(round(amount))
        else:
            amount = round(amount, 4)
        return [ControlEvent(mapping.target, "step", amount)]

    def feedback(self, mapping, value):
        """Build an outgoing message for one mapped target and its new value."""
        target = TARGETS[mapping.target]
        channel = mapping.channel or self.global_channel or 1
        number = mapping.number
        msg_type = mapping.msg_type
        if target.kind == "switch":
            on = bool(value)
            if msg_type == "note":
                return Feedback("note_on" if on else "note_off", channel, number,
                                127 if on else 0)
            return Feedback("cc", channel, number, 127 if on else 0)
        if target.kind == "tuning":
            low, high = target.low, target.high
            normalized = 0.0 if high == low else (float(value) - low) / (high - low)
            return Feedback("cc", channel, number,
                            int(round(max(0.0, min(1.0, normalized)) * 127)))
        if target.kind == "select_index":
            return Feedback("cc", channel, number,
                            int(round(max(0.0, min(1.0, float(value))) * 127)))
        if msg_type == "note":
            return Feedback("note_on", channel, number, 127)
        return Feedback("cc", channel, number, 127)

    def should_send(self, feedback, interval_ms):
        """Throttle + de-dupe. Returns True only for a real, due change."""
        import time

        key = feedback.key
        previous = self._last_sent.get(key)
        now_ms = time.monotonic() * 1000
        if previous is not None:
            if previous[0] == feedback.value:
                return False
            if now_ms - previous[1] < interval_ms:
                return False
        self._last_sent[key] = (feedback.value, now_ms)
        self.note_sent(feedback)
        # Same value later must still be allowed once the state changed back.
        return True

    def should_send_pulse(self, feedback, interval_ms):
        """Throttle a one-shot action pulse (no value de-duplication)."""
        import time

        key = feedback.key
        now_ms = time.monotonic() * 1000
        if now_ms - self._pulse_sent.get(key, -1e9) < interval_ms:
            return False
        self._pulse_sent[key] = now_ms
        self.note_sent(feedback)
        return True

    def clear_pending(self):
        """Drop transient state used for edge detection and loopback tests."""
        self._active.clear()
        self._recent_out.clear()
