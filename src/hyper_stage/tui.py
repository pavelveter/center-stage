"""Textual control panel for the hyper-stage video pipeline.

The app renders telemetry and controls on the Textual event loop while
VideoPipeline does capture, tracking and virtual-camera publishing in a daemon
thread. Settings are shared through SharedState (one lock) and persisted back to
the INI file every five seconds when something changed.
"""

import os

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    Rule,
    Select,
    Static,
    Switch,
)

from hyper_stage.config import save_config
from hyper_stage.lut import build_lut_table, cycle_index, list_lut_files
from hyper_stage.midi import MidiManager
from hyper_stage.midi_mapping import (
    SWITCHES,
    TUNING,
    TARGETS,
    build_mappings,
    target_for_control,
)
from hyper_stage.pipeline import VideoPipeline
from hyper_stage.pipmode import (
    CORNER_NAMES,
    is_image_file,
    list_pip_files,
    load_pip_image,
    pick_image_dialog,
)
from hyper_stage.shaders import list_shader_files
from hyper_stage.state import SharedState

# SWITCHES and TUNING now come from the single authoritative MIDI/TUI target
# registry in midi_mapping.py; they are imported above and re-exported here so
# existing imports (tests, plugins) keep working unchanged.
AUTOSAVE_SECONDS = 5.0
TELEMETRY_SECONDS = 0.25
PIP_DISCOVERY_SECONDS = 2.0
MIDI_LEARN_TIMEOUT = 15.0

# Targets offered by the minimal Learn workflow. Shutdown is excluded: it is
# opt-in configuration only and must never be assigned by a stray learn press.
LEARN_TARGETS = tuple(
    name for name, target in TARGETS.items() if not target.danger
)


def format_value(suffix, value):
    """Render a tuning value uniformly in inputs and telemetry."""
    if isinstance(TUNING[suffix][2], int):
        return str(int(value))
    return f"{float(value):.2f}"


def clamp_value(suffix, value):
    """Clamp a value to the documented range and its integer/float type."""
    _, _, low, high, _ = TUNING[suffix]
    value = min(high, max(low, float(value)))
    return int(round(value)) if isinstance(low, int) else round(value, 4)


class HyperStageApp(App):
    """Full-screen control panel with live telemetry and feature toggles."""

    TITLE = "hyper-stage"
    SUB_TITLE = "virtual camera"
    CSS_PATH = "app.tcss"
    # priority bindings keep the shortcuts working while a tuning input or the
    # LUT list has focus; the inputs accept numbers only.
    BINDINGS = [
        Binding("q", "shutdown", "Quit", priority=True),
        # MIDI Learn is key-driven: no extra widgets or config GUI needed.
        Binding("ctrl+m", "midi_learn", "MIDI Learn", priority=True),
        Binding("ctrl+n", "midi_learn_next", show=False, priority=True),
        Binding("ctrl+b", "midi_learn_prev", show=False, priority=True),
        Binding("t", "toggle_tracking", "Tracking", priority=True),
        Binding("c", "toggle_lut", "LUT", key_display="[c]", priority=True),
        Binding("h", "toggle_shader", "Shader", key_display="<h>", priority=True),
        Binding("s", "toggle_sharpen", "Sharpen", priority=True),
        Binding("l", "toggle_time_loop", "Loop", priority=True),
        Binding("m", "toggle_magic_loop", "Magic", priority=True),
        Binding("f", "toggle_deep_fry", "Fry", priority=True),
        Binding("p", "toggle_pip", "PiP", priority=True),
        Binding("r", "reload_files", "Reload", priority=True),
        # Focus the text overlay input (Latin and ЙЦУКЕН positions).
        Binding("i", "focus_text_input", "Text", priority=True),
        Binding("ш", "focus_text_input", show=False, priority=True),
        # Arrow keys move the PiP window along one axis each; up/down change
        # the vertical side, left/right the horizontal one. Horizontal sides
        # are swapped: the call app mirrors the self-preview, so the arrow
        # must match what the user sees, not the raw stream coordinates.
        Binding("left", "snap_corner(0, 1)", show=False, priority=True),
        Binding("up", "snap_corner(1, 0)", show=False, priority=True),
        Binding("down", "snap_corner(1, 1)", show=False, priority=True),
        Binding("right", "snap_corner(0, 0)", show=False, priority=True),
        # -/+ step through the discovered PiP images, 1-0 jump to the first
        # ten; the hint lives next to the PiP dropdown.
        Binding("minus,underscore", "pip_prev", show=False, priority=True),
        Binding("equals_sign,plus", "pip_next", show=False, priority=True),
        Binding("1", "pip_index(0)", show=False, priority=True),
        Binding("2", "pip_index(1)", show=False, priority=True),
        Binding("3", "pip_index(2)", show=False, priority=True),
        Binding("4", "pip_index(3)", show=False, priority=True),
        Binding("5", "pip_index(4)", show=False, priority=True),
        Binding("6", "pip_index(5)", show=False, priority=True),
        Binding("7", "pip_index(6)", show=False, priority=True),
        Binding("8", "pip_index(7)", show=False, priority=True),
        Binding("9", "pip_index(8)", show=False, priority=True),
        Binding("0", "pip_index(9)", show=False, priority=True),
        # LUT cycling stays bound but is not shown in the footer.
        Binding("]", "next_lut", show=False, priority=True),
        Binding("[", "prev_lut", show=False, priority=True),
        # Shader cycling, hidden like the LUT bracket keys.
        Binding(">", "next_shader", show=False, priority=True),
        Binding("<", "prev_shader", show=False, priority=True),
        # Russian layout equivalents: c/h/l/m/p sit on с/р/д/ь/з, the bracket
        # keys on ъ/х, and the angle-bracket keys on ю/б under ЙЦУКЕН.
        Binding("с", "toggle_lut", show=False, priority=True),
        Binding("р", "toggle_shader", show=False, priority=True),
        Binding("к", "reload_files", show=False, priority=True),
        Binding("д", "toggle_time_loop", show=False, priority=True),
        Binding("ь", "toggle_magic_loop", show=False, priority=True),
        Binding("з", "toggle_pip", show=False, priority=True),
        Binding("ъ", "next_lut", show=False, priority=True),
        Binding("х", "prev_lut", show=False, priority=True),
        Binding("ю", "next_shader", show=False, priority=True),
        Binding("б", "prev_shader", show=False, priority=True),
        # The rest of the hotkeys: q/t/s/f sit on й/е/ы/а under ЙЦУКЕН.
        Binding("й", "shutdown", show=False, priority=True),
        Binding("е", "toggle_tracking", show=False, priority=True),
        Binding("ы", "toggle_sharpen", show=False, priority=True),
        Binding("а", "toggle_deep_fry", show=False, priority=True),
    ]

    def __init__(self, cfg, config_path, preview=False):
        super().__init__()
        self.cfg = cfg
        self.config_path = config_path
        self.preview = preview
        self.state = SharedState(cfg)
        self.pipeline = VideoPipeline(cfg, self.state, preview=preview)
        self.lut_files = list_lut_files(cfg["lut_file"])
        self.lut_request = ""
        self.shader_files = list_shader_files(cfg["shader_file"])
        self.pip_files = list_pip_files(cfg["pip_image"])
        self.pip_request = ""
        # Tracking state to restore when Magic Loop turns off again.
        self._tracking_before_magic = None
        # MIDI is optional: constructing the manager never touches hardware.
        mappings, _ = build_mappings(cfg)
        self.midi = MidiManager(
            cfg,
            mappings,
            on_event=lambda event: self.call_from_thread(self._on_midi_event, event),
            on_status=lambda text: self.call_from_thread(self._midi_status, text),
            on_learn=lambda captured: self.call_from_thread(
                self._on_midi_learn, captured
            ),
        )
        self._learn_index = 0
        self._learn_replace_armed = False
        self._learn_timer = None
        self._shutdown_armed = False
        # Last control the *user* actually touched, as a MIDI target name.
        self._active_midi_target = None

    def tuner_controls(self, suffix, values, name=None):
        """Yield an optional mini-label plus the minus/input/plus controls."""
        _, key, _, _, _ = TUNING[suffix]
        if name is not None:
            yield Label(name, classes="tuner-name")
        yield Button("-", id=f"dec-{suffix}")
        yield Input(value=format_value(suffix, values[key]), id=f"in-{suffix}")
        yield Button("+", id=f"inc-{suffix}")

    def switch_row(self, label, widget_id, values, tuner=None, tuner_name=None):
        """Yield one inline row: label, switch (or spacer), optional tuner."""
        with Horizontal(classes="row"):
            yield Label(label, classes="name")
            if widget_id is None:
                yield Static("", classes="switch-spacer")
            else:
                key = SWITCHES[widget_id][1]
                yield Switch(value=bool(values[key]), id=widget_id)
            if tuner is not None:
                yield from self.tuner_controls(
                    tuner,
                    values,
                    name=TUNING[tuner][0] if tuner_name is None else tuner_name,
                )

    def lut_options(self):
        """(label, path) pairs for the LUT dropdown, blank included."""
        return [(os.path.basename(path), path) for path in self.lut_files]

    def pip_options(self):
        """(label, path) pairs for the PiP image dropdown, blank included.

        Labels include the parent folder, so same-named files from
        different directories stay distinguishable.
        """
        return [
            (
                f"…/{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}",
                path,
            )
            for path in self.pip_files
        ]

    def compose(self) -> ComposeResult:
        values = self.state.snapshot()
        if values["magic_loop_enabled"]:
            # The tracking switch renders off immediately; on_mount forces the
            # shared state to match.
            values = {**values, "tracking_enabled": False}
        current = os.path.abspath(values["lut_file"] or "")
        selected = current if current in self.lut_files else Select.NULL
        yield Header()
        with Vertical(id="panel"):
            yield Static("TELEMETRY", classes="panel-title first")
            yield Static("", classes="spacer")
            yield Static("", id="telemetry-body")
            yield Static("", id="tracking-debug", classes="notice")
            yield Static("starting", id="notice", classes="notice")
            yield Rule(classes="rule")
            yield Static("CONTROLS", classes="panel-title")
            yield Static("", classes="spacer")
            yield from self.switch_row(
                "Face Tracking", "sw-tracking", values, tuner="zoom"
            )
            # No switch on this row: spacers keep the tuner columns aligned.
            yield from self.switch_row(
                "Position Response", None, values, tuner="response", tuner_name=""
            )
            yield from self.switch_row("Flip Camera", "sw-cam-flip", values)
            yield from self.switch_row("Text Overlay", "sw-text", values)
            with Horizontal(classes="row"):
                yield Input(
                    value=str(values["text_string"]),
                    placeholder="type text, Enter to apply",
                    id="text-input",
                )
            yield from self.switch_row(
                "LUT Grading", "sw-lut", values, tuner="lutscale"
            )
            with Horizontal(classes="row"):
                yield Select(
                    self.lut_options(),
                    value=selected,
                    prompt="LUT .cube",
                    allow_blank=True,
                    compact=True,
                    type_to_search=False,
                    id="lut-select",
                )
                yield Button("[", id="btn-lut-prev")
                yield Button("]", id="btn-lut-next")
            shader_selected = values["shader_file"]
            if shader_selected not in self.shader_files:
                shader_selected = Select.NULL
            yield from self.switch_row("GLSL Shader", "sw-shader", values)
            with Horizontal(classes="row"):
                yield Select(
                    [(os.path.basename(p), p) for p in self.shader_files],
                    value=shader_selected,
                    prompt="shader .glsl",
                    allow_blank=True,
                    compact=True,
                    type_to_search=False,
                    id="shader-select",
                )
                yield Button("<", id="btn-shader-prev")
                yield Button(">", id="btn-shader-next")
            yield from self.switch_row(
                "Smart Sharpen", "sw-sharpen", values, tuner="sharpen"
            )
            yield from self.switch_row("Time Loop (1s)", "sw-loop", values)
            yield from self.switch_row("Magic Loop (AFK)", "sw-magic", values)
            yield from self.switch_row("Deep Fry", "sw-fry", values)

            with Horizontal(classes="row"):
                yield Label("PiP Slide", classes="name")
                yield Switch(value=bool(values["pip_enabled"]), id="sw-pip")
                yield Label("Flip", classes="tuner-name")
                yield Switch(value=bool(values["pip_flip"]), id="sw-pip-flip")

            # yield from self.switch_row("PiP Slide", "sw-pip", values)
            # yield from self.switch_row("PiP Flip", "sw-pip-flip", values)
            pip_selected = values["pip_image"]
            if pip_selected not in self.pip_files:
                pip_selected = Select.NULL
            with Horizontal(classes="row"):
                yield Select(
                    self.pip_options(),
                    value=pip_selected,
                    prompt="slide image",
                    allow_blank=True,
                    compact=True,
                    type_to_search=False,
                    id="pip-select",
                )
                yield Button("-", id="btn-pip-prev")
                yield Button("+", id="btn-pip-next")
                yield Button("Pick", id="btn-pip-pick")
                yield Label("1-0", classes="tuner-name hint-gap")
        yield Footer()

    def on_mount(self) -> None:
        self.pipeline.start()
        if self.cfg.get("midi_enabled"):
            self.start_midi()
        self.set_interval(TELEMETRY_SECONDS, self.refresh_telemetry)
        self.set_interval(AUTOSAVE_SECONDS, self.autosave)
        # Dropped image files appear in the PiP dropdown without a manual rescan.
        self.set_interval(PIP_DISCOVERY_SECONDS, self.refresh_pip_files)
        saved_theme = self.cfg.get("theme")
        if saved_theme and saved_theme in self.available_themes:
            # Assigning fires watch_theme, which persists the (unchanged)
            # value only when it differs from the config.
            self.theme = saved_theme
        self.sync_dependent_controls()
        if self.state.snapshot()["magic_loop_enabled"]:
            # Started with --magic-loop: force tracking off and dim its
            # switch so the UI matches the pipeline from the first frame.
            self.sync_magic_tracking(True)
        self.refresh_telemetry()
        current = self.state.snapshot()["lut_file"]
        if current:
            # Set before baking so the Select's own Changed event is a no-op.
            self.lut_request = current
            self.bake_lut(current)
        else:
            self.set_notice("no LUT selected")
        current = self.state.snapshot()["pip_image"]
        if current:
            # Set before decoding so the Select's own Changed event is a no-op.
            self.pip_request = current
            self.bake_pip_image(current)

    def on_unmount(self) -> None:
        self.midi.stop(notify=False)
        self._cancel_learn_timer()
        self.save_if_dirty()
        self.pipeline.stop()
        # join() refuses an unstarted thread: if compose or on_mount failed
        # early, the pipeline never ran and shutdown must not crash on it.
        if self.pipeline.ident is not None:
            self.pipeline.join(timeout=5.0)

    def watch_theme(self, theme: str) -> None:
        """Persist theme changes (e.g. from the command palette) to the INI."""
        cfg = getattr(self, "cfg", None)
        if cfg is None or theme == cfg.get("theme"):
            return
        cfg["theme"] = theme
        try:
            save_config(self.config_path, self.state.config(cfg))
        except OSError as exc:
            self.set_notice(f"save failed: {exc}")
            return
        self.set_notice(f"saved {os.path.basename(self.config_path)}")

    # --- telemetry -------------------------------------------------------

    def refresh_telemetry(self) -> None:
        """Repaint the single telemetry line from the shared snapshot."""
        # Workers publish raw counters; the TUI's existing 0.25 s timer copies
        # them into shared telemetry without ever waiting on a worker.
        stats = self.pipeline.stats() if hasattr(self.pipeline, "stats") else {}
        self.state.set_telemetry(**stats)
        values = self.state.telemetry()
        tracking = "TRACKING" if values["tracking"] else "IDLE"
        info = values["device"] if values["status"] == "running" else values["status"]
        line = (
            f"CAP {values['capture_fps']:.1f} | RENDER {values['render_fps']:.1f} | "
            f"Zoom {values['zoom']:.2f}x | {tracking} | {info}"
        )
        try:
            self.query_one("#telemetry-body", Static).update(line)
            if "detection_age" in values:
                age = values["detection_age"]
                debug = (
                    f"raw {values['raw_center'][0]:.2f},{values['raw_center'][1]:.2f} "
                    f"-> filt {values['filtered_center'][0]:.2f},"
                    f"{values['filtered_center'][1]:.2f} | size "
                    f"{values['raw_height']:.2f}->{values['filtered_height'] or 0:.2f} "
                    f"| zoom {values['desired_zoom']:.2f}/{values['actual_zoom']:.2f} "
                    f"| age {age:.2f}s | {values['tracking_state']}"
                )
                self.query_one("#tracking-debug", Static).update(debug)
        except NoMatches:
            # A shutdown tick can fire after the DOM is gone.
            pass

    def set_notice(self, message: str) -> None:
        """Show the latest control/save message in the telemetry panel."""
        try:
            self.query_one("#notice", Static).update(message)
        except NoMatches:
            pass

    # --- persistence -----------------------------------------------------

    def save_if_dirty(self) -> bool:
        """Write the INI file when there are unsaved changes; True on success."""
        revision = self.state.pending_revision()
        if revision is None:
            return True
        try:
            save_config(self.config_path, self.state.config(self.cfg))
        except OSError as exc:
            self.set_notice(f"save failed: {exc}")
            return False
        self.state.mark_saved(revision)
        return True

    def autosave(self) -> None:
        """Persist changed settings every few seconds."""
        if self.state.pending_revision() is None:
            return
        if self.save_if_dirty():
            self.set_notice(f"saved {os.path.basename(self.config_path)}")

    def action_shutdown(self) -> None:
        """Persist pending edits and leave the TUI."""
        self.save_if_dirty()
        self.exit()

    # --- MIDI integration ------------------------------------------------

    @work(thread=True)
    def start_midi(self) -> None:
        """Connect on a worker thread; a missing backend only sets a notice."""
        self.midi.start()

    def _midi_status(self, text: str) -> None:
        self.set_notice(f"MIDI: {text}")

    @staticmethod
    def _set_widget_value(widget, value, attribute="value") -> None:
        """Assign a widget value, marking the change as app-originated.

        Textual delivers Changed messages asynchronously, so a try/finally
        guard cannot tell a programmatic update from a user click. A counter on
        the widget is consumed by the matching interaction handler.
        """
        if getattr(widget, attribute) == value:
            return
        marker = "_hs_programmatic"
        setattr(widget, marker, getattr(widget, marker, 0) + 1)
        setattr(widget, attribute, value)

    def set_switch_value(self, widget_id, value) -> None:
        self._set_widget_value(self.query_one(f"#{widget_id}", Switch), value)

    def set_select_value(self, widget_id, value) -> None:
        self._set_widget_value(self.query_one(f"#{widget_id}", Select), value)

    def set_input_value(self, widget_id, value) -> None:
        self._set_widget_value(self.query_one(f"#{widget_id}", Input), value)

    def note_user_interaction(self, widget) -> None:
        """Remember the last user-touched control that is MIDI-mappable.

        Only real Textual event handlers call this; app-originated widget
        updates are marked with a counter that is consumed here and ignored.
        """
        marker = "_hs_programmatic"
        pending = getattr(widget, marker, 0)
        if pending:
            setattr(widget, marker, pending - 1)
            return
        target = target_for_control(widget.id if widget is not None else None)
        if target is not None and target in LEARN_TARGETS:
            self._active_midi_target = target

    def _on_midi_event(self, event) -> None:
        """MIDI-originated changes never change the Learn target context."""
        self._apply_midi_event(event)

    def _apply_midi_event(self, event) -> None:
        """Apply a decoded MIDI event through the existing app actions."""
        if not self.is_mounted or event.target not in TARGETS:
            return
        target, op = event.target, event.op
        if op == "toggle":
            # Tracking has its own guard so Magic Loop cannot be overridden.
            if target == "tracking_enabled":
                self.action_toggle_tracking(programmatic=True)
            else:
                self.flip_switch(TARGETS[target].widget_id, programmatic=True)
            return
        if op == "set_bool":
            if target == "tracking_enabled" and self.state.snapshot()[
                "magic_loop_enabled"
            ]:
                self.set_notice("tracking is frozen while Magic Loop is on")
                return
            self.set_switch_value(TARGETS[target].widget_id, bool(event.value))
            return
        if op in ("set_value", "step"):
            spec = TARGETS[target]
            if op == "set_value":
                value = spec.low + float(event.value) * (spec.high - spec.low)
            else:
                value = self.get_tuning(spec.suffix) + float(event.value)
            self.apply_tuning(spec.suffix, value)
            return
        if op == "select_index":
            files = self.pip_folder_files()
            if files:
                index = max(0, min(len(files) - 1, int(float(event.value) * len(files))))
                self.apply_pip_image(files[index])
            return
        if op == "trigger":
            self._midi_trigger(target, event.value)

    def _midi_trigger(self, target: str, direction) -> None:
        """Route one-shot MIDI actions to the same code the keys use."""
        if target.startswith("lut_"):
            step = -1 if direction == -1 else (1 if target == "lut_next" else -1)
            self.cycle_lut(step)
        elif target.startswith("shader_"):
            step = -1 if direction == -1 else (1 if target == "shader_next" else -1)
            self.cycle_shader(step)
        elif target in ("pip_next", "pip_prev"):
            step = -1 if (direction == -1 or target == "pip_prev") else 1
            self.cycle_pip_image(step)
        elif target.startswith("pip_corner_"):
            self.set_pip_corner(TARGETS[target].args[0])
        elif target == "reload_files":
            self.action_reload_files()
            self.midi.send_pulse("reload_files")
        elif target == "pip_pick":
            self.action_pick_pip_image()
        elif target == "shutdown":
            self._arm_midi_shutdown()

    def _arm_midi_shutdown(self) -> None:
        """Require a second, deliberate press before quitting from MIDI."""
        if self._shutdown_armed:
            self.action_shutdown()
            return
        self._shutdown_armed = True
        self.set_notice("MIDI quit armed - press again within 3s")
        self.set_timer(3.0, self._disarm_midi_shutdown)

    def _disarm_midi_shutdown(self) -> None:
        self._shutdown_armed = False

    # --- MIDI Learn ------------------------------------------------------

    def action_midi_learn(self) -> None:
        """Learn into the last user-touched control, not a hardcoded default."""
        if self.midi.learning:
            self.midi.cancel_learn()
            self._cancel_learn_timer()
            self._learn_replace_armed = False
            self.set_notice("MIDI LEARN → cancelled")
            return
        if not self._active_midi_target:
            self.set_notice("MIDI LEARN → select a TUI control first, then Ctrl+M")
            return
        self._learn_index = LEARN_TARGETS.index(self._active_midi_target)
        self._learn_replace_armed = False
        self.midi.start_learn()
        self._learn_timer = self.set_timer(MIDI_LEARN_TIMEOUT, self._midi_learn_timeout)
        self._show_learn_notice()

    def action_midi_learn_next(self) -> None:
        if self.midi.learning:
            self._learn_index = (self._learn_index + 1) % len(LEARN_TARGETS)
            self._learn_replace_armed = False
            self._show_learn_notice()

    def action_midi_learn_prev(self) -> None:
        if self.midi.learning:
            self._learn_index = (self._learn_index - 1) % len(LEARN_TARGETS)
            self._learn_replace_armed = False
            self._show_learn_notice()

    def _learn_text(self, captured=None) -> str:
        label = TARGETS[LEARN_TARGETS[self._learn_index]].label
        text = f"MIDI LEARN → {label}"
        if captured is not None:
            text += f" → {captured.type.upper()} {captured.number} CH {captured.channel}"
        return text

    def _show_learn_notice(self, captured=None) -> None:
        text = self._learn_text(captured)
        if captured is None:
            text += " (Ctrl+M cancels, Ctrl+N/B change target)"
        self.set_notice(text)

    def _cancel_learn_timer(self) -> None:
        if self._learn_timer is not None:
            self._learn_timer.stop()
            self._learn_timer = None

    def _midi_learn_timeout(self) -> None:
        if self.midi.learning:
            self.midi.cancel_learn()
            self.set_notice("MIDI learn: timed out")

    def _on_midi_learn(self, captured) -> None:
        """Assign a captured control to the selected target and persist it."""
        self._cancel_learn_timer()
        self.midi.cancel_learn()
        target = LEARN_TARGETS[self._learn_index]
        spec = self.midi.router.learn_mapping(captured, target)
        existing = self.cfg.setdefault("midi_mappings", {}).get(target)
        if existing and existing != spec and not self._learn_replace_armed:
            # Explicit replacement only: a second capture is the confirmation.
            self._learn_replace_armed = True
            self.midi.start_learn()
            self._learn_timer = self.set_timer(
                MIDI_LEARN_TIMEOUT, self._midi_learn_timeout
            )
            self.set_notice(self._learn_text(captured) + " - capture again to replace")
            return
        self.cfg.setdefault("midi_mappings", {})[target] = spec
        mappings, _ = build_mappings(self.cfg)
        self.midi.set_mappings(mappings)
        self._learn_replace_armed = False
        try:
            save_config(self.config_path, self.state.config(self.cfg))
            self.set_notice(self._learn_text(captured))
        except OSError as exc:
            self.set_notice(f"MIDI learn save failed: {exc}")

    # --- switches and shortcuts -----------------------------------------

    def set_tuning_disabled(self, suffix: str, disabled: bool) -> None:
        """Dim and disable one tuning row's input and buttons."""
        self.query_one(f"#in-{suffix}", Input).disabled = disabled
        self.query_one(f"#dec-{suffix}", Button).disabled = disabled
        self.query_one(f"#inc-{suffix}", Button).disabled = disabled

    def sync_dependent_controls(self) -> None:
        """Dim each feature's tuning controls only while that feature is on.

        Dropdowns and the PiP Pick button stay enabled with their feature
        off: pre-selecting an effect or image before switching it on is the
        whole point of keeping them separate from the toggle.
        """
        values = self.state.snapshot()
        self.set_tuning_disabled("lutscale", not bool(values["lut_enabled"]))
        self.set_tuning_disabled("sharpen", not bool(values["sharpen_enabled"]))

    def on_switch_changed(self, event: Switch.Changed) -> None:
        self.note_user_interaction(event.switch)
        spec = SWITCHES.get(event.switch.id or "")
        if spec is None:
            return
        label, key = spec
        if key == "magic_loop_enabled":
            self.sync_magic_tracking(event.switch.value)
        self.state.update(**{key: event.switch.value})
        self.midi.send_state(key, event.switch.value)
        self.sync_dependent_controls()
        self.set_notice(f"{label}: {'on' if event.switch.value else 'off'}")

    def sync_magic_tracking(self, magic_on: bool) -> None:
        """Mirror Magic Loop in the Face Tracking switch and shared state.

        Magic Loop freezes the framing, so tracking is forced off and its
        switch dims; turning Magic Loop off restores the previous tracking
        state so the UI always reflects what the pipeline actually does.
        """
        switch = self.query_one("#sw-tracking", Switch)
        if magic_on:
            if self._tracking_before_magic is None:
                self._tracking_before_magic = bool(
                    self.state.snapshot()["tracking_enabled"]
                )
            # One atomic update: if tracking went off a frame before the loop
            # went on, the crop would already start easing back to the base
            # framing and the frozen crop would not match the history.
            self.state.update(tracking_enabled=False, magic_loop_enabled=True)
            self.midi.send_state("tracking_enabled", False)
            self.midi.send_state("magic_loop_enabled", True)
            self.set_switch_value("sw-tracking", False)
            switch.disabled = True
        else:
            previous = self._tracking_before_magic
            self._tracking_before_magic = None
            switch.disabled = False
            if previous is not None:
                # Setting the same value fires no Changed message.
                self.set_switch_value("sw-tracking", previous)
            self.state.update(tracking_enabled=bool(switch.value))
            self.midi.send_state("tracking_enabled", bool(switch.value))

    def flip_switch(self, widget_id: str, programmatic: bool = False) -> None:
        """Toggle a switch; its Changed message updates the shared state."""
        switch = self.query_one(f"#{widget_id}", Switch)
        value = not switch.value
        if programmatic:
            self._set_widget_value(switch, value)
        else:
            switch.value = value

    def action_toggle_tracking(self, programmatic: bool = False) -> None:
        if self.state.snapshot()["magic_loop_enabled"]:
            self.set_notice("tracking is frozen while Magic Loop is on")
            return
        self.flip_switch("sw-tracking", programmatic=programmatic)

    def action_toggle_lut(self) -> None:
        self.flip_switch("sw-lut")

    def action_toggle_shader(self) -> None:
        self.flip_switch("sw-shader")

    def action_reload_files(self) -> None:
        """Rescan LUT/shader folders and image files, and reload the active files."""
        values = self.state.snapshot()
        self.lut_files = list_lut_files(values["lut_file"])
        self.shader_files = list_shader_files(values["shader_file"])
        self.pip_files = list_pip_files(values["pip_image"])
        lut = self.query_one("#lut-select", Select)
        lut.set_options([(os.path.basename(p), p) for p in self.lut_files])
        self.set_select_value(
            "lut-select",
            values["lut_file"] if values["lut_file"] in self.lut_files else Select.NULL,
        )
        shader = self.query_one("#shader-select", Select)
        shader.set_options([(os.path.basename(p), p) for p in self.shader_files])
        self.set_select_value(
            "shader-select",
            values["shader_file"]
            if values["shader_file"] in self.shader_files
            else Select.NULL,
        )
        pip = self.query_one("#pip-select", Select)
        pip.set_options(self.pip_options())
        self.set_select_value(
            "pip-select",
            values["pip_image"]
            if values["pip_image"] in self.pip_files
            else Select.NULL,
        )
        if values["lut_file"]:
            self.lut_request = values["lut_file"]
            self.bake_lut(values["lut_file"])
        if values["pip_image"]:
            self.pip_request = values["pip_image"]
            self.bake_pip_image(values["pip_image"])
        self.pipeline.reload_shader()
        self.set_notice("reloaded LUTs, shaders and PiP images")

    def action_toggle_sharpen(self) -> None:
        self.flip_switch("sw-sharpen")

    def action_toggle_time_loop(self) -> None:
        self.flip_switch("sw-loop")

    def action_toggle_magic_loop(self) -> None:
        self.flip_switch("sw-magic")

    def action_toggle_deep_fry(self) -> None:
        self.flip_switch("sw-fry")

    def action_toggle_pip(self) -> None:
        self.flip_switch("sw-pip")

    def action_snap_corner(self, axis: int, value: int) -> None:
        """Move the PiP window along one axis; the arrow keys land here.

        axis 0 = X (left/right side), axis 1 = Y (top/bottom side); value
        0 or 1 picks the side. The other axis keeps its current side, so
        each arrow moves the window instead of jumping to a fixed corner.
        """
        values = self.state.snapshot()
        if not values["pip_enabled"]:
            self.set_notice("PiP is off")
            return
        if not values["pip_image"]:
            self.set_notice("no PiP image loaded")
            return
        corner = int(values["pip_corner"])
        x_right, y_bottom = corner & 1, corner >> 1
        if axis == 1:
            y_bottom = 1 if value else 0
        else:
            x_right = 1 if value else 0
        corner = x_right + 2 * y_bottom
        self.state.update(pip_corner=corner)
        self.midi.send_pulse(
            ("pip_corner_tl", "pip_corner_tr", "pip_corner_bl", "pip_corner_br")[corner]
        )
        self.set_notice(f"PiP position: {CORNER_NAMES[corner]}")

    def set_pip_corner(self, corner: int) -> None:
        """Jump to an absolute PiP corner via the existing axis-snap action."""
        current = int(self.state.snapshot()["pip_corner"])
        wanted_x, wanted_y = corner & 1, corner >> 1
        if wanted_x != current & 1:
            self.action_snap_corner(0, wanted_x)
        if wanted_y != current >> 1:
            self.action_snap_corner(1, wanted_y)

    @work(thread=True)
    def action_pick_pip_image(self) -> None:
        """Open the native macOS file dialog off the TUI thread."""
        path = pick_image_dialog()
        if path:
            self.call_from_thread(self.apply_pip_image, path)

    # --- PiP image selection ---------------------------------------------

    def action_pip_prev(self) -> None:
        self.cycle_pip_image(-1)

    def action_pip_next(self) -> None:
        self.cycle_pip_image(1)

    def action_pip_index(self, index: int) -> None:
        files = self.pip_folder_files()
        if index < len(files):
            self.apply_pip_image(files[index])

    def pip_folder_files(self):
        """Images from the current image's folder; all files when none picked.

        Keeps -/+ and the digit jumps inside one directory instead of
        running off into the app folder's images.
        """
        current = self.state.snapshot()["pip_image"]
        if not current:
            return self.pip_files
        folder = os.path.dirname(current)
        files = [p for p in self.pip_files if os.path.dirname(p) == folder]
        return files or self.pip_files

    def cycle_pip_image(self, step: int) -> None:
        """Select the previous (step=-1) or next (step=1) image, clamped.

        Cycling stays inside the current image's folder and stops at its
        first/last file (sorted by name). With no image selected yet, next
        picks the first file and prev the last one. Works while PiP is
        off, like the dropdown and Pick.
        """
        files = self.pip_folder_files()
        current = self.state.snapshot()["pip_image"]
        index = (
            files.index(current)
            if current in files
            else (-1 if step > 0 else len(files))
        )
        index = min(max(index + step, 0), len(files) - 1)
        self.apply_pip_image(files[index])

    def apply_pip_image(self, path: str) -> None:
        """Validate, record and decode one image for the PiP background."""
        if not is_image_file(path):
            self.set_notice("PiP image must be .gif/.jpg/.jpeg/.png")
            return
        path = os.path.abspath(path)
        self.pip_request = path
        self.state.update(pip_image=path)
        self.midi.send_pulse("pip_next")
        if path not in self.pip_files:
            # A picked file outside the app directory still becomes an option.
            self.pip_files.append(path)
            self.pip_files.sort()
            self.query_one("#pip-select", Select).set_options(self.pip_options())
        self.set_select_value("pip-select", path)
        self.set_notice(f"PiP image: {os.path.basename(path)}")
        self.bake_pip_image(path)

    @work(thread=True, exclusive=True, group="pip")
    def bake_pip_image(self, path: str) -> None:
        """Decode and letterbox-fit the image off the TUI thread."""
        frame = load_pip_image(path, (int(self.cfg["out_w"]), int(self.cfg["out_h"])))
        self.call_from_thread(self.pip_image_loaded, path, frame)

    def pip_image_loaded(self, path: str, frame) -> None:
        """Apply a freshly decoded image, ignoring superseded selections."""
        if path != self.pip_request:
            return
        if frame is None:
            self.set_notice(f"PiP image failed: {os.path.basename(path)}")
            return
        self.state.set_pip_frame(frame)
        self.set_notice(f"PiP image: {os.path.basename(path)}")

    def refresh_pip_files(self) -> None:
        """Pick up files dropped into the app directory without a manual rescan."""
        if not self.is_mounted:
            # The discovery timer can fire while the app is shutting down.
            return
        current = self.state.snapshot()["pip_image"]
        # Merge instead of replacing: scanning follows the current image's
        # folder, so a Pick from another folder would otherwise drop every
        # previously discovered image from the dropdown and break cycling.
        found = set(list_pip_files(current))
        found.update(p for p in self.pip_files if os.path.exists(p))
        if current:
            found.add(current)
        files = sorted(found, key=str.lower)
        if files == self.pip_files:
            return
        self.pip_files = files
        select = self.query_one("#pip-select", Select)
        select.set_options(self.pip_options())
        self.set_select_value(
            "pip-select", current if current in files else Select.NULL
        )
        self.set_notice("found new PiP images")

    # --- LUT selection ---------------------------------------------------

    def cycle_lut(self, step: int) -> None:
        """Select the next/previous .cube file and bake it in the background."""
        if not self.lut_files:
            self.set_notice("no .cube files found in LUT/ or the current directory")
            return
        current = self.state.snapshot()["lut_file"]
        try:
            index = self.lut_files.index(current)
        except ValueError:
            # Nothing selected yet: move to the first file going forward and
            # to the last one going backward.
            index = -1 if step > 0 else 0
        next_index = cycle_index(index, len(self.lut_files), step)
        path = self.lut_files[next_index]
        if path == current and current:
            return
        self.lut_request = path
        self.state.update(lut_file=path)
        self.midi.send_pulse("lut_next")
        self.set_select_value("lut-select", path)
        self.set_notice(f"LUT: {os.path.basename(path)}")
        self.bake_lut(path)

    def action_next_lut(self) -> None:
        self.cycle_lut(1)

    def action_prev_lut(self) -> None:
        self.cycle_lut(-1)

    def cycle_shader(self, step: int) -> None:
        """Select the next/previous .glsl file for the video pipeline."""
        if not self.shader_files:
            self.set_notice(
                "no .glsl shaders found in shaders/ or the current directory"
            )
            return
        current = self.state.snapshot()["shader_file"]
        try:
            index = self.shader_files.index(current)
        except ValueError:
            # Nothing selected yet: move to the first file going forward and
            # to the last one going backward.
            index = -1 if step > 0 else 0
        next_index = cycle_index(index, len(self.shader_files), step)
        path = self.shader_files[next_index]
        if path == current and current:
            return
        self.state.update(shader_file=path)
        self.midi.send_pulse("shader_next")
        self.set_select_value("shader-select", path)
        self.set_notice(f"shader: {os.path.basename(path)}")

    def action_next_shader(self) -> None:
        self.cycle_shader(1)

    def action_prev_shader(self) -> None:
        self.cycle_shader(-1)

    def on_select_changed(self, event: Select.Changed) -> None:
        """Bake the chosen .cube file, or drop grading on a blank selection."""
        self.note_user_interaction(event.select)
        # A blank selection delivers the Select.NULL sentinel, not "".
        value = event.value if isinstance(event.value, str) else ""
        if event.select.id == "pip-select":
            # Two guards: value must match the select's current value (a
            # stale Changed event from an earlier programmatic set carries
            # the old value) and pip_request (the newest choice wins).
            if (
                value
                and value == event.select.value
                and value != self.pip_request
            ):
                self.apply_pip_image(value)
            return
        if event.select.id == "shader-select":
            if value and value != self.state.snapshot()["shader_file"]:
                self.state.update(shader_file=value)
                self.set_notice(f"shader: {os.path.basename(value)}")
            return
        if event.select.id != "lut-select":
            return
        if not value or value == self.lut_request:
            return
        self.lut_request = value
        self.state.update(lut_file=value)
        self.set_notice(f"loading {os.path.basename(value)}")
        self.bake_lut(value)

    @work(thread=True, exclusive=True, group="lut")
    def bake_lut(self, path: str) -> None:
        """Bake a LUT in the background so the video stream never stalls."""
        table = build_lut_table(path)
        self.call_from_thread(self.lut_baked, path, table)

    def lut_baked(self, path: str, table) -> None:
        """Apply a freshly baked table, ignoring superseded selections."""
        if path != self.lut_request:
            return
        if table is None:
            self.set_notice(f"LUT failed: {os.path.basename(path)}")
            return
        self.state.set_lut_table(table)
        self.set_notice(f"LUT: {os.path.basename(path)}")

    # --- tuning ----------------------------------------------------------

    def get_tuning(self, suffix: str) -> int | float:
        key = TUNING[suffix][1]
        return self.state.snapshot()[key]

    def apply_tuning(self, suffix: str, value) -> None:
        """Clamp, store and display one tuning value."""
        label, key, _, _, _ = TUNING[suffix]
        value = clamp_value(suffix, value)
        self.state.update(**{key: value})
        self.midi.send_state(key, value)
        self.set_input_value(f"in-{suffix}", format_value(suffix, value))
        self.set_notice(f"{label}: {format_value(suffix, value)}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        self.note_user_interaction(event.button)
        if button_id == "btn-pip-pick":
            self.action_pick_pip_image()
            return
        if button_id == "btn-pip-prev":
            self.cycle_pip_image(-1)
            return
        if button_id == "btn-pip-next":
            self.cycle_pip_image(1)
            return
        if button_id in ("btn-lut-prev", "btn-lut-next"):
            self.cycle_lut(1 if button_id == "btn-lut-next" else -1)
            return
        if button_id in ("btn-shader-prev", "btn-shader-next"):
            self.cycle_shader(1 if button_id == "btn-shader-next" else -1)
            return
        for prefix, direction in (("dec-", -1), ("inc-", 1)):
            if not button_id.startswith(prefix):
                continue
            suffix = button_id[len(prefix) :]
            if suffix not in TUNING:
                return
            step = TUNING[suffix][4]
            self.apply_tuning(suffix, self.get_tuning(suffix) + step * direction)
            return

    async def _check_bindings(self, key: str, priority: bool = False) -> bool:
        """Let the text field receive printable keys instead of hotkeys."""
        if priority and key not in ("escape", "tab", "shift+tab"):
            focused = self.focused
            if isinstance(focused, Input) and focused.id == "text-input":
                return False
        return await super()._check_bindings(key, priority=priority)

    def action_focus_text_input(self) -> None:
        """Jump to the overlay text field (i / ш)."""
        self.query_one("#text-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.note_user_interaction(event.input)
        if event.input.id == "text-input":
            text = event.value.strip()
            self.state.update(text_string=text)
            # Publishing a line implies showing it: turn the overlay on.
            switch = self.query_one("#sw-text", Switch)
            if not switch.value:
                self.set_switch_value("sw-text", True)
                self.state.update(text_enabled=True)
            self.set_notice(f"text overlay: {text or 'cleared'}")
            event.input.blur()
            return
        suffix = (event.input.id or "").removeprefix("in-")
        if suffix not in TUNING:
            return
        try:
            value = float(event.value)
        except ValueError:
            self.apply_tuning(suffix, self.get_tuning(suffix))
            return
        self.apply_tuning(suffix, value)
