"""Textual control panel for the center-stage video pipeline.

The app renders telemetry and controls on the Textual event loop while
VideoPipeline does capture, tracking and virtual-camera publishing in a daemon
thread. Settings are shared through SharedState (one lock) and persisted back to
the INI file every five seconds when something changed.
"""

import os

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
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

import tracker

# Widget id -> (label, SharedState key). Order defines the on-screen order.
# sw-lut and sw-sharpen are composed inside their own blocks and skipped in
# the FEATURES loop, but stay here so Changed messages update the state.
SWITCHES = {
    "sw-tracking": ("Face Tracking", "tracking_enabled"),
    "sw-lut": ("LUT Grading", "lut_enabled"),
    "sw-sharpen": ("Smart Sharpen", "sharpen_enabled"),
    "sw-loop": ("Time Loop (1s)", "time_loop_enabled"),
    "sw-fry": ("Deep Fry / Glitch", "deep_fry_enabled"),
}

# Input suffix -> (label, SharedState key, minimum, maximum, button step).
# Integer bounds mean an integer setting (LUT scale), floats otherwise.
TUNING = {
    "zoom": ("Zoom", "zoom", 1.0, 4.0, 0.05),
    "smoothing": ("Smoothing", "smoothing", 0.01, 0.30, 0.01),
    "sharpen": ("Strength", "sharpen_strength", 0.0, 1.0, 0.05),
    "lutscale": ("Intensity / Scale", "lut_scale", 2, 16, 1),
}

AUTOSAVE_SECONDS = 5.0
TELEMETRY_SECONDS = 0.25


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


class CenterStageApp(App):
    """Full-screen control panel with live telemetry and feature toggles."""

    TITLE = "center-stage"
    SUB_TITLE = "virtual camera"
    CSS = """
    #body {
        height: 1fr;
        padding: 0 1;
    }
    #body.compact {
        layout: vertical;
    }
    #left {
        width: 1fr;
        height: 1fr;
    }
    #right {
        width: 1fr;
        height: 1fr;
    }
    #body.compact #left,
    #body.compact #right {
        height: auto;
    }
    .sidebar {
        height: 1fr;
        border: round $primary;
        padding: 0 1 1 1;
    }
    #body.compact .sidebar {
        height: auto;
    }
    .panel-title {
        color: $accent;
        text-style: bold;
        margin-top: 1;
    }
    .panel-title.first {
        margin-top: 0;
    }
    .sub-title {
        color: $text-muted;
        text-style: bold;
        margin-top: 1;
    }
    #telemetry-body {
        padding: 1 0;
    }
    .notice {
        color: $text-muted;
        height: auto;
    }
    .row {
        height: auto;
        align-vertical: middle;
    }
    .row .name {
        width: 22;
    }
    .row Switch {
        margin-left: 1;
    }
    .row Input {
        width: 12;
        height: 1;
        border: none;
    }
    .row Button {
        width: 5;
        min-width: 3;
        height: 1;
        border: none;
        background: $surface-lighten-2;
    }
    #lut-select {
        width: 1fr;
        margin-left: 1;
        min-width: 16;
        height: 1;
        border: none;
        background: $surface-lighten-2;
    }
    #lut-select SelectCurrent {
        border: none !important;
        height: 1;
        padding: 0 1;
    }
    .rule {
        margin: 1 0;
        color: $primary-darken-1;
    }
    """
    # priority bindings keep the shortcuts working while a tuning input or the
    # LUT list has focus; the inputs accept numbers only.
    BINDINGS = [
        Binding("q", "shutdown", "Quit", priority=True),
        Binding("t", "toggle_tracking", "Tracking", priority=True),
        Binding("l", "toggle_lut", "LUT", priority=True),
        Binding("s", "toggle_sharpen", "Sharpen", priority=True),
        Binding("m", "toggle_time_loop", "Loop", priority=True),
        Binding("f", "toggle_deep_fry", "Fry", priority=True),
    ]

    def __init__(self, cfg, config_path, preview=False):
        super().__init__()
        self.cfg = cfg
        self.config_path = config_path
        self.preview = preview
        self.state = tracker.SharedState(cfg)
        self.pipeline = tracker.VideoPipeline(cfg, self.state, preview=preview)
        self.lut_files = tracker.list_lut_files(cfg["lut_file"])
        self.lut_request = ""

    def tuning_row(self, suffix, values):
        """Yield one labelled minus/input/plus row for a tuning setting."""
        label, key, _, _, _ = TUNING[suffix]
        with Horizontal(classes="row"):
            yield Label(label, classes="name")
            yield Button("-", id=f"dec-{suffix}")
            yield Input(value=format_value(suffix, values[key]), id=f"in-{suffix}")
            yield Button("+", id=f"inc-{suffix}")

    def lut_options(self):
        """(label, path) pairs for the LUT dropdown, blank included."""
        return [(os.path.basename(path), path) for path in self.lut_files]

    def compose(self) -> ComposeResult:
        values = self.state.snapshot()
        current = os.path.abspath(values["lut_file"] or "")
        selected = current if current in self.lut_files else Select.NULL
        yield Header()
        with Horizontal(id="body"):
            with Vertical(classes="sidebar", id="left"):
                yield Static("TELEMETRY", classes="panel-title first")
                yield Static(id="telemetry-body")
                yield Static("starting", id="notice", classes="notice")
                yield Static("TUNING", classes="sub-title")
                yield from self.tuning_row("zoom", values)
                yield from self.tuning_row("smoothing", values)
            with VerticalScroll(classes="sidebar", id="right"):
                yield Static(
                    "IMAGE PROCESSING & FEATURES", classes="panel-title first"
                )
                with Horizontal(classes="row"):
                    yield Label("LUT Grading", classes="name")
                    yield Switch(value=bool(values["lut_enabled"]), id="sw-lut")
                    yield Select(
                        self.lut_options(),
                        value=selected,
                        prompt="LUT .cube",
                        allow_blank=True,
                        compact=True,
                        type_to_search=False,
                        id="lut-select",
                    )
                yield from self.tuning_row("lutscale", values)
                yield Rule(classes="rule")
                with Horizontal(classes="row"):
                    yield Label("Smart Sharpen", classes="name")
                    yield Switch(
                        value=bool(values["sharpen_enabled"]), id="sw-sharpen"
                    )
                yield from self.tuning_row("sharpen", values)
                yield Rule(classes="rule")
                yield Static("FEATURES", classes="sub-title")
                for widget_id, (label, key) in SWITCHES.items():
                    if widget_id in ("sw-lut", "sw-sharpen"):
                        continue
                    with Horizontal(classes="row"):
                        yield Label(label, classes="name")
                        yield Switch(value=bool(values[key]), id=widget_id)
        yield Footer()

    def on_mount(self) -> None:
        self.pipeline.start()
        self.set_interval(TELEMETRY_SECONDS, self.refresh_telemetry)
        self.set_interval(AUTOSAVE_SECONDS, self.autosave)
        self.sync_dependent_controls()
        self.refresh_telemetry()
        current = self.state.snapshot()["lut_file"]
        if current:
            # Set before baking so the Select's own Changed event is a no-op.
            self.lut_request = current
            self.bake_lut(current)
        else:
            self.set_notice("no LUT selected")

    def on_unmount(self) -> None:
        self.save_if_dirty()
        self.pipeline.stop()
        self.pipeline.join(timeout=5.0)

    # --- telemetry -------------------------------------------------------

    def refresh_telemetry(self) -> None:
        """Repaint the counters from the shared telemetry snapshot."""
        values = self.state.telemetry()
        tracking = "TRACKING" if values["tracking"] else "IDLE"
        lines = [
            f"FPS            {values['fps']:5.1f}",
            f"Zoom           {values['zoom']:.2f}x",
            f"Face tracking  {tracking}",
            f"Virtual camera {values['device']}",
            f"Status         {values['status']}",
        ]
        try:
            self.query_one("#telemetry-body", Static).update("\n".join(lines))
        except NoMatches:
            # A shutdown tick can fire after the DOM is gone.
            pass

    def set_notice(self, message: str) -> None:
        """Show the latest control/save message in the telemetry panel."""
        try:
            self.query_one("#notice", Static).update(message)
        except NoMatches:
            pass

    def on_resize(self, event) -> None:
        """Stack the panels vertically on narrow terminals."""
        self.query_one("#body").set_class(event.size.width < 88, "compact")

    # --- persistence -----------------------------------------------------

    def save_if_dirty(self) -> bool:
        """Write the INI file when there are unsaved changes; True on success."""
        revision = self.state.pending_revision()
        if revision is None:
            return True
        try:
            tracker.save_config(
                self.config_path, self.state.config(self.cfg)
            )
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

    # --- switches and shortcuts -----------------------------------------

    def set_tuning_disabled(self, suffix: str, disabled: bool) -> None:
        """Dim and disable one tuning row's input and buttons."""
        self.query_one(f"#in-{suffix}", Input).disabled = disabled
        self.query_one(f"#dec-{suffix}", Button).disabled = disabled
        self.query_one(f"#inc-{suffix}", Button).disabled = disabled

    def sync_dependent_controls(self) -> None:
        """Enable each feature's controls only while that feature is on."""
        values = self.state.snapshot()
        lut_on = bool(values["lut_enabled"])
        self.query_one("#lut-select", Select).disabled = not lut_on
        self.set_tuning_disabled("lutscale", not lut_on)
        self.set_tuning_disabled("sharpen", not bool(values["sharpen_enabled"]))

    def on_switch_changed(self, event: Switch.Changed) -> None:
        spec = SWITCHES.get(event.switch.id or "")
        if spec is None:
            return
        label, key = spec
        self.state.update(**{key: event.switch.value})
        self.sync_dependent_controls()
        self.set_notice(f"{label}: {'on' if event.switch.value else 'off'}")

    def flip_switch(self, widget_id: str) -> None:
        """Toggle a switch; its Changed message updates the shared state."""
        switch = self.query_one(f"#{widget_id}", Switch)
        switch.value = not switch.value

    def action_toggle_tracking(self) -> None:
        self.flip_switch("sw-tracking")

    def action_toggle_lut(self) -> None:
        self.flip_switch("sw-lut")

    def action_toggle_sharpen(self) -> None:
        self.flip_switch("sw-sharpen")

    def action_toggle_time_loop(self) -> None:
        self.flip_switch("sw-loop")

    def action_toggle_deep_fry(self) -> None:
        self.flip_switch("sw-fry")

    # --- LUT selection ---------------------------------------------------

    def on_select_changed(self, event: Select.Changed) -> None:
        """Bake the chosen .cube file, or drop grading on a blank selection."""
        if event.select.id != "lut-select":
            return
        path = event.value
        if not path or path == self.lut_request:
            return
        self.lut_request = path
        self.state.update(lut_file=path)
        self.set_notice(f"loading {os.path.basename(path)}")
        self.bake_lut(path)
    @work(thread=True, exclusive=True, group="lut")
    def bake_lut(self, path: str) -> None:
        """Bake a LUT in the background so the video stream never stalls."""
        table = tracker.build_lut_table(path)
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
        self.query_one(f"#in-{suffix}", Input).value = format_value(suffix, value)
        self.set_notice(f"{label}: {format_value(suffix, value)}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        for prefix, direction in (("dec-", -1), ("inc-", 1)):
            if not button_id.startswith(prefix):
                continue
            suffix = button_id[len(prefix) :]
            if suffix not in TUNING:
                return
            step = TUNING[suffix][4]
            self.apply_tuning(suffix, self.get_tuning(suffix) + step * direction)
            return

    def on_input_submitted(self, event: Input.Submitted) -> None:
        suffix = (event.input.id or "").removeprefix("in-")
        if suffix not in TUNING:
            return
        try:
            value = float(event.value)
        except ValueError:
            self.apply_tuning(suffix, self.get_tuning(suffix))
            return
        self.apply_tuning(suffix, value)
