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

from config import save_config
from lut import build_lut_table, cycle_index, list_lut_files
from pipeline import VideoPipeline
from pipmode import (
    CORNER_NAMES,
    is_image_file,
    list_pip_files,
    load_pip_image,
    pick_image_dialog,
)
from shaders import list_shader_files
from state import SharedState

# Widget id -> (label, SharedState key). Order defines the on-screen order.
# sw-lut and sw-sharpen are composed inside their own blocks and skipped in
# the FEATURES loop, but stay here so Changed messages update the state.
SWITCHES = {
    "sw-tracking": ("Face Tracking", "tracking_enabled"),
    "sw-cam-flip": ("Flip Camera", "cam_flip"),
    "sw-lut": ("LUT Grading", "lut_enabled"),
    "sw-shader": ("GLSL Shader", "shader_enabled"),
    "sw-sharpen": ("Smart Sharpen", "sharpen_enabled"),
    "sw-loop": ("Time Loop (1s)", "time_loop_enabled"),
    "sw-magic": ("Magic Loop (AFK)", "magic_loop_enabled"),
    "sw-fry": ("Deep Fry", "deep_fry_enabled"),
    "sw-pip": ("PiP Slide", "pip_enabled"),
    "sw-pip-flip": ("PiP Flip", "pip_flip"),
}

# Input suffix -> (label, SharedState key, minimum, maximum, button step).
# Integer bounds mean an integer setting (LUT scale), floats otherwise.
TUNING = {
    "zoom": ("Zoom", "zoom", 1.0, 4.0, 0.05),
    "smoothing": ("Pan Smoothing", "smoothing", 0.01, 0.30, 0.01),
    "sharpen": ("Strength", "sharpen_strength", 0.0, 1.0, 0.05),
    "lutscale": ("Scale", "lut_scale", 2, 16, 1),
}

AUTOSAVE_SECONDS = 5.0
TELEMETRY_SECONDS = 0.25
PIP_DISCOVERY_SECONDS = 2.0


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
        Binding("t", "toggle_tracking", "Tracking", priority=True),
        Binding("c", "toggle_lut", "LUT", key_display="[c]", priority=True),
        Binding("h", "toggle_shader", "Shader", key_display="<h>", priority=True),
        Binding("s", "toggle_sharpen", "Sharpen", priority=True),
        Binding("l", "toggle_time_loop", "Loop", priority=True),
        Binding("m", "toggle_magic_loop", "Magic", priority=True),
        Binding("f", "toggle_deep_fry", "Fry", priority=True),
        Binding("p", "toggle_pip", "PiP", priority=True),
        Binding("r", "reload_files", "Reload", priority=True),
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
            yield Static("starting", id="notice", classes="notice")
            yield Rule(classes="rule")
            yield Static("CONTROLS", classes="panel-title")
            yield Static("", classes="spacer")
            yield from self.switch_row(
                "Face Tracking", "sw-tracking", values, tuner="zoom"
            )
            # No switch on this row: spacers keep the tuner columns aligned.
            yield from self.switch_row(
                "Pan Smoothing", None, values, tuner="smoothing", tuner_name=""
            )
            yield from self.switch_row("Flip Camera", "sw-cam-flip", values)
            yield from self.switch_row(
                "LUT Grading", "sw-lut", values, tuner="lutscale"
            )
            yield Select(
                self.lut_options(),
                value=selected,
                prompt="LUT .cube",
                allow_blank=True,
                compact=True,
                type_to_search=False,
                id="lut-select",
            )
            shader_selected = values["shader_file"]
            if shader_selected not in self.shader_files:
                shader_selected = Select.NULL
            yield from self.switch_row("GLSL Shader", "sw-shader", values)
            yield Select(
                [(os.path.basename(p), p) for p in self.shader_files],
                value=shader_selected,
                prompt="shader .glsl",
                allow_blank=True,
                compact=True,
                type_to_search=False,
                id="shader-select",
            )
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
                yield Button("Pick", id="btn-pip-pick")
                yield Label("- + 1-0", classes="tuner-name hint-gap")
        yield Footer()

    def on_mount(self) -> None:
        self.pipeline.start()
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
        values = self.state.telemetry()
        tracking = "TRACKING" if values["tracking"] else "IDLE"
        info = values["device"] if values["status"] == "running" else values["status"]
        line = (
            f"FPS {values['fps']:.1f} | Zoom {values['zoom']:.2f}x | "
            f"{tracking} | {info}"
        )
        try:
            self.query_one("#telemetry-body", Static).update(line)
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
        spec = SWITCHES.get(event.switch.id or "")
        if spec is None:
            return
        label, key = spec
        if key == "magic_loop_enabled":
            self.sync_magic_tracking(event.switch.value)
        self.state.update(**{key: event.switch.value})
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
            switch.value = False
            switch.disabled = True
        else:
            previous = self._tracking_before_magic
            self._tracking_before_magic = None
            switch.disabled = False
            if previous is not None:
                # Setting the same value fires no Changed message.
                switch.value = previous
            self.state.update(tracking_enabled=bool(switch.value))

    def flip_switch(self, widget_id: str) -> None:
        """Toggle a switch; its Changed message updates the shared state."""
        switch = self.query_one(f"#{widget_id}", Switch)
        switch.value = not switch.value

    def action_toggle_tracking(self) -> None:
        if self.state.snapshot()["magic_loop_enabled"]:
            self.set_notice("tracking is frozen while Magic Loop is on")
            return
        self.flip_switch("sw-tracking")

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
        lut.value = (
            values["lut_file"] if values["lut_file"] in self.lut_files else Select.NULL
        )
        shader = self.query_one("#shader-select", Select)
        shader.set_options([(os.path.basename(p), p) for p in self.shader_files])
        shader.value = (
            values["shader_file"]
            if values["shader_file"] in self.shader_files
            else Select.NULL
        )
        pip = self.query_one("#pip-select", Select)
        pip.set_options(self.pip_options())
        pip.value = (
            values["pip_image"]
            if values["pip_image"] in self.pip_files
            else Select.NULL
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
        self.set_notice(f"PiP position: {CORNER_NAMES[corner]}")

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
        if path not in self.pip_files:
            # A picked file outside the app directory still becomes an option.
            self.pip_files.append(path)
            self.pip_files.sort()
            self.query_one("#pip-select", Select).set_options(self.pip_options())
        self.query_one("#pip-select", Select).value = path
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
        files = list_pip_files(self.state.snapshot()["pip_image"])
        if files == self.pip_files:
            return
        self.pip_files = files
        select = self.query_one("#pip-select", Select)
        select.set_options(self.pip_options())
        select.value = (
            self.state.snapshot()["pip_image"]
            if self.state.snapshot()["pip_image"] in files
            else Select.NULL
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
        self.query_one("#lut-select", Select).value = path
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
        self.query_one("#shader-select", Select).value = path
        self.set_notice(f"shader: {os.path.basename(path)}")

    def action_next_shader(self) -> None:
        self.cycle_shader(1)

    def action_prev_shader(self) -> None:
        self.cycle_shader(-1)

    def on_select_changed(self, event: Select.Changed) -> None:
        """Bake the chosen .cube file, or drop grading on a blank selection."""
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
        self.query_one(f"#in-{suffix}", Input).value = format_value(suffix, value)
        self.set_notice(f"{label}: {format_value(suffix, value)}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "btn-pip-pick":
            self.action_pick_pip_image()
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
