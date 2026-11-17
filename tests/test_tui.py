import asyncio
import functools
import os
import tempfile
from unittest import mock

import numpy as np
import pytest

from hyper_stage import tui
from hyper_stage.cli import main
from hyper_stage.config import DEFAULTS, find_config_path, load_config, save_config
from hyper_stage.state import SharedState


def run_async(func):
    """Run an async Textual test without requiring pytest-asyncio."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return wrapper


class TestTui:
    class FakePipeline:
        instances = []

        # Mirrors Thread.ident: None until the thread is started.
        ident = None

        def __init__(self, cfg, state, preview=False):
            self.state = state
            self.started = False
            self.stopped = False
            self.reloads = 0
            TestTui.FakePipeline.instances.append(self)

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def reload_shader(self):
            self.reloads += 1

        def join(self, timeout=None):
            pass

    @run_async
    async def test_shortcuts_tuning_and_autosave(self):
        import os
        import tempfile

        from hyper_stage import tui

        original = tui.VideoPipeline
        tui.VideoPipeline = self.FakePipeline
        try:
            cfg = dict(DEFAULTS)
            cfg["lut_file"] = ""
            cfg["lut_enabled"] = False
            path = os.path.join(tempfile.mkdtemp(), "hyper-stage.ini")
            app = tui.HyperStageApp(cfg, path)
            async with app.run_test(size=(100, 45)) as pilot:
                await pilot.pause()
                assert self.FakePipeline.instances[-1].started

                await pilot.press("t")
                await pilot.pause()
                assert not (app.state.snapshot()['tracking_enabled'])

                # The Flip Camera switch mirrors the camera feed through the
                # shared state (no hotkey by design).
                app.query_one("#sw-cam-flip", tui.Switch).value = True
                await pilot.pause()
                assert app.state.snapshot()['cam_flip']

                # i/sh focus the text field; Enter publishes the text string
                # and turns the overlay back on if it was switched off.
                app.query_one("#sw-text", tui.Switch).value = False
                await pilot.pause()
                assert not (app.state.snapshot()['text_enabled'])
                await pilot.press("i")
                await pilot.pause()
                assert app.query_one('#text-input', tui.Input).has_focus
                assert app.query_one('#text-input', tui.Input).size.width == 69
                for char in "hello":
                    await pilot.press(char)
                await pilot.press("enter")
                await pilot.pause()
                assert app.state.snapshot()['text_string'] == 'hello'
                assert app.state.snapshot()['text_enabled']
                assert app.query_one('#sw-text', tui.Switch).value
                assert not (app.state.snapshot()['tracking_enabled'])

                await pilot.press("c")
                await pilot.pause()
                assert app.state.snapshot()['lut_enabled']

                await pilot.press("l")
                await pilot.pause()
                assert app.state.snapshot()['time_loop_enabled']

                await pilot.press("m")
                await pilot.pause()
                assert app.state.snapshot()['magic_loop_enabled']
                assert app.query_one('#sw-magic', tui.Switch).value

                zoom = app.query_one("#in-zoom", tui.Input)
                zoom.value = "2.5"
                zoom.focus()
                await pilot.press("enter")
                await pilot.pause()
                assert app.state.snapshot()['zoom'] == pytest.approx(2.5)

                app.autosave()
                assert load_config(path)['zoom'] == 2.5
        finally:
            tui.VideoPipeline = original

    @run_async
    async def test_magic_loop_syncs_the_tracking_switch(self):
        import os
        import tempfile

        from hyper_stage import tui

        original = tui.VideoPipeline
        tui.VideoPipeline = self.FakePipeline
        try:
            cfg = dict(DEFAULTS)
            cfg["lut_file"] = ""
            app = tui.HyperStageApp(cfg, os.path.join(tempfile.mkdtemp(), "t.ini"))
            async with app.run_test(size=(100, 45)) as pilot:
                await pilot.pause()
                tracking = app.query_one("#sw-tracking", tui.Switch)
                assert tracking.value

                await pilot.press("m")
                await pilot.pause()
                # Magic on: tracking forced off, switch dimmed and locked.
                state = app.state.snapshot()
                assert state['magic_loop_enabled']
                assert not (state['tracking_enabled'])
                assert not (tracking.value)
                assert tracking.disabled

                await pilot.press("m")
                await pilot.pause()
                # Magic off: the previous tracking state comes back.
                state = app.state.snapshot()
                assert not (state['magic_loop_enabled'])
                assert state['tracking_enabled']
                assert tracking.value
                assert not (tracking.disabled)
        finally:
            tui.VideoPipeline = original

    @run_async
    async def test_magic_loop_at_startup_forces_tracking_off(self):
        import os
        import tempfile

        from hyper_stage import tui

        original = tui.VideoPipeline
        tui.VideoPipeline = self.FakePipeline
        try:
            cfg = dict(DEFAULTS)
            cfg["lut_file"] = ""
            cfg["magic_loop_enabled"] = True
            app = tui.HyperStageApp(cfg, os.path.join(tempfile.mkdtemp(), "t.ini"))
            async with app.run_test(size=(100, 45)) as pilot:
                await pilot.pause()
                tracking = app.query_one("#sw-tracking", tui.Switch)
                assert not (app.state.snapshot()['tracking_enabled'])
                assert not (tracking.value)
                assert tracking.disabled

                await pilot.press("m")
                await pilot.pause()
                # Turning Magic Loop off restores the configured behavior.
                assert app.state.snapshot()['tracking_enabled']
                assert tracking.value
                assert not (tracking.disabled)
        finally:
            tui.VideoPipeline = original

    @run_async
    async def test_shader_controls_follow_their_switch_and_select(self):
        import os
        import tempfile

        from hyper_stage import tui

        original = tui.VideoPipeline
        tui.VideoPipeline = self.FakePipeline
        try:
            cfg = dict(DEFAULTS)
            cfg["lut_file"] = ""
            cfg["shader_enabled"] = False
            app = tui.HyperStageApp(cfg, os.path.join(tempfile.mkdtemp(), "t.ini"))
            async with app.run_test(size=(100, 45)) as pilot:
                await pilot.pause()
                select = app.query_one("#shader-select", tui.Select)
                # The dropdown stays usable while the shader is off...
                assert not (select.disabled)
                assert app.shader_files

                # The h shortcut flips the switch (the dropdown was never
                # locked).
                await pilot.press("h")
                await pilot.pause()
                assert app.state.snapshot()['shader_enabled']
                assert app.query_one('#sw-shader', tui.Switch).value

                # Picking a file updates the shared state.
                select.value = app.shader_files[0]
                await pilot.pause()
                assert app.state.snapshot()['shader_file'] == app.shader_files[0]

                # The blank option is a NoSelection sentinel, not a string;
                # choosing it must not crash the app.
                select.value = tui.Select.NULL
                await pilot.pause()
                assert app.state.snapshot()['shader_file'] == app.shader_files[0]
                # Same for the LUT dropdown's blank option.
                lut = app.query_one("#lut-select", tui.Select)
                lut.value = tui.Select.NULL
                await pilot.pause()
        finally:
            tui.VideoPipeline = original

    @run_async
    async def test_reload_rescans_files_and_pipeline(self):
        import os
        import tempfile

        from hyper_stage import tui

        original = tui.VideoPipeline
        tui.VideoPipeline = self.FakePipeline
        try:
            cfg = dict(DEFAULTS)
            cfg["lut_file"] = ""
            app = tui.HyperStageApp(cfg, os.path.join(tempfile.mkdtemp(), "t.ini"))
            async with app.run_test(size=(100, 45)) as pilot:
                await pilot.pause()
                pipeline = app.pipeline
                # Pretend the folder scan happened before new files appeared.
                app.shader_files = []
                app.query_one("#shader-select", tui.Select).set_options([])

                await pilot.press("r")
                await pilot.pause()
                # The rescan picked the shaders/ directory back up and the
                # pipeline was told to drop its compiled shader cache.
                assert app.shader_files
                assert len(app.query_one('#shader-select', tui.Select)._options) > 0
                assert pipeline.reloads >= 1
                # The cycle keys remain bound but are hidden from the footer.
                shown = {
                    b.action: getattr(b, "show", True) for b in app.BINDINGS
                }
                assert not (shown['next_lut'])
                assert not (shown['prev_lut'])
        finally:
            tui.VideoPipeline = original

    @run_async
    async def test_dependent_controls_follow_their_switch(self):
        import os
        import tempfile

        from hyper_stage import tui

        original = tui.VideoPipeline
        tui.VideoPipeline = self.FakePipeline
        try:
            cfg = dict(DEFAULTS)
            cfg["lut_file"] = ""
            cfg["lut_enabled"] = False
            cfg["sharpen_enabled"] = False
            cfg["pip_enabled"] = False
            app = tui.HyperStageApp(cfg, os.path.join(tempfile.mkdtemp(), "t.ini"))
            async with app.run_test(size=(100, 45)) as pilot:
                await pilot.pause()
                # Dropdowns and Pick stay usable while their feature is off:
                # an effect or image must be selectable before it is shown.
                assert not (app.query_one('#lut-select', tui.Select).disabled)
                assert not (app.query_one('#shader-select', tui.Select).disabled)
                assert not (app.query_one('#pip-select', tui.Select).disabled)
                assert not (app.query_one('#btn-pip-pick', tui.Button).disabled)
                # Tuning rows stay bound to their feature switch.
                assert app.query_one('#in-lutscale', tui.Input).disabled
                assert app.query_one('#in-sharpen', tui.Input).disabled

                await pilot.press("c")
                await pilot.press("s")
                await pilot.pause()
                assert not (app.query_one('#in-lutscale', tui.Input).disabled)
                assert not (app.query_one('#in-sharpen', tui.Input).disabled)

                await pilot.press("s")
                await pilot.pause()
                assert app.query_one('#in-sharpen', tui.Input).disabled
        finally:
            tui.VideoPipeline = original

    @run_async
    async def test_lut_select_updates_state(self):
        import os
        import tempfile

        from hyper_stage import tui

        original = tui.VideoPipeline
        tui.VideoPipeline = self.FakePipeline
        try:
            cfg = dict(DEFAULTS)
            cfg["lut_file"] = ""
            cfg["lut_enabled"] = True
            app = tui.HyperStageApp(cfg, os.path.join(tempfile.mkdtemp(), "t.ini"))
            async with app.run_test(size=(100, 45)) as pilot:
                await pilot.pause(0.1)
                select = app.query_one("#lut-select", tui.Select)
                assert app.lut_files
                select.value = app.lut_files[0]
                await pilot.pause(0.5)
                assert app.state.snapshot()['lut_file'] == app.lut_files[0]
                assert app.state.lut_table is not None
        finally:
            tui.VideoPipeline = original

    @run_async
    async def test_russian_layout_equivalents_cover_all_hotkeys(self):
        """Every Latin hotkey fires through its ЙЦУКЕН equivalent key."""
        import os
        import tempfile

        from hyper_stage import tui

        original = tui.VideoPipeline
        tui.VideoPipeline = self.FakePipeline
        try:
            cfg = dict(DEFAULTS)
            app = tui.HyperStageApp(cfg, os.path.join(tempfile.mkdtemp(), "t.ini"))
            async with app.run_test(size=(100, 45)) as pilot:
                await pilot.pause(0.1)
                cases = [
                    ("е", "sw-tracking", False),     # t: tracking off
                    ("ы", "sw-sharpen", True),       # s: sharpen on
                    ("а", "sw-fry", True),           # f: deep fry on
                ]
                for key, switch_id, expected in cases:
                    await pilot.press(key)
                    await pilot.pause(0.3)
                    assert app.query_one(f'#{switch_id}', tui.Switch).value is expected
                # Every Latin letter binding has a Russian counterpart
                # registered; multi-character navigation keys (arrows) and
                # layout-invariant keys (digits, -/+) are the same on both
                # layouts and need none.
                latin = {
                    b.action
                    for b in tui.HyperStageApp.BINDINGS
                    if len(b.key) == 1 and b.key.isascii() and not b.key.isdigit()
                }
                russian = {
                    b.action
                    for b in tui.HyperStageApp.BINDINGS
                    if len(b.key) == 1 and not b.key.isascii()
                }
                assert latin == russian
                # q equivalent actually shuts the app down (tested last).
                await pilot.press("й")
                await pilot.pause(0.3)
                assert not (app.is_running)
        finally:
            tui.VideoPipeline = original

    @run_async
    async def test_bracket_keys_cycle_luts(self):
        import os
        import tempfile

        from hyper_stage import tui

        original = tui.VideoPipeline
        tui.VideoPipeline = self.FakePipeline
        try:
            cfg = dict(DEFAULTS)
            cfg["lut_file"] = ""
            cfg["lut_enabled"] = True
            app = tui.HyperStageApp(cfg, os.path.join(tempfile.mkdtemp(), "t.ini"))
            async with app.run_test(size=(100, 45)) as pilot:
                await pilot.pause(0.1)
                files = app.lut_files
                assert files

                await pilot.press("]")
                await pilot.pause(0.3)
                assert app.state.snapshot()['lut_file'] == files[0]

                await pilot.press("]")
                await pilot.pause(0.3)
                assert app.state.snapshot()['lut_file'] == files[1]

                await pilot.press("[")
                await pilot.pause(0.3)
                assert app.state.snapshot()['lut_file'] == files[0]

                # Russian layout equivalents: the key where ] sits on a Russian
                # keyboard produces ъ (next) and [ produces х (previous).
                await pilot.press("ъ")
                await pilot.pause(0.3)
                assert app.state.snapshot()['lut_file'] == files[1]
                await pilot.press("х")
                await pilot.pause(0.3)
                assert app.state.snapshot()['lut_file'] == files[0]

                # Going backward from the first file wraps to the last.
                await pilot.press("[")
                await pilot.pause(0.3)
                assert app.state.snapshot()['lut_file'] == files[-1]
        finally:
            tui.VideoPipeline = original

    @run_async
    async def test_angle_bracket_keys_cycle_shaders(self):
        import os
        import tempfile

        from hyper_stage import tui

        original = tui.VideoPipeline
        tui.VideoPipeline = self.FakePipeline
        try:
            cfg = dict(DEFAULTS)
            cfg["shader_file"] = ""
            app = tui.HyperStageApp(cfg, os.path.join(tempfile.mkdtemp(), "t.ini"))
            async with app.run_test(size=(100, 45)) as pilot:
                await pilot.pause(0.1)
                files = app.shader_files
                assert files

                await pilot.press(">")
                await pilot.pause(0.3)
                assert app.state.snapshot()['shader_file'] == files[0]

                await pilot.press(">")
                await pilot.pause(0.3)
                assert app.state.snapshot()['shader_file'] == files[1]

                await pilot.press("<")
                await pilot.pause(0.3)
                assert app.state.snapshot()['shader_file'] == files[0]

                # Russian layout equivalents: the key where > sits on a Russian
                # keyboard produces ю (next) and < produces б (previous).
                await pilot.press("ю")
                await pilot.pause(0.3)
                assert app.state.snapshot()['shader_file'] == files[1]
                await pilot.press("б")
                await pilot.pause(0.3)
                assert app.state.snapshot()['shader_file'] == files[0]

                # Going backward from the first file wraps to the last.
                await pilot.press("<")
                await pilot.pause(0.3)
                assert app.state.snapshot()['shader_file'] == files[-1]
        finally:
            tui.VideoPipeline = original

    def test_shader_binding_shows_in_footer_as_h(self):
        from hyper_stage import tui

        display = {
            action: binding.key_display
            for binding in tui.HyperStageApp.BINDINGS
            if binding.key_display
            for action in (binding.action,)
        }
        assert display.get('toggle_shader') == '<h>'
        assert display.get('toggle_lut') == '[c]'

    @run_async
    async def test_theme_is_applied_and_persisted(self):
        import os
        import tempfile

        from hyper_stage import tui

        original = tui.VideoPipeline
        tui.VideoPipeline = self.FakePipeline
        try:
            with tempfile.TemporaryDirectory() as folder:
                ini = os.path.join(folder, "t.ini")
                cfg = dict(DEFAULTS)
                cfg["theme"] = "textual-light"
                app = tui.HyperStageApp(cfg, ini)
                async with app.run_test(size=(80, 24)) as pilot:
                    await pilot.pause(0.1)
                    # The saved theme is applied at startup...
                    assert app.theme == 'textual-light'
                    # ...and a theme change (command palette) is saved.
                    app.theme = "textual-dark"
                    await pilot.pause(0.1)
                assert load_config(ini)['theme'] == 'textual-dark'
        finally:
            tui.VideoPipeline = original

    @run_async
    async def test_pip_shortcuts_corner_snap_and_discovery(self):
        import os
        import tempfile

        from hyper_stage import tui

        original = tui.VideoPipeline
        tui.VideoPipeline = self.FakePipeline
        try:
            with tempfile.TemporaryDirectory() as folder:
                cfg = dict(DEFAULTS)
                cfg["pip_enabled"] = True
                image = os.path.join(folder, "slide.png")
                import cv2

                cv2.imwrite(image, np.zeros((40, 60, 3), dtype=np.uint8))
                app = tui.HyperStageApp(cfg, os.path.join(folder, "t.ini"))
                async with app.run_test(size=(100, 45)) as pilot:
                    await pilot.pause()
                    # A file dropped into the app directory is discovered...
                    from unittest import mock

                    with mock.patch("os.getcwd", return_value=folder):
                        app.refresh_pip_files()
                    await pilot.pause()
                    assert image in app.pip_files
                    assert image in [option[1] for option in app.pip_options()]

                    # ...and selecting it decodes a fitted background frame.
                    app.apply_pip_image(image)
                    await pilot.pause(0.3)
                    assert app.state.snapshot()['pip_image'] == image
                    assert app.state.pip_frame is not None

                    # Arrows move one axis at a time: down flips the vertical
                    # side (TL -> BL), left then flips the horizontal one
                    # (BL -> BR; swapped to match the mirrored self-preview).
                    await pilot.press("down")
                    await pilot.pause()
                    assert app.state.snapshot()['pip_corner'] == 2
                    await pilot.press("left")
                    await pilot.pause()
                    assert app.state.snapshot()['pip_corner'] == 3

                    # With PiP off the arrows are refused...
                    await pilot.press("p")
                    await pilot.pause()
                    assert not (app.state.snapshot()['pip_enabled'])
                    await pilot.press("right")
                    await pilot.pause()
                    assert app.state.snapshot()['pip_corner'] == 3

                    # ...and p/з toggles back on.
                    await pilot.press("з")
                    await pilot.pause()
                    assert app.state.snapshot()['pip_enabled']

                    # Arrows move the corner even with the dropdown focused:
                    # the bindings are priority, so no widget can swallow them.
                    app.query_one("#pip-select", tui.Select).focus()
                    await pilot.pause()
                    await pilot.press("up")
                    await pilot.pause()
                    # Up flips only the vertical side: BR -> TR.
                    assert app.state.snapshot()['pip_corner'] == 1

                    # The flip switch mirrors the background.
                    app.query_one("#sw-pip-flip", tui.Switch).value = True
                    await pilot.pause()
                    assert app.state.snapshot()['pip_flip']

                    # Dropdown labels show the parent folder, not just the
                    # file name.
                    second = os.path.join(folder, "meme.png")
                    cv2.imwrite(second, np.zeros((40, 60, 3), dtype=np.uint8))
                    with mock.patch("os.getcwd", return_value=folder):
                        app.refresh_pip_files()
                    await pilot.pause()
                    folder_label = f"…/{os.path.basename(folder)}/"
                    labels = [option[0] for option in app.pip_options()]
                    assert folder_label + 'meme.png' in labels
                    assert folder_label + 'slide.png' in labels
                    assert [option[1] for option in app.pip_options()].count(second) == 1

                    # +/- step through the current folder's images and stop
                    # at its ends instead of running into other folders...
                    await pilot.press("plus")
                    await pilot.pause()
                    assert app.state.snapshot()['pip_image'] == image
                    await pilot.press("minus")
                    await pilot.pause()
                    assert app.state.snapshot()['pip_image'] == second
                    await pilot.press("minus")
                    await pilot.pause()
                    # ...already at the folder's first file: minus stays.
                    assert app.state.snapshot()['pip_image'] == second

                    # A file in another folder does not leak into the cycle:
                    # with it selected, +/- has nothing to move to.
                    sub = os.path.join(folder, "sub")
                    os.mkdir(sub)
                    other = os.path.join(sub, "other.png")
                    cv2.imwrite(other, np.zeros((40, 60, 3), dtype=np.uint8))
                    app.apply_pip_image(other)
                    await pilot.pause()
                    await pilot.press("plus")
                    await pilot.pause()
                    assert app.state.snapshot()['pip_image'] == other
                    await pilot.press("minus")
                    await pilot.pause()
                    assert app.state.snapshot()['pip_image'] == other

                    # ...and digits jump inside the current folder (2 ->
                    # second entry of its folder). Blur the Select first so
                    # a stray option-list overlay cannot swallow the digit.
                    await pilot.press("escape")
                    await pilot.pause()
                    app.query_one("#sw-pip-flip", tui.Switch).focus()
                    app.apply_pip_image(second)
                    await pilot.pause(0.2)
                    await pilot.press("2")
                    await pilot.pause(0.2)
                    assert app.state.snapshot()['pip_image'] == image
        finally:
            tui.VideoPipeline = original


class TestMainCli:
    def test_list_cameras_exits_before_the_tui(self):
        from unittest import mock

        with (
            mock.patch("hyper_stage.cli.list_cameras", return_value=[]) as probe,
            mock.patch("builtins.print"),
        ):
            main(["--list-cameras"])
        probe.assert_called_once()

    def test_config_and_flag_overrides_reach_the_app(self):
        import os
        import tempfile
        from unittest import mock

        from hyper_stage import tui

        seen = {}

        class FakeApp:
            def __init__(self, cfg, config_path, preview=False):
                seen["cfg"] = cfg
                seen["path"] = config_path
                seen["preview"] = preview

            def run(self):
                seen["ran"] = True

        path = os.path.join(tempfile.mkdtemp(), "custom.ini")
        with mock.patch.object(tui, "HyperStageApp", FakeApp):
            main(
                [
                    "--config",
                    path,
                    "--no-tracking",
                    "--time-loop",
                    "--deep-fry",
                    "--magic-loop",
                    "--shader",
                    "/tmp/fx.glsl",
                    "--shader-enabled",
                ]
            )
        assert seen['ran']
        assert seen['path'] == path
        assert not (seen['cfg']['tracking_enabled'])
        assert seen['cfg']['time_loop_enabled']
        assert seen['cfg']['deep_fry_enabled']
        assert seen['cfg']['magic_loop_enabled']
        assert seen['cfg']['shader_file'] == '/tmp/fx.glsl'
        assert seen['cfg']['shader_enabled']

