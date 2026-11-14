import unittest

import tracker


class TestCropOrigin(unittest.TestCase):
    def test_centered(self):
        x1, y1, cw, ch = tracker.crop_origin(0.5, 0.5, 2.0, 1000, 1000)
        self.assertEqual(cw, 500)
        self.assertEqual(ch, 500)
        self.assertEqual(x1, 250)
        self.assertEqual(y1, 250)

    def test_clamped_left_top(self):
        x1, y1, cw, ch = tracker.crop_origin(0.0, 0.0, 2.0, 1000, 1000)
        self.assertEqual(x1, 0)
        self.assertEqual(y1, 0)

    def test_clamped_right_bottom(self):
        x1, y1, cw, ch = tracker.crop_origin(1.0, 1.0, 2.0, 1000, 1000)
        self.assertEqual(x1, 500)
        self.assertEqual(y1, 500)


class TestFaceCenters(unittest.TestCase):
    class BB:
        def __init__(self, ox, oy, w, h):
            self.origin_x, self.origin_y, self.width, self.height = ox, oy, w, h

    class Det:
        def __init__(self, bb):
            self.bounding_box = bb

    def test_single(self):
        d = self.Det(self.BB(100, 100, 200, 200))
        centers = tracker.face_centers([d], 640, 360)
        self.assertEqual(len(centers), 1)
        self.assertAlmostEqual(centers[0][0], (100 + 100) / 640)
        self.assertAlmostEqual(centers[0][1], (100 + 200 * 0.6) / 360)


class TestLoadConfig(unittest.TestCase):
    def test_defaults_when_missing(self):
        cfg = tracker.load_config(None)
        self.assertEqual(cfg["camera_index"], 1)
        self.assertEqual(cfg["zoom"], 1.5)

    def test_ini_override(self):
        import tempfile, os
        fd, path = tempfile.mkstemp(suffix=".ini")
        with os.fdopen(fd, "w") as f:
            f.write("[camera]\nindex = 3\n[tracking]\nzoom = 2.0\n")
        try:
            cfg = tracker.load_config(path)
            self.assertEqual(cfg["camera_index"], 3)
            self.assertEqual(cfg["zoom"], 2.0)
        finally:
            os.remove(path)


class TestLoadLut(unittest.TestCase):
    def test_loads_cube(self):
        files = tracker.list_lut_files(None)
        if not files:
            self.skipTest("no .cube files in LUT/")
        lut = tracker.load_lut(files[0])
        self.assertIsNotNone(lut)
        self.assertEqual(lut.dtype, __import__("numpy").float32)

    def test_missing_returns_none(self):
        self.assertIsNone(tracker.load_lut("/no/such/file.cube"))


class TestSmartSharpen(unittest.TestCase):
    def test_flat_area_untouched(self):
        import numpy as np
        frame = np.full((32, 32, 3), 120, dtype=np.uint8)
        out = tracker.apply_smart_sharpen(frame, strength=0.3, threshold=10)
        self.assertTrue((out == frame).all())

    def test_edge_is_sharpened(self):
        import numpy as np
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        frame[:, 16:] = 200
        out = tracker.apply_smart_sharpen(frame, strength=0.3, threshold=10)
        self.assertFalse((out == frame).all())

    def test_zero_strength_is_noop(self):
        import numpy as np
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        frame[:, 16:] = 200
        out = tracker.apply_smart_sharpen(frame, strength=0.0)
        self.assertTrue((out == frame).all())


class TestLutCycling(unittest.TestCase):
    def test_cycle_wraps_forward(self):
        self.assertEqual(tracker.cycle_index(2, 3, 1), 0)
        self.assertEqual(tracker.cycle_index(0, 3, 1), 1)

    def test_cycle_wraps_backward(self):
        self.assertEqual(tracker.cycle_index(0, 3, -1), 2)
        self.assertEqual(tracker.cycle_index(1, 3, -1), 0)

    def test_empty_list_is_safe(self):
        self.assertEqual(tracker.cycle_index(0, 0, 1), 0)

    def test_lut_files_sorted(self):
        files = tracker.list_lut_files(None)
        self.assertEqual(files, sorted(files))


class TestSpringStep(unittest.TestCase):
    def test_critically_damped_no_overshoot_and_converges(self):
        pos, vel, target = 1.0, 0.0, 1.5
        omega, zeta, dt = 2.0 * 3.14159 / 0.9, 1.0, 1.0 / 30.0
        for _ in range(300):
            pos, vel = tracker.spring_step(pos, vel, target, omega, zeta, dt)
            self.assertLessEqual(pos, target + 1e-6)
        self.assertAlmostEqual(pos, target, places=3)


class TestSaveConfig(unittest.TestCase):
    def test_roundtrip_preserves_unknown_sections_and_writes_atomically(self):
        import configparser
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "center-stage.ini")
            with open(path, "w") as f:
                f.write("[camera]\nindex = 2\n\n[custom]\nkeep = yes\n")
            cfg = tracker.load_config(path)
            cfg["zoom"] = 2.25
            cfg["lut_enabled"] = False
            cfg["time_loop_enabled"] = True
            cfg["lut_file"] = os.path.join(folder, "LUT", "a.cube")
            tracker.save_config(path, cfg)

            reloaded = tracker.load_config(path)
            self.assertEqual(reloaded["camera_index"], 2)
            self.assertEqual(reloaded["zoom"], 2.25)
            self.assertFalse(reloaded["lut_enabled"])
            self.assertTrue(reloaded["time_loop_enabled"])
            self.assertEqual(reloaded["lut_file"], os.path.join("LUT", "a.cube"))

            parser = configparser.ConfigParser()
            parser.read(path)
            self.assertEqual(parser.get("custom", "keep"), "yes")
            self.assertEqual(
                [n for n in os.listdir(folder) if n.startswith(".center-stage-")], []
            )

    def test_effects_section_is_loaded(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "center-stage.ini")
            with open(path, "w") as f:
                f.write("[effects]\ntime_loop = true\ndeep_fry = true\n")
            cfg = tracker.load_config(path)
            self.assertTrue(cfg["time_loop_enabled"])
            self.assertTrue(cfg["deep_fry_enabled"])


class TestSharedState(unittest.TestCase):
    def test_update_marks_dirty_and_merges_into_config(self):
        cfg = dict(tracker.DEFAULTS)
        state = tracker.SharedState(cfg)
        self.assertIsNone(state.pending_revision())
        state.update(zoom=2.0)
        self.assertIsNotNone(state.pending_revision())
        self.assertEqual(state.snapshot()["zoom"], 2.0)
        self.assertEqual(state.config(cfg)["zoom"], 2.0)
        self.assertEqual(state.config(cfg)["fps"], cfg["fps"])

    def test_telemetry_does_not_dirty_the_config(self):
        state = tracker.SharedState(dict(tracker.DEFAULTS))
        state.set_telemetry(fps=29.9, tracking=True)
        self.assertIsNone(state.pending_revision())
        self.assertEqual(state.telemetry()["fps"], 29.9)
        self.assertTrue(state.telemetry()["tracking"])

    def test_edit_during_save_stays_pending(self):
        state = tracker.SharedState(dict(tracker.DEFAULTS))
        state.update(zoom=2.0)
        revision = state.pending_revision()
        state.update(zoom=3.0)
        state.mark_saved(revision)
        self.assertIsNotNone(state.pending_revision())
        state.mark_saved(state.pending_revision())
        self.assertIsNone(state.pending_revision())

    def test_unknown_setting_is_rejected(self):
        state = tracker.SharedState(dict(tracker.DEFAULTS))
        with self.assertRaises(KeyError):
            state.update(no_such_setting=1)


class TestTimeLoop(unittest.TestCase):
    def test_records_while_off_and_replays_while_on(self):
        loop = tracker.TimeLoop(length=3)
        self.assertFalse(loop.ready)
        frames = [object(), object(), object()]
        for frame in frames:
            self.assertIs(loop.record(frame), frame)
        self.assertTrue(loop.ready)

        self.assertIs(loop.next_frame(), frames[0])
        self.assertIs(loop.next_frame(), frames[1])
        self.assertIs(loop.next_frame(), frames[2])
        self.assertIs(loop.next_frame(), frames[0])

    def test_resuming_live_resets_the_playhead(self):
        loop = tracker.TimeLoop(length=3)
        first, second = object(), object()
        loop.record(first)
        loop.record(second)
        loop.next_frame()
        live = object()
        loop.record(live)
        self.assertIs(loop.next_frame(), first)


class TestDeepFry(unittest.TestCase):
    def test_changes_pixels_and_preserves_layout(self):
        import numpy as np

        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :32] = (40, 90, 200)
        frame[:, 32:] = (210, 120, 30)
        out = tracker.apply_deep_fry(frame)
        self.assertEqual(out.shape, frame.shape)
        self.assertEqual(out.dtype, np.uint8)
        self.assertFalse((out == frame).all())


class TestLutDiscovery(unittest.TestCase):
    def test_scans_working_directory_and_lut_folder(self):
        import os
        import tempfile

        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as folder:
            os.makedirs(os.path.join(folder, "LUT"))
            for relative in ("root.cube", os.path.join("LUT", "nested.cube")):
                open(os.path.join(folder, relative), "w").close()
            os.chdir(folder)
            try:
                files = tracker.list_lut_files(None)
            finally:
                os.chdir(old_cwd)
        self.assertEqual(
            [os.path.basename(path) for path in files],
            ["nested.cube", "root.cube"],
        )


class TestVideoPipeline(unittest.TestCase):
    def test_processes_frames_and_reports_telemetry(self):
        import time
        import types
        from unittest import mock

        import numpy as np

        sent = []

        class FakeCapture:
            def __init__(self, *args, **kwargs):
                pass

            def isOpened(self):
                return True

            def set(self, *args):
                return True

            def read(self):
                frame = np.zeros((120, 160, 3), dtype=np.uint8)
                frame[..., 1] = 128
                return True, frame

            def release(self):
                pass

        class FakeCamera:
            device = "Fake Virtual Camera"

            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def send(self, frame):
                sent.append(frame)

            def sleep_until_next_frame(self):
                pass

        class FakeDetector:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def detect(self, image):
                return types.SimpleNamespace(detections=[])

        cfg = dict(tracker.DEFAULTS)
        cfg.update(out_w=160, out_h=90, tracking_enabled=False, sharpen_enabled=True)
        state = tracker.SharedState(cfg)
        pipeline = tracker.VideoPipeline(cfg, state)
        with (
            mock.patch.object(tracker, "ensure_model", return_value="fake"),
            mock.patch.object(tracker.cv2, "VideoCapture", FakeCapture),
            mock.patch.object(tracker.pyvirtualcam, "Camera", FakeCamera),
            mock.patch.object(
                tracker.vision.FaceDetector,
                "create_from_options",
                return_value=FakeDetector(),
            ),
        ):
            pipeline.start()
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if sent and state.telemetry()["fps"] > 0:
                    break
                time.sleep(0.005)
            telemetry = state.telemetry()
            pipeline.stop()
            pipeline.join(2.0)

        self.assertFalse(pipeline.is_alive())
        self.assertGreaterEqual(len(sent), 1)
        self.assertEqual(sent[0].shape, (90, 160, 3))
        self.assertEqual(telemetry["device"], "Fake Virtual Camera")
        self.assertEqual(telemetry["status"], "running")
        self.assertGreater(telemetry["fps"], 0.0)
        self.assertFalse(telemetry["tracking"])


class TestMainCli(unittest.TestCase):
    def test_list_cameras_exits_before_the_tui(self):
        from unittest import mock

        with (
            mock.patch.object(tracker, "list_cameras", return_value=[]) as probe,
            mock.patch("builtins.print"),
        ):
            tracker.main(["--list-cameras"])
        probe.assert_called_once()

    def test_config_and_flag_overrides_reach_the_app(self):
        import os
        import tempfile
        from unittest import mock

        import tui

        seen = {}

        class FakeApp:
            def __init__(self, cfg, config_path, preview=False):
                seen["cfg"] = cfg
                seen["path"] = config_path
                seen["preview"] = preview

            def run(self):
                seen["ran"] = True

        path = os.path.join(tempfile.mkdtemp(), "custom.ini")
        with mock.patch.object(tui, "CenterStageApp", FakeApp):
            tracker.main(
                ["--config", path, "--no-tracking", "--time-loop", "--deep-fry"]
            )
        self.assertTrue(seen["ran"])
        self.assertEqual(seen["path"], path)
        self.assertFalse(seen["cfg"]["tracking_enabled"])
        self.assertTrue(seen["cfg"]["time_loop_enabled"])
        self.assertTrue(seen["cfg"]["deep_fry_enabled"])


class TestTui(unittest.IsolatedAsyncioTestCase):
    class FakePipeline:
        instances = []

        def __init__(self, cfg, state, preview=False):
            self.state = state
            self.started = False
            self.stopped = False
            TestTui.FakePipeline.instances.append(self)

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def join(self, timeout=None):
            pass

    async def test_shortcuts_tuning_and_autosave(self):
        import os
        import tempfile

        import tui

        original = tracker.VideoPipeline
        tracker.VideoPipeline = self.FakePipeline
        try:
            cfg = dict(tracker.DEFAULTS)
            cfg["lut_file"] = ""
            cfg["lut_enabled"] = False
            path = os.path.join(tempfile.mkdtemp(), "center-stage.ini")
            app = tui.CenterStageApp(cfg, path)
            async with app.run_test(size=(100, 45)) as pilot:
                await pilot.pause()
                self.assertTrue(self.FakePipeline.instances[-1].started)

                await pilot.press("t")
                await pilot.pause()
                self.assertFalse(app.state.snapshot()["tracking_enabled"])

                await pilot.press("m")
                await pilot.pause()
                self.assertTrue(app.state.snapshot()["time_loop_enabled"])

                zoom = app.query_one("#in-zoom", tui.Input)
                zoom.value = "2.5"
                zoom.focus()
                await pilot.press("enter")
                await pilot.pause()
                self.assertAlmostEqual(app.state.snapshot()["zoom"], 2.5)

                app.autosave()
                self.assertEqual(tracker.load_config(path)["zoom"], 2.5)
        finally:
            tracker.VideoPipeline = original

    async def test_dependent_controls_follow_their_switch(self):
        import os
        import tempfile

        import tui

        original = tracker.VideoPipeline
        tracker.VideoPipeline = self.FakePipeline
        try:
            cfg = dict(tracker.DEFAULTS)
            cfg["lut_file"] = ""
            cfg["lut_enabled"] = False
            cfg["sharpen_enabled"] = False
            app = tui.CenterStageApp(cfg, os.path.join(tempfile.mkdtemp(), "t.ini"))
            async with app.run_test(size=(100, 45)) as pilot:
                await pilot.pause()
                self.assertTrue(app.query_one("#lut-select", tui.Select).disabled)
                self.assertTrue(app.query_one("#in-lutscale", tui.Input).disabled)
                self.assertTrue(app.query_one("#in-sharpen", tui.Input).disabled)

                await pilot.press("l")
                await pilot.press("s")
                await pilot.pause()
                self.assertFalse(app.query_one("#lut-select", tui.Select).disabled)
                self.assertFalse(app.query_one("#in-lutscale", tui.Input).disabled)
                self.assertFalse(app.query_one("#in-sharpen", tui.Input).disabled)

                await pilot.press("s")
                await pilot.pause()
                self.assertTrue(app.query_one("#in-sharpen", tui.Input).disabled)
        finally:
            tracker.VideoPipeline = original

    async def test_lut_select_updates_state(self):
        import os
        import tempfile

        import tui

        original = tracker.VideoPipeline
        tracker.VideoPipeline = self.FakePipeline
        try:
            cfg = dict(tracker.DEFAULTS)
            cfg["lut_file"] = ""
            cfg["lut_enabled"] = True
            app = tui.CenterStageApp(cfg, os.path.join(tempfile.mkdtemp(), "t.ini"))
            async with app.run_test(size=(100, 45)) as pilot:
                await pilot.pause(0.1)
                select = app.query_one("#lut-select", tui.Select)
                self.assertTrue(app.lut_files)
                select.value = app.lut_files[0]
                await pilot.pause(0.5)
                self.assertEqual(app.state.snapshot()["lut_file"], app.lut_files[0])
                self.assertIsNotNone(app.state.lut_table)
        finally:
            tracker.VideoPipeline = original



if __name__ == "__main__":
    unittest.main()