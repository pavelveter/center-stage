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
                f.write(
                    "[effects]\ntime_loop = true\ndeep_fry = true\n"
                    "magic_loop = true\n"
                )
            cfg = tracker.load_config(path)
            self.assertTrue(cfg["time_loop_enabled"])
            self.assertTrue(cfg["deep_fry_enabled"])
            self.assertTrue(cfg["magic_loop_enabled"])


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


class TestMagicLoop(unittest.TestCase):
    def test_build_loop_returns_none_without_enough_history(self):
        import numpy as np

        buf = tracker.MagicLoopBuffer()
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        buf.record(frame)
        self.assertIsNone(buf.build_loop())

    def test_seam_search_picks_the_calmest_window(self):
        import cv2
        import numpy as np

        buf = tracker.MagicLoopBuffer()
        still = np.full((180, 320, 3), 40, dtype=np.uint8)
        for i in range(tracker.MAGIC_LOOP_MIN_FRAMES + 30):
            if i < 60:
                frame = still.copy()
                # Busy motion early on: a bar jumps around every frame.
                frame[:, : ((i * 11) % 6) * 50 + 10] = 200
            else:
                frame = still  # dead still afterwards
            buf.record(frame)
        player = buf.build_loop()
        self.assertIsNotNone(player)
        # The loop is a full minimum-length window taken entirely from the
        # still zone, not from the moving part of the history.
        self.assertEqual(len(player._frames), tracker.MAGIC_LOOP_MIN_LOOP_FRAMES)
        first = cv2.imdecode(
            np.frombuffer(player._frames[0], np.uint8), cv2.IMREAD_COLOR
        )
        self.assertLess(int(first.max()), 100)

    def test_loop_rejects_a_window_shorter_than_two_seconds(self):
        import numpy as np

        buf = tracker.MagicLoopBuffer()
        still = np.full((180, 320, 3), 40, dtype=np.uint8)
        # Just under MAGIC_LOOP_MIN_FRAMES: a full calm window cannot exist.
        for _ in range(tracker.MAGIC_LOOP_MIN_FRAMES - 1):
            buf.record(still)
        self.assertIsNone(buf.build_loop())

    def test_player_crossfades_and_wraps(self):
        import cv2
        import numpy as np

        n = 10
        encoded = []
        for i in range(n):
            frame = np.full((180, 320, 3), i * 20 + 10, dtype=np.uint8)
            ok, jpeg = cv2.imencode(".jpg", frame)
            self.assertTrue(ok)
            encoded.append(jpeg.tobytes())
        player = tracker.MagicLoopPlayer(encoded)
        fade = n - 2  # clamped from MAGIC_LOOP_FADE by the short loop
        self.assertEqual(player._fade, fade)
        out_size = (320, 180)
        for _ in range(n):
            player.next_frame(out_size)
        # After one pass the playhead resumes past the crossfaded head.
        self.assertEqual(player._index, fade + 1)
        # The wrap is reported so the pipeline can pull a fresh loop.
        self.assertTrue(player.pass_done)
        player.pass_done = False
        # The final faded frame is exactly the head frame it blends into.
        tail = player.next_frame(out_size)  # i == n - 1, alpha == 1
        head = cv2.resize(
            cv2.imdecode(
                np.frombuffer(encoded[fade], np.uint8), cv2.IMREAD_COLOR
            ),
            out_size,
            interpolation=cv2.INTER_LINEAR,
        )
        self.assertTrue(np.allclose(tail, head, atol=1))
        self.assertEqual(tail.shape, (180, 320, 3))

    def test_short_loop_without_fade_wraps_to_start(self):
        import cv2
        import numpy as np

        encoded = []
        for i in range(2):
            frame = np.full((60, 80, 3), 100, dtype=np.uint8)
            ok, jpeg = cv2.imencode(".jpg", frame)
            self.assertTrue(ok)
            encoded.append(jpeg.tobytes())
        player = tracker.MagicLoopPlayer(encoded)
        self.assertEqual(player._fade, 0)
        player.next_frame((80, 60))
        player.next_frame((80, 60))
        self.assertEqual(player._index, 0)
        self.assertTrue(player.pass_done)


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

    def test_magic_loop_replays_a_frozen_seamless_loop(self):
        import time
        import types
        from unittest import mock

        import numpy as np

        sent = []
        counter = {"value": 0}

        class FakeCapture:
            def __init__(self, *args, **kwargs):
                pass

            def isOpened(self):
                return True

            def set(self, *args):
                return True

            def read(self):
                counter["value"] += 1
                value = counter["value"]
                frame = np.full((120, 160, 3), 90, dtype=np.uint8)
                # A moving column gives the first era distinct structure that
                # survives JPEG; afterwards the scene is perfectly still, so
                # a fresh loop pulled from the newest history has no column.
                if value <= 120:
                    frame[:, value % 160] = 255
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
        # Zoom 1.0 keeps the whole frame so the moving column stays visible
        # and every live frame remains distinct.
        cfg.update(
            out_w=160,
            out_h=90,
            magic_loop_enabled=True,
            tracking_enabled=False,
            zoom=1.0,
        )
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
            # Enough frames for the history to fill, the first loop (built
            # from the moving era) to play, and a fresh loop to be pulled
            # from the still era that follows.
            deadline = time.monotonic() + 12.0
            while time.monotonic() < deadline and len(sent) < 600:
                time.sleep(0.01)
            pipeline.stop()
            pipeline.join(2.0)

        self.assertFalse(pipeline.is_alive())
        self.assertGreater(len(sent), 400)
        bright = [int(frame.max()) for frame in sent]
        # Mid-stream: the live scene went still at frame 120, yet published
        # frames still show the bright column - recorded moving-era content
        # is being replayed, not the live feed.
        self.assertTrue(any(value >= 200 for value in bright[140:320]))
        # End: a regenerated loop from the newest (still) history plays - a
        # statically locked slice would keep showing the moving column.
        tail = {frame.tobytes() for frame in sent[-120:]}
        self.assertEqual(len(tail), 1)
        self.assertLess(bright[-1], 200)


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
                [
                    "--config",
                    path,
                    "--no-tracking",
                    "--time-loop",
                    "--deep-fry",
                    "--magic-loop",
                ]
            )
        self.assertTrue(seen["ran"])
        self.assertEqual(seen["path"], path)
        self.assertFalse(seen["cfg"]["tracking_enabled"])
        self.assertTrue(seen["cfg"]["time_loop_enabled"])
        self.assertTrue(seen["cfg"]["deep_fry_enabled"])
        self.assertTrue(seen["cfg"]["magic_loop_enabled"])


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

                await pilot.press("g")
                await pilot.pause()
                self.assertTrue(app.state.snapshot()["magic_loop_enabled"])
                self.assertTrue(app.query_one("#sw-magic", tui.Switch).value)

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

    async def test_magic_loop_syncs_the_tracking_switch(self):
        import os
        import tempfile

        import tui

        original = tracker.VideoPipeline
        tracker.VideoPipeline = self.FakePipeline
        try:
            cfg = dict(tracker.DEFAULTS)
            cfg["lut_file"] = ""
            app = tui.CenterStageApp(cfg, os.path.join(tempfile.mkdtemp(), "t.ini"))
            async with app.run_test(size=(100, 45)) as pilot:
                await pilot.pause()
                tracking = app.query_one("#sw-tracking", tui.Switch)
                self.assertTrue(tracking.value)

                await pilot.press("g")
                await pilot.pause()
                # Magic on: tracking forced off, switch dimmed and locked.
                state = app.state.snapshot()
                self.assertTrue(state["magic_loop_enabled"])
                self.assertFalse(state["tracking_enabled"])
                self.assertFalse(tracking.value)
                self.assertTrue(tracking.disabled)

                await pilot.press("g")
                await pilot.pause()
                # Magic off: the previous tracking state comes back.
                state = app.state.snapshot()
                self.assertFalse(state["magic_loop_enabled"])
                self.assertTrue(state["tracking_enabled"])
                self.assertTrue(tracking.value)
                self.assertFalse(tracking.disabled)
        finally:
            tracker.VideoPipeline = original

    async def test_magic_loop_at_startup_forces_tracking_off(self):
        import os
        import tempfile

        import tui

        original = tracker.VideoPipeline
        tracker.VideoPipeline = self.FakePipeline
        try:
            cfg = dict(tracker.DEFAULTS)
            cfg["lut_file"] = ""
            cfg["magic_loop_enabled"] = True
            app = tui.CenterStageApp(cfg, os.path.join(tempfile.mkdtemp(), "t.ini"))
            async with app.run_test(size=(100, 45)) as pilot:
                await pilot.pause()
                tracking = app.query_one("#sw-tracking", tui.Switch)
                self.assertFalse(app.state.snapshot()["tracking_enabled"])
                self.assertFalse(tracking.value)
                self.assertTrue(tracking.disabled)

                await pilot.press("g")
                await pilot.pause()
                # Turning Magic Loop off restores the configured behavior.
                self.assertTrue(app.state.snapshot()["tracking_enabled"])
                self.assertTrue(tracking.value)
                self.assertFalse(tracking.disabled)
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

    async def test_bracket_keys_cycle_luts(self):
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
                files = app.lut_files
                self.assertTrue(files)

                await pilot.press("]")
                await pilot.pause(0.3)
                self.assertEqual(app.state.snapshot()["lut_file"], files[0])

                await pilot.press("]")
                await pilot.pause(0.3)
                self.assertEqual(app.state.snapshot()["lut_file"], files[1])

                await pilot.press("[")
                await pilot.pause(0.3)
                self.assertEqual(app.state.snapshot()["lut_file"], files[0])

                # Russian layout equivalents: the key where ] sits on a Russian
                # keyboard produces ъ (next) and [ produces х (previous).
                await pilot.press("ъ")
                await pilot.pause(0.3)
                self.assertEqual(app.state.snapshot()["lut_file"], files[1])
                await pilot.press("х")
                await pilot.pause(0.3)
                self.assertEqual(app.state.snapshot()["lut_file"], files[0])

                # Going backward from the first file wraps to the last.
                await pilot.press("[")
                await pilot.pause(0.3)
                self.assertEqual(app.state.snapshot()["lut_file"], files[-1])
        finally:
            tracker.VideoPipeline = original



if __name__ == "__main__":
    unittest.main()