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
                    "[color]\nlut_enabled = true\nshader_enabled = true\n"
                    "shader_file = sh.glsl\n"
                    "[effects]\ntime_loop = true\ndeep_fry = true\n"
                    "magic_loop = true\n"
                )
            cfg = tracker.load_config(path)
            self.assertTrue(cfg["time_loop_enabled"])
            self.assertTrue(cfg["deep_fry_enabled"])
            self.assertTrue(cfg["magic_loop_enabled"])
            self.assertTrue(cfg["shader_enabled"])
            self.assertEqual(cfg["shader_file"], "sh.glsl")


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
    CROP = (0.5, 0.45, 1.5)

    def test_build_loop_returns_none_without_enough_history(self):
        import numpy as np

        buf = tracker.MagicLoopBuffer()
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        buf.record(frame, self.CROP)
        self.assertIsNone(buf.build_loop(self.CROP))

    def test_seam_search_picks_the_calmest_window(self):
        import cv2
        import numpy as np

        buf = tracker.MagicLoopBuffer()
        still = np.full((180, 320, 3), 40, dtype=np.uint8)
        # Busy first, then a still stretch long enough to hold a minimum
        # loop (2 s) plus its crossfade tail (span = loop + blend frames).
        for i in range(tracker.MAGIC_LOOP_MIN_FRAMES + 120):
            if i < 30:
                frame = still.copy()
                # Busy motion early on: a bar jumps around every frame.
                frame[:, : ((i * 11) % 6) * 50 + 10] = 200
            else:
                frame = still  # dead still afterwards
            buf.record(frame, self.CROP)
        player = buf.build_loop(self.CROP)
        self.assertIsNotNone(player)
        # The clip holds the loop plus its crossfade tail, and the loop
        # length is inside the 2-5 s window bounds.
        self.assertEqual(
            len(player._frames),
            player._length + player._blend,
        )
        self.assertGreaterEqual(player._length, tracker.MAGIC_LOOP_MIN_LOOP_FRAMES)
        first = cv2.imdecode(
            np.frombuffer(player._frames[0], np.uint8), cv2.IMREAD_COLOR
        )
        # The winner starts in the still zone, not in the busy part.
        self.assertLess(int(first.max()), 100)

    def test_loop_rejects_a_window_shorter_than_two_seconds(self):
        import numpy as np

        buf = tracker.MagicLoopBuffer()
        still = np.full((180, 320, 3), 40, dtype=np.uint8)
        # Just under MAGIC_LOOP_MIN_FRAMES: a full calm window cannot exist.
        for _ in range(tracker.MAGIC_LOOP_MIN_FRAMES - 1):
            buf.record(still, self.CROP)
        self.assertIsNone(buf.build_loop(self.CROP))

    def test_player_plays_in_order_and_wraps_with_crossfade(self):
        import cv2
        import numpy as np

        length, blend = 6, 2
        encoded = []
        for i in range(length + blend):
            frame = np.full((180, 320, 3), i * 20 + 10, dtype=np.uint8)
            ok, jpeg = cv2.imencode(".jpg", frame)
            self.assertTrue(ok)
            encoded.append(jpeg.tobytes())
        player = tracker.MagicLoopPlayer(encoded, length, blend)
        out_size = (320, 180)
        frames = [player.next_frame(out_size) for _ in range(length + blend)]

        def decode(i):
            return cv2.resize(
                cv2.imdecode(
                    np.frombuffer(encoded[i], np.uint8), cv2.IMREAD_COLOR
                ),
                out_size,
                interpolation=cv2.INTER_LINEAR,
            )

        # Plain frames play clip[t] in recording order after the blend head.
        for t in range(blend, length):
            self.assertTrue(np.array_equal(frames[t], decode(t)))
        # The first blended frame already moved away from clip[0] towards
        # the footage that followed the loop, so no motion reversal.
        self.assertFalse(np.array_equal(frames[0], decode(0)))


class TestShaderFiles(unittest.TestCase):
    def test_scans_working_directory_and_shaders_folder(self):
        import os
        import tempfile

        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as folder:
            os.makedirs(os.path.join(folder, "shaders"))
            for relative in (
                "root.glsl",
                os.path.join("shaders", "nested.glsl"),
                os.path.join("shaders", "ignore.txt"),
            ):
                open(os.path.join(folder, relative), "w").close()
            os.chdir(folder)
            try:
                files = tracker.list_shader_files(None)
            finally:
                os.chdir(old_cwd)
        self.assertEqual(
            [os.path.basename(path) for path in files],
            ["root.glsl", "nested.glsl"],
        )

    def test_load_shader_reads_source_and_rejects_bad_files(self):
        import os
        import tempfile

        self.assertIsNone(tracker.load_shader(""))
        self.assertIsNone(tracker.load_shader("/no/such/file.glsl"))
        with tempfile.TemporaryDirectory() as folder:
            plain = os.path.join(folder, "plain.glsl")
            open(plain, "w").write("vec3 noop(vec3 c) { return c; }")
            self.assertIsNone(tracker.load_shader(plain))
            good = os.path.join(folder, "good.glsl")
            source = "void mainImage(out vec4 c, in vec2 p) { c = vec4(1.0); }"
            open(good, "w").write(source)
            self.assertEqual(tracker.load_shader(good), source)


class TestShaderEngine(unittest.TestCase):
    PASSTHROUGH = """
    void mainImage(out vec4 fragColor, in vec2 fragCoord) {
        fragColor = vec4(texture(iChannel0, fragCoord.xy / iResolution.xy).rgb, 1.0);
    }
    """

    def _write(self, folder, name, source):
        import os

        path = os.path.join(folder, name)
        with open(path, "w") as f:
            f.write(source)
        return path

    def test_passthrough_preserves_the_frame(self):
        import os
        import tempfile

        import numpy as np

        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        frame[..., 2] = 200  # blue in BGR
        try:
            engine = tracker.ShaderEngine()
            with tempfile.TemporaryDirectory() as folder:
                path = self._write(folder, "pass.glsl", self.PASSTHROUGH)
                out = engine.apply(frame, path)
                # A broken file disables the effect, frame passes through.
                broken = self._write(folder, "broken.glsl", "vec3 noop;")
                self.assertTrue(
                    (engine.apply(frame, broken) == frame).all()
                )
        except Exception:
            self.skipTest("no usable GPU context in this environment")
        finally:
            if engine._ctx is not None:
                engine._ctx.release()
        self.assertEqual(out.shape, frame.shape)
        self.assertEqual(out.dtype, np.uint8)
        # Passthrough must not reorder or flip anything.
        self.assertTrue((out == frame).all())

    def test_inverting_shader_changes_pixels(self):
        import os
        import tempfile

        import numpy as np

        source = """
        void mainImage(out vec4 fragColor, in vec2 fragCoord) {
            vec3 c = texture(iChannel0, fragCoord.xy / iResolution.xy).rgb;
            fragColor = vec4(vec3(1.0) - c, 1.0);
        }
        """
        engine = tracker.ShaderEngine()
        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        frame[10:20, 20:30] = (30, 60, 90)
        try:
            with tempfile.TemporaryDirectory() as folder:
                path = self._write(folder, "invert.glsl", source)
                out = engine.apply(frame, path)
        except Exception:
            self.skipTest("no usable GPU context in this environment")
        finally:
            if engine._ctx is not None:
                engine._ctx.release()
        self.assertFalse((out == frame).all())
        # The marked region is inverted (255 - value, small tolerance).
        self.assertTrue(abs(int(out[15, 25, 0]) - (255 - 30)) <= 2)

    def test_shader_using_itime_compiles_and_runs(self):
        """iTime must be provided, or Shadertoy noise shaders never compile."""
        import numpy as np

        source = """
        void mainImage(out vec4 fragColor, in vec2 fragCoord) {
            float n = fract(sin(dot(fragCoord.xy, vec2(17.0, 180.0))) + iTime);
            fragColor = vec4(mix(
                texture(iChannel0, fragCoord.xy / iResolution.xy).rgb,
                vec3(n), 0.5), 1.0);
        }
        """
        engine = tracker.ShaderEngine()
        frame = np.full((60, 80, 3), 100, dtype=np.uint8)
        try:
            import tempfile

            with tempfile.TemporaryDirectory() as folder:
                path = self._write(folder, "itime.glsl", source)
                out = engine.apply(frame, path)
        except Exception:
            self.skipTest("no usable GPU context in this environment")
        finally:
            if engine._ctx is not None:
                engine._ctx.release()
        # Not blacklisted as broken and not passed through unchanged.
        self.assertNotIn(path, engine._broken)
        self.assertFalse((out == frame).all())

    def test_imouse_is_declared_for_shadertoy_imports(self):
        """Shaders reading iMouse must compile; default mouse is zero."""
        import os
        import tempfile

        import numpy as np

        source = """
        void mainImage(out vec4 fragColor, in vec2 fragCoord) {
            vec2 m = iMouse.xy / iResolution.xy;
            vec3 c = texture(iChannel0, fragCoord.xy / iResolution.xy).rgb;
            fragColor = vec4(mix(c, vec3(1.0), m.x), 1.0);
        }
        """
        engine = tracker.ShaderEngine()
        frame = np.full((40, 50, 3), 90, dtype=np.uint8)
        try:
            with tempfile.TemporaryDirectory() as folder:
                path = self._write(folder, "mouse.glsl", source)
                out = engine.apply(frame, path)
        except Exception:
            self.skipTest("no usable GPU context in this environment")
        finally:
            if engine._ctx is not None:
                engine._ctx.release()
        self.assertNotIn(path, engine._broken)
        # With the mouse at (0,0) the frame passes through untouched.
        self.assertTrue((out == frame).all())

    def test_auxiliary_noise_texture_drives_channel_one(self):
        """iChannel1 must hold live noise, not an unbound (garbage) unit."""
        import numpy as np

        source = """
        void mainImage(out vec4 fragColor, in vec2 fragCoord) {
            float n = texture(iChannel1, fragCoord.xy / 32.0).r;
            fragColor = vec4(vec3(n), 1.0);
        }
        """
        engine = tracker.ShaderEngine()
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        try:
            import tempfile

            with tempfile.TemporaryDirectory() as folder:
                path = self._write(folder, "noise.glsl", source)
                out = engine.apply(frame, path)
        except Exception:
            self.skipTest("no usable GPU context in this environment")
        finally:
            if engine._ctx is not None:
                engine._ctx.release()
        # Non-constant noise pattern with real variance.
        self.assertGreater(int(out.std()), 10)
        self.assertGreater(int(out.max()), 40)

    def test_font_texture_is_a_glyph_atlas(self):
        import numpy as np

        engine = tracker.ShaderEngine()
        try:
            engine._ensure_size(8, 8)
            font = engine._aux_texture("font")
            # Read while the context is alive; after release() it is gone.
            data = np.frombuffer(font.read(alignment=1), np.uint8).reshape(
                256, 256, 4
            )
        except Exception:
            self.skipTest("no usable GPU context in this environment")
        finally:
            if engine._ctx is not None:
                engine._ctx.release()
        self.assertIsNotNone(font)
        # Glyph cells are addressed as code = row_cell*16 + col_cell:
        # '0' is 48 (row 3, col 0) and 'A' is 65 (row 4, col 1); codes
        # below 32 stay blank.
        self.assertGreater(int(data[48:64, 0:16, 0].max()), 100)
        self.assertGreater(int(data[64:80, 16:32, 0].max()), 100)
        self.assertEqual(int(data[0:16, 0:16, 0].max()), 0)

    def test_channel_directive_is_parsed(self):
        source = (
            "// center-stage: 0=font 1=camera\n"
            "// center-stage: 2=noise\n"
            "void mainImage(out vec4 c, in vec2 p) {}\n"
        )
        self.assertEqual(
            tracker.ShaderEngine.parse_channels(source),
            {0: "font", 1: "camera", 2: "noise"},
        )
        self.assertEqual(
            tracker.ShaderEngine.parse_channels("void mainImage() {}"), {}
        )

    def test_dither_shader_matches_cpu_bayer_reference(self):
        """A Bayer-dither shader must decide every pixel identically to a
        CPU reference - no filtering blur or sampling offsets allowed."""
        import cv2
        import numpy as np

        source = """
        const int bayer[64] = int[64](
             0, 32,  8, 40,  2, 34, 10, 42,
            48, 16, 56, 24, 50, 18, 58, 26,
            12, 44,  4, 36, 14, 46,  6, 38,
            60, 28, 52, 20, 62, 30, 54, 22,
             3, 35, 11, 43,  1, 33,  9, 41,
            51, 19, 59, 27, 49, 17, 57, 25,
            15, 47,  7, 39, 13, 45,  5, 37,
            63, 31, 55, 23, 61, 29, 53, 21
        );
        void mainImage(out vec4 fragColor, in vec2 fragCoord) {
            vec2 uv = floor(fragCoord) / iResolution.xy;
            vec3 t = texture(iChannel0, uv).rgb;
            float luma = dot(t, vec3(0.299, 0.587, 0.114));
            luma = clamp((luma - 0.5) * 0.98 + 0.5 - 0.09, 0.0, 1.0);
            int x = int(mod(floor(fragCoord).x, 8.0));
            int y = int(mod(floor(fragCoord).y, 8.0));
            float d = step(float(bayer[x + y * 8]) / 64.0, luma);
            fragColor = vec4(mix(vec3(0.0), vec3(1.0), d), 1.0);
        }
        """
        engine = tracker.ShaderEngine()
        rng = np.random.default_rng(3)
        frame = rng.integers(0, 255, (256, 512, 3), dtype=np.uint8)
        try:
            import tempfile

            with tempfile.TemporaryDirectory() as folder:
                path = self._write(folder, "bayer.glsl", source)
                out = engine.apply(frame, path)
        except Exception:
            self.skipTest("no usable GPU context in this environment")
        finally:
            if engine._ctx is not None:
                engine._ctx.release()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        luma = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
        luma = np.clip((luma - 0.5) * 0.98 + 0.5 - 0.09, 0, 1)
        bayer = np.array(
            [
                [0, 32, 8, 40, 2, 34, 10, 42],
                [48, 16, 56, 24, 50, 18, 58, 26],
                [12, 44, 4, 36, 14, 46, 6, 38],
                [60, 28, 52, 20, 62, 30, 54, 22],
                [3, 35, 11, 43, 1, 33, 9, 41],
                [51, 19, 59, 27, 49, 17, 57, 25],
                [15, 47, 7, 39, 13, 45, 5, 37],
                [63, 31, 55, 23, 61, 29, 53, 21],
            ]
        )
        h, w = luma.shape
        # Shadertoy coordinates are y-up, so the tile anchors bottom-left.
        limit = np.tile(bayer[::-1], (h // 8 + 1, w // 8 + 1))[:h, :w] / 64.0
        expect = limit <= luma
        got = out[..., 0] > 128  # white in BGR
        self.assertGreaterEqual((expect == got).mean(), 0.999)

    def test_screen_space_shaders_get_shadertoy_y_up_orientation(self):
        """fragCoord.y > H/2 must land in the published TOP half.

        Shadertoy displays fragCoord.y = 0 at the bottom; without the
        boundary flips the raymarched scene in meta-crt.glsl appeared
        upside down. Camera sampling must stay pixel-exact meanwhile.
        """
        import os
        import tempfile

        import numpy as np

        screen = """
        void mainImage(out vec4 c, in vec2 p) {
            c = (p.y > iResolution.y * 0.5) ? vec4(1.0, 0.0, 0.0, 1.0)
                                            : vec4(0.0, 0.0, 1.0, 1.0);
        }
        """
        camera = """
        void mainImage(out vec4 c, in vec2 p) {
            c = vec4(texture(iChannel0, p / iResolution.xy).rgb, 1.0);
        }
        """
        engine = tracker.ShaderEngine()
        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        frame[5, 40] = (7, 8, 9)  # asymmetric marker near the top
        try:
            with tempfile.TemporaryDirectory() as folder:
                screen_path = self._write(folder, "screen.glsl", screen)
                out = engine.apply(frame, screen_path)
                camera_path = self._write(folder, "camera.glsl", camera)
                cam_out = engine.apply(frame, camera_path)
        except Exception:
            self.skipTest("no usable GPU context in this environment")
        finally:
            if engine._ctx is not None:
                engine._ctx.release()
        # Screen space: red (y > H/2) on the published top half. The shader
        # writes RGB; the engine's verbatim-import round trip publishes it
        # as BGR, so red (1,0,0) arrives as (0,0,255).
        self.assertEqual(int(out[10, 40, 0]), 0)
        self.assertEqual(int(out[10, 40, 2]), 255)
        self.assertEqual(int(out[50, 40, 0]), 255)
        self.assertEqual(int(out[50, 40, 2]), 0)
        # Camera sampling: still pixel-exact, no flip.
        self.assertTrue((cam_out == frame).all())

    def test_hardcoded_rgb_colors_publish_with_intended_hue(self):
        """A shader writing RGB palette constants must arrive hue-correct.

        1bit.glsl's orange #F27A2D published as BGR-preserving output
        appeared blue; single-file shaders are verbatim Shadertoy imports
        now, so RGB constants convert on readback.
        """
        import os
        import tempfile

        import numpy as np

        source = """
        void mainImage(out vec4 fragColor, in vec2 fragCoord) {
            fragColor = vec4(0.949, 0.478, 0.176, 1.0); // #F27A2D orange
        }
        """
        engine = tracker.ShaderEngine()
        frame = np.full((40, 50, 3), 90, dtype=np.uint8)
        try:
            with tempfile.TemporaryDirectory() as folder:
                path = self._write(folder, "palette.glsl", source)
                out = engine.apply(frame, path)
        except Exception:
            self.skipTest("no usable GPU context in this environment")
        finally:
            if engine._ctx is not None:
                engine._ctx.release()
        # Published BGR must read as the intended orange hue (values are
        # the palette floats through the 8-bit round trip).
        self.assertEqual(int(out[20, 25, 0]), 45)
        self.assertEqual(int(out[20, 25, 1]), 122)
        self.assertEqual(int(out[20, 25, 2]), 242)

    def test_multibuffer_pack_runs_passes_with_feedback(self):
        """A <name>.glsl folder runs Common + buffers + Image as one pack."""
        import os
        import tempfile

        import numpy as np

        pack_dir = None
        engine = tracker.ShaderEngine()
        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        frame[..., 0] = 200  # BGR blue
        try:
            with tempfile.TemporaryDirectory() as folder:
                os.makedirs(os.path.join(folder, "pack.glsl"))
                old_cwd = os.getcwd()
                os.chdir(folder)
                try:
                    # Discovered alongside single files.
                    found = tracker.list_shader_files(None)
                    self.assertIn(
                        os.path.realpath(os.path.join(folder, "pack.glsl")),
                        [os.path.realpath(p) for p in found],
                    )
                finally:
                    os.chdir(old_cwd)
                pack_dir = os.path.join(folder, "pack.glsl")
                passes = {
                    "Common.glsl": "vec3 chain(vec3 c) { return c; }\n",
                    "Buffer A.glsl": (
                        "void mainImage(out vec4 c, in vec2 p) {\n"
                        "    vec2 uv = p / iResolution.xy;\n"
                        "    vec4 cam = texture(iChannel0, uv);\n"
                        "    c = min(texture(iChannel1, uv) + 0.25, cam);\n"
                        "}\n"
                    ),
                    "Buffer B.glsl": (
                        "void mainImage(out vec4 c, in vec2 p) {\n"
                        "    vec2 uv = p / iResolution.xy;\n"
                        "    c = vec4(chain(texture(iChannel0, uv).rgb), 1.0);\n"
                        "}\n"
                    ),
                    "Image.glsl": (
                        "void mainImage(out vec4 c, in vec2 p) {\n"
                        "    vec2 uv = p / iResolution.xy;\n"
                        "    c = vec4(vec3(1.0)\n"
                        "        - texture(iChannel0, uv).rgb, 1.0);\n"
                        "}\n"
                    ),
                }
                for name, source in passes.items():
                    with open(os.path.join(pack_dir, name), "w") as f:
                        f.write(source)
                out1 = engine.apply(frame, pack_dir)
                out2 = engine.apply(frame, pack_dir)
        except Exception:
            self.skipTest("no usable GPU context in this environment")
        finally:
            if engine._ctx is not None and pack_dir in engine._programs:
                engine._programs[pack_dir].close()
            if engine._ctx is not None:
                engine._ctx.release()
        self.assertNotIn(pack_dir, engine._broken)
        # Pass order, feedback, and the RGB round trip: frame 1 Buffer A
        # accumulates 0.25 toward the camera's blue (0.784 in RGB), frame 2
        # reaches 0.5; Image inverts and the engine converts back to BGR.
        self.assertAlmostEqual(int(out1[5, 5, 0]), 255 - 64, delta=3)
        self.assertAlmostEqual(int(out2[5, 5, 0]), 255 - 128, delta=3)
        self.assertEqual(int(out1[5, 5, 1]), 255)
        self.assertEqual(int(out1[5, 5, 2]), 255)

    def test_multibuffer_pack_with_bad_passes_disables_cleanly(self):
        import os
        import tempfile

        import numpy as np

        frame = np.zeros((30, 40, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            pack_dir = os.path.join(folder, "pack.glsl")
            os.makedirs(pack_dir)
            with open(os.path.join(pack_dir, "Buffer A.glsl"), "w") as f:
                f.write("void mainImage(out vec4 c, in vec2 p) { c = vec4(0); }")
            # No Image.glsl: unusable, frame passes through.
            engine = tracker.ShaderEngine()
            try:
                out = engine.apply(frame, pack_dir)
            except Exception:
                self.skipTest("no usable GPU context in this environment")
            finally:
                engine._ctx.release()
            self.assertTrue((out == frame).all())
            self.assertIn(pack_dir, engine._broken)

    def test_pack_channel_directive_parsing(self):
        """Pack pass directives map units to aux names and buffer passes."""
        parse = tracker.MultiShader.parse_channels
        names = ["Buffer A.glsl", "Buffer B.glsl"]
        self.assertEqual(
            parse("// center-stage: 1=noise 2=Buffer B", names),
            {1: "noise", 2: ("buffer", 1)},
        )
        # Spaces optional, case ignored.
        self.assertEqual(
            parse("// center-stage: 0=feedback 1=bufferb", names),
            {0: "feedback", 1: ("buffer", 1)},
        )
        # Unknown names are ignored, leaving the default wiring.
        self.assertEqual(parse("// center-stage: 3=webcam", names), {})

    def test_multibuffer_pack_channel_directive(self):
        """A pass directive wires iChannelN to a named buffer's output."""
        import os
        import tempfile

        import numpy as np

        pack_dir = None
        engine = tracker.ShaderEngine()
        frame = np.zeros((30, 40, 3), dtype=np.uint8)
        try:
            with tempfile.TemporaryDirectory() as folder:
                pack_dir = os.path.join(folder, "pack.glsl")
                os.makedirs(pack_dir)
                passes = {
                    "Buffer A.glsl": (
                        "void mainImage(out vec4 c, in vec2 p) {\n"
                        "    c = vec4(1.0, 0.0, 0.0, 1.0);\n"
                        "}\n"
                    ),
                    "Buffer B.glsl": (
                        "void mainImage(out vec4 c, in vec2 p) {\n"
                        "    c = vec4(0.0, 1.0, 0.0, 1.0);\n"
                        "}\n"
                    ),
                    # The default Image wiring reads the last buffer (green)
                    # on iChannel0; the directive makes iChannel2 carry
                    # Buffer A (red) instead.
                    "Image.glsl": (
                        "// center-stage: 2=Buffer A\n"
                        "void mainImage(out vec4 c, in vec2 p) {\n"
                        "    c = texture(iChannel2, p / iResolution.xy);\n"
                        "}\n"
                    ),
                }
                for name, source in passes.items():
                    with open(os.path.join(pack_dir, name), "w") as f:
                        f.write(source)
                out = engine.apply(frame, pack_dir)
        except Exception:
            self.skipTest("no usable GPU context in this environment")
        finally:
            if engine._ctx is not None and pack_dir in engine._programs:
                engine._programs[pack_dir].close()
            if engine._ctx is not None:
                engine._ctx.release()
        self.assertNotIn(pack_dir, engine._broken)
        # Verbatim RGB import, so red publishes as BGR (R in the last slot).
        self.assertEqual(int(out[5, 5, 0]), 0)
        self.assertEqual(int(out[5, 5, 1]), 0)
        self.assertEqual(int(out[5, 5, 2]), 255)

    def test_multibuffer_pack_with_broken_image_disables_cleanly(self):
        import os
        import tempfile

        import numpy as np

        frame = np.zeros((30, 40, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            pack_dir = os.path.join(folder, "pack.glsl")
            os.makedirs(pack_dir)
            with open(os.path.join(pack_dir, "Buffer A.glsl"), "w") as f:
                f.write("void mainImage(out vec4 c, in vec2 p) { c = vec4(0); }")
            with open(os.path.join(pack_dir, "Image.glsl"), "w") as f:
                f.write("void mainImage(out vec4 c {")  # syntax error
            engine = tracker.ShaderEngine()
            try:
                out = engine.apply(frame, pack_dir)
            except Exception:
                self.skipTest("no usable GPU context in this environment")
            finally:
                if engine._programs.get(pack_dir):
                    engine._programs[pack_dir].close()
                engine._ctx.release()
            self.assertTrue((out == frame).all())
            self.assertIn(pack_dir, engine._broken)

    def test_ichannel_resolution_reports_texture_size(self):
        """Shadertoy's iChannelResolution uniform carries the texture size."""
        import os
        import tempfile

        import numpy as np

        path = None
        engine = tracker.ShaderEngine()
        frame = np.zeros((30, 40, 3), dtype=np.uint8)
        try:
            with tempfile.TemporaryDirectory() as folder:
                path = os.path.join(folder, "res.glsl")
                with open(path, "w") as f:
                    f.write(
                        "void mainImage(out vec4 c, in vec2 p) {\n"
                        "    vec2 res = iChannelResolution[0].xy;\n"
                        "    c = vec4(res / vec2(40.0, 30.0), 0.0, 1.0);\n"
                        "}\n"
                    )
                out = engine.apply(frame, path)
        except Exception:
            self.skipTest("no usable GPU context in this environment")
        finally:
            if engine._ctx is not None:
                engine._ctx.release()
        self.assertNotIn(path, engine._broken)
        # res / (40, 30) == (1, 1) -> RGB (1, 1, 0), published as BGR
        # (0, 255, 255) by the verbatim-import round trip.
        self.assertEqual(int(out[5, 5, 0]), 0)
        self.assertEqual(int(out[5, 5, 1]), 255)
        self.assertEqual(int(out[5, 5, 2]), 255)

    def test_pack_pass_texel_fetch_lod_reads_mipmaps(self):
        """Pack buffers carry mipmaps so explicit-LOD fetches work."""
        import os
        import tempfile

        import numpy as np

        pack_dir = None
        engine = tracker.ShaderEngine()
        frame = np.full((32, 32, 3), 200, dtype=np.uint8)
        try:
            with tempfile.TemporaryDirectory() as folder:
                pack_dir = os.path.join(folder, "pack.glsl")
                os.makedirs(pack_dir)
                passes = {
                    "Buffer A.glsl": (
                        "void mainImage(out vec4 c, in vec2 p) {\n"
                        "    c = texture(iChannel0, p / iResolution.xy);\n"
                        "}\n"
                    ),
                    # LOD-1 fetch of Buffer A: black without mipmaps, the
                    # 2x2-averaged gray with them.
                    "Buffer B.glsl": (
                        "void mainImage(out vec4 c, in vec2 p) {\n"
                        "    c = texelFetch(iChannel0, ivec2(0), 1);\n"
                        "}\n"
                    ),
                    "Image.glsl": (
                        "void mainImage(out vec4 c, in vec2 p) {\n"
                        "    c = texelFetch(iChannel0, ivec2(p), 0);\n"
                        "}\n"
                    ),
                }
                for name, source in passes.items():
                    with open(os.path.join(pack_dir, name), "w") as f:
                        f.write(source)
                out = engine.apply(frame, pack_dir)
        except Exception:
            self.skipTest("no usable GPU context in this environment")
        finally:
            if engine._ctx is not None and engine._programs.get(pack_dir):
                engine._programs[pack_dir].close()
            if engine._ctx is not None:
                engine._ctx.release()
        self.assertNotIn(pack_dir, engine._broken)
        # 200/255 through mip levels stays ~0.784; published BGR ~200 again.
        self.assertAlmostEqual(int(out[5, 5, 0]), 200, delta=4)

    def test_single_file_ichannel2_reads_noise(self):
        """iChannel2+ compile and bind noise instead of unbound garbage."""
        import os
        import tempfile

        import numpy as np

        path = None
        engine = tracker.ShaderEngine()
        frame = np.zeros((30, 40, 3), dtype=np.uint8)
        try:
            with tempfile.TemporaryDirectory() as folder:
                path = os.path.join(folder, "ch2.glsl")
                with open(path, "w") as f:
                    f.write(
                        "void mainImage(out vec4 c, in vec2 p) {\n"
                        "    c = vec4(texture(iChannel2,\n"
                        "        p / iResolution.xy).rgb, 1.0);\n"
                        "}\n"
                    )
                out = engine.apply(frame, path)
        except Exception:
            self.skipTest("no usable GPU context in this environment")
        finally:
            if engine._ctx is not None:
                engine._ctx.release()
        self.assertNotIn(path, engine._broken)
        # The noise texture's pixels, not driver garbage: varied values.
        self.assertGreater(len(np.unique(out[..., 0])), 4)


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
            # Enough frames for the settling period, the loop build, the
            # entry dissolve, and several full passes of the frozen loop.
            deadline = time.monotonic() + 12.0
            while time.monotonic() < deadline and len(sent) < 700:
                time.sleep(0.01)
            pipeline.stop()
            pipeline.join(2.0)

        self.assertFalse(pipeline.is_alive())
        self.assertGreater(len(sent), 500)
        # Settling grace: the first frames after activation are still the
        # live feed - distinct, moving-column frames, not a loop.
        early = {frame.tobytes() for frame in sent[:40]}
        self.assertGreater(len(early), 35)
        self.assertTrue(any(int(frame.max()) >= 200 for frame in sent[:40]))
        # Then the frozen loop repeats exactly: find its period in the tail
        # and verify every cycle is identical (clean restart, no crossfades).
        tail = sent[-400:]
        period = next(
            k
            for k in range(55, 200)
            if np.array_equal(tail[k], tail[0])
        )
        for k in range(period, len(tail)):
            self.assertTrue(np.array_equal(tail[k], tail[k - period]))


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
                    "--shader",
                    "/tmp/fx.glsl",
                    "--shader-enabled",
                ]
            )
        self.assertTrue(seen["ran"])
        self.assertEqual(seen["path"], path)
        self.assertFalse(seen["cfg"]["tracking_enabled"])
        self.assertTrue(seen["cfg"]["time_loop_enabled"])
        self.assertTrue(seen["cfg"]["deep_fry_enabled"])
        self.assertTrue(seen["cfg"]["magic_loop_enabled"])
        self.assertEqual(seen["cfg"]["shader_file"], "/tmp/fx.glsl")
        self.assertTrue(seen["cfg"]["shader_enabled"])


class TestTui(unittest.IsolatedAsyncioTestCase):
    class FakePipeline:
        instances = []

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

                await pilot.press("c")
                await pilot.pause()
                self.assertTrue(app.state.snapshot()["lut_enabled"])

                await pilot.press("l")
                await pilot.pause()
                self.assertTrue(app.state.snapshot()["time_loop_enabled"])

                await pilot.press("m")
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

                await pilot.press("m")
                await pilot.pause()
                # Magic on: tracking forced off, switch dimmed and locked.
                state = app.state.snapshot()
                self.assertTrue(state["magic_loop_enabled"])
                self.assertFalse(state["tracking_enabled"])
                self.assertFalse(tracking.value)
                self.assertTrue(tracking.disabled)

                await pilot.press("m")
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

                await pilot.press("m")
                await pilot.pause()
                # Turning Magic Loop off restores the configured behavior.
                self.assertTrue(app.state.snapshot()["tracking_enabled"])
                self.assertTrue(tracking.value)
                self.assertFalse(tracking.disabled)
        finally:
            tracker.VideoPipeline = original

    async def test_shader_controls_follow_their_switch_and_select(self):
        import os
        import tempfile

        import tui

        original = tracker.VideoPipeline
        tracker.VideoPipeline = self.FakePipeline
        try:
            cfg = dict(tracker.DEFAULTS)
            cfg["lut_file"] = ""
            cfg["shader_enabled"] = False
            app = tui.CenterStageApp(cfg, os.path.join(tempfile.mkdtemp(), "t.ini"))
            async with app.run_test(size=(100, 45)) as pilot:
                await pilot.pause()
                select = app.query_one("#shader-select", tui.Select)
                self.assertTrue(select.disabled)
                self.assertTrue(app.shader_files)

                # The h shortcut flips the switch and enables the dropdown.
                await pilot.press("h")
                await pilot.pause()
                self.assertTrue(app.state.snapshot()["shader_enabled"])
                self.assertTrue(app.query_one("#sw-shader", tui.Switch).value)
                self.assertFalse(select.disabled)

                # Picking a file updates the shared state.
                select.value = app.shader_files[0]
                await pilot.pause()
                self.assertEqual(
                    app.state.snapshot()["shader_file"], app.shader_files[0]
                )

                # The blank option is a NoSelection sentinel, not a string;
                # choosing it must not crash the app.
                select.value = tui.Select.NULL
                await pilot.pause()
                self.assertEqual(
                    app.state.snapshot()["shader_file"], app.shader_files[0]
                )
                # Same for the LUT dropdown's blank option.
                lut = app.query_one("#lut-select", tui.Select)
                lut.value = tui.Select.NULL
                await pilot.pause()
        finally:
            tracker.VideoPipeline = original

    async def test_reload_rescans_files_and_pipeline(self):
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
                pipeline = app.pipeline
                # Pretend the folder scan happened before new files appeared.
                app.shader_files = []
                app.query_one("#shader-select", tui.Select).set_options([])

                await pilot.press("r")
                await pilot.pause()
                # The rescan picked the shaders/ directory back up and the
                # pipeline was told to drop its compiled shader cache.
                self.assertTrue(app.shader_files)
                self.assertGreater(
                    len(app.query_one("#shader-select", tui.Select)._options), 0
                )
                self.assertGreaterEqual(pipeline.reloads, 1)
                # The cycle keys remain bound but are hidden from the footer.
                shown = {
                    b.action: getattr(b, "show", True) for b in app.BINDINGS
                }
                self.assertFalse(shown["next_lut"])
                self.assertFalse(shown["prev_lut"])
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

                await pilot.press("c")
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

    async def test_angle_bracket_keys_cycle_shaders(self):
        import os
        import tempfile

        import tui

        original = tracker.VideoPipeline
        tracker.VideoPipeline = self.FakePipeline
        try:
            cfg = dict(tracker.DEFAULTS)
            cfg["shader_file"] = ""
            app = tui.CenterStageApp(cfg, os.path.join(tempfile.mkdtemp(), "t.ini"))
            async with app.run_test(size=(100, 45)) as pilot:
                await pilot.pause(0.1)
                files = app.shader_files
                self.assertTrue(files)

                await pilot.press(">")
                await pilot.pause(0.3)
                self.assertEqual(app.state.snapshot()["shader_file"], files[0])

                await pilot.press(">")
                await pilot.pause(0.3)
                self.assertEqual(app.state.snapshot()["shader_file"], files[1])

                await pilot.press("<")
                await pilot.pause(0.3)
                self.assertEqual(app.state.snapshot()["shader_file"], files[0])

                # Russian layout equivalents: the key where > sits on a Russian
                # keyboard produces ю (next) and < produces б (previous).
                await pilot.press("ю")
                await pilot.pause(0.3)
                self.assertEqual(app.state.snapshot()["shader_file"], files[1])
                await pilot.press("б")
                await pilot.pause(0.3)
                self.assertEqual(app.state.snapshot()["shader_file"], files[0])

                # Going backward from the first file wraps to the last.
                await pilot.press("<")
                await pilot.pause(0.3)
                self.assertEqual(app.state.snapshot()["shader_file"], files[-1])
        finally:
            tracker.VideoPipeline = original

    def test_shader_binding_shows_in_footer_as_h(self):
        import tui

        display = {
            action: binding.key_display
            for binding in tui.CenterStageApp.BINDINGS
            if binding.key_display
            for action in (binding.action,)
        }
        self.assertEqual(display.get("toggle_shader"), "[h]")
        self.assertEqual(display.get("toggle_lut"), "[c]")



if __name__ == "__main__":
    unittest.main()