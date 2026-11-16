import unittest

from cli import *  # noqa: F401,F403
from config import *  # noqa: F401,F403
from effects import *  # noqa: F401,F403
from lut import *  # noqa: F401,F403
from magic_loop import *  # noqa: F401,F403
from pipeline import *  # noqa: F401,F403
from pipmode import *  # noqa: F401,F403
from shaders import *  # noqa: F401,F403
from state import *  # noqa: F401,F403
from tracking import *  # noqa: F401,F403

import cv2  # noqa: F401
import pyvirtualcam  # noqa: F401
from mediapipe.tasks import python as mp_python  # noqa: F401
from mediapipe.tasks.python import vision  # noqa: F401


class TestCropOrigin(unittest.TestCase):
    def test_centered(self):
        x1, y1, cw, ch = crop_origin(0.5, 0.5, 2.0, 1000, 1000)
        self.assertEqual(cw, 500)
        self.assertEqual(ch, 500)
        self.assertEqual(x1, 250)
        self.assertEqual(y1, 250)

    def test_clamped_left_top(self):
        x1, y1, cw, ch = crop_origin(0.0, 0.0, 2.0, 1000, 1000)
        self.assertEqual(x1, 0)
        self.assertEqual(y1, 0)

    def test_clamped_right_bottom(self):
        x1, y1, cw, ch = crop_origin(1.0, 1.0, 2.0, 1000, 1000)
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
        centers = face_centers([d], 640, 360)
        self.assertEqual(len(centers), 1)
        self.assertAlmostEqual(centers[0][0], (100 + 100) / 640)
        self.assertAlmostEqual(centers[0][1], (100 + 200 * 0.6) / 360)


class TestLoadConfig(unittest.TestCase):
    def test_defaults_when_missing(self):
        cfg = load_config(None)
        self.assertEqual(cfg["camera_index"], 1)
        self.assertEqual(cfg["zoom"], 1.5)

    def test_ini_override(self):
        import tempfile, os
        fd, path = tempfile.mkstemp(suffix=".ini")
        with os.fdopen(fd, "w") as f:
            f.write("[camera]\nindex = 3\n[tracking]\nzoom = 2.0\n")
        try:
            cfg = load_config(path)
            self.assertEqual(cfg["camera_index"], 3)
            self.assertEqual(cfg["zoom"], 2.0)
        finally:
            os.remove(path)

    def test_pip_animation_override(self):
        import tempfile, os
        fd, path = tempfile.mkstemp(suffix=".ini")
        with os.fdopen(fd, "w") as f:
            f.write("[pip]\nanimation = 1.5\n")
        try:
            cfg = load_config(path)
            self.assertEqual(cfg["pip_animation"], 1.5)
        finally:
            os.remove(path)


class TestLoadLut(unittest.TestCase):
    def test_loads_cube(self):
        files = list_lut_files(None)
        if not files:
            self.skipTest("no .cube files in LUT/")
        lut = load_lut(files[0])
        self.assertIsNotNone(lut)
        self.assertEqual(lut.dtype, __import__("numpy").float32)

    def test_missing_returns_none(self):
        self.assertIsNone(load_lut("/no/such/file.cube"))


class TestSmartSharpen(unittest.TestCase):
    def test_flat_area_untouched(self):
        import numpy as np
        frame = np.full((32, 32, 3), 120, dtype=np.uint8)
        out = apply_smart_sharpen(frame, strength=0.3, threshold=10)
        self.assertTrue((out == frame).all())

    def test_edge_is_sharpened(self):
        import numpy as np
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        frame[:, 16:] = 200
        out = apply_smart_sharpen(frame, strength=0.3, threshold=10)
        self.assertFalse((out == frame).all())

    def test_zero_strength_is_noop(self):
        import numpy as np
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        frame[:, 16:] = 200
        out = apply_smart_sharpen(frame, strength=0.0)
        self.assertTrue((out == frame).all())


def make_grey_lut(folder, name="bw.cube", size=5):
    """Write a synthetic pure-desaturation .cube LUT (out = luma, grey)."""
    import os

    lines = [f"LUT_3D_SIZE {size}"]
    for b in range(size):
        for g in range(size):
            for r in range(size):
                y = (0.114 * b + 0.587 * g + 0.299 * r) / (size - 1)
                lines.append(f"{y:.6f} {y:.6f} {y:.6f}")
    path = os.path.join(folder, name)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def make_blue_disc_frame():
    """A grey field with a noisy, saturated blue disc: leak amplifier."""
    import numpy as np

    frame = np.full((270, 480, 3), 128, np.uint8)
    yy, xx = np.mgrid[0:270, 0:480]
    mask = (xx - 240) ** 2 + (yy - 135) ** 2 < 80 ** 2
    rng = np.random.default_rng(3)
    frame[mask] = np.stack(
        [
            rng.integers(140, 256, int(mask.sum())),
            rng.integers(0, 60, int(mask.sum())),
            rng.integers(0, 60, int(mask.sum())),
        ],
        axis=1,
    )
    frame += rng.integers(-6, 7, frame.shape, dtype=np.int16).astype(np.uint8)
    return frame


class TestApplyLut(unittest.TestCase):
    def test_detail_mode_does_not_leak_chroma(self):
        """A B&W LUT with lut_detail=True must not re-tint from full-res
        chroma: the sharpness detail term is luma-only."""
        import os
        import tempfile

        import numpy as np

        with tempfile.TemporaryDirectory() as folder:
            path = make_grey_lut(folder)
            table = bake_lut(load_lut(path))
            frame = make_blue_disc_frame()
            soft = apply_lut(frame, table, scale=8, detail=False)
            sharp = apply_lut(frame, table, scale=8, detail=True)
            chroma_soft = soft.max(axis=2).astype(int) - soft.min(axis=2)
            chroma_sharp = sharp.max(axis=2).astype(int) - sharp.min(axis=2)
            # The graded low-res image defines how grey the output can be;
            # the detail path may add at most rounding on top of that.
            self.assertLessEqual(
                int(chroma_sharp.max()), int(chroma_soft.max()) + 2
            )

    def test_detail_mode_preserves_luma_detail(self):
        """The chroma-stripped detail term keeps edges sharp."""
        import os
        import tempfile

        import numpy as np

        with tempfile.TemporaryDirectory() as folder:
            path = make_grey_lut(folder)
            table = bake_lut(load_lut(path))
            # Grey step edge plus noise: luma detail must survive.
            frame = np.tile(
                np.linspace(60, 200, 480, dtype=np.uint8), (270, 1)
            )[..., None]
            frame = np.repeat(frame, 3, axis=2)
            rng = np.random.default_rng(5)
            frame += rng.integers(-6, 7, frame.shape, dtype=np.int16).astype(
                np.uint8
            )
            out = apply_lut(frame, table, scale=8, detail=True)
            luma = out.astype(np.float32) @ np.array([0.114, 0.587, 0.299])
            # Gradient energy in line direction stays close to the input's.
            grad_in = np.abs(np.diff(luma, axis=1)).mean()
            grad_out = np.abs(np.diff(luma, axis=1)).mean()
            self.assertGreater(grad_out, grad_in * 0.8)


class TestLutCycling(unittest.TestCase):
    def test_cycle_wraps_forward(self):
        self.assertEqual(cycle_index(2, 3, 1), 0)
        self.assertEqual(cycle_index(0, 3, 1), 1)

    def test_cycle_wraps_backward(self):
        self.assertEqual(cycle_index(0, 3, -1), 2)
        self.assertEqual(cycle_index(1, 3, -1), 0)

    def test_empty_list_is_safe(self):
        self.assertEqual(cycle_index(0, 0, 1), 0)

    def test_lut_files_sorted(self):
        files = list_lut_files(None)
        self.assertEqual(files, sorted(files))


class TestSpringStep(unittest.TestCase):
    def test_critically_damped_no_overshoot_and_converges(self):
        pos, vel, target = 1.0, 0.0, 1.5
        omega, zeta, dt = 2.0 * 3.14159 / 0.9, 1.0, 1.0 / 30.0
        for _ in range(300):
            pos, vel = spring_step(pos, vel, target, omega, zeta, dt)
            self.assertLessEqual(pos, target + 1e-6)
        self.assertAlmostEqual(pos, target, places=3)


class TestSaveConfig(unittest.TestCase):
    def test_roundtrip_preserves_unknown_sections_and_writes_atomically(self):
        import configparser
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "hyper-stage.ini")
            with open(path, "w") as f:
                f.write("[camera]\nindex = 2\n\n[custom]\nkeep = yes\n")
            cfg = load_config(path)
            cfg["zoom"] = 2.25
            cfg["lut_enabled"] = False
            cfg["time_loop_enabled"] = True
            cfg["lut_file"] = os.path.join(folder, "LUT", "a.cube")
            save_config(path, cfg)

            reloaded = load_config(path)
            self.assertEqual(reloaded["camera_index"], 2)
            self.assertEqual(reloaded["zoom"], 2.25)
            self.assertFalse(reloaded["lut_enabled"])
            self.assertTrue(reloaded["time_loop_enabled"])
            self.assertEqual(reloaded["lut_file"], os.path.join("LUT", "a.cube"))

            parser = configparser.ConfigParser()
            parser.read(path)
            self.assertEqual(parser.get("custom", "keep"), "yes")
            self.assertEqual(
                [n for n in os.listdir(folder) if n.startswith(".hyper-stage-")], []
            )

    def test_effects_section_is_loaded(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "hyper-stage.ini")
            with open(path, "w") as f:
                f.write(
                    "[color]\nlut_enabled = true\nshader_enabled = true\n"
                    "shader_file = sh.glsl\n"
                    "[effects]\ntime_loop = true\ndeep_fry = true\n"
                    "magic_loop = true\n"
                )
            cfg = load_config(path)
            self.assertTrue(cfg["time_loop_enabled"])
            self.assertTrue(cfg["deep_fry_enabled"])
            self.assertTrue(cfg["magic_loop_enabled"])
            self.assertTrue(cfg["shader_enabled"])
            self.assertEqual(cfg["shader_file"], "sh.glsl")

    def test_theme_is_loaded_and_saved(self):
        import configparser
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "hyper-stage.ini")
            with open(path, "w") as f:
                f.write("[appearance]\ntheme = textual-light\n")
            self.assertEqual(load_config(path)["theme"], "textual-light")

            cfg = dict(DEFAULTS)
            cfg["theme"] = "solarized-light"
            save_config(path, cfg)
            parser = configparser.ConfigParser()
            parser.read(path)
            self.assertEqual(parser.get("appearance", "theme"), "solarized-light")


class TestSharedState(unittest.TestCase):
    def test_update_marks_dirty_and_merges_into_config(self):
        cfg = dict(DEFAULTS)
        state = SharedState(cfg)
        self.assertIsNone(state.pending_revision())
        state.update(zoom=2.0)
        self.assertIsNotNone(state.pending_revision())
        self.assertEqual(state.snapshot()["zoom"], 2.0)
        self.assertEqual(state.config(cfg)["zoom"], 2.0)
        self.assertEqual(state.config(cfg)["fps"], cfg["fps"])

    def test_telemetry_does_not_dirty_the_config(self):
        state = SharedState(dict(DEFAULTS))
        state.set_telemetry(fps=29.9, tracking=True)
        self.assertIsNone(state.pending_revision())
        self.assertEqual(state.telemetry()["fps"], 29.9)
        self.assertTrue(state.telemetry()["tracking"])

    def test_edit_during_save_stays_pending(self):
        state = SharedState(dict(DEFAULTS))
        state.update(zoom=2.0)
        revision = state.pending_revision()
        state.update(zoom=3.0)
        state.mark_saved(revision)
        self.assertIsNotNone(state.pending_revision())
        state.mark_saved(state.pending_revision())
        self.assertIsNone(state.pending_revision())

    def test_unknown_setting_is_rejected(self):
        state = SharedState(dict(DEFAULTS))
        with self.assertRaises(KeyError):
            state.update(no_such_setting=1)


class TestTimeLoop(unittest.TestCase):
    def test_records_while_off_and_replays_while_on(self):
        loop = TimeLoop(length=3)
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
        loop = TimeLoop(length=3)
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
        out = apply_deep_fry(frame)
        self.assertEqual(out.shape, frame.shape)
        self.assertEqual(out.dtype, np.uint8)
        self.assertFalse((out == frame).all())


class TestMagicLoop(unittest.TestCase):
    CROP = (0.5, 0.45, 1.5)

    def test_build_loop_returns_none_without_enough_history(self):
        import numpy as np

        buf = MagicLoopBuffer()
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        buf.record(frame, self.CROP)
        self.assertIsNone(buf.build_loop(self.CROP))

    def test_seam_search_picks_the_calmest_window(self):
        import cv2
        import numpy as np

        buf = MagicLoopBuffer()
        still = np.full((180, 320, 3), 40, dtype=np.uint8)
        # Busy first, then a still stretch long enough to hold a minimum
        # loop (2 s) plus its crossfade tail (span = loop + blend frames).
        for i in range(MAGIC_LOOP_MIN_FRAMES + 120):
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
        self.assertGreaterEqual(player._length, MAGIC_LOOP_MIN_LOOP_FRAMES)
        first = cv2.imdecode(
            np.frombuffer(player._frames[0], np.uint8), cv2.IMREAD_COLOR
        )
        # The winner starts in the still zone, not in the busy part.
        self.assertLess(int(first.max()), 100)

    def test_loop_rejects_a_window_shorter_than_two_seconds(self):
        import numpy as np

        buf = MagicLoopBuffer()
        still = np.full((180, 320, 3), 40, dtype=np.uint8)
        # Just under MAGIC_LOOP_MIN_FRAMES: a full calm window cannot exist.
        for _ in range(MAGIC_LOOP_MIN_FRAMES - 1):
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
        player = MagicLoopPlayer(encoded, length, blend)
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
                files = list_shader_files(None)
            finally:
                os.chdir(old_cwd)
        self.assertEqual(
            [os.path.basename(path) for path in files],
            ["root.glsl", "nested.glsl"],
        )

    def test_load_shader_reads_source_and_rejects_bad_files(self):
        import os
        import tempfile

        self.assertIsNone(load_shader(""))
        self.assertIsNone(load_shader("/no/such/file.glsl"))
        with tempfile.TemporaryDirectory() as folder:
            plain = os.path.join(folder, "plain.glsl")
            open(plain, "w").write("vec3 noop(vec3 c) { return c; }")
            self.assertIsNone(load_shader(plain))
            good = os.path.join(folder, "good.glsl")
            source = "void mainImage(out vec4 c, in vec2 p) { c = vec4(1.0); }"
            open(good, "w").write(source)
            self.assertEqual(load_shader(good), source)


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
            engine = ShaderEngine()
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
        engine = ShaderEngine()
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
        engine = ShaderEngine()
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
        engine = ShaderEngine()
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
        engine = ShaderEngine()
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

        engine = ShaderEngine()
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
            "// hyper-stage: 0=font 1=camera\n"
            "// hyper-stage: 2=noise\n"
            "void mainImage(out vec4 c, in vec2 p) {}\n"
        )
        self.assertEqual(
            ShaderEngine.parse_channels(source),
            {0: "font", 1: "camera", 2: "noise"},
        )
        self.assertEqual(
            ShaderEngine.parse_channels("void mainImage() {}"), {}
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
        engine = ShaderEngine()
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
        engine = ShaderEngine()
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
        engine = ShaderEngine()
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
        engine = ShaderEngine()
        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        frame[..., 0] = 200  # BGR blue
        try:
            with tempfile.TemporaryDirectory() as folder:
                os.makedirs(os.path.join(folder, "pack.glsl"))
                old_cwd = os.getcwd()
                os.chdir(folder)
                try:
                    # Discovered alongside single files.
                    found = list_shader_files(None)
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
            engine = ShaderEngine()
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
        parse = MultiShader.parse_channels
        names = ["Buffer A.glsl", "Buffer B.glsl"]
        self.assertEqual(
            parse("// hyper-stage: 1=noise 2=Buffer B", names),
            {1: "noise", 2: ("buffer", 1)},
        )
        # Spaces optional, case ignored.
        self.assertEqual(
            parse("// hyper-stage: 0=feedback 1=bufferb", names),
            {0: "feedback", 1: ("buffer", 1)},
        )
        # Unknown names are ignored, leaving the default wiring.
        self.assertEqual(parse("// hyper-stage: 3=webcam", names), {})

    def test_multibuffer_pack_channel_directive(self):
        """A pass directive wires iChannelN to a named buffer's output."""
        import os
        import tempfile

        import numpy as np

        pack_dir = None
        engine = ShaderEngine()
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
                        "// hyper-stage: 2=Buffer A\n"
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
            engine = ShaderEngine()
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
        engine = ShaderEngine()
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
        engine = ShaderEngine()
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
        engine = ShaderEngine()
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
                files = list_lut_files(None)
            finally:
                os.chdir(old_cwd)
        self.assertEqual(
            [os.path.basename(path) for path in files],
            ["nested.cube", "root.cube"],
        )


class TestQuietNativeStderr(unittest.TestCase):
    def test_detector_setup_logs_are_silenced(self):
        """Native LiteRT logs at detector create/close never reach stderr."""
        import subprocess
        import sys

        code = (
            "import sys\n"
            "from mediapipe.tasks import python as mp_python\n"
            "from mediapipe.tasks.python import vision\n"
            "from tracking import ensure_model, quiet_native_stderr\n"
            "options = vision.FaceDetectorOptions(\n"
            "    base_options=mp_python.BaseOptions(\n"
            "        model_asset_path=ensure_model()))\n"
            "with quiet_native_stderr():\n"
            "    ctx = vision.FaceDetector.create_from_options(options)\n"
            "    detector = ctx.__enter__()\n"
            "detector.detect  # attribute access proves it is live\n"
            "with quiet_native_stderr():\n"
            "    ctx.__exit__(None, None, None)\n"
            "sys.stderr.write('AFTER-MARK\\n')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("INFO:", result.stderr)
        self.assertNotIn("TensorFlow Lite", result.stderr)
        # stderr outside the silenced windows still reaches the terminal.
        self.assertIn("AFTER-MARK", result.stderr)

    def test_quiet_native_stderr_swallows_fd2_writes_only(self):
        import os
        import subprocess
        import sys

        code = (
            "import os, sys\n"
            "from tracking import quiet_native_stderr\n"
            "with quiet_native_stderr():\n"
            "    os.write(2, b'HIDDEN\\n')\n"
            "os.write(2, b'VISIBLE\\n')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("HIDDEN", result.stderr)
        self.assertIn("VISIBLE", result.stderr)


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

        cfg = dict(DEFAULTS)
        cfg.update(out_w=160, out_h=90, tracking_enabled=False, sharpen_enabled=True)
        state = SharedState(cfg)
        pipeline = VideoPipeline(cfg, state)
        with (
            mock.patch("pipeline.ensure_model", return_value="fake"),
            mock.patch.object(cv2, "VideoCapture", FakeCapture),
            mock.patch.object(pyvirtualcam, "Camera", FakeCamera),
            mock.patch.object(
                vision.FaceDetector,
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

    def test_pip_composites_slide_background_with_corner_webcam(self):
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
                # Distinct green live frame: easy to tell from the background.
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

        cfg = dict(DEFAULTS)
        cfg.update(out_w=160, out_h=90, tracking_enabled=False, pip_enabled=True)
        state = SharedState(cfg)
        # An asymmetric blue/white background, pre-fitted to the output size
        # like the TUI worker would deliver it: blue right half, white left.
        background = np.zeros((90, 160, 3), dtype=np.uint8)
        background[:, :80] = (255, 255, 255)
        background[:, 80:] = (255, 10, 10)
        state.set_pip_frame(background)
        pipeline = VideoPipeline(cfg, state)
        with (
            mock.patch("pipeline.ensure_model", return_value="fake"),
            mock.patch.object(cv2, "VideoCapture", FakeCapture),
            mock.patch.object(pyvirtualcam, "Camera", FakeCamera),
            mock.patch.object(
                vision.FaceDetector,
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
            unflipped = sent[0]
            # Enabling PiP Flip mirrors the cached display background: the
            # white half moves right on the next frame, without touching the
            # stored one.
            state.update(pip_flip=True)
            sent.clear()
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and len(sent) < 3:
                time.sleep(0.005)
            pipeline.stop()
            pipeline.join(2.0)

        self.assertFalse(pipeline.is_alive())
        self.assertGreaterEqual(len(sent), 1)
        x, y, w, h = pip_corner_rect(
            state.snapshot()["pip_corner"],
            cfg["pip_scale"],
            cfg["pip_padding"],
            160,
            90,
        )
        # Flip off: the white half is on the left, as stored...
        self.assertEqual(tuple(unflipped[45, 40]), (255, 255, 255))
        self.assertEqual(tuple(unflipped[45, 120]), (255, 10, 10))
        # ...and the corner window holds the live webcam.
        self.assertEqual(tuple(unflipped[y + h // 2, x + w // 2]), (0, 128, 0))
        # Flip on: the published frame is mirrored...
        flipped = sent[-1]
        self.assertEqual(tuple(flipped[45, 120]), (255, 255, 255))
        self.assertEqual(tuple(flipped[45, 40]), (255, 10, 10))
        # ...and the stored background itself was never mutated.
        self.assertEqual(tuple(background[45, 120]), (255, 10, 10))

    def test_pip_transition_rect_eases_from_full_frame_to_corner(self):
        from pipmode import (
            CORNER_BR,
            compose_pip_rect,
            pip_corner_rect,
            pip_transition_rect,
        )

        w, h = 160, 90
        corner = pip_corner_rect(CORNER_BR, 0.25, 8, w, h)
        # progress 0: the full frame (PiP off)...
        self.assertEqual(pip_transition_rect(0.0, CORNER_BR, 0.25, 8, w, h),
                         (0, 0, w, h))
        # ...progress 1: the exact corner window (PiP on).
        self.assertEqual(pip_transition_rect(1.0, CORNER_BR, 0.25, 8, w, h),
                         corner)
        # Mid-flight the rect sits between the two and stays in frame.
        for progress in (0.25, 0.5, 0.75):
            x, y, rw, rh = pip_transition_rect(
                progress, CORNER_BR, 0.25, 8, w, h
            )
            self.assertGreater(x, 0)
            self.assertLess(x + rw, w)
            self.assertGreater(y, 0)
            self.assertLess(y + rh, h)
            self.assertGreater(rw, corner[2])
            self.assertLess(rw, w)

        # Composing at the corner rect matches the settled compose_pip,
        # and at the full-frame rect just pastes the webcam over the bg.
        background = np.full((h, w, 3), 10, dtype=np.uint8)
        webcam = np.full((30, 30, 3), 200, dtype=np.uint8)
        settled = compose_pip_rect(
            np.zeros((h, w, 3), dtype=np.uint8), background, webcam, corner
        )
        from pipmode import compose_pip

        self.assertTrue(
            (settled == compose_pip(np.zeros((h, w, 3), dtype=np.uint8),
                                    background, webcam, CORNER_BR, 0.25, 8)).all()
        )
        full = compose_pip_rect(
            np.zeros((h, w, 3), dtype=np.uint8), background, webcam, (0, 0, w, h)
        )
        self.assertEqual(tuple(full[45, 45]), (200, 200, 200))
        # The background was never mutated.
        self.assertEqual(tuple(background[45, 45]), (10, 10, 10))

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

        cfg = dict(DEFAULTS)
        # Zoom 1.0 keeps the whole frame so the moving column stays visible
        # and every live frame remains distinct.
        cfg.update(
            out_w=160,
            out_h=90,
            magic_loop_enabled=True,
            tracking_enabled=False,
            zoom=1.0,
        )
        state = SharedState(cfg)
        pipeline = VideoPipeline(cfg, state)
        with (
            mock.patch("pipeline.ensure_model", return_value="fake"),
            mock.patch.object(cv2, "VideoCapture", FakeCapture),
            mock.patch.object(pyvirtualcam, "Camera", FakeCamera),
            mock.patch.object(
                vision.FaceDetector,
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
            mock.patch("cli.list_cameras", return_value=[]) as probe,
            mock.patch("builtins.print"),
        ):
            main(["--list-cameras"])
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
        self.assertTrue(seen["ran"])
        self.assertEqual(seen["path"], path)
        self.assertFalse(seen["cfg"]["tracking_enabled"])
        self.assertTrue(seen["cfg"]["time_loop_enabled"])
        self.assertTrue(seen["cfg"]["deep_fry_enabled"])
        self.assertTrue(seen["cfg"]["magic_loop_enabled"])
        self.assertEqual(seen["cfg"]["shader_file"], "/tmp/fx.glsl")
        self.assertTrue(seen["cfg"]["shader_enabled"])


class TestPipMode(unittest.TestCase):
    def test_image_extension_matching_is_case_insensitive(self):
        for name in ("a.gif", "b.JPG", "c.Jpeg", "d.PNG", "e.png"):
            self.assertTrue(is_image_file(name), name)
        for name in ("f.mov", "g.txt", "h.cube", "i.glsl", "gif"):
            self.assertFalse(is_image_file(name), name)

    def test_list_pip_files_finds_dropped_images(self):
        import os
        import tempfile
        from unittest import mock

        with tempfile.TemporaryDirectory() as folder:
            for name in ("slide.png", "MEME.JPG", "notes.txt", "movie.mov"):
                open(os.path.join(folder, name), "w").close()
            with mock.patch("pipmode.os.getcwd", return_value=folder):
                files = list_pip_files("")
        self.assertEqual(
            files,
            sorted(
                os.path.abspath(os.path.join(folder, name))
                for name in ("MEME.JPG", "slide.png")
            ),
        )
        # Sorted by name, ignoring case: meme.png before slide.png even
        # though uppercase letters would sort first byte-wise.
        self.assertEqual(
            [os.path.basename(f) for f in files],
            ["MEME.JPG", "slide.png"],
        )

    def test_load_pip_image_letterboxes_to_output_size(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "wide.png")
            cv2.imwrite(path, np.full((60, 120, 3), 200, dtype=np.uint8))
            frame = load_pip_image(path, (100, 100))
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (100, 100, 3))
        self.assertEqual(frame.dtype, np.uint8)
        # Wide source: full width, black bars top and bottom.
        self.assertEqual(int(frame[0, 50].max()), 0)
        self.assertEqual(int(frame[99, 50].max()), 0)
        self.assertEqual(int(frame[50, 50].max()), 200)

    def test_load_pip_image_rejects_missing_and_unreadable(self):
        self.assertIsNone(load_pip_image("", (64, 64)))
        self.assertIsNone(load_pip_image("/no/such/file.png", (64, 64)))

    def test_corner_rect_places_the_window_with_margins(self):
        # Equal paddings on both axes: margin is a fraction of the frame
        # width applied to every side (96px on 1920x1080).
        self.assertEqual(pip_corner_rect(CORNER_TL, 0.25, 96, 1920, 1080), (96, 96, 480, 270))
        self.assertEqual(pip_corner_rect(CORNER_TR, 0.25, 96, 1920, 1080), (1344, 96, 480, 270))
        self.assertEqual(pip_corner_rect(CORNER_BL, 0.25, 96, 1920, 1080), (96, 714, 480, 270))
        self.assertEqual(pip_corner_rect(CORNER_BR, 0.25, 96, 1920, 1080), (1344, 714, 480, 270))

    def test_compose_pip_pastes_webcam_into_the_background(self):
        background = np.full((100, 100, 3), 255, dtype=np.uint8)
        webcam = np.zeros((10, 10, 3), dtype=np.uint8)
        webcam[:] = (0, 255, 0)
        out = compose_pip(np.zeros((100, 100, 3), np.uint8), background, webcam, CORNER_TL, 0.25, 5)
        # Window: x=5, y=5, 25x25 in the top-left corner of the background.
        self.assertEqual(tuple(out[10, 10]), (0, 255, 0))
        self.assertEqual(tuple(out[0, 0]), (255, 255, 255))
        self.assertEqual(tuple(out[40, 40]), (255, 255, 255))

    def test_pip_flip_is_applied_by_the_pipeline_not_per_paste(self):
        # compose_pip pastes the given background as-is; the mirror happens
        # once in the pipeline when the image or the flip flag changes.
        background = np.zeros((100, 100, 3), dtype=np.uint8)
        background[:, :50] = 255
        webcam = np.zeros((10, 10, 3), dtype=np.uint8)
        webcam[:] = (0, 255, 0)
        out = compose_pip(
            np.zeros((100, 100, 3), np.uint8),
            background,
            webcam,
            CORNER_TR,
            0.25,
            0.05,
        )
        # White half stays on the left; the window (x 70..94) holds the webcam.
        self.assertEqual(tuple(out[10, 40]), (255, 255, 255))
        self.assertEqual(tuple(out[10, 80]), (0, 255, 0))

    def test_pip_config_round_trips_through_ini(self):
        import os
        import tempfile

        fd, path = tempfile.mkstemp(suffix=".ini")
        with os.fdopen(fd, "w") as f:
            f.write("[pip]\nenabled = true\nimage = img/slide.gif\nflip = true\nscale = 0.3\npadding = 100\ncorner = 2\n")
        try:
            cfg = load_config(path)
            self.assertTrue(cfg["pip_enabled"])
            self.assertEqual(cfg["pip_image"], "img/slide.gif")
            self.assertTrue(cfg["pip_flip"])
            self.assertAlmostEqual(cfg["pip_scale"], 0.3)
            self.assertEqual(cfg["pip_padding"], 100)
            self.assertEqual(cfg["pip_corner"], 2)
            save_config(path, cfg)
            again = load_config(path)
            self.assertEqual(again["pip_enabled"], True)
            self.assertEqual(again["pip_flip"], True)
            self.assertAlmostEqual(again["pip_scale"], 0.3)
            self.assertEqual(again["pip_corner"], 2)
        finally:
            os.remove(path)

    def test_cli_pip_flags_override_config(self):
        import os
        import tempfile
        import tui
        from unittest import mock

        seen = {}

        class FakeApp:
            def __init__(self, cfg, config_path, preview=False):
                seen["cfg"] = cfg

            def run(self):
                pass

        with mock.patch.object(tui, "HyperStageApp", FakeApp):
            main(["--config", os.path.join(tempfile.mkdtemp(), "t.ini"), "--pip", "--pip-image", "/tmp/s.png"])
        self.assertTrue(seen["cfg"]["pip_enabled"])
        self.assertEqual(seen["cfg"]["pip_image"], "/tmp/s.png")

    def test_pip_tunables_and_frame_round_trip(self):
        state = SharedState(dict(DEFAULTS))
        state.update(pip_enabled=True, pip_corner=CORNER_BR)
        self.assertTrue(state.snapshot()["pip_enabled"])
        self.assertEqual(state.snapshot()["pip_corner"], CORNER_BR)
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        state.set_pip_frame(frame)
        self.assertIs(state.pip_frame, frame)


class TestTui(unittest.IsolatedAsyncioTestCase):
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

    async def test_shortcuts_tuning_and_autosave(self):
        import os
        import tempfile

        import tui

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
                self.assertEqual(load_config(path)["zoom"], 2.5)
        finally:
            tui.VideoPipeline = original

    async def test_magic_loop_syncs_the_tracking_switch(self):
        import os
        import tempfile

        import tui

        original = tui.VideoPipeline
        tui.VideoPipeline = self.FakePipeline
        try:
            cfg = dict(DEFAULTS)
            cfg["lut_file"] = ""
            app = tui.HyperStageApp(cfg, os.path.join(tempfile.mkdtemp(), "t.ini"))
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
            tui.VideoPipeline = original

    async def test_magic_loop_at_startup_forces_tracking_off(self):
        import os
        import tempfile

        import tui

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
            tui.VideoPipeline = original

    async def test_shader_controls_follow_their_switch_and_select(self):
        import os
        import tempfile

        import tui

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
                self.assertFalse(select.disabled)
                self.assertTrue(app.shader_files)

                # The h shortcut flips the switch (the dropdown was never
                # locked).
                await pilot.press("h")
                await pilot.pause()
                self.assertTrue(app.state.snapshot()["shader_enabled"])
                self.assertTrue(app.query_one("#sw-shader", tui.Switch).value)

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
            tui.VideoPipeline = original

    async def test_reload_rescans_files_and_pipeline(self):
        import os
        import tempfile

        import tui

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
            tui.VideoPipeline = original

    async def test_dependent_controls_follow_their_switch(self):
        import os
        import tempfile

        import tui

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
                self.assertFalse(app.query_one("#lut-select", tui.Select).disabled)
                self.assertFalse(app.query_one("#shader-select", tui.Select).disabled)
                self.assertFalse(app.query_one("#pip-select", tui.Select).disabled)
                self.assertFalse(app.query_one("#btn-pip-pick", tui.Button).disabled)
                # Tuning rows stay bound to their feature switch.
                self.assertTrue(app.query_one("#in-lutscale", tui.Input).disabled)
                self.assertTrue(app.query_one("#in-sharpen", tui.Input).disabled)

                await pilot.press("c")
                await pilot.press("s")
                await pilot.pause()
                self.assertFalse(app.query_one("#in-lutscale", tui.Input).disabled)
                self.assertFalse(app.query_one("#in-sharpen", tui.Input).disabled)

                await pilot.press("s")
                await pilot.pause()
                self.assertTrue(app.query_one("#in-sharpen", tui.Input).disabled)
        finally:
            tui.VideoPipeline = original

    async def test_lut_select_updates_state(self):
        import os
        import tempfile

        import tui

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
                self.assertTrue(app.lut_files)
                select.value = app.lut_files[0]
                await pilot.pause(0.5)
                self.assertEqual(app.state.snapshot()["lut_file"], app.lut_files[0])
                self.assertIsNotNone(app.state.lut_table)
        finally:
            tui.VideoPipeline = original

    async def test_russian_layout_equivalents_cover_all_hotkeys(self):
        """Every Latin hotkey fires through its ЙЦУКЕН equivalent key."""
        import os
        import tempfile

        import tui

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
                    self.assertTrue(
                        app.query_one(f"#{switch_id}", tui.Switch).value
                        is expected,
                        f"{key} did not toggle {switch_id}",
                    )
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
                self.assertEqual(latin, russian)
                # q equivalent actually shuts the app down (tested last).
                await pilot.press("й")
                await pilot.pause(0.3)
                self.assertFalse(app.is_running)
        finally:
            tui.VideoPipeline = original

    async def test_bracket_keys_cycle_luts(self):
        import os
        import tempfile

        import tui

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
            tui.VideoPipeline = original

    async def test_angle_bracket_keys_cycle_shaders(self):
        import os
        import tempfile

        import tui

        original = tui.VideoPipeline
        tui.VideoPipeline = self.FakePipeline
        try:
            cfg = dict(DEFAULTS)
            cfg["shader_file"] = ""
            app = tui.HyperStageApp(cfg, os.path.join(tempfile.mkdtemp(), "t.ini"))
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
            tui.VideoPipeline = original

    def test_shader_binding_shows_in_footer_as_h(self):
        import tui

        display = {
            action: binding.key_display
            for binding in tui.HyperStageApp.BINDINGS
            if binding.key_display
            for action in (binding.action,)
        }
        self.assertEqual(display.get("toggle_shader"), "<h>")
        self.assertEqual(display.get("toggle_lut"), "[c]")

    async def test_theme_is_applied_and_persisted(self):
        import os
        import tempfile

        import tui

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
                    self.assertEqual(app.theme, "textual-light")
                    # ...and a theme change (command palette) is saved.
                    app.theme = "textual-dark"
                    await pilot.pause(0.1)
                self.assertEqual(load_config(ini)["theme"], "textual-dark")
        finally:
            tui.VideoPipeline = original

    async def test_pip_shortcuts_corner_snap_and_discovery(self):
        import os
        import tempfile

        import tui

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
                    self.assertIn(image, app.pip_files)
                    self.assertIn(
                        image,
                        [option[1] for option in app.pip_options()],
                    )

                    # ...and selecting it decodes a fitted background frame.
                    app.apply_pip_image(image)
                    await pilot.pause(0.3)
                    self.assertEqual(app.state.snapshot()["pip_image"], image)
                    self.assertIsNotNone(app.state.pip_frame)

                    # Arrows move one axis at a time: down flips the vertical
                    # side (TL -> BL), left then flips the horizontal one
                    # (BL -> BR; swapped to match the mirrored self-preview).
                    await pilot.press("down")
                    await pilot.pause()
                    self.assertEqual(app.state.snapshot()["pip_corner"], 2)
                    await pilot.press("left")
                    await pilot.pause()
                    self.assertEqual(app.state.snapshot()["pip_corner"], 3)

                    # With PiP off the arrows are refused...
                    await pilot.press("p")
                    await pilot.pause()
                    self.assertFalse(app.state.snapshot()["pip_enabled"])
                    await pilot.press("right")
                    await pilot.pause()
                    self.assertEqual(app.state.snapshot()["pip_corner"], 3)

                    # ...and p/з toggles back on.
                    await pilot.press("з")
                    await pilot.pause()
                    self.assertTrue(app.state.snapshot()["pip_enabled"])

                    # Arrows move the corner even with the dropdown focused:
                    # the bindings are priority, so no widget can swallow them.
                    app.query_one("#pip-select", tui.Select).focus()
                    await pilot.pause()
                    await pilot.press("up")
                    await pilot.pause()
                    # Up flips only the vertical side: BR -> TR.
                    self.assertEqual(app.state.snapshot()["pip_corner"], 1)

                    # The flip switch mirrors the background.
                    app.query_one("#sw-pip-flip", tui.Switch).value = True
                    await pilot.pause()
                    self.assertTrue(app.state.snapshot()["pip_flip"])

                    # Dropdown labels show the parent folder, not just the
                    # file name.
                    second = os.path.join(folder, "meme.png")
                    cv2.imwrite(second, np.zeros((40, 60, 3), dtype=np.uint8))
                    with mock.patch("os.getcwd", return_value=folder):
                        app.refresh_pip_files()
                    await pilot.pause()
                    folder_label = f"…/{os.path.basename(folder)}/"
                    self.assertEqual(
                        [option[0] for option in app.pip_options()],
                        [folder_label + "meme.png", folder_label + "slide.png"],
                    )

                    # +/- step through the current folder's images and stop
                    # at its ends instead of running into other folders...
                    await pilot.press("plus")
                    await pilot.pause()
                    self.assertEqual(app.state.snapshot()["pip_image"], image)
                    await pilot.press("minus")
                    await pilot.pause()
                    self.assertEqual(app.state.snapshot()["pip_image"], second)
                    await pilot.press("minus")
                    await pilot.pause()
                    # ...already at the folder's first file: minus stays.
                    self.assertEqual(app.state.snapshot()["pip_image"], second)

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
                    self.assertEqual(app.state.snapshot()["pip_image"], other)
                    await pilot.press("minus")
                    await pilot.pause()
                    self.assertEqual(app.state.snapshot()["pip_image"], other)

                    # ...and digits jump inside the current folder (2 ->
                    # second entry of its folder).
                    app.apply_pip_image(second)
                    await pilot.pause()
                    await pilot.press("2")
                    await pilot.pause()
                    self.assertEqual(app.state.snapshot()["pip_image"], image)
        finally:
            tui.VideoPipeline = original


if __name__ == "__main__":
    unittest.main()