import os
import tempfile

import numpy as np
import pytest

from hyper_stage.lut import list_lut_files
from hyper_stage.shaders import (
    MultiShader, ShaderEngine, list_shader_files, load_shader,
)


class TestShaderFiles:
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
        assert [os.path.basename(path) for path in files] == ['root.glsl', 'nested.glsl']

    def test_load_shader_reads_source_and_rejects_bad_files(self):
        import os
        import tempfile

        assert load_shader('') is None
        assert load_shader('/no/such/file.glsl') is None
        with tempfile.TemporaryDirectory() as folder:
            plain = os.path.join(folder, "plain.glsl")
            open(plain, "w").write("vec3 noop(vec3 c) { return c; }")
            assert load_shader(plain) is None
            good = os.path.join(folder, "good.glsl")
            source = "void mainImage(out vec4 c, in vec2 p) { c = vec4(1.0); }"
            open(good, "w").write(source)
            assert load_shader(good) == source


class TestShaderEngine:
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
                assert (engine.apply(frame, broken) == frame).all()
        except Exception:
            pytest.skip("no usable GPU context in this environment")
        finally:
            if engine._ctx is not None:
                engine._ctx.release()
        assert out.shape == frame.shape
        assert out.dtype == np.uint8
        # Passthrough must not reorder or flip anything.
        assert (out == frame).all()

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
            pytest.skip("no usable GPU context in this environment")
        finally:
            if engine._ctx is not None:
                engine._ctx.release()
        assert not ((out == frame).all())
        # The marked region is inverted (255 - value, small tolerance).
        assert abs(int(out[15, 25, 0]) - (255 - 30)) <= 2

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
            pytest.skip("no usable GPU context in this environment")
        finally:
            if engine._ctx is not None:
                engine._ctx.release()
        # Not blacklisted as broken and not passed through unchanged.
        assert path not in engine._broken
        assert not ((out == frame).all())

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
            pytest.skip("no usable GPU context in this environment")
        finally:
            if engine._ctx is not None:
                engine._ctx.release()
        assert path not in engine._broken
        # With the mouse at (0,0) the frame passes through untouched.
        assert (out == frame).all()

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
            pytest.skip("no usable GPU context in this environment")
        finally:
            if engine._ctx is not None:
                engine._ctx.release()
        # Non-constant noise pattern with real variance.
        assert int(out.std()) > 10
        assert int(out.max()) > 40

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
            pytest.skip("no usable GPU context in this environment")
        finally:
            if engine._ctx is not None:
                engine._ctx.release()
        assert font is not None
        # Glyph cells are addressed as code = row_cell*16 + col_cell:
        # '0' is 48 (row 3, col 0) and 'A' is 65 (row 4, col 1); codes
        # below 32 stay blank.
        assert int(data[48:64, 0:16, 0].max()) > 100
        assert int(data[64:80, 16:32, 0].max()) > 100
        assert int(data[0:16, 0:16, 0].max()) == 0

    def test_channel_directive_is_parsed(self):
        source = (
            "// hyper-stage: 0=font 1=camera\n"
            "// hyper-stage: 2=noise\n"
            "void mainImage(out vec4 c, in vec2 p) {}\n"
        )
        assert ShaderEngine.parse_channels(source) == {0: 'font', 1: 'camera', 2: 'noise'}
        assert ShaderEngine.parse_channels('void mainImage() {}') == {}

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
            pytest.skip("no usable GPU context in this environment")
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
        assert (expect == got).mean() >= 0.999

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
            pytest.skip("no usable GPU context in this environment")
        finally:
            if engine._ctx is not None:
                engine._ctx.release()
        # Screen space: red (y > H/2) on the published top half. The shader
        # writes RGB; the engine's verbatim-import round trip publishes it
        # as BGR, so red (1,0,0) arrives as (0,0,255).
        assert int(out[10, 40, 0]) == 0
        assert int(out[10, 40, 2]) == 255
        assert int(out[50, 40, 0]) == 255
        assert int(out[50, 40, 2]) == 0
        # Camera sampling: still pixel-exact, no flip.
        assert (cam_out == frame).all()

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
            pytest.skip("no usable GPU context in this environment")
        finally:
            if engine._ctx is not None:
                engine._ctx.release()
        # Published BGR must read as the intended orange hue (values are
        # the palette floats through the 8-bit round trip).
        assert int(out[20, 25, 0]) == 45
        assert int(out[20, 25, 1]) == 122
        assert int(out[20, 25, 2]) == 242

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
                    assert os.path.realpath(os.path.join(folder, 'pack.glsl')) in [os.path.realpath(p) for p in found]
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
            pytest.skip("no usable GPU context in this environment")
        finally:
            if engine._ctx is not None and pack_dir in engine._programs:
                engine._programs[pack_dir].close()
            if engine._ctx is not None:
                engine._ctx.release()
        assert pack_dir not in engine._broken
        # Pass order, feedback, and the RGB round trip: frame 1 Buffer A
        # accumulates 0.25 toward the camera's blue (0.784 in RGB), frame 2
        # reaches 0.5; Image inverts and the engine converts back to BGR.
        assert abs(int(out1[5, 5, 0]) - (255 - 64)) <= (3)
        assert abs(int(out2[5, 5, 0]) - (255 - 128)) <= (3)
        assert int(out1[5, 5, 1]) == 255
        assert int(out1[5, 5, 2]) == 255

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
                pytest.skip("no usable GPU context in this environment")
            finally:
                engine._ctx.release()
            assert (out == frame).all()
            assert pack_dir in engine._broken

    def test_pack_channel_directive_parsing(self):
        """Pack pass directives map units to aux names and buffer passes."""
        parse = MultiShader.parse_channels
        names = ["Buffer A.glsl", "Buffer B.glsl"]
        assert parse('// hyper-stage: 1=noise 2=Buffer B', names) == {1: 'noise', 2: ('buffer', 1)}
        # Spaces optional, case ignored.
        assert parse('// hyper-stage: 0=feedback 1=bufferb', names) == {0: 'feedback', 1: ('buffer', 1)}
        # Unknown names are ignored, leaving the default wiring.
        assert parse('// hyper-stage: 3=webcam', names) == {}

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
            pytest.skip("no usable GPU context in this environment")
        finally:
            if engine._ctx is not None and pack_dir in engine._programs:
                engine._programs[pack_dir].close()
            if engine._ctx is not None:
                engine._ctx.release()
        assert pack_dir not in engine._broken
        # Verbatim RGB import, so red publishes as BGR (R in the last slot).
        assert int(out[5, 5, 0]) == 0
        assert int(out[5, 5, 1]) == 0
        assert int(out[5, 5, 2]) == 255

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
                pytest.skip("no usable GPU context in this environment")
            finally:
                if engine._programs.get(pack_dir):
                    engine._programs[pack_dir].close()
                engine._ctx.release()
            assert (out == frame).all()
            assert pack_dir in engine._broken

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
            pytest.skip("no usable GPU context in this environment")
        finally:
            if engine._ctx is not None:
                engine._ctx.release()
        assert path not in engine._broken
        # res / (40, 30) == (1, 1) -> RGB (1, 1, 0), published as BGR
        # (0, 255, 255) by the verbatim-import round trip.
        assert int(out[5, 5, 0]) == 0
        assert int(out[5, 5, 1]) == 255
        assert int(out[5, 5, 2]) == 255

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
            pytest.skip("no usable GPU context in this environment")
        finally:
            if engine._ctx is not None and engine._programs.get(pack_dir):
                engine._programs[pack_dir].close()
            if engine._ctx is not None:
                engine._ctx.release()
        assert pack_dir not in engine._broken
        # 200/255 through mip levels stays ~0.784; published BGR ~200 again.
        assert abs(int(out[5, 5, 0]) - (200)) <= (4)

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
            pytest.skip("no usable GPU context in this environment")
        finally:
            if engine._ctx is not None:
                engine._ctx.release()
        assert path not in engine._broken
        # The noise texture's pixels, not driver garbage: varied values.
        assert len(np.unique(out[..., 0])) > 4


class TestLutDiscovery:
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
        assert [os.path.basename(path) for path in files] == ['nested.cube', 'root.cube']

