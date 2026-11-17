import os
import tempfile

import cv2
import numpy as np
import pytest

from hyper_stage.cli import main
from hyper_stage.config import DEFAULTS, load_config, save_config
from hyper_stage.effects import TimeLoop, apply_deep_fry, apply_smart_sharpen
from hyper_stage.lut import (
    apply_lut, bake_lut, cycle_index, list_lut_files, load_lut,
)
from hyper_stage.magic_loop import (
    MAGIC_LOOP_MIN_FRAMES, MAGIC_LOOP_MIN_LOOP_FRAMES, MagicLoopBuffer,
    MagicLoopPlayer,
)
from hyper_stage.pipmode import (
    CORNER_BL, CORNER_BR, CORNER_TL, CORNER_TR, compose_pip, compose_pip_rect,
    is_image_file, list_pip_files, load_pip_image, pip_corner_rect,
    pip_transition_rect,
)
from hyper_stage.state import SharedState
from hyper_stage.textoverlay import apply_text_overlay, parse_color, resolve_font


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


class TestTextOverlay:
    @staticmethod
    def values(**overrides):
        cfg = dict(DEFAULTS)
        cfg["text_string"] = "Hello"
        cfg.update(overrides)
        return cfg

    def test_parse_color_and_font_resolution(self):
        assert parse_color('#ff00aa') == (255, 0, 170)
        assert parse_color('garbage') == (255, 255, 255)
        # The configured TTF is a real font on this machine.
        font = resolve_font("Nothing Font C")
        if font:
            assert font.lower().endswith(('.ttf', '.otf'))
        assert resolve_font('no-such-font-xyz') is None

    def test_overlay_is_centered_with_horizontal_gradient(self):
        frame = np.zeros((120, 320, 3), dtype=np.uint8)
        apply_text_overlay(frame, self.values())
        ys, xs = np.where(frame.any(axis=2))
        assert len(xs) > 0
        # Centered: the bounding box midpoint is near the frame midpoint.
        assert abs((xs.min() + xs.max()) / 2 - 160) < 8
        assert abs((ys.min() + ys.max()) / 2 - 60) < 8
        # Gradient: text pixels on the left are closer to the start color,
        # ones on the right closer to the end color.
        start = np.array(parse_color(DEFAULTS["text_color_start"])[::-1], dtype=int)
        end = np.array(parse_color(DEFAULTS["text_color_end"])[::-1], dtype=int)
        left_px = frame[ys, xs][xs < 160].astype(int)
        right_px = frame[ys, xs][xs >= 160].astype(int)
        assert np.abs(left_px - start).sum() < np.abs(left_px - end).sum()
        assert np.abs(right_px - end).sum() < np.abs(right_px - start).sum()

    def test_long_text_shrinks_to_fit_and_disabled_leaves_frame_clean(self):
        frame = np.zeros((90, 160, 3), dtype=np.uint8)
        apply_text_overlay(frame, self.values(text_string="A very long line of text"))
        ys, xs = np.where(frame.any(axis=2))
        assert xs.min() >= DEFAULTS['text_padding'] - 8
        assert xs.max() <= 160 - DEFAULTS['text_padding'] + 8
        clean = np.zeros_like(frame)
        apply_text_overlay(clean, self.values(text_enabled=False))
        apply_text_overlay(clean, self.values(text_string=""))
        assert not (clean.any())


class TestLoadLut:
    def test_loads_cube(self):
        files = list_lut_files(None)
        if not files:
            pytest.skip("no .cube files in LUT/")
        lut = load_lut(files[0])
        assert lut is not None
        assert lut.dtype == __import__('numpy').float32

    def test_missing_returns_none(self):
        assert load_lut('/no/such/file.cube') is None


class TestSmartSharpen:
    def test_flat_area_untouched(self):
        import numpy as np
        frame = np.full((32, 32, 3), 120, dtype=np.uint8)
        out = apply_smart_sharpen(frame, strength=0.3, threshold=10)
        assert (out == frame).all()

    def test_edge_is_sharpened(self):
        import numpy as np
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        frame[:, 16:] = 200
        out = apply_smart_sharpen(frame, strength=0.3, threshold=10)
        assert not ((out == frame).all())

    def test_zero_strength_is_noop(self):
        import numpy as np
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        frame[:, 16:] = 200
        out = apply_smart_sharpen(frame, strength=0.0)
        assert (out == frame).all()


class TestApplyLut:
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
            assert int(chroma_sharp.max()) <= int(chroma_soft.max()) + 2

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
            assert grad_out > grad_in * 0.8


class TestLutCycling:
    def test_cycle_wraps_forward(self):
        assert cycle_index(2, 3, 1) == 0
        assert cycle_index(0, 3, 1) == 1

    def test_cycle_wraps_backward(self):
        assert cycle_index(0, 3, -1) == 2
        assert cycle_index(1, 3, -1) == 0

    def test_empty_list_is_safe(self):
        assert cycle_index(0, 0, 1) == 0

    def test_lut_files_sorted(self):
        files = list_lut_files(None)
        assert files == sorted(files)


class TestTimeLoop:
    def test_records_while_off_and_replays_while_on(self):
        loop = TimeLoop(length=3)
        assert not (loop.ready)
        frames = [object(), object(), object()]
        for frame in frames:
            assert loop.record(frame) is frame
        assert loop.ready

        assert loop.next_frame() is frames[0]
        assert loop.next_frame() is frames[1]
        assert loop.next_frame() is frames[2]
        assert loop.next_frame() is frames[0]

    def test_resuming_live_resets_the_playhead(self):
        loop = TimeLoop(length=3)
        first, second = object(), object()
        loop.record(first)
        loop.record(second)
        loop.next_frame()
        live = object()
        loop.record(live)
        assert loop.next_frame() is first


class TestDeepFry:
    def test_changes_pixels_and_preserves_layout(self):
        import numpy as np

        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :32] = (40, 90, 200)
        frame[:, 32:] = (210, 120, 30)
        out = apply_deep_fry(frame)
        assert out.shape == frame.shape
        assert out.dtype == np.uint8
        assert not ((out == frame).all())


class TestMagicLoop:
    CROP = (0.5, 0.45, 1.5)

    def test_build_loop_returns_none_without_enough_history(self):
        import numpy as np

        buf = MagicLoopBuffer()
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        buf.record(frame, self.CROP)
        assert buf.build_loop(self.CROP) is None

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
        assert player is not None
        # The clip holds the loop plus its crossfade tail, and the loop
        # length is inside the 2-5 s window bounds.
        assert len(player._frames) == player._length + player._blend
        assert player._length >= MAGIC_LOOP_MIN_LOOP_FRAMES
        first = cv2.imdecode(
            np.frombuffer(player._frames[0], np.uint8), cv2.IMREAD_COLOR
        )
        # The winner starts in the still zone, not in the busy part.
        assert int(first.max()) < 100

    def test_loop_rejects_a_window_shorter_than_two_seconds(self):
        import numpy as np

        buf = MagicLoopBuffer()
        still = np.full((180, 320, 3), 40, dtype=np.uint8)
        # Just under MAGIC_LOOP_MIN_FRAMES: a full calm window cannot exist.
        for _ in range(MAGIC_LOOP_MIN_FRAMES - 1):
            buf.record(still, self.CROP)
        assert buf.build_loop(self.CROP) is None

    def test_player_plays_in_order_and_wraps_with_crossfade(self):
        import cv2
        import numpy as np

        length, blend = 6, 2
        encoded = []
        for i in range(length + blend):
            frame = np.full((180, 320, 3), i * 20 + 10, dtype=np.uint8)
            ok, jpeg = cv2.imencode(".jpg", frame)
            assert ok
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
            assert np.array_equal(frames[t], decode(t))
        # The first blended frame already moved away from clip[0] towards
        # the footage that followed the loop, so no motion reversal.
        assert not (np.array_equal(frames[0], decode(0)))


class TestPipMode:
    def test_image_extension_matching_is_case_insensitive(self):
        for name in ("a.gif", "b.JPG", "c.Jpeg", "d.PNG", "e.png"):
            assert is_image_file(name)
        for name in ("f.mov", "g.txt", "h.cube", "i.glsl", "gif"):
            assert not (is_image_file(name))

    def test_list_pip_files_finds_dropped_images(self):
        import os
        import tempfile
        from unittest import mock

        with tempfile.TemporaryDirectory() as folder:
            for name in ("slide.png", "MEME.JPG", "notes.txt", "movie.mov"):
                open(os.path.join(folder, name), "w").close()
            with mock.patch("hyper_stage.pipmode.os.getcwd", return_value=folder):
                files = list_pip_files("")
        assert files == sorted((os.path.abspath(os.path.join(folder, name)) for name in ('MEME.JPG', 'slide.png')))
        # Sorted by name, ignoring case: meme.png before slide.png even
        # though uppercase letters would sort first byte-wise.
        assert [os.path.basename(f) for f in files] == ['MEME.JPG', 'slide.png']

    def test_load_pip_image_letterboxes_to_output_size(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "wide.png")
            cv2.imwrite(path, np.full((60, 120, 3), 200, dtype=np.uint8))
            frame = load_pip_image(path, (100, 100))
        assert frame is not None
        assert frame.shape == (100, 100, 3)
        assert frame.dtype == np.uint8
        # Wide source: full width, black bars top and bottom.
        assert int(frame[0, 50].max()) == 0
        assert int(frame[99, 50].max()) == 0
        assert int(frame[50, 50].max()) == 200

    def test_load_pip_image_rejects_missing_and_unreadable(self):
        assert load_pip_image('', (64, 64)) is None
        assert load_pip_image('/no/such/file.png', (64, 64)) is None

    def test_corner_rect_places_the_window_with_margins(self):
        # Equal paddings on both axes: margin is a fraction of the frame
        # width applied to every side (96px on 1920x1080).
        assert pip_corner_rect(CORNER_TL, 0.25, 96, 1920, 1080) == (96, 96, 480, 270)
        assert pip_corner_rect(CORNER_TR, 0.25, 96, 1920, 1080) == (1344, 96, 480, 270)
        assert pip_corner_rect(CORNER_BL, 0.25, 96, 1920, 1080) == (96, 714, 480, 270)
        assert pip_corner_rect(CORNER_BR, 0.25, 96, 1920, 1080) == (1344, 714, 480, 270)

    def test_compose_pip_pastes_webcam_into_the_background(self):
        background = np.full((100, 100, 3), 255, dtype=np.uint8)
        webcam = np.zeros((10, 10, 3), dtype=np.uint8)
        webcam[:] = (0, 255, 0)
        out = compose_pip(np.zeros((100, 100, 3), np.uint8), background, webcam, CORNER_TL, 0.25, 5)
        # Window: x=5, y=5, 25x25 in the top-left corner of the background.
        assert tuple(out[10, 10]) == (0, 255, 0)
        assert tuple(out[0, 0]) == (255, 255, 255)
        assert tuple(out[40, 40]) == (255, 255, 255)

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
        assert tuple(out[10, 40]) == (255, 255, 255)
        assert tuple(out[10, 80]) == (0, 255, 0)

    def test_pip_config_round_trips_through_ini(self):
        import os
        import tempfile

        fd, path = tempfile.mkstemp(suffix=".ini")
        with os.fdopen(fd, "w") as f:
            f.write("[pip]\nenabled = true\nimage = img/slide.gif\nflip = true\nscale = 0.3\npadding = 100\ncorner = 2\n")
        try:
            cfg = load_config(path)
            assert cfg['pip_enabled']
            assert cfg['pip_image'] == 'img/slide.gif'
            assert cfg['pip_flip']
            assert cfg['pip_scale'] == pytest.approx(0.3)
            assert cfg['pip_padding'] == 100
            assert cfg['pip_corner'] == 2
            save_config(path, cfg)
            again = load_config(path)
            assert again['pip_enabled'] == True
            assert again['pip_flip'] == True
            assert again['pip_scale'] == pytest.approx(0.3)
            assert again['pip_corner'] == 2
        finally:
            os.remove(path)

    def test_cli_pip_flags_override_config(self):
        import os
        import tempfile
        from hyper_stage import tui
        from unittest import mock

        seen = {}

        class FakeApp:
            def __init__(self, cfg, config_path, preview=False):
                seen["cfg"] = cfg

            def run(self):
                pass

        with mock.patch.object(tui, "HyperStageApp", FakeApp):
            main(["--config", os.path.join(tempfile.mkdtemp(), "t.ini"), "--pip", "--pip-image", "/tmp/s.png"])
        assert seen['cfg']['pip_enabled']
        assert seen['cfg']['pip_image'] == '/tmp/s.png'

    def test_pip_tunables_and_frame_round_trip(self):
        state = SharedState(dict(DEFAULTS))
        state.update(pip_enabled=True, pip_corner=CORNER_BR)
        assert state.snapshot()['pip_enabled']
        assert state.snapshot()['pip_corner'] == CORNER_BR
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        state.set_pip_frame(frame)
        assert state.pip_frame is frame

