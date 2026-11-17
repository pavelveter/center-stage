import configparser
import os
import tempfile

import pytest

from hyper_stage.config import DEFAULTS, load_config, save_config


class TestLoadConfig:
    def test_defaults_when_missing(self):
        cfg = load_config(None)
        assert cfg['camera_index'] == 1
        assert cfg['zoom'] == 1.5

    def test_ini_override(self):
        import tempfile, os
        fd, path = tempfile.mkstemp(suffix=".ini")
        with os.fdopen(fd, "w") as f:
            f.write("[camera]\nindex = 3\nflip = true\n[tracking]\nzoom = 2.0\n")
        try:
            cfg = load_config(path)
            assert cfg['camera_index'] == 3
            assert cfg['cam_flip']
            assert cfg['zoom'] == 2.0
        finally:
            os.remove(path)

    def test_removed_legacy_keys_are_ignored(self):
        import tempfile, os
        fd, path = tempfile.mkstemp(suffix=".ini")
        with os.fdopen(fd, "w") as f:
            f.write(
                "[tracking]\nsmoothing = 0.03\ndeadzone = 0.08\n"
                "zoom_response_time = 0.5\nzoom_damping_ratio = 0.5\n"
                "position_response_time = 1.2\n"
            )
        try:
            cfg = load_config(path)
            assert cfg['position_response_time'] == 1.2
            assert cfg['position_deadband'] == DEFAULTS['position_deadband']
            for removed in ("smoothing", "deadzone", "zoom_response_time",
                            "zoom_damping_ratio"):
                assert removed not in cfg
            # save_config drops unknown keys instead of carrying them forward.
            save_config(path, cfg)
            with open(path) as saved:
                text = saved.read()
            for removed in ("smoothing =", "deadzone =", "zoom_response_time =",
                            "zoom_damping_ratio =", "margin ="):
                assert not (any((line.startswith(removed) for line in text.splitlines())))
        finally:
            os.remove(path)

    def test_text_config_round_trips_through_ini(self):
        import os
        import tempfile

        fd, path = tempfile.mkstemp(suffix=".ini")
        with os.fdopen(fd, "w") as f:
            f.write(
                "[text]\nenabled = false\ntext = Hello world\nfont_face = Nothing Font C\n"
                "font_scale = 2.0\nthickness = 4\npadding = 80\n"
                "color_start = #112233\ncolor_end = #aabbcc\n"
            )
        try:
            cfg = load_config(path)
            assert not (cfg['text_enabled'])
            assert cfg['text_string'] == 'Hello world'
            assert cfg['text_font_face'] == 'Nothing Font C'
            assert cfg['text_font_scale'] == 2.0
            assert cfg['text_thickness'] == 4
            assert cfg['text_padding'] == 80
            assert cfg['text_color_start'] == '#112233'
            assert cfg['text_color_end'] == '#aabbcc'
            save_config(path, cfg)
            again = load_config(path)
            assert again['text_padding'] == 80
            assert again['text_string'] == 'Hello world'
        finally:
            os.remove(path)

    def test_pip_animation_override(self):
        import tempfile, os
        fd, path = tempfile.mkstemp(suffix=".ini")
        with os.fdopen(fd, "w") as f:
            f.write("[pip]\nanimation = 1.5\n")
        try:
            cfg = load_config(path)
            assert cfg['pip_animation'] == 1.5
        finally:
            os.remove(path)


class TestSaveConfig:
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
            assert reloaded['camera_index'] == 2
            assert reloaded['zoom'] == 2.25
            assert not (reloaded['lut_enabled'])
            assert reloaded['time_loop_enabled']
            assert reloaded['lut_file'] == os.path.join('LUT', 'a.cube')

            parser = configparser.ConfigParser()
            parser.read(path)
            assert parser.get('custom', 'keep') == 'yes'
            assert [n for n in os.listdir(folder) if n.startswith('.hyper-stage-')] == []
            with open(path) as saved:
                text = saved.read()
            assert '; AVFoundation camera index' in text
            assert '[custom]' in text
            # Comments survive a second autosave.
            save_config(path, cfg)
            with open(path) as saved_again:
                assert saved_again.read() == text

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
            assert cfg['time_loop_enabled']
            assert cfg['deep_fry_enabled']
            assert cfg['magic_loop_enabled']
            assert cfg['shader_enabled']
            assert cfg['shader_file'] == 'sh.glsl'

    def test_theme_is_loaded_and_saved(self):
        import configparser
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "hyper-stage.ini")
            with open(path, "w") as f:
                f.write("[appearance]\ntheme = textual-light\n")
            assert load_config(path)['theme'] == 'textual-light'

            cfg = dict(DEFAULTS)
            cfg["theme"] = "solarized-light"
            save_config(path, cfg)
            parser = configparser.ConfigParser()
            parser.read(path)
            assert parser.get('appearance', 'theme') == 'solarized-light'

