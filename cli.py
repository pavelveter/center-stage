"""Command-line entry point: parse flags, apply config, start the TUI."""

import argparse
import os

from config import find_config_path, load_config
from lut import resolve_lut_path
from tracking import list_cameras


def parse_args(argv=None):
    """Parse CLI flags; all values override the corresponding INI settings."""
    p = argparse.ArgumentParser(description="Hyper Stage for a webcam")
    p.add_argument("--config", help="path to INI config file")
    p.add_argument("--camera", type=int, help="camera index override")
    p.add_argument("--lut", help="path to a .cube 3D LUT (overrides ini)")
    p.add_argument("--lut-scale", type=int, help="LUT downscale factor (default 8)")
    p.add_argument("--detect-every", type=int, help="run face detection every N frames")
    p.add_argument(
        "--sharpen",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable/disable edge-aware sharpening",
    )
    p.add_argument(
        "--sharpen-strength", type=float, help="sharpening amount (default 0.3)"
    )
    p.add_argument(
        "--tracking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable/disable face tracking (off = base zoom, centered)",
    )
    p.add_argument(
        "--time-loop",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="replay the last second of processed video in a loop",
    )
    p.add_argument(
        "--deep-fry",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="apply the deep-fry saturation and RGB channel-roll effect",
    )
    p.add_argument(
        "--magic-loop",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="AFK mode: replay a seamless crossfaded loop of recent video",
    )
    p.add_argument(
        "--shader",
        help="path to a .glsl fragment shader (overrides ini)",
    )
    p.add_argument(
        "--shader-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable/disable GPU shader processing",
    )
    p.add_argument(
        "--pip",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable/disable PiP slide mode (image background, small webcam)",
    )
    p.add_argument(
        "--pip-image",
        help="path to the PiP background image (.gif/.jpg/.jpeg/.png)",
    )
    p.add_argument(
        "--preview", action="store_true", help="show a debug preview window (costs CPU)"
    )
    p.add_argument(
        "--list-cameras",
        action="store_true",
        help="list available camera indices and exit",
    )
    return p.parse_args(argv)



def main(argv=None):
    """Start the video pipeline and the Textual control panel."""
    args = parse_args(argv)

    # Diagnostics mode: enumerate devices and exit without touching the camera.
    if args.list_cameras:
        print("Available cameras:")
        list_cameras()
        return

    # Config precedence: defaults < INI file < CLI flags.
    config_path = find_config_path(args.config)
    cfg = load_config(config_path)
    if args.camera is not None:
        cfg["camera_index"] = args.camera
    if args.lut is not None:
        cfg["lut_file"] = args.lut
    if args.lut_scale is not None:
        cfg["lut_scale"] = args.lut_scale
    if args.detect_every is not None:
        cfg["detect_every"] = args.detect_every
    if args.sharpen is not None:
        cfg["sharpen_enabled"] = args.sharpen
    if args.sharpen_strength is not None:
        cfg["sharpen_strength"] = args.sharpen_strength
    if args.tracking is not None:
        cfg["tracking_enabled"] = args.tracking
    if args.time_loop is not None:
        cfg["time_loop_enabled"] = args.time_loop
    if args.deep_fry is not None:
        cfg["deep_fry_enabled"] = args.deep_fry
    if args.magic_loop is not None:
        cfg["magic_loop_enabled"] = args.magic_loop
    if args.shader is not None:
        cfg["shader_file"] = args.shader
    if args.shader_enabled is not None:
        cfg["shader_enabled"] = args.shader_enabled
    if args.pip is not None:
        cfg["pip_enabled"] = args.pip
    if args.pip_image is not None:
        cfg["pip_image"] = args.pip_image

    base_dir = (
        os.path.dirname(os.path.abspath(config_path)) if config_path else os.getcwd()
    )
    # Keep the configured LUT resolvable from the config folder or the working
    # directory; the TUI lists it and the background worker bakes from it.
    cfg["lut_file"] = resolve_lut_path(cfg["lut_file"], base_dir) or ""
    # The same path resolution applies to the PiP image (config folder, cwd,
    # or absolute); load_pip_image reports a clear failure when it's missing.
    cfg["pip_image"] = resolve_lut_path(cfg["pip_image"], base_dir) or ""

    # Without an existing config, autosave creates one next to the user.
    if config_path is None:
        config_path = os.path.abspath("hyper-stage.ini")

    # Imported here so --list-cameras works even if the TUI stack is broken.
    from tui import HyperStageApp

    HyperStageApp(cfg, config_path, preview=args.preview).run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception:
        import traceback

        traceback.print_exc()
