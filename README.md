# center-stage

Apple-style Center Stage for a regular webcam: tracks your face and publishes
a cropped/zoomed view to a virtual camera.

The full-screen Textual control panel drives a video pipeline that runs in a
background daemon thread, so the published stream keeps running at 30 FPS while
you toggle features, pick LUTs, or tune values in the terminal.

## Run

```sh
uv run tracker.py
```

Quit with `q` (or `Ctrl+C`). The virtual camera starts as soon as the video
thread has opened the camera and the virtual-camera backend.

## Keyboard and mouse

| Key | Action |
| --- | --- |
| `q` | quit (settings are saved first) |
| `t` | toggle face tracking |
| `l` | toggle LUT color grading |
| `s` | toggle smart sharpening |
| `m` | toggle the one-second time loop |
| `f` | toggle deep fry / glitch |
| `]` / `[` | cycle to the next / previous `.cube` LUT (Russian layout: `ъ` / `х`) |

Everything is clickable: switches, the LUT list, the `-`/`+` tuning buttons, and
the footer key hints. Tuning fields accept a typed value on `Enter`.

## Panels

The screen is a single centered column (max 72 columns wide) with two blocks:

- **TELEMETRY** - one line with FPS, zoom, `TRACKING`/`IDLE`, and the active
  virtual camera (or the pipeline status when something is wrong), plus a
  dimmed notice line underneath.
- **CONTROLS** - every feature on one tight row: the switch inline with its
  numeric tuner (`-`, value, `+`) on the same line:
  - *Face Tracking* with the `Zoom` tuner.
  - *Pan Smoothing* tuner.
  - *LUT Grading* with the `Scale` tuner, and the `.cube` dropdown full-width
    directly below (every file found in `LUT/` and the working directory at
    startup). Switching grading off dims and disables the dropdown and the
    scale control.
  - *Smart Sharpen* with the `Strength` tuner; the tuner is disabled while
    sharpening is off.
  - *Time Loop (1s)* - replays a 30-frame ring buffer as a seamless loop.
  - *Deep Fry / Glitch* - boosted saturation/contrast with an RGB channel roll.

Picking a `.cube` file in the dropdown bakes its 128^3 lookup table in a
background worker, so the video stream never stalls.

Disabling face tracking eases the view back to the base zoom, centered at
`0.5, 0.45`, using the same spring/EMA smoothing as tracking.

## Auto-save

The INI file is rewritten atomically (temp file plus `os.replace`) every five
seconds when a setting changed, and once more on quit. Unknown sections and
keys already in the file are preserved.

## CLI

```sh
uv run tracker.py --list-cameras
uv run tracker.py --config /path/to/center-stage.ini
uv run tracker.py --camera 0 --lut LUT/tarantino-warm.cube
uv run tracker.py --no-tracking --time-loop --deep-fry
```

Other overrides: `--lut-scale`, `--detect-every`, `--sharpen/--no-sharpen`,
`--sharpen-strength`, `--preview` (opens a debug OpenCV window as well).

## Requirements

- macOS (AVFoundation capture + virtual camera)
- A webcam; by default uses camera index 1 (FaceTime is 0)
- Virtual camera backend (OBS / pyvirtualcam's macOS camera extension)

## Tuning

`center-stage.ini` holds `camera`, `capture`, `tracking`, `color`, `sharpen`, and
`effects` sections. Values changed in the TUI are written back there.

The `[color]` section applies a 3D `.cube` LUT. `lut_scale` downsamples during
grading for speed (2 = highest quality, higher = faster).

The `[sharpen]` section applies an edge-aware unsharp mask: only pixels whose
local contrast exceeds `threshold` are sharpened, so skin stays smooth.
`strength` 0.2-0.4 looks natural; 0 disables the effect.

The `[effects]` section controls the time loop and deep fry toggles.

## Tests

```sh
uv run python -m unittest discover
```
