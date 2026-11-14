# center-stage

Apple-style Center Stage for a regular webcam: tracks your face and publishes
a cropped/zoomed view to a virtual camera.

## Run

```sh
uv run tracker.py
```

Press `q` in the preview window to quit (or Ctrl+C).

## Live controls

While running in a terminal, these single keys act immediately:

| Key | Russian layout | Action |
| --- | --- | --- |
| `]` | `ъ` | next `.cube` LUT from the folder of the configured LUT |
| `[` | `х` | previous LUT |
| `Enter` | | toggle Center Stage on/off (off = full frame, still graded) |

Keys are only read when stdin is a TTY, so running in the background or from a
service keeps the old behavior.

Find available cameras:

```sh
uv run tracker.py --list-cameras
```

Select a camera via INI (`center-stage.ini`) or CLI:

```sh
uv run tracker.py --camera 0
uv run tracker.py --config /path/to/center-stage.ini
```

## Requirements

- macOS (AVFoundation capture + virtual camera)
- A webcam; by default uses camera index 1 (FaceTime is 0)
- Virtual camera backend (OBS / pyvirtualcam's macOS camera extension)

## Tuning

Edit `center-stage.ini` (camera, capture, tracking, color, sharpen sections) or
pass `--camera` / `--lut` / `--lut-scale` / `--detect-every` / `--sharpen` /
`--sharpen-strength`.

The `[color]` section applies a 3D `.cube` LUT. `lut_scale` downsamples
during grading for speed (1 = full quality, 4 = default, higher = faster).

The `[sharpen]` section applies an edge-aware unsharp mask: only pixels whose
local contrast exceeds `threshold` are sharpened, so skin stays smooth.
`strength` 0.2-0.4 looks natural; 0 disables the effect.
"""