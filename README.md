# hyper-stage

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
| `c` | toggle LUT color grading |
| `h` | toggle GLSL shader processing |
| `s` | toggle smart sharpening |
| `l` | toggle the one-second time loop |
| `m` | toggle Magic Loop (AFK) |
| `r` | rescan `LUT/` and `shaders/` folders and reload the active files (Russian layout: `к`) |
| `f` | toggle deep fry / glitch |
| `]` / `[` | cycle to the next / previous `.cube` LUT (not shown in the footer; Russian layout: `ъ` / `х`) |
| `>` / `<` | cycle to the next / previous `.glsl` shader (not shown in the footer; Russian layout: `ю` / `б`) |

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
  - *GLSL Shader* with the `.glsl` dropdown directly below (every file found
    in `shaders/` and the working directory at startup). Switching the shader
    off dims and disables the dropdown.
  - *Smart Sharpen* with the `Strength` tuner; the tuner is disabled while
    sharpening is off.
  - *Time Loop (1s)* - replays a 30-frame ring buffer as a seamless loop.
  - *Magic Loop (AFK)* - freezes the stream into a seamless loop of the last
    ten seconds of video and locks tracking while it is on (see below).
  - *Deep Fry / Glitch* - boosted saturation/contrast with an RGB channel roll.

Picking a `.cube` file in the dropdown bakes its 128^3 lookup table in a
background worker, so the video stream never stalls.

Picking a `.glsl` file runs it on the GPU in real time via a headless
`moderngl` context that lives inside the video thread. Shaders are
Shadertoy-style: they define `void mainImage(out vec4 fragColor, in vec2
fragCoord)`, read the current BGR frame from the `texture0` sampler (also
aliased as `iChannel0`) sized by the `iResolution` uniform, with `iTime` as
the animation clock, and output processed BGR pixels. A shader that fails to
compile is reported once and the frame passes through unchanged.

Shadertoy multibuffer exports are supported too: a folder named
`<name>.glsl/` holding `Common.glsl`, `Buffer A.glsl`, ..., `Image.glsl`
appears as one shader entry. `Common` is prepended to every pass, buffers
run in file-name order (the first buffer reads the camera on `iChannel0`,
each following pass reads the previous pass output, every buffer's
`iChannel1` is its own previous frame), and `Image.glsl` produces the
visible frame (reading the last buffer on `iChannel0`). Buffer targets are
float (RGBA16F) like Shadertoy's, so passes can carry negative/fractional
data (e.g. the Lucas-Kanade optical flow pack), and the camera frame is
converted to RGB for the pack, matching Shadertoy's channel order. Buffer
textures carry mipmaps like Shadertoy's, so passes that `texelFetch` an
explicit LOD (Ben-Day-Dots downsampling) work. Exports
wired differently (e.g. `Image.glsl` reading `Buffer B` on `iChannel2`)
take a directive comment in the pass file, e.g.
`// hyper-stage: 2=Buffer B`; accepted names are `camera`, `noise`,
`font`, `feedback` (the pass's own previous frame), and buffer pass names
like `Buffer B` (its latest output). All four `iChannel` slots are declared
in both single-file shaders and pack passes; a channel a pass samples
without a directive defaults to the noise texture (unbound samplers read
garbage on macOS). Shadertoy's `iChannelResolution[4]` uniform is provided
too, sized from the bound texture of each channel. Note desktop GL, unlike
Shadertoy's WebGL, does not zero-initial local variables - Shadertoy
imports relying on that need a zero-init (see `Pixel-Art.glsl`).

Single-file imports also get `iMouse` (always zero - drag interactions are
not available), and `iChannel1` defaults to a generated RGBA noise texture
(unbound samplers read garbage on macOS, not black). Single-file shaders
are verbatim Shadertoy imports: the camera frame is converted to RGB before
the shader and back to BGR after, so hardcoded RGB palette colors (1bit's
orange/purple, smiley's ASCII art) publish with their intended hues. A
shader that expects other Shadertoy channel assets can remap channels with
a directive comment, e.g. `// hyper-stage: 0=font 1=camera` - supported
stand-ins are `camera` (the processed frame), `noise`, and `font` (a
generated 16x16 ASCII glyph atlas approximating Shadertoy's font texture).

Disabling face tracking eases the view back to the base zoom, centered at
`0.5, 0.45`, using the same spring/EMA smoothing as tracking.

## Magic Loop (AFK)

Turning on *Magic Loop* (key `m`) freezes the published video into a seamless
loop so you can step away from a call:

- Face tracking is forced off (its switch dims and locks) and the previous
  tracking state is restored when Magic Loop is switched off again.
- The pipeline keeps a rolling ten-second history (300 frames) of the processed
  output, stored JPEG-compressed at reduced size.
- Activation follows a fixed sequence: tracking is forced off instantly (the
  switch dims and locks), then a ~1.75 second settling cool-down runs while
  the live feed continues - any motion from hitting the key flushes out of
  the history before anything is looped.
- Only after settling, the history is scanned for the calmest 2-5 second
  window: each candidate is scored by its internal motion (mean difference
  between consecutive frames) plus double the difference between its last and
  first frame (the jump at the splice), so the loop plays a moment where you
  sit still without gestures or blinks at the cut.
- Engaging the mode dissolves from the live picture into the loop over half a
  second. The chosen fragment then replays continuously, restarting cleanly
  at the head - the seam was picked to match, so there is no crossfade
  ghosting - until the switch is turned off and live capture resumes.

## Auto-save

The INI file is rewritten atomically (temp file plus `os.replace`) every five
seconds when a setting changed, and once more on quit. Unknown sections and
keys already in the file are preserved.

## CLI

```sh
uv run tracker.py --list-cameras
uv run tracker.py --config /path/to/hyper-stage.ini
uv run tracker.py --camera 0 --lut LUT/tarantino-warm.cube
uv run tracker.py --no-tracking --time-loop --deep-fry
```

Other overrides: `--lut-scale`, `--detect-every`, `--sharpen/--no-sharpen`,
`--sharpen-strength`, `--magic-loop/--no-magic-loop`,
`--shader <path.glsl>`, `--shader-enabled/--no-shader-enabled`,
`--preview` (opens a debug OpenCV window as well).

## Requirements

- macOS (AVFoundation capture + virtual camera)
- A webcam; by default uses camera index 1 (FaceTime is 0)
- Virtual camera backend (OBS / pyvirtualcam's macOS camera extension)

## Tuning

`hyper-stage.ini` holds `camera`, `capture`, `tracking`, `color`, `sharpen`,
`effects`, and `appearance` sections. Values changed in the TUI are written
back there.

The `[color]` section applies a 3D `.cube` LUT. `lut_scale` downsamples during
grading for speed (2 = highest quality, higher = faster). With `lut_detail`
the graded image is upscaled and the original's luma-only detail is added
back, keeping sharpness without letting full-res chroma bypass the grade
(a B&W LUT would otherwise leak the original colors at edges and in noise).
It also holds `shader_enabled` and `shader_file` for the GPU `.glsl` effect.

The `[sharpen]` section applies an edge-aware unsharp mask: only pixels whose
local contrast exceeds `threshold` are sharpened, so skin stays smooth.
`strength` 0.2-0.4 looks natural; 0 disables the effect.

The `[effects]` section controls the time loop, magic loop, and deep fry
toggles.

The `[appearance]` section holds `theme`, the Textual color theme of the TUI.
It is applied at startup and re-saved when you change the theme from the
command palette (`ctrl+p` -> theme).

## Tests

```sh
uv run python -m unittest discover
```
