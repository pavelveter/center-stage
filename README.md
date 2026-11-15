![hyper-stage banner](banner.png)

# hyper-stage 🎥✨

**Your regular webcam, but with main-character energy.**

Tired of looking like a grainy potato on 9 AM standups? `hyper-stage` hijacks your raw webcam feed, tracks your face with Apple-style Center Stage smoothness, slaps on cinema-grade 3D LUTs and real-time GPU GLSL shaders, and pipes the glow-up straight to a virtual camera at a buttery 30 FPS.

> **Peer-reviewed telemetry fact™:** Using `hyper-stage` makes video calls **4.73% less soul-crushing** (citation needed, vibes verified).

Everything runs from a slick, full-screen Textual TUI while a background daemon thread handles the heavy lifting. Toggle filters, tweak spring physics, or peace out for coffee without dropping a single frame.

---

## ⚡ Quickstart (No Cap)

Fire up the pipeline — `start.sh` installs `uv` if missing, syncs the locked dependencies, pre-downloads the face-detector model, and launches the app:

```sh
./start.sh
```

or, with `uv` already set up:

```sh
uv run main.py
```

Hit `q` (or `Ctrl+C`) to bounce — your settings auto-save before the lights go out. The virtual camera spins up the millisecond your webcam and virtual-cam backend handshake.

---

## 🎮 Controls & Keybinds

Click literally anything in the TUI (switches, dropdowns, `-`/`+` buttons, footer hints), or speedrun your call aesthetics with keyboard shortcuts (yes, even if you forgot to switch off your ЙЦУКЕН layout):

| Key       | Ru Layout | Vibe Check / Action                                                    |
| :-------- | :-------- | :--------------------------------------------------------------------- |
| `q`       | `й`       | **Rage-quit gracefully** (saves your config first)                     |
| `t`       | `е`       | Toggle **Face Tracking** (let the camera stalk you)                    |
| `c`       | `с`       | Toggle **LUT Color Grading** (instant indie-film mood)                 |
| `h`       | `р`       | Toggle **GLSL Shaders** (GPU goes brrr)                                |
| `s`       | `ы`       | Toggle **Smart Sharpen** (crisp edges, zero skin texture crime)        |
| `l`       | `д`       | Toggle **1s Time Loop** (freeze time for comedic effect)               |
| `m`       | `ь`       | Toggle **Magic Loop (AFK)** (ultimate stealth mode, see below)         |
| `f`       | `а`       | Toggle **Deep Fry / Glitch** (when the meeting could've been an email) |
| `r`       | `к`       | Hot-reload `LUT/` and `shaders/` folders on the fly                    |
| `]` / `[` | `ъ` / `х` | Cycle next / previous `.cube` LUT                                      |
| `>` / `<` | `ю` / `б` | Cycle next / previous `.glsl` shader                                   |

---

## 🕵️ Magic Loop (Elite AFK Tech)

Need to grab an iced oat latte or pet your cat mid-sync without anyone noticing? Smash **`m`** to engage **Magic Loop**:

1. **Instant Freeze:** Face tracking locks immediately so the crop doesn't awkwardly drift while you stand up.
2. **Anti-Sus Guard:** The pipeline ignores the last ~0.8 seconds right before you pressed the key (so nobody sees your hand reaching for the keyboard or your head turning away).
3. **Vibe-Checked Seamless Loop:** It scans your rolling 10-second compressed history for the chillest 2–5 second window with the lowest motion and cleanest seam, dissolves into it over 0.4s, and loops it seamlessly until you return.
4. **Clean Re-entry:** Tap `m` again when you're back in your chair—it smoothly crossfades back to live reality and wipes the old buffer. Zero evidence.

---

## 🎨 Shaders & LUTs (GPU Drip)

- **3D LUTs (`.cube`):** Drop your `.cube` files into `LUT/`. Selecting one bakes a $128^3$ lookup table in a background worker so your stream never stutters. Luma-only detail preservation keeps your face sharp without leaking original colors into B&W grades.
- **GLSL Shaders (`.glsl`):** Drop Shadertoy-style fragment shaders (`void mainImage(out vec4 fragColor, in vec2 fragCoord)`) into `shaders/` — the repo ships a starter pack (1-bit dither, CRT, halftone, pixel-art, optical-flow and friends). Runs headless via `moderngl` on your GPU.
  - Supports **multibuffer Shadertoy packs**! Drop a folder named `<effect>.glsl/` containing `Common.glsl`, `Buffer A.glsl`, ..., `Image.glsl` with `RGBA16F` ping-pong feedback, mipmaps, and custom channel routing directives (`// hyper-stage: 2=Buffer B`).
  - Built-in stand-ins for `camera`, `noise`, `font` (16x16 ASCII glyph atlas), and `feedback`.

---

## 🛠️ Dashboard Anatomy

The TUI keeps things clean in a centered 72-column layout:

- **TELEMETRY:** Live FPS, current zoom multiplier, `TRACKING` / `IDLE` status, active virtual camera device, and status notifications.
- **CONTROLS:**
  - _Face Tracking_ + `Zoom` spring tuner
  - _Pan Smoothing_ EMA tuner
  - _LUT Grading_ + `Scale` tuner + `.cube` selector
  - _GLSL Shader_ + `.glsl` selector
  - _Smart Sharpen_ + `Strength` tuner (edge-gated so you don't look deep-fried unless you want to)
  - _Time Loop (1s)_, _Magic Loop (AFK)_, and _Deep Fry_ switches

---

## 💻 CLI Flags & Config

Everything you tweak in the UI atomically auto-saves to `hyper-stage.ini` every 5 seconds. Want to launch straight into chaos from the terminal? We got flags:

```sh
uv run main.py --list-cameras
uv run main.py --config /path/to/hyper-stage.ini
uv run main.py --camera 0 --lut LUT/tarantino-warm.cube
uv run main.py --no-tracking --time-loop --deep-fry
uv run main.py --shader shaders/crt-tv.glsl --shader-enabled --preview
```

Other flags: `--lut-scale`, `--detect-every`, `--sharpen/--no-sharpen`, `--sharpen-strength`, `--magic-loop/--no-magic-loop`.

Change your TUI color theme anytime via the Textual command palette (`Ctrl+P` → `theme`) — it saves straight to `[appearance]` in your INI file.

---

## 📦 Requirements & Tests

- **OS:** macOS (uses AVFoundation capture + virtual camera backend)
- **Hardware:** Any webcam (defaults to index `1`; built-in FaceTime cam is usually `0`)
- **Virtual Cam:** OBS Virtual Camera or `pyvirtualcam` macOS camera extension

Run the test suite to verify nothing's broken:

```sh
uv run python -m unittest discover
```

---

## 📄 License

Distributed under the **MIT License**. Fork it, remix it, slap weird GLSL shaders on your face during quarterly town halls — just don't blame us if you get promoted for "always looking so engaged."
