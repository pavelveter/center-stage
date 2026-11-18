# MIDI Controller Integration — design / spec

## Intent

Let external MIDI hardware control every applicable TUI control of hyper-stage in
real time, and reflect application state back onto controller LEDs where the
hardware supports it. MIDI is an optional input/output subsystem: the app must
start and behave exactly as before with no MIDI backend, no ports and no config.

## Goals

- A declarative, single authoritative registry of MIDI targets.
- Hardware-independent mapping engine: CC, Note, absolute, relative encoders
  (two's complement, binary offset, `0/127`), toggle, momentary, next/prev.
- `SharedState` stays the single source of truth; MIDI and TUI call the *same*
  application methods.
- Bi-directional feedback with throttling, de-duplication and loop prevention.
- Device discovery, graceful missing-device behaviour, hot-plug / reconnect,
  clean shutdown.
- Hardware-free tests; documented in README.

## Non-goals / anti-goals

- No general-purpose MIDI framework or MIDI file playback.
- No arbitrary text entry over MIDI (text stays a TUI input).
- No default shutdown mapping; no native file picker driven by a MIDI value.
- No synchronous hardware I/O on the Textual event loop or video threads.

## Constraints

- Textual 8.2.8; `call_from_thread` is the supported cross-thread dispatch API
  and is already used by the LUT/PiP workers.
- `mido` callback runs in a foreign thread (verified in mido docs).
- Existing config save is atomic, comment-preserving and keyed by `INI_MAP`.
- The checked-in `hyper-stage.ini` is a user modification and is not edited.

## Architecture

| Module | Responsibility |
|---|---|
| `midi_mapping.py` | Single `TARGETS` registry (also source of `SWITCHES`/`TUNING`), mapping grammar, validation, pure decoder/router, edge detection, feedback message generation. No hardware imports. |
| `midi.py` | `MidiManager`: mido port lifecycle, bounded in/out queues, dispatcher + output + monitor threads, learning, status callbacks. Hardware optional. |
| `tui.py` | Thin hooks: route decoded events into existing actions, publish feedback, minimal Learn workflow, status notices. |
| `config.py` | `[midi]` defaults, scalar INI keys, mapping keys as bare INI options (never JSON), validation on load, persistence on save. |
| `cli.py` | `--list-midi` diagnostic and `--midi/--no-midi` override, mirroring camera flags. |

### Target registry (authoritative inventory)

- 11 switches: `tracking_enabled`, `cam_flip`, `text_enabled`, `lut_enabled`,
  `shader_enabled`, `sharpen_enabled`, `time_loop_enabled`,
  `magic_loop_enabled`, `deep_fry_enabled`, `pip_enabled`, `pip_flip`.
- 4 tunings: `zoom` (1–4, 0.05), `position_response_time` (0.2–3, 0.1),
  `sharpen_strength` (0–1, 0.05), `lut_scale` (2–16, 1).
- Actions: `lut_next`, `lut_prev`, `shader_next`, `shader_prev`, `pip_next`,
  `pip_prev`, `pip_index`, `pip_corner_tl/tr/bl/br`, `reload_files`,
  `pip_pick` (optional momentary), `shutdown` (opt-in only, never default).

### Mapping grammar

`type:channel:number:mode[:option...]`, with several specs for one target
separated by `|`.

- `type`: `cc` or `note`.
- `channel`: 1–16, or 0 = inherit global `[midi] channel`; global 0 = all.
- `number`: 0–127.
- `mode`: `absolute`, `toggle`, `momentary`, `trigger`, `next`, `prev`,
  `relative_twos`, `relative_binary`, `relative_0_127`.
- `option`: `invert` (direction/state inversion), `accel` (optional encoder
  acceleration, 1 step by default).
- `[midi] button_debounce` (ms, 0–1000, default 80) is a refractory window for
  edge-triggered controls: repeated press/release pairs from a bouncing button
  collapse into one action. `0` disables it.

Multiple mappings may share a target (macro), and one input message may map to
several targets. All matching mappings fire.

### Relative encoders

- `relative_twos`: 1–63 → +n, 64 → 0, 65–127 → n-128.
- `relative_binary`: 1–63 → +n, 64 → 0, 65–127 → -(n-64).
- `relative_0_127`: 0 → +1, 127 → -1, everything else ignored (never absolute).
- `invert` negates the decoded delta; `accel` multiplies step magnitude.
- Zero movement is ignored.

### Threading

mido callback → bounded queue → dispatcher thread (decode, coalesce continuous
events, batch) → `call_from_thread` → TUI actions on the event loop. Feedback is
enqueued from the TUI thread and sent by a dedicated output thread. A monitor
thread polls port availability and reconnects. All queues are bounded; burst
control drops *redundant continuous* items first but never drops button
press/release pairs.

### Feedback

Per-target last-sent value + time, throttled by `feedback_interval`, de-duped
on actual value changes only. Boolean → CC 0/127 (or Note on/off for note
mappings), tuning → 0–127, actions → short pulse. A short recent-output guard
prevents software loopback from being interpreted as input.

### MIDI Learn

`Ctrl+M` starts/cancels; `Ctrl+N`/`Ctrl+B` move through targets. The default
target is the **last user-interacted mappable TUI control**, resolved from a
registry-derived stable control-id → target table; no control yet means the UI
asks the user to pick one. Only real Textual interaction handlers update that
context: app-originated widget writes carry a per-widget counter consumed by the
handler, so programmatic updates, refreshes, MIDI traffic and feedback cannot
steal it. Non-mappable controls leave the previous target untouched. Captured
messages are consumed exclusively (never routed as controls), validated, and
converted to a mapping appropriate for the target kind. The notice shows
`MIDI LEARN → <control>` and then `MIDI LEARN → <control> → <MSG> <n> CH <ch>`.
Existing mappings are not overwritten unless the user explicitly confirms by
capturing again. 15 s timeout. Saved via existing `save_config`.

## Acceptance criteria

1. App starts unchanged without MIDI hardware/backends.
2. Every inventory target is reachable from MIDI using existing app logic.
3. MIDI ↔ TUI state stays in sync; feedback honours throttle/dedupe/loop guard.
4. Toggles are edge-triggered; momentary, note-off velocity 0 and both relative
   schemes plus `relative_0_127` behave as specified.
5. `[midi]` round-trips through the INI with comments preserved; invalid
   mappings warn and are skipped.
6. Disconnect/reconnect and shutdown are non-blocking and leak no threads.
7. `uv run pytest` passes, including existing TUI/config regression tests.
