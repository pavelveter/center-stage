# Three-stage concurrent pipeline — design

## Intent

MediaPipe inference currently runs inline in the single video thread, so every
detection stalls capture, control updates and virtual-camera output. Split the
stage into three threads with explicit ownership, single-slot transport and a
monotonic render scheduler. Preserve the exact effect order, tracking behaviour,
CLI flags, INI keys and TUI controls.

## Ownership map

| Resource | Owner | Created / released |
| --- | --- | --- |
| `cv2.VideoCapture` | `CameraWorker` | in its `run()` / `finally` |
| MediaPipe `FaceDetector` (+ context) | `DetectorWorker` | in its `run()` / `finally` |
| ModernGL context, shaders, textures, FBOs | `FrameProcessor` inside `RendererWorker` | lazily in render thread / `close()` in render thread |
| `pyvirtualcam.Camera` | `RendererWorker` | in its `run()` / `finally` |
| `PositionController`, `ZoomController`, Time/Magic Loop, PiP cache | `FrameProcessor` inside `RendererWorker` | render thread only |
| Tunables + telemetry | `SharedState` (one small lock) | TUI writes, workers read; workers publish telemetry |
| Frame / snapshot mailboxes | `framebus.py` (one small lock each) | producer publishes, consumers read immutable values |

No resource crosses a thread boundary. The TUI never joins a worker on its event
loop (the existing `on_unmount` join is bounded to 5 s and happens during exit).

## Copy and timestamp semantics

* Capture makes `cap.read()`'s array non-writeable and publishes it inside an
  immutable `FrameEnvelope` (sequence, `time.monotonic()` capture timestamp).
  This is a zero-copy handoff; consumers must never write to it. Every consumer
  creates a new array whenever it needs to transform the frame (`cv2.flip`,
  `cv2.resize`, `np.ascontiguousarray` on a crop all return fresh buffers).
* The detector downscales the envelope to 640x360; it never touches full-res data.
* `TrackingSnapshot` is a frozen dataclass carrying centre, size, confidence,
  capture timestamp/sequence, completion timestamp and validity. The snapshot
  bus only accepts monotonically newer sequence numbers, so a late result from
  an older frame cannot overwrite a newer accepted result.

## Scheduling

* Producer publishes every successful read and never waits for a consumer. A
  single-slot bus means an unread frame is overwritten (counted as a drop).
* Detector waits on an event signalled by the producer; after each inference it
  re-reads the bus, so intermediate frames are skipped rather than queued. It
  starts inference only when `frame.sequence - last_inference >= detect_every`.
* Renderer runs on a monotonic deadline schedule (`1/cfg["fps"]`). It waits on
  the stop event for the remaining time (no busy wait). A tick later than one
  full interval resets the deadline to `now + interval` and counts a dropped
  tick — no catch-up burst. When no newer frame exists, the last published
  output frame is re-sent to the virtual camera and counted as a repeat; loop
  players are not advanced on a repeat. `cam.sleep_until_next_frame()` is
  removed so there is a single pacing authority.
* Controllers `observe()` only on a snapshot with a previously unconsumed
  sequence; they `step()` every rendered frame with the real render delta.
  Detection age is derived from the snapshot's *capture* timestamp, so the
  existing timeout behaviour survives the decoupling.

## GIL / native-thread findings (measured on this M1, 2026-11-17)

* 40x `cv2.resize(2560x1440 -> 640x360)`: a competing Python counter ran at
  95% of its idle rate — OpenCV releases the GIL.
* 30x MediaPipe `FaceDetector.detect` with the cached BlazeFace model: ~28.4 ms
  per inference; the counter ran at 99% of its idle rate — MediaPipe releases
  the GIL. Thread-based overlap is real; no multiprocessing is warranted.
* `cv2.setNumThreads(1)` is already set in the app; no native thread counts are
  changed. NumPy ufuncs and ModernGL calls release the GIL in their C layers.

## Decision: benchmark environment

The automated benchmark uses a mock camera feeding synthetic 2560x1440 frames
and a mock virtual camera, so it runs in CI with no hardware. Live webcam
validation instructions are a deliverable, not an automated test.

## Benchmark results (`.venv/bin/python benchmark.py --frames 240`)

Same synthetic 2560x1440 input, 1920x1080 output, sharpening on, 30 fps target,
`detect_every=6`, mocked 28 ms inference. Representative run:

| Metric | Synchronous baseline | Three-stage |
| --- | --- | --- |
| Average output FPS | 16.13 | 29.33 |
| Frame interval p50 / p95 / p99 (ms) | 56.3 / 95.2 / 98.4 | 33.4 / 50.2 / 62.0 |
| Render duration p50 / p95 / p99 (ms) | 17.8 / 25.0 / 27.8 | 12.8 / 13.9 / 15.1 |
| Capture-to-output latency p50 (ms) | 17.9 | 2.9 |
| Inferences | 40 | 40 |

The primary goal — reduced jitter and predictable pacing — is met: median
interval moves from ~56 ms to ~33 ms and the p99 spread collapses from ~42 ms
to ~29 ms around the median. Render time is also lower because inference no
longer sits inside the render call. Results vary with host load; rerun the
benchmark on the target machine for a local baseline.
