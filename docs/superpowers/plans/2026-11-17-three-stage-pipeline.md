# Three-Stage Concurrent Video Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single synchronous video thread with independent capture, detector and renderer threads using single-slot mailboxes, immutable snapshots and a monotonic render scheduler.

**Architecture:** `framebus.py` holds one-slot frame/snapshot mailboxes; `capture.py`, `detector.py` and `renderer.py` each own their native resources in their own thread; `pipeline.py` becomes a `VideoPipeline` orchestrator with the same public API the TUI already uses. A shared `FrameProcessor` keeps the effect chain identical for the renderer and the baseline benchmark.

**Tech Stack:** Python 3.13, stdlib threading, OpenCV, MediaPipe Tasks, NumPy, ModernGL, pyvirtualcam, Textual, unittest.

**Spec:** `docs/specs/three-stage-pipeline.md`

## Global Constraints

- No new dependencies; standard-library threading primitives only.
- No multiprocessing; measured GIL release makes threads sufficient.
- Preserve CLI flags, INI keys, effect order, tracking behaviour and TUI controls.
- Keep the uncommitted `hyper-stage.ini` PiP edits untouched.
- No unbounded queues, no shared mutable numpy arrays, no locks held during inference/rendering.
- Test command: `.venv/bin/python -m unittest discover` (exit code 0).

## Review Focus

- Camera read failure mid-stream → capture worker keeps publishing statistics, stops cleanly after the configured consecutive-error ceiling.
- Detector exception on one frame → worker survives; telemetry surfaces the fault; no stale snapshot overwrite.
- Render falls behind → no catch-up burst, dropped ticks counted, output cadence steady.
- No new capture frame → last output is re-sent, loop players not advanced, repeats counted.
- Shutdown while a worker waits → stop event plus wake event returns promptly; every native resource is released in its owning thread.

---

### Task 1: Transport primitives and metrics

**Files:**
- Create: `framebus.py`, `metrics.py`
- Test: `test_pipeline.py`

**Interfaces:**
- Produces: `FrameEnvelope(frame, sequence, captured_at)`, `LatestFrame.publish(envelope)`, `.latest()`, `.stats()`; `TrackingSnapshot` (frozen), `LatestTracking.publish(snapshot)`, `.latest()`, `.stats()`; `RollingStats.add(v)`, `.p50()`, `.p95()`, `.fps(now)`, `.count()`.

- [ ] **Step 1: Write failing tests** for overwrite semantics, newest-frame retrieval, monotonic sequences, timestamp preservation, snapshot rejection of older sequences, and concurrent publish/read thread safety.
- [ ] **Step 2:** Run them, confirm they fail.
- [ ] **Step 3:** Implement the two modules. Each bus uses one `threading.Lock`, holds a single reference, and exposes a `wake` `threading.Event` set on publish. `TrackingSnapshot` is frozen. `RollingStats` uses a bounded `collections.deque`.
- [ ] **Step 4:** Run the tests, confirm they pass.
- [ ] **Step 5:** Commit `feat: add bounded frame and tracking mailboxes`.

### Task 2: Camera capture worker

**Files:**
- Create: `capture.py`
- Test: `test_pipeline.py`

**Interfaces:**
- Consumes: `LatestFrame`, `FrameEnvelope`, `RollingStats` from Task 1.
- Produces: `CameraWorker(cfg, bus, stop_event, on_ready, on_error)` with `start()`, `join(timeout)`, `stats()`; opens the camera in `run()`; sets `on_ready` after the first successful frame.

- [ ] **Step 1:** Write failing tests with a mock `cv2.VideoCapture`: malformed frames are rejected, failures increment stats, the bus receives immutable frames with increasing sequence and monotonic timestamps, camera is released on stop, and stats are exposed.
- [ ] **Step 2:** Confirm failure.
- [ ] **Step 3:** Implement `capture.py`. No blocking on consumers, `frame.setflags(write=False)` before publish, `time.monotonic()` timestamp, bounded consecutive-error handling.
- [ ] **Step 4:** Confirm pass.
- [ ] **Step 5:** Commit `feat: add dedicated camera capture worker`.

### Task 3: Detector worker

**Files:**
- Create: `detector.py`
- Test: `test_pipeline.py`

**Interfaces:**
- Consumes: `LatestFrame`, `LatestTracking`, `TrackingSnapshot` from Task 1; `quiet_native_stderr`, `ensure_model`, `DETECT_W/H`, `pick_subject`, `subject_center`, `face_size` from `tracking.py`.
- Produces: `DetectorWorker(cfg, state, frames, snapshots, stop_event, on_ready, on_error)` with `start()`, `join(timeout)`, `stats()`.

- [ ] **Step 1:** Write failing tests with a mock detector: respects `detect_every`, skips intermediate frames while inference is slow, stamps snapshots with the frame's timestamp/sequence, rejects late older results, survives an inference exception, and closes the detector context on stop.
- [ ] **Step 2:** Confirm failure.
- [ ] **Step 3:** Implement `detector.py` with event-driven wake, single in-flight inference in the worker thread, exception containment, and p50/p95 inference timing via `RollingStats`.
- [ ] **Step 4:** Confirm pass.
- [ ] **Step 5:** Commit `feat: add independent MediaPipe detector worker`.

### Task 4: Frame processor and renderer worker

**Files:**
- Create: `renderer.py`
- Modify: `pipeline.py` (move the existing loop body into `FrameProcessor`)
- Test: `test_pipeline.py`

**Interfaces:**
- Consumes: all effect modules; `PositionController`, `ZoomController`, `crop_origin` from `tracking.py`.
- Produces: `FrameProcessor(cfg, state, preview)` with `render(frame, snapshot, now, dt) -> np.ndarray` and `close()`; `RendererWorker(cfg, state, frames, snapshots, processor, cam_factory, stop_event, on_ready, on_error)` with `start()`, `join(timeout)`, `stats()`.

- [ ] **Step 1:** Write failing tests: rendering proceeds while the detector is unavailable; output is re-sent (not reprocessed) when no new frame exists; a late tick drops rather than bursts; controller `observe` is called once per new snapshot and `step` with the real delta; output frame order matches the current pipeline.
- [ ] **Step 2:** Confirm failure.
- [ ] **Step 3:** Implement `FrameProcessor` by moving the existing effect chain verbatim (crop → LUT → sharpen → shader → deep-fry → loop record → PiP → text). Implement `RendererWorker` with the monotonic scheduler, per-stage stats and resource cleanup in its own thread.
- [ ] **Step 4:** Confirm pass, run the full existing suite.
- [ ] **Step 5:** Commit `feat: add dedicated rendering thread with monotonic pacing`.

### Task 5: Orchestrator, state and TUI integration

**Files:**
- Modify: `pipeline.py`, `state.py`, `tui.py`
- Test: `test_pipeline.py`

**Interfaces:**
- Consumes: workers from Tasks 2–4.
- Produces: `VideoPipeline(cfg, state, preview=False)` with `start()`, `stop()`, `join(timeout)`, `reload_shader()`, `ident`, `is_alive()`, `failed`.

- [ ] **Step 1:** Write failing tests for ordered startup, partial-startup cleanup, bounded joins, no surviving workers after shutdown, and failure surfaced through `state.telemetry()["status"]`.
- [ ] **Step 2:** Confirm failure.
- [ ] **Step 3:** Rewrite `pipeline.py` as the orchestrator with the old public API. Extend `state.py` telemetry keys (capture/render/detector fps, p50/p95, latency, drops, repeats, stale count, detection age, status). Update `tui.py` telemetry line to show capture and render FPS and worker failures; leave `on_unmount` logic valid.
- [ ] **Step 4:** Run the full suite.
- [ ] **Step 5:** Commit `refactor: orchestrate three pipeline stages with deterministic lifecycle`.

### Task 6: Benchmark, docs and verification

**Files:**
- Create: `benchmark.py`, `docs/...` (already written)
- Modify: `README.md`

- [ ] **Step 1:** Add a deterministic harness comparing the synchronous baseline (using `FrameProcessor` with inline detection) to the concurrent pipeline using mock camera/detector/virtual-camera; report average FPS, p50/p95/p99 render duration, latency and drops.
- [ ] **Step 2:** Run it and record the numbers in `docs/specs/three-stage-pipeline.md`.
- [ ] **Step 3:** Document live webcam validation steps in `README.md`.
- [ ] **Step 4:** Run `.venv/bin/python -m unittest discover`, `git diff --check`, and confirm the ini PiP edits are untouched.
- [ ] **Step 5:** Commit `perf: add pipeline benchmark and live validation instructions`.
