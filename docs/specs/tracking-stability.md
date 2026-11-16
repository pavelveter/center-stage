# Tracking stability — diagnosis and design

## Diagnosis (reproduced numerically before changes)

1. `deadzone` is compared against the *current filtered position*, not the
   accepted target. As the view follows the face, the threshold follows with
   it, so small oscillating measurements repeatedly re-arm the target and the
   view wanders (simulated ±1.2% oscillation walked the view from 0.5 to 0.09).
2. Pan uses a per-frame EMA (`smoothing`), so its response depends on FPS and
   it has no velocity or spring behaviour.
3. Zoom uses the raw detection bbox height with no filtering or outlier
   rejection. Bbox height changes with head rotation/pitch, producing false
   zoom requests. `pick_target_zoom` applies zoom-out instantly and only
   hysteresis-guards zoom-in.
4. The zoom spring uses fixed `dt = 1/fps` instead of elapsed time: explicit
   Euler diverges at ~10 fps and reverses direction after a 200 ms stall.
5. Detections are synchronous (no stale-result overwrite), but results carry
   no capture timestamp or age. With `detect_every=6` the last measurement is
   reused forever; there is no timeout or graceful return to default framing.
6. Position averages *all* faces while zoom uses the *largest* face — two
   different subjects can drive the two controllers.

## Design

- `tracking.py` gains two independent controllers:
  - `PositionController`: hysteresis relative to the accepted target, damped
    spring with time-based omega/zeta and internally substepped integration
    (stable for large frame intervals), no zoom thresholds.
  - `ZoomController`: robust size from keypoint inter-ocular distance when
    available (bbox height fallback), rate-limited EMA of measurements,
    fit_zoom + relative deadband, separate exponential time constants for
    zoom-in (slow) and zoom-out (responsive), clamped to [1, base].
- One subject per frame (largest area × confidence) drives both controllers.
- Each detection is stamped with its capture time; missing detections beyond
  `detection_timeout` make both controllers ease toward default framing.
- Optional `tracking_debug` telemetry exposes raw/filtered center and size,
  desired/actual zoom, detection age/confidence, position velocity and state.
- Legacy ini keys (`smoothing`, `deadzone`, `zoom_response_time`,
  `zoom_damping_ratio`) keep working via migration to the new parameters.
- Performance: no extra image processing; controllers are pure numeric state,
  detection stays on the 640×360 frame.
