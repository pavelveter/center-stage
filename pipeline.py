"""Apple-style Center Stage for a regular webcam.

Captures a high-res frame, tracks face(s) with MediaPipe, and publishes a
cropped/zoomed view to a virtual camera so video-call apps see a framed shot.

The video pipeline runs in a daemon thread while tui.py renders a Textual
control panel. Settings come from an INI file (see hyper-stage.ini), can be
changed live in the TUI, and are written back when they change.
"""
import math
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from effects import TimeLoop, apply_deep_fry, apply_smart_sharpen
from lut import apply_lut
from magic_loop import (
    MAGIC_LOOP_ENTRY_FADE_SECONDS,
    MAGIC_LOOP_EXIT_FADE_SECONDS,
    MAGIC_LOOP_GUARD_SECONDS,
    MagicLoopBuffer,
)
from pipmode import compose_pip, compose_pip_rect, pip_transition_rect
from shaders import MultiShader, ShaderEngine, load_shader
from state import SharedState
from tracking import (
    DETECT_H,
    DETECT_W,
    MODEL_PATH,
    crop_origin,
    ensure_model,
    face_centers,
    fit_zoom,
    largest_face_height,
    pick_target_zoom,
    quiet_native_stderr,
    spring_step,
)


class VideoPipeline(threading.Thread):
    """Capture, detect, crop, grade and publish, isolated from the TUI thread.

    The detector, capture device and virtual camera are created and used only
    inside this daemon thread, preserving their thread affinity. All shared
    values are read from SharedState, so the TUI can change settings at any time
    without blocking or dropping frames.
    """

    def __init__(self, cfg, state, preview=False):
        super().__init__(name="hyper-stage-video", daemon=True)
        self.cfg = cfg
        self.state = state
        self.preview = preview
        self._stop_event = threading.Event()
        # Created lazily inside the video thread: the GL context must live
        # and be used on one thread only.
        self._shader_engine = None

    def stop(self):
        """Ask the loop to exit; the thread finishes the current frame first."""
        self._stop_event.set()

    def reload_shader(self):
        """Drop cached shader programs; the next frame recompiles from disk."""
        engine = self._shader_engine
        if engine is not None:
            # dict.clear() is atomic enough in CPython; the video thread only
            # reads entries and recompiles missing ones. Multibuffer packs
            # release their GL objects explicitly.
            for entry in engine._programs.values():
                if isinstance(entry, MultiShader):
                    entry.close()
            engine._programs.clear()

    def run(self):
        """Open the devices, run the frame loop, and always release resources."""
        cfg = self.cfg
        cap = None
        try:
            model_path = ensure_model()
            options = vision.FaceDetectorOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                min_detection_confidence=cfg["min_det_conf"],
            )
            # Single-threaded OpenCV avoids occasional macOS crashes when its
            # worker threads compete with MediaPipe's thread pool.
            cv2.setNumThreads(1)

            cap = cv2.VideoCapture(cfg["camera_index"], cv2.CAP_AVFOUNDATION)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["in_w"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["in_h"])
            cap.set(cv2.CAP_PROP_FPS, cfg["fps"])
            if not cap.isOpened():
                self.state.set_telemetry(
                    device="none",
                    status=f"cannot open camera {cfg['camera_index']}; "
                    "run --list-cameras to see indices",
                )
                return

            # Native LiteRT logs land on the terminal during detector
            # create/close; scope the silencing to exactly those two points
            # instead of the whole pipeline so real errors stay visible.
            with quiet_native_stderr():
                detector_ctx = vision.FaceDetector.create_from_options(options)
                detector = detector_ctx.__enter__()
            try:
                with pyvirtualcam.Camera(
                    width=cfg["out_w"],
                    height=cfg["out_h"],
                    fps=int(cfg["fps"]),
                    fmt=pyvirtualcam.PixelFormat.BGR,
                ) as cam:
                    self.state.set_telemetry(device=cam.device, status="running")
                    self._loop(cap, detector, cam)
            finally:
                with quiet_native_stderr():
                    detector_ctx.__exit__(None, None, None)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            self.state.set_telemetry(status=f"stopped: {exc}")
        finally:
            if cap is not None:
                cap.release()
            if self.preview:
                cv2.destroyAllWindows()
            self.state.set_telemetry(fps=0.0, tracking=False)

    def _loop(self, cap, detector, cam):
        """Frame loop; returns when stopped, when the camera dies, or on a fault."""
        cfg = self.cfg
        out_w, out_h = cfg["out_w"], cfg["out_h"]
        dt = 1.0 / max(1.0, float(cfg["fps"]))
        omega = 2.0 * math.pi / max(0.05, float(cfg["zoom_response_time"]))
        zeta = max(0.1, float(cfg["zoom_damping_ratio"]))
        detect_every = max(1, int(cfg["detect_every"]))
        deadzone = float(cfg["deadzone"])
        zoom_deadzone = float(cfg["zoom_deadzone"])
        max_head = float(cfg["max_head_height"])
        head_ratio = float(cfg["head_ratio"])

        curr_x, curr_y = 0.5, 0.45
        target_x, target_y = curr_x, curr_y
        curr_zoom = target_zoom = max(1.0, float(cfg["zoom"]))
        zoom_vel = 0.0
        frame_idx = 0
        consecutive_errors = 0
        fps_ema = 0.0
        last_time = time.monotonic()
        time_loop = TimeLoop()
        magic_loop = MagicLoopBuffer()
        magic_player = None
        magic_on_prev = False
        # Crop (x, y, zoom) held while Magic Loop is on, and the history
        # position of the key press (minus the guard) that bounds the loop.
        magic_frozen = None
        magic_mark = 0
        magic_guard = max(1, int(cfg["fps"] * MAGIC_LOOP_GUARD_SECONDS))
        magic_entry_fade = max(1, int(cfg["fps"] * MAGIC_LOOP_ENTRY_FADE_SECONDS))
        magic_entry_left = 0
        magic_exit_fade = max(1, int(cfg["fps"] * MAGIC_LOOP_EXIT_FADE_SECONDS))
        magic_exit_left = 0
        magic_exit_player = None
        # PiP background in its display orientation: re-mirrored only when the
        # image or the flip flag changes, never per frame.
        pip_display = None
        pip_display_src = None
        pip_flip_prev = None
        # PiP zoom animation: 1.0 is the settled corner window (PiP on),
        # 0.0 the full-frame webcam (PiP off); it eases toward its target
        # over cfg["pip_animation"] seconds, reversing smoothly mid-flight.
        # A zero/negative duration snaps instantly.
        pip_anim_progress = 1.0 if cfg["pip_enabled"] else 0.0
        pip_anim_ts = time.monotonic()

        while not self._stop_event.is_set():
            try:
                ret, frame = cap.read()

                # A camera can briefly hand back None or a malformed buffer.
                # Skip it rather than letting downstream OpenCV calls crash.
                if (
                    not ret
                    or frame is None
                    or not isinstance(frame, np.ndarray)
                    or frame.ndim != 3
                    or frame.shape[2] != 3
                ):
                    consecutive_errors += 1
                    if consecutive_errors > 30:
                        self.state.set_telemetry(
                            status="camera stopped delivering frames"
                        )
                        return
                    continue
                frame = np.ascontiguousarray(frame)
                consecutive_errors = 0
                state = self.state.snapshot()
                if state["cam_flip"]:
                    # Mirror the raw feed first, so tracking, the crop and
                    # the PiP window all operate on the published view.
                    frame = cv2.flip(frame, 1)
                base_zoom = max(1.0, float(state["zoom"]))

                magic_on = bool(state["magic_loop_enabled"])
                if magic_on != magic_on_prev:
                    magic_on_prev = magic_on
                    if magic_on:
                        # Freeze: tracking stops and the crop stays exactly
                        # where it is. The history was rendered with this
                        # framing, so any piece of it can be spliced into the
                        # live view without a pan or zoom jump.
                        if state["tracking_enabled"]:
                            self.state.update(tracking_enabled=False)
                            state["tracking_enabled"] = False
                        magic_frozen = (curr_x, curr_y, curr_zoom)
                        zoom_vel = 0.0
                        # Only footage from before the person reached for the
                        # key may be used; what comes after is not trusted.
                        magic_mark = magic_loop.count - magic_guard
                        magic_player = None
                        magic_entry_left = 0
                        magic_exit_left = 0
                        magic_exit_player = None
                    else:
                        # Dissolve back to live and drop the old history: it
                        # now shows the person leaving or an empty chair.
                        magic_exit_player = magic_player
                        magic_exit_left = (
                            magic_exit_fade if magic_player is not None else 0
                        )
                        magic_player = None
                        magic_entry_left = 0
                        magic_frozen = None
                        magic_loop.clear()

                if magic_on and magic_player is None:
                    # The loop is picked the moment the key is pressed, so the
                    # audience never sees the person leaving. Without enough
                    # history before the press (e.g. started with
                    # --magic-loop) keep the live feed and retry every frame
                    # with whatever has been recorded, minus the guard.
                    player = magic_loop.build_loop(magic_frozen, upto=magic_mark)
                    if player is None:
                        player = magic_loop.build_loop(
                            magic_frozen, upto=magic_loop.count - magic_guard
                        )
                    if player is not None:
                        magic_player = player
                        magic_entry_left = magic_entry_fade

                # The plain Time Loop replay skips the whole image pipeline.
                # Magic Loop keeps it running (crop frozen) until its loop is
                # chosen, and the live feed still dissolves in and out of it.
                looping = (
                    bool(state["time_loop_enabled"])
                    and time_loop.ready
                    and not magic_on
                )
                # Once the entry dissolve is over, the loop replaces the live
                # view completely. Detection, crop, grading, sharpening and
                # recording would all be thrown away, and an AFK call can run
                # for hours: skip them and just play the loop.
                loop_covers_live = magic_player is not None and magic_entry_left <= 0
                if looping:
                    out_frame = time_loop.next_frame()
                elif loop_covers_live:
                    out_frame = None
                else:
                    if magic_on:
                        # Magic Loop holds the crop exactly as frozen at the
                        # key press: no detection, no pan, no zoom spring.
                        self.state.set_telemetry(tracking=False)
                    else:
                        if not state["tracking_enabled"]:
                            # Disabled: ease the view back to the base framing.
                            target_x, target_y, target_zoom = 0.5, 0.45, base_zoom
                            self.state.set_telemetry(tracking=False)
                        elif frame_idx % detect_every == 0:
                            small = cv2.resize(frame, (DETECT_W, DETECT_H))
                            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                            result = detector.detect(mp_image)
                            centers = face_centers(result.detections, DETECT_W, DETECT_H)
                            if centers:
                                tx = sum(c[0] for c in centers) / len(centers)
                                ty = sum(c[1] for c in centers) / len(centers)
                                if abs(tx - curr_x) > deadzone:
                                    target_x = tx
                                if abs(ty - curr_y) > deadzone:
                                    target_y = ty
                                wanted = fit_zoom(
                                    largest_face_height(result.detections, DETECT_H),
                                    base_zoom,
                                    max_head,
                                    head_ratio,
                                )
                                target_zoom = pick_target_zoom(
                                    target_zoom, wanted, zoom_deadzone
                                )
                            self.state.set_telemetry(tracking=bool(centers))

                        # Exponential moving average smooths panning even though
                        # detection only updates every detect_every frames.
                        smoothing = float(state["smoothing"])
                        curr_x += (target_x - curr_x) * smoothing
                        curr_y += (target_y - curr_y) * smoothing

                        # Dolly zoom: damped spring, then clamped to the valid range.
                        curr_zoom, zoom_vel = spring_step(
                            curr_zoom,
                            zoom_vel,
                            max(1.0, min(base_zoom, target_zoom)),
                            omega,
                            zeta,
                            dt,
                        )
                        curr_zoom = max(1.0, min(base_zoom, curr_zoom))

                    # Use actual frame dimensions: the camera may negotiate a
                    # different resolution than requested.
                    fh, fw = frame.shape[:2]
                    x1, y1, crop_w, crop_h = crop_origin(
                        curr_x, curr_y, curr_zoom, fw, fh
                    )
                    cropped = frame[y1 : y1 + crop_h, x1 : x1 + crop_w]
                    if crop_w < 1 or crop_h < 1 or cropped.size == 0:
                        continue
                    out_frame = cv2.resize(
                        np.ascontiguousarray(cropped),
                        (out_w, out_h),
                        interpolation=cv2.INTER_LINEAR,
                    )

                    table = self.state.lut_table
                    if state["lut_enabled"] and table is not None:
                        out_frame = apply_lut(
                            out_frame,
                            table,
                            scale=int(state["lut_scale"]),
                            detail=bool(cfg["lut_detail"]),
                        )
                    strength = float(state["sharpen_strength"])
                    if state["sharpen_enabled"] and strength > 0:
                        out_frame = apply_smart_sharpen(
                            out_frame,
                            strength=strength,
                            threshold=int(cfg["sharpen_threshold"]),
                            sigma=float(cfg["sharpen_sigma"]),
                        )
                    if state["shader_enabled"] and state["shader_file"]:
                        if self._shader_engine is None:
                            self._shader_engine = ShaderEngine()
                        out_frame = self._shader_engine.apply(
                            out_frame, state["shader_file"]
                        )
                    if state["deep_fry_enabled"]:
                        out_frame = apply_deep_fry(out_frame)

                    # Feed both replay buffers with the pure processed frame;
                    # blending below must never leak into recorded history.
                    out_frame = time_loop.record(out_frame)
                    # The ~2 ms encode fits the frame budget at 30 fps. Once a
                    # loop is playing, recording stops: it would only capture
                    # the empty chair, and the history is cleared on exit.
                    if magic_player is None:
                        magic_loop.record(out_frame, (curr_x, curr_y, curr_zoom))

                if magic_player is not None:
                    # Seamless continuous playback: the crossfade seam is
                    # inside the player; only the entry dissolves from live.
                    loop_frame = magic_player.next_frame((out_w, out_h))
                    if magic_entry_left > 0:
                        alpha = 1.0 - magic_entry_left / magic_entry_fade
                        out_frame = cv2.addWeighted(
                            out_frame, 1.0 - alpha, loop_frame, alpha, 0.0
                        )
                        magic_entry_left -= 1
                    else:
                        out_frame = loop_frame
                elif magic_exit_left > 0:
                    # Magic Loop was just switched off: the loop keeps
                    # running underneath while it dissolves into the live feed.
                    loop_frame = magic_exit_player.next_frame((out_w, out_h))
                    alpha = magic_exit_left / magic_exit_fade
                    out_frame = cv2.addWeighted(
                        out_frame, 1.0 - alpha, loop_frame, alpha, 0.0
                    )
                    magic_exit_left -= 1
                    if magic_exit_left == 0:
                        magic_exit_player = None

                # PiP slide mode composites last: LUTs, shaders and sharpening
                # grade the webcam feed, never the static background image.
                # Toggling PiP zooms the webcam window to/from the corner
                # over the configured pip_animation seconds instead of cutting.
                now_ts = time.monotonic()
                anim_step = (now_ts - pip_anim_ts) / max(
                    float(cfg["pip_animation"]), 0.001
                )
                pip_anim_ts = now_ts
                pip_target = 1.0 if state["pip_enabled"] else 0.0
                if pip_anim_progress < pip_target:
                    pip_anim_progress = min(pip_anim_progress + anim_step, pip_target)
                elif pip_anim_progress > pip_target:
                    pip_anim_progress = max(pip_anim_progress - anim_step, pip_target)
                background = self.state.pip_frame
                if pip_anim_progress > 0.0 and background is not None:
                    flip = bool(state["pip_flip"])
                    if background is not pip_display_src or flip != pip_flip_prev:
                        pip_display = cv2.flip(background, 1) if flip else background
                        pip_display_src = background
                        pip_flip_prev = flip
                    if pip_anim_progress >= 1.0:
                        out_frame = compose_pip(
                            out_frame,
                            pip_display,
                            out_frame,
                            state["pip_corner"],
                            float(cfg["pip_scale"]),
                            int(cfg["pip_padding"]),
                        )
                    else:
                        h, w = out_frame.shape[:2]
                        rect = pip_transition_rect(
                            pip_anim_progress,
                            state["pip_corner"],
                            float(cfg["pip_scale"]),
                            int(cfg["pip_padding"]),
                            w,
                            h,
                        )
                        out_frame = compose_pip_rect(
                            out_frame, pip_display, out_frame, rect
                        )

                cam.send(out_frame)
                cam.sleep_until_next_frame()

                now = time.monotonic()
                elapsed = now - last_time
                if elapsed > 0:
                    measured = 1.0 / elapsed
                    if fps_ema == 0.0:
                        fps_ema = measured
                    else:
                        fps_ema = fps_ema * 0.9 + measured * 0.1
                last_time = now
                self.state.set_telemetry(fps=fps_ema, zoom=curr_zoom)

                if self.preview:
                    cv2.imshow("hyper-stage", out_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        return

                frame_idx += 1
            except Exception as exc:
                # Transient per-frame errors should not kill a call; only give
                # up after repeated failures, printing the stack for diagnosis.
                consecutive_errors += 1
                if consecutive_errors > 10:
                    import traceback

                    traceback.print_exc()
                    raise
                print(f"warning: frame processing error ({consecutive_errors}): {exc}")
                time.sleep(0.01)
