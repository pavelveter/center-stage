"""Rendering stage: owns the effect chain, the GL context and the virtual camera."""

import threading
import time

import cv2
import numpy as np
import pyvirtualcam

from hyper_stage.effects import TimeLoop, apply_deep_fry, apply_smart_sharpen
from hyper_stage.lut import apply_lut
from hyper_stage.magic_loop import (
    MAGIC_LOOP_ENTRY_FADE_SECONDS,
    MAGIC_LOOP_EXIT_FADE_SECONDS,
    MAGIC_LOOP_GUARD_SECONDS,
    MagicLoopBuffer,
)
from hyper_stage.metrics import RollingStats
from hyper_stage.pipmode import compose_pip, compose_pip_rect, pip_transition_rect
from hyper_stage.shaders import ShaderEngine
from hyper_stage.textoverlay import apply_text_overlay
from hyper_stage.tracking import PositionController, ZoomController, crop_origin


class FrameProcessor:
    """All per-frame processing state, owned exclusively by the renderer thread.

    The method bodies are the original pipeline effect chain, in the original
    order. Only the detection call has been removed: the caller hands in the
    newest immutable tracking snapshot instead.
    """

    def __init__(self, cfg, state, preview=False):
        self.cfg = cfg
        self.state = state
        self.preview = preview
        self.position = PositionController()
        self.zoom_ctrl = ZoomController(max(1.0, float(cfg["zoom"])))
        self.shader_engine = None

        self.time_loop = TimeLoop()
        self.magic_loop = MagicLoopBuffer()
        self.magic_player = None
        self.magic_on_prev = False
        self.magic_frozen = None
        self.magic_mark = 0
        fps = max(1, int(cfg["fps"]))
        self.magic_guard = max(1, int(fps * MAGIC_LOOP_GUARD_SECONDS))
        self.magic_entry_fade = max(1, int(fps * MAGIC_LOOP_ENTRY_FADE_SECONDS))
        self.magic_entry_left = 0
        self.magic_exit_fade = max(1, int(fps * MAGIC_LOOP_EXIT_FADE_SECONDS))
        self.magic_exit_left = 0
        self.magic_exit_player = None

        self.pip_display = None
        self.pip_display_src = None
        self.pip_flip_prev = None
        self.pip_anim_progress = 1.0 if cfg["pip_enabled"] else 0.0
        self.pip_anim_ts = time.monotonic()

    def observe(self, snapshot, values):
        """Feed one *new* measurement into the controllers (called once only)."""
        if not snapshot.valid:
            return
        self.position.observe(
            snapshot.center_x, snapshot.center_y, snapshot.captured_at,
            float(values["position_deadband"]), snapshot.confidence,
        )
        self.zoom_ctrl.observe(
            snapshot.size, snapshot.captured_at,
            float(values["zoom_measurement_smoothing"]),
            float(values["max_measurement_change"]), snapshot.confidence,
        )
        self.state.set_telemetry(tracking=True)

    def step(self, dt, now, values):
        """Advance springs every rendered frame, using the real render delta."""
        cfg = self.cfg
        if not values["tracking_enabled"]:
            self.position.relax()
            self.zoom_ctrl.relax()
            self.state.set_telemetry(tracking=False)
        timeout = float(values["detection_timeout"])
        curr_x, curr_y = self.position.step(
            dt, now, float(values["position_response_time"]),
            float(values["position_damping"]), timeout,
        )
        curr_zoom = self.zoom_ctrl.step(
            dt, now, max(1.0, float(values["zoom"])),
            float(cfg["max_head_height"]), float(cfg["head_ratio"]),
            float(values["zoom_deadzone"]),
            float(values["zoom_in_response_time"]),
            float(values["zoom_out_response_time"]), timeout,
        )
        if cfg["tracking_debug"]:
            self.state.set_telemetry(
                raw_center=(self.position.raw_x, self.position.raw_y),
                filtered_center=(self.position.x, self.position.y),
                raw_height=self.zoom_ctrl.raw_size,
                filtered_height=self.zoom_ctrl.filtered_size,
                desired_zoom=self.zoom_ctrl.desired,
                actual_zoom=self.zoom_ctrl.value,
                detection_age=now - max(
                    self.position.last_detection, self.zoom_ctrl.last_detection
                ),
                confidence=self.position.confidence,
                pos_velocity=(self.position.vx, self.position.vy),
                tracking_state=(
                    "detected"
                    if now - self.position.last_detection <= timeout
                    else "lost"
                ),
            )
        return curr_x, curr_y, curr_zoom

    def render(self, frame, values, now, curr_x, curr_y, curr_zoom):
        """Run the complete effect chain on one captured frame."""
        cfg = self.cfg
        out_w, out_h = cfg["out_w"], cfg["out_h"]
        if values["cam_flip"]:
            # Mirror the raw feed first, so tracking, crop and PiP all operate
            # on the published view (the detector mirrors its downscale too).
            frame = cv2.flip(frame, 1)

        magic_on = bool(values["magic_loop_enabled"])
        if magic_on != self.magic_on_prev:
            self.magic_on_prev = magic_on
            if magic_on:
                if values["tracking_enabled"]:
                    self.state.update(tracking_enabled=False)
                    values["tracking_enabled"] = False
                self.magic_frozen = (self.position.x, self.position.y,
                                     self.zoom_ctrl.value)
                self.position.relax()
                self.zoom_ctrl.relax()
                self.magic_mark = self.magic_loop.count - self.magic_guard
                self.magic_player = None
                self.magic_entry_left = 0
                self.magic_exit_left = 0
                self.magic_exit_player = None
                self.state.set_telemetry(tracking=False)
            else:
                self.magic_exit_player = self.magic_player
                self.magic_exit_left = (
                    self.magic_exit_fade if self.magic_player is not None else 0
                )
                self.magic_player = None
                self.magic_entry_left = 0
                self.magic_frozen = None
                self.magic_loop.clear()

        if magic_on and self.magic_player is None:
            player = self.magic_loop.build_loop(self.magic_frozen,
                                                upto=self.magic_mark)
            if player is None:
                player = self.magic_loop.build_loop(
                    self.magic_frozen, upto=self.magic_loop.count - self.magic_guard
                )
            if player is not None:
                self.magic_player = player
                self.magic_entry_left = self.magic_entry_fade

        looping = (
            bool(values["time_loop_enabled"])
            and self.time_loop.ready
            and not magic_on
        )
        loop_covers_live = self.magic_player is not None and self.magic_entry_left <= 0
        if looping:
            out_frame = self.time_loop.next_frame()
        elif loop_covers_live:
            out_frame = None
        else:
            fh, fw = frame.shape[:2]
            x1, y1, crop_w, crop_h = crop_origin(
                curr_x, curr_y, curr_zoom, fw, fh, out_w, out_h
            )
            cropped = frame[y1 : y1 + crop_h, x1 : x1 + crop_w]
            if crop_w < 1 or crop_h < 1 or cropped.size == 0:
                return None
            out_frame = cv2.resize(
                np.ascontiguousarray(cropped), (out_w, out_h),
                interpolation=cv2.INTER_LINEAR,
            )

            table = self.state.lut_table
            if values["lut_enabled"] and table is not None:
                out_frame = apply_lut(
                    out_frame, table, scale=int(values["lut_scale"]),
                    detail=bool(cfg["lut_detail"]),
                )
            strength = float(values["sharpen_strength"])
            if values["sharpen_enabled"] and strength > 0:
                out_frame = apply_smart_sharpen(
                    out_frame, strength=strength,
                    threshold=int(cfg["sharpen_threshold"]),
                    sigma=float(cfg["sharpen_sigma"]),
                )
            if values["shader_enabled"] and values["shader_file"]:
                if self.shader_engine is None:
                    self.shader_engine = ShaderEngine()
                out_frame = self.shader_engine.apply(out_frame, values["shader_file"])
            if values["deep_fry_enabled"]:
                out_frame = apply_deep_fry(out_frame)

            out_frame = self.time_loop.record(out_frame)
            if self.magic_player is None:
                self.magic_loop.record(out_frame, (curr_x, curr_y, curr_zoom))

        if self.magic_player is not None:
            loop_frame = self.magic_player.next_frame((out_w, out_h))
            if self.magic_entry_left > 0:
                alpha = 1.0 - self.magic_entry_left / self.magic_entry_fade
                out_frame = cv2.addWeighted(
                    out_frame, 1.0 - alpha, loop_frame, alpha, 0.0
                )
                self.magic_entry_left -= 1
            else:
                out_frame = loop_frame
        elif self.magic_exit_left > 0:
            loop_frame = self.magic_exit_player.next_frame((out_w, out_h))
            alpha = self.magic_exit_left / self.magic_exit_fade
            out_frame = cv2.addWeighted(
                out_frame, 1.0 - alpha, loop_frame, alpha, 0.0
            )
            self.magic_exit_left -= 1
            if self.magic_exit_left == 0:
                self.magic_exit_player = None

        now_ts = time.monotonic()
        anim_step = (now_ts - self.pip_anim_ts) / max(
            float(cfg["pip_animation"]), 0.001
        )
        self.pip_anim_ts = now_ts
        pip_target = 1.0 if values["pip_enabled"] else 0.0
        if self.pip_anim_progress < pip_target:
            self.pip_anim_progress = min(self.pip_anim_progress + anim_step, pip_target)
        elif self.pip_anim_progress > pip_target:
            self.pip_anim_progress = max(self.pip_anim_progress - anim_step, pip_target)
        background = self.state.pip_frame
        if self.pip_anim_progress > 0.0 and background is not None:
            flip = bool(values["pip_flip"])
            if background is not self.pip_display_src or flip != self.pip_flip_prev:
                self.pip_display = cv2.flip(background, 1) if flip else background
                self.pip_display_src = background
                self.pip_flip_prev = flip
            if self.pip_anim_progress >= 1.0:
                out_frame = compose_pip(
                    out_frame, self.pip_display, out_frame,
                    values["pip_corner"], float(cfg["pip_scale"]),
                    int(cfg["pip_padding"]),
                )
            else:
                h, w = out_frame.shape[:2]
                rect = pip_transition_rect(
                    self.pip_anim_progress, values["pip_corner"],
                    float(cfg["pip_scale"]), int(cfg["pip_padding"]), w, h,
                )
                out_frame = compose_pip_rect(
                    out_frame, self.pip_display, out_frame, rect
                )

        apply_text_overlay(out_frame, values)
        return out_frame

    def reload_shader(self):
        """Drop cached shader programs; the next frame recompiles from disk."""
        if self.shader_engine is not None:
            self.shader_engine.reload()

    def close(self):
        """Release GL resources in the rendering thread that created them."""
        if self.shader_engine is not None:
            self.shader_engine.release()
            self.shader_engine = None


class RendererWorker(threading.Thread):
    """Consumer thread: monotonic frame pacing, virtual camera and GL affinity."""

    def __init__(self, cfg, state, frames, snapshots, processor, stop_event,
                 on_ready=None, on_error=None, cam_factory=None,
                 reload_event=None):
        super().__init__(name="hyper-stage-renderer", daemon=True)
        self.cfg = cfg
        self.reload_event = reload_event
        self.state = state
        self.frames = frames
        self.snapshots = snapshots
        self.processor = processor
        self.stop_event = stop_event
        self._on_ready = on_ready
        self._on_error = on_error
        self._cam_factory = cam_factory
        self._ready = threading.Event()
        self._duration = RollingStats()
        self._rate = RollingStats()
        self._latency = RollingStats()
        self._drops = 0
        self._repeats = 0
        self._last_frame_seq = 0
        self._last_snapshot_seq = 0
        self._last_out = None
        self._telemetry_at = 0.0
        self._cam = None

    def _open_cam(self):
        return pyvirtualcam.Camera(
            width=self.cfg["out_w"], height=self.cfg["out_h"],
            fps=int(self.cfg["fps"]), fmt=pyvirtualcam.PixelFormat.BGR,
        )

    def wait_ready(self, timeout):
        return self._ready.wait(timeout)

    def run(self):
        try:
            if self._cam_factory:
                self._cam = self._cam_factory()
            else:
                self._cam = self._open_cam()
            self.state.set_telemetry(device=getattr(self._cam, "device", "virtual"))
            self._ready.set()
            if self._on_ready:
                self._on_ready()

            interval = 1.0 / max(1.0, float(self.cfg["fps"]))
            deadline = time.monotonic()
            last_tick = deadline
            while not self.stop_event.is_set():
                now = time.monotonic()
                remaining = deadline - now
                if remaining > 0:
                    # Sleep until the next tick; stop_event wakes us immediately.
                    self.stop_event.wait(remaining)
                    if self.stop_event.is_set():
                        break
                    now = time.monotonic()

                # No catch-up bursts: a late tick resets the schedule.
                if now - deadline > interval:
                    self._drops += 1
                    deadline = now + interval
                else:
                    deadline += interval

                if self.reload_event is not None and self.reload_event.is_set():
                    self.reload_event.clear()
                    self.processor.reload_shader()

                dt = min(0.5, max(1e-4, now - last_tick))
                last_tick = now
                envelope = self.frames.latest()
                values = self.state.snapshot()

                if envelope is not None and envelope.sequence != self._last_frame_seq:
                    self._last_frame_seq = envelope.sequence
                    started = time.monotonic()
                    snapshot = self.snapshots.latest()
                    if (
                        snapshot is not None
                        and snapshot.sequence != self._last_snapshot_seq
                    ):
                        self._last_snapshot_seq = snapshot.sequence
                        self.processor.observe(snapshot, values)
                    curr_x, curr_y, curr_zoom = self.processor.step(dt, now, values)
                    out_frame = self.processor.render(
                        envelope.frame, values, now, curr_x, curr_y, curr_zoom
                    )
                    if out_frame is not None:
                        self._last_out = out_frame
                        self._cam.send(out_frame)
                        self._rate.add(now)
                        self._latency.add(now - envelope.captured_at)
                        self._duration.add(time.monotonic() - started)
                        self.state.set_telemetry(zoom=curr_zoom)
                        if now - self._telemetry_at >= 0.25:
                            self._telemetry_at = now
                            measured = self._rate.fps()
                            # "fps" stays the published render rate for older
                            # telemetry consumers; render_fps is the explicit name.
                            self.state.set_telemetry(fps=measured,
                                                     render_fps=measured)
                        if self.processor.preview:
                            cv2.imshow("hyper-stage", out_frame)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                self.stop_event.set()
                elif self._last_out is not None:
                    # Keep the virtual-camera stream alive without re-running
                    # the effect chain or advancing loop players.
                    self._repeats += 1
                    self._cam.send(self._last_out)
                    self._rate.add(now)
        except Exception as exc:
            if self._on_error:
                self._on_error(exc)
            else:
                print(f"warning: renderer worker stopped: {exc}")
        finally:
            if self._cam is not None:
                try:
                    self._cam.close()
                except Exception:
                    pass
                self._cam = None
            self.processor.close()
            if self.processor.preview:
                cv2.destroyAllWindows()
            self._ready.set()

    def stats(self):
        return {
            "render_fps": self._rate.fps(),
            "render_p50": self._duration.p50(),
            "render_p95": self._duration.p95(),
            "capture_to_output": self._latency.p50(),
            "render_drops": self._drops,
            "render_repeats": self._repeats,
        }
