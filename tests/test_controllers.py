import math
import types

import numpy as np
import pytest

from hyper_stage.config import DEFAULTS
from hyper_stage.tracking import (
    DETECT_H, DETECT_W, PositionController, ZoomController, crop_origin,
    face_centers, face_size, pick_subject, spring_step,
)


class TestControllers:
    @staticmethod
    def make_detection(cx, cy, height, eye=None, score=0.9):
        det_w, det_h = DETECT_W, DETECT_H
        aspect = det_w / det_h
        w = height * det_h / aspect
        bb = types.SimpleNamespace(
            origin_x=(cx - w / 2) * det_w,
            origin_y=(cy - height * 0.6) * det_h,
            width=w * det_w,
            height=height * det_h,
        )
        kps = None
        if eye is not None:
            dx = eye / 2.8 / aspect  # eye distance expressed as height units
            kps = [
                types.SimpleNamespace(x=cx - dx, y=cy),
                types.SimpleNamespace(x=cx + dx, y=cy),
            ]
        return types.SimpleNamespace(
            bounding_box=bb, keypoints=kps,
            categories=[types.SimpleNamespace(score=score)],
        )

    @staticmethod
    def cfg(**overrides):
        cfg = dict(DEFAULTS)
        cfg.update(overrides)
        return cfg

    def test_small_position_oscillation_does_not_wander(self):
        ctrl = PositionController()
        cfg = self.cfg()
        now = 0.0
        for i in range(180):  # 6 seconds at 30 fps, detection every 6 frames
            if i % 6 == 0:
                x = 0.5 + 0.012 * (-1) ** (i // 6)
                ctrl.observe(x, 0.45, now, cfg["position_deadband"])
            ctrl.step(1 / 30, now, cfg["position_response_time"],
                      cfg["position_damping"], cfg["detection_timeout"])
            now += 1 / 30
            assert abs(ctrl.x - (0.5)) <= (0.01)
            assert abs(ctrl.y - (0.45)) <= (0.01)

    def test_slow_and_sudden_position_moves_converge_without_overshoot(self):
        cfg = self.cfg()
        for speed in ("slow", "jump"):
            ctrl = PositionController()
            now = 0.0
            for i in range(240):
                target = min(0.8, 0.5 + (i / 240) * 0.3) if speed == "slow" else 0.8
                ctrl.observe(target, 0.45, now, cfg["position_deadband"])
                ctrl.step(1 / 30, now, cfg["position_response_time"],
                          cfg["position_damping"], cfg["detection_timeout"])
                now += 1 / 30
                assert ctrl.x <= 0.8 + 1e-09  # no overshoot
            # A slow ramp legitimately lags a spring by a few percent.
            assert abs(ctrl.x - (0.8)) <= (0.04 if speed == 'slow' else 0.01)

    def test_position_frame_rate_independent(self):
        cfg = self.cfg()
        finals = []
        for fps in (15, 30, 60):
            ctrl = PositionController()
            for i in range(fps * 3):
                now = i / fps
                ctrl.observe(0.7, 0.45, now, cfg["position_deadband"])
                ctrl.step(1 / fps, now, cfg["position_response_time"],
                          cfg["position_damping"], cfg["detection_timeout"])
            finals.append(ctrl.x)
        assert max(finals) - min(finals) < 0.005

    def test_zoom_ignores_head_rotation_when_landmarks_stable(self):
        cfg = self.cfg()

        def run(rotating):
            z = ZoomController(1.25)
            now = 0.0
            for i in range(180):
                if i % 6 == 0:
                    # Bbox swings +-20% while the eyes (distance proxy) stay.
                    height = 0.32 * (1 + 0.2 * (-1) ** (i // 6)) if rotating else 0.32
                    d = self.make_detection(0.5, 0.45, height, eye=0.32 / 2.8)
                    z.observe(face_size(d, DETECT_W, DETECT_H), now,
                              cfg["zoom_measurement_smoothing"],
                              cfg["max_measurement_change"])
                z.step(1 / 30, now, 1.5, cfg["max_head_height"], cfg["head_ratio"],
                       cfg["zoom_deadzone"], cfg["zoom_in_response_time"],
                       cfg["zoom_out_response_time"], cfg["detection_timeout"])
                now += 1 / 30
            return z.value

        assert abs(run(True) - run(False)) < 0.005

    def test_zoom_small_changes_ignored_and_outliers_rejected(self):
        cfg = self.cfg()
        z = ZoomController(1.25)
        z.observe(0.4, 0.0, cfg["zoom_measurement_smoothing"],
                  cfg["max_measurement_change"])
        z.observe(0.405, 0.2, cfg["zoom_measurement_smoothing"],
                  cfg["max_measurement_change"])
        assert abs(z.filtered_size - 0.4) < 0.01
        z.observe(0.05, 0.21, cfg["zoom_measurement_smoothing"],
                  cfg["max_measurement_change"])
        assert z.filtered_size > 0.3  # implausible jump limited

    def test_zoom_asymmetric_response_and_recovery(self):
        cfg = self.cfg()
        t_in, t_out = cfg["zoom_in_response_time"], cfg["zoom_out_response_time"]
        assert t_in > t_out * 2
        z = ZoomController(1.0)
        # Same target step in each direction; a huge deadband freezes the
        # accepted target so only the asymmetric time constants are measured.
        z.target = 1.5
        z.step(1.0, 1.0, 2.0, cfg["max_head_height"], cfg["head_ratio"], 10.0,
               t_in, t_out, cfg["detection_timeout"])
        rise = z.value - 1.0
        z = ZoomController(2.0)
        z.target = 1.5
        z.step(1.0, 1.0, 2.0, cfg["max_head_height"], cfg["head_ratio"], 10.0,
               t_in, t_out, cfg["detection_timeout"])
        fall = 2.0 - z.value
        assert fall > rise * 2

        # Missing detections keep the state and ease toward base, no jump.
        z = ZoomController(1.5)
        z.observe(0.4, 0.0, cfg["zoom_measurement_smoothing"],
                  cfg["max_measurement_change"])
        before = z.value
        z.step(1 / 30, 5.0, 1.2, cfg["max_head_height"], cfg["head_ratio"],
               cfg["zoom_deadzone"], t_in, t_out, cfg["detection_timeout"])
        assert abs(z.value - before) < 0.05

    def test_subject_selection_is_consistent(self):
        big = self.make_detection(0.3, 0.4, 0.4, score=0.7)
        small = self.make_detection(0.7, 0.6, 0.25, score=0.99)
        assert pick_subject([small, big]) is big
        assert pick_subject([]) is None


class TestCropOrigin:
    def test_centered(self):
        x1, y1, cw, ch = crop_origin(0.5, 0.5, 2.0, 1000, 1000)
        assert cw == 500
        assert ch == 500
        assert x1 == 250
        assert y1 == 250

    def test_clamped_left_top(self):
        x1, y1, cw, ch = crop_origin(0.0, 0.0, 2.0, 1000, 1000)
        assert x1 == 0
        assert y1 == 0

    def test_clamped_right_bottom(self):
        x1, y1, cw, ch = crop_origin(1.0, 1.0, 2.0, 1000, 1000)
        assert x1 == 500
        assert y1 == 500


    def test_output_aspect_never_distorts_portrait_input(self):
        # Portrait camera, landscape virtual camera: crop must match the
        # output aspect so the final resize does not stretch the picture.
        x, y, w, h = crop_origin(0.5, 0.45, 1.5, 1080, 1920, 1920, 1080)
        assert w / h == 1920 / 1080
        assert w <= 1080
        assert h <= 1920
        assert 0 <= x <= 1080 - w and 0 <= y <= 1920 - h
        # Backward-compatible call keeps the input aspect.
        x, y, w, h = crop_origin(0.5, 0.5, 2.0, 1080, 1920)
        assert (w, h) == (540, 960)

    def test_min_max_zoom_stays_inside_the_frame(self):
        for zoom in (1.0, 1.25, 4.0, 100.0):
            x, y, w, h = crop_origin(1.0, 1.0, zoom, 640, 480, 1920, 1080)
            assert x >= 0
            assert y >= 0
            assert x + w <= 640
            assert y + h <= 480
            assert w > 0
            assert h > 0

    def test_crop_uses_mirrored_coordinates(self):
        # The pipeline mirrors the raw frame before detection and cropping,
        # so a face at raw x=0.2 becomes x=0.8 in the published view.
        raw_x = 0.2
        mirrored_x = 1.0 - raw_x
        x1, _, w, _ = crop_origin(mirrored_x, 0.5, 2.0, 1000, 1000, 1000, 1000)
        # Face at raw 0.2 -> mirrored 0.8 -> crop clamped to the right edge.
        assert x1 == 500


class TestFaceCenters:
    class BB:
        def __init__(self, ox, oy, w, h):
            self.origin_x, self.origin_y, self.width, self.height = ox, oy, w, h

    class Det:
        def __init__(self, bb):
            self.bounding_box = bb

    def test_single(self):
        d = self.Det(self.BB(100, 100, 200, 200))
        centers = face_centers([d], 640, 360)
        assert len(centers) == 1
        assert centers[0][0] == pytest.approx((100 + 100) / 640)
        assert centers[0][1] == pytest.approx((100 + 200 * 0.6) / 360)


class TestSpringStep:
    def test_stable_at_large_frame_intervals(self):
        # Explicit Euler with dt=0.2 and omega=12.5 diverges; the
        # substepped version must stay bounded and converge.
        pos, vel, target = 1.0, 0.0, 1.5
        omega, zeta = 2.0 * math.pi / 0.5, 1.0
        for _ in range(10):
            pos, vel = spring_step(pos, vel, target, omega, zeta, 0.2)
            assert pos <= target + 1e-06
            assert pos >= 1.0
        assert pos > 1.4

    def test_critically_damped_no_overshoot_and_converges(self):
        pos, vel, target = 1.0, 0.0, 1.5
        omega, zeta, dt = 2.0 * 3.14159 / 0.9, 1.0, 1.0 / 30.0
        for _ in range(300):
            pos, vel = spring_step(pos, vel, target, omega, zeta, dt)
            assert pos <= target + 1e-06
        assert pos == pytest.approx(target, abs=10 ** -(3))

