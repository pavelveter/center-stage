import unittest

import tracker


class TestCropOrigin(unittest.TestCase):
    def test_centered(self):
        x1, y1, cw, ch = tracker.crop_origin(0.5, 0.5, 2.0, 1000, 1000)
        self.assertEqual(cw, 500)
        self.assertEqual(ch, 500)
        self.assertEqual(x1, 250)
        self.assertEqual(y1, 250)

    def test_clamped_left_top(self):
        x1, y1, cw, ch = tracker.crop_origin(0.0, 0.0, 2.0, 1000, 1000)
        self.assertEqual(x1, 0)
        self.assertEqual(y1, 0)

    def test_clamped_right_bottom(self):
        x1, y1, cw, ch = tracker.crop_origin(1.0, 1.0, 2.0, 1000, 1000)
        self.assertEqual(x1, 500)
        self.assertEqual(y1, 500)


class TestFaceCenters(unittest.TestCase):
    class BB:
        def __init__(self, ox, oy, w, h):
            self.origin_x, self.origin_y, self.width, self.height = ox, oy, w, h

    class Det:
        def __init__(self, bb):
            self.bounding_box = bb

    def test_single(self):
        d = self.Det(self.BB(100, 100, 200, 200))
        centers = tracker.face_centers([d], 640, 360)
        self.assertEqual(len(centers), 1)
        self.assertAlmostEqual(centers[0][0], (100 + 100) / 640)
        self.assertAlmostEqual(centers[0][1], (100 + 200 * 0.6) / 360)


class TestLoadConfig(unittest.TestCase):
    def test_defaults_when_missing(self):
        cfg = tracker.load_config(None)
        self.assertEqual(cfg["camera_index"], 1)
        self.assertEqual(cfg["zoom"], 1.5)

    def test_ini_override(self):
        import tempfile, os
        fd, path = tempfile.mkstemp(suffix=".ini")
        with os.fdopen(fd, "w") as f:
            f.write("[camera]\nindex = 3\n[tracking]\nzoom = 2.0\n")
        try:
            cfg = tracker.load_config(path)
            self.assertEqual(cfg["camera_index"], 3)
            self.assertEqual(cfg["zoom"], 2.0)
        finally:
            os.remove(path)


class TestLoadLut(unittest.TestCase):
    def test_loads_cube(self):
        files = tracker.list_lut_files(None)
        if not files:
            self.skipTest("no .cube files in LUT/")
        lut = tracker.load_lut(files[0])
        self.assertIsNotNone(lut)
        self.assertEqual(lut.dtype, __import__("numpy").float32)

    def test_missing_returns_none(self):
        self.assertIsNone(tracker.load_lut("/no/such/file.cube"))


class TestSmartSharpen(unittest.TestCase):
    def test_flat_area_untouched(self):
        import numpy as np
        frame = np.full((32, 32, 3), 120, dtype=np.uint8)
        out = tracker.apply_smart_sharpen(frame, strength=0.3, threshold=10)
        self.assertTrue((out == frame).all())

    def test_edge_is_sharpened(self):
        import numpy as np
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        frame[:, 16:] = 200
        out = tracker.apply_smart_sharpen(frame, strength=0.3, threshold=10)
        self.assertFalse((out == frame).all())

    def test_zero_strength_is_noop(self):
        import numpy as np
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        frame[:, 16:] = 200
        out = tracker.apply_smart_sharpen(frame, strength=0.0)
        self.assertTrue((out == frame).all())


class TestLutCycling(unittest.TestCase):
    def test_cycle_wraps_forward(self):
        self.assertEqual(tracker.cycle_index(2, 3, 1), 0)
        self.assertEqual(tracker.cycle_index(0, 3, 1), 1)

    def test_cycle_wraps_backward(self):
        self.assertEqual(tracker.cycle_index(0, 3, -1), 2)
        self.assertEqual(tracker.cycle_index(1, 3, -1), 0)

    def test_empty_list_is_safe(self):
        self.assertEqual(tracker.cycle_index(0, 0, 1), 0)

    def test_lut_files_sorted(self):
        files = tracker.list_lut_files(None)
        self.assertEqual(files, sorted(files))


class TestSpringStep(unittest.TestCase):
    def test_critically_damped_no_overshoot_and_converges(self):
        pos, vel, target = 1.0, 0.0, 1.5
        omega, zeta, dt = 2.0 * 3.14159 / 0.9, 1.0, 1.0 / 30.0
        for _ in range(300):
            pos, vel = tracker.spring_step(pos, vel, target, omega, zeta, dt)
            self.assertLessEqual(pos, target + 1e-6)
        self.assertAlmostEqual(pos, target, places=3)


if __name__ == "__main__":
    unittest.main()