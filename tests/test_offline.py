"""Offline tests: geometry, projection, spacing, keyword mapping and blur detection (no network, no model)."""

import math
import random
import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageFilter

from streetseek import geo, panos, scoring


def textured_scene(seed=0, size=(640, 480)):
    """Sky on top, noisy blocky 'buildings' below: stands in for a sharp street photo."""
    rng = np.random.default_rng(seed)
    w, h = size
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[: h // 3] = [135, 180, 230]
    for _ in range(40):
        x, y = rng.integers(0, w - 60), rng.integers(h // 3, h - 40)
        img[y : y + rng.integers(20, 120), x : x + rng.integers(20, 160)] = rng.integers(40, 220, 3)
    img[h // 3 :] += rng.normal(0, 12, (h - h // 3, w, 3))
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


class GeoTests(unittest.TestCase):
    def test_bearing_cardinal_directions(self):
        self.assertAlmostEqual(geo.bearing_deg(0, 0, 1, 0), 0, places=3)
        self.assertAlmostEqual(geo.bearing_deg(0, 0, 0, 1), 90, places=3)
        self.assertAlmostEqual(geo.bearing_deg(0, 0, -1, 0), 180, places=3)

    def test_distance_one_degree_latitude(self):
        self.assertAlmostEqual(geo.distance_m(0, 0, 1, 0), 111_195, delta=50)

    def test_feature_keywords(self):
        self.assertTrue(geo.feature_filters("an old brick Church with a steeple"))
        self.assertTrue(geo.feature_filters("a gas station on a corner"))
        self.assertEqual(geo.feature_filters("a blue house with a red door"), [])


class PanoTests(unittest.TestCase):
    def test_render_view_faces_requested_heading(self):
        # Equirect with a red stripe at compass bearing 90 when the pano's center faces north (0).
        w, h = 720, 360
        eq = np.zeros((h, w, 3), dtype=np.uint8)
        col = int((90 / 360 + 0.5) * w)
        eq[:, col - 3 : col + 3] = [255, 0, 0]
        view = np.asarray(panos.render_view(eq, 0.0, 90, fov_deg=60, size=(101, 51)))
        self.assertGreater(view[25, 50, 0], 200)
        self.assertLess(view[25, 0, 0], 50)

    def test_spaced_keeps_points_apart(self):
        rng = random.Random(1)
        pts = [SimpleNamespace(lat=40 + rng.uniform(0, 0.005), lon=-83 + rng.uniform(0, 0.005)) for _ in range(400)]
        kept = panos.spaced(pts, 50)
        self.assertTrue(10 < len(kept) < 400)
        for i, a in enumerate(kept):
            for b in kept[i + 1 :]:
                self.assertGreaterEqual(geo.distance_m(a.lat, a.lon, b.lat, b.lon), 49)


class BlurTests(unittest.TestCase):
    def test_sharp_scene_is_not_blurred(self):
        for seed in range(5):
            self.assertLess(scoring.blur_fraction(textured_scene(seed)), 0.03)

    def test_privacy_style_blur_is_detected(self):
        for seed in range(5):
            img = textured_scene(seed)
            box = (150, 220, 500, 470)
            img.paste(img.crop(box).filter(ImageFilter.GaussianBlur(12)), box)
            self.assertGreaterEqual(scoring.blur_fraction(img), 0.03)

    def test_flat_sky_is_ignored(self):
        self.assertEqual(scoring.blur_fraction(Image.new("RGB", (640, 480), (140, 180, 230))), 0)


if __name__ == "__main__":
    unittest.main()
