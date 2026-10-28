"""Offline tests: geometry, projection, spacing, keyword mapping and blur detection (no network, no model)."""

import math
import random
import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageFilter

from streetseek import geo, panos, scoring, search


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
            box = (150, 180, 500, 400)  # a house: the blur rises past the horizon
            img.paste(img.crop(box).filter(ImageFilter.GaussianBlur(12)), box)
            self.assertGreaterEqual(scoring.blur_fraction(img), 0.03)

    def test_camera_car_smudge_is_ignored(self):
        for seed in range(5):
            img = textured_scene(seed)
            box = (100, 350, 540, 480)  # Google's smear over its own car: big, but all below the horizon
            img.paste(img.crop(box).filter(ImageFilter.GaussianBlur(12)), box)
            self.assertLess(scoring.blur_fraction(img), 0.03)

    def test_flat_sky_is_ignored(self):
        self.assertEqual(scoring.blur_fraction(Image.new("RGB", (640, 480), (140, 180, 230))), 0)

    def test_blur_words_are_detected_and_kept_from_clip(self):
        self.assertTrue(search.wants_blur("a blurred white house with a porch"))
        self.assertTrue(search.wants_blur("white house, Blurry"))
        self.assertFalse(search.wants_blur("a red brick church"))
        self.assertEqual(search.clip_text("a blurred white house with a porch"), "a white house with a porch")
        self.assertEqual(search.clip_text("white house, blurred"), "white house")

    def test_sweep_finds_blurred_house_beside_the_match(self):
        # Sharp panorama with a privacy-style blur centered 60 degrees right of the matched heading (0).
        eq = textured_scene(3, size=(2048, 1024))
        col = int((60 / 360 + 0.5) * 2048)
        box = (col - 110, 440, col + 110, 640)  # a house: from above the horizon to ~22 degrees down
        eq.paste(eq.crop(box).filter(ImageFilter.GaussianBlur(12)), box)
        here, (aim, side) = search.blur_sweep(np.asarray(eq), SimpleNamespace(heading=0.0), 0.0)
        self.assertLess(here, search.BLUR_THRESHOLD)
        self.assertAlmostEqual(aim, 60, delta=8)
        self.assertGreaterEqual(side, search.BLUR_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
