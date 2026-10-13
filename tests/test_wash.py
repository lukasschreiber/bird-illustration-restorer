import unittest

import cv2
import numpy as np

from pipeline.steps.fade_wash import FadePaintedWashStep
from pipeline import Pipeline
from pipeline.utils import PipelineImageContainer


class WashTests(unittest.TestCase):
    def setUp(self):
        self.step = FadePaintedWashStep("wash", "protected", max_size=400)
        self.source = np.full((240, 400, 3), 255, np.uint8)
        self.mask = np.zeros((240, 400), np.uint8)
        self.mask[20:220, 20:380] = 255
        self.source[self.mask > 0] = [240, 232, 226]
        self.protected = np.zeros_like(self.mask)
        self.protected[70:170, 130:270] = 255

    def test_exposed_wash_fades_but_protected_pale_subject_is_unchanged(self):
        fade = self.step._fade_amount(self.source, self.mask, self.protected)
        self.assertGreater(fade[100, 50], .8)
        self.assertTrue(np.all(fade[self.protected > 0] == 0))
        self.assertTrue(np.all(fade[self.mask == 0] == 0))

    def test_warm_ground_is_preserved(self):
        self.source[180:210, 20:380] = [185, 210, 230]
        fade = self.step._fade_amount(self.source, self.mask, self.protected)
        self.assertEqual(fade[195, 70], 0)

    def test_dark_scenery_strokes_are_preserved(self):
        cv2.line(self.source, (40, 40), (80, 200), (70, 80, 80), 3)
        fade = self.step._fade_amount(self.source, self.mask, self.protected)
        self.assertEqual(fade[120, 60], 0)

    def test_enclosed_pale_area_does_not_fade(self):
        self.protected[50:190, 110:290] = 255
        self.protected[80:160, 140:260] = 0
        self.source[80:160, 140:260] = [235, 235, 235]
        fade = self.step._fade_amount(self.source, self.mask, self.protected)
        self.assertEqual(fade[120, 200], 0)

    def test_blue_gap_between_protected_stems_clears(self):
        self.protected[50:190, 110:290] = 255
        self.protected[80:160, 140:260] = 0
        fade = self.step._fade_amount(self.source, self.mask, self.protected)
        self.assertGreater(fade[120, 200], .99)
        self.assertTrue(np.all(fade[self.protected > 0] == 0))

    def test_blue_margin_clears_without_touching_blue_subject(self):
        fade = self.step._fade_amount(self.source, self.mask, self.protected)
        self.assertGreater(fade[100, 128], .5)
        self.assertTrue(np.all(fade[self.protected > 0] == 0))

    def test_large_wash_between_dark_scenery_edges_can_fade(self):
        self.protected[:] = 0
        cv2.rectangle(self.source, (80, 50), (300, 190), (80, 90, 85), 3)
        fade = self.step._fade_amount(self.source, self.mask, self.protected)
        self.assertGreater(fade[120, 200], .8)
        self.assertEqual(fade[50, 200], 0)

    def test_zero_strength_is_a_noop(self):
        step = FadePaintedWashStep("wash", "protected", strength=0)
        self.assertTrue(np.all(step._fade_amount(self.source, self.mask, self.protected) == 0))

    def test_pipeline_keeps_protected_pixels_and_source_unchanged(self):
        def item(array):
            return PipelineImageContainer(array, 1, 0, "bird", "bird", 1, "book")
        pipeline = Pipeline()
        pipeline.cache["1_0_protected"] = item(self.protected)
        source_copy = self.source.copy()
        step = FadePaintedWashStep("wash", "protected", pipeline=pipeline)
        result = step.run(item(self.source), item(self.mask)).image
        np.testing.assert_array_equal(result[self.protected > 0], self.source[self.protected > 0])
        np.testing.assert_array_equal(self.source, source_copy)
        self.assertGreater(float(result[100, 50].mean()), float(self.source[100, 50].mean()))

    def test_paper_grain_in_confirmed_wash_does_not_leave_gray_speckles(self):
        rng = np.random.default_rng(0)
        grain = rng.integers(-3, 4, self.source.shape)
        source = np.clip(self.source.astype(int) + grain, 0, 255).astype(np.uint8)
        fade = self.step._fade_amount(source, self.mask, self.protected)
        self.assertTrue(np.all(fade[80:140, 40:80] > .999))
        cleaned = np.rint(source + (255-source.astype(float))*fade[:, :, None]).astype(np.uint8)
        self.assertTrue(np.all(cleaned[80:140, 40:80] == 255))

    def test_pale_wash_fragments_are_not_mistaken_for_scenery_ink(self):
        self.source[95:100, 60:65] = 220
        fade = self.step._fade_amount(self.source, self.mask, self.protected)
        self.assertGreater(fade[97, 62], .99)

    def test_cleanup_remains_gentle_next_to_dark_water_strokes(self):
        rng = np.random.default_rng(1)
        source = np.clip(self.source.astype(int) + rng.integers(-4, 5, self.source.shape), 0, 255).astype(np.uint8)
        cv2.line(source, (20, 100), (110, 100), (170, 170, 170), 2)
        fade = self.step._fade_amount(source, self.mask, self.protected)
        self.assertEqual(fade[100, 60], 0)
        self.assertLess(float(fade[105:108, 50:90].mean()), .99)

    def test_scan_resolution_does_not_reintroduce_gray_wash_border(self):
        source = cv2.resize(self.source, (800, 480), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(self.mask, (800, 480), interpolation=cv2.INTER_NEAREST)
        protected = cv2.resize(self.protected, (800, 480), interpolation=cv2.INTER_NEAREST)
        fade = self.step._fade_amount(source, mask, protected)
        self.assertEqual(fade[200, 759], 1)
        self.assertEqual(fade[200, 760], 0)


if __name__ == "__main__":
    unittest.main()
