import unittest
from unittest.mock import patch

import numpy as np
import cv2

from pipeline import Pipeline
from pipeline.steps.final_color import FinalColorEnhancementStep
from pipeline.utils import PipelineImageContainer


class FinalColorTests(unittest.TestCase):
    def test_original_formula_preserves_white_and_color_channel_highlights(self):
        source = np.array([[[255, 255, 255], [220, 220, 220],
                            [100, 160, 230], [20, 50, 80]]], np.uint8)
        image = PipelineImageContainer(source, 1, 0, "bird", "bird", 1, "book")
        step = FinalColorEnhancementStep("final", pipeline=Pipeline())
        with patch.object(step.balance, "_detect_background_color_kmeans", return_value=np.array([255, 255, 255])):
            result = step.run(image).image
        balanced = np.clip(source.astype(np.float32) * 1.2, 0, 255).astype(np.uint8)
        expected = balanced.copy()
        colored = np.any(balanced != 255, axis=2)
        expected[colored] = np.clip(balanced[colored] * (30/127+1) - 30 - 40, 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(result[0, :2], 255)
        np.testing.assert_array_equal(source, image.image)

    def test_color_enhancement_is_last_after_wash_and_crop(self):
        pipeline = Pipeline()
        pipeline.load_from_config("config.yaml")
        final, source, mask = pipeline.steps[-1]
        self.assertIsInstance(final, FinalColorEnhancementStep)
        self.assertEqual(source, "cleaned_image")
        self.assertIsNone(mask)
        self.assertEqual(type(pipeline.steps[-2][0]).__name__, "CropToMaskStep")
        self.assertEqual(type(pipeline.steps[-3][0]).__name__, "FadePaintedWashStep")

    def test_dark_cropped_artwork_uses_original_paper_as_color_reference(self):
        source = np.full((8, 8, 3), 100, np.uint8)
        image = PipelineImageContainer(source, 1, 0, "bird", "bird", 1, "book")
        pipeline = Pipeline()
        pipeline.cache["1_0_original"] = PipelineImageContainer(
            np.full((16, 16, 3), 220, np.uint8), 1, 0, "bird", "bird", 1, "book")
        step = FinalColorEnhancementStep("final", pipeline=pipeline, reference_image="original")
        result = step.run(image).image
        expected = int(120 * (30/127+1) - 70)
        self.assertTrue(np.all(result == expected))
        self.assertTrue(np.all(source == 100))

    def test_detail_preserving_grade_keeps_distinct_pale_shades(self):
        shades = np.arange(220, 255, 5, dtype=np.uint8)
        source = np.repeat(shades[None, :, None], 3, axis=2)
        image = PipelineImageContainer(source, 1, 0, "bird", "bird", 1, "book")
        result = FinalColorEnhancementStep("final", mode="detail_preserving").run(image).image
        self.assertTrue(np.all(np.diff(result[0, :, 0].astype(int)) > 0))
        self.assertTrue(np.all(result < 255))
        np.testing.assert_array_equal(source, image.image)

    def test_detail_preserving_grade_preserves_white_background_and_deepens_shadows(self):
        source = np.array([[[255, 255, 255], [60, 60, 60], [100, 120, 160]]], np.uint8)
        image = PipelineImageContainer(source, 1, 0, "bird", "bird", 1, "book")
        result = FinalColorEnhancementStep("final", mode="detail_preserving").run(image).image
        np.testing.assert_array_equal(result[0, 0], 255)
        self.assertLess(int(result[0, 1, 0]), 60)

    def test_vibrance_increases_muted_midtone_color(self):
        source = np.array([[[90, 120, 160]]], np.uint8)
        def grade(vibrance):
            image = PipelineImageContainer(source, 1, 0, "bird", "bird", 1, "book")
            result = FinalColorEnhancementStep("final", mode="detail_preserving", vibrance=vibrance).run(image).image
            lab = cv2.cvtColor(result.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
            return float(np.hypot(lab[0, 0, 1], lab[0, 0, 2]))
        self.assertGreater(grade(.25), grade(0)*1.05)


if __name__ == "__main__":
    unittest.main()
