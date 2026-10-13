import unittest

import cv2
import numpy as np

from pipeline.steps.paper_balance import PaperBalanceStep
from pipeline.steps.crop_to_mask import CropToMaskStep
from pipeline.steps.text_cleanup import AutomaticTextCleanupStep
from pipeline.steps.brightness_contrast import BrightnessContrastStep
from pipeline.utils import PipelineImageContainer


def item(image):
    return PipelineImageContainer(image, 1, 0, "test", "test", 1, "book")


class CleanupTests(unittest.TestCase):
    def test_pale_shading_survives_paper_correction(self):
        source = np.full((100, 120, 3), [210, 225, 235], np.uint8)
        source[30:70, 30:60] = [220, 230, 240]
        source[30:70, 60:90] = [235, 240, 245]
        mask = np.zeros((100, 120), np.uint8)
        mask[30:70, 30:90] = 255
        result = PaperBalanceStep("balance").run(item(source), item(mask)).image
        self.assertGreater(float(result[50, 75].mean()), float(result[50, 45].mean()) + 2)
        self.assertTrue(np.all(result[50, 75] < 255))
        np.testing.assert_array_equal(source[0, 0], [210, 225, 235])

    def test_missing_paper_samples_preserve_original_color(self):
        source = np.full((80, 100, 3), [130, 180, 200], np.uint8)
        result = PaperBalanceStep("balance").run(item(source), item(np.full((80, 100), 255, np.uint8)))
        np.testing.assert_array_equal(result.image, source)

    def test_final_contrast_keeps_distinct_highlight_shades(self):
        source = np.array([[[240, 240, 240], [250, 250, 250]]], np.uint8)
        result = BrightnessContrastStep("contrast", -5, 15, soft_highlights=True).run(item(source)).image
        self.assertGreater(int(result[0, 1, 0]), int(result[0, 0, 0]))
        self.assertLess(int(result[0, 1, 0]), 255)

    def test_local_illumination_adjustment_stays_within_its_limit(self):
        source = np.full((120, 160, 3), 230, np.uint8)
        source[:, :80] = 220
        mask = np.zeros((120, 160), np.uint8)
        mask[45:75, 60:100] = 255
        baseline = PaperBalanceStep("baseline", local_strength=0).run(item(source), item(mask)).image
        corrected = PaperBalanceStep("local", local_strength=1, max_local_shift=2).run(item(source), item(mask)).image
        difference = corrected.astype(int) - baseline.astype(int)
        self.assertTrue(np.any(difference != 0))
        self.assertLessEqual(np.max(np.abs(difference)), 3)

    def test_feather_has_soft_edge_white_exterior_and_unchanged_interior(self):
        source = np.full((40, 40, 3), 100, np.uint8)
        mask = np.zeros((40, 40), np.uint8)
        mask[10:30, 10:30] = 255
        result = CropToMaskStep("crop", padding=5, feather_radius=3).run(item(source), item(mask)).image
        self.assertTrue(np.all(result[0, 0] == 255))
        self.assertTrue(np.all(result[15, 15] == 100))
        self.assertTrue(np.all(result[5, 15] > 100))
        self.assertTrue(np.all(result[5, 15] < 255))

    def test_text_amid_dense_texture_is_not_removed(self):
        source = np.zeros((600, 240, 3), np.uint8)
        source[::2] = 255
        step = AutomaticTextCleanupStep("text", report_directory=None)
        box = [[50, 40], [180, 40], [180, 65], [50, 65]]
        step._ocr = lambda _: ([[box, "Signature", .99]], None)
        np.testing.assert_array_equal(step.run(item(source)).image, source)

    def test_confident_isolated_text_removed_without_whitening_rectangle(self):
        source = np.full((600, 240, 3), [190, 210, 225], np.uint8)
        cv2.putText(source, "TEST", (50, 65), cv2.FONT_HERSHEY_SIMPLEX, .7, (20, 20, 20), 2)
        step = AutomaticTextCleanupStep("text", report_directory=None)
        box = [[45, 45], [120, 45], [120, 72], [45, 72]]
        step._ocr = lambda _: ([[box, "TEST", .99]], None)
        result = step.run(item(source)).image
        self.assertLess(np.count_nonzero(result[:, :, 0] < 80), np.count_nonzero(source[:, :, 0] < 80) / 3)
        np.testing.assert_array_equal(result[10, 10], source[10, 10])
        self.assertFalse(np.any(np.all(result == 255, axis=2)))

    def test_low_confidence_or_single_character_is_preserved(self):
        source = np.full((120, 240, 3), 230, np.uint8)
        cv2.putText(source, "TEST", (50, 65), cv2.FONT_HERSHEY_SIMPLEX, .7, (20, 20, 20), 2)
        step = AutomaticTextCleanupStep("text", report_directory=None)
        box = [[45, 45], [120, 45], [120, 72], [45, 72]]
        step._ocr = lambda _: ([[box, "TEST", .5], [box, "X", .99]], None)
        np.testing.assert_array_equal(step.run(item(source)).image, source)

    def test_conventional_signature_still_requires_quiet_surroundings(self):
        source = np.full((600, 240, 3), 230, np.uint8)
        cv2.putText(source, "Lear.del", (40, 65), cv2.FONT_HERSHEY_SIMPLEX, .5, (20, 20, 20), 1)
        box = [[35, 45], [120, 45], [120, 72], [35, 72]]
        step = AutomaticTextCleanupStep("text", report_directory=None)
        step._ocr = lambda _: ([[box, "Lear.del", .77]], None)
        result = step.run(item(source)).image
        self.assertLess(np.count_nonzero(result[:, :, 0] < 80), np.count_nonzero(source[:, :, 0] < 80) / 3)
        source[::2] = 0
        np.testing.assert_array_equal(step.run(item(source)).image, source)


if __name__ == "__main__":
    unittest.main()
