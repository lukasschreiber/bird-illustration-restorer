import unittest

import cv2
import numpy as np

from pipeline import Pipeline
from pipeline.steps.find_subject import FindSubjectStep
from pipeline.utils import PipelineImageContainer


class SubjectMaskTests(unittest.TestCase):
    def setUp(self):
        self.step = FindSubjectStep("subject", 50, 0.5, pipeline=Pipeline(),
                                    method="source_edges", detection_max_size=400)

    def mask(self, gray):
        item = PipelineImageContainer(gray, 1, 0, "test", "test", 1, "test")
        return self.step.process_single(item).image

    def test_pale_subject_interior_and_separate_small_subject_survive(self):
        gray = np.full((300, 400), 225, np.uint8)
        cv2.ellipse(gray, (140, 150), (85, 100), 0, 0, 360, 200, 2)
        cv2.ellipse(gray, (320, 150), (25, 40), 0, 0, 360, 180, 2)
        mask = self.mask(gray)
        self.assertEqual(mask[150, 140], 255)
        self.assertEqual(mask[150, 320], 255)
        self.assertEqual(mask[20, 20], 0)

    def test_blank_scan_is_preserved_when_detection_is_uncertain(self):
        self.assertTrue(np.all(self.mask(np.full((100, 150), 230, np.uint8)) == 255))

    def test_thin_scan_border_does_not_expand_the_crop(self):
        gray = np.full((300, 400), 225, np.uint8)
        cv2.ellipse(gray, (200, 150), (70, 80), 0, 0, 360, 170, 3)
        cv2.line(gray, (0, 295), (399, 295), 100, 2)
        mask = self.mask(gray)
        self.assertEqual(mask[150, 200], 255)
        self.assertEqual(mask[295, 200], 0)

    def test_hole_fill_handles_foreground_at_corner(self):
        mask = np.zeros((30, 30), np.uint8)
        mask[:20, :20] = 255
        mask[5:10, 5:10] = 0
        filled = self.step._fill_binary_holes(mask)
        self.assertEqual(filled[7, 7], 255)
        self.assertEqual(filled[25, 25], 0)

    def test_resolution_does_not_change_subject_selection(self):
        gray = np.full((300, 400), 225, np.uint8)
        cv2.ellipse(gray, (200, 150), (100, 80), 0, 0, 360, 170, 3)
        small = self.mask(gray)
        large = self.mask(cv2.resize(gray, (800, 600), interpolation=cv2.INTER_NEAREST))
        reduced = cv2.resize(large, (400, 300), interpolation=cv2.INTER_NEAREST)
        self.assertGreater(np.count_nonzero((small > 0) & (reduced > 0)) /
                           np.count_nonzero((small > 0) | (reduced > 0)), 0.98)

    def test_config_detects_before_correction_and_has_no_manual_step(self):
        pipeline = Pipeline()
        pipeline.load_from_config("config.yaml")
        available = {"initial_image"}
        for step, source, mask in pipeline.steps:
            self.assertIn(source, available)
            if mask:
                self.assertIn(mask, available)
            self.assertNotEqual(type(step).__name__, "RemoveFeaturesLocalStep")
            if isinstance(step, FindSubjectStep):
                self.assertEqual(source, "grayscale_image")
                self.assertNotIn("white_balanced_image_to_use", available)
            available.add(step.name)


if __name__ == "__main__":
    unittest.main()
