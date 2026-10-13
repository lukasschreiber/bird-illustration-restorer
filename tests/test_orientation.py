import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from pipeline.steps.fix_orientation import FixOrientationStep
from pipeline.utils import PipelineImageContainer


class OrientationTests(unittest.TestCase):
    def test_visible_caption_rotation_does_not_depend_on_pdf_text(self):
        step = FixOrientationStep('orientation', pipeline=Mock())
        step._caption_rotation = Mock(return_value=cv2.ROTATE_90_CLOCKWISE)
        source = np.arange(18, dtype=np.uint8).reshape(3, 2, 3)
        item = PipelineImageContainer(source, 248, 0, 'Capercailzie', '', 34, 'book')
        result = step.run(item)
        np.testing.assert_array_equal(result.image, cv2.rotate(source, cv2.ROTATE_90_CLOCKWISE))
        step.pipeline.get_property.assert_not_called()

    def test_horizontal_matching_caption_selects_clockwise_rotation(self):
        step = FixOrientationStep('orientation')
        vertical = [[0, 0], [10, 0], [10, 100], [0, 100]]
        horizontal = [[0, 0], [100, 0], [100, 10], [0, 10]]
        step._ocr = Mock(side_effect=[
            ([[vertical, 'BRENT GOOSE', .95]], None),
            ([[horizontal, 'BRENT GOOSE', .95]], None),
            (None, None), (None, None)])
        self.assertEqual(step._caption_rotation(np.zeros((100, 60, 3), np.uint8),
                                               'Brent Goose'), cv2.ROTATE_90_CLOCKWISE)
        self.assertTrue(all(call.kwargs.get('use_cls') is False
                            for call in step._ocr.call_args_list))

    def test_unrelated_text_does_not_rotate_artwork(self):
        step = FixOrientationStep('orientation')
        box = [[0, 0], [100, 0], [100, 10], [0, 10]]
        step._ocr = Mock(return_value=([[box, 'LITHOGRAPH', .99]], None))
        self.assertIsNone(step._caption_rotation(np.zeros((100, 60, 3), np.uint8),
                                                'Brent Goose'))
