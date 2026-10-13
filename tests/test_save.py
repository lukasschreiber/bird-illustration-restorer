import unittest

from pipeline.save import output_filename
from pipeline.utils import PipelineImageContainer


class OutputFilenameTests(unittest.TestCase):
    def test_bird_name_is_safe_for_windows(self):
        image = PipelineImageContainer(None, 352, 0, 'Brent Goose: /bird?', '', 0, 'book')
        self.assertEqual(output_filename(image), '352-Brent-Goose-bird.jpg')

    def test_instances_have_distinct_names(self):
        image = PipelineImageContainer(None, 53, 1, 'White-bellied Swift', '', 0, 'book')
        self.assertEqual(output_filename(image), '53-v1-White-bellied-Swift.jpg')
