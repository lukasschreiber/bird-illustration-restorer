import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from pipeline import Pipeline
from pipeline.steps.semantic_mask import SemanticIllustrationMaskStep
from pipeline.utils import PipelineImageContainer


class SemanticMaskTests(unittest.TestCase):
    def step(self, **kwargs):
        return SemanticIllustrationMaskStep("subject", "unused.onnx", **kwargs)

    def test_multiclass_keeps_birds_and_scenery_excludes_paper_and_lettering(self):
        # Classes: paper, bird, scenery, lettering. One pixel per class.
        output = np.full((1, 4, 1, 4), -20, np.float32)
        for label in range(4):
            output[0, label, 0, label] = 20
        probability = self.step(foreground_classes=[1, 2])._foreground_probability(output)
        np.testing.assert_array_equal(probability > .5, [[False, True, True, False]])

    def test_flat_background_prediction_is_not_stretched_to_foreground(self):
        probability = self.step()._foreground_probability(np.full((1, 1, 4, 4), -20))
        self.assertTrue(np.all(probability < .01))

    def test_nonfinite_prediction_is_rejected(self):
        with self.assertRaises(ValueError):
            self.step()._foreground_probability(np.full((1, 1, 4, 4), np.nan))

    def test_auto_retries_gpu_failure_on_cpu(self):
        step = self.step(input_size=32)
        output = np.full((1, 1, 32, 32), -20, np.float32)
        output[:, :, 8:24, 8:24] = 20
        gpu = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name='input')],
                              get_providers=lambda: ['CUDAExecutionProvider'],
                              run=Mock(side_effect=RuntimeError('GPU out of memory')))
        cpu = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name='input')],
                              run=Mock(return_value=[output]))
        step._get_session = Mock(side_effect=[gpu, cpu])
        image = PipelineImageContainer(np.full((50, 80, 3), 220, np.uint8),
                                       246, 0, 'dove', 'dove', 26, 'book')
        self.assertEqual(step.process_single(image).image.shape, (50, 80))
        self.assertEqual(step.device, 'auto')
        cpu.run.assert_called_once()

    def test_optional_config_loads_without_loading_model_and_uses_raw_color(self):
        pipeline = Pipeline()
        pipeline.load_from_config("config.semantic.yaml")
        available = {"initial_image"}
        for step, source, mask in pipeline.steps:
            self.assertIn(source, available)
            if mask:
                self.assertIn(mask, available)
            if isinstance(step, SemanticIllustrationMaskStep):
                self.assertEqual(source, "oriented_image")
                self.assertIsNone(step._session)
            available.add(step.name)
        self.assertIn("semantic-review", pipeline.output_directory)

    def test_empty_prediction_fails_instead_of_retaining_print_artifacts(self):
        step = self.step(input_size=32)
        step._session = SimpleNamespace(
            get_inputs=lambda: [SimpleNamespace(name="input")],
            run=lambda *_: [np.full((1, 1, 32, 32), -20, np.float32)],
        )
        image = PipelineImageContainer(np.full((50, 80, 3), 220, np.uint8),
                                       355, 0, "swan", "swan", 50, "book")
        with self.assertRaisesRegex(ValueError, "book page 355"):
            step.process_single(image)

    def test_rgb_input_and_original_mask_geometry(self):
        step = self.step(input_size=32)
        def infer(_, feeds):
            batch = feeds["input"]
            np.testing.assert_allclose(batch[0, :, 0, 0],
                                       (np.array([1, 0, 0]) - step.mean) / step.std)
            output = np.full((1, 1, 32, 32), -20, np.float32)
            output[:, :, 8:24, 8:24] = 20
            return [output]
        step._session = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name="input")], run=infer)
        bgr = np.zeros((60, 80, 3), np.uint8)
        bgr[:, :, 2] = 255
        image = PipelineImageContainer(bgr, 1, 0, "bird", "bird", 1, "book")
        mask = step.process_single(image).image
        self.assertEqual(mask.shape, (60, 80))
        self.assertEqual(mask[30, 40], 255)
        self.assertEqual(mask[0, 0], 0)


if __name__ == "__main__":
    unittest.main()
