import cv2
from pipeline.base import PipelineStep, PipelineImageContainer
import numpy as np

class MaskColorStep(PipelineStep):
    def __init__(self, name, color: np.ndarray = np.array([255, 255, 255], dtype=np.uint8), pipeline=None):
        super().__init__(name, pipeline)
        self.color = color

    def process_single(self, input_item: PipelineImageContainer):
        input_item.image = np.all(input_item.image != self.color, axis=-1).astype(np.uint8) * 255

        return input_item