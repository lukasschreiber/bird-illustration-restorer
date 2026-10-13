import cv2
from pipeline.base import PipelineStep, PipelineImageContainer
import numpy as np

class BrightnessContrastStep(PipelineStep):
    def __init__(self, name, brightness: int, contrast: int, pipeline=None, soft_highlights: bool = False):
        super().__init__(name, pipeline)
        self.contrast = contrast
        self.brightness = brightness
        self.soft_highlights = soft_highlights

    def process_single(self, input_item: PipelineImageContainer):
        input_item.image = np.int16(input_item.image)
        input_item.image = (
            input_item.image * (self.contrast / 127 + 1) - self.contrast + self.brightness
        )
        if self.soft_highlights:
            above = np.maximum(input_item.image - 240, 0)
            input_item.image = np.where(input_item.image > 240,
                                       240 + 15 * (-np.expm1(-above / 15)), input_item.image)
        input_item.image = np.clip(input_item.image, 0, 255)
        input_item.image = np.uint8(input_item.image)
        return input_item
