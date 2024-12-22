from typing import Tuple
import cv2
from pipeline.base import PipelineStep

class ResizeStep(PipelineStep):
    def __init__(self, name, width: int|None = None, height: int|None = None):
        super().__init__(name)
        self.size = (width, height)

    def process_single(self, input_item):
        image = input_item
        if self.size[0] is not None and self.size[1] is not None:
            return cv2.resize(image, self.size)
        elif self.size[0] is not None:
            return cv2.resize(image, (self.size[0], int(self.size[0] * image.shape[0] / image.shape[1])))
        elif self.size[1] is not None:
            return cv2.resize(image, (int(self.size[1] * image.shape[1] / image.shape[0]), self.size[1]))
        else:
            return image