import cv2
from pipeline.base import PipelineStep

class GrayscaleStep(PipelineStep):
    def __init__(self, name):
        super().__init__(name)

    def process_single(self, input_item):
        image = input_item
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)