import cv2
from pipeline.base import PipelineStep, PipelineImageContainer

class GrayscaleStep(PipelineStep):
    def __init__(self, name, pipeline=None):
        super().__init__(name, pipeline)

    def process_single(self, input_item: PipelineImageContainer):
        input_item.image = cv2.cvtColor(input_item.image, cv2.COLOR_BGR2GRAY)
        return input_item