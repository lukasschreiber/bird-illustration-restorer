import cv2
from pipeline.base import PipelineStep, PipelineImageContainer

class BilateralFilterStep(PipelineStep):
    def __init__(self, name, diameter: int, sigma_color: int, sigma_space: int, pipeline=None):
        super().__init__(name, pipeline)
        self.diameter = diameter
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def process_single(self, input_item: PipelineImageContainer):
        input_item.image = cv2.bilateralFilter(input_item.image, self.diameter, self.sigma_color, self.sigma_space)
        return input_item