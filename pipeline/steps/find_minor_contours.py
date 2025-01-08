import cv2
from pipeline.base import PipelineStep, PipelineImageContainer
import numpy as np

class FindMinorContoursStep(PipelineStep):
    def __init__(self, name, pipeline=None):
        super().__init__(name, pipeline)

    def process_single(self, input_item: PipelineImageContainer):
        
        contours, _ = cv2.findContours(input_item.image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = cv2.contourArea(max(contours, key=cv2.contourArea))
        relevant_contours = [contour for contour in contours if cv2.contourArea(contour) > 0.5 * max_area]

        mask = np.zeros_like(input_item.image)
        for contour in relevant_contours:
            cv2.fillPoly(mask, [contour], 255)
            
        input_item.image = mask
        return input_item