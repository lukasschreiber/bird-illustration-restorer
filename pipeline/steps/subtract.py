import cv2
import numpy as np
from pipeline.base import PipelineStep, PipelineImageContainer

class ReduceYellowBySubtractionStep(PipelineStep):
    def __init__(self, name, image: str, shift: int, threshold: float, pipeline=None):
        super().__init__(name, pipeline)
        self.second_image = image
        self.shift = shift
        self.threshold = threshold

    def process_single(self, input_item: PipelineImageContainer):
        second_image = self.pipeline.cache[self.pipeline.get_cache_key(input_item, self.second_image)]
        input_item.image = cv2.absdiff(input_item.image, second_image.image)
        input_item.image = cv2.cvtColor(input_item.image, cv2.COLOR_BGR2GRAY)
        _, input_item.image = cv2.threshold(input_item.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # close morphological operation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        input_item.image = cv2.morphologyEx(input_item.image, cv2.MORPH_CLOSE, kernel)
        
        # find contours
        contours, _ = cv2.findContours(input_item.image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        max_area = cv2.contourArea(max(contours, key=cv2.contourArea))
        relevant_contours = [contour for contour in contours if cv2.contourArea(contour) > self.threshold * max_area]
                
        mask = np.zeros_like(input_item.image, dtype=np.uint8)
        for contour in relevant_contours:
            cv2.fillPoly(mask, [contour], 255)
            
        input_item.image = mask
        
        return input_item