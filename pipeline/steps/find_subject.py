import cv2
from pipeline.base import PipelineStep, PipelineImageContainer
import numpy as np

class FindSubjectStep(PipelineStep):
    def __init__(self, name, shift: int, threshold: float, pipeline=None):
        super().__init__(name, pipeline)
        self.shift = shift
        self.threshold = threshold

    def process_single(self, input_item: PipelineImageContainer):
        equalized = cv2.equalizeHist(input_item.image)
        blurred = cv2.GaussianBlur(equalized, (9, 9), 0)
        
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        edges = cv2.Canny(thresh, threshold1=30, threshold2=100)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((32, 32), np.uint8))
                
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = cv2.contourArea(max(contours, key=cv2.contourArea))
        relevant_contours = [contour for contour in contours if cv2.contourArea(contour) > self.threshold * max_area]

        mask = np.zeros_like(input_item.image, dtype=np.uint8)
        for contour in relevant_contours:
            cv2.fillPoly(mask, [contour], 255)
            
        input_item.image = mask
        return input_item
    
def resize_preview(image: np.ndarray, max_size: int = 600) -> np.ndarray:
    return cv2.resize(image, (max_size, int(max_size * image.shape[0] / image.shape[1])))