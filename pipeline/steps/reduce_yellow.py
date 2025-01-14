import cv2
from pipeline.base import PipelineStep, PipelineImageContainer
import numpy as np

class ReduceYellowStep(PipelineStep):
    def __init__(self, name, tolerance: int = 30, color: np.ndarray = np.array([245, 235, 225], dtype=np.uint8), grayscale_image: str = None, pipeline=None):
        self.tolerance = tolerance
        self.color = np.array(color, dtype=np.uint8)
        self.grayscale_image = grayscale_image
        super().__init__(name, pipeline)

    def process_single(self, input_item: PipelineImageContainer):
        # find the image where instance and page are the same in self.pipeline.cache[grayscale_image]
        original_image = self.pipeline.cache[self.pipeline.get_cache_key(input_item, self.grayscale_image)]

        corrected_image = self._reduce_yellow(input_item.image, original_image.image, self.tolerance, self.color)
        
        input_item.image = corrected_image
        return input_item
        
    def _reduce_yellow(self, 
            image: np.ndarray, 
            original_image: np.ndarray, 
            tolerance: int, 
            color: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]: # default [255, 245, 225]
        
        original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray[gray > 250] = 255
        equalized = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
        
        edges_original = cv2.Canny(original_gray, threshold1=50, threshold2=150)
        edges_processed = cv2.Canny(blurred, threshold1=50, threshold2=150)
        combined_edges = cv2.addWeighted(edges_original, 0.7, edges_processed, 0.3, 0)
        combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))
        
        enhanced_blended = cv2.addWeighted(blurred, 0.8, cv2.bitwise_not(combined_edges), 0.2, 0)
        
        subject = cv2.Canny(enhanced_blended, threshold1=300, threshold2=400)
        # subject = cv2.dilate(subject, np.ones((9,9), np.uint8), iterations=1)
        subject = cv2.morphologyEx(subject, cv2.MORPH_CLOSE, np.ones((32,32), np.uint8))
        contours, _ = cv2.findContours(subject, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        additional_contours = np.zeros_like(subject)
        
        cv2.drawContours(additional_contours, contours, -1, 255, thickness=cv2.FILLED)

        target_color_int = color.astype(np.int16)
        lower_bound = np.clip(target_color_int - tolerance, 0, 255).astype(np.uint8)
        upper_bound = np.clip(target_color_int + tolerance, 0, 255).astype(np.uint8)
        mask = cv2.inRange(image, lower_bound, upper_bound)
        mask = cv2.bitwise_not(mask)
            
        # if preview:
        #     cv2.imshow(f'Original Mask: {name}', resize_preview(mask, 600))
            
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) < 10000:
                cv2.fillPoly(mask, [contour], 0)
                continue
            
            # if one side of the contour spans the entire image, it is likely a border
            x, y, w, h = cv2.boundingRect(contour)
            if (x == 0 and w == mask.shape[1]) or (y == 0 and h == mask.shape[0]) or (x + w == mask.shape[1] and w == mask.shape[1]) or (y + h == mask.shape[0] and h == mask.shape[0]):
                cv2.fillPoly(mask, [contour], 0)
                continue
        
        subject_mask = np.zeros_like(mask)
        for contour in contours:
            cv2.fillPoly(subject_mask, [contour], 255)
                        
        additional_contours = cv2.bitwise_and(additional_contours, subject_mask)
            
        mask = cv2.add(mask, additional_contours)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((16,16), np.uint8))
        # mask = cv2.erode(mask, np.ones((9, 9), np.uint8), iterations=1)

        # if preview:
        #     cv2.imshow(f'Original Mask plus Additional Mask: {name}', resize_preview(mask, 600))
        
        # remove small artifacts by finding contours with a area threshold
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # if preview:
        #     p = np.zeros_like(mask)
        #     cv2.drawContours(p, contours, -1, 255, thickness=cv2.FILLED)
        #     cv2.imshow(f'Contours: {name}', resize_preview(p, 600))
                    
        for contour in contours:
        # Create a temporary mask for the current contour
            contour_mask = np.zeros_like(mask, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
            
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 10000:
                cv2.fillPoly(mask, [contour], 0)
                
            cv2.fillPoly(mask, [contour], 255)
        
        # if preview:
        #     cv2.imshow(f'Reduce Yellow Mask: {name}', resize_preview(mask, 600))
        
        neutral_color = np.array([255, 255, 255], dtype=np.uint8)
        normalized_mask = mask.astype(np.float32) / 255.0
        blended_image = (normalized_mask[:, :, None] * image.astype(np.float32) + (1 - normalized_mask[:, :, None]) * neutral_color.astype(np.float32)).astype(np.uint8)
        return blended_image
