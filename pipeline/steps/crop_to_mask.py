import cv2
from pipeline.base import PipelineStep, PipelineImageContainer
import numpy as np

class CropToMaskStep(PipelineStep):
    def __init__(self, name, background_color: np.ndarray = np.array([255, 255, 255], dtype=np.uint8), padding: int = 0, pipeline=None):
        super().__init__(name, pipeline)
        self.background_color = background_color
        self.padding = padding

    def process_masked_single(self, image: PipelineImageContainer, mask: PipelineImageContainer | None = None) -> PipelineImageContainer:
        if mask is None:
            raise ValueError("Mask must be provided")
        
        img = image.image.copy()
        
        # fill everything that is black in the mask with white
        img[mask.image == 0] = self.background_color

        x, y, w, h = cv2.boundingRect(mask.image)
        
        x = max(0, x)
        y = max(0, y)
        w = min(img.shape[1] - x, w)
        h = min(img.shape[0] - y, h)
        
        cropped_image = img[y:y + h, x:x + w]
        
        image.image = cv2.copyMakeBorder(cropped_image, self.padding, self.padding, self.padding, self.padding, cv2.BORDER_CONSTANT, value=self.background_color)
        return image