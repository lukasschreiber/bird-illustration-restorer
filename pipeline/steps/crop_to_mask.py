import cv2
from pipeline.base import PipelineStep, PipelineImageContainer
import numpy as np

class CropToMaskStep(PipelineStep):
    def __init__(self, name, background_color: np.ndarray = np.array([255, 255, 255], dtype=np.uint8), padding: int = 0, pipeline=None, feather_radius: float = 0):
        super().__init__(name, pipeline)
        self.background_color = tuple(int(channel) for channel in background_color)
        self.padding = padding
        if feather_radius < 0:
            raise ValueError("Feather radius cannot be negative")
        self.feather_radius = feather_radius

    def process_masked_single(self, image: PipelineImageContainer, mask: PipelineImageContainer | None = None) -> PipelineImageContainer:
        if mask is None:
            raise ValueError("Mask must be provided")
        
        img = image.image.copy()
        
        if mask.image.shape != img.shape[:2] or mask.image.dtype != np.uint8:
            raise ValueError("Crop mask must be an aligned uint8 image")
        if not np.any(mask.image):
            raise ValueError("Cannot crop to an empty subject mask")
        if self.feather_radius:
            # Blend inward only: never reintroduce paper outside the selected mask.
            distance = cv2.distanceTransform(np.uint8(mask.image > 0), cv2.DIST_L2, 3)
            alpha = np.clip(distance / self.feather_radius, 0, 1)
            alpha = alpha * alpha * (3 - 2 * alpha)
            img = np.rint(img * alpha[:, :, None] +
                          np.asarray(self.background_color, dtype=np.float32) * (1-alpha[:, :, None])).astype(np.uint8)
        else:
            img[mask.image == 0] = self.background_color

        x, y, w, h = cv2.boundingRect(mask.image)
        
        x = max(0, x)
        y = max(0, y)
        w = min(img.shape[1] - x, w)
        h = min(img.shape[0] - y, h)
        
        cropped_image = img[y:y + h, x:x + w]
        
        image.image = cv2.copyMakeBorder(cropped_image, self.padding, self.padding, self.padding, self.padding, cv2.BORDER_CONSTANT, value=self.background_color)
        return image
