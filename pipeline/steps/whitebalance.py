import cv2
from pipeline.base import PipelineStep, PipelineImageContainer
import numpy as np

class WhiteBalanceStep(PipelineStep):
    def __init__(self, name, pipeline=None, k: int = 3, max_size: int = 256, strength: int = 1.0):
        super().__init__(name, pipeline)
        self.k = k
        self.max_size = max_size
        self.strength = strength

    def process_single(self, input_item: PipelineImageContainer):
        input_item.image = self._correct_color_balance(input_item.image, self._detect_background_color_kmeans(input_item.image, self.k, self.max_size), self.strength)
        return input_item
    
    def _detect_background_color_kmeans(self, image: np.ndarray, k: int = 3, max_size: int = 256) -> np.ndarray:
        """
        Detect the dominant background color of an image using k-means clustering.
        
        Parameters:
        image (np.ndarray): The image to analyze.
        k (int): The number of clusters to use in k-means clustering.
        max_size (int): The maximum size of the image for processing, to speed up computation.
        
        Returns:
        np.ndarray: The dominant background color of the image.
        """
        height, width = image.shape[:2]
        scale = max_size / max(height, width)
        if scale < 1:
            resized_image = cv2.resize(image, (int(width * scale), int(height * scale)))
        else:
            resized_image = image

        pixels = resized_image.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        dominant_color = centers[np.argmax(np.bincount(labels.flatten()))]
        return np.uint8(dominant_color)
    
    def _correct_color_balance(self, image: np.ndarray, avg_bg_color: np.ndarray, strength: int = 1.0) -> np.ndarray:
        """
        Correct the color balance of an image based on the average background color.
        The strength parameter controls the intensity of the correction.
        
        Parameters:
        image (np.ndarray): The image to correct.
        avg_bg_color (np.ndarray): The average background color of the image.
        strength (float): The strength of the color correction.
        
        Returns:
        np.ndarray: The color-corrected image.
        """
        image = image.astype(np.float32)
        correction_factors = np.array([avg_bg_color[0] / 128.0, avg_bg_color[1] / 128.0, avg_bg_color[2] / 128.0])
        correction_factors = np.clip(correction_factors, 0.8, 1.2)  # Prevent over-correction
        corrected_image = image * correction_factors * strength
        corrected_image = np.clip(corrected_image, 0, 255).astype(np.uint8)
        
        return corrected_image
