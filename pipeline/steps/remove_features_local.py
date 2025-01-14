import cv2
import numpy as np
from typing import Dict, Tuple
from pipeline.base import PipelineStep, PipelineImageContainer

class RemoveFeaturesLocalStep(PipelineStep):
    def __init__(self, name, pipeline=None):
        super().__init__(name, pipeline)
        self.window_size = 800
        self.title = "[interactive] Remove Features Locally"
        self.points: list[Tuple[int, int]] = []
        self.rendered_points: list[Tuple[int, int]] = []
        self.patch_masks: Dict[Tuple[int, int], np.ndarray] = {}  # Corrected type hint


    def process_single(self, input_item: PipelineImageContainer):
        # Resize the image while maintaining aspect ratio
        self.original_image = input_item.image
        self.image = self.resize_image(self.original_image)
        self.title = f"[interactive] Remove Features Locally - {input_item.english_label}"
        self.display_image = self.image.copy()

        # Initialize window and set mouse callback
        cv2.namedWindow(self.title, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.title, self.mouse_callback)
        
        self.update_display()
        
        cv2.waitKey(0)
        cv2.destroyWindow(self.title)

        input_item.image = self.original_image
        return input_item

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        max_size = self.window_size
        h, w = image.shape[:2]
        aspect_ratio = w / h
        if w > h:
            new_w = max_size
            new_h = int(max_size / aspect_ratio)
        else:
            new_h = max_size
            new_w = int(max_size * aspect_ratio)
        
        return cv2.resize(image, (new_w, new_h))

    def mouse_callback(self, event, x, y, flags, param):
        # show a green dot at the current location
        self.display_image = self.image.copy()
        cv2.circle(self.display_image, (x, y), 5, (0, 0, 0), -1)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append(self.point_preview_to_original((x, y)))
        
        if event == cv2.EVENT_RBUTTONDOWN:
            # find the closest point and remove it, but only if it's within a certain radius
            min_distance = 30
            closest_point = None
            for point in self.points:
                distance = np.linalg.norm(np.array(point) - np.array([x, y]))
                if distance < min_distance:
                    min_distance = distance
                    closest_point = point
            
            if closest_point:
                self.points.remove(closest_point)
                self.patch_masks.pop(closest_point, None)
            
        self.update_display()

    def update_display(self):
        if self.points != self.rendered_points:
            # make changes to original image
            self.original_image = self.locally_remove_patches(self.original_image, self.points)
            self.image = self.resize_image(self.original_image)
            self.display_image = self.image.copy()
        
        for point in self.points:
            cv2.circle(self.display_image, self.point_original_to_preview(point), 5, (0, 0, 255), -1)
        
        self.rendered_points = self.points.copy()
        cv2.imshow(self.title, self.display_image)
        key = cv2.waitKey(1)

        if key == 27:  # ESC key to exit
            cv2.destroyWindow(self.title)
            
    def point_preview_to_original(self, point: Tuple[int, int]) -> Tuple[int, int]:
        """
        Convert a point from the preview image to the original image.
        
        Parameters:
        point (tuple[int, int]): The point in the preview image.
        
        Returns:
        tuple[int, int]: The point in the original image.
        """
        x, y = point
        scale = self.original_image.shape[1] / self.image.shape[1]
        return int(x * scale), int(y * scale)
    
    def point_original_to_preview(self, point: Tuple[int, int]) -> Tuple[int, int]:
        """
        Convert a point from the original image to the preview image.
        
        Parameters:
        point (tuple[int, int]): The point in the original image.
        
        Returns:
        tuple[int, int]: The point in the preview image.
        """
        x, y = point
        scale = self.image.shape[1] / self.original_image.shape[1]
        return int(x * scale), int(y * scale)
        
            
    def locally_remove_patches(self, image: np.ndarray, points: list[Tuple[int, int]]) -> np.ndarray:
        """
        Locally remove patches from an image.
        
        Parameters:
        image (np.ndarray): The input image.
        points (list[tuple[int, int]]): The points to remove.
        
        Returns:
        np.ndarray: The image with patches removed.
        """        
        patch_size = 15
        color_threshold = 20  # Color similarity threshold
        max_patch_size = 2000 * 2000

        for point in points:
            if point not in self.patch_masks:
                x, y = point
                # Ensure the patch boundaries are within the image dimensions
                y_min = max(0, y - patch_size)
                y_max = min(image.shape[0], y + patch_size)
                x_min = max(0, x - patch_size)
                x_max = min(image.shape[1], x + patch_size)

                # Find the average color of the patch
                patch = image[y_min:y_max, x_min:x_max]
                average_color = np.mean(patch, axis=(0, 1))
                        
                patch_mask = np.zeros(image.shape[:2], dtype=np.uint8)

                def flood_fill(x, y):
                    # Create a queue for BFS (breadth-first search)
                    queue = [(x, y)]
                    patch_mask[y, x] = 255  # Mark the starting point as part of the patch

                    patch_size = 0
                    while queue and patch_size < max_patch_size:
                        patch_size += 1
                        cx, cy = queue.pop(0)

                        # Explore the 4-connected neighbors (up, down, left, right)
                        for nx, ny in [(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)]:
                            if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0]:
                                # Check if the pixel is within the color threshold
                                if patch_mask[ny, nx] == 0 and np.linalg.norm(average_color - image[ny, nx]) <= color_threshold and not np.array_equal(image[ny, nx], [255, 255, 255]):
                                    patch_mask[ny, nx] = 255
                                    queue.append((nx, ny))
                                    
                flood_fill(x, y)
                kernel = np.ones((4, 4), np.uint8)
                dilated = cv2.dilate(patch_mask, kernel, iterations=1)
                eroded = cv2.erode(dilated, kernel, iterations=1)
                self.patch_masks[point] = eroded


        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for point in points:
            combined_mask |= self.patch_masks[point]
        image[combined_mask != 0] = [255, 255, 255]

        return image
