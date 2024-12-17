import numpy as np
import cv2

def detect_background_color_kmeans(image: np.ndarray, k: int = 3, max_size: int = 256) -> np.ndarray:
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

def correct_color_balance(image: np.ndarray, avg_bg_color: np.ndarray, strength: int = 1.0) -> np.ndarray:
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

def reduce_yellow(image: np.ndarray, original_image: np.ndarray, tolerance: int = 30, preview: bool = False, name: str = None, color: np.ndarray = np.array([245, 235, 225], dtype=np.uint8), feather_intensity: int = 5) -> np.ndarray: # default [255, 245, 225]
    """
    Reduce the yellow color in an image by replacing it with white.
    Also removes small artifacts and contours if their contrast is under a threshold.
    
    Parameters:
    image (np.ndarray): The image to process.
    tolerance (int): The color tolerance for the yellow color.
    preview (bool): Whether to display a preview of the processed image.
    name (str): The name of the image for display purposes, if preview is enabled.
    color (np.ndarray): The target color to reduce (default is yellow).
    
    Returns:
    np.ndarray: The processed image with reduced yellow
    """
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray[gray > 250] = 255
    equalized = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    
    target_color_int = color.astype(np.int16)
    lower_bound = np.clip(target_color_int - tolerance, 0, 255).astype(np.uint8)
    upper_bound = np.clip(target_color_int + tolerance, 0, 255).astype(np.uint8)
    mask = cv2.inRange(image, lower_bound, upper_bound)
    mask = cv2.bitwise_not(mask)
    
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
    
    if preview:
        cv2.imshow(f'Blurred: {name}', resize_preview(gray, 600))
    
    # remove small artifacts by finding contours with a area threshold
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    canny_preview = np.zeros_like(mask, dtype=np.uint8)
        
    i = 0
    for contour in contours:
        i += 1
        if cv2.contourArea(contour) < 10000:
            cv2.fillPoly(mask, [contour], 0)
            continue
        
        # if one side of the contour spans the entire image, it is likely a border
        x, y, w, h = cv2.boundingRect(contour)
        if (x == 0 and w == mask.shape[1]) or (y == 0 and h == mask.shape[0]) or (x + w == mask.shape[1] and w == mask.shape[1]) or (y + h == mask.shape[0] and h == mask.shape[0]):
            cv2.fillPoly(mask, [contour], 0)
            continue
        
       # Create a temporary mask for the current contour
        contour_mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # if preview:
        #     cv2.imshow(f"contour_mask {i}", resize_preview(contour_mask))

        # Compute the min/max color values in the contour region
        region = cv2.bitwise_and(image, image, mask=contour_mask)
        region_pixels = region[contour_mask == 255]
        min_val = region_pixels.min(axis=0)
        max_val = region_pixels.max(axis=0)
        
        # Check the color range within the contour
        color_range = np.max(max_val - min_val)
        if color_range < 30:  # Low contrast threshold
            cv2.fillPoly(mask, [contour], 0)
            
        extended_mask = cv2.dilate(contour_mask, np.ones((256, 256), np.uint8), iterations=1)
      
        edges = cv2.Canny(blurred, 200, 300)
        edges_original = cv2.Canny(original_gray, threshold1=200, threshold2=300)
        edges = cv2.bitwise_and(edges, edges, mask=extended_mask)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        combined_edges = cv2.addWeighted(edges_original, 0.7, edges, 0.3, 0)
        
        canny_contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for canny_contour in canny_contours:
            perimeter = cv2.arcLength(canny_contour, True)
            # this definition is probably incorrect, but it works for now
            compactness = perimeter ** 2 / (4 * np.pi * cv2.contourArea(canny_contour)) if cv2.contourArea(canny_contour) != 0 else 0
        
            if cv2.contourArea(canny_contour) > 5000 and compactness < 2550:
                print(f"compactness: {compactness}")
                cv2.fillPoly(mask, [canny_contour], 255)
                cv2.fillPoly(canny_preview, [canny_contour], 255)
    
    if feather_intensity > 0:
        mask = cv2.GaussianBlur(mask, (feather_intensity * 2 + 1, feather_intensity * 2 + 1), 0)
        
    mask = cv2.erode(mask, np.ones((9, 9), np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=1)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 10000:
            cv2.fillPoly(mask, [contour], 0)
            
        cv2.fillPoly(mask, [contour], 255)
    
    if preview:
        cv2.imshow(f'Reduce Yellow Mask: {name}', resize_preview(mask, 600))
        cv2.imshow(f'Canny Preview: {name}', resize_preview(canny_preview, 600))
    
    neutral_color = np.array([255, 255, 255], dtype=np.uint8)
    normalized_mask = mask.astype(np.float32) / 255.0
    blended_image = (normalized_mask[:, :, None] * image.astype(np.float32) +
                        (1 - normalized_mask[:, :, None]) * neutral_color.astype(np.float32)).astype(np.uint8)
    return blended_image


def crop_image_to_subject(image: np.ndarray, padding: int = 0, threshold: int = 240, preview: bool = False, name: str = None) -> np.ndarray:
    """
    Crop an image to the subject based on a luminance threshold.
    The subject is assumed to be the darkest part of the image.
    Padding is added around the cropped subject as a plain wide border.
    
    Parameters:
    image (np.ndarray): The image to crop.
    padding (int): The padding to add around the cropped subject.
    threshold (int): The luminance threshold for cropping.
    preview (bool): Whether to display a preview of the cropped image.
    name (str): The name of the image for display purposes, if preview is enabled.
    
    Returns:
    np.ndarray: The cropped image.
    """
    
    luminance = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(luminance, threshold, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    
    # Erode and dilate to refine the mask
    kernel_erode = np.ones((16, 16), np.uint8)
    kernel_dilate = np.ones((48, 48), np.uint8)
    mask_inv = cv2.erode(mask_inv, kernel_erode, iterations=1)
    mask_inv = cv2.dilate(mask_inv, kernel_dilate, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Create a blank mask and draw all contours
        combined_mask = np.zeros_like(mask_inv)
        for contour in contours:
            cv2.fillPoly(combined_mask, [contour], 255)
        
        # Dilate the combined mask to merge nearby contours
        # TODO Omitting this could improve reduction of non-relevant contours but that only works if yellow correction is better at keeping details inside of birds
        merged_mask = cv2.dilate(combined_mask, np.ones((120, 120), np.uint8), iterations=1)
        
        # Find merged contours
        merged_contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest merged contour and create a bounding box
        max_area = cv2.contourArea(max(merged_contours, key=cv2.contourArea))
        relevant_contours = [contour for contour in merged_contours if cv2.contourArea(contour) > 0.5 * max_area]
        
        # Create a new mask for relevant contours
        final_mask = np.zeros_like(mask_inv)
        for contour in relevant_contours:
            cv2.fillPoly(final_mask, [contour], 255)
        
        if preview:
            cv2.imshow(f'Crop Mask: {name}', resize_preview(final_mask, 600))
            
        # make everything outside the mask white
        image[final_mask == 0] = [255, 255, 255]
        
        # final_mask = cv2.dilate(final_mask, np.ones((48, 48), np.uint8), iterations=1)
        
        # Get the bounding box that includes all relevant contours
        x, y, w, h = cv2.boundingRect(final_mask)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        # Crop the image to the adjusted bounding box
        cropped_image = image[y:y + h, x:x + w]
        
        return cropped_image

    return image 

def find_subject_edges(image: np.ndarray, preview: bool = False, name: str = "") -> np.ndarray:
    # use the Canny edge detector to find edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Combine with Canny for stronger edges
    canny_edges = cv2.Canny(gray, 150, 400)  # Tune thresholds for weak edges
    edges = cv2.bitwise_or(adaptive_thresh, canny_edges)
    
    kernel = np.ones((5, 5), np.uint8)  # Larger kernel for better closure
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # closed_edges = cv2.erode(closed_edges, np.ones((3, 3), np.uint8), iterations=1)

    # Step 5: Find Contours
    contours, hierarchy = cv2.findContours(
        closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Step 6: Draw the largest contour (assume it includes both birds)
    output_image = np.zeros_like(gray)
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.fillPoly(output_image, [largest_contour], 255)
    
    if preview:
        cv2.imshow(f"Edges {name}", resize_preview(output_image))
    
def resize_preview(image: np.ndarray, max_size: int = 600) -> np.ndarray:
    return cv2.resize(image, (max_size, int(max_size * image.shape[0] / image.shape[1])))