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

def reduce_yellow(image: np.ndarray, original_image: np.ndarray, tolerance: int = 30, preview: bool = False, name: str = None, color: np.ndarray = np.array([245, 235, 225], dtype=np.uint8)) -> tuple[np.ndarray, np.ndarray]: # default [255, 245, 225]
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
        
    # if preview:
    #     cv2.imshow(f'Edges: {name}', resize_preview(additional_contours, 600))
    
    # enhanced_image = cv2.add(image, combined_edges)

    target_color_int = color.astype(np.int16)
    lower_bound = np.clip(target_color_int - tolerance, 0, 255).astype(np.uint8)
    upper_bound = np.clip(target_color_int + tolerance, 0, 255).astype(np.uint8)
    mask = cv2.inRange(image, lower_bound, upper_bound)
    mask = cv2.bitwise_not(mask)
        
    if preview:
        cv2.imshow(f'Original Mask: {name}', resize_preview(mask, 600))
        
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
        
    # subject_mask = cv2.dilate(subject_mask, np.ones((48,48), np.uint8), iterations=1)
        
    additional_contours = cv2.bitwise_and(additional_contours, subject_mask)
        
    mask = cv2.add(mask, additional_contours)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((16,16), np.uint8))
    # mask = cv2.erode(mask, np.ones((9, 9), np.uint8), iterations=1)

    if preview:
        cv2.imshow(f'Original Mask plus Additional Mask: {name}', resize_preview(mask, 600))
    
    # remove small artifacts by finding contours with a area threshold
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if preview:
        p = np.zeros_like(mask)
        cv2.drawContours(p, contours, -1, 255, thickness=cv2.FILLED)
        cv2.imshow(f'Contours: {name}', resize_preview(p, 600))
                
    for contour in contours:
       # Create a temporary mask for the current contour
        contour_mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
        
    # mask = cv2.erode(mask, np.ones((9, 9), np.uint8), iterations=1)
    # mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=1)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 10000:
            cv2.fillPoly(mask, [contour], 0)
            
        cv2.fillPoly(mask, [contour], 255)
    
    if preview:
        cv2.imshow(f'Reduce Yellow Mask: {name}', resize_preview(mask, 600))
    
    neutral_color = np.array([255, 255, 255], dtype=np.uint8)
    normalized_mask = mask.astype(np.float32) / 255.0
    blended_image = (normalized_mask[:, :, None] * image.astype(np.float32) +
                        (1 - normalized_mask[:, :, None]) * neutral_color.astype(np.float32)).astype(np.uint8)
    return blended_image, mask


def crop_image_to_subject(image: np.ndarray, mask: np.ndarray, padding: int = 0, threshold: int = 240, preview: bool = False, name: str = None) -> np.ndarray:
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

    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Create a blank mask and draw all contours
        combined_mask = np.zeros_like(mask)
        for contour in contours:
            cv2.fillPoly(combined_mask, [contour], 255)
        
        # Find merged contours
        merged_contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest merged contour and create a bounding box
        max_area = cv2.contourArea(max(merged_contours, key=cv2.contourArea))
        relevant_contours = [contour for contour in merged_contours if cv2.contourArea(contour) > 0.5 * max_area]
        
        # Create a new mask for relevant contours
        final_mask = np.zeros_like(mask)
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

def remove_text_remains(image: np.ndarray, preview: bool = False, name: str = "") -> np.ndarray:
    cleaned_image = image.copy()
    
    gray = cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.erode(binary, np.ones((16,16), np.uint8), iterations=1)
    
    if preview:
        cv2.imshow(f'Binary: {name}', resize_preview(binary, 600))
    
    height = image.shape[0]
    bottom_10_start = int(height * 0.9)
    mask = np.zeros_like(binary)
    mask[bottom_10_start:, :] = 255
    
    binary = cv2.bitwise_not(binary)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        if y >= bottom_10_start and (y + h) <= height:
            area = cv2.contourArea(contour)
            if area < 10000:
                cv2.drawContours(cleaned_image, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    return cleaned_image
    
def resize_preview(image: np.ndarray, max_size: int = 600) -> np.ndarray:
    return cv2.resize(image, (max_size, int(max_size * image.shape[0] / image.shape[1])))