import cv2
from pipeline.base import PipelineStep, PipelineImageContainer
import numpy as np

class FindSubjectStep(PipelineStep):
    def __init__(
        self,
        name,
        shift: int,
        threshold: float,
        low_contrast_boost: bool = True,
        blackhat_kernel: int = 15,
        min_component_area_ratio: float = 0.001,
        close_gaps_kernel: int = 13,
        fill_holes: bool = True,
        pipeline=None,
        method: str = "legacy",
        detection_max_size: int = 1600,
        edge_threshold: float = 6,
        edge_close_kernel: int = 7,
    ):
        if pipeline is None:
            raise ValueError("Pipeline must be provided")
        super().__init__(name, pipeline)
        self.shift = shift
        self.threshold = threshold
        self.low_contrast_boost = low_contrast_boost
        self.blackhat_kernel = blackhat_kernel
        self.min_component_area_ratio = min_component_area_ratio
        self.close_gaps_kernel = close_gaps_kernel
        self.fill_holes = fill_holes
        if method not in {"legacy", "source_edges"}:
            raise ValueError("Unknown subject detection method")
        if detection_max_size < 32 or edge_threshold <= 0 or edge_close_kernel < 1:
            raise ValueError("Invalid source edge detection parameters")
        self.method = method
        self.detection_max_size = detection_max_size
        self.edge_threshold = edge_threshold
        self.edge_close_kernel = edge_close_kernel

    def _source_edge_mask(self, gray: np.ndarray) -> np.ndarray:
        # Detect at a fixed scale so scan resolution does not change gap sizes.
        h, w = gray.shape
        scale = min(1.0, self.detection_max_size / max(h, w))
        working = cv2.resize(gray, (max(1, round(w * scale)), max(1, round(h * scale))),
                             interpolation=cv2.INTER_AREA) if scale < 1 else gray
        blurred = cv2.GaussianBlur(working, (5, 5), 0)
        edges = cv2.Canny(blurred, self.edge_threshold, self.edge_threshold * 3)
        size = self.edge_close_kernel | 1
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((size, size), np.uint8))
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        def is_scan_border(contour):
            x, y, cw, ch = cv2.boundingRect(contour)
            height, width = working.shape
            horizontal = cw > width * 0.5 and ch < height * 0.06
            vertical = ch > height * 0.5 and cw < width * 0.06
            # Broken scan edges can be much shorter than half the page. Reject
            # slender detached fragments only in the outermost margin too.
            horizontal_fragment = (cw > width * .15 and cw > ch * 8 and
                                   ch < height * .04 and
                                   (y < height * .04 or y + ch > height * .96))
            vertical_fragment = (ch > height * .15 and ch > cw * 8 and
                                 cw < width * .04 and
                                 (x < width * .04 or x + cw > width * .96))
            return ((horizontal and (y < height * 0.08 or y + ch > height * 0.92)) or
                    (vertical and (x < width * 0.08 or x + cw > width * 0.92)) or
                    horizontal_fragment or vertical_fragment)

        contours = [c for c in contours if not is_scan_border(c)]
        mask = np.zeros_like(working)
        largest = max((cv2.contourArea(c) for c in contours), default=0)
        # Keep multiple birds and disconnected illustration details, omit lettering.
        minimum = max(working.size * self.min_component_area_ratio, largest * 0.01)
        for contour in contours:
            if cv2.contourArea(contour) >= minimum:
                cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
        if not np.any(mask):
            # An uncertain detection should preserve the scan, not erase it.
            return np.full_like(gray, 255)
        if self.fill_holes:
            mask = self._fill_binary_holes(mask)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    def _fill_binary_holes(self, binary_mask: np.ndarray) -> np.ndarray:
        # A white corner is foreground, so seed flood fill outside the image.
        flood = cv2.copyMakeBorder(binary_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        h, w = flood.shape
        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(flood, flood_mask, (0, 0), (255,))
        holes = cv2.bitwise_not(flood)[1:-1, 1:-1]
        return cv2.bitwise_or(binary_mask, holes)

    def process_single(self, image: PipelineImageContainer):
        gray = image.image
        if self.method == "source_edges":
            image.image = self._source_edge_mask(gray)
            return image
        equalized = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(equalized, (9, 9), 0)
       
        # blurred[:50] = 255
        # blurred[-100:] = 255
        # blurred[:, :50] = 255
        # blurred[:, -50:] = 255
        
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        edges = cv2.Canny(thresh, threshold1=30, threshold2=100)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((16, 16), np.uint8))

        if self.low_contrast_boost:
            kernel_size = max(3, self.blackhat_kernel)
            if kernel_size % 2 == 0:
                kernel_size += 1

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            local_contrast = clahe.apply(gray)

            blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            dark_outlines = cv2.morphologyEx(local_contrast, cv2.MORPH_BLACKHAT, blackhat_kernel)
            dark_outlines = cv2.GaussianBlur(dark_outlines, (5, 5), 0)
            _, dark_outlines = cv2.threshold(dark_outlines, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            outline_edges = cv2.Canny(dark_outlines, threshold1=20, threshold2=80)
            outline_edges = cv2.dilate(outline_edges, np.ones((3, 3), np.uint8), iterations=1)
            outline_closed = cv2.morphologyEx(outline_edges, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

            closed = cv2.bitwise_or(closed, outline_closed)
                
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            image.image = np.zeros_like(gray, dtype=np.uint8)
            return image

        min_component_area = gray.shape[0] * gray.shape[1] * self.min_component_area_ratio
        contours = [contour for contour in contours if cv2.contourArea(contour) >= min_component_area]

        if not contours:
            image.image = np.zeros_like(gray, dtype=np.uint8)
            return image

        max_area = cv2.contourArea(max(contours, key=cv2.contourArea))
        relevant_contours = [contour for contour in contours if cv2.contourArea(contour) > self.threshold * max_area]

        mask = np.zeros_like(gray, dtype=np.uint8)
        for contour in relevant_contours:
            cv2.fillPoly(mask, [contour], (255,))

        if np.count_nonzero(mask) == 0:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.fillPoly(mask, [largest_contour], (255,))

        close_kernel_size = max(3, self.close_gaps_kernel)
        if close_kernel_size % 2 == 0:
            close_kernel_size += 1
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

        if self.fill_holes:
            mask = self._fill_binary_holes(mask)
            
        image.image = mask
        return image
    
def resize_preview(image: np.ndarray, max_size: int = 600) -> np.ndarray:
    return cv2.resize(image, (max_size, int(max_size * image.shape[0] / image.shape[1])))
