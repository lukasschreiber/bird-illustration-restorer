import cv2
import numpy as np

from pipeline.base import PipelineStep


class FadePaintedWashStep(PipelineStep):
    """Fade exposed cool/neutral wash, with subject and scenery protection."""

    def __init__(self, name, protected_mask, pipeline=None, max_size=1600,
                 strength=1.0, min_brightness=180, max_chroma=35,
                 detail_floor=1.0, detail_ceiling=5.0, protection_radius=3):
        super().__init__(name, pipeline)
        if max_size < 32 or not 0 <= strength <= 1:
            raise ValueError("Invalid wash size or strength")
        if not 0 <= detail_floor < detail_ceiling or protection_radius < 0:
            raise ValueError("Invalid wash detail protection")
        self.protected_mask = protected_mask
        self.max_size = max_size
        self.strength = strength
        self.min_brightness = min_brightness
        self.max_chroma = max_chroma
        self.detail_floor = detail_floor
        self.detail_ceiling = detail_ceiling
        self.protection_radius = protection_radius
        self.last_fade = None

    def _fade_amount(self, source, illustration, protected):
        h, w = source.shape[:2]
        scale = min(1.0, self.max_size / max(h, w))
        size = (max(1, round(w*scale)), max(1, round(h*scale)))
        small = cv2.resize(source, size, interpolation=cv2.INTER_AREA)
        selected = cv2.resize(illustration, size, interpolation=cv2.INTER_NEAREST) > 0
        guard = cv2.resize(protected, size, interpolation=cv2.INTER_NEAREST)
        raw_guard = guard > 0
        radius = self.protection_radius
        guard = cv2.dilate(guard, np.ones((radius*2+1, radius*2+1), np.uint8)) > 0
        # Enclosed gaps in the learned subject mask are uncertain pale detail.
        padded = cv2.copyMakeBorder(np.uint8(guard)*255, 1, 1, 1, 1,
                                    cv2.BORDER_CONSTANT, value=0)
        flood = padded.copy()
        cv2.floodFill(flood, None, (0, 0), 255)
        guard |= cv2.bitwise_not(flood)[1:-1, 1:-1] > 0
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        base = cv2.GaussianBlur(gray, (0, 0), 2)
        detail = cv2.GaussianBlur(np.abs(gray-base), (0, 0), 3)
        edges = cv2.Canny(base.astype(np.uint8), 15, 45)
        # Wash grain and its ragged pale boundary also produce Canny edges.
        # Protect edges supported by darker ink, not every pale wash boundary.
        nearby_dark = cv2.erode(base, np.ones((7, 7), np.uint8)) < 205
        edges[~nearby_dark] = 0
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8)) > 0
        colors = small.astype(np.float32)
        chroma = colors.max(axis=2) - colors.min(axis=2)
        # Preserve warm earth and ochre scenery. This targets gray/blue sky wash.
        cool_or_neutral = colors[:, :, 0] >= colors[:, :, 2] - 3
        # Filled holes and the safety margin can contain sky between reeds or
        # beside a pale bird. Release only smooth, distinctly blue wash there;
        # neutral feather shading and the learned foreground stay protected.
        blue_wash = ((colors[:, :, 0] - colors[:, :, 2] >= 6) &
                     (gray >= self.min_brightness) &
                     (chroma <= self.max_chroma) &
                     (detail < self.detail_ceiling) & ~edges & ~raw_guard)
        guard &= ~blue_wash
        eligible = (selected & ~guard & ~edges & (gray >= self.min_brightness) &
                    (chroma <= self.max_chroma) & cool_or_neutral &
                    (detail < self.detail_ceiling))
        # Only remove areas exposed to the exterior, not enclosed pale detail.
        exterior = np.uint8(~selected) * 255
        exposed = cv2.dilate(exterior, np.ones((5, 5), np.uint8)) > 0
        count, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(eligible), connectivity=8)
        touched = np.unique(labels[eligible & exposed])
        accepted = np.zeros(count, bool)
        accepted[touched] = True
        # Broad open gaps between scenery can be enclosed by stems/outline edges.
        accepted |= stats[:, cv2.CC_STAT_AREA] >= gray.size * .002
        blue_counts = np.bincount(labels[eligible & blue_wash], minlength=count)
        accepted |= blue_counts >= max(16, gray.size * .00002)
        accepted[0] = False
        connected = accepted[labels]
        t = np.clip((detail-self.detail_floor) / (self.detail_ceiling-self.detail_floor), 0, 1)
        opacity = 1 - t*t*(3-2*t)
        # Fully clear grain only in open wash far from dark scenery. Retain the
        # softer fade near water strokes, where flat shading belongs to the art.
        ink_density = cv2.GaussianBlur(np.float32((gray < 190) | edges), (0, 0), 12)
        open_wash = np.uint8(connected & (ink_density < .005)) * 255
        # Bridge tiny grain holes and pale boundary fragments in confirmed wash.
        open_wash = cv2.morphologyEx(open_wash, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        open_wash = cv2.dilate(open_wash, np.ones((3, 3), np.uint8)) > 0
        open_wash &= ((ink_density < .005) & (gray >= self.min_brightness) &
                      (chroma <= self.max_chroma) & cool_or_neutral & ~guard & ~edges)
        fade = np.where(open_wash, 1, opacity * connected) * self.strength
        fade = cv2.GaussianBlur(fade.astype(np.float32), (0, 0), 1)
        # Restore guards after smoothing so faint edges are not lifted.
        fade[guard | edges | ~selected | ~cool_or_neutral] = 0
        fade = cv2.resize(fade, (w, h), interpolation=cv2.INTER_LINEAR)
        # Do not feather confirmed wash back into gray at the illustration's
        # ragged boundary when expanding the detection mask to scan resolution.
        clear_at_scan_size = cv2.resize(np.uint8(open_wash), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        fade[clear_at_scan_size] = self.strength
        fade[(protected > 0) | (illustration == 0)] = 0
        return fade

    def process_masked_single(self, image, mask=None):
        if mask is None:
            raise ValueError("Wash cleanup requires an illustration mask")
        protected = self.pipeline.cache[self.pipeline.get_cache_key(image, self.protected_mask)].image
        if mask.image.shape != image.image.shape[:2] or protected.shape != mask.image.shape:
            raise ValueError("Wash protection masks must align with the color image")
        fade = self._fade_amount(image.image, mask.image, protected)
        result = image.image.copy()
        for channel in range(3):
            values = result[:, :, channel].astype(np.float32)
            result[:, :, channel] = np.rint(values + (255-values)*fade).astype(np.uint8)
        self.last_fade = fade
        image.image = result
        return image
