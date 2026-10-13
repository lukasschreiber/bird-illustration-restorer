import numpy as np
import cv2
from dataclasses import replace
from scipy.interpolate import PchipInterpolator

from pipeline.base import PipelineStep
from pipeline.steps.whitebalance import WhiteBalanceStep
from pipeline.steps.brightness_contrast import BrightnessContrastStep


class FinalColorEnhancementStep(PipelineStep):
    """Grade cleaned artwork last, with a highlight-safe or legacy finish."""

    def __init__(self, name, pipeline=None, strength=1.0, correction_min=.8,
                 correction_max=1.2, brightness=-40, contrast=30, k=3, max_size=256,
                 reference_image=None, mode="legacy", vibrance=.25):
        super().__init__(name, pipeline)
        self.balance = WhiteBalanceStep(name, pipeline=pipeline, strength=strength,
                                       correction_min=correction_min, correction_max=correction_max,
                                       k=k, max_size=max_size)
        self.contrast = BrightnessContrastStep(name, brightness=brightness,
                                              contrast=contrast, pipeline=pipeline)
        self.reference_image = reference_image
        if mode not in {"legacy", "detail_preserving"} or not 0 <= vibrance <= 1:
            raise ValueError("Invalid final color grading settings")
        self.mode = mode
        self.vibrance = vibrance

    def _grade_detail_preserving(self, source):
        white = np.all(source == 255, axis=2)
        # Float Lab keeps luminance separate from color, avoiding per-channel
        # clipping that destroyed pale feather shading in the legacy boost.
        lab = cv2.cvtColor(source.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
        curve = PchipInterpolator([0, 32, 64, 128, 192, 224, 240, 255],
                                  [0, 8, 28, 112, 205, 234, 246, 255])
        lightness = lab[:, :, 0].copy()
        lab[:, :, 0] = curve(lightness * 2.55) / 2.55
        chroma = np.hypot(lab[:, :, 1], lab[:, :, 2])
        highlight = np.clip((lightness-85)/15, 0, 1)
        highlight = highlight*highlight*(3-2*highlight)
        boost = 1 + self.vibrance * (1-np.clip(chroma/60, 0, 1)) * (1-highlight)
        lab[:, :, 1:] *= boost[:, :, None]
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        result = np.clip(np.rint(result*255), 0, 255).astype(np.uint8)
        result[white] = 255
        return result

    def process_single(self, image):
        if self.mode == "detail_preserving":
            image.image = self._grade_detail_preserving(image.image)
            return image
        if self.reference_image:
            reference = self.pipeline.cache[self.pipeline.get_cache_key(image, self.reference_image)].image
            background = self.balance._detect_background_color_kmeans(reference, self.balance.k, self.balance.max_size)
            corrected = self.balance._correct_color_balance(
                image.image, image.english_label, background, self.balance.strength,
                self.balance.correction_min, self.balance.correction_max)
            balanced = replace(image, image=corrected)
        else:
            balanced = self.balance.run(image)
        # Match the original order: newly clipped whites are excluded from
        # contrast adjustment, so cleaned background and white padding stay white.
        non_white = np.any(balanced.image != 255, axis=2)
        contrasted = self.contrast.run(balanced)
        result = balanced.image.copy()
        result[non_white] = contrasted.image[non_white]
        image.image = result
        return image
