import cv2
import numpy as np

from pipeline.base import PipelineStep


class PaperBalanceStep(PipelineStep):
    """Estimate paper only outside the illustration and preserve highlight detail."""

    def __init__(self, name, pipeline=None, max_size=512, target_white=245,
                 max_gain=1.1, local_strength=0.5, max_local_shift=10,
                 highlight_knee=220):
        super().__init__(name, pipeline)
        if max_size < 32 or not 0 <= local_strength <= 1 or max_local_shift < 0:
            raise ValueError("Invalid paper correction parameters")
        if not 0 < highlight_knee < target_white <= 255 or max_gain < 1:
            raise ValueError("Invalid paper highlight parameters")
        self.max_size = max_size
        self.target_white = target_white
        self.max_gain = max_gain
        self.local_strength = local_strength
        self.max_local_shift = max_local_shift
        self.highlight_knee = highlight_knee

    def process_masked_single(self, image, mask=None):
        if mask is None or mask.image.shape != image.image.shape[:2]:
            raise ValueError("Paper correction requires an aligned subject mask")
        source = image.image
        h, w = source.shape[:2]
        scale = min(1, self.max_size / max(h, w))
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        small = cv2.resize(source, size, interpolation=cv2.INTER_AREA).astype(np.float32)
        outside = cv2.resize(mask.image, size, interpolation=cv2.INTER_NEAREST) == 0
        # Scan-edge shadows are poor estimates of the paper color.
        margin = max(1, round(min(size) * .02))
        outside[:margin] = outside[-margin:] = False
        outside[:, :margin] = outside[:, -margin:] = False
        samples = small[outside]
        if len(samples) < 32:
            # No reliable paper samples: keep the artwork's original tone.
            image.image = source.copy()
            return image
        luminance = samples.mean(axis=1)
        samples = samples[luminance >= np.percentile(luminance, 50)]
        paper = np.median(samples, axis=0)
        paper_luma = float(paper.mean())
        if paper_luma < 80:
            image.image = source.copy()
            return image
        weights = (outside & (small.mean(axis=2) >= paper_luma - 12)).astype(np.float32)
        sigma = max(size) / 16
        support = cv2.GaussianBlur(weights, (0, 0), sigma)
        weighted = cv2.GaussianBlur(small.mean(axis=2) * weights, (0, 0), sigma)
        local = weighted / np.maximum(support, 1e-6)
        confidence = support / (support + .02)
        shift = np.clip((paper_luma - local) * confidence * self.local_strength,
                        -self.max_local_shift, self.max_local_shift)
        shift = cv2.resize(shift, (w, h), interpolation=cv2.INTER_LINEAR)
        neutral = np.clip(paper_luma / np.maximum(paper, 1), .9, 1.1)
        gain = min(self.max_gain, self.target_white / paper_luma)
        corrected = (source.astype(np.float32) * neutral + shift[:, :, None]) * gain
        # A smooth shoulder retains distinctions between pale feather shades.
        knee = self.highlight_knee
        headroom = 255 - knee
        above = np.maximum(corrected - knee, 0)
        corrected = np.where(corrected > knee, knee + headroom * (-np.expm1(-above / headroom)), corrected)
        image.image = np.clip(np.rint(corrected), 0, 255).astype(np.uint8)
        return image
