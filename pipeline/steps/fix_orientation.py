from pipeline.base import PipelineStep, PipelineImageContainer
import cv2
import re
import numpy as np

class FixOrientationStep(PipelineStep):
    def __init__(self, name, pipeline=None):
        super().__init__(name, pipeline)
        self._ocr = None

    def _caption_rotation(self, source, label):
        """Establish orientation from the visible caption, not PDF OCR geometry."""
        if self._ocr is None:
            from rapidocr_onnxruntime import RapidOCR
            self._ocr = RapidOCR(intra_op_num_threads=4, inter_op_num_threads=1)
        h, w = source.shape[:2]
        scale = min(1, 1600 / max(h, w))
        small = cv2.resize(source, (round(w*scale), round(h*scale)))
        tokens = [t for t in re.findall(r'[a-z]+', label.lower()) if len(t) >= 4]
        rotations = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180,
                     cv2.ROTATE_90_COUNTERCLOCKWISE]
        scores = []
        for rotation in rotations:
            candidate = small if rotation is None else cv2.rotate(small, rotation)
            # OCR's automatic 180-degree correction would make upside-down
            # captions readable too, defeating orientation scoring.
            detections, _ = self._ocr(candidate, use_cls=False)
            score = 0
            for box, text, confidence in detections or []:
                points = np.asarray(box)
                horizontal = np.linalg.norm(points[1]-points[0]) > 2*np.linalg.norm(points[3]-points[0])
                normalized = re.sub('[^a-z]', '', text.lower())
                matches = sum(len(t) for t in tokens if t in normalized)
                if horizontal and confidence >= .8 and matches >= 4:
                    score += matches * confidence
            scores.append(score)
        best = int(np.argmax(scores))
        return rotations[best] if scores[best] > 0 else None

    def process_single(self, input_item: PipelineImageContainer):
        # Embedded PDF OCR can be garbled or laid out as horizontal snippets
        # despite a sideways scan. Always check the actual printed caption.
        rotation = self._caption_rotation(input_item.image, input_item.english_label)
        if rotation is not None:
            input_item.image = cv2.rotate(input_item.image, rotation)
        return input_item
