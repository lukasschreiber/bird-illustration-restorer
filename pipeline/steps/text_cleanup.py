import json
import re
from pathlib import Path

import cv2
import numpy as np

from pipeline.base import PipelineStep


class AutomaticTextCleanupStep(PipelineStep):
    """Inpaint confidently recognized text only when its surroundings are quiet."""

    def __init__(self, name, pipeline=None, max_size=2400, min_confidence=.8,
                 max_ring_std=25, max_ring_edges=.1, signature_confidence=.75,
                 report_directory="./data/out/text-cleanup/"):
        super().__init__(name, pipeline)
        self.max_size = max_size
        self.min_confidence = min_confidence
        self.max_ring_std = max_ring_std
        self.max_ring_edges = max_ring_edges
        self.signature_confidence = signature_confidence
        self.report_directory = report_directory
        self._ocr = None

    def _get_ocr(self):
        if self._ocr is None:
            from rapidocr_onnxruntime import RapidOCR
            self._ocr = RapidOCR(intra_op_num_threads=4, inter_op_num_threads=1)
        return self._ocr

    def _safe_region(self, gray, polygon):
        x, y, w, h = cv2.boundingRect(polygon)
        if h > gray.shape[0] * .06 or w < h:
            return False
        pad = 12
        x0, y0 = max(0, x-pad), max(0, y-pad)
        x1, y1 = min(gray.shape[1], x+w+pad), min(gray.shape[0], y+h+pad)
        roi = gray[y0:y1, x0:x1]
        ring = np.ones(roi.shape, bool)
        ring[y-y0:y+h-y0, x-x0:x+w-x0] = False
        if not np.any(ring):
            return False
        edges = cv2.Canny(roi, 25, 75)
        return (np.std(roi[ring]) <= self.max_ring_std and
                np.mean(edges[ring] > 0) <= self.max_ring_edges)

    def process_single(self, image):
        source = image.image
        h, w = source.shape[:2]
        scale = min(1, self.max_size / max(h, w))
        small = cv2.resize(source, (max(1, round(w*scale)), max(1, round(h*scale))),
                           interpolation=cv2.INTER_AREA)
        detections, _ = self._get_ocr()(small)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        result = source.copy()
        reports = []
        for box, text, confidence in detections or []:
            polygon = np.rint(box).astype(np.int32)
            signature = bool(re.search(r"(?:\b|\.)(?:del|lith|sculp|pinx)\.?$", text.strip(), re.I))
            minimum = self.signature_confidence if signature else self.min_confidence
            recognized = confidence >= minimum and sum(c.isalpha() for c in text) >= 4
            safe = recognized and self._safe_region(gray, polygon)
            reports.append({"text": text, "confidence": float(confidence),
                            "box": (np.asarray(box)/scale).tolist(),
                            "status": "removed" if safe else "retained_uncertain_or_textured"})
            if not safe:
                continue
            full = np.rint(np.asarray(box)/scale).astype(np.int32)
            x, y, rw, rh = cv2.boundingRect(full)
            pad = max(4, round(8/scale))
            x0, y0 = max(0, x-pad), max(0, y-pad)
            x1, y1 = min(w, x+rw+pad), min(h, y+rh+pad)
            roi = result[y0:y1, x0:x1]
            region = np.zeros(roi.shape[:2], np.uint8)
            cv2.fillPoly(region, [full - [x0, y0]], 255)
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            background = cv2.GaussianBlur(roi_gray.astype(np.float32), (0, 0), max(2, 3/scale))
            strokes = np.uint8((background - roi_gray > 10) & (region > 0)) * 255
            radius = max(1, round(1/scale))
            strokes = cv2.dilate(strokes, np.ones((radius*2+1, radius*2+1), np.uint8))
            if not np.any(strokes):
                reports[-1]["status"] = "retained_no_strokes"
                continue
            roi[:] = cv2.inpaint(roi, strokes, max(3, round(2/scale)), cv2.INPAINT_TELEA)
        if self.report_directory:
            path = Path(self.report_directory) / image.book
            path.mkdir(parents=True, exist_ok=True)
            (path / f"{image.page}-v-{image.instance}.json").write_text(json.dumps(reports, indent=2))
        image.image = result
        return image
