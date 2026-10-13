from pathlib import Path

import cv2
import numpy as np

from pipeline.base import PipelineStep


class SemanticIllustrationMaskStep(PipelineStep):
    """Run a local NCHW ONNX segmentation model on the original color scan.

    Binary models predict artwork/background. Multiclass models can retain both
    bird and scenery classes while excluding paper, lettering, and damage.
    """

    def __init__(self, name, model_path, pipeline=None, input_size=1024,
                 foreground_classes=None, threshold=0.5, output_is_logits=True,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                 min_coverage=0.001, max_coverage=0.95, device="auto"):
        super().__init__(name, pipeline)
        if input_size < 32 or not 0 < threshold < 1:
            raise ValueError("Invalid segmentation size or threshold")
        if not 0 <= min_coverage < max_coverage <= 1:
            raise ValueError("Invalid segmentation coverage limits")
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.foreground_classes = [1] if foreground_classes is None else foreground_classes
        self.threshold = threshold
        self.output_is_logits = output_is_logits
        self.mean = np.asarray(mean, np.float32)
        self.std = np.asarray(std, np.float32)
        if self.mean.shape != (3,) or self.std.shape != (3,) or np.any(self.std <= 0):
            raise ValueError("Mean and standard deviation must describe RGB channels")
        self.min_coverage = min_coverage
        self.max_coverage = max_coverage
        self._session = None
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Segmentation device must be auto, cpu, or cuda")
        self.device = device

    def _get_session(self):
        if self._session is None:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"Segmentation model missing: {self.model_path}")
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError("Install requirements-semantic.txt to use semantic masking") from exc
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            providers = ["CPUExecutionProvider"]
            if self.device != "cpu":
                if hasattr(ort, 'preload_dlls') and 'CUDAExecutionProvider' in ort.get_available_providers():
                    ort.preload_dlls(directory="")
                if 'CUDAExecutionProvider' in ort.get_available_providers():
                    providers.insert(0, ("CUDAExecutionProvider", {
                        "cudnn_conv_algo_search": "HEURISTIC",
                        "cudnn_conv_use_max_workspace": "0",
                        "arena_extend_strategy": "kSameAsRequested",
                    }))
                elif self.device == "cuda":
                    raise RuntimeError("CUDA requested but ONNX Runtime GPU is unavailable")
            self._session = ort.InferenceSession(str(self.model_path), sess_options=options,
                                                providers=providers)
            active = self._session.get_providers()
            if self.device == "cuda" and "CUDAExecutionProvider" not in active:
                self._session = None
                raise RuntimeError("CUDA requested but the GPU provider could not initialize")
            print("Segmentation device: " + ("NVIDIA GPU (CUDA)" if
                  "CUDAExecutionProvider" in active else "CPU"), flush=True)
        return self._session

    def _foreground_probability(self, output):
        output = np.asarray(output, dtype=np.float32)
        if output.ndim != 4 or output.shape[0] != 1 or not np.all(np.isfinite(output)):
            raise ValueError("Expected finite ONNX segmentation output [1, classes, height, width]")
        scores = output[0]
        if scores.shape[0] == 1:
            if self.foreground_classes != [1]:
                raise ValueError("Binary models use foreground_classes: [1]")
            probability = (1 / (1 + np.exp(-np.clip(scores[0], -80, 80)))
                           if self.output_is_logits else scores[0])
        else:
            if not self.foreground_classes or any(c < 0 or c >= scores.shape[0] for c in self.foreground_classes):
                raise ValueError("Foreground classes do not match model output")
            if self.output_is_logits:
                scores = np.exp(scores - np.max(scores, axis=0, keepdims=True))
                scores /= scores.sum(axis=0, keepdims=True)
            probability = scores[list(set(self.foreground_classes))].sum(axis=0)
        if np.any(probability < 0) or np.any(probability > 1.0001):
            raise ValueError("Segmentation probabilities must be between zero and one")
        return probability

    def process_single(self, image):
        source = image.image
        if source.ndim != 3 or source.shape[2] != 3 or source.dtype != np.uint8:
            raise ValueError("Semantic masking requires an original uint8 BGR color image")
        rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        normalized = (resized.astype(np.float32) / 255 - self.mean) / self.std
        batch = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])
        session = self._get_session()
        try:
            output = session.run(None, {session.get_inputs()[0].name: batch})[0]
        except Exception as exc:
            if self.device != 'auto' or 'CUDAExecutionProvider' not in session.get_providers():
                raise
            print(f"GPU segmentation failed ({type(exc).__name__}); retrying on CPU", flush=True)
            self._session = None
            del session
            import gc
            gc.collect()
            self.device = 'cpu'
            try:
                session = self._get_session()
            finally:
                self.device = 'auto'
            output = session.run(None, {session.get_inputs()[0].name: batch})[0]
        probability = self._foreground_probability(output)
        probability = cv2.resize(probability, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask = np.uint8(probability >= self.threshold) * 255
        coverage = np.mean(mask > 0)
        if not self.min_coverage <= coverage <= self.max_coverage:
            raise ValueError(f"Review segmentation for {image.book} page {image.page}: coverage {coverage:.1%}")
        # Do not close or fill this mask: scenery can contain intentional gaps.
        image.image = mask
        return image
