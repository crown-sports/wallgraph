"""Inference adapters. No model or provider is selected at import time."""

import hashlib
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray


class WallBackend(Protocol):
    name: str

    def predict(self, rgb: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Return wall probabilities in the input image's pixel coordinates."""


class InkBackend:
    """Deterministic demo baseline; text and furniture also count as ink."""

    name = "ink-baseline"

    def predict(self, rgb: NDArray[np.uint8]) -> NDArray[np.float32]:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        return ink.astype(np.float32) / 255


class OnnxBackend:
    """One persistent session for a single NCHW segmentation model.

    Binary outputs are logits unless output_kind='probabilities'. Multiclass
    outputs use softmax then sum wall_classes; class semantics remain explicit.
    """

    name = "onnx"

    def __init__(
        self,
        model: Path,
        *,
        device: str = "cpu",
        wall_classes: tuple[int, ...] = (1,),
        preprocess: str = "rgb",
        output_kind: str = "logits",
        input_size: tuple[int, int] = (1024, 1024),
    ) -> None:
        import onnxruntime as ort

        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if preprocess not in {"rgb", "gray", "clahe"}:
            raise ValueError("preprocess must be rgb, gray or clahe")
        if output_kind not in {"logits", "probabilities"}:
            raise ValueError("output_kind must be logits or probabilities")
        if not wall_classes or min(wall_classes) < 0 or len(set(wall_classes)) != len(wall_classes):
            raise ValueError("wall_classes must contain distinct nonnegative indices")
        if len(input_size) != 2 or min(input_size) < 1:
            raise ValueError("input_size must contain positive height and width")
        if device == "cuda" and "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDA provider unavailable; install a compatible GPU runtime")
        if device == "cuda" and hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        # Explicit CUDA requests must fail when node placement would fall back to CPU.
        if device == "cuda":
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        providers = ["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model), sess_options=options, providers=providers)
        if device == "cuda" and "CUDAExecutionProvider" not in self.session.get_providers():
            raise RuntimeError("CUDA initialization failed; refusing a mislabeled CPU benchmark")
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or len(inputs[0].shape) != 4:
            raise ValueError("expected one NCHW model input")
        if inputs[0].type != "tensor(float)" or inputs[0].shape[1] not in {3, "channels", None}:
            raise ValueError("expected float32 RGB input with three channels")
        if isinstance(inputs[0].shape[0], int) and inputs[0].shape[0] != 1:
            raise ValueError("fixed batch size must be one")
        shape = inputs[0].shape
        self.input_size = tuple(
            dimension if isinstance(dimension, int) and dimension > 0 else fallback
            for dimension, fallback in zip(shape[2:], input_size, strict=True)
        )
        self.input_name = inputs[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.wall_classes = wall_classes
        self.preprocess = preprocess
        self.output_kind = output_kind
        self.providers = self.session.get_providers()
        digest = hashlib.sha256()
        with model.open("rb") as source:
            for block in iter(lambda: source.read(1 << 20), b""):
                digest.update(block)
        self.model_sha256 = digest.hexdigest()

    def predict(self, rgb: NDArray[np.uint8]) -> NDArray[np.float32]:
        image = rgb
        if self.preprocess != "rgb":
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            if self.preprocess == "clahe":
                image = np.stack(
                    [
                        gray,
                        cv2.createCLAHE(2, (20, 20)).apply(gray),
                        cv2.createCLAHE(2, (40, 40)).apply(gray),
                    ],
                    axis=-1,
                )
            else:
                image = np.repeat(gray[..., None], 3, axis=-1)
        height, width = image.shape[:2]
        target_h, target_w = self.input_size
        scale = min(target_h / height, target_w / width)
        resized_h = max(1, min(target_h, round(height * scale)))
        resized_w = max(1, min(target_w, round(width * scale)))
        top, left = (target_h - resized_h) // 2, (target_w - resized_w) // 2
        canvas = np.full((target_h, target_w, 3), 128, np.uint8)
        canvas[top : top + resized_h, left : left + resized_w] = cv2.resize(
            image,
            (resized_w, resized_h),
            interpolation=cv2.INTER_CUBIC,
        )
        tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / 255
        output = self.session.run([self.output_name], {self.input_name: tensor})[0]
        if output.ndim != 4 or output.shape[0] != 1:
            raise ValueError("expected NCHW segmentation output with batch size one")
        scores = output[0].astype(np.float32)
        if not np.isfinite(scores).all():
            raise ValueError("model produced nonfinite output")
        if scores.shape[0] == 1:
            if self.wall_classes != (1,):
                raise ValueError("binary model requires wall_classes=(1,)")
            probability = scores[0]
            if self.output_kind == "logits":
                probability = 1 / (1 + np.exp(-np.clip(probability, -80, 80)))
        else:
            if max(self.wall_classes) >= scores.shape[0]:
                raise ValueError("wall class index exceeds output channel count")
            if self.output_kind == "logits":
                scores = np.exp(scores - scores.max(axis=0, keepdims=True))
                scores /= scores.sum(axis=0, keepdims=True)
            elif not np.allclose(scores.sum(axis=0), 1, atol=0.02):
                raise ValueError("multiclass probabilities must sum to one")
            probability = scores[list(self.wall_classes)].sum(axis=0)
        if probability.min() < 0 or probability.max() > 1.0001:
            raise ValueError("model probabilities must be in [0, 1]")
        probability = cv2.resize(probability, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        probability = probability[top : top + resized_h, left : left + resized_w]
        return cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)
