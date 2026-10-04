from dataclasses import asdict
from time import perf_counter
from typing import Protocol

import cv2
import numpy as np

from .backends import WallBackend
from .config import WallConfig
from .domain import Junction, WallResult, WallSegment
from .tiling import fuse_tiles
from .vectorize import SkeletonVectorizer


class Vectorizer(Protocol):
    def extract(
        self,
        mask: np.ndarray,
        probability: np.ndarray,
    ) -> tuple[tuple[Junction, ...], tuple[WallSegment, ...]]: ...


class WallPipeline:
    """Compose inference, conservative cleanup and geometry via dependency injection."""

    def __init__(
        self,
        backend: WallBackend,
        config: WallConfig | None = None,
        vectorizer: Vectorizer | None = None,
    ) -> None:
        self.backend = backend
        self.config = config or WallConfig()
        self.vectorizer = vectorizer or SkeletonVectorizer(
            self.config.simplify_tolerance,
            self.config.min_segment_length,
        )

    def run(self, rgb: np.ndarray) -> WallResult:
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or min(rgb.shape) < 1:
            raise ValueError("input must be a nonempty HWC uint8 RGB image")
        start = perf_counter()
        if self.config.tile_size:
            probability = fuse_tiles(
                rgb,
                self.backend.predict,
                self.config.tile_size,
                self.config.overlap,
            )
        else:
            probability = self.backend.predict(rgb)
        if probability.shape != rgb.shape[:2] or not np.isfinite(probability).all():
            raise ValueError("backend returned invalid probability shape or values")
        if probability.min() < 0 or probability.max() > 1:
            raise ValueError("backend probabilities must be in [0, 1]")
        inferred = perf_counter()
        mask = (probability >= self.config.threshold).astype(np.uint8) * 255
        if self.config.close_radius:
            size = 2 * self.config.close_radius + 1
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((size, size), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = np.zeros(count, dtype=np.uint8)
        keep[1:] = (stats[1:, cv2.CC_STAT_AREA] >= self.config.min_component_area) * 255
        mask = keep[labels]
        cleaned = perf_counter()
        junctions, segments = self.vectorizer.extract(mask, probability)
        end = perf_counter()
        metadata = {
            "backend": self.backend.name,
            "providers": getattr(self.backend, "providers", []),
            "config": asdict(self.config),
            "timings_ms": {
                "inference": (inferred - start) * 1000,
                "cleanup": (cleaned - inferred) * 1000,
                "vectorization": (end - cleaned) * 1000,
                "total": (end - start) * 1000,
            },
            "confidence_kind": "mean_wall_probability_on_skeleton",
            "model_sha256": getattr(self.backend, "model_sha256", None),
            "preprocess": getattr(self.backend, "preprocess", None),
            "wall_classes": getattr(self.backend, "wall_classes", None),
            "input_size": getattr(self.backend, "input_size", None),
            "output_kind": getattr(self.backend, "output_kind", None),
        }
        return WallResult(mask, junctions, segments, metadata)
