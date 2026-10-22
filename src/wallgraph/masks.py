"""Import an already segmented wall mask without running an image model."""

from dataclasses import asdict
from time import perf_counter

import cv2
import numpy as np

from .config import WallConfig
from .domain import WallResult
from .vectorize import SkeletonVectorizer


def vectorize_mask(
    mask: np.ndarray, config: WallConfig | None = None, *, wall_value: int = 255
) -> WallResult:
    """Vectorize strict binary labels; preserve every supplied pixel by default.

    Segment confidence is mask membership, not an upstream model probability.
    Closing and component filtering are opt-in changes to the supplied mask.
    """
    if wall_value not in (0, 255) or isinstance(wall_value, bool):
        raise ValueError("wall_value must be 0 or 255")
    if (
        not isinstance(mask, np.ndarray)
        or mask.ndim != 2
        or mask.dtype != np.uint8
        or not mask.size
        or not np.isin(mask, (0, 255)).all()
    ):
        raise ValueError("mask must be a nonempty 2D uint8 binary array containing only 0 and 255")
    config = config or WallConfig(min_component_area=1)
    if config.tile_size:
        raise ValueError("mask vectorization has no inference tiles; tile_size must be zero")
    start = perf_counter()
    supplied = (mask == wall_value).astype(np.uint8) * 255
    probability = supplied.astype(np.float32) / 255
    cleaned = supplied.copy()
    if config.close_radius:
        size = 2 * config.close_radius + 1
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((size, size), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    keep = np.zeros(count, dtype=np.uint8)
    keep[1:] = (stats[1:, cv2.CC_STAT_AREA] >= config.min_component_area) * 255
    cleaned = keep[labels]
    nodes, segments = SkeletonVectorizer(
        config.simplify_tolerance, config.min_segment_length
    ).extract(cleaned, probability)
    return WallResult(
        cleaned,
        nodes,
        segments,
        {
            "backend": "supplied_binary_mask",
            "config": asdict(config),
            "source_wall_value": wall_value,
            "input_wall_pixels": int(np.count_nonzero(supplied)),
            "output_wall_pixels": int(np.count_nonzero(cleaned)),
            "changed_pixels": int(np.count_nonzero(supplied != cleaned)),
            "confidence_kind": "supplied_mask_membership_on_skeleton_not_model_probability",
            "timings_ms": {"total": (perf_counter() - start) * 1000},
        },
    )
