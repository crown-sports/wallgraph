from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def _starts(length: int, size: int, overlap: int) -> list[int]:
    if length <= size:
        return [0]
    return sorted(set(range(0, length - size + 1, size - overlap)) | {length - size})


def fuse_tiles(
    rgb: NDArray[np.uint8],
    predictor: Callable[[NDArray[np.uint8]], NDArray[np.float32]],
    size: int,
    overlap: int,
) -> NDArray[np.float32]:
    """Blend probabilities before thresholding, keeping coverage nonzero at edges."""
    if size < 1 or not 0 <= overlap < size:
        raise ValueError("invalid tile size or overlap")
    height, width = rgb.shape[:2]
    total = np.zeros((height, width), np.float32)
    weights = np.zeros_like(total)
    for top in _starts(height, size, overlap):
        for left in _starts(width, size, overlap):
            crop = rgb[top : top + size, left : left + size]
            ch, cw = crop.shape[:2]
            prediction = predictor(crop)
            if prediction.shape != (ch, cw) or not np.isfinite(prediction).all():
                raise ValueError("backend must return finite probabilities in crop coordinates")
            if prediction.min() < 0 or prediction.max() > 1:
                raise ValueError("backend probabilities must be in [0, 1]")
            window = np.outer(
                np.maximum(np.hanning(ch), 0.05),
                np.maximum(np.hanning(cw), 0.05),
            ).astype(np.float32)
            total[top : top + ch, left : left + cw] += prediction * window
            weights[top : top + ch, left : left + cw] += window
    return total / weights
