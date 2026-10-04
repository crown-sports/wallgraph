from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class WallConfig:
    threshold: float = 0.5
    min_component_area: int = 16
    close_radius: int = 0
    simplify_tolerance: float = 1.5
    min_segment_length: float = 3.0
    tile_size: int = 0
    overlap: int = 128

    def __post_init__(self) -> None:
        if not isfinite(self.threshold) or not 0 < self.threshold < 1:
            raise ValueError("threshold must be between zero and one")
        if self.min_component_area < 1 or self.close_radius < 0:
            raise ValueError("component area must be positive and close radius nonnegative")
        if not isfinite(self.simplify_tolerance) or self.simplify_tolerance <= 0:
            raise ValueError("simplify_tolerance must be positive and finite")
        if not isfinite(self.min_segment_length) or self.min_segment_length < 0:
            raise ValueError("min_segment_length must be nonnegative and finite")
        if self.tile_size < 0 or self.overlap < 0:
            raise ValueError("tile_size and overlap must be nonnegative")
        if self.tile_size and self.overlap >= self.tile_size:
            raise ValueError("overlap must be smaller than tile_size")
