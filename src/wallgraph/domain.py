from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Junction:
    id: int
    point: Point


@dataclass(frozen=True)
class WallSegment:
    id: int
    start_node: int
    end_node: int
    points: tuple[Point, ...]
    length_px: float
    thickness_px: float
    confidence: float


@dataclass
class WallResult:
    mask: NDArray[np.uint8]
    junctions: tuple[Junction, ...]
    segments: tuple[WallSegment, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        height, width = self.mask.shape
        return {
            "schema_version": "wallgraph/1",
            "image": {"width": width, "height": height},
            "coordinates": {"origin": "top_left", "x": "right", "y": "down", "unit": "px"},
            "mask": {"wall_value": 255, "background_value": 0},
            "junctions": [asdict(node) for node in self.junctions],
            "segments": [asdict(segment) for segment in self.segments],
            "metadata": self.metadata,
        }
