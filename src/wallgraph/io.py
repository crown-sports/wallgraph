import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .domain import WallResult


def read_rgb(path: Path) -> np.ndarray:
    # Keep the file's native orientation: masks and annotations share those pixels.
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def save_result(result: WallResult, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result.mask).save(destination / "walls.png")
    (destination / "walls.json").write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    # Scale the SVG viewBox directly in original input pixels.
    height, width = result.mask.shape
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">']
    for segment in result.segments:
        points = " ".join(f"{point.x:.3f},{point.y:.3f}" for point in segment.points)
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="#3366cc" '
            f'stroke-width="{segment.thickness_px:.3f}" stroke-linecap="butt"/>'
        )
    lines.append("</svg>")
    (destination / "walls.svg").write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_demo() -> np.ndarray:
    """One deliberately simple, original floor plan; no private dataset involved."""
    image = np.full((256, 384, 3), 255, np.uint8)
    cv2.rectangle(image, (24, 24), (360, 232), (0, 0, 0), 9)
    cv2.line(image, (192, 24), (192, 232), (0, 0, 0), 9)
    cv2.line(image, (192, 128), (360, 128), (0, 0, 0), 9)
    return image
