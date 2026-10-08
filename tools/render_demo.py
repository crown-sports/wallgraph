"""Render the single public before/after figure from generated geometry only."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from wallgraph import WallConfig, WallPipeline
from wallgraph.backends import InkBackend
from wallgraph.io import draw_demo

COLORS = ("#2274a5", "#8558a3", "#16877a", "#c46922", "#d04565", "#627a26")


def make_figure() -> Image.Image:
    image = draw_demo()
    result = WallPipeline(InkBackend(), WallConfig()).run(image)
    if len(result.segments) != 6 or len(result.junctions) != 4:
        raise ValueError("generated demo geometry changed; review the figure and README counts")
    board = Image.new("RGB", (512, 328), "#f8fafc")
    draw = ImageDraw.Draw(board)
    title = ImageFont.load_default(size=18)
    body = ImageFont.load_default(size=14)
    small = ImageFont.load_default(size=12)
    draw.text((18, 17), "Input drawing", font=title, fill="#182d3b")
    draw.text((275, 17), "WallGraph output", font=title, fill="#182d3b")
    left = (12, 52)
    right = (270, 52)
    size = (230, 154)
    board.paste(Image.fromarray(image).resize(size, Image.Resampling.NEAREST), left)
    evidence = np.where(result.mask[..., None] > 0, 225, 255).astype(np.uint8)
    evidence = np.repeat(evidence, 3, axis=2)
    board.paste(Image.fromarray(evidence).resize(size, Image.Resampling.NEAREST), right)
    height, width = result.mask.shape

    def point(x: float, y: float) -> tuple[float, float]:
        return right[0] + x * size[0] / width, right[1] + y * size[1] / height

    for segment in result.segments:
        draw.line(
            [point(p.x, p.y) for p in segment.points],
            fill=COLORS[segment.id % len(COLORS)],
            width=3,
            joint="curve",
        )
    for node in result.junctions:
        x, y = point(node.point.x, node.point.y)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#f3a52b", outline="#182d3b")
        label = str(node.id)
        label_x, label_y = x + 7, y + 3
        bounds = draw.textbbox((label_x, label_y), label, font=small)
        draw.rectangle((bounds[0] - 1, bounds[1] - 1, bounds[2] + 1, bounds[3] + 1), fill="white")
        draw.text((label_x, label_y), label, font=small, fill="#182d3b")
    draw.line((18, 224, 43, 224), fill=COLORS[0], width=3)
    draw.text((51, 216), "Wall path", font=body, fill="#344c5c")
    draw.ellipse((180, 220, 188, 228), fill="#f3a52b", outline="#182d3b")
    draw.text((199, 216), "Connection node", font=body, fill="#344c5c")
    draw.line((18, 249, 494, 249), fill="#dbe3e8", width=1)
    draw.text(
        (18, 262), "6 paths  |  4 nodes  |  384 x 256 original pixels", font=body, fill="#182d3b"
    )
    draw.text((18, 287), "walls.png  +  walls.json  +  walls.svg", font=body, fill="#344c5c")
    draw.text(
        (18, 310),
        "Generated example; no trained model or dataset used.",
        font=small,
        fill="#536876",
    )
    return board


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1] / "examples/simple.png"
    )
    args = parser.parse_args()
    figure = make_figure()
    if args.check:
        if not args.output.is_file():
            parser.error("demo figure is missing; run tools/render_demo.py")
        with Image.open(args.output) as existing:
            if existing.mode != "RGB" or not np.array_equal(
                np.asarray(existing), np.asarray(figure)
            ):
                parser.error("demo figure differs from actual output; run tools/render_demo.py")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.save(args.output)
    print("Generated demo figure verified" if args.check else "Generated input/output figure")


if __name__ == "__main__":
    main()
