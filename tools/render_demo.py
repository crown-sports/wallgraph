"""Check the reviewed public figure or render real stages with a supplied model."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from draw_complex_demo import draw_complex_demo
from PIL import Image
from pipeline_figure import run_and_render

PUBLIC_FIGURE = Path(__file__).resolve().parents[1] / "examples/simple.png"
PUBLIC_FIGURE_SHA256 = "f6d711b40268d34c0b9fc81b7597f05c3aebbe53ef26a241d6a1e246d1f5f8e8"


def check_figure(path: Path) -> None:
    """Check the reviewed asset, not inference reproducibility without weights."""
    if hashlib.sha256(path.read_bytes()).hexdigest() != PUBLIC_FIGURE_SHA256:
        raise ValueError("figure differs from the reviewed public asset")
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != (1560, 1000):
            raise ValueError("reviewed figure must be a 1560x1000 RGB PNG")
        if image.info:
            raise ValueError("reviewed figure must contain no metadata")
        image.load()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--image", type=Path, help="optional input; otherwise generate a complex plan"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path, help="optional local JSON record; never uploaded")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--preprocess", choices=("rgb", "gray", "clahe"), default="clahe")
    parser.add_argument("--wall-classes", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--output-kind", choices=("logits", "probabilities"), default="logits")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--close-radius", type=int, default=2)
    args = parser.parse_args()
    if args.check:
        if args.model or args.image or args.summary:
            parser.error(
                "--check only validates the fixed public figure; do not supply model/image"
            )
        try:
            check_figure(args.output or PUBLIC_FIGURE)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        print("Reviewed figure hash, dimensions and metadata verified; no inference run")
        return
    if args.model is None or args.output is None:
        parser.error("rendering requires --model and --output; match options to your model")
    if args.image is None:
        rgb = draw_complex_demo()
    else:
        with Image.open(args.image) as image:
            rgb = np.asarray(image.convert("RGB"))
    figure, summary = run_and_render(
        rgb,
        args.model,
        device=args.device,
        preprocess=args.preprocess,
        wall_classes=tuple(args.wall_classes),
        threshold=args.threshold,
        close_radius=args.close_radius,
        output_kind=args.output_kind,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.save(args.output, format="PNG")
    summary["source_kind"] = (
        "generated_complex_plan" if args.image is None else "caller_supplied_image"
    )
    summary["figure_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"Rendered one actual inference: {summary['measurements']['paths']} paths, "
        f"{summary['measurements']['nodes']} nodes; no manual prediction edits"
    )


if __name__ == "__main__":
    main()
