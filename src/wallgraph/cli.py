import argparse
import json
from pathlib import Path

from .backends import InkBackend, OnnxBackend
from .benchmark import benchmark
from .config import WallConfig
from .evaluate import evaluate
from .io import draw_demo, read_rgb, save_result
from .pipeline import WallPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract walls and their connectivity graph")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("detect", "demo", "evaluate", "benchmark"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        if name == "detect":
            command.add_argument("--image", type=Path, required=True)
        if name == "evaluate":
            command.add_argument("--manifest", type=Path, required=True)
        if name == "benchmark":
            command.add_argument("--images", type=Path, nargs="+", required=True)
            command.add_argument("--warmup", type=int, default=1)
            command.add_argument("--repeats", type=int, default=3)
        command.add_argument("--model", type=Path)
        command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        command.add_argument("--wall-classes", type=int, nargs="+", default=[1])
        command.add_argument("--preprocess", choices=("rgb", "gray", "clahe"), default="rgb")
        command.add_argument("--output-kind", choices=("logits", "probabilities"), default="logits")
        command.add_argument("--input-size", type=int, nargs=2, default=[1024, 1024])
        command.add_argument("--tile-size", type=int, default=0)
        command.add_argument("--overlap", type=int, default=128)
        command.add_argument("--close-radius", type=int, default=0)
        command.add_argument("--threshold", type=float, default=0.5)
    train = commands.add_parser("train")
    train.add_argument("--train-manifest", type=Path, required=True)
    train.add_argument("--val-manifest", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--size", type=int, default=512)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    train.add_argument("--seed", type=int, default=42)
    export = commands.add_parser("export-onnx")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    if args.command == "train":
        from .training import train_model

        train_model(args)
        return
    if args.command == "export-onnx":
        from .training import export_model

        export_model(args)
        return
    if args.device == "cuda" and args.model is None:
        parser.error("--device cuda requires --model")
    if args.model:
        backend = OnnxBackend(
            args.model,
            device=args.device,
            wall_classes=tuple(args.wall_classes),
            preprocess=args.preprocess,
            output_kind=args.output_kind,
            input_size=tuple(args.input_size),
        )
    else:
        backend = InkBackend()
    pipeline = WallPipeline(
        backend,
        WallConfig(
            tile_size=args.tile_size,
            overlap=args.overlap,
            close_radius=args.close_radius,
            threshold=args.threshold,
        ),
    )
    if args.command in {"evaluate", "benchmark"}:
        report = (
            evaluate(pipeline, args.manifest)
            if args.command == "evaluate"
            else benchmark(pipeline, args.images, args.warmup, args.repeats)
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(report))
    else:
        image = draw_demo() if args.command == "demo" else read_rgb(args.image)
        result = pipeline.run(image)
        save_result(result, args.output)
        print(json.dumps({"segment_count": len(result.segments), "metadata": result.metadata}))


if __name__ == "__main__":
    main()
