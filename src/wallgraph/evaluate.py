"""Private manifests in; aggregate, path-free reports out."""

import hashlib
import json
import platform
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image

from .io import read_rgb
from .metrics import wall_metrics
from .pipeline import WallPipeline


def read_manifest(path: Path) -> list[dict]:
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not records:
        raise ValueError("manifest is empty")
    for record in records:
        if not all(
            isinstance(record.get(key), str) and record[key] for key in ("image", "mask", "group")
        ):
            raise ValueError("each record must contain image, mask and group strings")
    return records


def evaluate(pipeline: WallPipeline, manifest: Path) -> dict:
    records = read_manifest(manifest)
    totals = defaultdict(int)
    samples = []
    timings = []
    for record in records:
        rgb = read_rgb(manifest.parent / record["image"])
        with Image.open(manifest.parent / record["mask"]) as image:
            truth = np.asarray(image.convert("L"))
        result = pipeline.run(rgb)
        scores = wall_metrics(result.mask, truth)
        for key in ("tp", "fp", "fn"):
            totals[key] += scores[key]
        samples.append(scores)
        timings.append(result.metadata["timings_ms"]["total"])
    tp, fp, fn = (totals[key] for key in ("tp", "fp", "fn"))
    return {
        "schema_version": "wallgraph-evaluation/1",
        "sample_count": len(records),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "micro_iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        "macro_skeleton_f1": float(np.mean([s["skeleton_f1"] for s in samples])),
        "latency_ms": {
            "median": float(np.median(timings)),
            "p95": float(np.percentile(timings, 95)),
        },
        "timing_scope": "pipeline, excludes file I/O and session creation; no warmup",
        "backend": pipeline.backend.name,
        "config": result.metadata["config"],
        "providers": result.metadata["providers"],
        "model_sha256": result.metadata["model_sha256"],
        "preprocess": result.metadata["preprocess"],
        "wall_classes": result.metadata["wall_classes"],
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": version("numpy"),
            "opencv": version("opencv-python-headless"),
        },
    }
