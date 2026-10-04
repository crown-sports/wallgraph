import platform
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np

from .io import read_rgb
from .pipeline import WallPipeline


def benchmark(pipeline: WallPipeline, images: list[Path], warmup: int, repeats: int) -> dict:
    if not images or warmup < 0 or repeats < 1:
        raise ValueError("nonempty image list, nonnegative warmup and positive repeats required")
    timings = []
    inference_timings = []
    counts = []
    shapes = set()
    load_start = perf_counter()
    loaded = [read_rgb(path) for path in images]
    load_ms = (perf_counter() - load_start) * 1000
    for image in loaded:
        shapes.add(tuple(image.shape[:2]))
        for _ in range(warmup):
            pipeline.run(image)
        for _ in range(repeats):
            result = pipeline.run(image)
            timings.append(result.metadata["timings_ms"]["total"])
            inference_timings.append(result.metadata["timings_ms"]["inference"])
            counts.append(len(result.segments))
    return {
        "schema_version": "wallgraph-benchmark/1",
        "unique_image_count": len(images),
        "warmup_per_image": warmup,
        "repeats_per_image": repeats,
        "measured_runs": len(timings),
        "input_shapes_hw": sorted(shapes),
        "pipeline_ms": {
            "median": float(np.median(timings)),
            "p95": float(np.percentile(timings, 95)),
        },
        "inference_ms": {
            "median": float(np.median(inference_timings)),
            "p95": float(np.percentile(inference_timings, 95)),
        },
        "image_loading_ms": load_ms,
        "segment_count_range": [min(counts), max(counts)],
        "timing_scope": "pipeline, excludes file I/O and model initialization",
        "accuracy": None,
        "accuracy_reason": "benchmark has no ground-truth annotations",
        "backend": result.metadata,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": version("numpy"),
            "opencv": version("opencv-python-headless"),
        },
    }
