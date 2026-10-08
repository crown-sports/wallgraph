"""Render six measured stages from one caller-supplied image and ONNX model.

This tool does not bundle or select an image, dataset, or model. It records the
actual session input and normal pipeline result; display resizing never feeds
back into inference. The returned image embeds no filenames or image metadata.
"""

from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from wallgraph import WallConfig, WallPipeline
from wallgraph.backends import OnnxBackend
from wallgraph.domain import WallResult


class _RecordingSession:
    """Observe a session call while forwarding its original arguments unchanged."""

    def __init__(self, session, input_name: str) -> None:
        self.session = session
        self.input_name = input_name
        self.calls = 0
        self.tensor = None
        self.output_shape = None

    def __getattr__(self, name):
        return getattr(self.session, name)

    def run(self, outputs, feeds, *args, **kwargs):
        if self.calls:
            raise RuntimeError("a pipeline figure must use exactly one inference call")
        self.calls += 1
        self.tensor = feeds[self.input_name].copy()
        result = self.session.run(outputs, feeds, *args, **kwargs)
        self.output_shape = list(result[0].shape)
        return result


class _RecordingBackend(OnnxBackend):
    def predict(self, rgb: np.ndarray) -> np.ndarray:
        probability = super().predict(rgb)
        self.probability = probability.copy()
        return probability


def _font(size: int):
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow 10.0 uses the fixed-size default font.
        return ImageFont.load_default()


def _model_panels(tensor: np.ndarray, preprocess: str) -> tuple[list, str]:
    if tensor.ndim != 4 or tensor.shape[:2] != (1, 3):
        raise ValueError("captured model input must have shape 1x3xHxW")
    if tensor.dtype != np.float32 or not np.isfinite(tensor).all():
        raise ValueError("captured model input must contain finite float32 values")
    if tensor.min() < 0 or tensor.max() > 1:
        raise ValueError("captured model input is outside the expected uint8/255 range")
    displayed = np.rint(tensor[0].transpose(1, 2, 0) * 255).astype(np.uint8)
    if preprocess == "clahe":
        panels = [
            (
                "2. Model input 1/3: gray",
                "Actual tensor channel; letterbox included",
                displayed[:, :, 0],
            ),
            (
                "3. Model input 2/3: CLAHE20",
                "Actual channel; clip=2, grid=20x20",
                displayed[:, :, 1],
            ),
        ]
        note = "Shown input channels: gray and CLAHE20; third channel is CLAHE40."
    elif preprocess == "gray":
        panels = [
            (
                "2. Model input 1/3: gray",
                "Actual tensor channel; letterbox included",
                displayed[:, :, 0],
            ),
            (
                "3. Model input 2/3: gray",
                "Gray is repeated in all three channels",
                displayed[:, :, 1],
            ),
        ]
        note = "Gray mode repeats grayscale in all three model input channels."
    else:
        panels = [
            ("2. Actual model input: RGB", "Captured tensor; letterbox included", displayed),
            ("3. Model input 1/3: R", "Actual red channel; no CLAHE applied", displayed[:, :, 0]),
        ]
        note = "RGB mode: the captured RGB tensor and its red channel are shown."
    return panels, note


def _red_overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = rgb.copy()
    selected = mask > 0
    overlay[selected] = np.rint(
        0.35 * rgb[selected].astype(np.float32) + 0.65 * np.array((230, 35, 35))
    ).astype(np.uint8)
    return overlay


def _paste_panel(
    canvas: Image.Image,
    pixels: np.ndarray,
    index: int,
    title: str,
    description: str,
    *,
    nearest: bool = False,
    result: WallResult | None = None,
) -> None:
    column, row = index % 3, index // 3
    x, y = 20 + column * 512, 70 + row * 424
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((x, y, x + 495, y + 407), radius=10, fill="white", outline="#d8dee7")
    draw.text((x + 12, y + 10), title, fill="#152339", font=_font(20))
    draw.text((x + 12, y + 38), description, fill="#5c687a", font=_font(13))
    image = Image.fromarray(pixels).convert("RGB")
    resampling = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    image.thumbnail((472, 338), resampling)
    image_x, image_y = x + (496 - image.width) // 2, y + 61 + (338 - image.height) // 2
    canvas.paste(image, (image_x, image_y))
    if result is None:
        return
    source_h, source_w = pixels.shape[:2]
    scale_x, scale_y = image.width / source_w, image.height / source_h

    def position(point) -> tuple[float, float]:
        # Match the pixel-center mapping of the displayed source thumbnail.
        return (
            image_x + (point.x + 0.5) * scale_x - 0.5,
            image_y + (point.y + 0.5) * scale_y - 0.5,
        )

    for segment in result.segments:
        points = [position(point) for point in segment.points]
        if len(points) >= 2:
            draw.line(points, fill=(20, 90, 230), width=2)
    for node in result.junctions:
        node_x, node_y = position(node.point)
        draw.ellipse(
            (node_x - 2, node_y - 2, node_x + 2, node_y + 2),
            fill=(20, 90, 230),
            outline="white",
            width=1,
        )


def run_and_render(
    rgb: np.ndarray,
    model: Path,
    *,
    device: str = "cpu",
    preprocess: str = "clahe",
    wall_classes: tuple[int, ...] = (1, 2),
    threshold: float = 0.5,
    close_radius: int = 2,
    output_kind: str = "logits",
) -> tuple[Image.Image, dict]:
    """Run once, then render the measured stages without editing predictions.

    The caller is responsible for matching preprocessing, output kind and class
    indices to the model's contract. No tiling, additional prediction, bespoke
    mask repair, or accuracy evaluation occurs. The final mask is the regular
    WallPipeline threshold, configured closing and component filtering result.
    Counts use original-resolution outputs; thumbnails may omit tiny features.
    Returned summary fields contain no input/model paths or raw image metadata.
    """
    config = WallConfig(threshold=threshold, close_radius=close_radius)
    # Reject malformed input before constructing an expensive inference session.
    if (
        not isinstance(rgb, np.ndarray)
        or rgb.ndim != 3
        or rgb.shape[2] != 3
        or rgb.dtype != np.uint8
        or min(rgb.shape) < 1
    ):
        raise ValueError("input must be a nonempty HWC uint8 RGB image")
    backend = _RecordingBackend(
        Path(model),
        device=device,
        preprocess=preprocess,
        wall_classes=wall_classes,
        output_kind=output_kind,
    )
    recorder = _RecordingSession(backend.session, backend.input_name)
    backend.session = recorder
    result = WallPipeline(backend, config).run(rgb)
    if recorder.calls != 1 or recorder.tensor is None:
        raise RuntimeError("pipeline figure did not capture exactly one inference call")
    input_panels, channel_note = _model_panels(recorder.tensor, preprocess)
    raw_mask = backend.probability >= threshold
    final_mask = result.mask > 0
    panels = [
        ("1. Original RGB image", "Caller-supplied image; full frame", rgb),
        *input_panels,
        ("4. Final predicted wall mask", "Normal configured cleanup; nearest preview", result.mask),
        (
            "5. Predicted walls in red",
            "Final wall pixels; red overlay at opacity 0.65",
            _red_overlay(rgb, result.mask),
        ),
        ("6. Wall paths and nodes in blue", "Actual segment points and all junctions", rgb),
    ]
    canvas = Image.new("RGB", (1560, 1000), "#f1f4f8")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (20, 17),
        "WallGraph: one prediction, six measured stages",
        fill="#152339",
        font=_font(27),
    )
    for index, (title, description, pixels) in enumerate(panels):
        _paste_panel(
            canvas,
            pixels,
            index,
            title,
            description,
            nearest=index == 3,
            result=result if index == 5 else None,
        )
    height, width = rgb.shape[:2]
    lines = [
        (
            f"Original {width}x{height} | model SHA256 {backend.model_sha256[:12]} | "
            f"preprocess={preprocess} | wall classes={list(wall_classes)} | "
            f"threshold={threshold:g} | closing radius={close_radius}px"
        ),
        (
            f"Actual input tensor {'x'.join(map(str, recorder.tensor.shape))} | "
            f"min component={config.min_component_area}px | paths={len(result.segments)} | "
            f"nodes={len(result.junctions)} | final wall pixels={int(final_mask.sum())}"
        ),
        channel_note,
        "One inference run; no manual edits; not an accuracy benchmark. "
        "Thumbnails may omit tiny features.",
    ]
    for index, line in enumerate(lines):
        draw.text((20, 919 + index * 18), line, fill="#344359", font=_font(14))
    summary = {
        "schema_version": "wallgraph-pipeline-figure/1",
        "scope": "one_inference_visualization_not_accuracy_benchmark",
        "image": {"width": width, "height": height, "color": "RGB"},
        "figure": {"width": canvas.width, "height": canvas.height, "color": "RGB"},
        "model_sha256": backend.model_sha256,
        "device": device,
        "providers": list(backend.providers),
        "preprocess": preprocess,
        "output_kind": output_kind,
        "wall_classes": list(wall_classes),
        "config": asdict(config),
        "actual_input_tensor": {
            "shape": list(recorder.tensor.shape),
            "dtype": str(recorder.tensor.dtype),
            "normalization": "uint8/255",
            "geometry": "captured_after_aspect_preserving_resize_and_center_letterbox",
        },
        "output_tensor_shape": recorder.output_shape,
        "inference_calls": recorder.calls,
        "measurements": {
            "total_pixels": int(result.mask.size),
            "raw_threshold_wall_pixels": int(raw_mask.sum()),
            "final_wall_pixels": int(final_mask.sum()),
            "cleanup_added_pixels": int((~raw_mask & final_mask).sum()),
            "cleanup_removed_pixels": int((raw_mask & ~final_mask).sum()),
            "paths": len(result.segments),
            "nodes": len(result.junctions),
            "probability_min": float(backend.probability.min()),
            "probability_max": float(backend.probability.max()),
        },
        "display": {
            "input_panels": [title for title, _, _ in input_panels],
            "channel_note": channel_note,
            "model_inputs_shown_from_recorded_feed": True,
            "red_overlay_opacity": 0.65,
            "mask_preview_resampling": "nearest",
            "path_preview_width_px": 2,
            "node_markers": "all_junctions",
            "manual_edits": False,
        },
    }
    return canvas, summary
