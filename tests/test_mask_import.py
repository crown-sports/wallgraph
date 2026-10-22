import json
import subprocess
import sys

import cv2
import numpy as np
import pytest
from PIL import Image

from wallgraph import WallConfig, vectorize_mask


def test_import_preserves_coordinates_polarity_and_tiny_evidence():
    mask = np.zeros((80, 240), np.uint8)
    cv2.line(mask, (20, 20), (220, 60), 255, 7)
    mask[2, 2] = 255
    original = mask.copy()
    result = vectorize_mask(mask)
    assert np.array_equal(mask, original)
    assert np.array_equal(result.mask, mask)
    assert np.array_equal(vectorize_mask(255 - mask, wall_value=0).mask, mask)
    assert result.to_dict()["image"] == {"width": 240, "height": 80}
    assert 190 < max(s.length_px for s in result.segments) < 215
    assert result.metadata["changed_pixels"] == 0
    assert "not_model_probability" in result.metadata["confidence_kind"]
    filtered = vectorize_mask(mask, WallConfig(min_component_area=2))
    assert filtered.metadata["changed_pixels"] == 1


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((0, 2), np.uint8),
        np.ones((3, 3), np.uint8),
        np.zeros((3, 3, 3), np.uint8),
        np.zeros((3, 3)),
    ],
)
def test_invalid_masks_are_rejected(mask):
    with pytest.raises(ValueError, match="binary"):
        vectorize_mask(mask)


def test_cli_exports_contract_and_rejects_ambiguous_or_overwritten_inputs(tmp_path):
    mask = np.zeros((50, 80), np.uint8)
    mask[10:40, 30:35] = 255
    path = tmp_path / "source.png"
    Image.fromarray(mask).save(path)
    output = tmp_path / "out"
    command = [
        sys.executable,
        "-m",
        "wallgraph.cli",
        "vectorize",
        "--mask",
        str(path),
        "--output",
        str(output),
    ]
    assert subprocess.run(command, capture_output=True).returncode == 0
    report = json.loads((output / "walls.json").read_text())
    assert report["schema_version"] == "wallgraph/1"
    assert len(report["metadata"]["source_sha256"]) == 64
    assert np.array_equal(np.asarray(Image.open(output / "walls.png")), mask)
    Image.fromarray(np.full((3, 3), 128, np.uint8)).save(path)
    assert subprocess.run(command, capture_output=True).returncode == 2
    before = (output / "walls.png").read_bytes()
    command[command.index(str(path))] = str(output / "walls.png")
    assert subprocess.run(command, capture_output=True).returncode == 2
    assert (output / "walls.png").read_bytes() == before


def test_closed_ring_retains_anchor_closing_edge_and_perimeter():
    mask = np.zeros((90, 120), np.uint8)
    mask[10:15, 10:-10] = mask[-15:-10, 10:-10] = 255
    mask[10:-10, 10:15] = mask[10:-10, -15:-10] = 255
    result = vectorize_mask(mask)
    assert len(result.segments) == 1
    ring = result.segments[0]
    anchor = result.junctions[0].point
    assert ring.start_node == ring.end_node == result.junctions[0].id
    assert ring.points[0] == ring.points[-1] == anchor
    assert 310 < ring.length_px < 325
    measured = sum(
        np.hypot(b.x - a.x, b.y - a.y) for a, b in zip(ring.points, ring.points[1:], strict=False)
    )
    assert ring.length_px == pytest.approx(measured, rel=1e-6)
