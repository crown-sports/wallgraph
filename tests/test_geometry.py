import cv2
import numpy as np
import pytest

from wallgraph import WallConfig, WallPipeline
from wallgraph.backends import InkBackend
from wallgraph.io import draw_demo
from wallgraph.metrics import wall_metrics
from wallgraph.tiling import fuse_tiles


def test_t_junction_has_three_edges_and_common_node():
    image = np.full((100, 100, 3), 255, np.uint8)
    cv2.line(image, (15, 20), (85, 20), (0, 0, 0), 5)
    cv2.line(image, (50, 20), (50, 85), (0, 0, 0), 5)
    result = WallPipeline(InkBackend()).run(image)
    degrees = {}
    for edge in result.segments:
        for node in (edge.start_node, edge.end_node):
            degrees[node] = degrees.get(node, 0) + 1
    assert len(result.segments) == 3
    assert sorted(degrees.values()) == [1, 1, 1, 3]
    assert all(4 <= edge.thickness_px <= 8 for edge in result.segments)


def test_closed_loop_is_preserved():
    image = np.full((100, 100, 3), 255, np.uint8)
    cv2.rectangle(image, (20, 20), (80, 80), (0, 0, 0), 5)
    result = WallPipeline(InkBackend()).run(image)
    assert len(result.segments) == 1
    assert result.segments[0].start_node == result.segments[0].end_node
    assert len(result.segments[0].points) >= 5


def test_oblique_wall_and_non_square_coordinates():
    image = np.full((80, 240, 3), 255, np.uint8)
    cv2.line(image, (20, 20), (220, 60), (0, 0, 0), 7)
    result = WallPipeline(InkBackend()).run(image)
    assert len(result.segments) == 1
    assert 190 < result.segments[0].length_px < 215
    assert result.to_dict()["image"] == {"width": 240, "height": 80}


def test_tiling_covers_small_and_uneven_edges_without_changing_predictions():
    image = draw_demo()[:83, :211]

    def predictor(crop):
        return crop[:, :, 0].astype(np.float32) / 255

    assert np.allclose(fuse_tiles(image, predictor, 64, 13), predictor(image))
    assert np.allclose(fuse_tiles(image[:1], predictor, 64, 13), predictor(image[:1]))


def test_empty_input_returns_empty_graph_and_finite_scores():
    image = np.full((50, 100, 3), 255, np.uint8)
    result = WallPipeline(InkBackend()).run(image)
    assert not result.segments and not result.junctions
    assert wall_metrics(result.mask, result.mask)["iou"] == 1


def test_noise_is_removed_and_metrics_penalize_missing_walls():
    image = np.full((50, 100, 3), 255, np.uint8)
    image[10, 10] = 0
    assert not WallPipeline(InkBackend()).run(image).mask.any()
    truth = np.zeros((20, 20), np.uint8)
    truth[5:15, 5:15] = 255
    assert wall_metrics(np.zeros_like(truth), truth)["iou"] == 0
    assert wall_metrics(truth, truth)["skeleton_f1"] == 1


def test_bad_config_and_backend_fail_early():
    with pytest.raises(ValueError):
        WallConfig(tile_size=10, overlap=10)
    with pytest.raises(ValueError):
        WallConfig(threshold=float("nan"))

    class WrongBackend:
        name = "invalid"

        def predict(self, image):
            return np.full(image.shape[:2], np.nan, np.float32)

    with pytest.raises(ValueError):
        WallPipeline(WrongBackend()).run(draw_demo())
