import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

_SPEC = importlib.util.spec_from_file_location(
    "pipeline_figure", Path(__file__).resolve().parents[1] / "tools/pipeline_figure.py"
)
figure_tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(figure_tool)


@pytest.fixture
def model(tmp_path):
    helper = onnx.helper
    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["images"], ["output"], axes=[1], keepdims=1)],
        "observable-input",
        [helper.make_tensor_value_info("images", onnx.TensorProto.FLOAT, [1, 3, 32, 64])],
        [helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 1, 32, 64])],
    )
    value = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    value.ir_version = 8
    path = tmp_path / "private-test-model.onnx"
    onnx.save(value, path)
    return path


@pytest.mark.parametrize("preprocess", ["clahe", "rgb", "gray"])
def test_figure_observes_one_real_session_and_unmodified_pipeline(
    model, tmp_path, monkeypatch, preprocess
):
    rgb = np.empty((15, 39, 3), np.uint8)
    rgb[:, :, 0] = np.linspace(5, 185, 39, dtype=np.uint8)
    rgb[:, :, 1] = np.linspace(60, 210, 15, dtype=np.uint8)[:, None]
    rgb[:, :, 2] = 20 + (np.indices((15, 39)).sum(axis=0) % 2) * 100
    original = rgb.copy()
    rgb.flags.writeable = False
    feeds, results, panels = [], [], []
    original_run = ort.InferenceSession.run

    def observed_run(session, names, input_feed, *args, **kwargs):
        feeds.append({name: value.copy() for name, value in input_feed.items()})
        return original_run(session, names, input_feed, *args, **kwargs)

    monkeypatch.setattr(ort.InferenceSession, "run", observed_run)
    original_pipeline_run = figure_tool.WallPipeline.run

    def observed_pipeline(pipeline, image):
        result = original_pipeline_run(pipeline, image)
        results.append(result)
        return result

    monkeypatch.setattr(figure_tool.WallPipeline, "run", observed_pipeline)
    original_paste = figure_tool._paste_panel

    def observed_panel(canvas, pixels, index, title, description, **kwargs):
        panels.append({"pixels": pixels.copy(), "index": index, "title": title, **kwargs})
        return original_paste(canvas, pixels, index, title, description, **kwargs)

    monkeypatch.setattr(figure_tool, "_paste_panel", observed_panel)
    figure, summary = figure_tool.run_and_render(
        rgb,
        model,
        preprocess=preprocess,
        wall_classes=(1,),
        output_kind="probabilities",
        close_radius=0,
    )
    assert len(feeds) == len(results) == summary["inference_calls"] == 1
    assert [panel["index"] for panel in panels] == list(range(6))
    tensor = feeds[0]["images"]
    assert summary["actual_input_tensor"]["shape"] == list(tensor.shape) == [1, 3, 32, 64]
    assert summary["actual_input_tensor"]["dtype"] == str(tensor.dtype) == "float32"
    assert summary["output_tensor_shape"] == [1, 1, 32, 64]
    # Compare display pixels to the actual intercepted feed, independently of
    # the renderer's preprocessing; no recomputed CLAHE or letterbox is assumed.
    captured_display = np.rint(tensor[0].transpose(1, 2, 0) * 255).astype(np.uint8)
    if preprocess == "rgb":
        np.testing.assert_array_equal(panels[1]["pixels"], captured_display)
        np.testing.assert_array_equal(panels[2]["pixels"], captured_display[:, :, 0])
        assert "RGB" in panels[1]["title"] and ": R" in panels[2]["title"]
        assert all("CLAHE" not in panel["title"] for panel in panels)
    else:
        np.testing.assert_array_equal(panels[1]["pixels"], captured_display[:, :, 0])
        np.testing.assert_array_equal(panels[2]["pixels"], captured_display[:, :, 1])
        if preprocess == "clahe":
            assert "gray" in panels[1]["title"] and "CLAHE20" in panels[2]["title"]
        else:
            assert all("gray" in panel["title"] for panel in panels[1:3])
            assert all("CLAHE" not in panel["title"] for panel in panels)
    result = results[0]
    np.testing.assert_array_equal(panels[3]["pixels"], result.mask)
    assert panels[3]["nearest"] is True
    assert panels[5]["result"] is result
    background = result.mask == 0
    np.testing.assert_array_equal(panels[4]["pixels"][background], rgb[background])
    assert summary["measurements"]["final_wall_pixels"] == int((result.mask > 0).sum())
    assert summary["measurements"]["paths"] == len(result.segments)
    assert summary["measurements"]["nodes"] == len(result.junctions)
    np.testing.assert_array_equal(rgb, original)
    assert figure.size == (1560, 1000) and max(figure.size) <= 2048
    assert figure.mode == "RGB" and figure.info == {}
    output = tmp_path / "six-stages.png"
    figure.save(output)
    with Image.open(output) as saved:
        assert saved.mode == "RGB" and saved.size == figure.size and saved.info == {}
    serialized = json.dumps(summary, allow_nan=False)
    assert str(tmp_path) not in serialized and model.name not in serialized
    assert summary["display"]["manual_edits"] is False


@pytest.mark.parametrize("problem", ["input", "threshold", "closing"])
def test_invalid_figure_request_fails_before_model_construction(monkeypatch, problem):
    def forbidden_backend(*args, **kwargs):
        pytest.fail("invalid input constructed an inference session")

    monkeypatch.setattr(figure_tool, "_RecordingBackend", forbidden_backend)
    rgb = np.zeros((3, 7, 3), np.uint8)
    arguments = {}
    if problem == "input":
        rgb = rgb.astype(np.float32)
    elif problem == "threshold":
        arguments["threshold"] = 0
    else:
        arguments["close_radius"] = -1
    with pytest.raises(ValueError):
        figure_tool.run_and_render(rgb, Path("missing.onnx"), **arguments)
