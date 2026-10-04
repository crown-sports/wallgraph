from pathlib import Path

import numpy as np
import pytest

from wallgraph.backends import OnnxBackend

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")


def make_model(path: Path):
    helper = onnx.helper
    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["images"], ["output"], axes=[1], keepdims=1)],
        "test",
        [helper.make_tensor_value_info("images", onnx.TensorProto.FLOAT, [1, 3, 32, 64])],
        [helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 1, 32, 64])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)


def test_letterbox_restores_probability_and_original_size(tmp_path):
    path = tmp_path / "test.onnx"
    make_model(path)
    backend = OnnxBackend(path, output_kind="probabilities")
    image = np.full((19, 93, 3), 255, np.uint8)
    probability = backend.predict(image)
    assert probability.shape == (19, 93)
    assert np.allclose(probability, 1)
    assert backend.session is backend.session
    with pytest.raises(ValueError):
        OnnxBackend(path, wall_classes=(0,)).predict(image)


def test_cuda_is_never_silently_reported_on_cpu(tmp_path):
    import onnxruntime

    if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
        pytest.skip("CPU-only runtime guard test")
    with pytest.raises(RuntimeError, match="CUDA provider unavailable"):
        OnnxBackend(tmp_path / "missing.onnx", device="cuda")
