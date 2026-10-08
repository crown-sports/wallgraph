import importlib.util
from pathlib import Path

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

spec = importlib.util.spec_from_file_location(
    "release", Path(__file__).parents[1] / "tools/release.py"
)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def test_public_export_excludes_models_data_and_generated_results():
    files = release.public_files()
    assert all(path.suffix not in {".onnx", ".pt", ".jsonl", ".npz"} for path in files)
    assert sum(path.suffix == ".png" for path in files) == 1


def test_extra_public_image_and_symlink_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(release, "ROOT", tmp_path)
    monkeypatch.setattr(release, "TOP_LEVEL", set())
    (tmp_path / "examples").mkdir()
    unexpected = tmp_path / "examples/private.png"
    unexpected.write_bytes(b"unused")
    with pytest.raises(ValueError, match="single reviewed"):
        release.public_files()
    unexpected.unlink()
    (tmp_path / "examples/link.png").symlink_to(tmp_path / "elsewhere.png")
    with pytest.raises(ValueError, match="symlink"):
        release.public_files()


def test_oversized_or_metadata_bearing_demo_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(release, "ROOT", tmp_path)
    monkeypatch.setattr(release, "TOP_LEVEL", set())
    (tmp_path / "examples").mkdir()
    demo = tmp_path / "examples/simple.png"
    Image.new("RGB", (2049, 1)).save(demo)
    with pytest.raises(ValueError, match="public limit"):
        release.public_files()
    metadata = PngInfo()
    metadata.add_text("source", "unreviewed source details")
    Image.new("RGB", (32, 32)).save(demo, pnginfo=metadata)
    with pytest.raises(ValueError, match="no metadata"):
        release.public_files()
