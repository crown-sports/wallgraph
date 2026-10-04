import json

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from wallgraph.training import (  # noqa: E402
    ManifestDataset,
    SmallUNet,
    segmentation_loss,
    validate_split,
)


def test_training_loss_has_finite_gradients_and_ignores_padding():
    model = SmallUNet()
    logits = model(torch.rand(2, 3, 32, 64))
    truth = torch.zeros_like(logits)
    valid = torch.ones_like(logits)
    valid[:, :, :4] = 0
    loss = segmentation_loss(logits, truth, valid)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_group_leakage_and_duplicate_images_are_rejected(tmp_path):
    Image.fromarray(np.zeros((20, 50, 3), np.uint8)).save(tmp_path / "image.png")
    Image.fromarray(np.zeros((20, 50), np.uint8)).save(tmp_path / "mask.png")
    for group, name in (("same", "train"), ("same", "val")):
        (tmp_path / f"{name}.jsonl").write_text(
            json.dumps(
                {"image": "image.png", "mask": "mask.png", "group": group},
            )
            + "\n"
        )
    train = ManifestDataset(tmp_path / "train.jsonl", 32)
    validation = ManifestDataset(tmp_path / "val.jsonl", 32)
    with pytest.raises(ValueError, match="groups overlap"):
        validate_split(train, validation)
    validation.records[0]["group"] = "different"
    with pytest.raises(ValueError, match="identical image"):
        validate_split(train, validation)
    image, truth, valid = train[0]
    assert image.shape == (3, 32, 32)
    assert valid.sum() < 32 * 32
    assert not truth.any()
