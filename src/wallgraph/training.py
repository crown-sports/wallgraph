"""A new compact segmentation baseline, trained only from user-owned manifests."""

import hashlib
import json
import random
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .evaluate import read_manifest
from .io import read_rgb


class ConvBlock(nn.Sequential):
    def __init__(self, incoming: int, outgoing: int) -> None:
        super().__init__(
            nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(4, outgoing),
            nn.SiLU(),
            nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(4, outgoing),
            nn.SiLU(),
        )


class SmallUNet(nn.Module):
    """Two encoder levels; a conventional U-Net baseline, not a novelty claim."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder1 = ConvBlock(3, 16)
        self.encoder2 = ConvBlock(16, 32)
        self.center = ConvBlock(32, 64)
        self.decoder2 = ConvBlock(96, 32)
        self.decoder1 = ConvBlock(48, 16)
        self.output = nn.Conv2d(16, 1, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        first = self.encoder1(image)
        second = self.encoder2(nn.functional.max_pool2d(first, 2))
        center = self.center(nn.functional.max_pool2d(second, 2))
        up2 = nn.functional.interpolate(
            center, size=second.shape[2:], mode="bilinear", align_corners=False
        )
        second_out = self.decoder2(torch.cat([up2, second], dim=1))
        up1 = nn.functional.interpolate(
            second_out, size=first.shape[2:], mode="bilinear", align_corners=False
        )
        return self.output(self.decoder1(torch.cat([up1, first], dim=1)))


class ManifestDataset(Dataset):
    def __init__(self, manifest: Path, size: int) -> None:
        self.manifest = manifest
        self.records = read_manifest(manifest)
        self.size = size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[index]
        image = read_rgb(self.manifest.parent / record["image"])
        with Image.open(self.manifest.parent / record["mask"]) as source:
            mask = np.asarray(source.convert("L"))
        if image.shape[:2] != mask.shape:
            raise ValueError("image and binary annotation sizes must match")
        if not set(np.unique(mask)).issubset({0, 1, 255}):
            raise ValueError("training requires binary wall masks, with 0 background")
        height, width = mask.shape
        scale = min(self.size / height, self.size / width)
        ch, cw = max(1, round(height * scale)), max(1, round(width * scale))
        top, left = (self.size - ch) // 2, (self.size - cw) // 2
        canvas = np.full((self.size, self.size, 3), 128, np.uint8)
        truth = np.zeros((self.size, self.size), np.float32)
        valid = np.zeros_like(truth)
        canvas[top : top + ch, left : left + cw] = cv2.resize(
            image,
            (cw, ch),
            interpolation=cv2.INTER_CUBIC,
        )
        truth[top : top + ch, left : left + cw] = cv2.resize(
            (mask > 0).astype(np.float32),
            (cw, ch),
            interpolation=cv2.INTER_NEAREST,
        )
        valid[top : top + ch, left : left + cw] = 1
        return (
            torch.from_numpy(canvas.transpose(2, 0, 1).copy()).float() / 255,
            torch.from_numpy(truth[None]),
            torch.from_numpy(valid[None]),
        )


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_split(train: ManifestDataset, validation: ManifestDataset) -> None:
    if {r["group"] for r in train.records} & {r["group"] for r in validation.records}:
        raise ValueError("floor-plan groups overlap between training and validation")
    train_images = {_digest(train.manifest.parent / r["image"]) for r in train.records}
    val_images = {_digest(validation.manifest.parent / r["image"]) for r in validation.records}
    if train_images & val_images:
        raise ValueError("identical image bytes occur in both splits")


def segmentation_loss(
    logits: torch.Tensor, truth: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, truth, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp_min(1)
    probabilities = logits.sigmoid() * valid
    dice = (2 * (probabilities * truth).sum() + 1) / (
        probabilities.sum() + (truth * valid).sum() + 1
    )
    return bce + 1 - dice


def train_model(args: Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1 or args.size < 16 or args.size % 4:
        raise ValueError("positive epochs/batch size and size >=16 divisible by four required")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    training = ManifestDataset(args.train_manifest, args.size)
    validation = ManifestDataset(args.val_manifest, args.size)
    validate_split(training, validation)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(training, args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(validation, args.batch_size)
    model = SmallUNet().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    best_iou = -1.0
    args.output.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        for image, truth, valid in loader:
            image, truth, valid = (value.to(args.device) for value in (image, truth, valid))
            optimizer.zero_grad(set_to_none=True)
            loss = segmentation_loss(model(image), truth, valid)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(image)
        model.eval()
        tp = fp = fn = 0
        with torch.inference_mode():
            for image, truth, valid in val_loader:
                pred = (model(image.to(args.device)).sigmoid() >= 0.5).cpu()
                ground = truth > 0.5
                usable = valid > 0
                tp += int((pred & ground & usable).sum())
                fp += int((pred & ~ground & usable).sum())
                fn += int((~pred & ground & usable).sum())
        iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
        row = {"epoch": epoch + 1, "train_loss": loss_sum / len(training), "val_iou": iou}
        history.append(row)
        print(json.dumps(row), flush=True)
        if iou > best_iou:
            best_iou = iou
            torch.save(
                {key: value.detach().cpu() for key, value in model.state_dict().items()},
                args.output / "best.pt",
            )
    report = {
        "architecture": "wallgraph-small-unet/1",
        "seed": args.seed,
        "input_size": args.size,
        "train_count": len(training),
        "val_count": len(validation),
        "train_manifest_sha256": _digest(args.train_manifest),
        "val_manifest_sha256": _digest(args.val_manifest),
        "checkpoint_sha256": _digest(args.output / "best.pt"),
        "device": args.device,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "metric_scope": "validation pixels at letterboxed training resolution, padding excluded",
        "selection": "best validation IoU; this is model selection, not held-out test accuracy",
        "history": history,
    }
    (args.output / "training.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def export_model(args: Namespace) -> None:
    if args.size < 16 or args.size % 4:
        raise ValueError("size must be >=16 and divisible by four")
    model = SmallUNet()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.zeros(1, 3, args.size, args.size),
        str(args.output),
        input_names=["images"],
        output_names=["wall_logits"],
        opset_version=17,
        dynamo=False,
    )
