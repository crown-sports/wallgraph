# Train a wall model on your own drawings

[中文](training-guide.zh-CN.md) · [Methods](methods.md) · [Measured results](validation.md) · [GPU runtime](runtime.md)

This guide takes you from a local import to a binary wall model, an ONNX export, and a held-out evaluation. It follows the current SmallUNet implementation. Bring images and annotations you are allowed to use; the repository supplies code and one generated demonstration image, not a training dataset or pretrained checkpoint.

## 1. Install the code you will actually import

Use Python 3.10 or newer. From a local clone, choose the CPU environment below. The `train` extra installs PyTorch and ONNX; the separate `onnx` extra installs the CPU inference runtime needed for `detect` and `evaluate`.

```bash
git clone https://github.com/crown-sports/wallgraph.git
cd wallgraph
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train,onnx]'
export WG_DEVICE=cpu
python -c 'import wallgraph; print(wallgraph.__version__, wallgraph.__file__)'
python -m wallgraph.cli train --help
```

Editable installation makes `src/wallgraph` importable from the active interpreter. Use `python -m wallgraph.cli` if the shell's `wallgraph` command points to a different environment. Avoid naming your own script `wallgraph.py` or `torch.py`. The top-level import does not load PyTorch or create an ONNX session; `wallgraph.training` requires the training extra, while constructing `OnnxBackend` requires an ONNX Runtime wheel.

For Linux NVIDIA execution, choose a fresh environment and install `'.[train,gpu]'` instead, then set `export WG_DEVICE=cuda`. Do not install both `onnxruntime` and `onnxruntime-gpu` in that environment. Confirm both frameworks before a long run:

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python -c 'import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())'
```

GPU availability in PyTorch does not establish ONNX CUDA availability. Follow the [runtime notes](runtime.md) for the driver and framework combination. This CLI supports `cpu` and `cuda`; Apple MPS is not a selectable training device.

To check a direct model import without data:

```python
import torch
from wallgraph.training import SmallUNet

model = SmallUNet().eval()
with torch.inference_mode():
    logits = model(torch.zeros(1, 3, 32, 64))
print(tuple(logits.shape))  # (1, 1, 32, 64)
```

These are randomly initialized weights. This check verifies tensor plumbing, not recognition quality.

## 2. Give the model an unambiguous target

Store data outside the checkout. Set the following variable to your own absolute directory; all later shell commands use it.

```bash
export WG_DATA="/absolute/path/to/private-data"
```

A possible layout is:

```text
private-data/
  images/plan-a.png
  masks/plan-a.png
  train.jsonl
  val.jsonl
  test.jsonl
```

Each manifest is UTF-8 JSONL, one object per line, not a JSON array:

```json
{"image":"images/plan-a.png","mask":"masks/plan-a.png","group":"plan-a"}
```

`image`, `mask`, and `group` must be nonempty strings. Relative file paths resolve from the manifest's directory, not the shell's working directory. Each split must contain at least one record.

The image is decoded as RGB. Supply a single-channel lossless mask with the same height and width: `0` is background; `1` or `255` is wall. A mixture of `1` and `255` is accepted and both mean foreground. Other values are rejected by the training loader. Do not use JPEG labels, antialiased outlines, colored overlay images, or unmapped multiclass IDs. Convert semantic class IDs to a binary wall mask explicitly before creating the manifest. Evaluation treats every nonzero mask value as foreground; it does not perform that semantic mapping for you.

Define what “wall” includes before annotating: openings, window structures, railings, and furniture are not interchangeable. Keep that rule identical across all splits. A structural wall label that includes a door opening is not automatically a closed barrier suitable for room extraction.

Split original floor plans before generating crops or augmentations. Every derivative of one plan belongs to the same `group`; related pages from the same drawing or building may need an even broader group. Assign each group to exactly one split. Preserve official splits when applicable, and record the selection seed and annotation version privately. The trainer rejects overlapping train/validation groups and identical image bytes across those two splits. It does not automatically check the test split, visually similar duplicates, or undocumented provenance.

## 3. Check all three splits before using the GPU

This preflight reuses the actual loader, checks every image/mask pair, and applies the available overlap check to all split pairs. It reads data without publishing it.

```bash
python - <<'PY'
import os
from itertools import combinations
from pathlib import Path
from wallgraph.training import ManifestDataset, validate_split

root = Path(os.environ["WG_DATA"])
datasets = {
    name: ManifestDataset(root / f"{name}.jsonl", size=512)
    for name in ("train", "val", "test")
}
for first, second in combinations(datasets, 2):
    validate_split(datasets[first], datasets[second])
for name, dataset in datasets.items():
    for index in range(len(dataset)):
        image, truth, valid = dataset[index]
        assert image.shape == (3, 512, 512)
        assert truth.shape == valid.shape == (1, 512, 512)
    print(name, "records:", len(dataset))
PY
```

Also inspect some overlays locally, especially thin walls and empty masks. Valid shapes and values cannot detect a shifted annotation, a wrong wall definition, or near-duplicate plans. Re-encoded copies have different file hashes, so grouping still matters.

## 4. Run a small check, then the baseline

Use a new output directory for each run: the command can overwrite `best.pt` and `training.json`. First run one epoch to find data or device errors:

```bash
python -m wallgraph.cli train \
  --train-manifest "$WG_DATA/train.jsonl" \
  --val-manifest "$WG_DATA/val.jsonl" \
  --device "$WG_DEVICE" --size 128 --epochs 1 --batch-size 2 --seed 42 \
  --output "$WG_DATA/runs/smoke"
```

For deterministic CUDA runs, set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before starting Python. The trainer enables deterministic algorithms and may reject unsupported operations. Seeds and deterministic settings still do not guarantee identical results across devices or framework versions; see [PyTorch's reproducibility notes](https://docs.pytorch.org/docs/2.8/notes/randomness.html).

Run the baseline only after the check succeeds:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m wallgraph.cli train \
  --train-manifest "$WG_DATA/train.jsonl" \
  --val-manifest "$WG_DATA/val.jsonl" \
  --device "$WG_DEVICE" --size 512 --epochs 20 --batch-size 4 --seed 42 \
  --output "$WG_DATA/runs/baseline"
```

For a CUDA smoke run, apply the same environment-variable prefix to the smoke command. `size` must be at least 16 and divisible by four; epochs and batch size must be positive. CPU is useful for checking the chain but has not been assigned a promised training time.

The current recipe is fixed: RGB divided by 255, two encoder levels with 16/32/64 channels, GroupNorm, SiLU, BCE plus soft Dice, and AdamW with learning rate `0.001` and weight decay `0.0001`. It uses FP32, without augmentation, learning-rate scheduling, mixed precision, gradient accumulation, or distributed training. Those are not hidden CLI switches. `--seed` controls initialization and training shuffle; it does not generate a dataset split.

Each epoch prints `epoch`, `train_loss`, and `val_iou`. Validation uses threshold `0.5` and pools TP/FP/FN over all valid pixels: this is micro IoU at training resolution. The first epoch that attains a new best score saves `best.pt`; ties retain the earlier checkpoint. `training.json`, written after all epochs finish, records the history, counts, manifest/checkpoint hashes, runtime, and selection rule. It does not record every command argument; retain the command, code commit, and `python -m pip freeze` output with the private run. There is no resume command or saved optimizer state.

## 5. Understand resizing before exporting

A non-square image is scaled to fit a square `size × size` canvas without stretching, centered, and padded with RGB value 128. Image interpolation is bicubic; mask interpolation is nearest-neighbor. The loader returns `(image, truth, valid)`. The validity mask excludes padding from both loss and validation counts. It is not an arbitrary annotation-ignore mask: value `255` in your label still means wall.

Export the chosen checkpoint at the same size used for training:

```bash
python -m wallgraph.cli export-onnx \
  --checkpoint "$WG_DATA/runs/baseline/best.pt" --size 512 \
  --output "$WG_DATA/runs/baseline/model.onnx"
```

| Exported contract | Value |
| --- | --- |
| Input name and layout | `images`, float32 `[1, 3, 512, 512]`, RGB in `[0, 1]` |
| Output name and layout | `wall_logits`, `[1, 1, 512, 512]` logits |
| Graph | ONNX opset 17; fixed batch and spatial dimensions |
| Postprocessing | Sigmoid, removal of letterbox padding, probability resize to original coordinates, then thresholding |

`export-onnx` builds SmallUNet and loads its state dictionary on CPU; it is not a general converter for arbitrary checkpoints. `--size` sets a fixed export shape rather than dynamic axes. For this binary model use `--preprocess rgb --wall-classes 1 --output-kind logits`. The `1` denotes the foreground contract, not a second output channel. Changing inference to CLAHE or grayscale would change the input distribution from this training recipe. An exported fixed shape takes precedence over inference `--input-size`.

Load your own saved model directly when debugging PyTorch:

```python
import os
from pathlib import Path
import torch
from wallgraph.training import ManifestDataset, SmallUNet

root = Path(os.environ["WG_DATA"])
model = SmallUNet().eval()
model.load_state_dict(
    torch.load(root / "runs/baseline/best.pt", map_location="cpu", weights_only=True)
)
image, truth, valid = ManifestDataset(root / "val.jsonl", 512)[0]
with torch.inference_mode():
    probability = model(image[None]).sigmoid()[0, 0]
print(tuple(probability.shape), float(probability.min()), float(probability.max()))
```

The result here is in padded training coordinates. For an original-coordinate mask, use the inference pipeline below.

## 6. Evaluate the exported model on held-out drawings

```bash
python -m wallgraph.cli evaluate \
  --manifest "$WG_DATA/test.jsonl" \
  --model "$WG_DATA/runs/baseline/model.onnx" \
  --device "$WG_DEVICE" --preprocess rgb --wall-classes 1 \
  --output-kind logits --threshold 0.5 --close-radius 0 \
  --output "$WG_DATA/runs/baseline/test.json"
```

Always supply `--model`: without it, this CLI evaluates the dark-ink demo backend. The evaluation report contains micro IoU, micro F1, mean skeleton-coverage F1, configuration, model/manifest hashes, and latency. It does not currently contain macro pixel IoU or graph-connectivity accuracy. Predictions are evaluated against the original-resolution mask. Its latency excludes file I/O and session creation and has no warmup; use `benchmark` for a controlled warmup/repetition protocol.

For local visual inspection, replace `plan-a.png` with an image you own:

```bash
python -m wallgraph.cli detect \
  --image "$WG_DATA/images/plan-a.png" \
  --model "$WG_DATA/runs/baseline/model.onnx" \
  --device "$WG_DEVICE" --preprocess rgb --wall-classes 1 \
  --output-kind logits --threshold 0.5 --close-radius 0 \
  --output "$WG_DATA/runs/baseline/inspection"
```

Before accepting an export, compare PyTorch and ONNX sigmoid probabilities on the same preprocessed validation tensors. Record maximum/mean absolute error and binary agreement with a tolerance chosen beforehand. Export success alone is insufficient: in our recorded experiment, default CPU and CUDA runtimes exceeded a `0.001` maximum-error threshold on one tensor. Disabling CUDA TF32 improved that diagnostic; CPU differences remain unresolved. The CLI does not expose a TF32 switch. See [the numerical results](validation.md#new-training-baseline-and-export) before claiming equivalence.

## 7. Troubleshoot the first run

| Symptom | Check |
| --- | --- |
| `No module named wallgraph` | Activate the intended environment, run its `python -m pip install -e ...` from the checkout, and print `wallgraph.__file__`. |
| `No module named torch` or `onnxruntime` | Install the relevant extra into that same interpreter; `train` does not include the inference runtime. |
| Manifest is empty / record keys missing | Use nonempty UTF-8 JSONL with `image`, `mask`, and `group` strings. |
| File not found / annotation-size error | Resolve paths from the manifest directory; compare the decoded image and mask dimensions. |
| Binary-mask error | Inspect stored label values; explicitly convert semantic classes and remove interpolated gray values. |
| Groups or identical bytes overlap | Rebuild the split by original plan. Renaming a group to bypass the error preserves the leakage. |
| CUDA unavailable or no compatible kernel | Check the actual PyTorch wheel and GPU architecture; ONNX additionally needs its CUDA provider. CPU remains an explicit alternative. |
| Deterministic cuBLAS error | Set the workspace variable before Python starts; retain the deterministic setting and inspect other unsupported operations. |
| Out of memory | Reduce batch size first. Reducing resolution also changes the experiment and may remove thin walls; record it. |
| Low loss but poor useful walls | Inspect empty/shifted masks, label definitions, foreground prevalence, and train/validation domain differences before adding epochs. |
| Good pixel score but merged rooms | Inspect thin-wall recall and openings. Closing may also erase real gaps; select it on validation and measure downstream regions. |

## 8. Choose the next experiment with a fair comparison

The measured baseline used 90 training and 15 validation drawings, 20 epochs, batch four, RGB 512, seed 42, and an RTX 5090. Validation selected epoch 18 with micro IoU `0.61542` at training resolution. On 30 held-out drawings, original-coordinate test macro IoU was `0.59297`, micro IoU `0.56866`, and micro F1 `0.72502`. These scores use different splits and aggregations; the validation score is not the test score. The private manifests and research weights are not distributed, so the public repository cannot reconstruct that exact table. The legacy model's budget and training exposure are unknown; this run does not establish an equal-budget improvement over it.

The following is a proposed experiment matrix, not a list of demonstrated gains:

| Question | Controlled comparison | Extra requirement |
| --- | --- | --- |
| How much does the seed matter? | Repeat the same recipe with several prespecified seeds | Same groups, training budget, and selection rule; report the distribution |
| Are small walls lost by resizing? | Compare 512 with 768 input | Export each at its own training size; report memory, time, and downstream room errors |
| Is postprocessing helping? | Same checkpoint, threshold `0.4/0.5/0.6`, closing `0/1/2` | Select only on validation; preserve opening-related failure cases |
| Would rotation or brightness augmentation help? | Baseline versus one added augmentation | Requires code changes; keep image and mask geometry synchronized |
| Would a different loss or learning rate help? | Change one training choice at a time | Requires code changes; the current CLI has no loss or learning-rate flag |

For architecture or loss comparisons, keep the data, initialization policy, update budget, resolution, augmentation, optimizer, checkpoint rule, and inference settings aligned except for the stated variable. Equal epochs with equal data and batch size give equal update counts; they do not give equal compute at different resolutions. Report measured device time and, if compute efficiency is the question, run a separate fixed-time comparison. Keep one final test set untouched while selecting candidates. If a test set has already guided changes, use a fresh holdout. Archive private manifests, commands, environment versions, hashes, and per-plan results so another authorized user can audit the conclusion.
