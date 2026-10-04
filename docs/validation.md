# Validation snapshot — 2026-10-04

This is a small, frozen engineering comparison, not a production accuracy guarantee. Source code is being released as an experimental toolkit; private models and raw experiment data are not distributed.

## Data and controls

The private evaluation used [CubiCasa5k](https://github.com/CubiCasa/CubiCasa5k) SVG human annotations from the [official archive](https://zenodo.org/records/2613548). Data uses CC BY-NC 4.0. Only aggregate measurements appear here.

Official train/validation/test splits supplied 90/15/30 sampled drawings. Each split balanced colorful, high_quality, and high_quality_architectural; test had ten drawings per source. Selection seed 20261003 was fixed before inference. Floor-plan group IDs and identical image bytes did not cross splits; visually near-duplicate drawings were not exhaustively excluded. These balanced samples do not estimate the whole dataset's natural distribution. Exposure of the legacy weights to these drawings is unknown.

Wall truth uses structural SVG Wall label 2, including annotated door/window opening structures and excluding railings. It differs from the concept of a perfectly closed opaque barrier. Predictions were restored to original image coordinates and positive wall polarity.

Legacy and new pipelines used the same private 1024×1024 ONNX wall model. All 30 legacy classifier routes selected it. Legacy preprocessing/routing/postprocessing stayed intact; WallGraph used fixed CLAHE and wall classes `[1, 2]`. The comparison therefore includes actual processing differences. It is not a behavior-identical code reorganization.

Only the 15 validation drawings selected the configuration: threshold 0.5 and closing radius 2. Thresholds 0.6/0.7 performed worse. The library's general default still has closing disabled. Test samples were not used to tune these settings.

## Test accuracy

All 30 cases completed; no parsing or inference failures were dropped. Macro means an equal-weight image average; micro aggregates all pixel counts.

| Output | Macro IoU | Macro F1 | Micro IoU |
| --- | ---: | ---: | ---: |
| Legacy final API mask | 0.78478 | 0.87768 | 0.76436 |
| Legacy raw segmentation reference | 0.86334 | 0.92555 | 0.85357 |
| WallGraph defaults | 0.84089 | 0.91190 | 0.81948 |
| Validation-selected WallGraph | 0.84203 | 0.91259 | 0.82097 |

Paired bootstrap within each source stratum, 10,000 repetitions, seed 20261003, percentile 95% intervals:

- Selected WallGraph minus legacy final mask: **+5.72 percentage points**, interval **[+4.94, +6.52]**; 30 improvements.
- Selected WallGraph minus legacy raw reference: **−2.13 points**, interval **[−3.25, −1.10]**; six improvements and 24 regressions.

Mean precision/recall changed from legacy final 0.81499/0.95247 to new selected 0.94116/0.88687. Fewer false walls came with more missed walls. New-wall plus new-region end-to-end macro PQ fell from the legacy two-stage 0.68034 to 0.56580. Fine-wall gaps and room merges matter even when wall pixel IoU improves.

Skeleton coverage F1 was 0.94347 for legacy final, 0.95910 for raw, and 0.95770 for new selected. It checks one mask's skeleton against the other mask dilated by two four-neighbor iterations. It does not measure graph connections, door semantics, or traversal correctness.

## Timing and perturbations

On one server with four CPU inference threads, seven fixed images received one warmup and three measured calls per method, giving 21 measurements. Full wall preprocessing, inference, cleanup, and vectorization were included; file I/O was excluded. New sessions were persistent and construction excluded. The legacy API naturally reconstructed classifier/wall sessions each call, and that cost was included.

| Full wall pipeline | Median | P95 |
| --- | ---: | ---: |
| Legacy CPU | 3207.78 ms | 4190.56 ms |
| New CPU | 1446.38 ms | 1849.59 ms |
| New RTX 5090 CUDA | 166.65 ms | 599.93 ms |

The numbers describe these deployment choices and inputs, not an isolated algorithm speedup or production tail latency. New CPU/CUDA mask agreement IoU was at least 0.999617, not bit-identical. CUDA requests disabled CPU node fallback; earlier profiling found all 149 executed model nodes on CUDA.

Three fixed validation images had mean IoU 0.83351 unchanged, 0.83449 after JPEG quality 50, 0.83793 with noise sigma 8, and 0.77697 after a 90-degree rotation restored to original coordinates. Rotation averaged −5.65 points, with a worst −12.06 points. There was no paired legacy perturbation comparison or multiple noise-seed study.

## New training baseline and export

SmallUNet uses 16/32/64 channels, GroupNorm, SiLU, BCE+Dice, AdamW at learning rate 0.001, RGB 512 input, batch four, and seed 42. Ninety training and 15 validation images ran for 20 epochs on RTX 5090. Padding was excluded and validation micro IoU selected epoch 18 (0.61542 at training resolution).

On the 30 held-out images in original coordinates, test macro IoU was **0.59297**, micro IoU **0.56866**, and micro F1 **0.72502**. It cannot replace the existing model. Different resolutions, training budgets, and unknown legacy training exposure prevent an equal-budget architecture claim. These research weights are not published.

Against PyTorch CPU on one fixed validation tensor, default ORT CPU maximum sigmoid-probability error was 0.002421 and default CUDA 0.001225; both failed the chosen threshold of 0.001. Binary pixel agreements were 99.99847%/99.99886%. CUDA with TF32 disabled had maximum error 0.000003636 and complete binary agreement. CPU differences remain unresolved. See [ORT's TF32 documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#use_tf32); one tensor does not establish a general error bound.

## Engineering scope

The paired runtime used Python 3.12.14, NumPy 1.26.4, OpenCV 4.11.0.86, SciPy 1.14.1, scikit-image 0.24.0, ORT GPU 1.26.0, PyTorch 2.8.0+cu128, CUDA 12.8, and driver 580.95.05. Legacy dependency incompatibilities were resolved in a common environment rather than counted as accuracy failures.

WallGraph has 13 local tests including CPU training and ONNX contracts. Both extracted packages together passed 33 local tests, 32 server tests with one CPU-only skip, and 100 generated-layout geometry/protocol invariant checks. Fresh wheel installation and the model-free demo worked without the legacy application, PyTorch, or ORT. These are engineering checks, not wall graph ground-truth validation.

See [the detailed Chinese record](results.md) for historical measurements. Next priorities are fine-wall recall, opening-aware barriers, graph ground truth, export precision controls, and a newly frozen independent business-data test set after further tuning.
