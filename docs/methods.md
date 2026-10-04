# From wall pixels to a connection graph

[中文](methods.zh-CN.md) · [Reproduction notes](reproduction.md) · [Validation](validation.md)

A wall mask locates pixels. A spatial application also needs turns, junctions, and loops. WallGraph carries the mask into an inspectable graph in original image coordinates, with replaceable segmentation and geometry stages.

| Core choice | Purpose | Implementation |
| --- | --- | --- |
| Original-coordinate probabilities and explicit backends | Keep model preprocessing, class IDs, device, and output shape inspectable | [backends.py](../src/wallgraph/backends.py), [pipeline.py](../src/wallgraph/pipeline.py) |
| Skeleton nodes and edge tracing | Cluster junction pixels, remove corner shortcuts, trace edges, and anchor degree-two cycles | [vectorize.py](../src/wallgraph/vectorize.py) |
| Separate geometry from pixel evidence | Simplify paths while retaining masks, pixel thickness, and uncalibrated probability confidence | [domain.py](../src/wallgraph/domain.py), [vectorize.py](../src/wallgraph/vectorize.py) |
| Blend overlapping probabilities before thresholding | Avoid hard tile seams and retain positive coverage at image edges | [tiling.py](../src/wallgraph/tiling.py) |
| Compact private-data training | Train, select on validation, and export, excluding padding from loss | [training.py](../src/wallgraph/training.py) |

Tracing cannot restore missing walls. Adjacency rules and short-path filtering can affect connections. Generated tests are not annotated graph accuracy. Tiling has not shown an accuracy gain in the frozen 30-drawing comparison.

## Research lineage

| Original work | Used here | Scope |
| --- | --- | --- |
| Ronneberger, Fischer, Brox, [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597), MICCAI 2015 | Encoder/decoder and skip concatenation in `SmallUNet` | Adapted two-level 16/32/64-channel network, padded convolution, bilinear upsampling, GN/SiLU, BCE+Dice. Original architecture, medical experiments, training recipe, and results were not reproduced |
| Wu, He, [Group Normalization](https://arxiv.org/abs/1803.08494), ECCV 2018 | `nn.GroupNorm(4, channels)` | A batch-four design choice. No GN/BN ablation supports an accuracy benefit here |
| Zhang, Suen, [A fast parallel algorithm for thinning digital patterns](https://doi.org/10.1145/357994.358023), CACM 1984 | Two-dimensional thinning through `skimage.morphology.skeletonize` | Library implementation; [its documentation](https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.skeletonize) identifies Zhang as the 2D default. Subsequent graph tracing is implemented here |
| Kalervo et al., [CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis](https://arxiv.org/abs/1904.01920), 2019 | Official drawings and SVG human annotations for private evaluation | Data/task reference. No reproduction of its complete multi-task model, full training, or leaderboard result |

Thickness uses [SciPy's exact Euclidean distance transform](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html) and the median path estimate `max(1, 2*distance-1)`. It measures pixels; diagonals, junctions, and resolution introduce error. OpenCV `approxPolyDP` simplifies polylines. Weighted probability fusion is a general engineering implementation, without a claimed paper-result reproduction.

## Contribution and interpretation

The work is in coordinate restoration, connection structure, model contracts, and auditable experiments. A backend can change without changing the output convention; a vectorizer can change without silently rewriting the mask.

The private reference ONNX model's architecture lineage and training exposure are unconfirmed. It cannot be attributed to an official paper checkpoint. Test macro IoU 0.84203 uses that external reference plus the new processing pipeline; the separately trained `SmallUNet` reached 0.59297. Read them as separate experiments. [Validation](validation.md) gives controls and limitations.
