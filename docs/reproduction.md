# From a running model to an interpretable result

[中文](reproduction.zh-CN.md) · [Methods and papers](methods.md) · [Validation](validation.md)

The useful part of this engineering reproduction was finding the gap between successful execution and usable geometry. These notes come from actual training, inference, and paired evaluation. They cover adapted methods on drawings; they do not recreate U-Net's medical experiments.

## Establish truth before comparing

The first four real drawings supported integration and timing checks. Following the original loader's `F1_scaled.png`, `model.svg`, and split conventions identified the annotation source. A fixed seed then selected 90/15/30 train/validation/test drawings. Validation chose settings; test measured them. The [dataset paper](https://arxiv.org/abs/1904.01920) gives provenance; private selections and weights are not distributed.

Structural wall truth includes annotated opening structures. Pixel agreement and a closed room barrier therefore ask different questions. We retained both legacy raw segmentation and the final API mask rather than choosing only the weaker reference.

## Resolution, padding, and budget

Non-square drawings were letterboxed to 512. A validity mask excludes padding from loss and validation counts. The adapted SmallUNet uses two downsampling levels, 16/32/64 channels, GroupNorm/SiLU, BCE+Dice, and AdamW. Ninety annotated drawings ran on an RTX 5090 for 20 epochs, batch four, seed 42; validation selected epoch 18.

Validation micro IoU at training resolution was 0.61542; native-coordinate test macro IoU was 0.59297. They differ in split, resolution, and aggregation. The legacy reference uses 1024 input and has unknown training budget/exposure. This establishes a baseline under our budget, not an equal-budget U-Net architecture comparison or replacement checkpoint.

## Better pixel overlap can still lose rooms

Selected wall macro IoU 0.84203 exceeded the legacy final mask's 0.78478, but remained below raw segmentation's 0.86334. Mean recall fell from 0.95247 to 0.88687. Full-pipeline region PQ fell from 0.68034 to 0.56580. Private case review showed missing thin internal walls connecting several annotated rooms.

That finding puts fine-wall recall, barrier closure, and region instances alongside pixel IoU in future evaluation. Skeleton F1 currently measures coverage; annotated graph accuracy and a complete manual error taxonomy remain missing.

## Export needs a numerical check

On one fixed validation tensor, default ORT CPU/CUDA maximum probability errors against PyTorch CPU were 0.002421/0.001225, above the preset 0.001 limit. Disabling CUDA TF32 reduced the error to 0.000003636. CPU differences remain unresolved. Reported test scores used actual default CUDA, with no model reselection after that diagnostic. [ORT documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#use_tf32) describes the option; one tensor does not establish a universal bound.

## Reuse the protocol

Training, evaluation, and timing commands are public. Start with truth definitions and independent plan groups, select settings only on validation, and evaluate in original coordinates. Check preprocessing, class IDs, model identity, device execution, and padding. Closing and tiling remain opt-in and need their own validation.

The frozen selection and reference weights are private, so downloading the package cannot reconstruct the exact paired score table. The reusable outputs are implementation and protocol. Future tuning needs a fresh holdout rather than repeated selection on these 30 test drawings. Real images, annotations, and research weights remain private.
