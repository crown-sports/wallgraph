# Changelog

## Unreleased

- Replace the single public figure with six measured stages from a program-generated apartment drawing: original, captured gray and CLAHE20 inputs, final wall mask, red wall overlay, and blue paths/nodes.
- Record the private model digest and actual inference settings; the third input channel, CLAHE40, is not displayed. Models, real drawings, attachments, datasets and intermediate files remain private.
- Add a source tool for rendering the stages with a user-supplied ONNX model. CI checks the reviewed figure’s hash and format; it does not repeat private-model inference.
- Keep the READMEs concise and explain text/furniture interference, model adaptation and coordinate reuse. The model-free CLI demo, runtime algorithms and existing tagged releases are unchanged.

## 0.1.2 — 2026-10-08

- Explain the model-to-geometry integration on the README first screen and link the interactive downstream region example.
- Link PlanRegions' local visual review for inspecting model changes. Segmentation, training and vectorization behavior are unchanged.

## 0.1.1 — 2026-10-08

- Explain core technical choices, paper lineage, adaptation scope, and actual reproduction lessons in English and Chinese.
- Add practical use cases and a verified private-data training/import/export guide.
- Connect wall-model review to PlanRegions' region-change diagnostics and continuity research roadmap.
- Use the current `crown-sports` repository URLs. Segmentation and vectorization behavior are unchanged.

## 0.1.0 — 2026-10-04

First public experimental release of the independently packaged wall-processing toolkit.

- Pluggable probability backends, persistent ONNX sessions, immutable configuration, and explicit CPU/CUDA selection.
- Original-image mask coordinates, optional overlapping tiles, skeleton node/path graphs, loop anchors, and thickness estimates.
- PNG/JSON/SVG export, command-line demo, evaluation, and timing tools.
- SmallUNet training and ONNX export using external private manifests, padding exclusion, and split overlap checks.
- Public validation summaries include improved final-mask IoU, the stronger legacy raw reference, downstream region regression, and numerical limitations.
- English/Chinese documentation, contribution and security policies, issue/PR templates, citation metadata, CI, and source/package downloads.

No datasets, real drawings, annotations, pretrained weights, application services, or prior repository history are included. The one generated diagram is an interface demo.
