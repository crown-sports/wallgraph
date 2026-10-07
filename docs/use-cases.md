# What WallGraph helps you build

[中文](use-cases.zh-CN.md) · [Quick start](../README.md#install-and-run) · [Methods](methods.md) · [Measured results](validation.md)

WallGraph is useful when you already have wall predictions and need to turn them into geometry that another application can inspect. It gives the model, mask, and connection graph clear boundaries, so improving one stage does not require rebuilding the whole application.

## Choose it for a concrete task

| Your task | Input | Useful output | What you still provide |
| --- | --- | --- | --- |
| Add wall structure to a drawing review tool | An RGB plan and your compatible ONNX wall model | Original-coordinate polylines, junction IDs, mask, and SVG preview | The review UI and checks against your drawings |
| Compare segmentation or cleanup choices | Independently labeled plans and configured pipelines | Pixel IoU/F1, skeleton coverage, stage timings, and saved geometry | A held-out split and separate connectivity checks |
| Connect wall inference to region analysis | A wall result | A binary mask and metadata accepted by PlanRegions | Evidence for open boundaries or missing separators |
| Reuse a model outside ONNX | A Python backend that returns an aligned probability map | The same cleanup and graph pipeline | Your inference adapter and its preprocessing contract |

For a review interface, `walls.json` lets you draw a path and follow its endpoint references to junctions. `walls.png` retains the pixel evidence beside that geometry. This makes a disconnected wall or an unexpected loop inspectable; the library does not decide whether the drawing is correct.

## Try the full handoff

After installing both projects using their READMEs, run:

```bash
wallgraph demo --output runs/walls
planregions detect --walls runs/walls/walls.png \
  --wall-metadata runs/walls/walls.json --output runs/regions
```

The single generated layout produces six wall paths and three regions. WallGraph writes `walls.png`, `walls.json`, and `walls.svg`. [PlanRegions](https://github.com/crown-sports/planregions) adds an exact instance map and region polygons. The metadata check verifies dimensions, coordinates, and wall polarity before the handoff. This verifies an integration path; a simple generated drawing cannot establish recognition accuracy.

For real plans, follow the [ONNX input contract](../README.md#install-and-run). A compatible shape alone is insufficient: color order, normalization, class IDs, and output semantics must match the model's training.

## Where its value ends today

The included ink detector also finds text and furniture. Real wall recognition requires your own licensed model. There is no built-in editor, DXF/BIM exporter, physical scale inference, or door-accessibility graph. Coordinates and thickness are in pixels; a wall connection is not a walking route.

An attractive SVG is not evidence of correct topology. Published tests found lower wall recall and worse downstream region results in the new full pipeline, despite improvement over one processed-mask reference. Use the [validation report](validation.md) to design your own acceptance test. If your task only needs a binary mask, graph extraction may add little value.

## A 30-second introduction

> WallGraph helps developers turn floor-plan wall predictions into inspectable structure: a mask, connected polylines, junctions, and closed loops in the original image coordinates. You bring the model; WallGraph provides replaceable inference and geometry stages, a CLI, and evaluation tools. Its output connects directly to PlanRegions for region analysis. The project is experimental, with measured limitations published alongside the code. It is a starting point for building and checking a drawing workflow.

For the technical background and actual implementation lessons, read [methods and papers](methods.md) and [reproduction notes](reproduction.md).
