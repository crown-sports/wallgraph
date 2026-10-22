# Wall geometry notes

[README](../README.md) · [中文](../README.zh-CN.md)

`vectorize` starts with a binary wall mask. It skeletonizes the walls, traces paths between junctions and simplifies those paths. Coordinates stay on the source grid. The default import preserves every wall pixel; cleanup is opt-in. See [mask import](../src/wallgraph/masks.py) and [graph tracing](../src/wallgraph/vectorize.py).

A closed ring needs its final segment back to the first point. Branches need shared node IDs. Those cases, plus unchanged mask imports, are covered in [geometry tests](../tests/test_geometry.py) and [mask tests](../tests/test_mask_import.py). A path is a centerline, not either face of a wall.

The README figure reads the mask and JSON produced by the [shared generated example](https://github.com/crown-sports/planportals/blob/main/docs/workflow.md). The blue lines are exported polylines; circles are exported junctions. No detector or manual geometry correction is involved. The separate [ONNX figure](../examples/simple.png) uses a private model; its [settings](demo-provenance.md) are recorded, but the weights are not distributed.

If you only need a mask operation, OpenCV or scikit-image may be enough. This package is for retaining paths, junctions and import metadata across a drawing workflow. It does not infer room types, doors or scale. For model integration, start with the [training/import guide](training-guide.md) ([中文](training-guide.zh-CN.md)).

Mask quality still matters: a missing wall can join rooms, and closing a real opening can change the drawing. The [private evaluation](validation.md) includes the lower downstream region score after cleanup; use it to understand the current failure modes.
