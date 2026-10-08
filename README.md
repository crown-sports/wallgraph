# WallGraph

Turn a wall segmentation mask into **centerlines, junctions and closed loops in the original image coordinates**.

[中文](README.zh-CN.md) · [Releases](https://github.com/crown-sports/wallgraph/releases)

![Generated apartment plan, actual gray and CLAHE model inputs, predicted mask, red wall overlay, and blue paths with nodes](examples/simple.png)

One generated apartment plan, one private ONNX model run: original → actual gray channel → actual CLAHE20 channel → final wall mask → red wall overlay → blue centerlines and nodes. **25 paths, 21 nodes**, with no manual edits. The model’s third channel, CLAHE40, is not shown. This run closes some door/window openings and leaves a short living-room divider disconnected. [Source and settings](docs/demo-provenance.md).

## Install and run

Python 3.10+. Try the separate, simple demo without a model:

```bash
python -m pip install "git+https://github.com/crown-sports/wallgraph.git@v0.1.2"
wallgraph demo --output runs/demo
```

Open `runs/demo/walls.svg` to see the result.

| Output | What you can use it for |
| --- | --- |
| `walls.svg` | View the extracted wall paths |
| `walls.json` | Draw wall paths, follow endpoint connections, read pixel thickness and confidence |
| `walls.png` | Keep the binary mask or pass it to region analysis |

The graph uses the original image size: origin at the top left, x right, y down, units in pixels. Closed loops retain an anchor node. There is no physical scale inference.

## Use your wall model

```bash
python -m pip install "wallgraph[onnx] @ git+https://github.com/crown-sports/wallgraph.git@v0.1.2"
wallgraph detect --image /private/plan.png --model /private/wall.onnx \
  --preprocess rgb --wall-classes 1 --device cpu --output runs/prediction
```

The ONNX adapter accepts float32 NCHW three-channel input in 0–1 and binary or multiclass segmentation logits. Match preprocessing and wall class IDs to your model. For probability outputs, add `--output-kind probabilities`; dynamic spatial inputs need `--input-size H W`.

From a main source checkout, render the six stages above with your own compatible model:

```bash
python tools/render_demo.py --model /private/wall.onnx --preprocess clahe \
  --wall-classes 1 2 --output /private/steps.png
```

**Bring your own compatible model.** No weights or real datasets are included. The CLI’s simple demo uses ink thresholding, which also finds text and furniture; the six-stage figure uses an actual ONNX segmentation model. Neither example measures recognition accuracy.

## Problems it helps solve

| You have | You need | WallGraph provides |
| --- | --- | --- |
| A wall mask made of thick pixels | A path to overlay or select in your drawing UI | Centerline polylines with endpoint node IDs |
| Predictions resized for model inference | Geometry aligned with the source drawing | Masks and paths restored to original coordinates |
| Different segmentation models | A shared geometry interface | Replaceable ONNX or Python inference backend |
| Wall predictions for room extraction | An aligned boundary input | A mask and metadata accepted by [PlanRegions](https://github.com/crown-sports/planregions) |

Furniture, labels and missing thin walls can confuse a model. Keeping the mask beside the geometry lets you inspect these errors before reusing the coordinates. A review editor, CAD/BIM export, door semantics and dimension recognition remain application work. See [integration examples](docs/use-cases.md).

## Measured limits

On 30 annotated drawings using the same private model, validation-selected WallGraph reached wall macro IoU **0.84203**, versus **0.78478** for the legacy final mask and **0.86334** for its raw segmentation. Lower wall recall also reduced downstream region PQ from **0.68034 to 0.56580**. The included training baseline reached IoU **0.59297**; one export check exceeded the selected numerical error threshold. These results support further development, not a production accuracy claim. [Full protocol, intervals and timing](docs/validation.md).

A good pixel score can still hide a broken room boundary. [The generated interactive example](https://crown-sports.github.io/planregions/) shows how one missing wall pixel can merge two regions.

## Documentation

[Training and data import](docs/training-guide.md) · [Methods and papers](docs/methods.md) · [Implementation notes](docs/reproduction.md) · [Architecture](docs/architecture.md) · [GPU runtime](docs/runtime.md)

[Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [Code of conduct](CODE_OF_CONDUCT.md) · [Release checks](docs/releasing.md)

MIT for repository code; external models and data retain their own licenses. [Provenance](NOTICE.md). Software citation: `CITATION.cff`.
