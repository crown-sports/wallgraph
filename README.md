# WallGraph

Turn a wall segmentation mask into **centerlines, junctions and closed loops in the original image coordinates**.

[中文](README.zh-CN.md) · [Releases](https://github.com/crown-sports/wallgraph/releases)

![Generated plan on the left; centerlines and graph nodes from the WallGraph demo on the right](examples/simple.png)

Left: the generated input. Right: six wall paths and four connection nodes, drawn from the demo’s `walls.json`. Colors distinguish paths; numbered dots are nodes. This shows geometry extraction, not recognition accuracy.

## Install and run

Python 3.10+. Run the demo without a model:

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

The ONNX adapter accepts float32 NCHW RGB input in 0–1 and binary or multiclass segmentation logits. Match preprocessing and wall class IDs to your model. For probability outputs, add `--output-kind probabilities`; dynamic spatial inputs need `--input-size H W`.

**Bring your own compatible model.** No weights or real datasets are included. The demo's ink detector also finds text and furniture; it is suitable for the simple example, not real wall recognition.

## Problems it helps solve

| You have | You need | WallGraph provides |
| --- | --- | --- |
| A wall mask made of thick pixels | A path to overlay or select in your drawing UI | Centerline polylines with endpoint node IDs |
| Predictions resized for model inference | Geometry aligned with the source drawing | Masks and paths restored to original coordinates |
| Different segmentation models | A shared geometry interface | Replaceable ONNX or Python inference backend |
| Wall predictions for room extraction | An aligned boundary input | A mask and metadata accepted by [PlanRegions](https://github.com/crown-sports/planregions) |

This is a geometry toolkit. A review editor, CAD/BIM export, door semantics and dimension recognition remain application work. See [integration examples](docs/use-cases.md).

## Measured limits

On 30 annotated drawings using the same private model, validation-selected WallGraph reached wall macro IoU **0.84203**, versus **0.78478** for the legacy final mask and **0.86334** for its raw segmentation. Lower wall recall also reduced downstream region PQ from **0.68034 to 0.56580**. The included training baseline reached IoU **0.59297**; one export check exceeded the selected numerical error threshold. These results support further development, not a production accuracy claim. [Full protocol, intervals and timing](docs/validation.md).

A good pixel score can still hide a broken room boundary. [The generated interactive example](https://crown-sports.github.io/planregions/) shows how one missing wall pixel can merge two regions.

## Documentation

[Training and data import](docs/training-guide.md) · [Methods and papers](docs/methods.md) · [Implementation notes](docs/reproduction.md) · [Architecture](docs/architecture.md) · [GPU runtime](docs/runtime.md)

[Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [Code of conduct](CODE_OF_CONDUCT.md) · [Release checks](docs/releasing.md)

MIT for repository code; external models and data retain their own licenses. [Provenance](NOTICE.md). Software citation: `CITATION.cff`.
