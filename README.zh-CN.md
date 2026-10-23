# WallGraph

把墙体掩码转成原图坐标下的中心线和交点，方便叠加、选中和测量墙线。

[English](README.md) · [v0.2.1](https://github.com/crown-sports/wallgraph/releases/tag/v0.2.1)

![左侧为二值墙体掩码，右侧为实际提取的蓝色墙线和交点](examples/workflow.png)

生成示例：得到 **11 条路径、7 个交点**，61,849 个输入墙体像素全部保留。[运行这张五房间示例](https://github.com/crown-sports/planportals/blob/main/docs/workflow.zh-CN.md)。

## 先跑起来

Python 3.10+。下面的小型独立 demo 不需要模型：

```bash
python -m pip install "git+https://github.com/crown-sports/wallgraph.git@v0.2.1"
wallgraph demo --output runs/demo
```

原始输入与结果请分开放；重复使用同一个输出目录，会替换其中生成的结果文件。

打开 `runs/demo/walls.svg` 看结果。`walls.json` 保存墙线、端点节点 ID、像素厚度和分数，`walls.png` 保存墙体掩码。

已有墙体掩码时直接导入：

```bash
wallgraph vectorize --mask /private/walls.png --output runs/vectorized
```

输入为只含 0 和 255 的二维 uint8 PNG，白色表示墙；黑墙加 `--wall-value 0`。RGB 图和概率图会被拒绝。默认不改像素，只有显式设置 `--close-radius` 或 `--min-component-area` 才做清理。

## 接到自己的工具里

坐标原点在左上角，x 向右、y 向下，单位为像素。可按路径绘制和选中墙线，通过端点 ID 查连接，或将掩码和元数据交给 [PlanRegions](https://github.com/crown-sports/planregions) 划分区域。有尺寸依据后，可用 [PlanMetric](https://github.com/crown-sports/planmetric) 换算长度。

导入掩码的分数表示像素是否属于输入墙体，不是检测器置信度。这里不推算真实比例尺、门的类别或 CAD/BIM 模型。

## 只有扫描图时

需要自备兼容的分割模型，仓库不提供公开权重。[训练与数据导入](docs/training-guide.zh-CN.md)介绍了基线和文件格式。

```bash
python -m pip install "wallgraph[onnx] @ git+https://github.com/crown-sports/wallgraph.git@v0.2.1"
wallgraph detect --image /private/plan.png --model /private/wall.onnx \
  --preprocess rgb --wall-classes 1 --device cpu --output runs/prediction
```

预处理和类别 ID 要与模型一致。接口接受 0–1 的 float32 NCHW 三通道输入及分割 logits；概率输出加 `--output-kind probabilities`，动态输入加 `--input-size H W`。

[查看一次真实 ONNX 推理的六步图](examples/simple.png) · [输入与参数](docs/demo-provenance.md)。这个私有模型会把部分开口封住，也漏了一段隔墙；独立 demo 的墨迹阈值同样可能把家具和文字当成墙。

私有 30 张图纸评估中，清理后的墙体 IoU 低于原始分割，下游区域 PQ 也下降了。首页生成图不代表真实识别精度。[实验结果与限制](docs/validation.md)。

## 继续看

[实现说明](docs/product.md) · [接入场景](docs/use-cases.zh-CN.md) · [方法](docs/methods.zh-CN.md) · [架构](docs/architecture.zh-CN.md) · [复现记录](docs/reproduction.zh-CN.md)

[贡献](CONTRIBUTING.md) · [发布检查](docs/releasing.md) · [安全报告](SECURITY.md)

代码与原创生成图使用 MIT 许可；外部模型和数据保留各自许可。[来源说明](NOTICE.md)。
