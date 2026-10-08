# WallGraph

把墙体分割掩码变成**原图坐标下的中心线、交点和闭环**。

[English](README.md) · [发布版本](https://github.com/crown-sports/wallgraph/releases)

![左侧为生成的简单图纸，右侧为 WallGraph 实际输出的中心线与图节点](examples/simple.png)

左侧是生成的输入图，右侧按 demo 的 `walls.json` 绘制：**6 条墙体路径、4 个连接节点**。颜色区分路径，圆点旁是节点编号。这张图展示几何提取，不代表识别精度。

## 安装与运行

需要 Python 3.10 或更新版本。无需模型即可运行 demo：

```bash
python -m pip install "git+https://github.com/crown-sports/wallgraph.git@v0.1.2"
wallgraph demo --output runs/demo
```

打开 `runs/demo/walls.svg` 查看结果。

| 输出 | 能拿来做什么 |
| --- | --- |
| `walls.svg` | 查看提取的墙体路径 |
| `walls.json` | 绘制墙线、按端点查连接关系、读取像素厚度和置信度 |
| `walls.png` | 保留二值掩码，或传入区域分析 |

坐标对应原图：左上角为原点，x 向右、y 向下，单位为像素。闭合墙环保留锚点节点。工具不推算真实比例尺。

## 接入自己的墙体模型

```bash
python -m pip install "wallgraph[onnx] @ git+https://github.com/crown-sports/wallgraph.git@v0.1.2"
wallgraph detect --image /private/plan.png --model /private/wall.onnx \
  --preprocess rgb --wall-classes 1 --device cpu --output runs/prediction
```

ONNX 接口接受 float32 NCHW RGB 输入，范围为 0–1，输出为二分类或多分类分割 logits。预处理和墙体类别需与自己的模型一致；概率输出加 `--output-kind probabilities`，动态空间尺寸加 `--input-size H W`。

**真实识别需自备兼容模型。** 仓库不提供权重或真实数据集。demo 的墨迹检测也会把文字和家具当成墙，适合简单示例，不适合真实墙体识别。

## 具体解决哪些问题

| 你手上有什么 | 接下来要做什么 | WallGraph 提供什么 |
| --- | --- | --- |
| 由粗像素组成的墙体掩码 | 在图纸界面叠加或选中一条墙线 | 中心线折线与端点节点 ID |
| 按模型尺寸缩放过的预测 | 将几何结果对齐原始图纸 | 恢复到原图坐标的掩码与路径 |
| 不同的分割模型 | 复用同一套几何处理接口 | 可替换的 ONNX 或 Python 推理后端 |
| 用于房间提取的墙体预测 | 给区域算法传入对齐的边界 | [PlanRegions](https://github.com/crown-sports/planregions) 可读取的掩码与元信息 |

这是几何工具库。复核编辑器、CAD/BIM 导出、门的语义和尺寸识别仍需应用层实现。接入方式见[使用场景](docs/use-cases.zh-CN.md)。

## 实测与限制

使用同一个私有模型，在 30 张标注图纸上，验证集选定配置的墙体 macro IoU 为 **0.84203**，旧工程最终掩码为 **0.78478**，原始分割为 **0.86334**。墙体召回下降，下游区域 PQ 也从 **0.68034 降至 0.56580**。仓库训练基线的 IoU 为 **0.59297**，一次导出检查超出了选定的数值误差阈值。结果支持继续改进，不能作为生产精度保证。[完整实验、区间和耗时](docs/validation.md)。

像素分数高，房间边界也可能出问题。[生成的交互示例](https://crown-sports.github.io/planregions/)展示了少一个墙像素时，两个区域怎样合并。

## 文档

[训练与数据导入](docs/training-guide.zh-CN.md) · [方法与论文](docs/methods.zh-CN.md) · [实现笔记](docs/reproduction.zh-CN.md) · [架构](docs/architecture.zh-CN.md) · [GPU 运行](docs/runtime.md)

[贡献指南](CONTRIBUTING.md) · [安全报告](SECURITY.md) · [行为准则](CODE_OF_CONDUCT.md) · [发布检查](docs/releasing.md)

仓库代码使用 MIT 许可；外部模型和数据保留各自许可。见[来源说明](NOTICE.md)。软件引用信息见 `CITATION.cff`。
