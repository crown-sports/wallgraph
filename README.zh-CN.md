# WallGraph

[English](README.md) · [使用场景](docs/use-cases.zh-CN.md) · [训练指南](docs/training-guide.zh-CN.md) · [核心方法](docs/methods.zh-CN.md) · [完整验证](docs/validation.md) · [发布版本](https://github.com/crown-sports/wallgraph/releases)

读懂一面墙，既要知道它占据哪些像素，也要知道它在哪里转弯、怎样与别的墙相接。WallGraph 把分割结果整理为原图坐标下的中心线、厚度和连接图，让下一步的空间分析有明确的几何依据。

它适合接入已有墙体模型、搭建图纸复核工具，以及为区域分析准备坐标一致的输入。[使用场景](docs/use-cases.zh-CN.md)从具体任务说明如何接入；[训练指南](docs/training-guide.zh-CN.md)走完导入、数据预检、训练、导出和独立评测。

**当前状态：实验工程工具库。** 提供独立实现、训练入口和评测方法，真实图纸需自备模型。现有 30 张标注测试仍有分割与下游区域退步，具体数字和区间保留在验证文档；数据和权重不公开。

## 背景与问题

空间分析、装修估算和网络规划需要墙的几何结构，而分割模型通常只输出像素。直接把像素轮廓当成墙，会出现同一面墙的双边、交叉处断连、斜墙失真，以及缩放后坐标不一致的问题。工程上，模型初始化、图像处理和业务接口混在一起，也会使替换模型与复现实验变得困难。

WallGraph 将这些职责拆开：模型返回原图坐标的概率，清理流程形成掩码，骨架追踪组织节点与路径。换模型时可以保留几何实现，改几何时也能单独核对掩码。所有输出使用左上角原点、x 向右、y 向下、单位为像素；不猜测真实比例尺。

## 快速开始

从 GitHub 安装已发布的版本：

```bash
python -m pip install "git+https://github.com/crown-sports/wallgraph.git@v0.1.1"
wallgraph demo --output runs/demo
```

也可下载 [Release 中的 wheel](https://github.com/crown-sports/wallgraph/releases/tag/v0.1.1) 并校验 SHA256。以下开发安装命令在克隆本仓库后执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,onnx]'
wallgraph demo --output runs/demo
wallgraph detect --image examples/simple.png --output runs/simple
```

公开素材只有 `examples/simple.png` 一张原创简单图。`demo` 使用相同的图形生成函数；它是接口演示，不是训练集，也不是准确率基准。没有提供真实数据、标注、训练权重或旧工程的模型。

结果包括 `walls.png`（255 为墙、0 为背景）、`walls.json`（节点、折线、厚度、置信度与分阶段耗时）和 `walls.svg`。每条折线通过 `start_node` / `end_node` 引用节点，闭合墙环允许首尾引用同一节点。

无模型时采用 `ink-baseline`：把深色像素识别为墨迹。它会把文字和家具也当成墙，适合简单 demo 和测试。真实图纸应使用训练过的分割模型。

```bash
wallgraph detect --image /private/plan.png --model /private/wall.onnx \
  --preprocess rgb --wall-classes 1 --device cpu --output runs/prediction
# 大图：在概率层做融合，再进行阈值与矢量化
wallgraph detect --image /private/large.png --model /private/wall.onnx \
  --tile-size 1024 --overlap 128 --output runs/large
```

模型接口为 float32 NCHW RGB 输入（0–1），单通道二分类 logits 或多通道 logits 输出。动态空间尺寸使用 `--input-size H W`，固定尺寸自动读取。多类模型必须明确指定 `--wall-classes`，模型预处理可选 `rgb`、`gray`、`clahe`；输出为概率时使用 `--output-kind probabilities`。不同模型的类别和预处理不能靠文件名猜测。

```python
from wallgraph import WallPipeline, WallConfig
from wallgraph.backends import InkBackend
from wallgraph.io import draw_demo

pipeline = WallPipeline(InkBackend(), WallConfig(close_radius=0))
result = pipeline.run(draw_demo())
geometry = result.to_dict()
```

## 技术难点与工程贡献

| 难点 | 处理方式 | 当前边界 |
| --- | --- | --- |
| 非正方形图与大图 | 保持长宽比的 letterbox；切片概率加权融合；恢复原图尺寸 | 局部切片缺少全局上下文，需真实数据对比 |
| 交叉处多余边与断连 | 骨架邻接去除对角捷径；聚合同一交叉处的关键像素；追踪节点之间的路径 | 分割错漏仍会传到拓扑，微小分叉可能保留 |
| 闭环与斜墙 | 为全二度闭环创建锚点；对路径做容差简化，保留斜向几何 | 几何精度受简化容差和像素分辨率影响 |
| 厚度不稳定 | 沿骨架用距离变换估计宽度，取路径样本中位数 | 是像素宽度估计，斜墙和连接处有量化误差 |
| 验证结果无法复核 | 保存参数、运行时、manifest 与模型摘要；检查跨 split 的组和图像重复 | 验证集用于选模型；需要独立测试集才能报告最终精度 |

训练基线借鉴 [U-Net](https://arxiv.org/abs/1505.04597) 与 [Group Normalization](https://arxiv.org/abs/1803.08494)，二维骨架通过库调用 Zhang–Suen 方法。论文对应哪段代码、哪些地方做了改编，见 [核心方法与论文](docs/methods.zh-CN.md)。本工程重点在节点与闭环追踪、坐标一致性和可替换后端，基础算法沿用成熟方法。

[复现实跑笔记](docs/reproduction.zh-CN.md) 记录了标注怎样找到、padding 怎样排除、细墙漏检怎样影响房间，以及为什么导出成功之后仍要检查数值。训练和配对结果来自真实执行，未重做原论文实验。完整结果见 [实测记录](docs/results.md)。

改模型或阈值时，还可以接入 [PlanRegions 变化诊断](https://github.com/crown-sports/planregions/blob/main/docs/comparison.zh-CN.md)：找出哪些区域合并、分裂或消失，把墙体改动追溯到具体空间。[后续研究路线](https://github.com/crown-sports/planregions/blob/main/docs/research-roadmap.zh-CN.md)说明连续性、开口语义和新测试集怎样验证。

## 私有数据训练与评估

安装 `.[train]`。在公开仓库外准备 JSONL manifest，每行包含 `image`、`mask`、`group` 三个字符串。图像/掩码路径相对于 manifest 所在目录，也可以是绝对路径；掩码 0 表示背景、1 或 255 表示墙。`group` 是同一原始平面图的分组标识，同图的裁剪/增强必须处于同一组。

```bash
wallgraph train --train-manifest /private/train.jsonl --val-manifest /private/val.jsonl \
  --device cuda --size 512 --epochs 20 --output /private/runs/wallgraph
wallgraph export-onnx --checkpoint /private/runs/wallgraph/best.pt \
  --size 512 --output /private/runs/wallgraph/model.onnx
wallgraph evaluate --manifest /private/test.jsonl \
  --model /private/runs/wallgraph/model.onnx --output /private/reports/walls.json
wallgraph benchmark --images /private/plan.png --model /private/wall.onnx \
  --device cuda --warmup 1 --repeats 3 --output /private/reports/latency.json
```

新训练入口使用小型 U-Net、BCE+Dice、AdamW 与固定 seed。训练时排除 letterbox padding；按验证集 IoU 选取权重。评估在原图坐标报告 pixel IoU/F1 和容差为 2 像素的 skeleton F1；骨架指标评估覆盖情况，不能替代图连通性指标。评估报告不包含单张文件名和像素，但汇总指标仍需发布者审阅。GPU 安装和严格设备检查见 [运行说明](docs/runtime.md)。

## 工程规范与发布

```bash
ruff check .
ruff format --check .
pytest
python tools/release.py --check
python -m build
python tools/release.py --output dist/wallgraph-source.zip
```

发布工具按白名单打包，拒绝额外图像、凭据、内网地址与非预期文件。它不会递归打包工作目录。`data/`、`weights/`、`runs/` 和模型后缀均被忽略。自动扫描只能减少误打包风险，发布前仍需审阅具体归档。新实现使用 MIT 许可；外部权重和数据各自的许可不随之改变，详见 [来源说明](NOTICE.md)。

协作规范见 [CONTRIBUTING](CONTRIBUTING.md)、[行为准则](CODE_OF_CONDUCT.md)、[安全报告](SECURITY.md) 和 [发布流程](docs/releasing.md)。`CITATION.cff` 提供软件引用信息。
