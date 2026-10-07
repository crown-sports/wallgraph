# 用自己的图纸训练墙体模型

[English](training-guide.md) · [方法与论文](methods.zh-CN.md) · [实测结果](validation.md) · [GPU 运行说明](runtime.md)

这份手册从本地导入开始，走完二值墙体训练、ONNX 导出和独立测试。命令对应当前 SmallUNet 实现。你需要准备有权使用的图纸和标注；仓库提供代码和一张生成的演示图，不提供训练集或预训练权重。

## 1. 先确认导入的是哪份代码

需要 Python 3.10 或更新版本。克隆后，在仓库根目录安装。下面使用 CPU 环境：`train` 安装 PyTorch 和 ONNX，`onnx` 另行安装推理所需的 CPU ONNX Runtime。

```bash
git clone https://github.com/crown-sports/wallgraph.git
cd wallgraph
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train,onnx]'
export WG_DEVICE=cpu
python -c 'import wallgraph; print(wallgraph.__version__, wallgraph.__file__)'
python -m wallgraph.cli train --help
```

可编辑安装会让当前 Python 找到 `src/wallgraph`。如果终端里的 `wallgraph` 命令来自另一个环境，就用 `python -m wallgraph.cli`。自己的脚本也不要命名为 `wallgraph.py` 或 `torch.py`，以免遮蔽包名。顶层 `import wallgraph` 不会加载 PyTorch 或创建 ONNX 会话；导入 `wallgraph.training` 才需要训练依赖，构造 `OnnxBackend` 才需要 ONNX Runtime。

Linux NVIDIA 用户可选择一个全新环境，改装 `'.[train,gpu]'`，并设置 `export WG_DEVICE=cuda`。同一环境不要同时安装 CPU 和 GPU 两种 ONNX Runtime wheel。训练前分别确认：

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python -c 'import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())'
```

PyTorch 能用 GPU，不代表 ONNX CUDA 也已经可用。驱动与框架组合见[运行说明](runtime.md)。当前 CLI 的设备选项只有 `cpu` 和 `cuda`，没有 Apple MPS。

不依赖数据，也可以先检查模型导入和张量形状：

```python
import torch
from wallgraph.training import SmallUNet

model = SmallUNet().eval()
with torch.inference_mode():
    logits = model(torch.zeros(1, 3, 32, 64))
print(tuple(logits.shape))  # (1, 1, 32, 64)
```

这里使用随机初始化权重，只验证输入输出能否接通，不代表识别效果。

## 2. 先说清楚要识别什么

数据放在仓库之外。把下面的变量改成自己的绝对路径；后续命令都会使用它。

```bash
export WG_DATA="/absolute/path/to/private-data"
```

目录可以这样组织：

```text
private-data/
  images/plan-a.png
  masks/plan-a.png
  train.jsonl
  val.jsonl
  test.jsonl
```

manifest 是 UTF-8 编码的 JSONL，一行一个对象，不是 JSON 数组：

```json
{"image":"images/plan-a.png","mask":"masks/plan-a.png","group":"plan-a"}
```

`image`、`mask`、`group` 都必须是非空字符串。相对路径从 manifest 所在目录解析，与执行命令的目录无关。训练、验证、测试文件都至少需要一条记录。

图片会解码为 RGB。标签应是与图片宽高一致的单通道无损掩码：`0` 是背景，`1` 或 `255` 是墙体，混用 `1` 和 `255` 也都按前景处理。训练加载器拒绝其他数值。不要直接使用 JPEG 标签、抗锯齿线条、彩色覆盖图或未经转换的多类标签；先按语义明确选出墙体类别，再保存二值掩码。评测程序会把所有非零值视为前景，不会替你完成类别映射。

标注前先统一墙体口径：门窗结构是否计入，栏杆、家具如何处理。三个划分必须使用同一套定义。结构墙标签包含门洞，并不代表它能直接充当房间提取所需的封闭障碍。

先按原始图纸划分，再做裁剪或增强。同一张图纸的所有派生样本必须共用 `group`；同一建筑或同一套图纸的关联页面，可能还需要更大的分组。每个组只能进入一个划分。有官方划分时应保留其边界，并私下记录抽样种子和标注版本。训练命令自动检查训练与验证的组重叠、图像字节完全重复；它不会自动检查测试集、视觉近似重复或数据来历。

## 3. 上 GPU 前检查三份数据

下面直接复用实际加载器，逐张检查图像与标签，并对三个划分两两执行现有的重叠检查。它只读取数据，不会上传。

```bash
python - <<'PY'
import os
from itertools import combinations
from pathlib import Path
from wallgraph.training import ManifestDataset, validate_split

root = Path(os.environ["WG_DATA"])
datasets = {
    name: ManifestDataset(root / f"{name}.jsonl", size=512)
    for name in ("train", "val", "test")
}
for first, second in combinations(datasets, 2):
    validate_split(datasets[first], datasets[second])
for name, dataset in datasets.items():
    for index in range(len(dataset)):
        image, truth, valid = dataset[index]
        assert image.shape == (3, 512, 512)
        assert truth.shape == valid.shape == (1, 512, 512)
    print(name, "records:", len(dataset))
PY
```

还应在本地看几张图像与标注的叠加结果，特别是细墙和空标签。尺寸、数值正确，仍可能存在标注偏移或定义错误。重新编码的相同图片会产生不同文件哈希，所以分组管理不可省略。

## 4. 先跑通一轮，再训练基线

每次运行用新的输出目录，命令可能覆盖其中的 `best.pt` 和 `training.json`。先跑一轮，尽早发现数据或设备问题：

```bash
python -m wallgraph.cli train \
  --train-manifest "$WG_DATA/train.jsonl" \
  --val-manifest "$WG_DATA/val.jsonl" \
  --device "$WG_DEVICE" --size 128 --epochs 1 --batch-size 2 --seed 42 \
  --output "$WG_DATA/runs/smoke"
```

CUDA 确定性训练需要在启动 Python 前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；CUDA 的这次试跑也应加上同样的前缀。训练器启用了确定性算法，遇到不支持的操作会报错。即使种子相同，也不能保证跨设备、跨框架版本逐位一致，原因见 [PyTorch 复现说明](https://docs.pytorch.org/docs/2.8/notes/randomness.html)。

确认试跑成功后，执行基线配置：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m wallgraph.cli train \
  --train-manifest "$WG_DATA/train.jsonl" \
  --val-manifest "$WG_DATA/val.jsonl" \
  --device "$WG_DEVICE" --size 512 --epochs 20 --batch-size 4 --seed 42 \
  --output "$WG_DATA/runs/baseline"
```

`size` 至少为 16，且必须是 4 的倍数；epoch 和 batch size 必须为正。CPU 适合验证流程，这里没有承诺它的完整训练耗时。

当前训练配方固定为：RGB 除以 255，两级编码器、16/32/64 通道、GroupNorm、SiLU、BCE 加 soft Dice，AdamW 学习率 `0.001`、weight decay `0.0001`。使用 FP32，没有数据增强、学习率调度、混合精度、梯度累积或分布式训练，也没有对应的隐藏命令参数。`--seed` 控制初始化和训练打乱，不负责生成数据划分。

每轮输出 `epoch`、`train_loss` 和 `val_iou`。验证在阈值 `0.5` 下累加所有有效像素的 TP/FP/FN，得到训练分辨率的 micro IoU。只有分数严格提高时才保存 `best.pt`，同分保留更早的权重。全部 epoch 完成后，`training.json` 记录历史、数量、manifest/权重哈希、运行环境和选择规则。它没有记录所有命令参数，因此还要保存完整命令、代码 commit 和 `python -m pip freeze` 输出。当前没有恢复训练命令，也没有保存优化器状态。

## 5. 理解非方形图纸怎样进入模型

图片按比例缩放到 `size × size` 画布，居中放置，用 RGB 128 补边，不做拉伸。图片用双三次插值，标签用最近邻插值。加载器返回 `(image, truth, valid)`；`valid` 把补边从损失和验证计数中排除。它不是通用的忽略标签机制：原始标签中的 `255` 仍然表示墙体。

按训练时的尺寸导出所选权重：

```bash
python -m wallgraph.cli export-onnx \
  --checkpoint "$WG_DATA/runs/baseline/best.pt" --size 512 \
  --output "$WG_DATA/runs/baseline/model.onnx"
```

| 导出约定 | 内容 |
| --- | --- |
| 输入名称与形状 | `images`，float32 `[1, 3, 512, 512]`，RGB，范围 `[0, 1]` |
| 输出名称与形状 | `wall_logits`，`[1, 1, 512, 512]`，未经 sigmoid 的 logits |
| 图格式 | ONNX opset 17，固定 batch 和空间尺寸 |
| 后处理 | sigmoid，移除补边，把概率图还原到原图坐标，然后二值化 |

`export-onnx` 在 CPU 上构建 SmallUNet 并加载它的 state dictionary，不是任意网络权重的通用转换器。`--size` 指定固定导出尺寸，不会产生动态轴。这里的二值模型应使用 `--preprocess rgb --wall-classes 1 --output-kind logits`；`1` 表示前景约定，不是第二个输出通道。改用 CLAHE 或灰度推理，会改变这套训练配方的输入分布。固定形状模型的尺寸优先于推理命令里的 `--input-size`。

调试 PyTorch 时，可以直接导入自己保存的模型：

```python
import os
from pathlib import Path
import torch
from wallgraph.training import ManifestDataset, SmallUNet

root = Path(os.environ["WG_DATA"])
model = SmallUNet().eval()
model.load_state_dict(
    torch.load(root / "runs/baseline/best.pt", map_location="cpu", weights_only=True)
)
image, truth, valid = ManifestDataset(root / "val.jsonl", 512)[0]
with torch.inference_mode():
    probability = model(image[None]).sigmoid()[0, 0]
print(tuple(probability.shape), float(probability.min()), float(probability.max()))
```

这里得到的仍是带补边的训练坐标。需要原图坐标结果时，使用下面的推理管线。

## 6. 在独立测试集评测导出的模型

```bash
python -m wallgraph.cli evaluate \
  --manifest "$WG_DATA/test.jsonl" \
  --model "$WG_DATA/runs/baseline/model.onnx" \
  --device "$WG_DEVICE" --preprocess rgb --wall-classes 1 \
  --output-kind logits --threshold 0.5 --close-radius 0 \
  --output "$WG_DATA/runs/baseline/test.json"
```

务必提供 `--model`；省略时评测的是深色墨迹演示后端。报告包含 micro IoU、micro F1、平均骨架覆盖 F1、配置、模型/manifest 哈希和耗时，目前不含像素 macro IoU，也不含图连接准确率。预测与原分辨率标签比较。这里的耗时不含文件读写和会话创建，没有预热；需要受控的预热、重复计时时使用 `benchmark`。

查看自己的单张结果时，把 `plan-a.png` 换成实际图片：

```bash
python -m wallgraph.cli detect \
  --image "$WG_DATA/images/plan-a.png" \
  --model "$WG_DATA/runs/baseline/model.onnx" \
  --device "$WG_DEVICE" --preprocess rgb --wall-classes 1 \
  --output-kind logits --threshold 0.5 --close-radius 0 \
  --output "$WG_DATA/runs/baseline/inspection"
```

导出之后还应在相同预处理的验证张量上，对比 PyTorch 与 ONNX 的 sigmoid 概率，记录最大误差、平均误差和二值像素一致率，并事先定下容差。成功生成文件不等于数值已经对齐。已有实验中，默认 CPU/CUDA 在一张固定张量上均超过 `0.001` 最大误差门槛；关闭 CUDA TF32 改善了该次诊断，CPU 差异仍未解决。CLI 没有 TF32 开关。具体范围见[导出实测](validation.md#new-training-baseline-and-export)。

## 7. 第一次训练卡住时查什么

| 现象 | 排查方式 |
| --- | --- |
| `No module named wallgraph` | 激活目标环境，从仓库执行该解释器的 `python -m pip install -e ...`，打印 `wallgraph.__file__`。 |
| 缺少 `torch` 或 `onnxruntime` | 在同一个解释器中安装对应 extra；`train` 不包含推理 runtime。 |
| manifest 为空或字段缺失 | 检查非空 UTF-8 JSONL，以及 `image`、`mask`、`group` 字符串。 |
| 文件找不到、标注尺寸不一致 | 按 manifest 所在目录解析路径，核对解码后的图像和标签尺寸。 |
| 标签不是二值掩码 | 查看实际存储值，显式转换语义类别，移除插值生成的灰度值。 |
| 分组或相同字节重叠 | 按原始图纸重新划分；只改 group 名绕过报错，数据泄漏仍在。 |
| CUDA 不可用或内核不支持 | 检查 PyTorch wheel 与 GPU 架构；ONNX 还需要自己的 CUDA provider。也可明确选择 CPU。 |
| cuBLAS 确定性报错 | 在 Python 启动前设置 workspace 变量，保留确定性设置，再检查其他不受支持的操作。 |
| 显存不足 | 优先减小 batch；降低分辨率也会改变实验，可能丢失细墙，必须记录。 |
| loss 降了，墙体仍不好 | 先看空标签、错位、标注定义、前景比例及训练/验证分布差异，再考虑增加 epoch。 |
| 像素指标不错，房间却合并 | 检查细墙召回和开口；闭运算也可能封掉真实缝隙，需在验证集选择并评测下游区域。 |

## 8. 下一步实验怎样做才有说服力

已有基线使用 90 张训练图、15 张验证图，在 RTX 5090 上训练 20 个 epoch，batch 4、RGB 512、seed 42。验证选择 epoch 18，训练分辨率 micro IoU 为 `0.61542`。30 张保留测试图在原图坐标的 macro IoU 为 `0.59297`、micro IoU 为 `0.56866`、micro F1 为 `0.72502`。这些分数的划分和聚合口径不同，验证分数不是测试分数。私有 manifest 和研究权重没有发布，因此公开仓库不能重建原始数值表。既有模型的训练预算与接触过的数据未知，这次训练也不构成同预算胜出的证据。

下面是待做的对比，不是已经证实有效的调参结论：

| 想回答的问题 | 对照方式 | 必须补充的控制 |
| --- | --- | --- |
| 随机种子影响多大？ | 同一配方重复几个事先选定的种子 | 固定分组、训练预算和选择规则，报告分布 |
| 缩放是否丢掉细墙？ | 比较 512 与 768 输入 | 分别按训练尺寸导出，记录显存、耗时和下游房间错误 |
| 后处理是否有帮助？ | 同一权重比较阈值 `0.4/0.5/0.6`、闭运算半径 `0/1/2` | 只用验证集选择，保留开口相关失败样例 |
| 旋转或亮度增强是否有帮助？ | 基线对照一种新增增强 | 需要改代码，图像与标签的几何变换必须同步 |
| 换损失或学习率是否有帮助？ | 一次改变一个训练选项 | 需要改代码，当前 CLI 没有损失或学习率参数 |

比较网络或损失时，除声明的变量外，应对齐数据、初始化策略、更新次数、分辨率、增强、优化器、权重选择和推理设置。同样数据与 batch 下，同 epoch 意味着同更新次数；换了分辨率，就不再是同计算量。应报告实际设备耗时；若研究计算效率，再单独做固定时长对照。选择候选期间保留一个不参与决策的最终测试集；如果某份测试结果已经指导了修改，就换新的保留集。私下保存 manifest、命令、环境版本、哈希和逐图结果，让有权限的协作者能够核对结论。
