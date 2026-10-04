# 墙体像素怎样成为连接图

[English](methods.md) · [复现与实跑笔记](reproduction.zh-CN.md) · [完整验证](validation.md)

墙体分割给出一片像素；空间分析还需要知道墙在哪里转弯、如何相交、是否形成闭环。WallGraph 的主线是把分割结果组织为原图坐标下可检查的连接结构，并让模型、清理和矢量化各自可替换。

## 核心技术落在哪里

| 技术环节 | 解决的问题 | 对应实现 |
| --- | --- | --- |
| 原图概率与可替换后端 | 明确预处理、类别和设备，让不同模型遵守同一个输入输出约定 | [backends.py](../src/wallgraph/backends.py)、[pipeline.py](../src/wallgraph/pipeline.py) |
| 骨架到节点与路径 | 聚合交叉处关键像素、去除角点的对角捷径，逐边追踪；全二度闭环增加锚点 | [vectorize.py](../src/wallgraph/vectorize.py) |
| 几何与像素依据分开 | 简化折线，同时保留掩码；用距离估计像素厚度、用概率记录未校准置信度 | [domain.py](../src/wallgraph/domain.py)、[vectorize.py](../src/wallgraph/vectorize.py) |
| 重叠切片先融合概率 | 减轻切片边缘的硬接缝，在融合后统一阈值化；窗口权重保持正值 | [tiling.py](../src/wallgraph/tiling.py) |
| 小型训练基线 | 从私有标注跑通训练、验证选模和 ONNX 导出，padding 不参与损失 | [training.py](../src/wallgraph/training.py) |

骨架追踪不能补回漏检的墙。短路径过滤和像素邻接规则也会影响连接关系，现有测试覆盖构造几何，尚缺真实连接图标注。切片策略没有在本轮 30 张测试上证明优于整图推理。

## 论文依据与使用方式

| 论文或原始资料 | 本工程使用的部分 | 复现范围 |
| --- | --- | --- |
| Ronneberger、Fischer、Brox：[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)，MICCAI 2015 | 编码器—解码器及跳跃连接，落在 `SmallUNet` | 工程改编：两级下采样、16/32/64 通道、padding 卷积、双线性上采样、GN/SiLU、BCE+Dice；未复现原论文网络、医学数据、训练方案或成绩 |
| Wu、He：[Group Normalization](https://arxiv.org/abs/1803.08494)，ECCV 2018 | `nn.GroupNorm(4, channels)`，不使用批间统计 | 面向 batch 4 的设计选择，未做 GN 与 BN 消融，不能据此声称精度收益 |
| Zhang、Suen：[A fast parallel algorithm for thinning digital patterns](https://doi.org/10.1145/357994.358023)，CACM 1984 | 通过 `skimage.morphology.skeletonize` 调用二维细化 | 依赖库实现；[官方文档](https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.skeletonize) 明确二维默认使用 Zhang 方法，本工程独立实现后续图追踪 |
| Kalervo 等：[CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis](https://arxiv.org/abs/1904.01920)，2019 | 使用其官方图纸与 SVG 人工标注构建私有评测 | 数据与任务参考；未复现论文的完整多任务模型、全量训练或榜单成绩 |

厚度估计调用 [SciPy 的欧氏距离变换](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html)。路径样本采用 `max(1, 2*distance-1)` 的中位数，属于像素宽度估计；斜墙、交叉处和分辨率都会带来误差。折线简化调用 OpenCV `approxPolyDP`。概率加权融合是本工程的通用实现，没有对应已完成的论文成绩复现。

## 本工程的贡献怎样理解

主要工作集中在坐标恢复、连接结构、模型契约与可复核实验。读者可以换一个后端，仍然得到相同约定的 mask 和 graph；也可以替换矢量化策略，单独检查几何变化。

现有私人参考 ONNX 模型的网络来源和训练暴露未确认，不能把它认作某篇论文的官方权重。30 张测试中的 0.84203 来自外部参考模型加新处理流程；独立训练 `SmallUNet` 的成绩是 0.59297，两者必须分开解读。完整数字、配置和局限见验证文档。
