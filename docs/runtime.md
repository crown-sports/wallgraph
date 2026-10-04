# 运行与 GPU 验证

CPU 推理安装 `.[onnx]`；Linux NVIDIA 推理安装 `.[gpu]`。两种 ONNX Runtime wheel 提供相同 import，不应在同一环境同时安装 CPU 与 GPU wheel。建议为每个任务创建独立虚拟环境。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[gpu,train]'
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python -c 'import onnxruntime as o; print(o.__version__, o.get_available_providers())'
CUDA_VISIBLE_DEVICES=0 wallgraph detect --image /private/plan.png \
  --model /private/model.onnx --device cuda --output /private/run
```

GPU wheel、驱动、CUDA 与 cuDNN 版本应按 [ONNX Runtime 官方兼容表](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) 匹配。不能仅凭 `nvidia-smi` 正常就判断 PyTorch / ONNX 可以执行。运行时报告真实 provider；严格 CUDA 后端禁用 CPU 节点回退，不满足时明确失败。若某模型必须使用 CPU 算子，需要另行设计混合执行和测量，不应在严格 GPU 实测中省略这个事实。

5090 等新架构需要框架 wheel 支持其计算能力。训练前执行一次前向与反向，导出后执行一次 ONNX 推理并和 PyTorch 输出比较。GPU 性能报告应注明冷启动/预热、重复次数、模型摘要、输入尺寸和是否计入几何后处理；不能把未经标注的推理结果写成准确率。

训练基线默认为 FP32，不使用自动混合精度或分布式训练。先验证基础数值和数据协议，再基于真实对比引入性能优化。
