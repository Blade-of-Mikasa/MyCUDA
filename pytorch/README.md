# MyCUDA PyTorch 扩展

该目录把 `src/kernel/tensor_core_gemm.cuh` 注册为 PyTorch 自定义算子：

```python
import torch
import mycuda

a = torch.randn(1024, 1024, device="cuda")
b = torch.randn(1024, 1024, device="cuda")
c = mycuda.gemm(a, b)
# 原始 Dispatcher 入口同样可用：torch.ops.mycuda.gemm(a, b)
```

## 构建与验证

```bash
cd ~/workspace/MyCUDA/pytorch
source ~/.venvs/pytorch/bin/activate
python -m pip install -e . --no-build-isolation
python test_extension.py
python benchmark.py
```

开发时若只想在源码目录生成扩展，也可以运行 `python setup.py build_ext --inplace`。

算子要求 Ampere（SM80）或更新的 NVIDIA GPU，以及同一设备上的二维 CUDA `float32` 输入。输入可以不连续，扩展会在需要时生成连续副本；输出始终是连续行主序张量。前向使用 TF32 Tensor Core 和 FP32 累加，因此误差特性与启用 TF32 的 `torch.mm` 相近。

当前 PyTorch 2.14 构建需要支持 C++20 的主机编译器。

Python 层注册了 autograd：`dA=dC@B.T`、`dB=A.T@dC`，反向当前使用 PyTorch 自带 matmul。FakeTensor 实现使算子能够进入 `torch.compile` 图。原生实现遵循 PyTorch 当前 CUDA device 和 stream，不会擅自同步。
