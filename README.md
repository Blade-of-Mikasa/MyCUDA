# MyCUDA

CUDA GEMM 学习与性能优化项目。

- CUDA 内核与独立测试：`src/kernel/`
- PyTorch 自定义算子：`pytorch/`
- 优化记录：`learning-note/GEMM优化实测.md`

PyTorch 扩展快速开始：

```bash
cd ~/workspace/MyCUDA/pytorch
source ~/.venvs/pytorch/bin/activate
python -m pip install -e . --no-build-isolation
python test_extension.py
```

详细接口、限制与 benchmark 方法见 `pytorch/README.md`。
