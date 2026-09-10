# GEMM 优化实测

## 测试口径

- 设备：RTX 4060 Laptop（SM89），CUDA 12.6。
- 输入/输出为行主序 `float`；计算使用 TF32 Tensor Core、FP32 累加。
- 使用 CUDA event 计时，预热不计入结果；cuBLAS 使用 `CUBLAS_COMPUTE_32F_FAST_TF32`。
- 移动 GPU 会随温度和功耗动态变频，因此下表取连续三轮测试的中位数。

## 优化过程

1. **先调 tile，而不是先堆指令。** 原配置为 block `64×64×32`、warp `32×32`、双缓冲，大矩阵约 `5 TFLOPS`。扩大到 block `128×128×16`、warp `64×64` 后，每个 warp 复用更多 A/B fragment，同时仍能保持每个 SM 两个 block，提升到约 `7.7–8.2 TFLOPS`。
2. **区分快路径和通用路径。** 完整且对齐的 tile 使用无边界分支的 16-byte `cp.async.cg`；完整输出直接由 WMMA 写到 global memory。尾块仍走逐项检查路径，所以非整块和非对齐输入仍然正确。
3. **复用 shared memory。** 输入流水线结束后，原空间复用为尾块写回区，避免 A/B buffer 与 C staging 同时占用 shared memory。
4. **三阶段流水。** 预取两个 K tile，并用 `wait_group 1/2` 保持后续拷贝在途，最终大矩阵达到约 `8.6–8.7 TFLOPS`。
5. **保留失败实验。** BK=8 配四阶段流水同步过多；限制寄存器使吞吐降至约 `7.5–7.9 TFLOPS`；更大的 block 降低并发；shared padding 和 `.ca`/`.cg` 的差异不稳定，因此没有加入默认配置。

## 结果与结论

连续三轮的中位数如下（`TF` 单位为 TFLOPS）：

| N | 旧内核 TF | 优化后 TF | cuBLAS TF | 优化后/cuBLAS |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 4.90 | 6.21 | 9.54 | 65.2% |
| 2048 | 5.58 | 7.95 | 9.91 | 80.2% |
| 4096 | 5.38 | 8.56 | 10.22 | 83.8% |
| 8192 | 4.94 | 8.73 | 10.54 | 82.8% |

在 `4096–8192` 方阵上，最终内核达到 cuBLAS TF32 的约 `83%`，相对最初版本提升约 `59%–77%`。剩余差距主要来自 WMMA 抽象、shared-memory 布局和指令级流水；cuBLAS 使用更低层的架构专用 MMA/`ldmatrix` 内核及更完整的尺寸调度。

最终默认配置：

```text
Block tile: 128×128×16
Warp tile:  64×64
Threads:    128
Pipeline:   3 stages
Shared:     48 KiB/block
Registers:  241/thread（SM89，nvcc 12.6，-O3）
```

复现：

```bash
cd ~/workspace/MyCUDA/src/kernel
nvcc -O3 -std=c++17 -arch=sm_89 -lineinfo benchmark_tensor_core_gemm.cu -lcublas -o /tmp/benchmark_tensor_core_gemm
/tmp/benchmark_tensor_core_gemm
```

正确性测试覆盖边界尺寸、K 尾块、非对齐子视图、空点积和另一组模板参数，均已通过。WSL 当前未启用 WDDM debugger interface，因此无法采集 Compute Sanitizer/Nsight Compute 的硬件检查结果。
