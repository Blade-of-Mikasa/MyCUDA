# Tensor Core + Multi-stage Pipeline + Warp Tiling

实现位于 `tensor_core_gemm.cuh`，延续 `tiling_gemm.cuh` 的行主序接口和参数顺序：

```cpp
// A[M,K] * B[K,N] -> C[M,N]，三个指针均指向连续的 device 数组。
cudaError_t status = launch_tensor_core_gemm(A, B, C, M, K, N, stream);
// 检查 status；使用结果前同步 stream，并检查执行错误。
```

需要 CUDA 11+ 和 Ampere（SM80）或更新的 GPU。A/B/C 仍是 `float`，计算时将 A/B 舍入为 TF32，使用 FP32 累加，因此和原来的完整 FP32 GEMM 可能有精度差异。C 覆盖写入，不得与 A/B 重叠。实现没有 alpha、beta、转置或独立 leading dimension 参数。

默认分块：

| 层级 | 大小 | 工作 |
| --- | --- | --- |
| Block tile | `BM=128, BN=128, BK=16` | 一个 block 负责 C 的 `128×128` 区域 |
| Warp tile | `WM=64, WN=64` | 4 个 warp 按 `2×2` 排列，每个计算 `64×64` |
| WMMA | `16×16×8` | 每个 warp 保存 `4×4` 个累加 fragment |
| Shared memory | 48 KiB/block | 三阶段 A/B 流水；完成计算后复用为尾块写回区 |

例如 warp 0 负责 C 的左上 `64×64`，warp 1 负责右上，warp 2 负责左下，warp 3 负责右下。warp 内的 32 个线程共同完成 `mma_sync`，同一 A/B fragment 会在更多输出 fragment 间复用。

默认三阶段流水执行顺序：

```text
预取 tile 0/1，等待 tile 0 并同步 block
预取 tile 2，同时计算 tile 0
等待 tile 1 并同步 block
预取 tile 3，同时计算 tile 1
……
最后一个 tile 只计算，不发起新的预取
```

`cp.async.wait_group 1/2` 保留后续拷贝在途；随后的 `__syncthreads()` 确保整个 block 完成所需加载并停止读取待复用 buffer。两者都需要。

完整对齐的 tile 使用无边界分支的 16 字节异步拷贝；非对齐输入行或尾部使用 4 字节拷贝，越界元素通过 `src-size=0` 补零。完整输出 tile 由 WMMA 直接写 global memory；尾块复用输入 shared memory 作安全中转。因此 M/N/K 不必是 tile 大小的倍数。`M=0` 或 `N=0` 不启动 kernel，`K=0` 将 C 写成 0。

在 NVIDIA GPU 环境，从本目录运行：

```bash
nvcc -O3 -std=c++17 -arch=sm_80 -lineinfo test_tensor_core_gemm.cu -o /tmp/test_tensor_core_gemm
/tmp/test_tensor_core_gemm
compute-sanitizer --tool memcheck --error-exitcode 1 /tmp/test_tensor_core_gemm
compute-sanitizer --tool racecheck --error-exitcode 1 /tmp/test_tensor_core_gemm
compute-sanitizer --tool synccheck --error-exitcode 1 /tmp/test_tensor_core_gemm
```

可以把 `sm_80` 换成目标 GPU 的架构。测试包含精确可表示输入与 CPU double 参考值的严格比较、普通 float 输入的 TF32 误差检查、1/2/3/4 个 K tile、多轮缓冲复用、非整块尺寸、非 16 字节对齐的输入/输出、空维度、重复覆盖写，以及另一组模板参数。普通输入的容差按点积各项绝对值之和设置，避免正负抵消使相对误差失真；严格用例的容差为 0。

该实现已在 RTX 4060 Laptop（SM89）与 CUDA 12.6 上完成编译、正确性和性能测试。详细过程见 `../../learning-note/GEMM优化实测.md`。

接口约束参考：[NVIDIA WMMA 文档](https://docs.nvidia.com/cuda/archive/12.5.0/cuda-c-programming-guide/index.html#warp-matrix-functions)；异步拷贝与同步语义参考：[NVIDIA PTX cp.async 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async)。
