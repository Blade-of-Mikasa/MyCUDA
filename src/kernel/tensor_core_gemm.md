# Tensor Core + Double Buffer + Warp Tiling

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
| Block tile | `BM=64, BN=64, BK=32` | 一个 block 负责 C 的 `64×64` 区域 |
| Warp tile | `WM=32, WN=32` | 4 个 warp 按 `2×2` 排列，每个计算 `32×32` |
| WMMA | `16×16×8` | 每个 warp 保存 `2×2` 个累加 fragment |
| Shared memory | 36 KiB/block | 双份 A/B tile 共 32 KiB，写回中转区共 4 KiB |

例如 warp 0 负责 C 的左上 `32×32`，warp 1 负责右上，warp 2 负责左下，warp 3 负责右下。warp 内的 32 个线程共同完成一次 `mma_sync`。每次 K 前进 8，加载两个 A fragment、两个 B fragment，组合成四次 MMA，实现 fragment 复用。

双缓冲执行顺序：

```text
预取 tile 0 -> buffer 0，等待并同步 block
预取 tile 1 -> buffer 1，同时计算 buffer 0
等待并同步 block，切换 read buffer
预取 tile 2 -> buffer 0，同时计算 buffer 1
等待并同步 block，切换 read buffer
……
最后一个 tile 只计算，不发起新的预取
```

`cp.async.wait_group 0` 等待当前线程的拷贝；随后的 `__syncthreads()` 确保整个 block 完成加载和旧 buffer 的读取。两者都需要。能否充分隐藏搬运延迟、能提升多少性能，需要在目标 GPU 上测量。

对齐且完整的连续四个 float 用 16 字节异步拷贝；非对齐输入行或尾部使用 4 字节拷贝，越界元素通过 `src-size=0` 补零。shared memory 按 32 字节对齐。最后先把完整 WMMA fragment 写入每个 warp 私有的 shared 中转区，再检查行列边界写 C，所以 M/N/K 不必是 tile 大小的倍数。`M=0` 或 `N=0` 不启动 kernel，`K=0` 将 C 写成 0。

在 NVIDIA GPU 环境，从本目录运行：

```bash
nvcc -O3 -std=c++17 -arch=sm_80 -lineinfo test_tensor_core_gemm.cu -o /tmp/test_tensor_core_gemm
/tmp/test_tensor_core_gemm
compute-sanitizer --tool memcheck --error-exitcode 1 /tmp/test_tensor_core_gemm
compute-sanitizer --tool racecheck --error-exitcode 1 /tmp/test_tensor_core_gemm
compute-sanitizer --tool synccheck --error-exitcode 1 /tmp/test_tensor_core_gemm
```

可以把 `sm_80` 换成目标 GPU 的架构。测试包含精确可表示输入与 CPU double 参考值的严格比较、普通 float 输入的 TF32 误差检查、1/2/3/4 个 K tile、多轮缓冲复用、非整块尺寸、非 16 字节对齐的输入/输出、空维度、重复覆盖写，以及另一组模板参数。普通输入的容差按点积各项绝对值之和设置，避免正负抵消使相对误差失真；严格用例的容差为 0。

本次开发环境为 macOS ARM，没有 nvcc/NVIDIA GPU，因此上述 GPU 编译、正确性测试、sanitizer 和性能测试尚未执行。`test_tensor_core_gemm.cu` 是独立测试入口，仓库原有 `src/run.cu` 目前为空。

接口约束参考：[NVIDIA WMMA 文档](https://docs.nvidia.com/cuda/archive/12.5.0/cuda-c-programming-guide/index.html#warp-matrix-functions)；异步拷贝与同步语义参考：[NVIDIA PTX cp.async 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async)。
