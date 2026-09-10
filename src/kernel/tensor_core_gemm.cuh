#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <mma.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#error "tensor_core_gemm requires SM80 or newer; compile with -arch=sm_80 or newer"
#endif

namespace tensor_core_gemm_detail {

// global -> shared 异步拷贝。16 字节路径要求两端都按 16 字节对齐。
template<int Bytes>
__device__ __forceinline__ void copy_async(float* dst, const float* src) {
    const unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    if constexpr (Bytes == 16) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                     :: "r"(smem), "l"(src) : "memory");
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2;"
                     :: "r"(smem), "l"(src), "n"(Bytes) : "memory");
    }
}

// Fast path for complete, 16-byte-aligned tiles. The caller guarantees that
// both the source rows and the destination rows preserve 16-byte alignment.
template<int Rows, int Cols, int DstStride, int Threads>
__device__ __forceinline__ void load_tile_async_full(
    float* dst, const float* src, std::size_t row0, std::size_t col0,
    int src_cols
) {
    static_assert(Cols % 4 == 0, "tile rows must contain whole float4 groups");
    for (int i = static_cast<int>(threadIdx.x) * 4;
         i < Rows * Cols; i += Threads * 4) {
        const int row = i / Cols;
        const int col = i % Cols;
        copy_async<16>(dst + row * DstStride + col,
                       src + (row0 + row) * src_cols + col0 + col);
    }
}

// src-size=0：不读取 src，直接把目标位置补零。
__device__ __forceinline__ void zero_async(float* dst, const float* safe_src) {
    const unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, 0;"
                 :: "r"(smem), "l"(safe_src) : "memory");
}

__device__ __forceinline__ void commit_async() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}

template<int Pending>
__device__ __forceinline__ void wait_async() {
    asm volatile("cp.async.wait_group %0;" :: "n"(Pending) : "memory");
}

// 所有线程共同搬运一个行主序 tile，每次分配连续 4 个 float。
// 整块且对齐时用 16 字节拷贝；尾部/非对齐行用 4 字节拷贝或补零。
template<int Rows, int Cols, int DstStride, int Threads>
__device__ __forceinline__ void load_tile_async(
    float* dst, const float* src, std::size_t row0, std::size_t col0,
    int rows, int cols
) {
    static_assert(Cols % 4 == 0, "tile rows must contain whole float4 groups");
    for (int i = static_cast<int>(threadIdx.x) * 4;
         i < Rows * Cols; i += Threads * 4) {
        const std::size_t row = row0 + i / Cols;
        const std::size_t col = col0 + i % Cols;
        const int dst_row = i / Cols;
        const int dst_col = i % Cols;
        if (row < static_cast<std::size_t>(rows) &&
            col + 3 < static_cast<std::size_t>(cols)) {
            const float* ptr = src + row * cols + col;
            if ((reinterpret_cast<std::uintptr_t>(ptr) & 15u) == 0) {
                copy_async<16>(dst + dst_row * DstStride + dst_col, ptr);
                continue;
            }
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (row < static_cast<std::size_t>(rows) &&
                col + j < static_cast<std::size_t>(cols)) {
                copy_async<4>(dst + dst_row * DstStride + dst_col + j,
                              src + row * cols + col + j);
            } else {
                // 使用矩阵起始地址，避免构造越界指针。
                zero_async(dst + dst_row * DstStride + dst_col + j, src);
            }
        }
    }
}

} // namespace tensor_core_gemm_detail

// 与 tiling_gemm 一样：行主序 A[M,K] * B[K,N] -> C[M,N]，覆盖写入 C。
// A/B 在 Tensor Core 计算前舍入为 TF32，FP32 累加；不是完整 FP32 精度。
// A/B/C 必须是连续的 device 数组，C 不得与输入重叠。
// block 必须为 dim3((BM/WM)*(BN/WN)*32)，建议通过下方 launch 函数调用。
template<int BM = 128, int BN = 128, int BK = 16, int WM = 64, int WN = 64,
         int PadA = 0, int PadB = 0, int Stages = 3>
__global__ void tensor_core_gemm(
    const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, const int M, const int K, const int N
) {
    static_assert(BM > 0 && BN > 0 && BK > 0 && WM > 0 && WN > 0,
                  "tile dimensions must be positive");
    static_assert(WM % 16 == 0 && WN % 16 == 0 && BK % 8 == 0,
                  "TF32 WMMA shape is 16x16x8");
    static_assert(BM % WM == 0 && BN % WN == 0,
                  "block tile must contain whole warp tiles");
    constexpr int Warps = (BM / WM) * (BN / WN);
    constexpr int Threads = Warps * 32;
    constexpr int WarpRows = WM / 16;
    constexpr int WarpCols = WN / 16;
    constexpr int AStride = BK + PadA;
    constexpr int BStride = BN + PadB;
    static_assert(Threads <= 1024, "too many threads in a block");
    static_assert(Stages >= 2 && Stages <= 4, "two to four pipeline stages are supported");
    static_assert(PadA >= 0 && PadB >= 0 && AStride % 4 == 0 && BStride % 4 == 0,
                  "shared-memory strides must preserve 16-byte row alignment");
    constexpr std::size_t InputFloats = Stages * (BM * AStride + BK * BStride);
    constexpr std::size_t OutputFloats = Warps * 16 * 16;
    constexpr std::size_t SharedFloats =
        InputFloats > OutputFloats ? InputFloats : OutputFloats;
    static_assert(SharedFloats * sizeof(float) <= 48 * 1024,
                  "this kernel uses at most 48 KiB of static shared memory");

    namespace wmma = nvcuda::wmma;
    using namespace tensor_core_gemm_detail;

    // 多阶段流水：读 As[read]/Bs[read] 的同时异步预取后续 tile。
    // 32 字节对齐满足 WMMA；BK/BN 和每个 fragment 起点也满足对齐要求。
    struct __align__(32) InputStorage {
        float As[Stages][BM][AStride];
        float Bs[Stages][BK][BStride];
    };
    union __align__(32) SharedStorage {
        InputStorage input;
        float Cs[Warps][16 * 16];
    };
    // 输入流水缓冲在计算完成后复用为边界写回区，降低 shared-memory 占用。
    __shared__ SharedStorage storage;
#define As storage.input.As
#define Bs storage.input.Bs
#define Cs storage.Cs

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int warp_m = (warp / (BN / WN)) * WM;
    const int warp_n = (warp % (BN / WN)) * WN;
    const std::size_t block_m = static_cast<std::size_t>(blockIdx.y) * BM;
    const std::size_t block_n = static_cast<std::size_t>(blockIdx.x) * BN;
    const bool full_a_rows = block_m + BM <= static_cast<std::size_t>(M);
    const bool full_b_cols = block_n + BN <= static_cast<std::size_t>(N);
    const bool aligned_a = (reinterpret_cast<std::uintptr_t>(A) & 15u) == 0 && K % 4 == 0;
    const bool aligned_b = (reinterpret_cast<std::uintptr_t>(B) & 15u) == 0 && N % 4 == 0;

    wmma::fragment<wmma::matrix_a, 16, 16, 8,
                   wmma::precision::tf32, wmma::row_major> a_frag[WarpRows];
    wmma::fragment<wmma::matrix_b, 16, 16, 8,
                   wmma::precision::tf32, wmma::row_major> b_frag[WarpCols];
    wmma::fragment<wmma::accumulator, 16, 16, 8, float>
        c_frag[WarpRows][WarpCols];
    #pragma unroll
    for (int i = 0; i < WarpRows; ++i) {
        #pragma unroll
        for (int j = 0; j < WarpCols; ++j) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    const int tiles = K / BK + (K % BK != 0);
    // Prologue：双缓冲预取 1 个 tile；三/四缓冲预取 2/3 个 tile。
    if (tiles > 0) {
        if (full_a_rows && BK <= K && aligned_a) {
            load_tile_async_full<BM, BK, AStride, Threads>(&As[0][0][0], A, block_m, 0, K);
        } else {
            load_tile_async<BM, BK, AStride, Threads>(&As[0][0][0], A, block_m, 0, M, K);
        }
        if (full_b_cols && BK <= K && aligned_b) {
            load_tile_async_full<BK, BN, BStride, Threads>(&Bs[0][0][0], B, 0, block_n, N);
        } else {
            load_tile_async<BK, BN, BStride, Threads>(&Bs[0][0][0], B, 0, block_n, K, N);
        }
        commit_async();
        if constexpr (Stages >= 3) {
            if (tiles > 1) {
                if (full_a_rows && 2 * BK <= K && aligned_a) {
                    load_tile_async_full<BM, BK, AStride, Threads>(
                        &As[1][0][0], A, block_m, BK, K);
                } else {
                    load_tile_async<BM, BK, AStride, Threads>(
                        &As[1][0][0], A, block_m, BK, M, K);
                }
                if (full_b_cols && 2 * BK <= K && aligned_b) {
                    load_tile_async_full<BK, BN, BStride, Threads>(
                        &Bs[1][0][0], B, BK, block_n, N);
                } else {
                    load_tile_async<BK, BN, BStride, Threads>(
                        &Bs[1][0][0], B, BK, block_n, K, N);
                }
                commit_async();
            }
        }
        if constexpr (Stages == 4) {
            if (tiles > 2) {
                if (full_a_rows && 3 * BK <= K && aligned_a) {
                    load_tile_async_full<BM, BK, AStride, Threads>(
                        &As[2][0][0], A, block_m, 2 * BK, K);
                } else {
                    load_tile_async<BM, BK, AStride, Threads>(
                        &As[2][0][0], A, block_m, 2 * BK, M, K);
                }
                if (full_b_cols && 3 * BK <= K && aligned_b) {
                    load_tile_async_full<BK, BN, BStride, Threads>(
                        &Bs[2][0][0], B, 2 * BK, block_n, N);
                } else {
                    load_tile_async<BK, BN, BStride, Threads>(
                        &Bs[2][0][0], B, 2 * BK, block_n, K, N);
                }
                commit_async();
                wait_async<2>();
            } else if (tiles > 1) {
                wait_async<1>();
            } else {
                wait_async<0>();
            }
        } else if constexpr (Stages == 3) {
            if (tiles > 1) wait_async<1>();
            else wait_async<0>();
        } else {
            wait_async<0>();
        }
        // wait 只等待本线程的拷贝；block barrier 让其他线程也能安全读取。
        __syncthreads();
    }

    for (int tile = 0; tile < tiles; ++tile) {
        const int read = tile % Stages;
        const int next_tile = tile + Stages - 1;
        const bool has_prefetch = next_tile < tiles;
        if (has_prefetch) {
            const int write = next_tile % Stages;
            const std::size_t next_k = static_cast<std::size_t>(next_tile) * BK;
            if (full_a_rows && next_k + BK <= static_cast<std::size_t>(K) && aligned_a) {
                load_tile_async_full<BM, BK, AStride, Threads>(
                    &As[write][0][0], A, block_m, next_k, K);
            } else {
                load_tile_async<BM, BK, AStride, Threads>(
                    &As[write][0][0], A, block_m, next_k, M, K);
            }
            if (full_b_cols && next_k + BK <= static_cast<std::size_t>(K) && aligned_b) {
                load_tile_async_full<BK, BN, BStride, Threads>(
                    &Bs[write][0][0], B, next_k, block_n, N);
            } else {
                load_tile_async<BK, BN, BStride, Threads>(
                    &Bs[write][0][0], B, next_k, block_n, K, N);
            }
            commit_async();
        }

        // Warp tiling：默认每个 warp 保存 2x2 个 16x16 累加 fragment。
        // 同一个 A fragment 复用于两列，同一个 B fragment 复用于两行。
        #pragma unroll
        for (int k = 0; k < BK; k += 8) {
            #pragma unroll
            for (int i = 0; i < WarpRows; ++i) {
                wmma::load_matrix_sync(a_frag[i], &As[read][warp_m + i * 16][k], AStride);
                #pragma unroll
                for (int t = 0; t < a_frag[i].num_elements; ++t) {
                    a_frag[i].x[t] = wmma::__float_to_tf32(a_frag[i].x[t]);
                }
            }
            #pragma unroll
            for (int j = 0; j < WarpCols; ++j) {
                wmma::load_matrix_sync(b_frag[j], &Bs[read][k][warp_n + j * 16], BStride);
                #pragma unroll
                for (int t = 0; t < b_frag[j].num_elements; ++t) {
                    b_frag[j].x[t] = wmma::__float_to_tf32(b_frag[j].x[t]);
                }
            }
            #pragma unroll
            for (int i = 0; i < WarpRows; ++i) {
                #pragma unroll
                for (int j = 0; j < WarpCols; ++j) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }

        if (tile + 1 < tiles) {
            if constexpr (Stages == 4) {
                if (has_prefetch) {
                    wait_async<2>();
                } else {
                    wait_async<0>();
                }
            } else if constexpr (Stages == 3) {
                if (has_prefetch) {
                    wait_async<1>();
                } else {
                    wait_async<0>();
                }
            } else {
                wait_async<0>();
            }
            // 同时保证：下一组已加载，所有 warp 已读完当前组。
            // 下一轮才能安全地覆盖旧的 read buffer。
            __syncthreads();
        }
    }

    // 完整且对齐的输出 tile 直接由 WMMA 写 global memory，避免 shared 中转。
    const bool direct_store =
        block_m + BM <= static_cast<std::size_t>(M) &&
        block_n + BN <= static_cast<std::size_t>(N) &&
        (reinterpret_cast<std::uintptr_t>(C) & 31u) == 0 && N % 8 == 0;
    if (direct_store) {
        #pragma unroll
        for (int i = 0; i < WarpRows; ++i) {
            #pragma unroll
            for (int j = 0; j < WarpCols; ++j) {
                wmma::store_matrix_sync(
                    C + (block_m + warp_m + i * 16) * N + block_n + warp_n + j * 16,
                    c_frag[i][j], N, wmma::mem_row_major);
            }
        }
        return;
    }

    // 尾块先等待所有 warp 停止读取输入缓冲，再把它复用为写回区。
    __syncthreads();
    // WMMA 不能让部分 lane 退出，也不能把越界 fragment 直接写到 C。
    // 先完整写 shared，再由 32 个 lane 各自检查行列边界。
    #pragma unroll
    for (int i = 0; i < WarpRows; ++i) {
        #pragma unroll
        for (int j = 0; j < WarpCols; ++j) {
            wmma::store_matrix_sync(Cs[warp], c_frag[i][j], 16, wmma::mem_row_major);
            __syncwarp();
            for (int p = lane; p < 16 * 16; p += 32) {
                const std::size_t row = block_m + warp_m + i * 16 + p / 16;
                const std::size_t col = block_n + warp_n + j * 16 + p % 16;
                if (row < static_cast<std::size_t>(M) && col < static_cast<std::size_t>(N)) {
                    C[row * N + col] = Cs[warp][p];
                }
            }
            __syncwarp(); // 所有 lane 读完，才能复用该 warp 的中转区。
        }
    }
#undef As
#undef Bs
#undef Cs
}

// 异步 launch，返回参数/启动错误；执行错误由调用方同步 stream 后检查。
// M/N 为 0 时不启动 kernel；K 为 0 时把 C 写成 0，允许 A/B=nullptr。
template<int BM = 128, int BN = 128, int BK = 16, int WM = 64, int WN = 64,
         int PadA = 0, int PadB = 0, int Stages = 3>
inline cudaError_t launch_tensor_core_gemm(
    const float* A, const float* B, float* C, int M, int K, int N,
    cudaStream_t stream = nullptr
) {
    if (M < 0 || K < 0 || N < 0) return cudaErrorInvalidValue;
    if (M == 0 || N == 0) return cudaSuccess;
    if (C == nullptr || (K > 0 && (A == nullptr || B == nullptr))) {
        return cudaErrorInvalidValue;
    }
    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) return status;
    cudaDeviceProp prop{};
    status = cudaGetDeviceProperties(&prop, device);
    if (status != cudaSuccess) return status;
    if (prop.major < 8) return cudaErrorNotSupported;

    const dim3 grid(N / BN + (N % BN != 0), M / BM + (M % BM != 0));
    const dim3 block((BM / WM) * (BN / WN) * 32);
    if (grid.x > static_cast<unsigned>(prop.maxGridSize[0]) ||
        grid.y > static_cast<unsigned>(prop.maxGridSize[1])) {
        return cudaErrorInvalidConfiguration;
    }
    tensor_core_gemm<BM, BN, BK, WM, WN, PadA, PadB, Stages>
        <<<grid, block, 0, stream>>>(A, B, C, M, K, N);
    return cudaGetLastError();
}
