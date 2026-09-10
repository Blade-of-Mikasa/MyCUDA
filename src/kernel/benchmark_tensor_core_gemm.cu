#include "tensor_core_gemm.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        const cudaError_t status = (call);                                      \
        if (status != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,     \
                         cudaGetErrorString(status));                            \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

#define CHECK_CUBLAS(call)                                                      \
    do {                                                                        \
        const cublasStatus_t status = (call);                                   \
        if (status != CUBLAS_STATUS_SUCCESS) {                                  \
            std::fprintf(stderr, "cuBLAS %s:%d: %d\n", __FILE__, __LINE__,   \
                         static_cast<int>(status));                              \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

template<class Launch>
float measure_ms(Launch launch, int warmup, int iterations) {
    for (int i = 0; i < warmup; ++i) launch();
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) launch();
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float elapsed = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return elapsed / iterations;
}

double throughput(int n, float ms) {
    return 2.0 * static_cast<double>(n) * n * n / (ms * 1.0e9);
}

template<int BM, int BN, int BK, int WM, int WN, int Stages>
float measure_custom(float* a, float* b, float* c, int n, int iterations) {
    const dim3 grid((n + BN - 1) / BN, (n + BM - 1) / BM);
    const dim3 block((BM / WM) * (BN / WN) * 32);
    return measure_ms([&] {
        tensor_core_gemm<BM, BN, BK, WM, WN, 0, 0, Stages>
            <<<grid, block>>>(a, b, c, n, n, n);
    }, 10, iterations);
}

void run_case(int n, int iterations, cublasHandle_t handle) {
    const size_t bytes = static_cast<size_t>(n) * n * sizeof(float);
    float *a, *b, *c;
    CHECK_CUDA(cudaMalloc(&a, bytes));
    CHECK_CUDA(cudaMalloc(&b, bytes));
    CHECK_CUDA(cudaMalloc(&c, bytes));
    CHECK_CUDA(cudaMemset(a, 0, bytes));
    CHECK_CUDA(cudaMemset(b, 0, bytes));

    const float old_ms = measure_custom<64, 64, 32, 32, 32, 2>(
        a, b, c, n, iterations);
    const float optimized_ms = measure_custom<128, 128, 16, 64, 64, 3>(
        a, b, c, n, iterations);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const float cublas_ms = measure_ms([&] {
        // Row-major C=A*B is column-major C^T=B^T*A^T.
        CHECK_CUBLAS(cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
            b, CUDA_R_32F, n, a, CUDA_R_32F, n, &beta,
            c, CUDA_R_32F, n, CUBLAS_COMPUTE_32F_FAST_TF32,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }, 10, iterations);

    const double old_tf = throughput(n, old_ms);
    const double optimized_tf = throughput(n, optimized_ms);
    const double cublas_tf = throughput(n, cublas_ms);
    std::printf("%5d %5d  %8.3f %7.2f  %8.3f %7.2f  %8.3f %7.2f  %6.1f%%\n",
                n, iterations,
                old_ms, old_tf, optimized_ms, optimized_tf,
                cublas_ms, cublas_tf, 100.0 * optimized_tf / cublas_tf);

    CHECK_CUDA(cudaFree(a));
    CHECK_CUDA(cudaFree(b));
    CHECK_CUDA(cudaFree(c));
}

int main() {
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s (SM%d%d)\n", prop.name, prop.major, prop.minor);
    std::puts("FP32 input, TF32 Tensor Core compute; time excludes warm-up");
    std::puts("    N  iter    old_ms  old_TF    opt_ms  opt_TF  cublas_ms  cu_TF   opt/cu");

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    run_case(1024, 200, handle);
    run_case(2048, 100, handle);
    run_case(4096, 30, handle);
    run_case(8192, 10, handle);
    CHECK_CUBLAS(cublasDestroy(handle));
    return 0;
}
