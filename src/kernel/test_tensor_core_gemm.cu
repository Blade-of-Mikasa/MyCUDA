#include "tensor_core_gemm.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check_cuda(cudaError_t status, const char* expression, int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "line %d: %s: %s\n", line, expression, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK_CUDA(expr) check_cuda((expr), #expr, __LINE__)

static void expect_status(cudaError_t actual, cudaError_t expected) {
    if (actual != expected) {
        std::fprintf(stderr, "expected %s, got %s\n",
                     cudaGetErrorString(expected), cudaGetErrorString(actual));
        std::exit(EXIT_FAILURE);
    }
}

template<int BM = 64, int BN = 64, int BK = 32, int WM = 32, int WN = 32>
static void run_case(int M, int K, int N, bool exact, int offset, cudaStream_t stream) {
    const std::size_t a_size = static_cast<std::size_t>(M) * K;
    const std::size_t b_size = static_cast<std::size_t>(K) * N;
    const std::size_t c_size = static_cast<std::size_t>(M) * N;
    std::vector<float> a(a_size), b(b_size);
    unsigned state = 12345;
    auto value = [&]() {
        state = state * 1664525u + 1013904223u;
        // /16 的输入和这些短点积都能被 TF32/FP32 精确表示。
        // /1000 另外检查 TF32 舍入后的数值误差。
        return exact ? (static_cast<int>((state >> 16) % 33) - 16) / 16.0f
                     : (static_cast<int>((state >> 16) % 2001) - 1000) / 1000.0f;
    };
    for (float& x : a) x = value();
    for (float& x : b) x = value();

    float *a_base = nullptr, *b_base = nullptr, *c_base = nullptr;
    if (K > 0) {
        // offset=1 专门验证只有 4 字节对齐的输入子视图。
        CHECK_CUDA(cudaMalloc(&a_base, (a_size + offset) * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&b_base, (b_size + offset) * sizeof(float)));
        CHECK_CUDA(cudaMemcpyAsync(a_base + offset, a.data(), a_size * sizeof(float), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(b_base + offset, b.data(), b_size * sizeof(float), cudaMemcpyHostToDevice, stream));
    }
    constexpr std::size_t guard = 17;
    constexpr float sentinel = 12345.0f;
    std::vector<float> output(c_size + 2 * guard, sentinel);
    CHECK_CUDA(cudaMalloc(&c_base, output.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpyAsync(c_base, output.data(), output.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    const float* da = a_base ? a_base + offset : nullptr;
    const float* db = b_base ? b_base + offset : nullptr;

    // 重复覆盖写，检查实现没有依赖旧 C。默认使用非默认 stream。
    for (int repeat = 0; repeat < 3; ++repeat) {
        const cudaError_t status = launch_tensor_core_gemm<BM, BN, BK, WM, WN>(
            da, db, c_base + guard, M, K, N, stream);
        CHECK_CUDA(status);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaMemcpy(output.data(), c_base, output.size() * sizeof(float), cudaMemcpyDeviceToHost));

    for (std::size_t i = 0; i < guard; ++i) {
        if (output[i] != sentinel || output[guard + c_size + i] != sentinel) {
            std::fprintf(stderr, "C guard overwritten for M=%d K=%d N=%d\n", M, K, N);
            std::exit(EXIT_FAILURE);
        }
    }
    double max_error = 0.0;
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            double reference = 0.0, sum_abs = 0.0;
            for (int k = 0; k < K; ++k) {
                const double product = static_cast<double>(a[static_cast<std::size_t>(row) * K + k])
                                       * b[static_cast<std::size_t>(k) * N + col];
                reference += product;
                sum_abs += std::abs(product);
            }
            const float actual = output[guard + static_cast<std::size_t>(row) * N + col];
            const double error = std::abs(static_cast<double>(actual) - reference);
            // 普通输入：按点积各项绝对值之和容纳两次 TF32 输入舍入，
            // 避免仅按最终结果设置相对误差时，被正负抵消误导。
            const double tolerance = exact ? 0.0 : 0.0011 * sum_abs + 1e-5;
            if (!std::isfinite(actual) || error > tolerance) {
                std::fprintf(stderr,
                    "FAIL M=%d K=%d N=%d (%d,%d): got %.9g, expected %.12g, tolerance %.6g\n",
                    M, K, N, row, col, actual, reference, tolerance);
                std::exit(EXIT_FAILURE);
            }
            max_error = std::max(max_error, error);
        }
    }
    std::printf("PASS M=%d K=%d N=%d exact=%d offset=%d tile=%dx%dx%d warp=%dx%d max_error=%.6g\n",
                M, K, N, exact, offset, BM, BN, BK, WM, WN, max_error);
    if (a_base) CHECK_CUDA(cudaFree(a_base));
    if (b_base) CHECK_CUDA(cudaFree(b_base));
    CHECK_CUDA(cudaFree(c_base));
}

int main() {
    expect_status(launch_tensor_core_gemm(nullptr, nullptr, nullptr, -1, 1, 1), cudaErrorInvalidValue);
    expect_status(launch_tensor_core_gemm(nullptr, nullptr, nullptr, 1, -1, 1), cudaErrorInvalidValue);
    expect_status(launch_tensor_core_gemm(nullptr, nullptr, nullptr, 1, 1, -1), cudaErrorInvalidValue);
    expect_status(launch_tensor_core_gemm(nullptr, nullptr, nullptr, 0, 7, 8), cudaSuccess);
    expect_status(launch_tensor_core_gemm(nullptr, nullptr, nullptr, 7, 8, 0), cudaSuccess);
    expect_status(launch_tensor_core_gemm(nullptr, nullptr, nullptr, 1, 1, 1), cudaErrorInvalidValue);

    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    if (prop.major < 8) {
        std::fprintf(stderr, "SM80+ required, got %s (SM%d%d)\n", prop.name, prop.major, prop.minor);
        return EXIT_FAILURE;
    }
    std::printf("GPU: %s (SM%d%d)\n", prop.name, prop.major, prop.minor);
    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // 参数顺序始终是 M, K, N。
    const int shapes[][3] = {
        {1, 1, 1}, {16, 8, 16}, {31, 7, 19},
        {64, 32, 64},       // 1 个 K tile
        {64, 64, 64},       // 2 个 K tile
        {64, 96, 64},       // 第一次复用 buffer 0
        {64, 128, 64},      // 第一次复用 buffer 1
        {65, 33, 67},       // M/N/K 都有尾部，行地址不总是 16 字节对齐
        {127, 129, 130}, {128, 256, 128},
        {33, 1025, 29},     // 多轮缓冲切换且最后一个 tile 仅有 1 个 K
        {65, 0, 67}        // 空点积必须覆盖 C 为 0
    };
    for (const auto& shape : shapes) {
        run_case(shape[0], shape[1], shape[2], true, 0, stream);
    }
    run_case(67, 131, 69, false, 0, stream);
    run_case(96, 257, 80, false, 1, stream);
    run_case(64, 64, 64, true, 1, stream);
    run_case<32, 64, 16, 16, 32>(37, 49, 71, true, 0, stream);
    CHECK_CUDA(cudaStreamDestroy(stream));
    std::puts("All Tensor Core GEMM tests passed.");
    return EXIT_SUCCESS;
}
