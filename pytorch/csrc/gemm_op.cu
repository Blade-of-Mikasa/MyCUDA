#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include <cstdint>
#include <limits>

#include "tensor_core_gemm.cuh"

namespace {

at::Tensor gemm_cuda(const at::Tensor& a, const at::Tensor& b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "mycuda::gemm expects CUDA tensors");
    TORCH_CHECK(a.layout() == c10::kStrided && b.layout() == c10::kStrided,
                "mycuda::gemm expects strided tensors");
    TORCH_CHECK(a.scalar_type() == at::kFloat && b.scalar_type() == at::kFloat,
                "mycuda::gemm supports torch.float32 only");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2,
                "mycuda::gemm expects two 2-D tensors");
    TORCH_CHECK(a.device() == b.device(),
                "mycuda::gemm expects tensors on the same CUDA device");
    TORCH_CHECK(a.size(1) == b.size(0),
                "mycuda::gemm shape mismatch: A is ", a.sizes(),
                " and B is ", b.sizes());

    constexpr int64_t kIntMax = std::numeric_limits<int>::max();
    const int64_t m64 = a.size(0);
    const int64_t k64 = a.size(1);
    const int64_t n64 = b.size(1);
    TORCH_CHECK(m64 <= kIntMax && k64 <= kIntMax && n64 <= kIntMax,
                "mycuda::gemm dimensions must fit in a 32-bit integer");

    const c10::cuda::CUDAGuard device_guard(a.device());
    auto output = at::empty({m64, n64}, a.options());
    if (m64 == 0 || n64 == 0) {
        return output;
    }

    // The CUDA kernel consumes contiguous row-major matrices. contiguous() is
    // a no-op for the common case and makes sliced/transposed inputs safe.
    const auto a_contiguous = a.contiguous();
    const auto b_contiguous = b.contiguous();

    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 16;
    constexpr int WM = 64;
    constexpr int WN = 64;
    constexpr int Stages = 3;
    constexpr int Threads = (BM / WM) * (BN / WN) * 32;

    const int m = static_cast<int>(m64);
    const int k = static_cast<int>(k64);
    const int n = static_cast<int>(n64);
    const uint64_t grid_x = (static_cast<uint64_t>(n) + BN - 1) / BN;
    const uint64_t grid_y = (static_cast<uint64_t>(m) + BM - 1) / BM;
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(grid_x <= static_cast<uint64_t>(properties->maxGridSize[0]) &&
                    grid_y <= static_cast<uint64_t>(properties->maxGridSize[1]),
                "mycuda::gemm launch grid is too large for this CUDA device");

    const dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y));
    const dim3 block(Threads);
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(a.get_device());
    tensor_core_gemm<BM, BN, BK, WM, WN, 0, 0, Stages>
        <<<grid, block, 0, stream>>>(
            a_contiguous.data_ptr<float>(), b_contiguous.data_ptr<float>(),
            output.data_ptr<float>(), m, k, n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

}  // namespace

TORCH_LIBRARY(mycuda, m) {
    m.def("gemm(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(mycuda, CUDA, m) {
    m.impl("gemm", TORCH_FN(gemm_cuda));
}
