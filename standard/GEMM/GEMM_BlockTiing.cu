#include <cuda_runtime.h>

template <int TM, int TN, int TK>
__global__ void matrix_multiplication_kernel(
const float* A, const float* B, float* C,
int M, int N, int K
) {
    __shared__ float ta[TM * TN], tb[TN * TK];
    int yid = threadIdx.y, xid = threadIdx.x;
    int i = blockIdx.y * blockDim.y + yid, j = blockIdx.x * blockDim.x + xid;
    float s = 0;
    for(int p = 0; p < N; p += TN){
        ta[yid * TN + xid] = (i < M && p + xid < N) ? A[i * N + (p + xid)] : 0.0;
        tb[yid * TK + xid] = (p + yid < N && j < K) ? B[(p + yid) * K + j] : 0.0;
        __syncthreads();
        #pragma unroll
        for(int t = 0; t < TN; ++ t){
            s += ta[yid * TN + t] * tb[t * TK + xid];
        }
        __syncthreads();
    }
    if(i < M && j < K){
        C[i * K + j] = s;
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    // [M, N] [N, K]
    matrix_multiplication_kernel<16, 16, 16><<<blocksPerGrid, threadsPerBlock>>>( A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
