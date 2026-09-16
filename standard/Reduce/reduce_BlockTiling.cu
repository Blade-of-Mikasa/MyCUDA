#include <cuda_runtime.h>

__global__ void reduce_kernel(
const float *A, int N,
float *S
){
    extern __shared__  float s[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    s[tid] = (i < N) ? A[i] : 0.0;
    __syncthreads();
    for(int len = blockDim.x / 2; len > 0; len >>= 1){
        if(tid < len){
            s[tid] += s[tid + len];
        }
        __syncthreads();
    }
    if(tid == 0){
        atomicAdd(S, s[0]);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadPerBlock = 256;
    int BlockPerGrid = (N + threadPerBlock - 1) / threadPerBlock;
    cudaMemset(output, 0, sizeof(float));
    reduce_kernel<<<BlockPerGrid, threadPerBlock, threadPerBlock * sizeof(float)>>>(input, N, output);
}
