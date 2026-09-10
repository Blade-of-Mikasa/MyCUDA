#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>


// [M, K] * [K, N] = [M, N]
template<
int BM = 32,
int BN = 32,
int BK = 16,
int TM = 2,
int TN = 2
>
__global__ void register_tiling_gemm(
    const float *A, 
    const float *B, 
    float *C, 
    const int M, 
    const int K, 
    const int N
){
    int y_idx = threadIdx.y, x_idx = threadIdx.x;
    int row = blockIdx.y * BM + y_idx, col = blockIdx.x * BN + x_idx;
    __shared__ float []
}