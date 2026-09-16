#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// [M, K] * [K, N] = [M, N]
template<int TILE>
__global__ void gemm_block_tiling(
    int M, int K, int N,
    float * a, float * b, float * c
){
    __shared__ float ta[TILE], tb[TILE];

    int yid = threadIdx.y, xid = threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(){
        // load a tiling
    }
    if(){
        // load b tiling
    }

    __syncthreads();


}

int main(){
    Dim3 threadNum = {16, 16};
    int3 blockNum = (N + threadNum - 1) / threadNum;
    gemm_block_tiling<<<threadNum
}