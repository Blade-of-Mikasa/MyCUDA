#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>


// [M, K] * [K, N] = [M, N]
template<int TILE>
__global__ void gemm(
    const float *A, 
    const float *B, 
    float *C, 
    const int M, 
    const int K, 
    const int N
){
    
}