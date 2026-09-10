#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>


// [M, K] * [K, N] = [M, N]
template<int TILE>
__global__ void tiling_gemm(
    const float *A, 
    const float *B, 
    float *C, 
    const int M, 
    const int K, 
    const int N
){
    __shared__ float As[TILE][TILE], Bs[TILE][TILE];

    int yid = threadIdx.y, xid = threadIdx.x;
    int row = blockIdx.y * TILE + yid;
    int col = blockIdx.x * TILE + xid;
    float sum = 0;

    // C[row][col] += A[row][k] * B[k][col]
    for(int k = 0; k < K; k += TILE){
        // load
        int A_row = row, A_col = k + xid;
        if(A_row < M && A_col < K){
            // A[row][k + xid]
            As[yid][xid] = A[A_row * K + A_col];
        } else{
            As[yid][xid] = 0.0;
        }

        int B_row = k + yid, B_col = col;
        if(B_row < K && B_col < N){
            // B[k + yid][col]
            Bs[yid][xid] = B[B_row * N + B_col]; 
        } else {
            Bs[yid][xid] = 0.0;
        }

        __syncthreads();

        // cal

        #pragma unroll
        for(int i = 0; i < TILE; ++ i){
            sum += As[yid][i] * Bs[i][xid];
        }

        __syncthreads();
    }

    if(row < M && col < N){
        C[row * N + col] = sum;
    }
}