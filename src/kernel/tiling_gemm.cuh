#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// [M, K] * [K, N] = [M, N]
__global__ void tiling_gemm(
    const float *A, 
    const float *B, 
    float *C, 
    const int M, 
    const int K, 
    const int N, 
    const int TILE
){
    __shared__ As[TILE][TILE], Bs[TILE][TILE];

    int yid = threadIdx.y, xid = threadIdx.x;
    int row = blockIdx.y * blockDim.y + yid;
    int col = blockIdx.x * blockDim.x + xid;

    // C[row][col] += A[row][k] * B[k][col]
    for(int k = 0; k < K; k += TILE){
        // load
        int A_row = row, A_col = k * TILE + xid;
        if(A_row < M && A_col < K){
            // A[row][k * tile + xid]
            As[yid][xid] = A[A_row * K + A_col];
        }

        int B_row = k * TILE + yid, B_col = col;
        if(B_row < K && B_col < N){
            // B[k * tile + yid][col]
            Bs[yid][xid] = B[B_row * N + B_col]; 
        }

        __sync_thread();

        // cal
        float sum = 0;
        #pragma unroll
        for(int i = 0; i < TILE && k * TILE + i < K; ++ i){
            sum += As[yid][i] * Bs[i][xid];
        }

        C[row][col] = sum;
        __sync_thread();
    }
}