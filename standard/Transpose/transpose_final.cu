#include <cuda_runtime.h>

constexpr int TILE = 32;

__global__ void transpose_tiled(
    const float* input,
    float* output,
    int M,
    int N
) {
    // +1 用于避免 shared memory bank conflict
    __shared__ float tile[TILE][TILE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // input 是 M x N
    int x = blockIdx.x * TILE + tx;
    int y = blockIdx.y * TILE + ty;

    // 1. global memory -> shared memory
    // 相邻线程访问连续地址，coalesced
    if (x < N && y < M) {
        tile[ty][tx] = input[y * N + x];
    }

    __syncthreads();

    /*
        转置后：

        原 block (blockIdx.y, blockIdx.x)

        要写到 output 中对应的
        (blockIdx.x, blockIdx.y) 转置位置。

        output 大小为 N x M
    */
    int out_x = blockIdx.y * TILE + tx;
    int out_y = blockIdx.x * TILE + ty;

    // 2. shared memory -> global memory
    //
    // tile[tx][ty] 完成 block 内部转置
    //
    // 因为 shared 是 [32][33]，
    // 所以按列读取时不会发生 32-way bank conflict。
    //
    // 同时 output 相邻线程写连续地址，coalesced。
    if (out_x < M && out_y < N) {
        output[out_y * M + out_x] = tile[tx][ty];
    }
}


// input:  M x N
// output: N x M
extern "C" void solve(
    const float* input,
    float* output,
    int M,
    int N
) {
    dim3 threads(TILE, TILE);

    dim3 blocks(
        (N + TILE - 1) / TILE,
        (M + TILE - 1) / TILE
    );

    transpose_tiled<<<blocks, threads>>>(
        input,
        output,
        M,
        N
    );
}