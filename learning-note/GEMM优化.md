# GEMM 优化

## Tiling

本质不是优化读写次数，而是通过编排把接下来要频繁读取的放到更快的内存上

block tiling：利用 shared memory

thread tiling：利用 register

问题：直接把 block 的 16*16 shared memory 放到 register 上，效果不也一样吗？

## Vectorized Load

本质用黑魔法让GPU一下读4个

float4 v = reinterpret_cast<const float4*>(A + p)

## Double Buffer

计算一个 tile 时异步搬运下一个 tile

cp.async.cg.shared.global [dst], [src], 16;

cp.async.commit_group;

cp.async.wait_group 0;

## Warp Tiling & Tensor Core

Tensor Core 用来替代 thread 级的 tiling。其他读取和更高层 tiling 还需要手写。

```Cuda
wmma::fragment<
    wmma::matrix_a,
    16, 16, 16,
    half,
    wmma::row_major
> a_frag;
```

wmma::fill_fragment(...)

wmma::load_matrix_sync(...)

wmma::mma_sync(...)

wmma::store_matrix_sync(...)