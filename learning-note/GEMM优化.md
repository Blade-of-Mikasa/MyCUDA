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

## 内嵌汇编

```Cpp
asm(
    "order %index, %index, %index;"
    : output
    : input
);
```

## Bank

为了同时兼顾成本（不能每个字节都一个端口）和正常访问顺序（一般是 thread_i <-> 地址 x+i ），所以分 Bank 时安排成交叉形 （Bank1, Bank2, ..., Bank32, Bank1, ...）

访问一个 Bank 的不同地址时退化为串行。

# 优化指标

## key

核心优化指标

- latency
- throughput

## objective

用来观察，分析和解释核心指标

- Memory Bandwidth
- Compute Utilization
- Arithmetic Intensity
- Roofline