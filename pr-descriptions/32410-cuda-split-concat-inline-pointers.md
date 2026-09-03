# Avoid pinned buffers in CUDA Split and Concat fast paths

Pull request: [microsoft/onnxruntime#32410](https://github.com/microsoft/onnxruntime/pull/32410)

## Description

Avoid allocating pinned host pointer buffers in the CUDA Split and Concat fast
paths when all partitions are equal-sized and there are at most 32 pointers.

These paths pass a `TArray` directly as a kernel argument, so the
`CudaAsyncBuffer` allocation and destruction were unnecessary. In a Qwen 3.8
decode trace, their removal reduced pinned allocation/free pairs from 1,635 to
1,355 and reduced time attributed to `cudaFreeHost` from 4.90 seconds to
11.3 milliseconds.

The GPU pointer-buffer fallback is unchanged for more than 32 pointers and for
unequal Split/Concat dimensions.

## Motivation

`cudaFreeHost` can synchronize outstanding device work. Allocating pinned
pointer metadata before selecting the inline fast path introduced avoidable
decode gaps even though the allocation was never copied to or consumed by the
GPU.

## Testing

- Release ARM64 CUDA build with CUDA 13.4 for SM120/SM121
- `onnxruntime_provider_test --gtest_filter=*ConcatOpTest*:*SplitOperatorTest*`
  (53 tests passed across four suites)
- Equal-sized fast paths and existing fallback-path coverage
