# Tune FP4 GEMV scheduling for client SM12x GPUs

Pull request: [microsoft/onnxruntime#32408](https://github.com/microsoft/onnxruntime/pull/32408)

## Description

Tune the tensor-core GEMV decomposition used by
`MatMulBlockQuantizedFp4Weight` for low-SM-count SM12x GPUs.

For long reductions on devices with at most 64 SMs, the existing H200-oriented
selector can provide insufficient K parallelism. The new selector uses
`KSplit=16, ColTiles=1` when the four-column output grid provides fewer than
eight waves. Already-saturated wide-output shapes retain the existing
selection.

## Performance

Measured on a 48-SM SM121 GPU with representative Qwen 3.8 27B decode shapes:

| Shape | M | Existing | Proposed | Improvement |
|---|---:|---:|---:|---:|
| 17408x5120 | 1 | 0.264 ms | 0.202 ms | 23.4% |
| 17408x5120 | 8 | 0.262 ms | 0.204 ms | 22.0% |
| 5120x17408 | 1 | 0.236 ms | 0.223 ms | 5.5% |
| 5120x17408 | 8 | 0.241 ms | 0.224 ms | 7.4% |

The selector remains unchanged for pre-SM12x GPUs, devices with more than 64
SMs, short reductions, and output grids that already provide at least eight
waves.

## Testing

- Release ARM64 CUDA build with CUDA 13.4 for SM120/SM121
- `onnxruntime_provider_test
  --gtest_filter=MatMulBlockQuantizedFp4WeightOpTest.*`
  (19 selector and FP16/BF16 operator tests passed)
- Selector boundary coverage for client SM12x, large SM12x, Hopper, short K,
  and wide N
