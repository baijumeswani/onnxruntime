# Add FP8 GEMV KSplit32 scheduling for client SM12x GPUs

Pull request: [microsoft/onnxruntime#32409](https://github.com/microsoft/onnxruntime/pull/32409)

## Description

Add a KSplit-32 tensor-core GEMV specialization for
`MatMulBlockQuantizedFp8Weight` and select it for low-M decode on low-SM-count
SM12x GPUs.

The selector keeps KSplit 16 for M greater than eight and for narrow multi-row
projections, avoiding the resource-limit failure observed with 1,024-thread
KSplit-32 blocks at larger M. Other architectures retain their existing
selection.

## Performance

Measured on a 48-SM SM121 GPU:

| Workload | Existing | Proposed | Improvement |
|---|---:|---:|---:|
| Qwen FP16-KV, 4K | 29.57 tok/s | 30.07 tok/s | 1.7% |
| Qwen FP16-KV, 16K | 29.53 tok/s | 29.97 tok/s | 1.5% |
| Qwen INT8-KV, 16K at matching acceptance | 31.05 tok/s | 31.47 tok/s | 1.4% |
| Qwen INT8-KV, batch 4 | 69.3 tok/s | 70.9 tok/s | 2.3% |

Representative projection kernels improved 4.7-8.3%, including a 7.8%
improvement for the 248320x5120 LM head at M=8.

More than 99.7% of compared FP16 outputs were bit-identical between KSplit 16
and 32. The maximum observed difference was 0.000244141, caused by FP32
reduction order.

## Testing

- Release ARM64 CUDA build with CUDA 13.4 for SM120/SM121
- `onnxruntime_provider_test
  --gtest_filter=MatMulBlockQuantizedFp8WeightOpTest.*`
  (12 selector and FP16/BF16 operator tests passed)
- Selector coverage for KSplit 8, 16, and 32, including the M=16 fallback
