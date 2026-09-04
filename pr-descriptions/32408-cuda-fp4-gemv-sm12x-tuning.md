# Tune FP4 GEMV scheduling for the 48-SM SM121 GPU

Pull request: [microsoft/onnxruntime#32408](https://github.com/microsoft/onnxruntime/pull/32408)

## Description

Tune the tensor-core GEMV decomposition used by
`MatMulBlockQuantizedFp4Weight` for the qualified 48-SM SM121 GPU.

The existing H200-oriented selector can provide insufficient K parallelism for
specific wide-grid and long-reduction decode shapes on SM121. The new selector
uses `KSplit=16, ColTiles=1` only when all of these conditions hold:

- compute capability is exactly 12.1;
- the device has exactly 48 SMs;
- M is at most 16;
- K contains at least 40 128-element windows;
- the shape either provides four wide-grid waves or has at least 128 K windows;
- the wide output grid provides fewer than eight waves.

SM120, other SM12x minors, other SM counts, M above 16, short reductions,
narrow attention grids, and already-saturated output grids retain the generic
selector.

## Performance

Measured on RTX Spark (SM121, 48 SMs) as end-to-end ORT `Run()` latency with an
explicit shared CUDA stream. Each cell includes three alternating process
rounds, with 100 warmups and 1,000 measured calls per arm per round.

| Shape | M | Existing | Proposed | Latency reduction |
|---|---:|---:|---:|---:|
| 17408x5120 | 1 | 0.2416 ms | 0.2005 ms | 16.9% |
| 17408x5120 | 8 | 0.3594 ms | 0.3258 ms | 9.3% |
| 17408x5120 | 16 | 0.3634 ms | 0.3336 ms | 8.0% |
| 5120x17408 | 1 | 0.2196 ms | 0.2099 ms | 3.9% |
| 5120x17408 | 8 | 0.2260 ms | 0.2136 ms | 5.4% |
| 5120x17408 | 16 | 0.2283 ms | 0.2182 ms | 4.6% |
| 24512x5120 | 1 | 0.3276 ms | 0.2711 ms | 17.2% |
| 24512x5120 | 8 | 0.4469 ms | 0.3982 ms | 11.2% |

Negative controls established the required exclusions:

| Excluded regime | Existing | Forced `16/1` | Impact |
|---|---:|---:|---:|
| 7168x5120, M=8 attention | 0.1660 ms | 0.1788 ms | 7.7% slower |
| 5120x7168, M=8 attention | 0.1652 ms | 0.1851 ms | 12.0% slower |
| 17408x2048, M=8 short K | 0.0480 ms | 0.0624 ms | 30.6% slower |
| 5120x8192, M=8 narrow grid | 0.1745 ms | 0.1879 ms | 7.7% slower |
| 17408x5120, M=32 | 0.2453 ms | 0.3977 ms | 62.1% slower |
| 5120x17408, M=32 | 0.2964 ms | 0.4005 ms | 35.2% slower |

No-override runs tracked the expected forced arm for every qualified and
excluded case, confirming that the compiled selector applies the intended
schedule. Randomized activations and varying block scales produced identical
natural/expected output and baseline/proposed output within one FP16 ULP.

At the exact eight-wave boundary (`N=24576, K=5120`), forced `16/1` was still
11% to 17% faster. The selector deliberately retains the generic schedule
there rather than extrapolating the optimization beyond the conservative
less-than-eight-wave rule.

## Testing

- Release ARM64 CUDA build with CUDA 13.4 for SM120/SM121
- `onnxruntime_provider_test
  --gtest_filter=MatMulBlockQuantizedFp4WeightOpTest.*`
  (19 selector and FP16/BF16 operator tests passed)
- Exact and ragged KSplit16 numerical coverage at K=2048 and K=2304
- Selector boundaries at M=16/17/32, 47/48/49/64/65 SMs, SM120/SM121, both
  K-window floors, narrow attention grids, and the eight-wave output-grid
  threshold
