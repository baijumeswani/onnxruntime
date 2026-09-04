# Add FP8 GEMV KSplit32 scheduling for the 48-SM SM121 GPU

Pull request: [microsoft/onnxruntime#32409](https://github.com/microsoft/onnxruntime/pull/32409)

## Description

Add a KSplit32 tensor-core GEMV specialization for
`MatMulBlockQuantizedFp8Weight` and select it only for qualified low-M decode
shapes on the 48-SM SM121 GB10 GPU.

The final selector preserves the generic schedule unless all of these
conditions hold:

- compute capability is exactly 12.1;
- the device has exactly 48 SMs;
- the original request has M at most 8;
- the shape is either:
  - a wide output with `N >= 16384` and at least 80 64-element K windows; or
  - a long reduction with `N >= 5120` and at least 128 K windows.

SM120, other SM12x minors, other SM counts, M above 8, attention-sized
projections, small outputs, short reductions, and shapes immediately outside
both boundaries retain the generic selector.

KSplit32 dispatches directly to the only qualified launch,
`<KSplit=32, MTiles=1>`. This avoids instantiating unreachable `<32,2>` and
`<32,4>` kernels; `<32,4>` would require 64 KiB of static shared memory.
Requests above M=8 are recursively tiled with GB10 KSplit32 tuning disabled, so
the original request size cannot be lost during dispatch.

## Kernel performance

Measured on RTX Spark (SM121, 48 SMs) as end-to-end ORT `Run()` latency with an
explicit shared CUDA stream. Each final-selector cell used three alternating
process rounds, with 100 warmups and 1,000 measured calls per arm per round.

| Shape | M | Generic | KSplit32 | Latency reduction |
|---|---:|---:|---:|---:|
| 17408x5120 gate/up | 1 | 0.3674 ms | 0.3611 ms | 1.7% |
| 17408x5120 gate/up | 8 | 0.4901 ms | 0.4792 ms | 2.3% |
| 5120x17408 down | 1 | 0.3771 ms | 0.3684 ms | 2.3% |
| 5120x17408 down | 8 | 0.3771 ms | 0.3696 ms | 2.0% |
| 32768x5120 wide output | 1 | 0.6704 ms | 0.6552 ms | 2.2% |
| 32768x5120 wide output | 8 | 0.7986 ms | 0.7729 ms | 3.2% |

Exact boundary cases were also positive across M=1, 2, 4, and 8:

| Boundary | Latency reduction |
|---|---:|
| `N=16384`, 80 K windows | 1.2% to 2.5% |
| `N=5120`, 128 K windows | 3.2% to 5.9% |

The broad initial heuristic was rejected after finding regressions at M=16/32,
on attention and small-output shapes, at 32 K windows, and around N=4096. The
final selector excludes all of those regimes. Negative controls at N=16383,
79 windows, N=5119, and 127 windows also retain the generic schedule.

No-override runs followed the expected schedule for every selected and
excluded case. Randomized activations and varying block scales matched the
generic output in every compared full tensor.

## Whole-model performance

The exact final CUDA plugin was compared with GB10 tuning disabled versus
enabled. Each value is the median paired gain from three alternating process
pairs. The code-copy workload keeps generated tokens, speculative acceptance,
tokens per target forward, and target-forward counts identical between arms.

| Qwen 3.8 27B DFlash2 workload | Decode improvement |
|---|---:|
| FP16 KV, batch 1, 4K | 1.19% |
| FP16 KV, batch 1, 16K | 1.03% |
| INT8 KV, batch 1, 4K | 1.73% |
| INT8 KV, batch 1, 16K | 1.05% |
| INT8 KV, batch 4, 4K | 0.16% |

Batch 4 is effectively neutral, with paired changes from -0.17% to +0.41%,
while every batch-1 pair improved. Median TTFT did not regress in these
matched-acceptance comparisons.

## Testing

- Release Windows ARM64 CUDA 13.4 build for SM120/SM121
- Dedicated CUDA plugin EP build for SM120/SM121
- `onnxruntime_provider_test
  --gtest_filter=MatMulBlockQuantizedFp8WeightOpTest.*`
  (13 selector and FP16/BF16 operator tests passed)
- The same 13 tests passed with `--gtest_shuffle --gtest_random_seed=2`
- Forced FP16 and BF16 ragged KSplit32 numerical coverage at M=8, N=17,
  K=2112 in a fresh subprocess
- Selector boundaries for SM120/SM121, 47/48/49 SMs, M=8/9/16, both N floors,
  both K-window floors, and representative excluded shapes

## Suggested response to review

Review comment:
[discussion_r3924587719](https://github.com/microsoft/onnxruntime/pull/32409#discussion_r3924587719)

> Good catch. KSplit32 now dispatches directly to `<32,1>`, so the unreachable
> `<32,2>` and `<32,4>` kernels are not instantiated. The selector and runtime
> guard keep KSplit32 at M <= 8; `<32,4>` would otherwise require 64 KiB of
> static shared memory.
>
> I also added a forced ragged KSplit32 numerical test for FP16 and BF16. It
> runs in a fresh subprocess so the environment override is applied before any
> GEMV static initialization, including when the suite is shuffled.

## Suggested qualification follow-up

> I narrowed the selector to the measured 48-SM SM121 GB10 device and to two
> qualified low-M regions: N >= 16384 with at least 80 K windows, or N >= 5120
> with at least 128 K windows. All other devices and shapes keep the generic
> selector.
>
> On the final binary, representative gate/up, down, and wide-output kernels
> improved by 1.4% to 3.2%. Exact selected boundaries improved by 1.2% to 5.9%.
> Negative controls immediately outside each boundary, plus attention, small
> output, short-K, N around 4096, and M=16/32 cases, all retained the generic
> path.
>
> Matched-acceptance Qwen 3.8 27B DFlash2 runs improved batch-1 decode by 1.0%
> to 1.7% across FP16/INT8 KV at 4K/16K. INT8 batch 4 was neutral at +0.16%
> median, with a -0.17% to +0.41% paired range. Generated tokens, acceptance,
> target-forward counts, and tokens per target forward matched between arms.
>
> The FP8 suite passes 13/13 tests in normal and shuffled order, including
> forced FP16/BF16 ragged KSplit32 coverage. No-override microbenchmarks followed
> the expected selector route and all full output tensors matched.
