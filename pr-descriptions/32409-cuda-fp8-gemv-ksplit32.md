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
  - at least 1,024 output blocks and 80 64-element K windows; or
  - at least 320 output blocks and 128 K windows.

Each FP8 GEMV output block owns 16 columns, so the thresholds use
`ceil(N / 16)` rather than raw N. This gives shapes with identical launch
geometry the same SM121 override: the wide boundary starts at N=16369, and the
long-reduction boundary starts at N=5105. The generic selector is unchanged,
preserving behavior on other devices.

SM120, other SM12x minors, other SM counts, M above 8, attention-sized
projections, small outputs, short reductions, and shapes immediately below
both output-block boundaries retain the generic selector.

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

The review specifically raised the possibility that KSplit32 could regress
already-saturated, vocabulary-sized output grids. Additional three-round GB10
negative-control measurements instead showed that KSplit32 remained beneficial
as N increased through the Qwen 3.8 lm-head dimension:

| Shape (M=1, K=5120) | Output blocks | Output waves on 48 SMs | KSplit8 | KSplit32 | Latency reduction |
|---|---:|---:|---:|---:|---:|
| N=32769 | 2,049 | 42.7 | 0.6836 ms | 0.6559 ms | 3.95% |
| N=65536 | 4,096 | 85.3 | 1.3174 ms | 1.2693 ms | 3.63% |
| N=131072 | 8,192 | 170.7 | 2.6234 ms | 2.5102 ms | 4.32% |
| N=248320 lm head | 15,520 | 323.3 | 5.2056 ms | 4.8773 ms | 6.54% |

Every cell used three alternating process rounds, 100 warmups, and 1,000
CUDA-event-timed ORT calls per arm. Full outputs matched with `atol=0.5` and
`rtol=1e-3`. A final-selector smoke run confirmed that the natural selector
matched forced KSplit32 at N=248320 (4.8339 ms natural versus 4.8683 ms
forced). Because the proposed regression did not reproduce and the benefit
persisted through more than 323 output waves, no arbitrary upper N cap is
applied.

Exact boundary cases were also positive across M=1, 2, 4, and 8:

| Boundary | Latency reduction |
|---|---:|
| 1,024 output blocks (measured at `N=16384`), 80 K windows | 1.2% to 2.5% |
| 320 output blocks (measured at `N=5120`), 128 K windows | 3.2% to 5.9% |

The broad initial heuristic was rejected after finding regressions at M=16/32,
on attention and small-output shapes, at 32 K windows, and around N=4096. The
final selector excludes all of those regimes. Negative controls at 1,023
output blocks (`N=16368`), 79 windows, 319 output blocks (`N=5104`), and 127
windows retain the generic schedule.

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
- The forced subprocess explicitly sets `ORT_FP8_GEMV_MMA=1` and
  `ORT_FP8_GEMV_MAX_M=32`, preventing inherited environment variables from
  silently selecting scalar GEMV or dequantize/GEMM
- CUDA availability is checked in the parent process, so no-GPU environments
  report the forced execution test as skipped rather than passed
- Selector boundaries for SM120/SM121, 47/48/49 SMs, M=8/9/16, both N floors,
  both K-window floors, launch-geometry-equivalent N values, measured
  vocabulary-sized outputs, and representative excluded shapes

The PR branch is rebased onto current ORT main at `da238410bb`. The full 13-test
FP8 suite passed before the conflict-free rebase. A fresh local Windows ARM64
build of current main is presently blocked in unrelated DeepGEMM SM90 code:
CUDA 13.4 rejects `__int128_t` and GNU-style inline PTX in
`deep_gemm_sm90.cu`. The FP8 KSplit32 files compile and test successfully on
the prior base, and the rebase did not conflict with any PR file.

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
> qualified low-M regions expressed in launch geometry: at least 1,024
> 16-column output blocks with 80 K windows, or at least 320 output blocks with
> 128 K windows. This removes the raw-N discontinuities: N=16369..16384 and
> N=5105..5120 now receive the same schedule as their identical launch grids.
> The generic selector remains unchanged on all other devices.
>
> On the final binary, representative gate/up, down, and wide-output kernels
> improved by 1.4% to 3.2%. Exact selected boundaries improved by 1.2% to 5.9%.
> I also measured the very-wide-output concern directly. At M=1, K=5120,
> forced KSplit32 improved latency by 3.95% at N=32769, 3.63% at N=65536,
> 4.32% at N=131072, and 6.54% at the N=248320 lm-head shape. The latter
> launches 15,520 blocks, or more than 323 waves on 48 SMs. The natural
> selector matched forced KSplit32, and all full outputs matched KSplit8.
> Since the proposed regression did not reproduce and the gain persisted
> through the model's lm head, I did not add an arbitrary upper cap.
>
> Negative controls immediately below each output-block boundary, plus
> attention, small output, short-K, N around 4096, and M=16/32 cases, all
> retained the generic path.
>
> Matched-acceptance Qwen 3.8 27B DFlash2 runs improved batch-1 decode by 1.0%
> to 1.7% across FP16/INT8 KV at 4K/16K. INT8 batch 4 was neutral at +0.16%
> median, with a -0.17% to +0.41% paired range. Generated tokens, acceptance,
> target-forward counts, and tokens per target forward matched between arms.
>
> The FP8 suite passes 13/13 tests in normal and shuffled order, including
> forced FP16/BF16 ragged KSplit32 coverage. The child process explicitly
> enables tensor-core GEMV with M up to 32, and the parent reports unavailable
> CUDA environments as skipped. No-override microbenchmarks followed the
> expected selector route and all full output tensors matched.

## Suggested response to follow-up review

> Addressed both remaining test issues. The forced child process now sets
> `ORT_FP8_GEMV_MMA=1` and `ORT_FP8_GEMV_MAX_M=32`, so inherited environment
> variables cannot silently route the test through scalar GEMV or
> dequantize/GEMM.
>
> `HasCudaEnvironment(800)` is now checked before the parent spawns the child.
> A machine without a usable CUDA device therefore reports the outer test as
> skipped instead of interpreting the child's successful skip exit as a pass.
