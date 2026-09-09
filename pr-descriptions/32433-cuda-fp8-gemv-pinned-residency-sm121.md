# SM121 qualification for FP8 GEMV pinned residency

Pull request: [microsoft/onnxruntime#32433](https://github.com/microsoft/onnxruntime/pull/32433)

Qualified head: `7cba02a1f6a5ea7d298b6fd9424b52e9dee3b206`

## Verdict

**Approval is recommended after required CI is green.**

The residency-pinned KSplit16 kernel is measurably positive on a 48-SM SM121
GB10 when a matrix falls inside the PR's dispatch window. The strongest and
most stable batch-1 decode results show approximately **8-10% lower FP8 GEMV
operator latency** at M=1 for Qwen-like K dimensions.

The result should not be described as an end-to-end Qwen 3.8 27B improvement
on RTX Spark. None of that model's FP8 projection dimensions enters the
SM-count-scaled residency window on this 48-SM GPU, so this specific model is
expected to remain neutral.

## Qualified system

- GPU: NVIDIA GB10 / RTX Spark
- Compute capability: SM121
- SM count: 48
- CUDA: 13.4
- Model inspected: Qwen 3.8 27B NVFP4, INT8 KV cache, DFlash2
- Workload policy: batch size 1, greedy target sampling, DFlash2 width 7

## Dispatch applicability

The PR adds a second KSplit16 FP8 GEMV entry point with:

```cpp
__launch_bounds__(512, 3)
```

The pinned entry point is selected only when:

```text
compute capability >= 8.9
KSplit == 16
M <= 8
2 * SM_count < ceil(N / 16) <= 3 * SM_count
```

For 48 SMs, the eligible output grid is 97-144 blocks, corresponding to:

```text
1537 <= N <= 2304
```

KSplit8, KSplit32, M greater than 8, pre-SM89 devices, and grids outside this
interval continue to use the ordinary kernel.

## Qwen 3.8 27B applicability

The target and drafter ONNX graphs were enumerated for
`MatMulBlockQuantizedFp8Weight` output dimensions.

| Output dimension N | Target nodes | Drafter nodes | Eligible on 48-SM SM121 |
|---:|---:|---:|:---:|
| 1,024 | 32 | 0 | No |
| 5,120 | 72 | 0 | No |
| 6,144 | 48 | 0 | No |
| 10,240 | 48 | 0 | No |
| 12,288 | 16 | 0 | No |
| 17,408 | 16 | 0 | No |
| 248,320 | 1 | 1 | No |

None of the 233 target FP8 nodes or the drafter FP8 lm-head falls in the
1,537-2,304 window. The new kernel therefore cannot be selected by this model
on the 48-SM RTX Spark.

This differs from devices such as H200, RTX 4090, and RTX 5060 Ti because the
eligible N interval scales with SM count. Their intervals intersect model
dimensions that the GB10 interval does not.

## Matched SM121 performance qualification

Two binaries were built from the same PR head:

- **Baseline:** residency dispatch locally disabled, using the ordinary
  KSplit16 kernel.
- **PR:** unmodified residency predicate and pinned kernel.

Only the CUDA provider differed. The ORT host and provider-shared DLLs were
identical. Each arm ran in a separate process on an explicit CUDA stream with
100 warmups and 1,000 CUDA-event-timed calls. Order alternated across three
rounds.

| N | K | M | Dispatch | Baseline p50 (ms) | PR p50 (ms) | Median latency reduction | Three rounds |
|---:|---:|---:|:---:|---:|---:|---:|---|
| 1,536 | 5,120 | 1 | Lower control | 0.026912 | 0.026912 | +0.24% | -1.08%, +0.48%, +0.24% |
| 1,537 | 5,120 | 1 | Pinned | 0.029312 | 0.026976 | **+7.97%** | +7.44%, +8.95%, +7.97% |
| 2,048 | 5,120 | 1 | Pinned | 0.032208 | 0.029152 | **+9.49%** | +9.49%, +10.00%, +9.19% |
| 2,048 | 5,120 | 8 | Pinned | 0.157312 | 0.149280 | **+5.11%** | +3.02%, +5.11%, +7.03% |
| 2,304 | 5,120 | 1 | Pinned | 0.032480 | 0.029408 | **+9.52%** | +9.52%, +9.46%, +9.63% |
| 2,305 | 5,120 | 1 | Upper control | 0.032576 | 0.032416 | -0.20% | -0.29%, -0.20%, +0.69% |
| 2,048 | 6,144 | 1 | Pinned | 0.034080 | 0.031168 | **+8.63%** | +8.84%, +7.51%, +8.63% |
| 2,048 | 6,144 | 8 | Pinned | 0.158784 | 0.157216 | +0.99% | +0.73%, +0.99%, +4.82% |
| 2,048 | 17,408 | 1 | Pinned | 0.167760 | 0.163984 | +2.11% | +4.64%, -0.24%, +2.11% |
| 2,048 | 17,408 | 8 | Pinned | 0.168576 | 0.165952 | +1.56% | +2.95%, -3.27%, +1.56% |

The intended M=1 decode case is positive and repeatable at K=5,120 and
K=6,144. The exact lower and upper boundary controls are neutral, showing that
the measured effect follows the residency predicate rather than a general
binary difference.

The K=17,408 cases have a smaller absolute difference and noisier individual
rounds. Their positive medians are supporting data, not headline evidence.

## Correctness

The native build passed all 15
`MatMulBlockQuantizedFp8WeightOpTest.*` tests on SM121, including:

- Predicate boundary coverage.
- Forced execution of the pinned-residency kernel.
- KSplit selection and forced KSplit32 coverage.
- FP16 and BF16 execution.
- Speculative-decode shapes.
- Lane ownership, scratch tiling, and zero-K behavior.

Every paired microbenchmark compared the full FP16 output tensor. The maximum
absolute difference was `0.0` for all 30 baseline/PR pairs.

## End-to-end qualification note

A targeted 4K Qwen run was attempted, but the local GenAI/current-ORT provider
combination crashed during integration, including after rebuilding GenAI
against the PR ORT SDK. No timing from those attempts is included.

The graph census provides the relevant model-level conclusion independently:
no Qwen FP8 matrix can satisfy the new predicate on this device. A measurable
Qwen delta on 48-SM GB10 would indicate an uncontrolled binary or harness
difference rather than execution of the pinned kernel.

## Approval rationale

Approval is justified because:

1. The dispatch is narrowly limited to the occupancy interval it addresses.
2. The kernel is measurably positive on real SM121 hardware when selected.
3. Exact dispatch-boundary controls remain neutral.
4. Outputs are bit-identical across the matched baseline and PR binaries.
5. The full focused FP8 operator suite passes on SM121.
6. Upstream H200, SM120, and SM89 results show the same mechanism benefits
   model shapes when their device-specific interval intersects those shapes.

Recommended qualification wording:

> PR #32433 improves eligible FP8 GEMV shapes on SM121. It is neutral for
> Qwen 3.8 27B on the 48-SM RTX Spark because none of that model's FP8
> matrices falls inside the residency-pinned dispatch window.

At qualification time, the PR head had 69 successful checks, 16 checks still
in progress, and failures in Android NNAPI, React Native Android, and OpenVINO
jobs. Approval should be submitted after required CI is green or those
failures are confirmed unrelated.
