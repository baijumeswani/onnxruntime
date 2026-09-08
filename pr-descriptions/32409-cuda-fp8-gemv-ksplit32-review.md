# Review feedback for FP8 GEMV KSplit32 scheduling

Pull request: [microsoft/onnxruntime#32409](https://github.com/microsoft/onnxruntime/pull/32409)

Reviewed head: `88ea12b05af3f534d652c994beebd0df5398a32a`

## Verdict

Do not approve yet. The implementation appears correct, and the earlier
KSplit32 shared-memory instantiation issue is fixed, but the performance
selector is broader than the demonstrated qualification.

## Finding: KSplit32 applies to already saturated output grids

File:
`onnxruntime/contrib_ops/cuda/math/matmul_block_scaled_fp8_tiling.h:27-30`

The wide-output condition has only lower bounds on N and K windows. On the
targeted SM121/48-SM device, it therefore changes shapes such as
`M=1, N=248320, K=5120` from KSplit8 to KSplit32. That shape launches 15,520
output blocks, more than 323 waves across 48 SMs, so it does not lack
inter-block parallelism.

KSplit32 instead increases each block from 256 to 1,024 threads, expands its
reduction storage from 4 KiB to 16 KiB, and performs a 32-way rather than
8-way reduction. This is a plausible regression for vocabulary/lm-head-sized
outputs and conflicts with the generic selector's rationale that wide N
should use fewer warps.

Before approval, do one of the following:

- cap the override based on output-block or grid waves;
- restrict it to the specifically qualified shapes; or
- provide GB10 negative-control measurements showing that KSplit32 remains
  beneficial for very wide outputs such as the model's lm-head dimension.

The threshold boundaries also need justification. `N=16383` and `N=16384`
both launch 1,024 blocks, but the selector jumps from KSplit8 to KSplit32.
Likewise, `N=5119` and `N=5120` both launch 320 blocks but switch from
KSplit16 to KSplit32. A launch-geometry-based threshold would generalize more
coherently unless these are intentional exact model-shape boundaries.

## Correctness and code quality

The CUDA implementation itself looks sound:

- KSplit32 instantiates only `<32,1>`, avoiding the invalid `<32,4>`
  specialization and its 64 KiB static shared-memory requirement.
- The reachable KSplit32 kernel uses 16 KiB of shared memory.
- Surplus warps and ragged K windows are handled correctly.
- A real forced KSplit32 test covers FP16 and BF16 execution.
- The forced KSplit32 test and the complete 13-test FP8 operator suite passed
  in Linux CUDA CI.

At review time, the PR description was empty, so it did not document benchmark
methodology, positive results, or negative-control coverage for the selected
region. The failed WebGPU plugin check and cancelled web build also remained
unresolved.
