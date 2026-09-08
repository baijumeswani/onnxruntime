# Follow-up review for FP8 GEMV KSplit32 scheduling

Pull request: [microsoft/onnxruntime#32409](https://github.com/microsoft/onnxruntime/pull/32409)

Reviewed head: `da238410bb8906c43fa662ee18f980053f4b1335`

## Verdict

The previous production and performance concerns are resolved. The production
CUDA implementation looks correct and appropriately qualified. Approval is
recommended after two small fixes make the forced KSplit32 test hermetic and
ensure no-GPU runs are reported as skipped.

## Remaining findings

### 1. The forced test can silently execute another path

File:
`onnxruntime/test/contrib_ops/matmul_block_scaled_fp8_test.cc:124-129`

The subprocess inherits environment variables that the test does not
explicitly override. If `ORT_FP8_GEMV_MMA=0`, the test uses the scalar GEMV
path. If `ORT_FP8_GEMV_MAX_M` is below 8, the operator can use the
dequantize/GEMM path. Both paths can produce the expected output without
executing the KSplit32 kernel that this test is intended to cover.

Add these values to the child environment:

```cpp
{"ORT_FP8_GEMV_MMA", "1"},
{"ORT_FP8_GEMV_MAX_M", "32"},
```

### 2. A no-GPU run reports a pass instead of a skip

File:
`onnxruntime/test/contrib_ops/matmul_block_scaled_fp8_test.cc:130-140`

CUDA availability is checked only in the child process. `GTEST_SKIP()` exits
that process successfully, so the parent sees `std::system(...) == 0` and
reports the outer test as passed even though KSplit32 was not exercised.

Check `HasCudaEnvironment(800)` before spawning the child so the outer test
correctly reports a skip when no usable CUDA device is available.

## Resolution of previous concerns

| Previous concern | Status |
|---|---|
| KSplit32 instantiated invalid `<32,4>` with 64 KiB shared memory | Resolved: dispatch directly instantiates only `<32,1>` |
| KSplit32 had only host-side selector coverage | Resolved, subject to the hermeticity fixes above: a forced FP16/BF16 execution test exists |
| Raw-N thresholds split shapes with identical launch geometry | Resolved: thresholds now use `ceil(N / 16)` output blocks |
| Already-saturated or lm-head grids could regress | Resolved: measurements through `N=248320` show 3.6% to 6.5% improvement, including 323 output waves |
| The heuristic was too broad across architectures and M values | Resolved: it is restricted to SM121, exactly 48 SMs, and M at most 8 |
| Performance rationale and negative controls were missing | Resolved: the PR description now documents methodology, boundaries, exclusions, kernel results, and whole-model results |

## Additional assessment

No new correctness, memory-safety, ABI/API, or production code-quality issue
was found in the rebased diff. The launch-geometry thresholds and expanded
selector tests cover the important boundary cases.

At review time, most CI jobs for the rebased head were still running. The
Optional Lint failure was unrelated to the change: the misspell action's
Debian image rejected expired repository metadata.
