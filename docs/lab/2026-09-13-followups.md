# Lab notebook: follow-up optimization tracks

Date: 2026-09-13 UTC
Branch: `research/next-optimizations`

## Prefill/decode interference (#4)

The client harness `bench/mixed_workload.py` ran one synthetic long-prefill request and four
interactive streams against the frozen `chunked_prefill_size=4096`, overlap-scheduler-on
server. The run used a unique nonce and 500 context lines.

| metric | result |
|---|---:|
| interactive TTFT median | 8.897 s |
| interactive TTFT p95 | 9.969 s |
| long-prefill TTFT | 8.312 s |
| long-prefill end-to-end | 11.779 s |

This is a contention signal, not a standalone regression claim: there is no idle-server control
in this run and the harness reports character decode rate. The next A/B should repeat the exact
workload with chunk sizes 1024/2048/4096/8192 and with overlap scheduling disabled, recording
token-level ITL from Prometheus. Raw result:
[`mixed-chunk4096-overlap-on-20260913.json`](../../results/mixed-chunk4096-overlap-on-20260913.json).

## SM80 QSA verify attention (#2)

`bench/qsa_splitk_sweep.py` and the research image overlay expose
`SGLANG_QSA_BLOCK_S` and `SGLANG_QSA_WARPS`. The existing patched kernel uses one program per
(batch, head), so this is a tile/warp sweep and correctness check; it is explicitly **not** a
true split-KV reduction yet. A true split-KV version needs partial `(max, sum, value)` buffers
and a numerically stable reduction, with an additional workspace/capture audit. The image was
built successfully as `sglang-fp6:research-20260913`. On B=3, H=16, D=128, S=128, all 12
tile/warp choices had max absolute error `4.77e-7`; latency ranged from 0.1054 to 0.1542 ms.
The fastest tested choice was BLOCK_S=256/WARPS=8 at 0.1054 ms, effectively tied with the
128/4 default at 0.1056 ms. This does not justify changing the default without a real QSA shape
sweep. Raw result: [`qsa-sweep-b3-h16-d128-s128-20260913.json`](../../results/qsa-sweep-b3-h16-d128-s128-20260913.json).

## Dense INT8 prefill (#3)

`patches/apply_research_overlays.py` adds opt-in `SGLANG_W8_TIMING=1` logging around the
W8A16 linear path. It labels each call `gemv` or `prefill` and records M/N/K and CUDA elapsed
milliseconds. A synthetic K=2560, N=4096 sweep reported warmed kernel medians of 0.394 ms
(M=1), 0.394 ms (M=16), 0.407 ms (M=64), 0.411 ms (M=256), and 0.591 ms (M=1024); the
first call includes JIT compilation. These are isolated linear timings, not end-to-end prefill
speedups. The default is off and no production flag is promoted. Raw result:
[`w8-prefill-sweep-20260913.json`](../../results/w8-prefill-sweep-20260913.json).

## DFlash2-style context drafting (#5)

`bench/context_draft_upper_bound.py` measures an offline suffix/prefix token match. A control
case where the prompt ends in `return json` measured a two-token candidate; an unrelated
continuation measured zero. This is an upper bound only: it does not call the model, alter
NEXTN, or claim acceptance. Native DFlash2 integration remains blocked on a Flash-Next drafter
interface and a quality corpus.

## PLE INT8/INT4 (#6)

The PLE manifest is `raw_fp8_e4m3`, 320,001,536 rows × 160 bytes, 51,200,245,760 bytes.
`tools/ple_quality.py` decoded a contiguous 100,000-row sample and evaluated symmetric
per-row quantization:

| candidate | MAE | RMSE | mean cosine | 1% cosine | estimated sidecar |
|---|---:|---:|---:|---:|---:|
| INT8 | 4.46e-5 | 5.18e-5 | 0.999979 | 0.999959 | 51.20 GB |
| INT4 | 8.01e-4 | 9.45e-4 | 0.992922 | 0.987377 | 25.60 GB |

These are reconstruction metrics, not model-quality or lookup-bandwidth results. The current
runtime still consumes the FP8 raw sidecar. INT8 is a plausible next candidate; INT4 requires
needle/semantic quality tests and a packed lookup kernel before promotion. Raw result:
[`ple-quality-100k-20260913.json`](../../results/ple-quality-100k-20260913.json).
`tools/ple_quantize.py` now provides a bounded-memory converter for producing candidate sidecars
and row-scale files; it is intentionally not enabled by the launcher.

## Serving sanity after research image

After the controlled restart and first-use compilation, `bench/quick_bench.py` completed
without errors. The three 1,024-token decode runs were 68.7, 68.9, and 69.0 tok/s (median
68.9 tok/s); smoke TTFT was 1.09 s and smoke decode was 61.6 tok/s. This is close to the
previous 67.9 tok/s baseline and is not attributed to QSA/W8 changes without a matched A/B.
