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
built successfully as `sglang-fp6:research-20260913`; the GPU sweep is pending server warmup.

## Dense INT8 prefill (#3)

`patches/apply_research_overlays.py` adds opt-in `SGLANG_W8_TIMING=1` logging around the
W8A16 linear path. It labels each call `gemv` or `prefill` and records M/N/K and CUDA elapsed
milliseconds. The default is off and no production flag is promoted. The same image can be
restarted with the flag for a 1K/4K/16K/64K prefill sweep.

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
