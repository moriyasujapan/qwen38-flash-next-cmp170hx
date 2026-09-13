# Research roadmap

This branch records the six follow-up optimization tracks requested after the initial SM80
baseline. Every experiment must preserve the command, runtime configuration, raw result,
interpretation, and rollback path. Negative results are first-class results.

## Tracking issues

| Track | Issue | First measurement | Exit gate |
|---|---|---|---|
| Workload-aware MTP tuning | [#1](https://github.com/moriyasujapan/qwen38-flash-next-cmp170hx/issues/1) | API streaming + Prometheus counter deltas | recommendation across repeatable workload/config matrix |
| SM80 split-KV verify attention | [#2](https://github.com/moriyasujapan/qwen38-flash-next-cmp170hx/issues/2) | isolated kernel correctness and latency sweep | winning patch or documented negative result |
| Dense INT8 prefill | [#3](https://github.com/moriyasujapan/qwen38-flash-next-cmp170hx/issues/3) | phase timing at 1K/4K/16K/64K | bottleneck attribution plus A/B |
| Prefill/decode interference | [#4](https://github.com/moriyasujapan/qwen38-flash-next-cmp170hx/issues/4) | mixed long-prefill and streaming-decode load | interactive and throughput profiles |
| DFlash2-style context drafting | [#5](https://github.com/moriyasujapan/qwen38-flash-next-cmp170hx/issues/5) | offline context-match upper bound | prototype or explicit blocker |
| PLE INT8/INT4 | [#6](https://github.com/moriyasujapan/qwen38-flash-next-cmp170hx/issues/6) | sampled quantization error and lookup bandwidth | quality/bandwidth decision and runtime plan |

## Experiment rules

1. Keep the production launcher unchanged until an A/B passes.
2. Use a new image tag for kernel/runtime changes.
3. Record exact base image digest, model revision, runtime flags, and GPU state.
4. Use unique prompt nonces for cold-prefix measurements.
5. Separate TTFT, decode throughput, end-to-end throughput, and aggregate throughput.
6. Record MTP verify-call counters; do not treat the last log-line acceptance gauge as a run average.
7. Mark a run contaminated when another request reaches the server during its measurement window.
8. Run short deterministic quality checks and long-context needle retrieval before promotion.
9. Give every change an environment-variable kill switch or a one-command rollback.
10. Commit raw data and the conclusion, including failures.

## Promotion flow

```text
hypothesis
  -> isolated measurement
  -> experimental patch/image
  -> same-protocol A/B
  -> quality and soak checks
  -> documented rollback
  -> merge candidate
```

The two CMP 170HX cards are already occupied by the serving model, so a second full model
instance cannot coexist on another port. Instrumentation-only tests can use the running server.
Configuration and kernel A/Bs require a controlled restart and are logged as service-impacting
experiments.
