# syntax=docker/dockerfile:1
ARG BASE_IMAGE=lmsysorg/sglang:qwen38flashnext@sha256:5ae5816783d58e2e56e84d2e863f5441425056f500b7fbd7448c4aae017a2521
FROM ${BASE_IMAGE}

COPY vendor/sglang-fp6.patch /tmp/sglang-fp6.patch
COPY patches/logprob_guard.py patches/apply_logprob_guard.py /tmp/

RUN set -eux; \
    cd /sgl-workspace/sglang; \
    git apply --reject /tmp/sglang-fp6.patch || true; \
    test -f python/sglang/srt/managers/scheduler.py.rej; \
    test -f python/sglang/srt/managers/tokenizer_manager.py.rej; \
    test -f python/sglang/srt/model_executor/runner_backend/breakable_cuda_graph_backend.py.rej; \
    test "$(find . -name '*.rej' | wc -l)" -eq 3; \
    sed -i 's/batch_result\.copy_done = self\.device_module\.Event()/batch_result.copy_done = self.device_module.Event(blocking=True)/' \
      python/sglang/srt/managers/scheduler.py; \
    grep -q 'Event(blocking=True)' python/sglang/srt/managers/scheduler.py; \
    python3 /tmp/apply_logprob_guard.py; \
    test "$(grep -c '_check_input_logprob_span(' python/sglang/srt/managers/tokenizer_manager.py)" -ge 2; \
    for f in \
      python/sglang/srt/layers/quantization/fp6.py \
      python/sglang/srt/layers/quantization/w8_dense.py \
      python/sglang/kernels/ops/moe/fp6_moe_kernels.py \
      python/sglang/kernels/ops/attention/qsa_packed_decode.py; do \
        test -f "$f"; \
    done; \
    find . -name '*.rej' -delete; \
    python3 -c "import sglang.srt.managers.tokenizer_manager as t; assert callable(t._check_input_logprob_span)"
