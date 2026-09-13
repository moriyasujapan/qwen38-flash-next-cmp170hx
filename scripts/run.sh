#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

NAME=${NAME:-qwen38-flashnext-fp6}
MODEL_DIR=${MODEL_DIR:-/opt/models/qwen38-flashnext-fp6int8}
PORT=${PORT:-18020}
IMAGE=${IMAGE:-sglang-fp6:sm80-fix1}
MAMBA_DTYPE=${MAMBA_DTYPE:-float16}

case "${1:-up}" in
  logs) exec docker logs -f "$NAME" ;;
  stop) docker rm -f "$NAME"; exit 0 ;;
  up) ;;
  *) echo "Usage: $0 [up|logs|stop]" >&2; exit 2 ;;
esac

[[ -n "${GPU_DEVICES:-}" ]] || {
  echo 'GPU_DEVICES is required. Copy .env.example to .env and set two GPU UUIDs.' >&2
  exit 1
}
[[ -d "$MODEL_DIR" ]] || { echo "Model directory not found: $MODEL_DIR" >&2; exit 1; }
[[ -f "$MODEL_DIR/ple/ngram.bin" ]] || { echo 'Missing ple/ngram.bin' >&2; exit 1; }

available_gib=$(awk '/MemAvailable/ {print int($2/1024/1024)}' /proc/meminfo)
(( available_gib >= 52 )) || {
  echo "MemAvailable ${available_gib} GiB < 52 GiB. Stop other memory-heavy jobs first." >&2
  exit 1
}

echo "MemAvailable: ${available_gib} GiB"
echo "GDN/Mamba temporal state dtype: ${MAMBA_DTYPE}"
docker rm -f "$NAME" 2>/dev/null || true

exec docker run -d --name "$NAME" \
  --gpus "\"device=${GPU_DEVICES}\"" \
  --ipc=host --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --restart=no \
  -p "${PORT}:${PORT}" \
  -v "${MODEL_DIR}:/model:ro" \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e SGLANG_PLE_SIDECAR=/model/ple/ngram.bin \
  -e NCCL_P2P_DISABLE=1 \
  -e NCCL_SHM_DISABLE=0 \
  -e SGLANG_FP6_DECODE_FP16=1 \
  -e SGLANG_FP6_GEMV3=1 \
  -e SGLANG_MAX_INPUT_LOGPROB_TOKENS=1024 \
  --entrypoint python3 \
  "$IMAGE" -m sglang.launch_server \
    --model-path /model \
    --served-model-name qwen3.8-flash-next \
    --host 0.0.0.0 --port "$PORT" \
    --tp-size 2 \
    --context-length "${CTX:-262144}" \
    --mem-fraction-static "${MEM:-0.93}" \
    --chunked-prefill-size "${CHUNKED_PREFILL:-4096}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS:-3}" \
    --attention-backend triton \
    --image-processor-backend pil \
    --linear-attn-backend triton \
    --max-mamba-cache-size "${MAMBA_SLOTS:-24}" \
    --mamba-ssm-dtype "$MAMBA_DTYPE" \
    --speculative-algorithm "${SPEC_ALGO:-NEXTN}" \
    --speculative-num-steps "${SPEC_STEPS:-1}" \
    --speculative-eagle-topk "${SPEC_TOPK:-1}" \
    --speculative-num-draft-tokens "${SPEC_DRAFT:-2}" \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --cuda-graph-backend-prefill disabled \
    --cuda-graph-backend-decode "${GRAPH_DECODE:-full}" \
    --cuda-graph-bs-decode 1 2 3 \
    --cuda-graph-max-bs-decode 3 \
    --enable-metrics \
    --trust-remote-code
