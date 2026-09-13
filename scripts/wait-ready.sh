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
PORT=${PORT:-18020}

for _ in $(seq 1 180); do
  status=$(docker inspect --format '{{.State.Status}}' "$NAME" 2>/dev/null || true)
  if [[ "$status" != running ]]; then
    echo "Container stopped: status=${status}" >&2
    docker logs --tail 80 "$NAME" 2>&1 || true
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null; then
    echo 'ENGINE_READY'
    python3 "${ROOT_DIR}/scripts/warmup.py" \
      --url "http://127.0.0.1:${PORT}/v1/chat/completions"
    echo 'READY'
    exit 0
  fi
  sleep 10
done

echo 'Timed out waiting for SGLang.' >&2
exit 1
