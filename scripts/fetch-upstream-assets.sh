#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO=${UPSTREAM_REPO:-Soomin33/Qwen3.8-Flash-Next-FP6-INT8}
REV=${UPSTREAM_REV:-42f2700ed675bab6176995e0c6a839240418bfba}
BASE_URL="https://huggingface.co/${REPO}/resolve/${REV}"

mkdir -p "${ROOT_DIR}/vendor"
curl -fL --retry 3 "${BASE_URL}/patches/sglang-fp6.patch?download=true" \
  -o "${ROOT_DIR}/vendor/sglang-fp6.patch"
curl -fL --retry 3 "${BASE_URL}/LICENSE?download=true" \
  -o "${ROOT_DIR}/vendor/MODEL-LICENSE"

test -s "${ROOT_DIR}/vendor/sglang-fp6.patch"
test -s "${ROOT_DIR}/vendor/MODEL-LICENSE"
printf 'Fetched upstream assets from %s@%s\n' "$REPO" "$REV"
