#!/usr/bin/env bash
set -euo pipefail

MODEL_REPO=${MODEL_REPO:-Soomin33/Qwen3.8-Flash-Next-FP6-INT8}
MODEL_REV=${MODEL_REV:-42f2700ed675bab6176995e0c6a839240418bfba}
MODEL_DIR=${MODEL_DIR:-/opt/models/qwen38-flashnext-fp6int8}

command -v hf >/dev/null || {
  echo 'The `hf` command is missing. Install it with: python3 -m pip install -U huggingface_hub' >&2
  exit 1
}

echo "About 146 GiB will be downloaded to: ${MODEL_DIR}"
hf download "$MODEL_REPO" --revision "$MODEL_REV" --local-dir "$MODEL_DIR"
test -f "$MODEL_DIR/config.json"
test -f "$MODEL_DIR/ple/ngram.bin"
echo 'Model download complete.'
