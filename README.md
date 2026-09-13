# Qwen3.8-Flash-Next on 2× CMP 170HX

[日本語版 / Japanese](README_ja.md)

Reproducible tooling, an SM80-patched SGLang image, benchmarks, and stability checks for
serving `Soomin33/Qwen3.8-Flash-Next-FP6-INT8` on two NVIDIA CMP 170HX 64 GB GPUs.

This is not a stock-SGLang recipe for a generic FP8 checkpoint. It applies the model's
FP6/INT8, PLE-offload, and QSA patches to a pinned SGLang image and adds guards required by
the tested image revision.

## Measured results

Measured on 2026-09-13. These are results from one machine, not guaranteed performance.

| Test | Result |
|---|---:|
| Single-stream decode while hardware Power Brake was active | 24.2 tok/s median |
| Single-stream decode after resolving B30/PWRBRK# | **67.9 tok/s median** |
| Improvement | **2.81×** |
| FP16 GDN state, concurrency 1 aggregate decode | 64.25 tok/s median |
| FP16 GDN state, concurrency 2 aggregate decode | 99.02 tok/s median |
| FP16 GDN state, concurrency 3 aggregate decode | **119.18 tok/s median** |
| Cold TTFT for a ~9.9K-token prefix | 9.201 s |
| RadixCache-hit TTFT for the same prefix | **0.612 s** |
| RadixCache speedup | **15.02×** |
| KV token capacity, FP32 → FP16 GDN state | 415,296 → **481,344** |

`quick_bench.py` and `bench_matrix.py` use different prompts and output lengths. The 67.9
tok/s result is the median of three 1,024-token generations; 64.25 tok/s is the median of
three 512-token matrix runs with unique prompts. Do not treat them as the same test series.

See [detailed results](docs/RESULTS.md) and [tuning notes](docs/TUNING.md).

## Tested hardware profile

- 2× CMP 170HX with 64 GB unlocked VRAM per card
- Approximately 96 GB of system RAM; at least 52 GiB `MemAvailable` before launch
- At least 160 GB of free model storage
- Docker and NVIDIA Container Toolkit
- PCIe Gen2 x4 per card in the tested machine
- No NVLink and no CUDA P2P between the two cards

The checkpoint consists of approximately 97.6 GiB of weight shards and a 47.7 GiB PLE FP8
sidecar. The local model directory occupies about 146 GiB. PLE is held in pinned host memory,
so unlimited container memlock is required.

## Check hardware Power Brake first

If decode performance is stuck in the 20 tok/s range, check this before changing kernels:

```bash
nvidia-smi -q | grep -A1 "HW Power Brake Slowdown"
```

An unexpected `Active` state may mean the platform is asserting `PWRBRK#` on PCIe edge pin
B30. Resolving it improved this machine from 24.2 to 67.9 tok/s.

This is motherboard-dependent. Do nothing if the state is `Not Active`. Masking a physical
edge-connector pin can damage the contact or hardware if performed incorrectly. This
repository does not recommend or warranty a physical modification.

## Precision and runtime map

| Component | Format or path |
|---|---|
| Routed and MTP experts | Packed FP6 e2m3, custom Triton kernels |
| Selected dense projections | Symmetric INT8, group 32, FP16 scales |
| Numerically sensitive components | BF16 |
| PLE n-gram table | FP8 e4m3 sidecar in pinned host RAM |
| KV cache | BF16 |
| GDN/Mamba recurrent state | FP16 |
| QSA and linear attention | Triton |
| Speculative decoding | NEXTN, one step, two draft tokens |

FP8 and FP6 are not executed as native SM80 formats. PLE uses FP8 as a compact lookup
sidecar, while packed FP6 expert weights are decoded inside specialized kernels.

## Installation

### 1. Clone this repository and download the model

```bash
git clone https://github.com/moriyasujapan/qwen38-flash-next-cmp170hx.git
cd qwen38-flash-next-cmp170hx

python3 -m pip install -U huggingface_hub
MODEL_DIR=/opt/models/qwen38-flashnext-fp6int8 ./scripts/download-model.sh
```

The model and its patch are governed by the Qwen Community License 1.0. Review the
[model page](https://huggingface.co/Soomin33/Qwen3.8-Flash-Next-FP6-INT8) and license before
use.

### 2. Fetch the pinned upstream patch and build the image

```bash
./scripts/fetch-upstream-assets.sh
docker build -t sglang-fp6:sm80-fix1 .
```

The fetch script pins Hugging Face revision
`42f2700ed675bab6176995e0c6a839240418bfba`. If you update either the model patch or SGLang
base image, re-audit every rejected hunk before serving requests.

### 3. Configure the machine-specific values

```bash
cp .env.example .env
nvidia-smi -L
$EDITOR .env
./scripts/check-host.sh
```

Use GPU UUIDs instead of numeric indices in `GPU_DEVICES`; index ordering can change after a
driver reload or when another GPU is added.

### 4. Start and warm up the server

```bash
./scripts/run.sh
./scripts/wait-ready.sh
```

The first launch includes weight loading, JIT compilation, and CUDA Graph capture. On the
tested system, cold API startup took about 16 minutes and the full warm-up completed after
about 18 minutes. Restart time can be much shorter while weights remain in the OS page cache.

### 5. Test the OpenAI-compatible endpoint

```bash
curl http://127.0.0.1:18020/v1/models

curl http://127.0.0.1:18020/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3.8-flash-next",
    "messages":[{"role":"user","content":"Hello"}],
    "max_tokens":128,
    "chat_template_kwargs":{"enable_thinking":false}
  }'
```

## Important runtime choices

- Tensor parallel size 2
- PLE offload with `--ulimit memlock=-1`
- P2P explicitly disabled; NCCL shared-memory transport retained
- FP6 GEMV3 expert path and FP16-placement decode enabled
- GDN/Mamba temporal state stored as FP16
- Production MTP profile: `NEXTN`, one speculative step, two draft tokens
- Production QSA profile: Triton packed decode, `BLOCK_S=128`, `WARPS=8`
- Dense W8 timing instrumentation disabled (`SGLANG_W8_TIMING=0`)

These are the recommended production defaults from the research branch. The QSA sweep found
`BLOCK_S=256/WARPS=8` marginally faster for one synthetic shape, but not enough to justify a
global default change. PLE INT8/INT4 candidates remain experimental; production uses the raw
FP8 sidecar.
- Chunked prefill size 4,096
- Maximum three running requests
- Decode CUDA Graphs captured for batch sizes 1, 2, and 3
- Prefill CUDA Graph disabled
- RadixCache enabled
- CPU/PIL image preprocessing
- Input-logprob scored span capped at 1,024 positions to prevent full-server OOM

The upstream patch was authored against SGLang commit `7c66045d71`, while the pinned image
contains `g593134d17`. Three hunks are expected to reject. The Docker build verifies the exact
reject set, ports the scheduler event change and input-logprob guard, checks all core FP6/QSA
files, and imports the patched tokenizer manager. An unexpected reject fails the build.

## Benchmarks

Quick single-stream test:

```bash
python3 bench/quick_bench.py
```

Concurrency, prefill, and cold/warm RadixCache matrix:

```bash
python3 bench/bench_matrix.py \
  --concurrency 1 2 3 \
  --rounds 3 \
  --output-tokens 512 \
  --prefill-tokens 1024 4096 16384 \
  --radix-tokens 8192
```

The matrix adds a unique nonce to cold prompts so a previous run cannot accidentally satisfy
them from RadixCache.

Run short deterministic checks and long-context needle retrieval after changing the GDN state
dtype:

```bash
python3 bench/gdn_quality.py fp16 --needle-tokens 16384
```

For an FP32/FP16 A/B:

```bash
MAMBA_DTYPE=float32 ./scripts/run.sh
./scripts/wait-ready.sh
python3 bench/gdn_quality.py fp32

MAMBA_DTYPE=float16 ./scripts/run.sh
./scripts/wait-ready.sh
python3 bench/gdn_quality.py fp16
```

## OpenWebUI

[openwebui/compose.override.yml](openwebui/compose.override.yml) connects an OpenWebUI Docker
service to SGLang through the Linux host's published port:

```bash
docker compose -f docker-compose.yml \
  -f qwen38-flash-next-cmp170hx/openwebui/compose.override.yml up -d
```

If OpenWebUI already has connections saved through its admin UI, its persistent database
configuration may override environment variables. Update the saved OpenAI-compatible URL to
`http://host.docker.internal:18020/v1` and use served model ID `qwen3.8-flash-next`.

## Acknowledgements

- [QwenLM/Qwen3.8-Flash-Next](https://github.com/QwenLM/Qwen3.8-Flash-Next)
- [Soomin33/Qwen3.8-Flash-Next-FP6-INT8](https://huggingface.co/Soomin33/Qwen3.8-Flash-Next-FP6-INT8)
- [SGLang](https://github.com/sgl-project/sglang)
- [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090)
- [allover326/deepseek-v4-cmp170hx](https://github.com/allover326/deepseek-v4-cmp170hx)

The core FP6/INT8, PLE-offload, QSA, and MTP support comes from the patch distributed with
Soomin33's checkpoint. This repository provides audited integration, launch, warm-up,
benchmarking, and operational safeguards for the tested CMP 170HX setup.

## License

Original scripts and documentation in this repository are MIT-licensed. Model weights,
tokenizers, the downloaded SGLang patch, the base container image, and their dependencies
retain their respective upstream licenses. See [NOTICE.md](NOTICE.md).
