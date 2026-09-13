#!/usr/bin/env python3
"""Apply opt-in research instrumentation after the model-supplied SGLang patch.

The upstream patch is downloaded at build time, so these small idempotent source overlays
are kept separate and are guarded by environment variables at runtime.
"""
from pathlib import Path


ROOT = Path("/sgl-workspace/sglang")


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if new in text:
        return
    if old not in text:
        raise SystemExit(f"overlay context missing: {path}")
    path.write_text(text.replace(old, new, 1))


qsa = ROOT / "python/sglang/kernels/ops/attention/qsa_packed_decode.py"
replace_once(
    qsa,
    "import triton.language as tl\n",
    "import triton.language as tl\nimport os\n",
)
replace_once(
    qsa,
    "                      block_s=128, num_warps=8):\n    B, H, D = q.shape\n",
    "                      block_s=128, num_warps=8):\n"
    "    # Research-only knobs; true split-KV reduction is tracked separately.\n"
    "    block_s = int(os.environ.get(\"SGLANG_QSA_BLOCK_S\", str(block_s)))\n"
    "    num_warps = int(os.environ.get(\"SGLANG_QSA_WARPS\", str(num_warps)))\n"
    "    B, H, D = q.shape\n",
)

w8 = ROOT / "python/sglang/srt/layers/quantization/w8_dense.py"
replace_once(w8, "import os\n", "import os\nimport time\n")
replace_once(
    w8,
    "              bias: Optional[torch.Tensor] = None) -> torch.Tensor:\n    N, K = w.shape\n",
    "              bias: Optional[torch.Tensor] = None) -> torch.Tensor:\n"
    "    timing = os.environ.get(\"SGLANG_W8_TIMING\") == \"1\"\n"
    "    if timing:\n        torch.cuda.synchronize()\n        t0 = time.perf_counter()\n"
    "    N, K = w.shape\n",
)
replace_once(
    w8,
    "    if bias is not None:\n        out = out + bias\n    return out.reshape(*x.shape[:-1], N)\n",
    "    if bias is not None:\n        out = out + bias\n"
    "    if timing:\n"
    "        torch.cuda.synchronize()\n"
    "        print(f\"W8TIME M={M} N={N} K={K} path={'gemv' if _use_gemv(M, x2) else 'prefill'} ms={(time.perf_counter() - t0) * 1e3:.3f}\", flush=True)\n"
    "    return out.reshape(*x.shape[:-1], N)\n",
)
print("research overlays applied")
