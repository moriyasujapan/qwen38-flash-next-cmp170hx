#!/usr/bin/env python3
"""Synthetic W8A16 shape sweep for the dense INT8 prefill path."""
from __future__ import annotations

import argparse
import json

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()
    from sglang.srt.layers.quantization.w8_dense import w8_linear

    device = torch.device("cuda")
    K, N, group = 2560, 4096, 32
    w = torch.randint(-127, 128, (N, K), device=device, dtype=torch.int8)
    s = torch.rand(N, K // group, device=device, dtype=torch.float16) * 0.02
    records = []
    for m in (1, 16, 64, 256, 1024):
        x = torch.randn(m, K, device=device, dtype=torch.bfloat16)
        for _ in range(3):
            w8_linear(x, w, s, group)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            out = w8_linear(x, w, s, group)
        end.record()
        end.synchronize()
        ref = torch.matmul(x.float(), (w.float() * s.repeat_interleave(group, dim=1).float()).t())
        err = (out.float() - ref).abs()
        records.append({
            "M": m, "N": N, "K": K, "group": group,
            "ms": start.elapsed_time(end) / args.iters,
            "max_abs_error": float(err.max()),
            "mean_abs_error": float(err.mean()),
            "path": "gemv" if m <= 16 else "prefill",
        })
    result = {"schema": "w8-prefill-sweep-v1", "records": records}
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
