#!/usr/bin/env python3
"""Microbenchmark the patched QSA packed-decode Triton kernel.

This measures tile/warp choices only. The current upstream kernel assigns one program to
each (batch, head), so this harness explicitly reports that it is a tile sweep, not a claim
that split-KV reduction has already been implemented.
"""
from __future__ import annotations

import argparse
import json
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; run inside the SGLang container")
    from sglang.kernels.ops.attention.qsa_packed_decode import qsa_packed_decode

    device = torch.device("cuda")
    q = torch.randn(args.batch, args.heads, args.dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(args.batch * args.seq, 1, args.dim, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    cu = torch.arange(0, (args.batch + 1) * args.seq, args.seq, device=device, dtype=torch.int32)
    ref = torch.empty_like(q)
    for b in range(args.batch):
        scores = (q[b].float() @ k[b * args.seq:(b + 1) * args.seq, 0].float().T) / (args.dim ** 0.5)
        ref[b] = (scores.softmax(-1) @ v[b * args.seq:(b + 1) * args.seq, 0].float()).to(q.dtype)

    records = []
    for block_s in (32, 64, 128, 256):
        for warps in (2, 4, 8):
            try:
                for _ in range(args.warmup):
                    qsa_packed_decode(q, k, v, cu, args.dim ** -0.5, block_s=block_s, num_warps=warps)
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.iters):
                    out = qsa_packed_decode(q, k, v, cu, args.dim ** -0.5, block_s=block_s, num_warps=warps)
                end.record()
                end.synchronize()
                err = (out.float() - ref.float()).abs().max().item()
                records.append({"block_s": block_s, "warps": warps, "ms": start.elapsed_time(end) / args.iters, "max_abs_error": err})
            except Exception as exc:
                records.append({"block_s": block_s, "warps": warps, "error": repr(exc)})
    result = {
        "schema": "qsa-tile-sweep-v1",
        "shape": {"batch": args.batch, "heads": args.heads, "dim": args.dim, "seq": args.seq},
        "kernel_note": "one (batch,head) program; tile sweep only, no split-KV reduction",
        "records": records,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
