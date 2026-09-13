#!/usr/bin/env python3
"""Sample PLE FP8 rows and estimate INT8/INT4 sidecar error and size.

The complete PLE table is tens of gigabytes, so this tool uses deterministic mmap sampling
and never materializes the sidecar. It evaluates symmetric per-row quantization, which is a
quality/bandwidth screen rather than a runtime implementation.
"""
from __future__ import annotations

import argparse
import json
import mmap
import os

import numpy as np


def decode_e4m3(raw: np.ndarray) -> np.ndarray:
    raw = raw.astype(np.uint32)
    sign = raw >> 7
    exp = (raw >> 3) & 0xF
    man = raw & 0x7
    bits = (sign << 31) | ((exp + 120) << 23) | (man << 20)
    normal = bits.view(np.float32)
    subnormal = man.astype(np.float32) * (2.0 ** -9)
    return np.where(exp == 0, subnormal * np.where(sign, -1.0, 1.0), normal)


def metrics(values: np.ndarray, bits: int) -> dict:
    levels = (1 << (bits - 1)) - 1
    scales = np.max(np.abs(values), axis=1, keepdims=True) / levels
    scales = np.maximum(scales, np.float32(1e-12))
    q = np.clip(np.rint(values / scales), -levels - 1, levels).astype(np.int8)
    deq = q.astype(np.float32) * scales
    err = deq - values
    denom = np.linalg.norm(values, axis=1) * np.linalg.norm(deq, axis=1)
    cosine = np.sum(values * deq, axis=1) / np.maximum(denom, 1e-12)
    return {
        "bits": bits,
        "scale": "symmetric_per_row",
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "max_abs": float(np.max(np.abs(err))),
        "cosine_mean": float(np.mean(cosine)),
        "cosine_p01": float(np.quantile(cosine, 0.01)),
        "scale_min": float(np.min(scales)),
        "scale_max": float(np.max(scales)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/opt/models-ssd/qwen38-flashnext-fp6int8/ple/ngram-manifest.json")
    ap.add_argument("--sidecar", default="/opt/models-ssd/qwen38-flashnext-fp6int8/ple/ngram.bin")
    ap.add_argument("--rows", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=170)
    ap.add_argument("--random", action="store_true", help="use random rows (many disk seeks)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    with open(args.manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    dim = int(manifest["row_dim"])
    stride = int(manifest["row_stride_bytes"])
    total = int(manifest["total_rows"])
    scale = float(manifest["scale"])
    if stride != dim:
        raise SystemExit("only one-byte-per-element PLE rows are supported")
    rng = np.random.default_rng(args.seed)
    count = min(args.rows, total)
    if args.random:
        rows = np.sort(rng.choice(total, size=count, replace=False))
    else:
        start_row = int(rng.integers(0, max(1, total - count)))
        rows = np.arange(start_row, start_row + count, dtype=np.int64)
    raw = np.empty((len(rows), dim), dtype=np.uint8)
    with open(args.sidecar, "rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        if len(rows) and np.all(np.diff(rows) == 1):
            start = int(rows[0]) * stride
            raw[:] = np.frombuffer(mm, dtype=np.uint8, count=len(rows) * dim, offset=start).reshape(len(rows), dim)
        else:
            for i, row in enumerate(rows):
                start = int(row) * stride
                raw[i] = np.frombuffer(mm, dtype=np.uint8, count=dim, offset=start)
    values = decode_e4m3(raw) * np.float32(scale)
    result = {
        "schema": "ple-quality-v1",
        "manifest": {"format": manifest.get("format"), "row_dim": dim, "total_rows": total, "scale": scale},
        "sample": {"rows": len(rows), "seed": args.seed},
        "source_bytes": os.path.getsize(args.sidecar),
        "estimated_bytes": {"fp8": total * dim, "int8": total * dim, "int4": (total * dim + 1) // 2},
        "quantization": {"int8": metrics(values, 8), "int4": metrics(values, 4)},
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
