#!/usr/bin/env python3
"""Stream-convert the raw PLE FP8 sidecar to an experimental INT8/INT4 sidecar.

Output is deliberately not wired into SGLang: the runtime needs a matching packed lookup
kernel and quality gate first. Row scales are stored separately as float16 so conversion can
run with bounded RAM over the 51 GB source file.
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
from pathlib import Path

import numpy as np


def decode_e4m3(raw: np.ndarray) -> np.ndarray:
    raw = raw.astype(np.uint32)
    sign = raw >> 7
    exp = (raw >> 3) & 0xF
    man = raw & 0x7
    normal = ((sign << 31) | ((exp + 120) << 23) | (man << 20)).view(np.float32)
    sub = man.astype(np.float32) * (2.0 ** -9)
    return np.where(exp == 0, sub * np.where(sign, -1.0, 1.0), normal)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--scale-output", required=True)
    ap.add_argument("--format", choices=("int8", "int4"), required=True)
    ap.add_argument("--chunk-rows", type=int, default=65536)
    ap.add_argument("--start-row", type=int, default=0)
    ap.add_argument("--rows", type=int, default=None)
    args = ap.parse_args()
    with open(args.manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    dim = int(manifest["row_dim"])
    total = int(manifest["total_rows"])
    stride = int(manifest["row_stride_bytes"])
    source_scale = np.float32(manifest["scale"])
    if stride != dim:
        raise SystemExit("expected one raw FP8 byte per PLE value")
    start = max(0, args.start_row)
    count = total - start if args.rows is None else min(args.rows, total - start)
    if count <= 0:
        raise SystemExit("no rows selected")
    levels = 127 if args.format == "int8" else 7
    packed_width = (dim + 1) // 2 if args.format == "int4" else dim
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.scale_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.sidecar, "rb") as source, mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as mm, open(args.output, "wb") as dst, open(args.scale_output, "wb") as scales:
        for offset in range(0, count, args.chunk_rows):
            rows = min(args.chunk_rows, count - offset)
            file_offset = (start + offset) * stride
            raw = np.frombuffer(mm, dtype=np.uint8, count=rows * dim, offset=file_offset).reshape(rows, dim).copy()
            values = decode_e4m3(raw) * source_scale
            row_scale = np.maximum(np.max(np.abs(values), axis=1) / levels, 1e-12).astype(np.float16)
            q = np.clip(np.rint(values / row_scale[:, None]), -levels - 1, levels).astype(np.int8)
            if args.format == "int4":
                packed = np.zeros((rows, packed_width), dtype=np.uint8)
                lo = q[:, 0::2] & 0x0F
                hi = q[:, 1::2] & 0x0F
                packed[:, : lo.shape[1]] = lo | (hi << 4)
                packed.tofile(dst)
            else:
                q.tofile(dst)
            row_scale.tofile(scales)
    out_manifest = {
        "format": args.format,
        "source_format": manifest.get("format"),
        "row_dim": dim,
        "total_rows": count,
        "start_row": start,
        "row_stride_bytes": packed_width,
        "scale_dtype": "float16",
        "scale_file": os.path.basename(args.scale_output),
    }
    with open(f"{args.output}.manifest.json", "w", encoding="utf-8") as handle:
        json.dump(out_manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps(out_manifest, indent=2))


if __name__ == "__main__":
    main()
