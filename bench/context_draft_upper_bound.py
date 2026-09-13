#!/usr/bin/env python3
"""Measure a text-context match upper bound for DFlash2-style drafting.

This does not alter SGLang. It asks: if a continuation starts with a suffix already present
in the prompt, how many tokens could a context drafter propose without model computation?
It is useful for deciding whether a native Flash-Next drafter is worth implementing.
"""
from __future__ import annotations

import argparse
import json

from transformers import AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="/opt/models-ssd/qwen38-flashnext-fp6int8")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--continuation", required=True)
    ap.add_argument("--max-draft", type=int, default=8)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    prompt = tokenizer.encode(args.prompt, add_special_tokens=False)
    continuation = tokenizer.encode(args.continuation, add_special_tokens=False)
    best = 0
    for n in range(min(args.max_draft, len(continuation)), 0, -1):
        if len(prompt) >= n and prompt[-n:] == continuation[:n]:
            best = n
            break
    result = {
        "schema": "context-draft-upper-bound-v1",
        "prompt_tokens": len(prompt),
        "continuation_tokens": len(continuation),
        "max_draft": args.max_draft,
        "matched_prefix_tokens": best,
        "candidate": tokenizer.decode(continuation[:best]) if best else "",
        "interpretation": "offline suffix/prefix match only; not a serving-quality or acceptance result",
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
