#!/usr/bin/env python3
"""Measure interactive streaming while a long prefill is in flight.

This is intentionally a client-side harness: it does not change server flags and therefore
can be run against the production endpoint. The result separates TTFT from decode throughput
and records the exact workload/config labels for later chunked-prefill A/B runs.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor


def _request(url: str, prompt: str, max_tokens: int, label: str) -> dict:
    body = json.dumps({
        "model": "qwen3.8-flash-next",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    start = time.perf_counter()
    first = None
    pieces = []
    with urllib.request.urlopen(req, timeout=900) as response:
        for raw in response:
            if not raw.startswith(b"data:"):
                continue
            payload = raw[5:].strip()
            if payload == b"[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                text = delta.get("content") or ""
                if text and first is None:
                    first = time.perf_counter()
                pieces.append(text)
    end = time.perf_counter()
    text = "".join(pieces)
    ttft = (first or end) - start
    decode_time = max(end - (first or end), 1e-9)
    return {
        "label": label,
        "started_at": start,
        "ttft_s": ttft,
        "e2e_s": end - start,
        "output_chars": len(text),
        "decode_chars_per_s": len(text) / decode_time,
        "error": None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18020/v1/chat/completions")
    ap.add_argument("--long-repeat", type=int, default=900)
    ap.add_argument("--interactive-count", type=int, default=4)
    ap.add_argument("--interactive-delay", type=float, default=0.5)
    ap.add_argument("--long-max-tokens", type=int, default=128)
    ap.add_argument("--interactive-max-tokens", type=int, default=128)
    ap.add_argument("--label", default="chunk4096-overlap-on")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    nonce = uuid.uuid4().hex[:12]
    long_prompt = (
        f"Synthetic long-prefill workload nonce={nonce}. Summarize the following numbered "
        "context and return a concise checklist.\n" + "\n".join(
            f"item {i}: repository context sentence for prefill interference measurement."
            for i in range(args.long_repeat)
        )
    )
    short = [
        f"Interactive request {i} nonce={nonce}: explain one Python exception in two sentences."
        for i in range(args.interactive_count)
    ]

    results = []
    with ThreadPoolExecutor(max_workers=args.interactive_count + 1) as pool:
        futures = [pool.submit(_request, args.url, long_prompt, args.long_max_tokens, "long_prefill")]
        time.sleep(args.interactive_delay)
        futures.extend(
            pool.submit(_request, args.url, prompt, args.interactive_max_tokens, f"interactive_{i}")
            for i, prompt in enumerate(short)
        )
        for future in futures:
            try:
                results.append(future.result())
            except Exception as exc:  # retain failures in the raw record
                results.append({"label": "unknown", "error": repr(exc)})

    record = {
        "schema": "mixed-workload-v1",
        "config_label": args.label,
        "url": args.url.rsplit("/", 1)[0],
        "long_repeat": args.long_repeat,
        "interactive_count": args.interactive_count,
        "interactive_delay_s": args.interactive_delay,
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    good = [r for r in results if not r.get("error") and r["label"].startswith("interactive")]
    if good:
        print(
            f"interactive ttft median={statistics.median(r['ttft_s'] for r in good):.3f}s "
            f"p95={sorted(r['ttft_s'] for r in good)[max(0, int(len(good) * .95) - 1)]:.3f}s"
        )
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
