#!/usr/bin/env python3
"""Small streaming benchmark that separates TTFT from decode speed."""

import json
import statistics
import time
import urllib.request


URL = "http://127.0.0.1:18020/v1/chat/completions"
MODEL = "qwen3.8-flash-next"


def run(prompt, max_tokens, thinking=False):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    req = urllib.request.Request(
        URL, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    started = time.perf_counter()
    first = None
    usage = None
    pieces = []
    with urllib.request.urlopen(req, timeout=600) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                piece = (delta.get("content") or "") + (
                    delta.get("reasoning_content") or ""
                )
                if piece:
                    first = first or time.perf_counter()
                    pieces.append(piece)
    ended = time.perf_counter()
    completion_tokens = usage["completion_tokens"]
    decode_tps = (completion_tokens - 1) / (ended - first)
    return {
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": completion_tokens,
        "ttft": first - started,
        "decode_tps": decode_tps,
        "text": "".join(pieces),
    }


def main():
    smoke = run("日本の首都と、その都市の特徴を2文で説明して。", 120)
    print(
        f"SMOKE prompt={smoke['prompt_tokens']} "
        f"completion={smoke['completion_tokens']} "
        f"TTFT={smoke['ttft']:.2f}s decode={smoke['decode_tps']:.1f}tok/s"
    )
    print("RESPONSE", smoke["text"].strip().replace("\n", " ")[:160])

    values = []
    for index in range(1, 4):
        result = run(
            "Pythonで二分探索木を実装し、挿入・削除・探索の計算量を詳しく解説してください。",
            1024,
        )
        values.append(result["decode_tps"])
        print(
            f"RUN {index} completion={result['completion_tokens']} "
            f"TTFT={result['ttft']:.2f}s decode={result['decode_tps']:.1f}tok/s"
        )
    print(f"MEDIAN decode={statistics.median(values):.1f}tok/s")


if __name__ == "__main__":
    main()
