#!/usr/bin/env python3
"""Deterministic smoke and long-context probes for GDN state dtype A/B."""

import argparse
import hashlib
import json
import time
import urllib.request


NEEDLE = "ZXCVBNM-170HX-583921"


def chat(url, model, prompt, max_tokens):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=1800) as response:
        result = json.load(response)
    elapsed = time.perf_counter() - started
    message = result["choices"][0]["message"]
    text = (message.get("reasoning_content") or "") + (message.get("content") or "")
    return {
        "elapsed": elapsed,
        "usage": result.get("usage"),
        "finish_reason": result["choices"][0].get("finish_reason"),
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text": text,
    }


def needle_prompt(approx_tokens, depth):
    filler = "All work and no play makes Jack a dull boy. " * max(
        1, approx_tokens // 11
    )
    location = int(len(filler) * depth)
    context = (
        filler[:location]
        + f"\nThe secret passcode is {NEEDLE}. Remember it exactly.\n"
        + filler[location:]
    )
    return context + "\nWhat is the passcode? Reply with the passcode only."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tag")
    parser.add_argument(
        "--url", default="http://127.0.0.1:18020/v1/chat/completions"
    )
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--needle-tokens", type=int, default=16384)
    args = parser.parse_args()

    prompts = {
        "arithmetic": (
            "青い箱が37個、赤い箱が青い箱の3倍より5個少なくあります。"
            "赤い箱を28個出荷しました。残った箱の総数を式と答えだけで示してください。"
        ),
        "code": (
            "Write a Python function that returns the longest strictly increasing "
            "subsequence length in O(n log n). Return only one fenced code block."
        ),
        "logic": (
            "All glippets are trangs. No trang is a vorp. Some nims are glippets. "
            "Can any of those nims be vorps? Answer yes or no and justify in one sentence."
        ),
    }
    result = {"tag": args.tag, "short": {}, "needle": {}}
    for name, prompt in prompts.items():
        item = chat(args.url, args.model, prompt, 256)
        result["short"][name] = item
        print(
            f"SHORT name={name} elapsed={item['elapsed']:.3f}s "
            f"tokens={item['usage']['completion_tokens']} sha256={item['sha256']}",
            flush=True,
        )

    for depth in (0.1, 0.5, 0.9):
        item = chat(
            args.url,
            args.model,
            needle_prompt(args.needle_tokens, depth),
            32,
        )
        item["retrieved"] = NEEDLE in item["text"]
        result["needle"][str(depth)] = item
        print(
            f"NEEDLE depth={depth:.1f} elapsed={item['elapsed']:.3f}s "
            f"prompt_tokens={item['usage']['prompt_tokens']} "
            f"retrieved={item['retrieved']}",
            flush=True,
        )

    output = f"gdn-quality-{args.tag}.json"
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(f"WROTE {output}", flush=True)


if __name__ == "__main__":
    main()
