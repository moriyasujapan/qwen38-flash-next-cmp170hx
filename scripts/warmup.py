#!/usr/bin/env python3
"""Compile common serving-time paths before declaring the API ready."""

import argparse
import concurrent.futures
import json
import urllib.request
import uuid


def request(url, prompt, max_tokens):
    body = {
        "model": "qwen3.8-flash-next",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        url,
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url", default="http://127.0.0.1:18020/v1/chat/completions"
    )
    args = parser.parse_args()

    for concurrency in (1, 2, 3):
        nonce = uuid.uuid4().hex
        with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
            futures = [
                pool.submit(
                    request,
                    args.url,
                    f"warmup-{nonce}-{index}: count from 1 to 16",
                    16,
                )
                for index in range(concurrency)
            ]
            for future in futures:
                future.result()
        print(f"WARMUP_DECODE concurrency={concurrency}", flush=True)

    prefix = ("alpha beta gamma delta epsilon zeta eta theta. " * 600) + nonce
    request(args.url, prefix + " summarize in one word", 1)
    request(args.url, prefix + " summarize in two words", 1)
    print("WARMUP_PREFIX complete", flush=True)


if __name__ == "__main__":
    main()
