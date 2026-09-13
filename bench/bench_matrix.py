#!/usr/bin/env python3
"""Measure decode concurrency, prefill and RadixCache behavior."""

import argparse
import concurrent.futures
import json
import statistics
import threading
import time
import urllib.request
import uuid


def percentile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    low = int(index)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (index - low)


def stream_request(url, model, prompt, max_tokens, start_gate=None):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    if start_gate is not None:
        start_gate.wait()
    started = time.perf_counter()
    first = None
    chunks = 0
    usage = None
    with urllib.request.urlopen(request, timeout=900) as response:
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
                    chunks += 1
    ended = time.perf_counter()
    completion_tokens = usage.get("completion_tokens") if usage else chunks
    decode_seconds = ended - first if first is not None else 0.0
    return {
        "first": first,
        "ended": ended,
        "prompt_tokens": usage.get("prompt_tokens") if usage else None,
        "completion_tokens": completion_tokens,
        "ttft": first - started if first is not None else None,
        "decode_tps": (
            max(completion_tokens - 1, 0) / decode_seconds if decode_seconds else 0.0
        ),
    }


def run_concurrency(url, model, concurrency, output_tokens, round_index):
    gate = threading.Event()
    nonce = uuid.uuid4().hex
    prompts = [
        f"benchmark nonce {nonce}-{index}-{round_index}. "
        "Pythonで永続的赤黒木を実装し、計算量と設計判断を詳しく説明してください。"
        for index in range(concurrency)
    ]
    with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
        futures = [
            pool.submit(stream_request, url, model, prompt, output_tokens, gate)
            for prompt in prompts
        ]
        common_start = time.perf_counter()
        gate.set()
        results = [future.result() for future in futures]
    common_end = max(result["ended"] for result in results)
    first_token = min(result["first"] for result in results)
    total_tokens = sum(result["completion_tokens"] for result in results)
    decode_tokens = sum(max(result["completion_tokens"] - 1, 0) for result in results)
    return {
        "concurrency": concurrency,
        "round": round_index,
        "total_tokens": total_tokens,
        "wall_seconds": common_end - common_start,
        "e2e_output_tps": total_tokens / (common_end - common_start),
        "aggregate_decode_tps": decode_tokens / (common_end - first_token),
        "ttft_mean": statistics.mean(result["ttft"] for result in results),
        "ttft_p95": percentile([result["ttft"] for result in results], 0.95),
        "request_decode_mean": statistics.mean(
            result["decode_tps"] for result in results
        ),
    }


def make_prefill_prompt(approx_tokens, nonce):
    sentence = (
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
        "nu xi omicron pi rho sigma tau upsilon phi chi psi omega. "
    )
    return f"cold-prefix-{nonce}. " + sentence * max(1, approx_tokens // 24) + (
        "\nSummarize in one word."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url", default="http://127.0.0.1:18020/v1/chat/completions"
    )
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument(
        "--prefill-tokens", nargs="+", type=int, default=[1024, 4096, 16384]
    )
    parser.add_argument("--radix-tokens", type=int, default=8192)
    args = parser.parse_args()

    warmup = stream_request(
        args.url, args.model, f"warmup-{uuid.uuid4().hex}: count to 32", 32
    )
    print(
        f"WARMUP tokens={warmup['completion_tokens']} ttft={warmup['ttft']:.3f}s "
        f"decode={warmup['decode_tps']:.2f}tok/s",
        flush=True,
    )

    all_results = []
    for concurrency in args.concurrency:
        for round_index in range(1, args.rounds + 1):
            result = run_concurrency(
                args.url, args.model, concurrency, args.output_tokens, round_index
            )
            all_results.append(result)
            print(
                "CONCURRENCY "
                f"c={concurrency} round={round_index} "
                f"tokens={result['total_tokens']} wall={result['wall_seconds']:.3f}s "
                f"e2e_output={result['e2e_output_tps']:.2f}tok/s "
                f"aggregate_decode={result['aggregate_decode_tps']:.2f}tok/s "
                f"request_decode_mean={result['request_decode_mean']:.2f}tok/s "
                f"ttft_mean={result['ttft_mean']:.3f}s "
                f"ttft_p95={result['ttft_p95']:.3f}s",
                flush=True,
            )

    for concurrency in args.concurrency:
        group = [r for r in all_results if r["concurrency"] == concurrency]
        print(
            f"CONCURRENCY_SUMMARY c={concurrency} rounds={len(group)} "
            f"aggregate_decode_median="
            f"{statistics.median(r['aggregate_decode_tps'] for r in group):.2f}tok/s "
            f"e2e_output_median="
            f"{statistics.median(r['e2e_output_tps'] for r in group):.2f}tok/s "
            f"ttft_mean={statistics.mean(r['ttft_mean'] for r in group):.3f}s",
            flush=True,
        )

    for approx_tokens in args.prefill_tokens:
        prompt = make_prefill_prompt(approx_tokens, uuid.uuid4().hex)
        result = stream_request(args.url, args.model, prompt, 1)
        actual = result["prompt_tokens"]
        print(
            f"PREFILL requested_approx={approx_tokens} actual={actual} "
            f"ttft={result['ttft']:.3f}s "
            f"e2e_prefill={actual / result['ttft']:.2f}tok/s",
            flush=True,
        )

    radix_prompt = make_prefill_prompt(args.radix_tokens, uuid.uuid4().hex)
    cold = stream_request(args.url, args.model, radix_prompt, 1)
    warm = stream_request(args.url, args.model, radix_prompt, 1)
    print(
        f"RADIX actual={cold['prompt_tokens']} cold_ttft={cold['ttft']:.3f}s "
        f"warm_ttft={warm['ttft']:.3f}s "
        f"speedup={cold['ttft'] / warm['ttft']:.2f}x",
        flush=True,
    )


if __name__ == "__main__":
    main()
