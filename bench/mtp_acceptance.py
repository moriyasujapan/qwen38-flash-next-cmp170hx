#!/usr/bin/env python3
"""Record workload-level MTP effectiveness using API timings and Prometheus deltas.

Run this against an otherwise idle server. The script marks a sample contaminated when more
than its own streaming request reaches the server during the measurement window.
"""

import argparse
import hashlib
import json
import math
import statistics
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path


def get_text(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode()


def parse_prometheus(text):
    samples = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        head, raw_value = line.rsplit(" ", 1)
        if "{" in head:
            name, raw_labels = head.split("{", 1)
            raw_labels = raw_labels[:-1]
            labels = {}
            if raw_labels:
                for item in raw_labels.split(","):
                    key, value = item.split("=", 1)
                    labels[key] = value.strip('"')
        else:
            name, labels = head, {}
        try:
            value = float(raw_value)
        except ValueError:
            continue
        samples.append((name, labels, value))
    return samples


def metric_sum(samples, name, required_labels=None):
    required_labels = required_labels or {}
    values = [
        value
        for sample_name, labels, value in samples
        if sample_name == name
        and all(labels.get(key) == expected for key, expected in required_labels.items())
    ]
    if not values:
        raise KeyError(f"metric not found: {name} labels={required_labels}")
    return sum(values)


def metric_one(samples, name):
    values = [value for sample_name, _, value in samples if sample_name == name]
    if not values:
        raise KeyError(f"metric not found: {name}")
    return values[0]


def stream_chat(url, model, prompt, max_tokens, temperature, thinking):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first = None
    ended = None
    usage = None
    output = []
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                piece = (delta.get("reasoning_content") or "") + (
                    delta.get("content") or ""
                )
                if piece:
                    first = first or time.perf_counter()
                    output.append(piece)
    ended = time.perf_counter()
    if first is None or usage is None:
        raise RuntimeError("stream did not return output and usage")
    completion_tokens = usage["completion_tokens"]
    return {
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": completion_tokens,
        "ttft_seconds": first - started,
        "e2e_seconds": ended - started,
        "decode_seconds": ended - first,
        "decode_tps": (completion_tokens - 1) / (ended - first),
        "output_sha256": hashlib.sha256("".join(output).encode()).hexdigest(),
    }


def make_context(target_tokens, nonce):
    if target_tokens <= 0:
        return ""
    unit = (
        "Repository note: immutable snapshots preserve parent links; readers validate "
        "checksums before publication. "
    )
    # The API usage count remains authoritative; this is only an approximate input size.
    return f"context-nonce={nonce}\n" + unit * max(1, target_tokens // 16) + "\n"


def load_workloads(path):
    data = json.loads(Path(path).read_text())
    required = {"id", "category", "prompt"}
    for item in data:
        missing = required - item.keys()
        if missing:
            raise ValueError(f"workload entry is missing {sorted(missing)}")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18020")
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--workloads", default=str(Path(__file__).with_name("workloads.json")))
    parser.add_argument("--contexts", nargs="+", type=int, default=[0, 8192])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--config-label", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    workloads = load_workloads(args.workloads)
    metrics_url = args.base_url.rstrip("/") + "/metrics"
    chat_url = args.base_url.rstrip("/") + "/v1/chat/completions"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    initial_metrics = parse_prometheus(get_text(metrics_url))
    spec_steps = metric_one(initial_metrics, "sglang:spec_num_steps")
    draft_tokens = metric_one(initial_metrics, "sglang:spec_num_draft_tokens")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    records = []

    for context_tokens in args.contexts:
        for workload in workloads:
            for repeat in range(1, args.repeats + 1):
                nonce = uuid.uuid4().hex
                prompt = make_context(context_tokens, nonce) + workload["prompt"]
                before = parse_prometheus(get_text(metrics_url))
                before_verify = metric_sum(before, "sglang:spec_verify_calls_total")
                before_generation = metric_sum(
                    before,
                    "sglang:generation_tokens_total",
                    {"is_streaming": "true"},
                )
                before_requests = metric_sum(
                    before,
                    "sglang:num_requests_total",
                    {"is_streaming": "true"},
                )

                api = stream_chat(
                    chat_url,
                    args.model,
                    prompt,
                    args.max_tokens,
                    args.temperature,
                    args.thinking,
                )
                time.sleep(0.1)
                after = parse_prometheus(get_text(metrics_url))
                verify_delta = metric_sum(
                    after, "sglang:spec_verify_calls_total"
                ) - before_verify
                generation_delta = metric_sum(
                    after,
                    "sglang:generation_tokens_total",
                    {"is_streaming": "true"},
                ) - before_generation
                request_delta = metric_sum(
                    after,
                    "sglang:num_requests_total",
                    {"is_streaming": "true"},
                ) - before_requests
                accept_length = (
                    generation_delta / verify_delta if verify_delta > 0 else math.nan
                )
                proposed_drafts = max(draft_tokens - 1.0, 1.0)
                accept_rate = (accept_length - 1.0) / proposed_drafts
                record = {
                    "schema": 1,
                    "run_id": run_id,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "config_label": args.config_label,
                    "spec_steps": spec_steps,
                    "draft_tokens": draft_tokens,
                    "workload_id": workload["id"],
                    "category": workload["category"],
                    "context_target_tokens": context_tokens,
                    "repeat": repeat,
                    "temperature": args.temperature,
                    "thinking": args.thinking,
                    **api,
                    "verify_calls_delta": verify_delta,
                    "generation_counter_delta": generation_delta,
                    "request_counter_delta": request_delta,
                    "counter_accept_length": accept_length,
                    "counter_accept_rate": accept_rate,
                    "last_batch_accept_length": metric_one(
                        after, "sglang:spec_accept_length"
                    ),
                    "last_batch_accept_rate": metric_one(
                        after, "sglang:spec_accept_rate"
                    ),
                    "contaminated": request_delta != 1.0,
                }
                records.append(record)
                with output_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(
                    f"MTP workload={workload['id']} context={context_tokens} "
                    f"repeat={repeat} accept_len={accept_length:.3f} "
                    f"accept_rate={accept_rate:.3f} decode={api['decode_tps']:.2f}tok/s "
                    f"ttft={api['ttft_seconds']:.3f}s contaminated={request_delta != 1.0}",
                    flush=True,
                )

    valid = [record for record in records if not record["contaminated"]]
    print(
        "SUMMARY "
        f"samples={len(records)} valid={len(valid)} "
        f"accept_length_mean={statistics.mean(r['counter_accept_length'] for r in valid):.3f} "
        f"decode_median={statistics.median(r['decode_tps'] for r in valid):.2f}tok/s",
        flush=True,
    )


if __name__ == "__main__":
    main()
