#!/usr/bin/env python3
"""Rank recorded MTP configurations while retaining per-workload regressions."""

import argparse
import collections
import json
import math
import statistics
from pathlib import Path


def geometric_mean(values):
    return math.exp(statistics.mean(math.log(value) for value in values if value > 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--max-ttft-regression", type=float, default=1.20)
    args = parser.parse_args()

    records = []
    for input_path in args.inputs:
        with Path(input_path).open(encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    records = [record for record in records if not record.get("contaminated")]
    if not records:
        raise SystemExit("no uncontaminated records")

    grouped = collections.defaultdict(list)
    for record in records:
        grouped[record["config_label"]].append(record)

    baseline_label = sorted(grouped)[0]
    baseline_ttft = statistics.median(
        record["ttft_seconds"] for record in grouped[baseline_label]
    )
    ranking = []
    for label, group in grouped.items():
        by_workload = collections.defaultdict(list)
        for record in group:
            key = (record["workload_id"], record["context_target_tokens"])
            by_workload[key].append(record["decode_tps"])
        workload_medians = [statistics.median(values) for values in by_workload.values()]
        median_ttft = statistics.median(record["ttft_seconds"] for record in group)
        ranking.append(
            {
                "label": label,
                "steps": group[0]["spec_steps"],
                "draft_tokens": group[0]["draft_tokens"],
                "score_geomean_decode": geometric_mean(workload_medians),
                "worst_workload_decode": min(workload_medians),
                "median_ttft": median_ttft,
                "ttft_ratio_to_baseline": median_ttft / baseline_ttft,
                "eligible": median_ttft
                <= baseline_ttft * args.max_ttft_regression,
                "samples": len(group),
            }
        )
    ranking.sort(key=lambda item: (item["eligible"], item["score_geomean_decode"]), reverse=True)
    print(json.dumps({"baseline": baseline_label, "ranking": ranking}, indent=2))
    winner = next((item for item in ranking if item["eligible"]), None)
    if winner:
        print(
            "RECOMMENDED_ENV "
            f"SPEC_STEPS={int(winner['steps'])} "
            f"SPEC_DRAFT={int(winner['draft_tokens'])} "
            f"# {winner['label']}"
        )


if __name__ == "__main__":
    main()
