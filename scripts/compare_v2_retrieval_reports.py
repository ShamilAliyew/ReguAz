#!/usr/bin/env python3
"""Compare sequential and optimized V2 evaluation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    optimized = json.loads(args.optimized.read_text(encoding="utf-8"))
    if baseline["dataset_sha256"] != optimized["dataset_sha256"]:
        raise ValueError("reports use different frozen datasets")
    baseline_traces = {row["record_id"]: row for row in baseline["traces"]}
    optimized_traces = {row["record_id"]: row for row in optimized["traces"]}
    stages = ("dense", "sparse", "bm25", "rrf", "reranked")
    parity = {}
    for stage in stages:
        mismatches = [
            record_id
            for record_id in sorted(set(baseline_traces) | set(optimized_traces))
            if baseline_traces.get(record_id, {}).get("stage_ids", {}).get(stage)
            != optimized_traces.get(record_id, {}).get("stage_ids", {}).get(stage)
        ]
        parity[stage] = {
            "exact_ordered_parity": not mismatches,
            "mismatch_count": len(mismatches),
            "mismatch_record_ids": mismatches,
        }
    latency = {}
    for key in sorted(set(baseline["latency_ms"]) & set(optimized["latency_ms"])):
        before = float(baseline["latency_ms"][key]["mean"])
        after = float(optimized["latency_ms"][key]["mean"])
        latency[key] = {
            "baseline_mean_ms": before,
            "optimized_mean_ms": after,
            "delta_ms": after - before,
            "change_percent": 100.0 * (after - before) / before if before else 0.0,
        }
    quality_equal = baseline["quality"] == optimized["quality"]
    report = {
        "schema_version": "2.0",
        "dataset_sha256": baseline["dataset_sha256"],
        "parity_policy": "Exact ordered chunk-ID parity for dense, sparse, BM25, RRF and reranked top-20 lists on every query.",
        "ranking_parity": parity,
        "quality_metrics_exactly_equal": quality_equal,
        "baseline_strategy": baseline["strategy"],
        "optimized_strategy": optimized["strategy"],
        "latency_comparison": latency,
        "baseline_failures": baseline["failures"],
        "optimized_failures": optimized["failures"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return (
        0
        if quality_equal
        and all(item["exact_ordered_parity"] for item in parity.values())
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
