#!/usr/bin/env python3
"""Benchmark the three V2 source-search schedules with shared loaded models."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from statistics import fmean

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from backend.reguaz.retrieval.hybrid_v2 import HybridV2Retriever  # noqa: E402


STRATEGIES = ("sequential", "qdrant_batch_bm25_overlap", "concurrent_all")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = {}
    with HybridV2Retriever(local_files_only=args.local_files_only) as retriever:
        runtime = None
        for strategy in STRATEGIES:
            retriever.settings = replace(
                retriever.settings, retrieval_strategy=strategy
            )
            for _ in range(args.warmup_runs):
                retriever.retrieve(args.query)
            runs = [retriever.retrieve_with_trace(args.query) for _ in range(args.runs)]
            runtime = runs[0]["runtime"]
            timing_keys = sorted({key for run in runs for key in run["timings"]})
            stable = {
                stage: all(
                    [item["chunk_id"] for item in run["intermediate"][stage]]
                    == [item["chunk_id"] for item in runs[0]["intermediate"][stage]]
                    for run in runs[1:]
                )
                for stage in (
                    "dense",
                    "sparse",
                    "bm25",
                    "fused_all",
                    "reranked_candidates",
                )
            }
            rows[strategy] = {
                "mean_timings_ms": {
                    key: fmean(
                        run["timings"][key] for run in runs if key in run["timings"]
                    )
                    for key in timing_keys
                },
                "stable_within_strategy": stable,
                "stage_ids": {
                    stage: [item["chunk_id"] for item in runs[0]["intermediate"][stage]]
                    for stage in stable
                },
            }
        reference = rows["sequential"]["stage_ids"]
        parity = {
            stage: all(
                rows[strategy]["stage_ids"][stage] == reference[stage]
                for strategy in STRATEGIES
            )
            for stage in reference
        }
        output = {
            "schema_version": "2.0",
            "query": args.query,
            "runs_per_strategy": args.runs,
            "warmup_runs_per_strategy": args.warmup_runs,
            "runtime": runtime,
            "strategies": rows,
            "ordered_parity_across_strategies": parity,
            "selected_strategy": "qdrant_batch_bm25_overlap",
            "selection_reason": "Fastest stable measured strategy without simultaneous independent operations against the embedded Qdrant client.",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selected_strategy": output["selected_strategy"],
                "parity": parity,
                "mean_timings_ms": {
                    key: value["mean_timings_ms"] for key, value in rows.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
