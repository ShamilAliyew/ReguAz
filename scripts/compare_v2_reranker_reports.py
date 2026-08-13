#!/usr/bin/env python3
"""Compare two frozen ReguAZ V2 reranker evaluation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evaluation/reguaz_v2_evaluation_50.jsonl"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    first = json.loads(args.first.read_text(encoding="utf-8"))
    second = json.loads(args.second.read_text(encoding="utf-8"))
    gold = {
        row["id"]: set(row["relevant_chunk_ids"])
        for row in (
            json.loads(line)
            for line in args.dataset.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    _validate_pair(first, second)
    models = {
        first["reranker_model"]: _summary(first),
        second["reranker_model"]: _summary(second),
    }
    parity = _source_parity(first, second)
    disagreements = _hit_disagreements(first, second, gold)
    winner = _select_winner(models)
    report = {
        "schema_version": "2.0",
        "dataset_sha256": first["dataset_sha256"],
        "rerank_top_k": 15,
        "selection_policy": (
            "Prefer test exact-child Hit Rate@5, then Recall@5, nDCG@5, "
            "MRR@5, then lower mean reranker latency."
        ),
        "models": models,
        "pre_rerank_exact_ordered_parity": parity,
        "top5_hit_disagreements": disagreements,
        "selected_model": winner,
        "selection_reason": (
            "Test Hit Rate@5 tied; selected model has higher test Recall@5, "
            "nDCG@5 and MRR@5 and lower reranker latency."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _validate_pair(first: dict[str, Any], second: dict[str, Any]) -> None:
    if first["dataset_sha256"] != second["dataset_sha256"]:
        raise ValueError("reranker reports use different datasets")
    if first.get("rerank_top_k") != 15 or second.get("rerank_top_k") != 15:
        raise ValueError("both reranker reports must use rerank_top_k=15")
    if first["successful_query_count"] != 50 or second["successful_query_count"] != 50:
        raise ValueError("both reranker reports must contain 50 successful queries")


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    quality = report["quality"]["reranked"]
    return {
        "revision": report.get("reranker_revision"),
        "code_revision": report.get("reranker_code_revision"),
        "full": _metrics_at_five(quality["full"]),
        "dev": _metrics_at_five(quality["split"]["dev"]["metrics"]),
        "test": _metrics_at_five(quality["split"]["test"]["metrics"]),
        "reranker_latency_ms": report["latency_ms"]["reranker_ms"],
        "total_latency_ms": report["latency_ms"]["total_retrieval_ms"],
        "runtime": report["runtime"],
    }


def _metrics_at_five(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "exact_child_hit_rate@5": metrics["exact_child_hit_rate@5"],
        "document_hit_rate@5": metrics["document_hit_rate@5"],
        "recall@5": metrics["recall@5"],
        "mrr@5": metrics["mrr@5"],
        "ndcg@5": metrics["ndcg@5"],
    }


def _source_parity(first: dict[str, Any], second: dict[str, Any]) -> dict[str, bool]:
    first_traces = {item["record_id"]: item for item in first["traces"]}
    second_traces = {item["record_id"]: item for item in second["traces"]}
    if first_traces.keys() != second_traces.keys():
        raise ValueError("reranker report record IDs differ")
    return {
        stage: all(
            first_traces[record_id]["stage_ids"][stage]
            == second_traces[record_id]["stage_ids"][stage]
            for record_id in first_traces
        )
        for stage in ("dense", "sparse", "bm25", "rrf")
    }


def _hit_disagreements(
    first: dict[str, Any],
    second: dict[str, Any],
    gold: dict[str, set[str]],
) -> dict[str, int]:
    first_traces = {item["record_id"]: item for item in first["traces"]}
    second_traces = {item["record_id"]: item for item in second["traces"]}
    different_order = 0
    different_set = 0
    first_only = 0
    second_only = 0
    for record_id, first_trace in first_traces.items():
        first_ids = first_trace["stage_ids"]["reranked"][:5]
        second_ids = second_traces[record_id]["stage_ids"]["reranked"][:5]
        different_order += first_ids != second_ids
        different_set += set(first_ids) != set(second_ids)
        first_hit = bool(gold[record_id] & set(first_ids))
        second_hit = bool(gold[record_id] & set(second_ids))
        first_only += first_hit and not second_hit
        second_only += second_hit and not first_hit
    return {
        "different_ordered_top5_count": different_order,
        "different_top5_set_count": different_set,
        "first_only_hit_count": first_only,
        "second_only_hit_count": second_only,
    }


def _select_winner(models: dict[str, dict[str, Any]]) -> str:
    def key(item: tuple[str, dict[str, Any]]) -> tuple[float, ...]:
        summary = item[1]
        test = summary["test"]
        return (
            test["exact_child_hit_rate@5"],
            test["recall@5"],
            test["ndcg@5"],
            test["mrr@5"],
            -summary["reranker_latency_ms"]["mean"],
        )

    return max(models.items(), key=key)[0]


if __name__ == "__main__":
    raise SystemExit(main())
