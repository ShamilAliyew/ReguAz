from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from statistics import fmean, median
from typing import Any

from backend.reguaz.evaluation.v2_dataset import EvaluationRecord
from backend.reguaz.retrieval.v2_artifacts import V2ArtifactStore


CUTOFFS = (5, 10, 20, 30)


def evaluate_ranking(
    retrieved_ids: Sequence[str],
    record: EvaluationRecord,
    store: V2ArtifactStore,
    *,
    cutoffs: Sequence[int] = CUTOFFS,
) -> dict[str, float]:
    gold_chunks = set(record.relevant_chunk_ids)
    gold_logical = set(record.relevant_logical_chunk_ids)
    gold_documents = set(record.relevant_document_version_ids)
    retrieved = list(dict.fromkeys(str(value) for value in retrieved_ids))
    metadata = [store.get_child(chunk_id) for chunk_id in retrieved]
    output: dict[str, float] = {}
    for cutoff in cutoffs:
        values = retrieved[:cutoff]
        values_meta = metadata[:cutoff]
        hits = [
            rank for rank, chunk_id in enumerate(values, 1) if chunk_id in gold_chunks
        ]
        relevant_count = len(set(values) & gold_chunks)
        dcg = sum(1.0 / math.log2(rank + 1) for rank in hits)
        ideal = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, min(len(gold_chunks), cutoff) + 1)
        )
        logical = {
            str(item["logical_chunk_id"]) for item in values_meta if item is not None
        }
        documents = {
            str(item["document_version_id"]) for item in values_meta if item is not None
        }
        output[f"hit_rate@{cutoff}"] = float(bool(hits))
        output[f"exact_child_hit_rate@{cutoff}"] = float(bool(hits))
        output[f"recall@{cutoff}"] = relevant_count / len(gold_chunks)
        output[f"mrr@{cutoff}"] = 1.0 / hits[0] if hits else 0.0
        output[f"ndcg@{cutoff}"] = dcg / ideal if ideal else 0.0
        output[f"logical_chunk_hit_rate@{cutoff}"] = float(bool(logical & gold_logical))
        output[f"document_hit_rate@{cutoff}"] = float(bool(documents & gold_documents))
    return output


def aggregate_quality(
    per_query: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def summarize(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
        selected = list(rows)
        keys = sorted({key for row in selected for key in row["metrics"]})
        return {
            key: fmean(
                float(row["metrics"][key]) for row in selected if key in row["metrics"]
            )
            for key in keys
        }

    dimensions = {
        "split": lambda row: row["split"],
        "difficulty": lambda row: row["difficulty"],
        "category": lambda row: row["category"],
        "question_type": lambda row: row["question_type"],
        "evidence_cardinality": lambda row: (
            "multi_chunk" if row["requires_multiple_chunks"] else "single_chunk"
        ),
    }
    output: dict[str, Any] = {"full": summarize(per_query)}
    for dimension, getter in dimensions.items():
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in per_query:
            grouped[str(getter(row))].append(row)
        output[dimension] = {
            key: {"query_count": len(rows), "metrics": summarize(rows)}
            for key, rows in sorted(grouped.items())
        }
    return output


def latency_summary(values: Sequence[float]) -> dict[str, float]:
    samples = sorted(float(value) for value in values)
    if not samples:
        return {key: 0.0 for key in ("mean", "median", "p95", "min", "max")}
    return {
        "mean": fmean(samples),
        "median": median(samples),
        "p95": samples[max(0, math.ceil(0.95 * len(samples)) - 1)],
        "min": samples[0],
        "max": samples[-1],
    }


def compare_rankings(
    baseline: Mapping[str, Sequence[str]], optimized: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    ids = sorted(set(baseline) | set(optimized))
    mismatches = [
        {
            "record_id": record_id,
            "baseline": list(baseline.get(record_id, ())),
            "optimized": list(optimized.get(record_id, ())),
        }
        for record_id in ids
        if list(baseline.get(record_id, ())) != list(optimized.get(record_id, ()))
    ]
    return {
        "policy": "Exact ordered chunk-ID parity at every compared stage.",
        "compared_query_count": len(ids),
        "exact_parity": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }
