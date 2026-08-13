from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.reguaz.evaluation.retrieval_v2 import evaluate_ranking, latency_summary
from backend.reguaz.evaluation.v2_dataset import EvaluationRecord


class FakeStore:
    def __init__(self, records: dict[str, dict[str, str]]) -> None:
        self.records = records

    def get_child(self, chunk_id: str) -> dict[str, str] | None:
        return self.records.get(chunk_id)


def test_multi_evidence_recall_and_ranking_metrics() -> None:
    root = Path(__file__).resolve().parents[3]
    record = next(
        EvaluationRecord.model_validate_json(line)
        for line in (root / "data/evaluation/reguaz_v2_evaluation_50.jsonl")
        .read_text()
        .splitlines()
        if len(json.loads(line)["relevant_chunk_ids"]) > 1
    )
    first, second = record.evidence[:2]
    store = FakeStore(
        {
            first.chunk_id: {
                "logical_chunk_id": first.logical_chunk_id,
                "document_version_id": first.document_version_id,
            },
            second.chunk_id: {
                "logical_chunk_id": second.logical_chunk_id,
                "document_version_id": second.document_version_id,
            },
        }
    )
    metrics = evaluate_ranking(
        ["irrelevant", first.chunk_id, second.chunk_id], record, store, cutoffs=(2, 3)
    )
    assert metrics["hit_rate@2"] == 1.0
    assert metrics["mrr@2"] == pytest.approx(0.5)
    assert metrics["recall@2"] == pytest.approx(1 / len(record.relevant_chunk_ids))
    assert metrics["recall@3"] == pytest.approx(2 / len(record.relevant_chunk_ids))


def test_latency_summary_exact_percentiles() -> None:
    assert latency_summary([1, 2, 3, 4, 100]) == {
        "mean": 22.0,
        "median": 3.0,
        "p95": 100.0,
        "min": 1.0,
        "max": 100.0,
    }
