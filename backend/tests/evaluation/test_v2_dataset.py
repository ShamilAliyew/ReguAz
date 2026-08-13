from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from backend.reguaz.evaluation.v2_dataset import EvaluationRecord, validate_dataset


ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "data/evaluation/reguaz_v2_evaluation_50.jsonl"
MANIFEST = ROOT / "data/evaluation/reguaz_v2_evaluation_50_manifest.json"


def test_frozen_v2_dataset_schema_and_exact_distributions() -> None:
    records = [
        EvaluationRecord.model_validate_json(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 50
    assert Counter(record.split for record in records) == {"dev": 10, "test": 40}
    assert Counter(record.difficulty for record in records) == {
        "easy": 20,
        "medium": 20,
        "hard": 10,
    }
    assert len({record.id for record in records}) == 50
    assert len({record.category for record in records}) == 8
    assert any(record.metadata.requires_multiple_chunks for record in records)


def test_frozen_v2_dataset_matches_authoritative_artifacts() -> None:
    result = validate_dataset(DATASET, MANIFEST, v2_root=ROOT / "data/processed/v2")
    assert result["status"] == "valid"
    assert result["record_count"] == 50
    assert (
        result["dataset_sha256"]
        == json.loads(MANIFEST.read_text())["dataset_jsonl_sha256"]
    )
