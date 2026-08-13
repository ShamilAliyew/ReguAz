from __future__ import annotations

import json
from pathlib import Path

from backend.reguaz.retrieval.v2_artifacts import V2ArtifactStore


def test_lean_hydration_and_lazy_parent_resolution(v2_root: Path) -> None:
    store = V2ArtifactStore(v2_root=v2_root)
    path = next((v2_root / "documents").glob("*/chunks.jsonl"))
    child = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    hydrated = store.hydrate_for_reranker(
        [{"chunk_id": child["chunk_id"], "rrf_rank": 1}]
    )
    assert set(("content", "document_title", "canonical_locator", "hierarchy")) <= set(
        hydrated[0]
    )
    assert "embedding_text" not in hydrated[0]

    payload = store.get_child(child["chunk_id"])
    parents, warnings = store.resolve_parents([{"payload": payload}])
    assert payload["parent_chunk_id"] in parents
    assert warnings == []


def test_missing_parent_is_reported_structurally(v2_root: Path) -> None:
    store = V2ArtifactStore(v2_root=v2_root)
    _, warnings = store.resolve_parents(
        [
            {
                "payload": {
                    "parent_chunk_id": "missing-parent",
                    "document_id": "missing-doc",
                }
            }
        ]
    )
    assert warnings == [
        {
            "code": "missing_parent",
            "parent_chunk_id": "missing-parent",
            "document_id": "missing-doc",
        }
    ]
