from __future__ import annotations

import pytest

from backend.reguaz.retrieval.fusion_v2 import fuse_v2_results


def result(chunk_id: str, rank: int, score: float = 1.0) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "point_id": f"point-{chunk_id}",
        "rank": rank,
        "score": score,
        "content": f"content {chunk_id}",
        "document_title": "Qanun",
        "canonical_locator": f"Maddə {rank}",
        "hierarchy": {"article": {"number": str(rank), "title": None}},
        "source_file": "Qanun.md",
        "page_start": 1,
        "page_end": 1,
        "parent_chunk_id": "parent-1",
        "relations": {"parent": "parent-1"},
        "payload": {"chunk_id": chunk_id, "content": f"content {chunk_id}"},
    }


def test_rrf_exact_formula_and_missing_source_fields() -> None:
    fused = fuse_v2_results(
        dense_results=[result("a", 1), result("b", 2)],
        sparse_results=[result("b", 1), result("c", 2)],
        bm25_results=[result("a", 1), result("c", 2)],
        rrf_k=60,
        candidate_count=None,
    )
    by_id = {item["chunk_id"]: item for item in fused}

    assert by_id["a"]["rrf_score"] == pytest.approx(2 / 61)
    assert by_id["b"]["rrf_score"] == pytest.approx(1 / 61 + 1 / 62)
    assert by_id["a"]["sparse_rank"] is None
    assert by_id["b"]["bm25_rank"] is None
    assert by_id["c"]["dense_score"] is None


def test_rrf_deduplicates_within_and_across_sources() -> None:
    fused = fuse_v2_results(
        dense_results=[result("a", 1), result("a", 2)],
        sparse_results=[result("a", 1)],
        bm25_results=[],
        rrf_k=60,
        candidate_count=None,
    )
    assert len(fused) == 1
    assert fused[0]["rrf_score"] == pytest.approx(2 / 61)
    assert fused[0]["dense_rank"] == 1


def test_rrf_ties_break_by_chunk_id_deterministically() -> None:
    fused = fuse_v2_results(
        dense_results=[result("z", 1)],
        sparse_results=[result("a", 1)],
        bm25_results=[],
        candidate_count=None,
    )
    assert [item["chunk_id"] for item in fused] == ["a", "z"]


def test_rrf_default_selects_top_twenty() -> None:
    fused = fuse_v2_results(
        dense_results=[result(f"c{index:02d}", index + 1) for index in range(30)],
        sparse_results=[],
        bm25_results=[],
    )
    assert len(fused) == 20
    assert [item["rrf_rank"] for item in fused] == list(range(1, 21))
