from __future__ import annotations

from pathlib import Path
import time

import pytest

from backend.reguaz.retrieval.hybrid_v2 import (
    HybridV2Retriever,
    HybridV2Settings,
    build_reranker_text,
)
from backend.reguaz.retrieval.v2_contract import RetrievalFilters, V2RetrievalContract
from backend.reguaz.services.embeddings.v2_models import SparseEmbedding


def source_result(chunk_id: str, rank: int, source: str) -> dict[str, object]:
    hierarchy = {"article": {"number": str(rank), "title": "Kapital"}}
    payload = {
        "schema_version": "2.0",
        "chunk_id": chunk_id,
        "content": f"{source} content {chunk_id}",
        "document_title": "Banklar haqqında Qanun",
        "canonical_locator": f"Maddə {rank}",
        "hierarchy": hierarchy,
        "source_file": "Qanun.md",
        "page_start": 1,
        "page_end": 1,
        "parent_chunk_id": "parent-1",
        "relations": {"parent": "parent-1"},
        "category": "laws",
        "document_id": "doc-1",
        "is_current": True,
    }
    return {
        "chunk_id": chunk_id,
        "point_id": f"point-{chunk_id}" if source != "bm25" else None,
        "score": 1.0 / rank,
        "rank": rank,
        "content": payload["content"],
        "document_title": payload["document_title"],
        "canonical_locator": payload["canonical_locator"],
        "hierarchy": hierarchy,
        "source_file": payload["source_file"],
        "page_start": 1,
        "page_end": 1,
        "parent_chunk_id": "parent-1",
        "relations": payload["relations"],
        "payload": payload,
    }


class FakeEncoder:
    model_id = "BAAI/bge-m3"
    model_revision = "test-revision"
    dense_dimension = 1024

    def __init__(self) -> None:
        self.calls: list[str] = []

    def encode_query(
        self, text: str, *, max_length: int = 512
    ) -> tuple[list[float], SparseEmbedding]:
        self.calls.append(text)
        return [1.0] + [0.0] * 1023, SparseEmbedding(indices=[17], values=[0.5])


class FakeQdrant:
    def __init__(self) -> None:
        self.dense_calls: list[tuple[int, RetrievalFilters]] = []
        self.sparse_calls: list[tuple[int, RetrievalFilters]] = []
        self.closed = False
        self.batch_calls = 0
        self.fetch_calls: list[list[str]] = []

    def dense_search(
        self, vector: object, *, top_k: int, filters: RetrievalFilters
    ) -> list[dict[str, object]]:
        self.dense_calls.append((top_k, filters))
        return [
            source_result(f"c{index:02d}", index + 1, "dense") for index in range(30)
        ]

    def sparse_search(
        self, vector: object, *, top_k: int, filters: RetrievalFilters
    ) -> list[dict[str, object]]:
        self.sparse_calls.append((top_k, filters))
        return [
            source_result(f"c{index:02d}", index - 9, "sparse")
            for index in range(10, 40)
        ]

    def batch_search(
        self,
        dense_vector: object,
        sparse_vector: object,
        *,
        dense_top_k: int,
        sparse_top_k: int,
        filters: RetrievalFilters,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        self.batch_calls += 1
        time.sleep(0.03)
        return (
            self.dense_search(dense_vector, top_k=dense_top_k, filters=filters),
            self.sparse_search(sparse_vector, top_k=sparse_top_k, filters=filters),
        )

    def fetch_full_payloads(
        self, candidates: list[dict[str, object]]
    ) -> dict[str, dict[str, object]]:
        self.fetch_calls.append([str(item["chunk_id"]) for item in candidates])
        return {str(item["chunk_id"]): dict(item["payload"]) for item in candidates}

    def close(self) -> None:
        self.closed = True


class FakeBM25:
    def __init__(self, corpus_size: int) -> None:
        self.corpus_size = corpus_size
        self.calls: list[tuple[int, RetrievalFilters]] = []

    def search(
        self, query: str, *, top_k: int, filters: RetrievalFilters
    ) -> list[dict[str, object]]:
        self.calls.append((top_k, filters))
        time.sleep(0.03)
        return [
            source_result(f"c{index:02d}", index - 19, "bm25")
            for index in range(20, 50)
        ]


class FakeReranker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], int | None]] = []

    def rerank(
        self, query: str, documents: list[str], *, batch_size: int | None = None
    ) -> list[float]:
        self.calls.append((query, documents, batch_size))
        return [float(index) for index in range(len(documents))]


def build_hybrid(
    v2_root: Path,
) -> tuple[HybridV2Retriever, FakeEncoder, FakeQdrant, FakeBM25, FakeReranker]:
    contract = V2RetrievalContract.load(v2_root)
    encoder = FakeEncoder()
    qdrant = FakeQdrant()
    bm25 = FakeBM25(contract.chunk_manifest.child_count)
    reranker = FakeReranker()
    hybrid = HybridV2Retriever(
        v2_root=v2_root,
        contract=contract,
        settings=HybridV2Settings(),
        encoder=encoder,
        qdrant_retriever=qdrant,
        bm25_retriever=bm25,
        reranker=reranker,
    )
    return hybrid, encoder, qdrant, bm25, reranker


def test_default_hybrid_pipeline_preserves_intermediates_and_latencies(
    v2_root: Path,
) -> None:
    hybrid, encoder, qdrant, bm25, reranker = build_hybrid(v2_root)
    filters = RetrievalFilters(category="laws", document_id="doc-1")

    run = hybrid.retrieve_with_trace("kapital normativi", filters=filters)

    assert encoder.calls == ["kapital normativi"]
    assert qdrant.dense_calls[0][0] == 30
    assert qdrant.sparse_calls[0][0] == 30
    assert bm25.calls[0][0] == 30
    assert qdrant.dense_calls[0][1] == filters
    assert run["counts"]["dense_result_count"] == 30
    assert run["counts"]["sparse_result_count"] == 30
    assert run["counts"]["bm25_result_count"] == 30
    assert len(run["intermediate"]["fused_candidates"]) == 15
    assert len(reranker.calls[0][1]) == 15
    assert len(run["results"]) == 5
    assert qdrant.batch_calls == 1
    assert len(qdrant.fetch_calls) == 1 and len(qdrant.fetch_calls[0]) == 5
    assert run["timings"]["parallel_retrieval_ms"] < 55
    assert [item["final_rank"] for item in run["results"]] == [1, 2, 3, 4, 5]
    assert all("rrf_rank" in item and "dense_rank" in item for item in run["results"])
    expected_timings = {
        "query_encoding_ms",
        "dense_search_ms",
        "sparse_search_ms",
        "qdrant_batch_ms",
        "bm25_search_ms",
        "parallel_retrieval_ms",
        "fusion_ms",
        "top20_hydration_ms",
        "reranker_ms",
        "final_payload_fetch_ms",
        "parent_lookup_ms",
        "total_retrieval_ms",
    }
    assert set(run["timings"]) == expected_timings
    assert all(value >= 0 for value in run["timings"].values())
    assert run["overlap"]["all_three"] == 10

    hybrid.close()
    assert qdrant.closed is True


def test_rerank_15_vs_20_reuses_same_retrieval(v2_root: Path) -> None:
    hybrid, encoder, qdrant, bm25, reranker = build_hybrid(v2_root)

    comparison = hybrid.compare_rerank_depths("likvidlik", depths=(15, 20))

    assert encoder.calls == ["likvidlik"]
    assert len(qdrant.dense_calls) == len(qdrant.sparse_calls) == len(bm25.calls) == 1
    assert [len(call[1]) for call in reranker.calls] == [15, 20]
    assert [item["actual_reranked_count"] for item in comparison["comparisons"]] == [
        15,
        20,
    ]
    assert all(len(item["results"]) == 5 for item in comparison["comparisons"])
    hybrid.close()


def test_hybrid_rejects_empty_query_before_encoding(v2_root: Path) -> None:
    hybrid, encoder, _, _, _ = build_hybrid(v2_root)
    with pytest.raises(ValueError, match="query must not be empty"):
        hybrid.retrieve("   ")
    assert encoder.calls == []
    hybrid.close()


def test_compact_reranker_text_excludes_embedding_text() -> None:
    candidate = source_result("c1", 1, "dense")
    candidate["payload"]["embedding_text"] = "must not be included"
    text = build_reranker_text(candidate)
    assert text.startswith("Document: Banklar haqqında Qanun\nLocator: Maddə 1")
    assert "Hierarchy: article=1 — Kapital" in text
    assert "Content: dense content c1" in text
    assert "must not be included" not in text
