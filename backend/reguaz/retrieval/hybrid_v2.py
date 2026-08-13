from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from backend.reguaz.retrieval.bm25_v2_retriever import BM25V2Retriever
from backend.reguaz.retrieval.fusion_v2 import fuse_v2_results
from backend.reguaz.retrieval.qdrant_v2_retriever import QdrantV2Retriever
from backend.reguaz.retrieval.reranker import CrossEncoderReranker
from backend.reguaz.retrieval.v2_contract import (
    RetrievalFilters,
    V2RetrievalContract,
)
from backend.reguaz.retrieval.v2_artifacts import V2ArtifactStore
from backend.reguaz.services.embeddings.bge_m3_v2 import BgeM3DenseSparseEncoder
from backend.reguaz.services.embeddings.v2_models import SparseEmbedding


DEFAULT_V2_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_V2_RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


class QueryEncoder(Protocol):
    model_id: str
    model_revision: str
    dense_dimension: int

    def encode_query(
        self, text: str, *, max_length: int = 512
    ) -> tuple[list[float], SparseEmbedding]: ...


class Reranker(Protocol):
    def rerank(
        self, query: str, documents: list[str], *, batch_size: int | None = None
    ) -> list[float]: ...


@dataclass(frozen=True, slots=True)
class HybridV2Settings:
    dense_top_k: int = 30
    sparse_top_k: int = 30
    bm25_top_k: int = 30
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    bm25_weight: float = 1.0
    rrf_k: int = 60
    rerank_top_k: int = 15
    final_top_k: int = 5
    query_max_length: int = 512
    reranker_batch_size: int = 8
    retrieval_strategy: str = "qdrant_batch_bm25_overlap"

    def __post_init__(self) -> None:
        positive = (
            self.dense_top_k,
            self.sparse_top_k,
            self.bm25_top_k,
            self.rrf_k,
            self.rerank_top_k,
            self.final_top_k,
            self.query_max_length,
            self.reranker_batch_size,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("hybrid V2 count and length settings must be positive")
        if self.final_top_k > self.rerank_top_k:
            raise ValueError("final_top_k cannot exceed rerank_top_k")
        if any(
            weight < 0
            for weight in (self.dense_weight, self.sparse_weight, self.bm25_weight)
        ):
            raise ValueError("hybrid V2 source weights must be non-negative")
        allowed = {"sequential", "qdrant_batch_bm25_overlap", "concurrent_all"}
        if self.retrieval_strategy not in allowed:
            raise ValueError(
                f"unsupported V2 retrieval strategy: {self.retrieval_strategy}"
            )


class HybridV2Retriever:
    """Dense + learned sparse + BM25, equal-weight RRF, then cross-encoder."""

    def __init__(
        self,
        *,
        v2_root: Path = Path("data/processed/v2"),
        qdrant_path: Path = Path("data/processed/v2/qdrant"),
        alias: str = "reguaz_v2_current",
        settings: HybridV2Settings | None = None,
        device: str | None = None,
        local_files_only: bool = False,
        reranker_model: str = DEFAULT_V2_RERANKER_MODEL,
        reranker_revision: str | None = DEFAULT_V2_RERANKER_REVISION,
        reranker_trust_remote_code: bool = False,
        reranker_code_revision: str | None = None,
        contract: V2RetrievalContract | None = None,
        encoder: QueryEncoder | None = None,
        qdrant_retriever: Any | None = None,
        bm25_retriever: Any | None = None,
        reranker: Reranker | None = None,
        artifact_store: V2ArtifactStore | None = None,
    ) -> None:
        started = time.perf_counter()
        self.settings = settings or HybridV2Settings()
        self.contract = contract or V2RetrievalContract.load(v2_root)
        self._closed = False
        self._qdrant: Any | None = None
        try:
            self._encoder = encoder or BgeM3DenseSparseEncoder(
                device=device,
                local_files_only=local_files_only,
                model_revision=self.contract.embedding_manifest.model_revision,
            )
            self._validate_encoder()
            self._qdrant = qdrant_retriever or QdrantV2Retriever(
                v2_root=v2_root,
                qdrant_path=qdrant_path,
                alias=alias,
                contract=self.contract,
            )
            self._artifacts = (
                artifact_store
                or getattr(bm25_retriever, "artifact_store", None)
                or V2ArtifactStore(v2_root=v2_root, contract=self.contract)
            )
            self._bm25 = bm25_retriever or BM25V2Retriever(
                v2_root=v2_root,
                contract=self.contract,
                artifact_store=self._artifacts,
            )
            if self._bm25.corpus_size != self.contract.chunk_manifest.child_count:
                raise ValueError("V2 BM25 corpus size does not match chunk manifest")
            self._reranker = reranker or CrossEncoderReranker(
                model_name=reranker_model,
                device=device,
                batch_size=self.settings.reranker_batch_size,
                revision=reranker_revision,
                local_files_only=local_files_only,
                trust_remote_code=reranker_trust_remote_code,
                code_revision=reranker_code_revision,
                show_progress_bar=False,
            )
        except Exception:
            self.close()
            raise
        self.startup_ms = _elapsed_ms(started)
        self.last_run: dict[str, Any] | None = None

    def _validate_encoder(self) -> None:
        manifest = self.contract.embedding_manifest
        if self._encoder.model_id != manifest.model_id:
            raise ValueError("V2 query encoder model ID mismatch")
        if self._encoder.model_revision != manifest.model_revision:
            raise ValueError("V2 query encoder model revision mismatch")
        if self._encoder.dense_dimension != manifest.dense_dimension:
            raise ValueError("V2 query encoder dense dimension mismatch")

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: RetrievalFilters | Mapping[str, object] | None = None,
    ) -> list[dict[str, Any]]:
        run = self.retrieve_with_trace(query, top_k=top_k, filters=filters)
        return run["results"]

    def retrieve_with_trace(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: RetrievalFilters | Mapping[str, object] | None = None,
    ) -> dict[str, Any]:
        final_top_k = top_k if top_k is not None else self.settings.final_top_k
        if final_top_k <= 0 or final_top_k > self.settings.rerank_top_k:
            raise ValueError("top_k must be positive and no larger than rerank_top_k")
        base = self._retrieve_base(query, filters)
        run = self._complete_run(
            base,
            rerank_top_k=self.settings.rerank_top_k,
            final_top_k=final_top_k,
        )
        self.last_run = run
        return run

    def compare_rerank_depths(
        self,
        query: str,
        *,
        depths: Sequence[int] = (15, 20),
        final_top_k: int | None = None,
        filters: RetrievalFilters | Mapping[str, object] | None = None,
    ) -> dict[str, Any]:
        if not depths or any(depth <= 0 for depth in depths):
            raise ValueError("rerank comparison depths must be positive")
        base = self._retrieve_base(query, filters)
        available = len(base["intermediate"]["fused_all"])
        final = final_top_k or self.settings.final_top_k
        comparisons = []
        for depth in depths:
            if final > depth:
                raise ValueError("final_top_k cannot exceed a comparison depth")
            run = self._complete_run(
                base,
                rerank_top_k=min(depth, available),
                final_top_k=final,
            )
            comparisons.append(
                {
                    "rerank_top_k": depth,
                    "actual_reranked_count": run["counts"]["reranked_candidate_count"],
                    "reranker_ms": run["timings"]["reranker_ms"],
                    "total_retrieval_ms": run["timings"]["total_retrieval_ms"],
                    "results": run["results"],
                }
            )
        output = {
            "query": query,
            "filters": base["filters"],
            "startup_ms": self.startup_ms,
            "shared_retrieval": {
                "timings": base["timings"],
                "counts": base["counts"],
                "overlap": base["overlap"],
                "conditions": base["conditions"],
                "intermediate": base["intermediate"],
            },
            "comparisons": comparisons,
        }
        self.last_run = output
        return output

    def _retrieve_base(
        self,
        query: str,
        filters: RetrievalFilters | Mapping[str, object] | None,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Hybrid V2 retriever is closed")
        if not query.strip():
            raise ValueError("query must not be empty")
        normalized_filters = RetrievalFilters.from_value(filters)
        started = time.perf_counter()

        stage = time.perf_counter()
        dense_vector, sparse_vector = self._encoder.encode_query(
            query, max_length=self.settings.query_max_length
        )
        query_encoding_ms = _elapsed_ms(stage)

        retrieval_started = time.perf_counter()
        dense, sparse, bm25, branch_timings = self._search_sources(
            query, dense_vector, sparse_vector, normalized_filters
        )
        parallel_retrieval_ms = _elapsed_ms(retrieval_started)

        stage = time.perf_counter()
        fused_all = fuse_v2_results(
            dense_results=dense,
            sparse_results=sparse,
            bm25_results=bm25,
            dense_weight=self.settings.dense_weight,
            sparse_weight=self.settings.sparse_weight,
            bm25_weight=self.settings.bm25_weight,
            rrf_k=self.settings.rrf_k,
            candidate_count=None,
        )
        fusion_ms = _elapsed_ms(stage)
        dense_ids = [result["chunk_id"] for result in dense]
        sparse_ids = [result["chunk_id"] for result in sparse]
        bm25_ids = [result["chunk_id"] for result in bm25]
        counts = {
            "dense_result_count": len(dense),
            "sparse_result_count": len(sparse),
            "bm25_result_count": len(bm25),
            "fused_unique_candidate_count": len(fused_all),
        }
        conditions = {
            "bm25_empty_token_document_count": getattr(
                self._bm25, "empty_token_document_count", 0
            ),
            "duplicate_counts": {
                "dense": len(dense_ids) - len(set(dense_ids)),
                "sparse": len(sparse_ids) - len(set(sparse_ids)),
                "bm25": len(bm25_ids) - len(set(bm25_ids)),
            },
            "empty_sources": [
                name
                for name, values in (
                    ("dense", dense),
                    ("sparse", sparse),
                    ("bm25", bm25),
                )
                if not values
            ],
            "fused_empty": not fused_all,
        }
        return {
            "query": query,
            "filters": normalized_filters.as_dict(),
            "started": started,
            "base_retrieval_ms": _elapsed_ms(started),
            "timings": {
                "query_encoding_ms": query_encoding_ms,
                **branch_timings,
                "parallel_retrieval_ms": parallel_retrieval_ms,
                "fusion_ms": fusion_ms,
            },
            "counts": counts,
            "overlap": _source_overlap(dense_ids, sparse_ids, bm25_ids),
            "conditions": conditions,
            "intermediate": {
                "dense": dense,
                "sparse": sparse,
                "bm25": bm25,
                "fused_all": fused_all,
            },
        }

    def _search_sources(
        self,
        query: str,
        dense_vector: list[float],
        sparse_vector: SparseEmbedding,
        filters: RetrievalFilters,
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        dict[str, float],
    ]:
        def dense_call() -> tuple[list[dict[str, Any]], float]:
            started = time.perf_counter()
            value = self._qdrant.dense_search(
                dense_vector, top_k=self.settings.dense_top_k, filters=filters
            )
            return value, _elapsed_ms(started)

        def sparse_call() -> tuple[list[dict[str, Any]], float]:
            started = time.perf_counter()
            value = self._qdrant.sparse_search(
                sparse_vector, top_k=self.settings.sparse_top_k, filters=filters
            )
            return value, _elapsed_ms(started)

        def bm25_call() -> tuple[list[dict[str, Any]], float]:
            started = time.perf_counter()
            value = self._bm25.search(
                query, top_k=self.settings.bm25_top_k, filters=filters
            )
            return value, _elapsed_ms(started)

        strategy = self.settings.retrieval_strategy
        if strategy == "sequential":
            dense, dense_ms = dense_call()
            sparse, sparse_ms = sparse_call()
            bm25, bm25_ms = bm25_call()
            return (
                dense,
                sparse,
                bm25,
                {
                    "dense_search_ms": dense_ms,
                    "sparse_search_ms": sparse_ms,
                    "bm25_search_ms": bm25_ms,
                },
            )
        if strategy == "qdrant_batch_bm25_overlap" and hasattr(
            self._qdrant, "batch_search"
        ):

            def qdrant_batch() -> (
                tuple[tuple[list[dict[str, Any]], list[dict[str, Any]]], float]
            ):
                started = time.perf_counter()
                value = self._qdrant.batch_search(
                    dense_vector,
                    sparse_vector,
                    dense_top_k=self.settings.dense_top_k,
                    sparse_top_k=self.settings.sparse_top_k,
                    filters=filters,
                )
                return value, _elapsed_ms(started)

            with ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="reguaz-v2"
            ) as pool:
                qdrant_future = pool.submit(qdrant_batch)
                bm25_future = pool.submit(bm25_call)
                (dense, sparse), qdrant_ms = qdrant_future.result()
                bm25, bm25_ms = bm25_future.result()
            return (
                dense,
                sparse,
                bm25,
                {
                    "dense_search_ms": qdrant_ms,
                    "sparse_search_ms": qdrant_ms,
                    "bm25_search_ms": bm25_ms,
                    "qdrant_batch_ms": qdrant_ms,
                },
            )
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="reguaz-v2") as pool:
            dense_future = pool.submit(dense_call)
            sparse_future = pool.submit(sparse_call)
            bm25_future = pool.submit(bm25_call)
            dense, dense_ms = dense_future.result()
            sparse, sparse_ms = sparse_future.result()
            bm25, bm25_ms = bm25_future.result()
        return (
            dense,
            sparse,
            bm25,
            {
                "dense_search_ms": dense_ms,
                "sparse_search_ms": sparse_ms,
                "bm25_search_ms": bm25_ms,
            },
        )

    def _complete_run(
        self,
        base: dict[str, Any],
        *,
        rerank_top_k: int,
        final_top_k: int,
    ) -> dict[str, Any]:
        lean_candidates = base["intermediate"]["fused_all"][:rerank_top_k]
        stage = time.perf_counter()
        candidates = self._artifacts.hydrate_for_reranker(lean_candidates)
        hydration_ms = _elapsed_ms(stage)
        reranker_texts = [build_reranker_text(candidate) for candidate in candidates]
        stage = time.perf_counter()
        scores = self._reranker.rerank(
            base["query"],
            reranker_texts,
            batch_size=self.settings.reranker_batch_size,
        )
        reranker_ms = _elapsed_ms(stage)
        if len(scores) != len(candidates):
            raise ValueError("reranker score count does not match candidate count")
        reranked = []
        for candidate, score in zip(candidates, scores, strict=True):
            item = dict(candidate)
            item["reranker_score"] = float(score)
            reranked.append(item)
        reranked.sort(
            key=lambda candidate: (-candidate["reranker_score"], candidate["chunk_id"])
        )
        for rank, candidate in enumerate(reranked, start=1):
            candidate["reranker_rank"] = rank

        finalists = reranked[:final_top_k]
        stage = time.perf_counter()
        if hasattr(self._qdrant, "fetch_full_payloads"):
            full_payloads = self._qdrant.fetch_full_payloads(finalists)
        else:
            full_payloads = {
                str(item["chunk_id"]): (
                    self._artifacts.get_child(str(item["chunk_id"]))
                    or item.get("payload")
                )
                for item in finalists
            }
        final_payload_fetch_ms = _elapsed_ms(stage)

        results = []
        for rank, candidate in enumerate(finalists, start=1):
            item = dict(candidate)
            payload = full_payloads.get(str(item["chunk_id"]))
            if payload is None:
                raise ValueError(f"final V2 payload is missing: {item['chunk_id']}")
            item["payload"] = payload
            for key in (
                "content",
                "document_title",
                "canonical_locator",
                "hierarchy",
                "source_file",
                "page_start",
                "page_end",
                "parent_chunk_id",
                "relations",
            ):
                item[key] = payload.get(key)
            item["final_rank"] = rank
            results.append(item)

        stage = time.perf_counter()
        parents, parent_warnings = self._artifacts.resolve_parents(results)
        for item in results:
            parent_id = item["payload"].get("parent_chunk_id")
            item["parent"] = parents.get(str(parent_id)) if parent_id else None
        parent_lookup_ms = _elapsed_ms(stage)

        timings = dict(base["timings"])
        timings["top20_hydration_ms"] = hydration_ms
        timings["reranker_ms"] = reranker_ms
        timings["final_payload_fetch_ms"] = final_payload_fetch_ms
        timings["parent_lookup_ms"] = parent_lookup_ms
        timings["total_retrieval_ms"] = (
            base["base_retrieval_ms"]
            + hydration_ms
            + reranker_ms
            + final_payload_fetch_ms
            + parent_lookup_ms
        )
        counts = dict(base["counts"])
        counts.update(
            {
                "reranked_candidate_count": len(candidates),
                "final_result_count": len(results),
                "resolved_parent_count": len(parents),
            }
        )
        return {
            "query": base["query"],
            "filters": base["filters"],
            "startup_ms": self.startup_ms,
            "config": {
                "dense_top_k": self.settings.dense_top_k,
                "sparse_top_k": self.settings.sparse_top_k,
                "bm25_top_k": self.settings.bm25_top_k,
                "dense_weight": self.settings.dense_weight,
                "sparse_weight": self.settings.sparse_weight,
                "bm25_weight": self.settings.bm25_weight,
                "rrf_k": self.settings.rrf_k,
                "rerank_top_k": rerank_top_k,
                "final_top_k": final_top_k,
                "concurrency_strategy": self.settings.retrieval_strategy,
            },
            "runtime": {
                "query_encoder_device": str(
                    getattr(self._encoder, "device", "unknown")
                ),
                "reranker_device": str(getattr(self._reranker, "device", "unknown")),
                "reranker_model": str(
                    getattr(self._reranker, "model_name", "injected")
                ),
                "reranker_revision": getattr(self._reranker, "model_revision", None),
                "reranker_code_revision": getattr(
                    self._reranker, "code_revision", None
                ),
                "qdrant_device": "cpu",
                "bm25_device": "cpu",
                "concurrency_strategy": self.settings.retrieval_strategy,
            },
            "timings": timings,
            "counts": counts,
            "overlap": base["overlap"],
            "conditions": {**base["conditions"], "parent_warnings": parent_warnings},
            "intermediate": {
                **base["intermediate"],
                "fused_candidates": candidates,
                "reranked_candidates": reranked,
            },
            "results": results,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._qdrant is not None:
            self._qdrant.close()

    def __enter__(self) -> HybridV2Retriever:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def build_reranker_text(candidate: Mapping[str, Any]) -> str:
    return (
        f"Document: {candidate['document_title']}\n"
        f"Locator: {candidate['canonical_locator']}\n"
        f"Hierarchy: {_compact_hierarchy(candidate.get('hierarchy'))}\n"
        f"Content: {candidate['content']}"
    )


def _compact_hierarchy(value: object) -> str:
    if not isinstance(value, Mapping):
        return ""
    parts: list[str] = []
    for key in (
        "division",
        "chapter",
        "article",
        "section",
        "clause",
        "subclause",
        "item",
        "annex",
    ):
        current = value.get(key)
        if isinstance(current, Mapping):
            label = " — ".join(
                str(item)
                for item in (current.get("number"), current.get("title"))
                if item
            )
        else:
            label = str(current) if current else ""
        if label:
            parts.append(f"{key}={label}")
    return " | ".join(parts)


def _source_overlap(
    dense_ids: Sequence[str], sparse_ids: Sequence[str], bm25_ids: Sequence[str]
) -> dict[str, int]:
    dense = set(dense_ids)
    sparse = set(sparse_ids)
    bm25 = set(bm25_ids)
    return {
        "dense_sparse": len(dense & sparse),
        "dense_bm25": len(dense & bm25),
        "sparse_bm25": len(sparse & bm25),
        "all_three": len(dense & sparse & bm25),
    }


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0
