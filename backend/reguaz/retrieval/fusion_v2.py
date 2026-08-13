from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


SOURCE_NAMES = ("dense", "sparse", "bm25")


def fuse_v2_results(
    *,
    dense_results: Sequence[Mapping[str, Any]],
    sparse_results: Sequence[Mapping[str, Any]],
    bm25_results: Sequence[Mapping[str, Any]],
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    bm25_weight: float = 1.0,
    rrf_k: int = 60,
    candidate_count: int | None = 20,
) -> list[dict[str, Any]]:
    """Fuse three ranked lists using explicit weighted reciprocal ranks."""
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    if candidate_count is not None and candidate_count <= 0:
        raise ValueError("candidate_count must be positive or null")
    weights = {
        "dense": float(dense_weight),
        "sparse": float(sparse_weight),
        "bm25": float(bm25_weight),
    }
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("RRF weights must be non-negative")

    sources = {
        "dense": dense_results,
        "sparse": sparse_results,
        "bm25": bm25_results,
    }
    candidates: dict[str, dict[str, Any]] = {}
    for source_name in SOURCE_NAMES:
        seen_in_source: set[str] = set()
        ordered = sorted(
            sources[source_name],
            key=lambda result: (int(result["rank"]), str(result["chunk_id"])),
        )
        for result in ordered:
            chunk_id = str(result["chunk_id"])
            if chunk_id in seen_in_source:
                continue
            seen_in_source.add(chunk_id)
            rank = int(result["rank"])
            if rank <= 0:
                raise ValueError("source ranks must be one-based positive integers")
            candidate = candidates.setdefault(chunk_id, _new_candidate(result))
            candidate[f"{source_name}_rank"] = rank
            candidate[f"{source_name}_score"] = float(result["score"])
            contribution = weights[source_name] / (rrf_k + rank)
            candidate["rrf_contributions"][source_name] = contribution
            candidate["retrievers"].append(source_name)
            candidate["rrf_score"] += contribution
            if candidate["point_id"] is None and result.get("point_id") is not None:
                candidate["point_id"] = str(result["point_id"])

    fused = sorted(
        candidates.values(),
        key=lambda candidate: (-candidate["rrf_score"], candidate["chunk_id"]),
    )
    for rank, candidate in enumerate(fused, start=1):
        candidate["rrf_rank"] = rank
    return fused if candidate_count is None else fused[:candidate_count]


def _new_candidate(result: Mapping[str, Any]) -> dict[str, Any]:
    candidate = {
        "chunk_id": str(result["chunk_id"]),
        "point_id": (
            str(result["point_id"]) if result.get("point_id") is not None else None
        ),
        "retrievers": [],
        "rrf_contributions": {"dense": None, "sparse": None, "bm25": None},
        "dense_rank": None,
        "dense_score": None,
        "sparse_rank": None,
        "sparse_score": None,
        "bm25_rank": None,
        "bm25_score": None,
        "rrf_rank": 0,
        "rrf_score": 0.0,
    }
    # Compatibility for injected/custom retrievers; production V2 sources are lean.
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
        "payload",
    ):
        if key in result:
            candidate[key] = result[key]
    return candidate
