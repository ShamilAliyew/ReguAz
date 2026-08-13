from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from backend.reguaz.retrieval.v2_contract import (
    RetrievalFilters,
    V2RetrievalContract,
)
from backend.reguaz.retrieval.v2_artifacts import V2ArtifactStore


TOKEN_PATTERN = re.compile(
    r"\d+(?:[-.]\d+)*|[^\W\d_]+(?:['’][^\W\d_]+)*",
    flags=re.UNICODE,
)


def tokenize_az_legal(text: str) -> list[str]:
    """Deterministically tokenize Azerbaijani legal prose and locators."""
    normalized = unicodedata.normalize("NFC", text).casefold()
    normalized = unicodedata.normalize("NFC", normalized.replace("i\u0307", "i"))
    return TOKEN_PATTERN.findall(normalized)


class BM25V2Retriever:
    """BM25 over V2 child ``content`` only; parent and embedding text are excluded."""

    def __init__(
        self,
        *,
        v2_root: Path = Path("data/processed/v2"),
        contract: V2RetrievalContract | None = None,
        artifact_store: V2ArtifactStore | None = None,
    ) -> None:
        self.contract = contract or V2RetrievalContract.load(v2_root)
        self.artifact_store = artifact_store or V2ArtifactStore(
            v2_root=v2_root, contract=self.contract
        )
        self._chunk_ids: list[str] = []
        self._contents: list[str] = []
        self._filter_fields: list[dict[str, Any]] = []
        self._load_children()
        tokenized = [tokenize_az_legal(content) for content in self._contents]
        self.empty_token_document_count = sum(not tokens for tokens in tokenized)
        self._index = BM25Okapi(tokenized)

    def _load_children(self) -> None:
        seen: set[str] = set()
        for child in self.artifact_store.iter_children():
            chunk_id = str(child["chunk_id"])
            if chunk_id in seen:
                raise ValueError(f"duplicate V2 BM25 chunk_id: {chunk_id}")
            seen.add(chunk_id)
            self._chunk_ids.append(chunk_id)
            self._contents.append(str(child["content"]))
            self._filter_fields.append(
                {
                    "category": child["category"],
                    "document_id": child["document_id"],
                    "is_current": child["is_current"],
                }
            )

        expected = self.contract.chunk_manifest.child_count
        if len(self._chunk_ids) != expected:
            raise ValueError(
                f"V2 BM25 child count mismatch: expected {expected}, got {len(self._chunk_ids)}"
            )

    @property
    def corpus_size(self) -> int:
        return len(self._chunk_ids)

    def get_payload(self, chunk_id: str) -> dict[str, Any] | None:
        return self.artifact_store.get_child(chunk_id)

    def search(
        self,
        query: str,
        *,
        top_k: int = 30,
        filters: RetrievalFilters | Mapping[str, object] | None = None,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        tokens = tokenize_az_legal(query)
        if not tokens:
            raise ValueError("query produced no BM25 tokens")
        normalized_filters = RetrievalFilters.from_value(filters)
        scores = self._index.get_scores(tokens).tolist()
        eligible = [
            index
            for index, payload in enumerate(self._filter_fields)
            if _matches_filters(payload, normalized_filters)
        ]
        ranked = sorted(
            eligible,
            key=lambda index: (-float(scores[index]), self._chunk_ids[index]),
        )[:top_k]
        results: list[dict[str, Any]] = []
        for rank, index in enumerate(ranked, start=1):
            results.append(
                {
                    "chunk_id": self._chunk_ids[index],
                    "point_id": None,
                    "score": float(scores[index]),
                    "rank": rank,
                }
            )
        return results


def _matches_filters(payload: dict[str, Any], filters: RetrievalFilters) -> bool:
    if filters.category is not None and payload.get("category") != filters.category:
        return False
    if (
        filters.document_id is not None
        and payload.get("document_id") != filters.document_id
    ):
        return False
    return filters.is_current is None or payload.get("is_current") is filters.is_current
