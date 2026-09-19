from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import httpx
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException

from backend.reguaz.retrieval.v2_contract import (
    RetrievalFilters,
    V2RetrievalContract,
)
from backend.reguaz.services.embeddings.v2_models import SparseEmbedding
from backend.reguaz.services.ingestion.point_ids import v2_point_id
from backend.reguaz.utils.logger import get_logger


logger = get_logger(__name__, "retrieval.log")


REQUIRED_PAYLOAD_FIELDS = (
    "chunk_id",
    "content",
    "document_title",
    "canonical_locator",
    "hierarchy",
    "source_file",
    "page_start",
    "page_end",
    "parent_chunk_id",
    "relations",
    "category",
    "document_id",
    "is_current",
)


class QdrantV2Retriever:
    """Read-only dense and learned-sparse access to the validated V2 alias."""

    def __init__(
        self,
        *,
        v2_root: Path = Path("data/processed/v2"),
        qdrant_path: Path = Path("data/processed/v2/qdrant"),
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
        timeout_seconds: float = 30.0,
        alias: str = "reguaz_v2_current",
        contract: V2RetrievalContract | None = None,
        client_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.contract = contract or V2RetrievalContract.load(v2_root)
        self.qdrant_path = qdrant_path.resolve()
        self.qdrant_url = qdrant_url.strip() if qdrant_url else None
        if self.qdrant_url is not None and not self.qdrant_url.startswith(
            ("http://", "https://")
        ):
            raise ValueError("remote Qdrant URL must use http:// or https://")
        if timeout_seconds <= 0:
            raise ValueError("Qdrant timeout must be positive")
        self.alias = alias
        self.mode = "remote" if self.qdrant_url else "local_embedded"
        self._closed = False
        self._qdrant_api_key = qdrant_api_key
        self._timeout_seconds = timeout_seconds
        self._client_factory = client_factory
        self._reconnect_lock = threading.Lock()
        self._client = self._create_client()
        try:
            self.physical_collection = self._validate_collection()
        except Exception:
            self.close()
            raise

    def _validate_collection(self) -> str:
        return str(
            self._remote_read(
                "validate_collection",
                self._validate_collection_with_client,
                attempts=5,
                base_delay_seconds=0.5,
            )
        )

    def _validate_collection_with_client(self, client: Any) -> str:
        aliases = {
            item.alias_name: item.collection_name
            for item in client.get_aliases().aliases
        }
        physical = aliases.get(self.alias)
        if physical is None:
            raise ValueError(f"required Qdrant alias does not exist: {self.alias}")
        if not client.collection_exists(physical):
            raise ValueError(f"Qdrant alias target does not exist: {physical}")

        info = client.get_collection(physical)
        metadata = info.config.metadata or {}
        embedding = self.contract.embedding_manifest
        chunks = self.contract.chunk_manifest
        expected_metadata = {
            "corpus_version": chunks.corpus_version,
            "embedding_model_id": embedding.model_id,
            "embedding_model_revision": embedding.model_revision,
        }
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                raise ValueError(
                    f"Qdrant collection metadata mismatch for {key}: "
                    f"expected {expected!r}, got {metadata.get(key)!r}"
                )

        vectors = info.config.params.vectors
        if not isinstance(vectors, dict) or "dense" not in vectors:
            raise ValueError("Qdrant V2 collection is missing named dense vector")
        dense = vectors["dense"]
        if dense.size != embedding.dense_dimension or dense.size != 1024:
            raise ValueError("Qdrant V2 dense vector dimension mismatch")
        if dense.distance != models.Distance.COSINE:
            raise ValueError("Qdrant V2 dense vector must use cosine distance")
        sparse = info.config.params.sparse_vectors or {}
        if "sparse" not in sparse:
            raise ValueError("Qdrant V2 collection is missing named sparse vector")

        exact_count = client.count(collection_name=physical, exact=True).count
        if exact_count != chunks.child_count:
            raise ValueError(
                "Qdrant V2 point count mismatch: "
                f"expected {chunks.child_count}, got {exact_count}"
            )
        return physical

    def dense_search(
        self,
        query_vector: list[float],
        *,
        top_k: int = 30,
        filters: RetrievalFilters | Mapping[str, object] | None = None,
    ) -> list[dict[str, Any]]:
        if len(query_vector) != self.contract.embedding_manifest.dense_dimension:
            raise ValueError("dense query vector dimension mismatch")
        if any(not math.isfinite(value) for value in query_vector):
            raise ValueError("dense query vector contains NaN or infinity")
        return self._search(
            query=[float(value) for value in query_vector],
            using="dense",
            top_k=top_k,
            filters=filters,
        )

    def sparse_search(
        self,
        query_vector: SparseEmbedding,
        *,
        top_k: int = 30,
        filters: RetrievalFilters | Mapping[str, object] | None = None,
    ) -> list[dict[str, Any]]:
        if not query_vector.indices:
            raise ValueError("sparse query vector must not be empty")
        if any(not math.isfinite(value) for value in query_vector.values):
            raise ValueError("sparse query vector contains NaN or infinity")
        return self._search(
            query=models.SparseVector(
                indices=query_vector.indices,
                values=query_vector.values,
            ),
            using="sparse",
            top_k=top_k,
            filters=filters,
        )

    def batch_search(
        self,
        dense_vector: list[float],
        sparse_vector: SparseEmbedding,
        *,
        dense_top_k: int = 30,
        sparse_top_k: int = 30,
        filters: RetrievalFilters | Mapping[str, object] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Issue dense and sparse requests in one Qdrant batch call."""
        if self._closed:
            raise RuntimeError("Qdrant V2 retriever is closed")
        if len(dense_vector) != self.contract.embedding_manifest.dense_dimension:
            raise ValueError("dense query vector dimension mismatch")
        if not sparse_vector.indices:
            raise ValueError("sparse query vector must not be empty")
        if dense_top_k <= 0 or sparse_top_k <= 0:
            raise ValueError("top_k must be positive")
        normalized = RetrievalFilters.from_value(filters)
        requests = [
            models.QueryRequest(
                query=[float(value) for value in dense_vector],
                using="dense",
                filter=_qdrant_filter(normalized),
                limit=dense_top_k,
                with_payload=["chunk_id"],
                with_vector=False,
            ),
            models.QueryRequest(
                query=models.SparseVector(
                    indices=sparse_vector.indices, values=sparse_vector.values
                ),
                using="sparse",
                filter=_qdrant_filter(normalized),
                limit=sparse_top_k,
                with_payload=["chunk_id"],
                with_vector=False,
            ),
        ]
        try:
            response = self._remote_read(
                "query_batch_points",
                lambda client: client.query_batch_points(
                    collection_name=self.alias,
                    requests=requests,
                ),
                attempts=2,
            )
        except Exception as exc:
            if not self._is_transient_transport_error(exc):
                raise
            logger.warning(
                "Qdrant batch search remained unavailable after retry; "
                "falling back to individual dense and sparse reads"
            )
            self._replace_remote_client(self._client)
            return self.dense_search(
                dense_vector, top_k=dense_top_k, filters=normalized
            ), self.sparse_search(sparse_vector, top_k=sparse_top_k, filters=normalized)
        if len(response) != 2:
            raise RuntimeError("Qdrant V2 batch search returned an unexpected response")
        return self._lean_results(response[0].points), self._lean_results(
            response[1].points
        )

    def _search(
        self,
        *,
        query: list[float] | models.SparseVector,
        using: str,
        top_k: int,
        filters: RetrievalFilters | Mapping[str, object] | None,
    ) -> list[dict[str, Any]]:
        if self._closed:
            raise RuntimeError("Qdrant V2 retriever is closed")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        normalized_filters = RetrievalFilters.from_value(filters)
        response = self._remote_read(
            f"query_points:{using}",
            lambda client: client.query_points(
                collection_name=self.alias,
                query=query,
                using=using,
                query_filter=_qdrant_filter(normalized_filters),
                limit=top_k,
                with_payload=["chunk_id"],
                with_vectors=False,
            ),
        )
        return self._lean_results(response.points)

    def _lean_results(self, points: list[Any]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        for rank, point in enumerate(points, start=1):
            payload = dict(point.payload or {})
            if "chunk_id" not in payload:
                raise ValueError("Qdrant V2 lean payload is missing chunk_id")
            chunk_id = str(payload["chunk_id"])
            if chunk_id in seen:
                raise ValueError(f"duplicate chunk_id returned by Qdrant: {chunk_id}")
            seen.add(chunk_id)
            results.append(
                {
                    "chunk_id": chunk_id,
                    "point_id": str(point.id),
                    "score": float(point.score),
                    "rank": rank,
                }
            )
        return results

    def fetch_full_payloads(
        self, candidates: list[Mapping[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Fetch all final payloads in one request and validate identity mapping."""
        if self._closed:
            raise RuntimeError("Qdrant V2 retriever is closed")
        expected = {
            str(item["chunk_id"]): v2_point_id(str(item["chunk_id"]))
            for item in candidates
        }
        if not expected:
            return {}
        points = self._remote_read(
            "retrieve_final_payloads",
            lambda client: client.retrieve(
                collection_name=self.alias,
                ids=list(expected.values()),
                with_payload=True,
                with_vectors=False,
            ),
        )
        by_id = {str(point.id): point for point in points}
        output: dict[str, dict[str, Any]] = {}
        for chunk_id, point_id in expected.items():
            point = by_id.get(point_id)
            if point is None:
                raise ValueError(f"final V2 point is missing: {chunk_id} ({point_id})")
            payload = dict(point.payload or {})
            missing = [key for key in REQUIRED_PAYLOAD_FIELDS if key not in payload]
            if missing:
                raise ValueError(f"Qdrant V2 payload is missing fields: {missing}")
            if payload.get("schema_version") != "2.0":
                raise ValueError("Qdrant V2 payload schema version mismatch")
            if str(payload.get("chunk_id")) != chunk_id:
                raise ValueError(
                    f"final V2 point/payload mismatch: expected {chunk_id}, got {payload.get('chunk_id')}"
                )
            output[chunk_id] = payload
        return output

    def _create_client(self) -> Any:
        target = self.qdrant_url or str(self.qdrant_path)
        if self._client_factory is not None:
            return self._client_factory(target)
        if self.qdrant_url is not None:
            return QdrantClient(
                url=self.qdrant_url,
                api_key=self._qdrant_api_key,
                timeout=self._timeout_seconds,
                check_compatibility=False,
            )
        return QdrantClient(path=str(self.qdrant_path))

    def _remote_read(
        self,
        operation_name: str,
        operation: Callable[[Any], Any],
        *,
        attempts: int = 3,
        base_delay_seconds: float = 0.2,
    ) -> Any:
        """Retry transient remote transport failures with a fresh HTTP pool."""
        if self.mode != "remote":
            return operation(self._client)
        for attempt in range(1, attempts + 1):
            client = self._client
            try:
                return operation(client)
            except Exception as exc:
                if not self._is_transient_transport_error(exc) or attempt >= attempts:
                    raise
                logger.warning(
                    "Transient Qdrant transport failure operation=%s attempt=%d/%d; "
                    "reconnecting before retry",
                    operation_name,
                    attempt,
                    attempts,
                )
                self._replace_remote_client(client)
                time.sleep(base_delay_seconds * (2 ** (attempt - 1)))
        raise AssertionError("Qdrant retry loop exited unexpectedly")

    def _replace_remote_client(self, failed_client: Any) -> None:
        with self._reconnect_lock:
            if self._closed:
                raise RuntimeError("Qdrant V2 retriever is closed")
            if self._client is not failed_client:
                return
            replacement = self._create_client()
            self._client = replacement
            failed_client.close()

    @staticmethod
    def _is_transient_transport_error(exc: Exception) -> bool:
        current: BaseException | None = exc
        visited: set[int] = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            if isinstance(current, httpx.TransportError):
                return True
            if isinstance(current, ResponseHandlingException):
                current = current.source
                continue
            current = current.__cause__ or current.__context__
        return False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._client.close()

    def __enter__(self) -> QdrantV2Retriever:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _qdrant_filter(filters: RetrievalFilters) -> models.Filter | None:
    conditions: list[models.FieldCondition] = []
    if filters.category is not None:
        conditions.append(
            models.FieldCondition(
                key="category", match=models.MatchValue(value=filters.category)
            )
        )
    if filters.document_id is not None:
        conditions.append(
            models.FieldCondition(
                key="document_id", match=models.MatchValue(value=filters.document_id)
            )
        )
    if filters.is_current is not None:
        conditions.append(
            models.FieldCondition(
                key="is_current", match=models.MatchValue(value=filters.is_current)
            )
        )
    return models.Filter(must=conditions) if conditions else None
