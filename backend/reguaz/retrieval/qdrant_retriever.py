"""
backend/reguaz/retrieval/qdrant_retriever.py

Lightweight wrapper around a local Qdrant collection.
Performs pure vector-search; embedding generation is the caller's responsibility.

This module is the Qdrant equivalent of retriever.py (ChromaRetriever).
The public API, constructor signature, return contract, validation strategy,
and coding style deliberately mirror ChromaRetriever so that both classes
are immediately recognisable as parallel implementations of the same role.
"""

from __future__ import annotations

import logging
from typing import Any

# pyrefly: ignore [missing-import]
from qdrant_client import QdrantClient
# pyrefly: ignore [missing-import]
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger(__name__)


class QdrantRetriever:
    """
    Opens an existing local Qdrant collection and exposes a vector-search method.

    This class is the Qdrant equivalent of ChromaRetriever.  The constructor
    signature, the :meth:`search` return contract, and the validation checks
    are intentionally identical to ChromaRetriever so that either class can
    be used interchangeably by the hybrid retriever layer.

    Parameters
    ----------
    persist_directory : str
        Path to the local Qdrant storage directory (e.g. ``"data/qdrant"``).
    collection_name : str
        Name of the collection to open (e.g. ``"reguaz_bge_m3"``).
    """

    def __init__(self, persist_directory: str, collection_name: str) -> None:
        logger.info(
            "QdrantRetriever: opening collection '%s' from '%s'",
            collection_name,
            persist_directory,
        )
        self._collection_name = collection_name
        self._client = QdrantClient(path=str(persist_directory))

        # Validate the collection exists before attempting to open it.
        try:
            exists = self._client.collection_exists(collection_name)
        except (UnexpectedResponse, Exception) as exc:
            logger.error(
                "QdrantRetriever: failed to check collection '%s' in '%s': %s",
                collection_name,
                persist_directory,
                exc,
            )
            raise

        if not exists:
            message = (
                f"QdrantRetriever: collection '{collection_name}' not found "
                f"in '{persist_directory}'."
            )
            logger.error(message)
            raise ValueError(message)

        # Validate the collection is non-empty.
        try:
            info = self._client.get_collection(collection_name)
        except (UnexpectedResponse, Exception) as exc:
            logger.error(
                "QdrantRetriever: cannot retrieve info for collection '%s': %s",
                collection_name,
                exc,
            )
            raise

        count: int = info.points_count or 0
        logger.info(
            "QdrantRetriever: collection '%s' has %d vectors.",
            collection_name,
            count,
        )
        if count == 0:
            raise ValueError(
                f"QdrantRetriever: collection '{collection_name}' is empty."
            )

        # Log stored vector configuration for transparency, mirroring the
        # embedding_model / embedding_dimension metadata check in ChromaRetriever.
        vectors_config = info.config.params.vectors
        stored_dim: int | None = getattr(vectors_config, "size", None)
        stored_distance: Any = getattr(vectors_config, "distance", None)
        logger.info(
            "QdrantRetriever: collection config — dimension=%s, distance=%s.",
            stored_dim,
            stored_distance,
        )

    def close(self) -> None:
        """
        Close the underlying Qdrant client.

        This is primarily useful for local Qdrant storage, where the client
        keeps a file lock on the database. Calling this method releases all
        resources cleanly.

        It is safe to call multiple times.
        """
        try:
            self._client.close()
        except Exception:
            logger.exception("QdrantRetriever: failed to close Qdrant client.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> dict[str, Any]:
        """
        Run a nearest-neighbour search against the collection.

        Parameters
        ----------
        query_embedding : list[float]
            Pre-computed embedding vector for the query.
        top_k : int
            Number of nearest neighbours to retrieve.

        Returns
        -------
        dict with keys:
            ``ids``        – list of matched chunk IDs (original string form)
            ``documents``  – list of matched document texts
            ``metadatas``  – list of metadata dicts (payload fields, text excluded)
            ``distances``  – list of distances (lower = more similar; computed as
                             ``1 − score`` to mirror the ChromaRetriever convention
                             where cosine distance is returned rather than score)

        Notes
        -----
        The ingestion pipeline (run_qdrant_ingestion.py) maps every plain
        ``chunk_id`` string (e.g. ``"law_001"``) to a UUID5 Qdrant point id,
        but preserves the original string in the payload under the key
        ``"chunk_id"``.  This method always returns the original string form
        so that downstream consumers (RRF, BM25, evaluation scripts) can
        match IDs across retrieval backends without modification.
        """
        # Coerce the query embedding to a plain list[float] so that numpy
        # arrays, torch tensors, or other array-like types are all handled
        # safely before being forwarded to the Qdrant client.
        if not isinstance(query_embedding, list):
            query_embedding = [float(v) for v in query_embedding]

        # client.search() was removed in qdrant-client ≥1.7; query_points() is
        # the current unified search API.  The query parameter accepts a plain
        # list[float] for nearest-neighbour search, mirroring the old interface.
        response = self._client.query_points(
            collection_name=self._collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
        )

        # Map each ScoredPoint back to the same envelope shape as ChromaRetriever.
        # query_points() returns a QueryResponse; the actual ScoredPoint list is
        # under response.points (ChromaDB instead wrapped each field in an outer
        # list per query vector).  Both are unwrapped here so callers always
        # receive plain lists of strings / dicts / floats.
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        distances: list[float] = []

        for point in response.points:
            payload: dict[str, Any] = point.payload or {}

            # Recover the original chunk_id string stored by the ingestion pipeline.
            ids.append(payload.get("chunk_id", ""))
            # Expose the chunk text separately, mirroring ChromaDB's "documents" field.
            documents.append(payload.get("text", ""))
            # All remaining payload fields become the metadata dict.
            metadatas.append({k: v for k, v in payload.items() if k != "text"})
            # Qdrant returns cosine *score* (higher = more similar).
            # Convert to cosine *distance* (lower = more similar) to match
            # the ChromaRetriever.search return contract.
            distances.append(1.0 - point.score)

        logger.debug(
            "QdrantRetriever.search returned %d results (top_k=%d).",
            len(ids),
            top_k,
        )

        return {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
            "distances": distances,
        }
