"""
backend/reguaz/retrieval/retriever.py

Lightweight wrapper around a ChromaDB PersistentClient collection.
Performs pure vector-search; embedding generation is the caller's responsibility.
"""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.errors import NotFoundError

logger = logging.getLogger(__name__)


class ChromaRetriever:
    """
    Opens an existing ChromaDB collection and exposes a vector-search method.

    Parameters
    ----------
    persist_directory : str
        Path to the ChromaDB persistence directory (e.g. ``"data/chroma"``).
    collection_name : str
        Name of the collection to open (e.g. ``"reguaz_e5"``).
    """

    def __init__(self, persist_directory: str, collection_name: str) -> None:
        logger.info(
            "ChromaRetriever: opening collection '%s' from '%s'",
            collection_name,
            persist_directory,
        )
        self._client = chromadb.PersistentClient(path=persist_directory)

        # Validate the collection exists before attempting to open it.
        try:
            self._collection = self._client.get_collection(name=collection_name)
        except (NotFoundError, Exception) as exc:
            logger.error(
                "ChromaRetriever: collection '%s' not found in '%s': %s",
                collection_name,
                persist_directory,
                exc,
            )
            raise

        # Validate the collection is non-empty.
        count = self._collection.count()
        logger.info(
            "ChromaRetriever: collection '%s' has %d vectors.",
            collection_name,
            count,
        )
        if count == 0:
            raise ValueError(
                f"ChromaRetriever: collection '{collection_name}' is empty."
            )

        # Validate stored metadata is consistent with the requested collection.
        # The embedding_model field is written by the ingestion pipeline.
        coll_meta: dict[str, Any] = self._collection.metadata or {}
        stored_model: str | None = coll_meta.get("embedding_model")
        if stored_model is not None:
            # Normalise both sides the same way the ingestion script does.
            norm_stored = stored_model.lower().replace("-", "_")
            norm_name = collection_name.lower()
            if norm_stored not in norm_name:
                logger.error(
                    "ChromaRetriever: metadata mismatch for collection '%s'. "
                    "Stored embedding_model='%s' does not match collection name. "
                    "Stopping evaluation to avoid using the wrong embeddings.",
                    collection_name,
                    stored_model,
                )
                raise ValueError(
                    f"Metadata mismatch: collection '{collection_name}' stores "
                    f"embedding_model='{stored_model}'."
                )

        stored_dim: int | None = coll_meta.get("embedding_dimension")
        logger.info(
            "ChromaRetriever: collection metadata — embedding_model=%s, dimension=%s.",
            stored_model,
            stored_dim,
        )

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
            ``ids``        – list of matched document IDs
            ``documents``  – list of matched document texts
            ``metadatas``  – list of metadata dicts (schema as stored in Chroma)
            ``distances``  – list of distances (lower = more similar for cosine)
        """
        # Coerce the query embedding to a plain list[float] so that numpy
        # arrays, torch tensors, or other array-like types are all handled
        # safely before being forwarded to the ChromaDB client.
        if not isinstance(query_embedding, list):
            query_embedding = [float(v) for v in query_embedding]

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # ChromaDB wraps each field in an outer list (one per query vector).
        # Guard defensively against None values in the response envelope.
        ids: list[str] = (results.get("ids") or [[]])[0]
        documents: list[str] = (results.get("documents") or [[]])[0]
        metadatas: list[dict[str, Any]] = (results.get("metadatas") or [[]])[0]
        distances: list[float] = (results.get("distances") or [[]])[0]

        logger.debug(
            "ChromaRetriever.search returned %d results (top_k=%d).",
            len(ids),
            top_k,
        )

        return {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
            "distances": distances,
        }
