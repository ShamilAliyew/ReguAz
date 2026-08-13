"""Local Qdrant database manager.

This module encapsulates all low-level interactions with a locally
persisted Qdrant vector database instance. It is intentionally scoped to
pure database operations only (client lifecycle, collection management,
and batched point upserts). It does not implement retrieval, hybrid
search, reranking, embedding generation, or any ingestion pipeline
logic — those concerns belong to other layers of the application.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams

from backend.reguaz import config as reguaz_config
from backend.reguaz.utils.logger import get_logger

logger = get_logger(__name__, "qdrant_ingestion.log")

DEFAULT_DB_PATH = reguaz_config.QDRANT_PATH

# Qdrant only accepts unsigned 64-bit integers or UUID strings as point
# ids. Caller-supplied ids (e.g. "law_001") are deterministically mapped
# to a UUID5 derived from this fixed namespace, so the same id always
# maps to the same Qdrant point across repeated ingestion runs. The
# original id is expected to be preserved by the caller in the payload
# (e.g. as "chunk_id") if it needs to be recovered later.
_POINT_ID_NAMESPACE = uuid.UUID("d95a4f3e-4a3b-4f8e-9c2a-8e2f6b9a1d70")


class QdrantDBManager:
    """Manages a local, file-persisted Qdrant database instance.

    This class is responsible solely for database-level operations:
    client initialization, collection lifecycle management, and batched
    point upserts. It has no knowledge of embedding models, retrieval
    strategies, or payload schemas — payloads are treated as opaque
    dictionaries supplied by the caller.

    Attributes:
        embedding_dim: Dimensionality of vectors stored in collections
            managed by this instance.
        db_path: Filesystem path where Qdrant persists its local data.
    """

    def __init__(
        self,
        embedding_dim: int,
        db_path: str | Path = DEFAULT_DB_PATH,
    ) -> None:
        """Initializes a local Qdrant client with persistent storage.

        Args:
            embedding_dim: Dimensionality of the vectors that will be
                stored. Must be a positive integer.
            db_path: Directory in which Qdrant will persist its local
                database files. Defaults to ``data/qdrant``.

        Raises:
            ValueError: If ``embedding_dim`` is not a positive integer.
            Exception: If the underlying Qdrant client fails to
                initialize.
        """
        if embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be a positive integer, got {embedding_dim}."
            )

        self.embedding_dim = embedding_dim
        self.db_path = Path(db_path)

        try:
            self.db_path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(self.db_path))
            logger.info(
                "Initialized local Qdrant client at '%s' (embedding_dim=%d).",
                self.db_path,
                self.embedding_dim,
            )
        except Exception:
            logger.exception(
                "Failed to initialize Qdrant client at '%s'.", self.db_path
            )
            raise

    @property
    def client(self) -> QdrantClient:
        """Returns the underlying Qdrant client instance.

        Returns:
            The raw ``QdrantClient`` object, for callers that need
            direct access beyond what this manager exposes.
        """
        return self._client

    def collection_exists(self, collection_name: str) -> bool:
        """Checks whether a collection exists.

        Args:
            collection_name: Name of the collection to check.

        Returns:
            True if the collection exists, False otherwise.

        Raises:
            Exception: If the existence check fails for a reason other
                than the collection not existing.
        """
        try:
            return self._client.collection_exists(collection_name)
        except Exception:
            logger.exception(
                "Failed to check existence of collection '%s'.", collection_name
            )
            raise

    def create_collection(
        self,
        collection_name: str,
        force_recreate: bool = False,
    ) -> None:
        """Creates a collection if it does not already exist.

        The collection is configured to use cosine similarity with the
        embedding dimension provided at construction time.

        Args:
            collection_name: Name of the collection to create.
            force_recreate: If True, an existing collection with the
                same name is deleted and recreated. If False (default),
                an existing collection is left untouched.

        Raises:
            Exception: If collection creation fails.
        """
        try:
            exists = self.collection_exists(collection_name)

            if exists and force_recreate:
                logger.info(
                    "Collection '%s' already exists; force_recreate is set, "
                    "deleting before recreation.",
                    collection_name,
                )
                self.delete_collection(collection_name)
                exists = False

            if exists:
                logger.info(
                    "Collection '%s' already exists; skipping creation.",
                    collection_name,
                )
                return

            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                "Created collection '%s' (dim=%d, distance=COSINE).",
                collection_name,
                self.embedding_dim,
            )
        except Exception:
            logger.exception("Failed to create collection '%s'.", collection_name)
            raise

    def delete_collection(self, collection_name: str) -> None:
        """Deletes a collection.

        Args:
            collection_name: Name of the collection to delete.

        Raises:
            Exception: If collection deletion fails.
        """
        try:
            self._client.delete_collection(collection_name=collection_name)
            logger.info("Deleted collection '%s'.", collection_name)
        except Exception:
            logger.exception("Failed to delete collection '%s'.", collection_name)
            raise

    def get_collection_info(self, collection_name: str) -> Any:
        """Retrieves metadata and statistics for a collection.

        Args:
            collection_name: Name of the collection to inspect.

        Returns:
            The collection information object returned by the Qdrant
            client (e.g. vector configuration, point count, status).

        Raises:
            Exception: If the collection does not exist or the request
                fails.
        """
        try:
            return self._client.get_collection(collection_name=collection_name)
        except (UnexpectedResponse, Exception):
            logger.exception(
                "Failed to retrieve info for collection '%s'.", collection_name
            )
            raise

    def upsert_points(
        self,
        collection_name: str,
        ids: list[str | int],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
        batch_size: int = 128,
    ) -> None:
        """Upserts points into a collection in batches.

        Args:
            collection_name: Name of the target collection.
            ids: List of point (chunk) ids.
            vectors: List of embedding vectors, one per id.
            payloads: List of arbitrary metadata dictionaries, one per
                id. No schema is assumed or enforced.
            batch_size: Maximum number of points to send per upsert
                request. Defaults to 128.

        Raises:
            ValueError: If ``ids``, ``vectors``, and ``payloads`` do not
                all have the same length, or if ``batch_size`` is not
                positive.
            Exception: If the upsert operation fails.
        """
        self._validate_batch_inputs(ids, vectors, payloads, batch_size)

        total = len(ids)
        logger.info(
            "Upserting %d points into collection '%s' (batch_size=%d).",
            total,
            collection_name,
            batch_size,
        )

        try:
            for start in range(0, total, batch_size):
                end = start + batch_size
                points = [
                    PointStruct(
                        id=self._to_point_id(point_id),
                        vector=vector,
                        payload=payload,
                    )
                    for point_id, vector, payload in zip(
                        ids[start:end], vectors[start:end], payloads[start:end]
                    )
                ]
                self._client.upsert(collection_name=collection_name, points=points)
                logger.info(
                    "Upserted batch [%d:%d] (%d points) into collection '%s'.",
                    start,
                    min(end, total),
                    len(points),
                    collection_name,
                )

            logger.info(
                "Successfully upserted %d points into collection '%s'.",
                total,
                collection_name,
            )
        except Exception:
            logger.exception(
                "Failed to upsert points into collection '%s'.", collection_name
            )
            raise

    @staticmethod
    def _to_point_id(point_id: str | int) -> str | int:
        """Converts a caller-supplied id into a Qdrant-compatible point id.

        Qdrant only accepts unsigned 64-bit integers or UUID strings as
        point ids. Integers and already-valid UUID strings are passed
        through unchanged. Any other string (e.g. a human-readable
        ``chunk_id`` such as ``"law_001"``) is deterministically mapped
        to a UUID5 derived from a fixed namespace, so the same input
        always produces the same Qdrant point id across runs.

        Args:
            point_id: The caller-supplied point id.

        Returns:
            An integer or UUID string accepted by Qdrant as a point id.
        """
        if isinstance(point_id, int):
            return point_id

        try:
            return str(uuid.UUID(str(point_id)))
        except ValueError:
            return str(uuid.uuid5(_POINT_ID_NAMESPACE, str(point_id)))

    def close(self) -> None:
        """Closes the underlying Qdrant client connection, if applicable.

        Safe to call multiple times. Errors during close are logged but
        not re-raised, since callers typically invoke this during
        teardown/cleanup.
        """
        try:
            close_fn = getattr(self._client, "close", None)
            if callable(close_fn):
                close_fn()
                logger.info("Closed Qdrant client at '%s'.", self.db_path)
        except Exception:
            logger.exception("Error while closing Qdrant client at '%s'.", self.db_path)

    def __enter__(self) -> "QdrantDBManager":
        """Enables use as a context manager.

        Returns:
            This ``QdrantDBManager`` instance.
        """
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Ensures the client is closed on context manager exit."""
        self.close()

    @staticmethod
    def _validate_batch_inputs(
        ids: list[str | int],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
        batch_size: int,
    ) -> None:
        """Validates that batch upsert inputs are consistent.

        Args:
            ids: List of point ids.
            vectors: List of embedding vectors.
            payloads: List of payload dictionaries.
            batch_size: Requested batch size.

        Raises:
            ValueError: If the lengths of ``ids``, ``vectors``, and
                ``payloads`` differ, or if ``batch_size`` is not
                positive.
        """
        if not (len(ids) == len(vectors) == len(payloads)):
            message = (
                "ids, vectors, and payloads must have equal lengths, got "
                f"{len(ids)}, {len(vectors)}, {len(payloads)} respectively."
            )
            logger.error(message)
            raise ValueError(message)

        if batch_size <= 0:
            message = f"batch_size must be a positive integer, got {batch_size}."
            logger.error(message)
            raise ValueError(message)