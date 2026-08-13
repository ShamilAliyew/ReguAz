"""Qdrant ingestion pipeline for the production BGE-M3 embedding model.

This script is responsible solely for ingesting pre-computed BGE-M3
embeddings into a local Qdrant collection. It joins embedding records
with their corresponding chunks on ``chunk_id``, builds Qdrant
payloads, and upserts the resulting points in configurable batches.

Every run drops and recreates the target collection before ingesting,
so reruns always produce a clean, fully-consistent collection —
stale points from chunks that were removed, renamed, or re-chunked
since the last run never persist.

It does NOT implement retrieval, BM25, hybrid search, reranking, or
evaluation logic — those concerns live elsewhere in the project.

Usage:
    python scripts/run_qdrant_ingestion.py
    python scripts/run_qdrant_ingestion.py --batch-size 256
    python scripts/run_qdrant_ingestion.py \\
        --embeddings-dir data/processed/embeddings \\
        --chunks-dir data/processed/chunks \\
        --qdrant-dir data/qdrant
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.reguaz.config import (
    CHUNKS_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTION_PREFIX,
    EMBEDDINGS_DIR,
    QDRANT_PATH,
)
from backend.reguaz.database.qdrant import QdrantDBManager
from backend.reguaz.services.chunks.chunk_reader import ChunkReader
from backend.reguaz.services.embeddings.embedding_reader import EmbeddingReader
from backend.reguaz.utils.logger import get_logger

logger = get_logger(__name__, "qdrant_ingestion.log")

# This pipeline is dedicated to the production embedding model. Adding
# support for additional models is out of scope at this stage.
EMBEDDING_MODEL = "bge_m3"

# Payload fields copied verbatim from each resolved chunk. The raw
# embedding vector is intentionally excluded from the payload.
PAYLOAD_FIELDS: tuple[str, ...] = (
    "chunk_id",
    "document_id",
    "title",
    "category",
    "chapter",
    "article",
    "section",
    "subsection",
    "page_start",
    "page_end",
    "source_file",
    "text",
)


@dataclass(frozen=True)
class IngestionSettings:
    """Resolved settings for a single ingestion run.

    Attributes:
        embeddings_dir: Root directory containing per-model embedding
            subdirectories (e.g. ``{EMBEDDINGS_DIR}``, which contains
            ``bge_m3/``).
        chunks_dir: Root directory containing chunk JSONL files.
        qdrant_dir: Directory where the local Qdrant database is
            persisted.
        batch_size: Number of points to upsert per Qdrant batch.
        collection_name: Name of the target Qdrant collection.
        model_name: Name of the embedding model being ingested.
    """

    embeddings_dir: Path
    chunks_dir: Path
    qdrant_dir: Path
    batch_size: int
    collection_name: str
    model_name: str = EMBEDDING_MODEL


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the ingestion script.

    Returns:
        Parsed CLI arguments. Any omitted argument falls back to the
        corresponding default from ``backend.reguaz.config``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Ingest BGE-M3 embeddings into the local Qdrant "
            f"'{DEFAULT_COLLECTION_PREFIX}_{EMBEDDING_MODEL}' collection."
        )
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default=None,
        help=(
            "Root embeddings directory containing per-model "
            f"subdirectories (default: {EMBEDDINGS_DIR})."
        ),
    )
    parser.add_argument(
        "--chunks-dir",
        type=str,
        default=None,
        help=f"Root chunks directory (default: {CHUNKS_DIR}).",
    )
    parser.add_argument(
        "--qdrant-dir",
        type=str,
        default=None,
        help=f"Local Qdrant storage directory (default: {QDRANT_PATH}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"Upsert batch size (default: {DEFAULT_BATCH_SIZE}).",
    )
    return parser.parse_args()


def build_settings(args: argparse.Namespace) -> IngestionSettings:
    """Builds resolved ingestion settings from CLI args and config defaults.

    CLI arguments take precedence over ``backend.reguaz.config``
    defaults. ``embeddings_dir`` is kept as the embeddings root (not
    joined with the model name) since ``EmbeddingReader`` resolves the
    per-model subdirectory itself given ``model_name``.

    Args:
        args: Parsed CLI arguments.

    Returns:
        A fully resolved ``IngestionSettings`` instance.
    """
    embeddings_dir = Path(args.embeddings_dir or EMBEDDINGS_DIR)
    chunks_dir = Path(args.chunks_dir or CHUNKS_DIR)
    qdrant_dir = Path(args.qdrant_dir or QDRANT_PATH)
    batch_size = args.batch_size or DEFAULT_BATCH_SIZE

    collection_name = f"{DEFAULT_COLLECTION_PREFIX}_{EMBEDDING_MODEL}"

    return IngestionSettings(
        embeddings_dir=embeddings_dir,
        chunks_dir=chunks_dir,
        qdrant_dir=qdrant_dir,
        batch_size=batch_size,
        collection_name=collection_name,
    )


class QdrantIngestionPipeline:
    """Ingests BGE-M3 embeddings into a local Qdrant collection.

    The pipeline reads pre-computed embedding records and their
    corresponding chunks, joins them on ``chunk_id``, builds Qdrant
    payloads, and upserts the resulting vectors in batches via
    ``QdrantDBManager``. It performs no chunking, no embedding
    generation, and no retrieval.
    """

    def __init__(self, settings: IngestionSettings) -> None:
        """Initializes the pipeline with resolved settings and readers.

        Args:
            settings: Fully resolved ingestion settings.
        """
        self._settings = settings
        self._embedding_reader = EmbeddingReader()
        self._chunk_reader = ChunkReader()

    def run(self) -> None:
        """Executes the full ingestion pipeline end to end.

        Raises:
            ValueError: If a chunk_id cannot be resolved, a duplicate
                chunk_id is encountered, the embedding dimension
                changes mid-stream, or collected batches are
                inconsistent.
            Exception: If any underlying reader or database operation
                fails.
        """
        start_time = time.monotonic()
        logger.info(
            "Starting Qdrant ingestion pipeline for model '%s'.",
            self._settings.model_name,
        )
        self._log_configuration()

        chunk_lookup = self._build_chunk_lookup()
        embedding_files = self._discover_embedding_files()
        ids, vectors, payloads = self._collect_points(embedding_files, chunk_lookup)

        if not ids:
            logger.warning(
                "No embedding records found under '%s'; nothing to ingest.",
                self._settings.embeddings_dir,
            )
            return

        self._ingest_points(ids, vectors, payloads)

        elapsed_seconds = time.monotonic() - start_time
        logger.info(
            "Ingestion completed: %d vector(s) inserted into collection "
            "'%s' in %.2fs.",
            len(ids),
            self._settings.collection_name,
            elapsed_seconds,
        )

    def _log_configuration(self) -> None:
        """Logs the resolved configuration for this ingestion run."""
        settings = self._settings
        logger.info(
            "Configuration: embeddings_dir='%s' chunks_dir='%s' "
            "qdrant_dir='%s' batch_size=%d collection='%s'.",
            settings.embeddings_dir,
            settings.chunks_dir,
            settings.qdrant_dir,
            settings.batch_size,
            settings.collection_name,
        )

    def _build_chunk_lookup(self) -> dict[str, dict[str, Any]]:
        """Builds the chunk_id -> chunk lookup table via ChunkReader.

        Returns:
            A dictionary mapping each chunk_id to its full chunk
            object.
        """
        logger.info("Building chunk lookup from '%s'.", self._settings.chunks_dir)
        lookup = self._chunk_reader.build_lookup(self._settings.chunks_dir)
        logger.info("Chunk lookup built with %d chunk(s).", len(lookup))
        return lookup

    def _discover_embedding_files(self) -> list[Path]:
        """Discovers BGE-M3 embedding files via EmbeddingReader.

        Returns:
            A list of paths to discovered embedding JSONL files.
        """
        logger.info(
            "Discovering embedding files under '%s/%s'.",
            self._settings.embeddings_dir,
            self._settings.model_name,
        )
        files = self._embedding_reader.discover_files(
            self._settings.embeddings_dir, self._settings.model_name
        )
        logger.info("Discovered %d embedding file(s).", len(files))
        return files

    def _collect_points(
        self,
        embedding_files: list[Path],
        chunk_lookup: dict[str, dict[str, Any]],
    ) -> tuple[list[str], list[list[float]], list[dict[str, Any]]]:
        """Joins embeddings with chunks and builds points to upsert.

        Embedding records whose ``chunk_id`` has already been processed
        are skipped with a logged warning (first occurrence wins).  This
        mirrors the tolerance strategy used by
        ``ChunkReader.build_lookup`` and ensures that duplicate
        ``chunk_id`` values originating from identically-titled
        documents in different category directories do not crash the
        pipeline.

        Args:
            embedding_files: Discovered embedding JSONL file paths.
            chunk_lookup: chunk_id -> chunk lookup table.

        Returns:
            A tuple of ``(ids, vectors, payloads)`` lists, aligned by
            index and ready for batched upsert.

        Raises:
            ValueError: If a chunk_id cannot be resolved or the
                embedding dimension changes mid-stream.
        """
        ids: list[str] = []
        vectors: list[list[float]] = []
        payloads: list[dict[str, Any]] = []
        seen_chunk_ids: set[str] = set()
        expected_dim: int | None = None
        duplicate_count = 0

        for file_path in embedding_files:
            for record in self._embedding_reader.read_file(file_path):
                chunk_id = self._extract_chunk_id(record, file_path)

                if self._is_duplicate_embedding(chunk_id, seen_chunk_ids, file_path):
                    duplicate_count += 1
                    continue

                vector = self._extract_vector(record, file_path)
                expected_dim = self._validate_dimension(
                    vector, expected_dim, chunk_id, file_path
                )
                chunk = self._resolve_chunk(chunk_id, chunk_lookup, file_path)

                ids.append(chunk_id)
                vectors.append(vector)
                payloads.append(self._build_payload(chunk))
                seen_chunk_ids.add(chunk_id)

        if duplicate_count:
            logger.warning(
                "Skipped %d duplicate embedding record(s) across %d file(s). "
                "Consider deduplicating the source data.",
                duplicate_count,
                len(embedding_files),
            )

        self._assert_consistent_lengths(ids, vectors, payloads)
        return ids, vectors, payloads

    def _ingest_points(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        """Recreates the target collection and upserts points in batches.

        The collection is always dropped and recreated before ingestion
        so that every run produces a clean, fully-consistent state —
        stale points from previously removed or renamed chunks never
        linger in the collection.

        Args:
            ids: Point (chunk) ids.
            vectors: Embedding vectors, aligned with ``ids``.
            payloads: Payload dictionaries, aligned with ``ids``.
        """
        embedding_dim = len(vectors[0])
        qdrant = QdrantDBManager(
            embedding_dim=embedding_dim, db_path=self._settings.qdrant_dir
        )

        try:
            qdrant.create_collection(
                self._settings.collection_name, force_recreate=True
            )
            qdrant.upsert_points(
                collection_name=self._settings.collection_name,
                ids=ids,
                vectors=vectors,
                payloads=payloads,
                batch_size=self._settings.batch_size,
            )
        finally:
            qdrant.close()

    @staticmethod
    def _extract_chunk_id(record: dict[str, Any], file_path: Path) -> str:
        """Extracts and validates the chunk_id field from an embedding record.

        Args:
            record: Embedding record parsed from a JSONL line.
            file_path: Path of the file the record was read from, used
                for error context.

        Returns:
            The record's ``chunk_id`` value.

        Raises:
            ValueError: If the record has no ``chunk_id`` field.
        """
        chunk_id = record.get("chunk_id")
        if not chunk_id:
            message = f"Embedding record missing 'chunk_id' in file '{file_path}'."
            logger.error(message)
            raise ValueError(message)
        return chunk_id

    @staticmethod
    def _extract_vector(record: dict[str, Any], file_path: Path) -> list[float]:
        """Extracts and validates the embedding vector from a record.

        Args:
            record: Embedding record parsed from a JSONL line.
            file_path: Path of the file the record was read from, used
                for error context.

        Returns:
            The record's ``embedding`` vector.

        Raises:
            ValueError: If the record has no ``embedding`` field.
        """
        vector = record.get("embedding")
        if not vector:
            message = f"Embedding record missing 'embedding' in file '{file_path}'."
            logger.error(message)
            raise ValueError(message)
        return vector

    @staticmethod
    def _is_duplicate_embedding(
        chunk_id: str, seen_chunk_ids: set[str], file_path: Path
    ) -> bool:
        """Checks whether a chunk_id has already been seen among embeddings.

        If the chunk_id is a duplicate, a warning is logged and the
        caller should skip the record (first occurrence wins).

        Args:
            chunk_id: Chunk id extracted from the current record.
            seen_chunk_ids: Set of chunk_ids already processed.
            file_path: Path of the file the record was read from, used
                for warning context.

        Returns:
            True if ``chunk_id`` was already processed (duplicate),
            False otherwise.
        """
        if chunk_id in seen_chunk_ids:
            logger.warning(
                "Duplicate chunk_id '%s' encountered in embedding "
                "file '%s'; keeping the first occurrence and skipping "
                "this one.",
                chunk_id,
                file_path,
            )
            return True
        return False

    @staticmethod
    def _validate_dimension(
        vector: list[float],
        expected_dim: int | None,
        chunk_id: str,
        file_path: Path,
    ) -> int:
        """Validates that the embedding dimension is stable across records.

        Args:
            vector: Embedding vector for the current record.
            expected_dim: Dimension established by the first record
                encountered, or None if not yet set.
            chunk_id: Chunk id of the current record, used for error
                context.
            file_path: Path of the file the record was read from, used
                for error context.

        Returns:
            The dimension to treat as expected for subsequent records.

        Raises:
            ValueError: If the current vector's dimension differs from
                ``expected_dim``.
        """
        dimension = len(vector)

        if expected_dim is None:
            return dimension

        if dimension != expected_dim:
            message = (
                f"Embedding dimension changed for chunk_id '{chunk_id}' in "
                f"file '{file_path}': expected {expected_dim}, got "
                f"{dimension}."
            )
            logger.error(message)
            raise ValueError(message)

        return expected_dim

    @staticmethod
    def _resolve_chunk(
        chunk_id: str,
        chunk_lookup: dict[str, dict[str, Any]],
        file_path: Path,
    ) -> dict[str, Any]:
        """Resolves the full chunk object for a given chunk_id.

        Args:
            chunk_id: Chunk id to resolve.
            chunk_lookup: chunk_id -> chunk lookup table.
            file_path: Path of the embedding file the chunk_id came
                from, used for error context.

        Returns:
            The full chunk dictionary matching ``chunk_id``.

        Raises:
            ValueError: If ``chunk_id`` is not present in
                ``chunk_lookup``.
        """
        chunk = chunk_lookup.get(chunk_id)
        if chunk is None:
            message = (
                f"chunk_id '{chunk_id}' from embedding file '{file_path}' "
                "was not found in the chunk lookup."
            )
            logger.error(message)
            raise ValueError(message)
        return chunk

    @staticmethod
    def _build_payload(chunk: dict[str, Any]) -> dict[str, Any]:
        """Builds a Qdrant payload dictionary from a chunk.

        Only the fields listed in ``PAYLOAD_FIELDS`` are copied. The
        embedding vector is never included in the payload.

        Args:
            chunk: Full chunk dictionary resolved from the chunk
                lookup.

        Returns:
            A payload dictionary containing exactly the configured
            chunk fields.
        """
        return {field: chunk.get(field) for field in PAYLOAD_FIELDS}

    @staticmethod
    def _assert_consistent_lengths(
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        """Ensures collected ids, vectors, and payloads stay aligned.

        Args:
            ids: Collected point ids.
            vectors: Collected embedding vectors.
            payloads: Collected payload dictionaries.

        Raises:
            ValueError: If the three lists do not have equal lengths.
        """
        if not (len(ids) == len(vectors) == len(payloads)):
            message = (
                "Inconsistent batch sizes collected for ingestion: "
                f"{len(ids)} ids, {len(vectors)} vectors, "
                f"{len(payloads)} payloads."
            )
            logger.error(message)
            raise ValueError(message)


def main() -> None:
    """Entry point: parses arguments and runs the ingestion pipeline.

    Raises:
        Exception: Re-raises any exception encountered during the
            pipeline run, after logging it.
    """
    args = parse_args()
    settings = build_settings(args)
    pipeline = QdrantIngestionPipeline(settings)

    try:
        pipeline.run()
    except Exception:
        logger.exception("Qdrant ingestion pipeline failed.")
        raise


if __name__ == "__main__":
    main()