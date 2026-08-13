from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import QdrantClient, models

from backend.reguaz.services.embeddings.v2_models import (
    EmbeddingManifest,
    EmbeddingRecord,
)
from backend.reguaz.services.ingestion.point_ids import v2_point_id
from backend.reguaz.services.embeddings.v2_pipeline import (
    CATEGORY_ALLOWLIST,
    V2CorpusReader,
    read_embedding_records,
)


PAYLOAD_INDEXES: tuple[tuple[str, models.PayloadSchemaType], ...] = (
    ("category", models.PayloadSchemaType.KEYWORD),
    ("document_id", models.PayloadSchemaType.KEYWORD),
    ("document_version_id", models.PayloadSchemaType.KEYWORD),
    ("logical_chunk_id", models.PayloadSchemaType.KEYWORD),
    ("chunk_type", models.PayloadSchemaType.KEYWORD),
    ("node_type", models.PayloadSchemaType.KEYWORD),
    ("status", models.PayloadSchemaType.KEYWORD),
    ("is_current", models.PayloadSchemaType.BOOL),
    ("hierarchy.article.number", models.PayloadSchemaType.KEYWORD),
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QdrantIngestionReport(StrictModel):
    schema_version: str = "2.0"
    status: str
    corpus_version: str
    embedding_model_id: str
    embedding_model_revision: str
    dense_dimension: int = Field(gt=0)
    vector_modes: list[str]
    source_embedding_manifest: str
    source_embedding_manifest_sha256: str
    qdrant_path: str
    qdrant_mode: str
    physical_collection: str
    alias: str
    point_count: int = Field(ge=0)
    expected_point_count: int = Field(ge=0)
    payload_indexes: list[str]
    payload_indexes_effective: bool
    dense_smoke_result_count: int = Field(ge=0)
    sparse_smoke_result_count: int = Field(ge=0)
    completed_at: datetime


@dataclass(frozen=True, slots=True)
class QdrantV2Settings:
    v2_root: Path = Path("data/processed/v2")
    embeddings_root: Path = Path("data/processed/v2/embeddings")
    qdrant_path: Path = Path("data/processed/v2/qdrant")
    collection_prefix: str = "reguaz_v2_bge_m3"
    alias: str = "reguaz_v2_current"
    upload_batch_size: int = 128
    parallel: int = 1
    force_collection: bool = False

    def __post_init__(self) -> None:
        if self.upload_batch_size <= 0 or self.parallel <= 0:
            raise ValueError("upload_batch_size and parallel must be positive")


class EmbeddingArtifactReader:
    def __init__(self, directory: Path) -> None:
        self.directory = directory.resolve()
        self.manifest_path = self.directory / "manifest.json"
        self.manifest = EmbeddingManifest.model_validate_json(
            self.manifest_path.read_text(encoding="utf-8")
        )

    def validate_and_iter(self) -> Iterator[EmbeddingRecord]:
        count = 0
        seen: set[str] = set()
        for relative, expected_hash in sorted(self.manifest.shard_sha256.items()):
            path = self.directory / relative
            if _sha256_file(path) != expected_hash:
                raise ValueError(f"embedding shard hash mismatch: {path}")
            for record in read_embedding_records(path):
                if record.chunk_id in seen:
                    raise ValueError(f"duplicate embedding record: {record.chunk_id}")
                seen.add(record.chunk_id)
                self._validate_record(record)
                count += 1
                yield record
        if count != self.manifest.embedded_chunk_count:
            raise ValueError(
                f"embedding count mismatch: expected {self.manifest.embedded_chunk_count}, got {count}"
            )

    def validate(self) -> None:
        for _ in self.validate_and_iter():
            pass

    def _validate_record(self, record: EmbeddingRecord) -> None:
        if record.model_id != self.manifest.model_id:
            raise ValueError("embedding model ID mismatch")
        if record.model_revision != self.manifest.model_revision:
            raise ValueError("embedding model revision mismatch")
        if len(record.dense) != self.manifest.dense_dimension:
            raise ValueError("dense vector dimension mismatch")
        if any(
            not math.isfinite(value) for value in record.dense + record.sparse.values
        ):
            raise ValueError("embedding contains NaN or infinity")


class QdrantV2IngestionPipeline:
    def __init__(self, settings: QdrantV2Settings) -> None:
        self.settings = settings
        self.corpus = V2CorpusReader(settings.v2_root)
        embedding_directory = (
            settings.embeddings_root.resolve()
            / "bge_m3"
            / self.corpus.manifest.corpus_version
        )
        self.artifacts = EmbeddingArtifactReader(embedding_directory)
        if (
            self.artifacts.manifest.corpus_version
            != self.corpus.manifest.corpus_version
        ):
            raise ValueError("chunk and embedding corpus versions differ")
        self.physical_collection = self._collection_name()

    def dry_run(self) -> dict[str, object]:
        chunks = self.corpus.read_all()
        self.artifacts.validate()
        return {
            "corpus_version": self.corpus.manifest.corpus_version,
            "expected_points": len(chunks),
            "embedding_records": self.artifacts.manifest.embedded_chunk_count,
            "dense_dimension": self.artifacts.manifest.dense_dimension,
            "vector_modes": self.artifacts.manifest.vector_modes,
            "physical_collection": self.physical_collection,
            "alias": self.settings.alias,
            "qdrant_path": str(self.settings.qdrant_path.resolve()),
            "payload_indexes": [field for field, _ in PAYLOAD_INDEXES],
        }

    def run(self) -> QdrantIngestionReport:
        chunk_inputs = self.corpus.read_all()
        chunks = {item.chunk.chunk_id: item for item in chunk_inputs}
        expected = self.artifacts.manifest.embedded_chunk_count
        if expected != len(chunks):
            raise ValueError(
                f"embedding/chunk count mismatch: {expected} embeddings, {len(chunks)} chunks"
            )
        qdrant_path = self.settings.qdrant_path.resolve()
        self._validate_qdrant_target(qdrant_path)
        qdrant_path.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(qdrant_path))
        try:
            exists = client.collection_exists(self.physical_collection)
            if exists and self.settings.force_collection:
                client.delete_collection(self.physical_collection)
                exists = False
            if not exists:
                self._create_collection(client)
                self._create_payload_indexes(client)
                client.upload_points(
                    collection_name=self.physical_collection,
                    points=self._points(chunks),
                    batch_size=self.settings.upload_batch_size,
                    parallel=self.settings.parallel,
                    max_retries=3,
                    wait=True,
                )
            point_count, dense_hits, sparse_hits = self._validate_collection(
                client, chunks, expected
            )
            self._switch_alias(client)
            report = QdrantIngestionReport(
                status="success",
                corpus_version=self.corpus.manifest.corpus_version,
                embedding_model_id=self.artifacts.manifest.model_id,
                embedding_model_revision=self.artifacts.manifest.model_revision,
                dense_dimension=self.artifacts.manifest.dense_dimension,
                vector_modes=self.artifacts.manifest.vector_modes,
                source_embedding_manifest=str(self.artifacts.manifest_path),
                source_embedding_manifest_sha256=_sha256_file(
                    self.artifacts.manifest_path
                ),
                qdrant_path=str(qdrant_path),
                qdrant_mode="local_embedded",
                physical_collection=self.physical_collection,
                alias=self.settings.alias,
                point_count=point_count,
                expected_point_count=expected,
                payload_indexes=[field for field, _ in PAYLOAD_INDEXES],
                payload_indexes_effective=False,
                dense_smoke_result_count=dense_hits,
                sparse_smoke_result_count=sparse_hits,
                completed_at=datetime.now(timezone.utc),
            )
            _write_report(self.settings.v2_root / "qdrant_ingestion.json", report)
            return report
        except Exception:
            # Never publish an alias to a collection that failed validation.
            raise
        finally:
            client.close()

    def _points(self, chunks: dict[str, object]) -> Iterator[models.PointStruct]:
        for record in self.artifacts.validate_and_iter():
            item = chunks.get(record.chunk_id)
            if item is None:
                raise ValueError(
                    f"embedding has no matching V2 chunk: {record.chunk_id}"
                )
            chunk = item.chunk
            if record.embedding_input_sha256 != _sha256_text(chunk.embedding_text):
                raise ValueError(f"stale embedding for chunk: {chunk.chunk_id}")
            if item.category not in CATEGORY_ALLOWLIST:
                raise ValueError(f"unsupported category in payload: {item.category}")
            payload = {
                "schema_version": "2.0",
                "corpus_version": self.corpus.manifest.corpus_version,
                "embedding_model_id": record.model_id,
                "embedding_model_revision": record.model_revision,
                "embedding_input_sha256": record.embedding_input_sha256,
                "chunk_id": chunk.chunk_id,
                "logical_chunk_id": chunk.logical_chunk_id,
                "document_id": chunk.document_id,
                "document_version_id": chunk.document_version_id,
                "parent_chunk_id": chunk.parent_chunk_id,
                "article_root_id": chunk.article_root_id,
                "category": item.category,
                "document_title": chunk.document_title,
                "short_title": chunk.short_title,
                "source_file": chunk.source_file,
                "source_url": chunk.source_url,
                "chunk_type": chunk.chunk_type,
                "node_type": chunk.node_type,
                "language": chunk.language,
                "hierarchy": chunk.hierarchy.model_dump(mode="json"),
                "canonical_locator": chunk.canonical_locator,
                "atomic_unit_ids": chunk.atomic_unit_ids,
                "content": chunk.content,
                "content_sha256": chunk.content_sha256,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "source_spans": [
                    span.model_dump(mode="json") for span in chunk.source_spans
                ],
                "relations": chunk.relations.model_dump(mode="json"),
                "split_index": chunk.split_index,
                "split_count": chunk.split_count,
                "is_continuation": chunk.is_continuation,
                "continuation_of": chunk.continuation_of,
                "effective_from": chunk.effective_from,
                "effective_to": chunk.effective_to,
                "status": chunk.status,
                "is_current": chunk.is_current,
                "quality": chunk.quality.model_dump(mode="json"),
                "table": chunk.table.model_dump(mode="json") if chunk.table else None,
            }
            yield models.PointStruct(
                id=v2_point_id(chunk.chunk_id),
                vector={
                    "dense": record.dense,
                    "sparse": models.SparseVector(
                        indices=record.sparse.indices, values=record.sparse.values
                    ),
                },
                payload=payload,
            )

    def _create_collection(self, client: QdrantClient) -> None:
        client.create_collection(
            collection_name=self.physical_collection,
            vectors_config={
                "dense": models.VectorParams(
                    size=self.artifacts.manifest.dense_dimension,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
            metadata={
                "schema_version": "2.0",
                "corpus_version": self.corpus.manifest.corpus_version,
                "embedding_model_id": self.artifacts.manifest.model_id,
                "embedding_model_revision": self.artifacts.manifest.model_revision,
                "source_embedding_manifest_sha256": _sha256_file(
                    self.artifacts.manifest_path
                ),
                "vector_modes": ["dense", "sparse"],
            },
        )

    def _create_payload_indexes(self, client: QdrantClient) -> None:
        for field, schema in PAYLOAD_INDEXES:
            client.create_payload_index(
                collection_name=self.physical_collection,
                field_name=field,
                field_schema=schema,
                wait=True,
            )

    def _validate_collection(
        self, client: QdrantClient, chunks: dict[str, object], expected: int
    ) -> tuple[int, int, int]:
        info = client.get_collection(self.physical_collection)
        point_count = client.count(
            collection_name=self.physical_collection, exact=True
        ).count
        if point_count != expected:
            raise ValueError(
                f"Qdrant point count mismatch: expected {expected}, got {point_count}"
            )
        metadata = info.config.metadata or {}
        if metadata.get("corpus_version") != self.corpus.manifest.corpus_version:
            raise ValueError("Qdrant collection corpus metadata mismatch")
        first_record = next(self.artifacts.validate_and_iter(), None)
        sparse_record = next(
            (
                record
                for record in self.artifacts.validate_and_iter()
                if record.sparse.indices
            ),
            None,
        )
        if first_record is None or sparse_record is None:
            raise ValueError("cannot validate an empty embedding corpus")
        dense = client.query_points(
            collection_name=self.physical_collection,
            query=first_record.dense,
            using="dense",
            limit=3,
            with_payload=False,
        ).points
        sparse = client.query_points(
            collection_name=self.physical_collection,
            query=models.SparseVector(
                indices=sparse_record.sparse.indices,
                values=sparse_record.sparse.values,
            ),
            using="sparse",
            limit=3,
            with_payload=False,
        ).points
        if not dense or not sparse:
            raise ValueError("Qdrant dense/sparse smoke search returned no results")
        expected_ids = {v2_point_id(chunk_id) for chunk_id in chunks}
        if (
            str(dense[0].id) not in expected_ids
            or str(sparse[0].id) not in expected_ids
        ):
            raise ValueError("Qdrant smoke search returned an unknown point")
        return point_count, len(dense), len(sparse)

    def _switch_alias(self, client: QdrantClient) -> None:
        aliases = {
            alias.alias_name: alias.collection_name
            for alias in client.get_aliases().aliases
        }
        current = aliases.get(self.settings.alias)
        if current == self.physical_collection:
            return
        operations: list[object] = []
        if current is not None:
            operations.append(
                models.DeleteAliasOperation(
                    delete_alias=models.DeleteAlias(alias_name=self.settings.alias)
                )
            )
        operations.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=self.physical_collection,
                    alias_name=self.settings.alias,
                )
            )
        )
        client.update_collection_aliases(change_aliases_operations=operations)

    def _collection_name(self) -> str:
        corpus = re.sub(
            r"[^a-zA-Z0-9]+", "_", self.corpus.manifest.corpus_version
        ).strip("_")
        revision = self.artifacts.manifest.model_revision[:10]
        return f"{self.settings.collection_prefix}_{corpus}_{revision}".lower()

    @staticmethod
    def _validate_qdrant_target(path: Path) -> None:
        if path.name != "qdrant" or path.parent.name != "v2":
            raise ValueError(
                "V2 local Qdrant path must resolve to data/processed/v2/qdrant-style target"
            )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_report(path: Path, report: QdrantIngestionReport) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
