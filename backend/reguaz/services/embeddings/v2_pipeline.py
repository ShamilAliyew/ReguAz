from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Iterator

from backend.reguaz.services.chunking.models import ChildChunk, DocumentMetadata, Manifest

from .bge_m3_v2 import DenseSparseEncoder
from .v2_models import EmbeddingManifest, EmbeddingRecord


CATEGORY_ALLOWLIST = frozenset(
    {
        "laws",
        "risk_management",
        "reporting_and_audit",
        "prudential_regulations",
        "governance_and_compliance",
        "guidance_and_methodology",
        "payments_and_banking_operations",
        "aml_kyc",
    }
)


@dataclass(frozen=True, slots=True)
class ChunkInput:
    chunk: ChildChunk
    category: str


@dataclass(frozen=True, slots=True)
class EmbeddingSettings:
    input_root: Path = Path("data/processed/v2")
    output_root: Path = Path("data/processed/v2/embeddings")
    batch_size: int = 8
    max_length: int = 512
    shard_size: int = 256
    force: bool = False
    resume: bool = True

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.max_length <= 0 or self.shard_size <= 0:
            raise ValueError("batch_size, max_length, and shard_size must be positive")


class V2CorpusReader:
    def __init__(self, input_root: Path) -> None:
        self.input_root = input_root.resolve()
        manifest_path = self.input_root / "manifest.json"
        self.manifest = Manifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.failed_document_count:
            raise ValueError("V2 chunk manifest contains failed documents")
        self.manifest_path = manifest_path

    def read_all(self) -> list[ChunkInput]:
        documents_root = self.input_root / "documents"
        files = sorted(documents_root.glob("*/chunks.jsonl"))
        if len(files) != self.manifest.successful_document_count:
            raise ValueError(
                "V2 document/chunk file count does not match the chunk manifest"
            )
        result: list[ChunkInput] = []
        seen: set[str] = set()
        for chunks_path in files:
            document_path = chunks_path.parent / "document.json"
            document = DocumentMetadata.model_validate_json(
                document_path.read_text(encoding="utf-8")
            )
            if document.category not in CATEGORY_ALLOWLIST:
                raise ValueError(
                    f"unsupported category '{document.category}' in {document.source_relative_path}"
                )
            for line_number, line in enumerate(
                chunks_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                chunk = ChildChunk.model_validate_json(line)
                if chunk.document_id != document.document_id:
                    raise ValueError(
                        f"document mismatch in {chunks_path}:{line_number}"
                    )
                if chunk.chunk_id in seen:
                    raise ValueError(f"duplicate chunk_id in V2 corpus: {chunk.chunk_id}")
                seen.add(chunk.chunk_id)
                result.append(ChunkInput(chunk=chunk, category=document.category))
        result.sort(key=lambda item: item.chunk.chunk_id)
        if len(result) != self.manifest.child_count:
            raise ValueError(
                f"expected {self.manifest.child_count} V2 children, discovered {len(result)}"
            )
        return result


class V2EmbeddingPipeline:
    def __init__(self, settings: EmbeddingSettings, encoder: DenseSparseEncoder) -> None:
        self.settings = settings
        self.encoder = encoder
        self.reader = V2CorpusReader(settings.input_root)

    @property
    def final_directory(self) -> Path:
        return (
            self.settings.output_root.resolve()
            / "bge_m3"
            / self.reader.manifest.corpus_version
        )

    @property
    def staging_directory(self) -> Path:
        return self.final_directory.parent / f".{self.final_directory.name}.inprogress"

    def dry_run(self) -> dict[str, object]:
        chunks = self.reader.read_all()
        return {
            "corpus_version": self.reader.manifest.corpus_version,
            "source_child_count": len(chunks),
            "model_id": self.encoder.model_id,
            "model_revision": self.encoder.model_revision,
            "vector_modes": ["dense", "sparse"],
            "dense_dimension": self.encoder.dense_dimension,
            "batch_size": self.settings.batch_size,
            "max_length": self.settings.max_length,
            "shard_size": self.settings.shard_size,
            "expected_shards": math.ceil(len(chunks) / self.settings.shard_size),
            "output": str(self.final_directory),
        }

    def run(self) -> EmbeddingManifest:
        chunks = self.reader.read_all()
        final = self.final_directory
        staging = self.staging_directory
        if final.exists() and not self.settings.force:
            return self.validate_artifacts(final)
        if final.exists():
            self._validate_managed_target(final)
        if staging.exists() and not self.settings.resume:
            raise FileExistsError(
                f"in-progress embedding build exists: {staging}; enable resume or move it aside"
            )
        staging.mkdir(parents=True, exist_ok=True)
        shards_dir = staging / "shards"
        shards_dir.mkdir(exist_ok=True)
        shard_hashes: dict[str, str] = {}
        shard_count = math.ceil(len(chunks) / self.settings.shard_size)
        for shard_index in range(shard_count):
            start = shard_index * self.settings.shard_size
            shard_inputs = chunks[start : start + self.settings.shard_size]
            shard_name = f"part-{shard_index:05d}.jsonl"
            shard_path = shards_dir / shard_name
            if shard_path.exists() and self.settings.resume:
                self._validate_shard(shard_path, shard_inputs)
            else:
                self._write_shard(shard_path, shard_inputs)
            shard_hashes[f"shards/{shard_name}"] = _sha256_file(shard_path)
        manifest = self._build_manifest(len(chunks), shard_count, shard_hashes)
        _write_json(staging / "manifest.json", manifest.model_dump(mode="json"))
        self.validate_artifacts(staging)
        replaced = final.parent / f".{final.name}.replaced-{os.getpid()}"
        if replaced.exists():
            raise RuntimeError(f"unexpected embedding backup exists: {replaced}")
        if final.exists():
            final.rename(replaced)
        try:
            staging.rename(final)
        except Exception:
            if replaced.exists() and not final.exists():
                replaced.rename(final)
            raise
        if replaced.exists():
            shutil.rmtree(replaced)
        return self.validate_artifacts(final)

    def _write_shard(self, path: Path, inputs: list[ChunkInput]) -> None:
        temporary = path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for start in range(0, len(inputs), self.settings.batch_size):
                batch_inputs = inputs[start : start + self.settings.batch_size]
                output = self.encoder.encode_documents(
                    [item.chunk.embedding_text for item in batch_inputs],
                    batch_size=self.settings.batch_size,
                    max_length=self.settings.max_length,
                )
                for item, dense, sparse in zip(
                    batch_inputs, output.dense, output.sparse, strict=True
                ):
                    record = EmbeddingRecord(
                        chunk_id=item.chunk.chunk_id,
                        logical_chunk_id=item.chunk.logical_chunk_id,
                        document_id=item.chunk.document_id,
                        document_version_id=item.chunk.document_version_id,
                        embedding_input_sha256=_sha256_text(item.chunk.embedding_text),
                        model_id=self.encoder.model_id,
                        model_revision=self.encoder.model_revision,
                        dense=dense,
                        sparse=sparse,
                    )
                    stream.write(
                        json.dumps(record.model_dump(mode="json"), ensure_ascii=False)
                        + "\n"
                    )
                stream.flush()
                os.fsync(stream.fileno())
        temporary.replace(path)

    def _validate_shard(self, path: Path, expected: list[ChunkInput]) -> None:
        records = list(read_embedding_records(path))
        if len(records) != len(expected):
            raise ValueError(f"incomplete embedding shard: {path}")
        for record, item in zip(records, expected, strict=True):
            if record.chunk_id != item.chunk.chunk_id:
                raise ValueError(f"embedding shard order mismatch: {path}")
            if record.embedding_input_sha256 != _sha256_text(item.chunk.embedding_text):
                raise ValueError(f"stale embedding input in shard: {path}")
            self._validate_record(record)

    def validate_artifacts(self, directory: Path) -> EmbeddingManifest:
        manifest = EmbeddingManifest.model_validate_json(
            (directory / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest.corpus_version != self.reader.manifest.corpus_version:
            raise ValueError("embedding artifacts belong to a different corpus version")
        seen: set[str] = set()
        count = 0
        for relative, expected_hash in sorted(manifest.shard_sha256.items()):
            path = directory / relative
            if _sha256_file(path) != expected_hash:
                raise ValueError(f"embedding shard hash mismatch: {path}")
            for record in read_embedding_records(path):
                if record.chunk_id in seen:
                    raise ValueError(f"duplicate embedding chunk_id: {record.chunk_id}")
                seen.add(record.chunk_id)
                self._validate_record(record)
                count += 1
        if count != manifest.embedded_chunk_count or count != manifest.source_child_count:
            raise ValueError("embedding artifact count does not match manifest")
        return manifest

    def _validate_record(self, record: EmbeddingRecord) -> None:
        if record.model_id != self.encoder.model_id or record.model_revision != self.encoder.model_revision:
            raise ValueError("embedding record model identity mismatch")
        if len(record.dense) != self.encoder.dense_dimension:
            raise ValueError("embedding record dense dimension mismatch")
        if any(not math.isfinite(value) for value in record.dense + record.sparse.values):
            raise ValueError("embedding record contains non-finite values")
        norm = math.sqrt(sum(value * value for value in record.dense))
        if not 0.995 <= norm <= 1.005:
            raise ValueError(f"embedding record is not normalized: {norm}")

    def _build_manifest(
        self, child_count: int, shard_count: int, shard_hashes: dict[str, str]
    ) -> EmbeddingManifest:
        source_path = self.reader.manifest_path
        return EmbeddingManifest(
            corpus_version=self.reader.manifest.corpus_version,
            source_chunk_manifest=str(source_path),
            source_chunk_manifest_sha256=_sha256_file(source_path),
            source_child_count=child_count,
            embedded_chunk_count=child_count,
            model_id=self.encoder.model_id,
            model_revision=self.encoder.model_revision,
            model_library_version=version("FlagEmbedding"),
            transformers_version=version("transformers"),
            torch_version=version("torch"),
            vector_modes=["dense", "sparse"],
            dense_dimension=self.encoder.dense_dimension,
            dense_normalized=self.encoder.normalized,
            max_length=self.settings.max_length,
            batch_size=self.settings.batch_size,
            shard_size=self.settings.shard_size,
            shard_count=shard_count,
            shard_sha256=dict(sorted(shard_hashes.items())),
            category_allowlist=sorted(CATEGORY_ALLOWLIST),
            build_timestamp=datetime.now(timezone.utc),
        )

    @staticmethod
    def _validate_managed_target(path: Path) -> None:
        if path.parent.name != "bge_m3" or path.parent.parent.name != "embeddings":
            raise ValueError(f"refusing to replace unmanaged embedding target: {path}")


def read_embedding_records(path: Path) -> Iterator[EmbeddingRecord]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                yield EmbeddingRecord.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"invalid embedding record at {path}:{line_number}") from exc


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
