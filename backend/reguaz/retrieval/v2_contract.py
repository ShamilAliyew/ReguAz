from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from backend.reguaz.services.chunking.models import Manifest
from backend.reguaz.services.embeddings.v2_models import EmbeddingManifest
from backend.reguaz.services.embeddings.v2_pipeline import CATEGORY_ALLOWLIST


@dataclass(frozen=True, slots=True)
class RetrievalFilters:
    category: str | None = None
    document_id: str | None = None
    is_current: bool | None = True

    def __post_init__(self) -> None:
        if self.category is not None and self.category not in CATEGORY_ALLOWLIST:
            raise ValueError(f"unsupported V2 category filter: {self.category}")
        if self.document_id is not None and not self.document_id.strip():
            raise ValueError("document_id filter must not be blank")

    @classmethod
    def from_value(
        cls, value: RetrievalFilters | Mapping[str, object] | None
    ) -> RetrievalFilters:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        allowed = {"category", "document_id", "is_current"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unsupported V2 filters: {sorted(unknown)}")
        return cls(
            category=_optional_string(value.get("category"), "category"),
            document_id=_optional_string(value.get("document_id"), "document_id"),
            is_current=_optional_bool(value.get("is_current", True)),
        )

    def as_dict(self) -> dict[str, str | bool | None]:
        return {
            "category": self.category,
            "document_id": self.document_id,
            "is_current": self.is_current,
        }


@dataclass(frozen=True, slots=True)
class V2RetrievalContract:
    v2_root: Path
    chunk_manifest_path: Path
    embedding_manifest_path: Path
    chunk_manifest: Manifest
    embedding_manifest: EmbeddingManifest

    @classmethod
    def load(cls, v2_root: Path) -> V2RetrievalContract:
        root = v2_root.resolve()
        chunk_manifest_path = root / "manifest.json"
        chunk_manifest = Manifest.model_validate_json(
            chunk_manifest_path.read_text(encoding="utf-8")
        )
        if chunk_manifest.failed_document_count:
            raise ValueError("V2 chunk manifest contains failed documents")

        embedding_manifest_path = (
            root
            / "embeddings"
            / "bge_m3"
            / chunk_manifest.corpus_version
            / "manifest.json"
        )
        embedding_manifest = EmbeddingManifest.model_validate_json(
            embedding_manifest_path.read_text(encoding="utf-8")
        )
        if embedding_manifest.corpus_version != chunk_manifest.corpus_version:
            raise ValueError("V2 chunk and embedding corpus versions differ")
        if embedding_manifest.source_child_count != chunk_manifest.child_count:
            raise ValueError("V2 chunk and embedding child counts differ")
        if embedding_manifest.embedded_chunk_count != chunk_manifest.child_count:
            raise ValueError("V2 embedding count does not match child count")
        if embedding_manifest.model_id != "BAAI/bge-m3":
            raise ValueError("V2 retrieval requires the BAAI/bge-m3 embedding model")
        if embedding_manifest.dense_dimension != 1024:
            raise ValueError("V2 retrieval requires 1024-dimensional dense vectors")
        if embedding_manifest.vector_modes != ["dense", "sparse"]:
            raise ValueError("V2 embedding manifest must contain dense and sparse modes")
        return cls(
            v2_root=root,
            chunk_manifest_path=chunk_manifest_path,
            embedding_manifest_path=embedding_manifest_path,
            chunk_manifest=chunk_manifest,
            embedding_manifest=embedding_manifest,
        )


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} filter must be a string or null")
    return value


def _optional_bool(value: object) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    raise ValueError("is_current filter must be a boolean or null")
