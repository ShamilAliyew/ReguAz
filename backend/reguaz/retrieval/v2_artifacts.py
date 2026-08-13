from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
import hashlib
from pathlib import Path
from typing import Any

from backend.reguaz.retrieval.v2_contract import V2RetrievalContract
from backend.reguaz.services.chunking.models import (
    ChildChunk,
    DocumentMetadata,
    ParentChunk,
)
from backend.reguaz.services.embeddings.v2_pipeline import CATEGORY_ALLOWLIST


class V2ArtifactStore:
    """Process-local authoritative child lookup with lazy per-document parents."""

    def __init__(
        self,
        *,
        v2_root: Path = Path("data/processed/v2"),
        contract: V2RetrievalContract | None = None,
    ) -> None:
        self.contract = contract or V2RetrievalContract.load(v2_root)
        self._children: dict[str, dict[str, Any]] = {}
        self._document_dirs: dict[str, Path] = {}
        self._document_metadata: dict[str, dict[str, Any]] = {}
        self._parent_document_ids: dict[str, str] = {}
        self._continuations: dict[str, list[str]] = defaultdict(list)
        self._parent_cache: dict[str, dict[str, Any]] = {}
        self._loaded_parent_documents: set[str] = set()
        self._source_cache: dict[str, str] = {}
        self._load_children()

    def _load_children(self) -> None:
        files = sorted((self.contract.v2_root / "documents").glob("*/chunks.jsonl"))
        expected_documents = self.contract.chunk_manifest.successful_document_count
        if len(files) != expected_documents:
            raise ValueError(
                f"V2 child artifact count mismatch: expected {expected_documents}, got {len(files)}"
            )
        for path in files:
            metadata = DocumentMetadata.model_validate_json(
                (path.parent / "document.json").read_text(encoding="utf-8")
            )
            if metadata.category not in CATEGORY_ALLOWLIST:
                raise ValueError(f"unsupported V2 category: {metadata.category}")
            self._document_dirs[metadata.document_id] = path.parent
            self._document_metadata[metadata.document_id] = metadata.model_dump(
                mode="json"
            )
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        child = ChildChunk.model_validate_json(line)
                    except Exception as exc:
                        raise ValueError(
                            f"invalid V2 child at {path}:{line_number}"
                        ) from exc
                    if child.document_id != metadata.document_id:
                        raise ValueError(
                            f"V2 child/document mismatch at {path}:{line_number}"
                        )
                    if child.chunk_id in self._children:
                        raise ValueError(f"duplicate V2 chunk_id: {child.chunk_id}")
                    record = child.model_dump(mode="json")
                    record.pop("embedding_text", None)
                    record.pop("atomic_units", None)
                    record["category"] = metadata.category
                    record["corpus_version"] = (
                        self.contract.chunk_manifest.corpus_version
                    )
                    self._children[child.chunk_id] = record
                    self._parent_document_ids[child.parent_chunk_id] = child.document_id
                    if child.continuation_of:
                        self._continuations[child.continuation_of].append(
                            child.chunk_id
                        )
        expected = self.contract.chunk_manifest.child_count
        if len(self._children) != expected:
            raise ValueError(
                f"V2 child count mismatch: expected {expected}, got {len(self._children)}"
            )

    @property
    def corpus_size(self) -> int:
        return len(self._children)

    def iter_children(self) -> Iterable[dict[str, Any]]:
        return self._children.values()

    def get_child(self, chunk_id: str) -> dict[str, Any] | None:
        value = self._children.get(chunk_id)
        return dict(value) if value is not None else None

    def get_parent(self, parent_id: str) -> dict[str, Any] | None:
        if parent_id not in self._parent_cache:
            document_id = self._parent_document_ids.get(parent_id)
            if document_id:
                self._load_document_parents(document_id)
        value = self._parent_cache.get(parent_id)
        return dict(value) if value is not None else None

    def get_artifact(self, artifact_id: str) -> tuple[str, dict[str, Any]] | None:
        child = self.get_child(artifact_id)
        if child is not None:
            return "child", child
        parent = self.get_parent(artifact_id)
        if parent is not None:
            return "parent", parent
        return None

    def get_continuations(self, chunk_id: str) -> list[dict[str, Any]]:
        return [
            dict(self._children[target])
            for target in sorted(self._continuations.get(chunk_id, []))
            if target in self._children
        ]

    def get_document_metadata(self, document_id: str) -> dict[str, Any] | None:
        value = self._document_metadata.get(document_id)
        return dict(value) if value is not None else None

    def iter_document_metadata(self) -> Iterable[dict[str, Any]]:
        return (dict(value) for value in self._document_metadata.values())

    def get_source_representation(self, document_id: str) -> dict[str, Any] | None:
        metadata = self._document_metadata.get(document_id)
        if metadata is None:
            return None
        text = self._source_cache.get(document_id)
        path = (
            self.contract.v2_root.parent
            / "cleaned_documents"
            / str(metadata["source_relative_path"])
        )
        if text is None:
            if not path.is_file():
                return None
            text = path.read_text(encoding="utf-8")
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest != metadata["source_sha256"]:
                raise ValueError(
                    f"cleaned source hash mismatch for document: {document_id}"
                )
            self._source_cache[document_id] = text
        return {
            "representation_id": (
                f"cleaned:{metadata['document_version_id']}:{metadata['source_sha256']}"
            ),
            "representation_type": "cleaned_markdown",
            "document_id": document_id,
            "document_version_id": metadata["document_version_id"],
            "source_sha256": metadata["source_sha256"],
            "source_relative_path": metadata["source_relative_path"],
            "path": path,
            "text": text,
        }

    def hydrate_for_reranker(
        self, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        hydrated: list[dict[str, Any]] = []
        for candidate in candidates:
            chunk_id = str(candidate["chunk_id"])
            child = self._children.get(chunk_id)
            if child is None:
                required = (
                    "content",
                    "document_title",
                    "canonical_locator",
                    "hierarchy",
                )
                if all(key in candidate for key in required):
                    hydrated.append(dict(candidate))
                    continue
                raise ValueError(
                    f"RRF candidate is absent from V2 child artifacts: {chunk_id}"
                )
            item = dict(candidate)
            for key in ("content", "document_title", "canonical_locator", "hierarchy"):
                item[key] = child[key]
            hydrated.append(item)
        return hydrated

    def resolve_parents(
        self, results: list[dict[str, Any]]
    ) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
        required: dict[str, str] = {}
        for result in results:
            payload = result.get("payload") or {}
            parent_id = payload.get("parent_chunk_id")
            document_id = payload.get("document_id")
            if parent_id and document_id:
                required[str(parent_id)] = str(document_id)

        by_document: dict[str, set[str]] = defaultdict(set)
        for parent_id, document_id in required.items():
            if parent_id not in self._parent_cache:
                by_document[document_id].add(parent_id)
        for document_id in by_document:
            self._load_document_parents(document_id)

        found = {
            parent_id: dict(self._parent_cache[parent_id])
            for parent_id in required
            if parent_id in self._parent_cache
        }
        warnings = [
            {
                "code": "missing_parent",
                "parent_chunk_id": parent_id,
                "document_id": document_id,
            }
            for parent_id, document_id in sorted(required.items())
            if parent_id not in found
        ]
        return found, warnings

    def _load_document_parents(self, document_id: str) -> None:
        if document_id in self._loaded_parent_documents:
            return
        directory = self._document_dirs.get(document_id)
        if directory is None:
            self._loaded_parent_documents.add(document_id)
            return
        path = directory / "parents.jsonl"
        if path.is_file():
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        parent = ParentChunk.model_validate_json(line)
                    except Exception as exc:
                        raise ValueError(
                            f"invalid V2 parent at {path}:{line_number}"
                        ) from exc
                    if parent.document_id != document_id:
                        raise ValueError(
                            f"V2 parent/document mismatch at {path}:{line_number}"
                        )
                    self._parent_cache[parent.parent_chunk_id] = parent.model_dump(
                        mode="json"
                    )
        self._loaded_parent_documents.add(document_id)
