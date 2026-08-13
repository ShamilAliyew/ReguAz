from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.reguaz.retrieval.v2_artifacts import V2ArtifactStore
from backend.reguaz.retrieval.v2_contract import V2RetrievalContract


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Evidence(StrictModel):
    chunk_id: str
    logical_chunk_id: str
    document_version_id: str
    document_title: str
    canonical_locator: str
    page_start: int | None
    page_end: int | None
    content_sha256: str
    content: str


class ExpectedCitation(StrictModel):
    document_version_id: str
    document_title: str
    canonical_locator: str
    page_start: int | None
    page_end: int | None


class EvaluationMetadata(StrictModel):
    source_profile: str
    requires_multiple_chunks: bool
    generation_notes: str


class EvaluationRecord(StrictModel):
    id: str
    language: Literal["az"] = "az"
    split: Literal["dev", "test"]
    difficulty: Literal["easy", "medium", "hard"]
    category: str
    question_type: str
    question: str = Field(min_length=1)
    user_input: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    reference_contexts: list[str] = Field(min_length=1)
    relevant_chunk_ids: list[str] = Field(min_length=1)
    relevant_logical_chunk_ids: list[str] = Field(min_length=1)
    relevant_document_version_ids: list[str] = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)
    expected_citations: list[ExpectedCitation] = Field(min_length=1)
    metadata: EvaluationMetadata

    @model_validator(mode="after")
    def canonical_aliases_and_evidence(self) -> "EvaluationRecord":
        if self.question != self.user_input:
            raise ValueError("question and user_input must be identical")
        if self.reference != self.reference_answer:
            raise ValueError("reference and reference_answer must be identical")
        if self.reference_contexts != [item.content for item in self.evidence]:
            raise ValueError("reference_contexts must exactly follow evidence content")
        if self.relevant_chunk_ids != [item.chunk_id for item in self.evidence]:
            raise ValueError("relevant_chunk_ids must exactly follow evidence")
        if self.metadata.requires_multiple_chunks != (len(self.evidence) > 1):
            raise ValueError("requires_multiple_chunks disagrees with evidence count")
        return self


class DatasetManifest(StrictModel):
    schema_version: Literal["2.0"] = "2.0"
    dataset_name: str
    dataset_version: str
    creation_timestamp: datetime
    deterministic_seed: int
    total_record_count: int
    dev_count: int
    test_count: int
    difficulty_distribution: dict[str, int]
    category_distribution: dict[str, int]
    question_type_distribution: dict[str, int]
    corpus_id: str
    v2_manifest_sha256: str
    source_chunk_artifact_sha256: str
    source_parent_artifact_sha256: str
    generator_implementation_version: str
    drafting_model_id: str | None
    drafting_model_revision: str | None
    validation_status: str
    dataset_jsonl_sha256: str
    near_duplicate_threshold: float
    known_limitations: list[str]


def stable_record_id(question: str, chunk_ids: list[str]) -> str:
    value = unicodedata.normalize("NFC", question).strip() + "\n" + "\n".join(chunk_ids)
    return "reguaz_v2_eval_" + hashlib.sha256(value.encode()).hexdigest()[:20]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_files(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def validate_dataset(
    dataset_path: Path,
    manifest_path: Path,
    *,
    v2_root: Path = Path("data/processed/v2"),
) -> dict[str, Any]:
    contract = V2RetrievalContract.load(v2_root)
    store = V2ArtifactStore(v2_root=v2_root, contract=contract)
    records = [
        EvaluationRecord.model_validate_json(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = DatasetManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    failures: list[str] = []
    if len(records) != 50:
        failures.append(f"record count is {len(records)}, expected 50")
    ids = [record.id for record in records]
    if len(ids) != len(set(ids)):
        failures.append("record IDs are not unique")
    expected_counts = {
        "split": {"dev": 10, "test": 40},
        "difficulty": {"easy": 20, "medium": 20, "hard": 10},
    }
    for name, expected in expected_counts.items():
        actual = Counter(getattr(record, name) for record in records)
        if dict(actual) != expected:
            failures.append(f"{name} distribution mismatch: {dict(actual)}")

    questions = [_normalize(record.question) for record in records]
    for index, left in enumerate(questions):
        for right in questions[index + 1 :]:
            if _jaccard(left, right) > manifest.near_duplicate_threshold:
                failures.append("near-duplicate question threshold exceeded")
                break

    for record in records:
        if record.id != stable_record_id(record.question, record.relevant_chunk_ids):
            failures.append(f"unstable record ID: {record.id}")
        if not _looks_azerbaijani(record.question):
            failures.append(f"question does not look Azerbaijani: {record.id}")
        for item in record.evidence:
            child = store.get_child(item.chunk_id)
            if child is None:
                failures.append(f"missing chunk: {item.chunk_id}")
                continue
            expected = {
                "logical_chunk_id": child["logical_chunk_id"],
                "document_version_id": child["document_version_id"],
                "document_title": child["document_title"],
                "canonical_locator": child["canonical_locator"],
                "page_start": child["page_start"],
                "page_end": child["page_end"],
                "content_sha256": child["content_sha256"],
                "content": child["content"],
            }
            actual = item.model_dump(exclude={"chunk_id"})
            if actual != expected:
                failures.append(f"authoritative evidence mismatch: {item.chunk_id}")
            if child["content_sha256"] == hashlib.sha256(b"").hexdigest():
                failures.append(f"empty evidence chunk: {item.chunk_id}")
            if (
                item.page_start is not None
                and item.page_end is not None
                and item.page_start > item.page_end
            ):
                failures.append(f"invalid page range: {item.chunk_id}")
        if set(record.relevant_logical_chunk_ids) != {
            item.logical_chunk_id for item in record.evidence
        }:
            failures.append(f"logical IDs mismatch: {record.id}")
        if set(record.relevant_document_version_ids) != {
            item.document_version_id for item in record.evidence
        }:
            failures.append(f"document version IDs mismatch: {record.id}")
        answer_tokens = _answer_tokens(record.reference_answer)
        evidence_tokens = _answer_tokens(" ".join(record.reference_contexts))
        overlap = len(answer_tokens & evidence_tokens) / len(answer_tokens)
        if overlap < 0.30:
            failures.append(
                f"reference answer has insufficient lexical evidence support ({overlap:.2f}): {record.id}"
            )

    actual_dataset_hash = sha256_file(dataset_path)
    if manifest.dataset_jsonl_sha256 != actual_dataset_hash:
        failures.append("dataset SHA-256 mismatch")
    chunk_files = list((contract.v2_root / "documents").glob("*/chunks.jsonl"))
    parent_files = list((contract.v2_root / "documents").glob("*/parents.jsonl"))
    checks = {
        "v2_manifest_sha256": sha256_file(contract.chunk_manifest_path),
        "source_chunk_artifact_sha256": sha256_files(chunk_files, contract.v2_root),
        "source_parent_artifact_sha256": sha256_files(parent_files, contract.v2_root),
    }
    for field, actual in checks.items():
        if getattr(manifest, field) != actual:
            failures.append(f"manifest {field} mismatch")
    if manifest.total_record_count != len(records):
        failures.append("manifest record count mismatch")
    if failures:
        raise ValueError("V2 evaluation validation failed:\n- " + "\n- ".join(failures))
    return {
        "status": "valid",
        "record_count": len(records),
        "dataset_sha256": actual_dataset_hash,
        **checks,
    }


def _normalize(value: str) -> set[str]:
    return {
        part for part in unicodedata.normalize("NFC", value).casefold().split() if part
    }


def _answer_tokens(value: str) -> set[str]:
    return set(
        re.findall(
            r"\d+(?:[-.]\d+)*|[^\W\d_]+",
            unicodedata.normalize("NFC", value).casefold(),
            flags=re.UNICODE,
        )
    )


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left or right else 1.0


def _looks_azerbaijani(value: str) -> bool:
    lowered = value.casefold()
    markers = (
        " nə",
        " hansı",
        " necə",
        " neçə",
        " üçün",
        " üzrə",
        " olduqda",
        " edilməlidir",
    )
    return value.rstrip().endswith(("?", ".")) and (
        any(marker in " " + lowered for marker in markers)
        or any(character in lowered for character in "əöüğışç")
    )
