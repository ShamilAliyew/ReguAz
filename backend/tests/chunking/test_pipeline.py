from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.reguaz.services.chunking import ApproximateTokenizer, ChunkingConfig, ChunkingPipeline
from backend.reguaz.services.chunking.models import ChildChunk


def make_roots(tmp_path: Path) -> tuple[Path, Path]:
    input_root = tmp_path / "data" / "processed" / "cleaned_documents"
    output_root = tmp_path / "data" / "processed" / "v2"
    input_root.mkdir(parents=True)
    return input_root, output_root


def test_pipeline_stable_versioned_ids_relations_and_schemas(tmp_path: Path) -> None:
    input_root, output_root = make_roots(tmp_path)
    category = input_root / "laws"
    category.mkdir()
    source = category / "Qanun.md"
    original = "<!-- PAGE: 1 -->\nBank Qanunu\nMaddə 1. Başlıq\n1.1. Birinci mətn.\n1.2. 1.1-ci bənddə olan qayda.\nƏlavə 2\n| A | B |\n| --- | --- |\n| x | y |"
    source.write_text(original, encoding="utf-8")
    pipeline = ChunkingPipeline(ChunkingConfig(input_root=input_root, output_root=output_root), ApproximateTokenizer())
    first = pipeline.run()
    doc_dir = next((output_root / "documents").iterdir())
    first_chunks = [json.loads(line) for line in (doc_dir / "chunks.jsonl").read_text().splitlines()]
    first_doc = json.loads((doc_dir / "document.json").read_text())
    first_ids = [chunk["chunk_id"] for chunk in first_chunks]
    first_logical_ids = [chunk["logical_chunk_id"] for chunk in first_chunks]
    for payload in first_chunks:
        ChildChunk.model_validate(payload)
    assert first.succeeded
    assert all(chunk["parent_chunk_id"] for chunk in first_chunks)
    assert any(chunk["relations"]["references"] for chunk in first_chunks)
    assert any(chunk["relations"]["annex_parent"] for chunk in first_chunks)
    assert any(chunk["relations"]["table_parent"] for chunk in first_chunks)

    second = pipeline.run(force=True)
    doc_dir = next((output_root / "documents").iterdir())
    second_chunks = [json.loads(line) for line in (doc_dir / "chunks.jsonl").read_text().splitlines()]
    assert second.succeeded
    assert [chunk["chunk_id"] for chunk in second_chunks] == first_ids
    assert json.loads((doc_dir / "document.json").read_text())["document_version_id"] == first_doc["document_version_id"]

    source.write_text(original.replace("olan qayda", "olan yeni qayda"), encoding="utf-8")
    pipeline.run(force=True)
    doc_dir = next((output_root / "documents").iterdir())
    changed_doc = json.loads((doc_dir / "document.json").read_text())
    changed_chunks = [json.loads(line) for line in (doc_dir / "chunks.jsonl").read_text().splitlines()]
    assert changed_doc["document_id"] == first_doc["document_id"]
    assert changed_doc["document_version_id"] != first_doc["document_version_id"]
    assert [chunk["chunk_id"] for chunk in changed_chunks] != first_ids
    assert [chunk["logical_chunk_id"] for chunk in changed_chunks] == first_logical_ids


def test_unresolved_reference_and_approved_document_relation(tmp_path: Path) -> None:
    input_root, output_root = make_roots(tmp_path)
    category = input_root / "rules"
    category.mkdir()
    (category / "Qərar.md").write_text(
        "Qaydaların təsdiq edilməsi barədə\nQƏRARA ALIR:\n"
        "1. Qaydalar təsdiq edilsin.\n2. 99.7-ci bənd tətbiq edilir.\n"
        "1. Ümumi müddəalar\n1.1. Qaydanın mətni.",
        encoding="utf-8",
    )
    pipeline = ChunkingPipeline(
        ChunkingConfig(input_root=input_root, output_root=output_root), ApproximateTokenizer()
    )
    pipeline.run()
    chunks_path = next(output_root.glob("documents/*/chunks.jsonl"))
    chunks = [json.loads(line) for line in chunks_path.read_text().splitlines()]
    assert any(
        not reference["resolved"]
        for chunk in chunks
        for reference in chunk["relations"]["references"]
    )
    assert any(chunk["relations"]["approved_document"] for chunk in chunks)


def test_duplicate_documents_and_unicode_filename(tmp_path: Path) -> None:
    input_root, output_root = make_roots(tmp_path)
    content = "Maddə 1. Eyni qanun\n1.1. Eyni mətn."
    for category_name, filename in (("laws", "Qanun.md"), ("other", "Qanun surəti.md")):
        category = input_root / category_name
        category.mkdir()
        (category / filename).write_text(content, encoding="utf-8")
    pipeline = ChunkingPipeline(ChunkingConfig(input_root=input_root, output_root=output_root), ApproximateTokenizer())
    result = pipeline.run()
    assert result.manifest.discovered_document_count == 2
    assert result.manifest.duplicate_document_count == 1
    documents = [json.loads(path.read_text()) for path in output_root.glob("documents/*/document.json")]
    assert sum(not item["is_canonical_document"] for item in documents) == 1


def test_strict_schema_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ChildChunk.model_validate({"unexpected": True})


def test_existing_v2_requires_force_and_v1_is_untouched(tmp_path: Path) -> None:
    input_root, output_root = make_roots(tmp_path)
    category = input_root / "laws"
    category.mkdir()
    (category / "Qanun.md").write_text("Maddə 1. Mətn", encoding="utf-8")
    v1 = tmp_path / "data" / "processed" / "chunks"
    v1.mkdir()
    sentinel = v1 / "keep.jsonl"
    sentinel.write_text("v1", encoding="utf-8")
    pipeline = ChunkingPipeline(ChunkingConfig(input_root=input_root, output_root=output_root), ApproximateTokenizer())
    pipeline.run()
    with pytest.raises(FileExistsError):
        pipeline.run()
    assert sentinel.read_text(encoding="utf-8") == "v1"
