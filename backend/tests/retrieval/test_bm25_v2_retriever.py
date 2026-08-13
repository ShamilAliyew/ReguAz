from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.reguaz.retrieval.bm25_v2_retriever import (
    BM25V2Retriever,
    tokenize_az_legal,
)


def test_azerbaijani_legal_tokenization_is_deterministic() -> None:
    text = "İCARƏÇİNİN öhdəliyi Maddə 7-1.2-də göstərilir."
    expected = ["icarəçinin", "öhdəliyi", "maddə", "7-1.2", "də", "göstərilir"]
    assert tokenize_az_legal(text) == expected
    assert tokenize_az_legal(text) == tokenize_az_legal(text)


def test_bm25_v2_indexes_only_child_content_and_joins_category(v2_root: Path) -> None:
    chunks_path = next((v2_root / "documents").glob("*/chunks.jsonl"))
    rows = [json.loads(line) for line in chunks_path.read_text().splitlines()]
    for row in rows:
        row["embedding_text"] += " EMBEDDINGONLYTOKEN"
    chunks_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    (chunks_path.parent / "parents.jsonl").write_text(
        json.dumps({"content": "PARENTONLYTOKEN"}) + "\n", encoding="utf-8"
    )

    retriever = BM25V2Retriever(v2_root=v2_root)
    content_hit = retriever.search("kapital", top_k=1)[0]
    embedding_only = retriever.search("EMBEDDINGONLYTOKEN", top_k=1)[0]
    parent_only = retriever.search("PARENTONLYTOKEN", top_k=1)[0]

    manifest = json.loads((v2_root / "manifest.json").read_text())
    assert retriever.corpus_size == manifest["child_count"]
    assert content_hit["score"] != 0
    assert set(content_hit) == {"chunk_id", "point_id", "score", "rank"}
    assert retriever.get_payload(content_hit["chunk_id"])["category"] == "laws"
    assert "embedding_text" not in retriever.get_payload(content_hit["chunk_id"])
    assert embedding_only["score"] == 0
    assert parent_only["score"] == 0


def test_bm25_v2_duplicate_chunk_id_fails_fast(v2_root: Path) -> None:
    chunks_path = next((v2_root / "documents").glob("*/chunks.jsonl"))
    lines = chunks_path.read_text(encoding="utf-8").splitlines()
    chunks_path.write_text("\n".join([*lines, lines[0]]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate V2 (BM25 )?chunk_id"):
        BM25V2Retriever(v2_root=v2_root)


def test_bm25_v2_rejects_empty_query(v2_root: Path) -> None:
    retriever = BM25V2Retriever(v2_root=v2_root)
    with pytest.raises(ValueError, match="query must not be empty"):
        retriever.search("   ")


def test_bm25_v2_keeps_child_with_symbol_only_content(v2_root: Path) -> None:
    chunks_path = next((v2_root / "documents").glob("*/chunks.jsonl"))
    rows = [json.loads(line) for line in chunks_path.read_text().splitlines()]
    rows[0]["content"] = "—"
    rows[0]["character_count"] = 1
    second = dict(rows[0])
    second["chunk_id"] = second["chunk_id"] + "-second"
    second["logical_chunk_id"] = second["logical_chunk_id"] + "-second"
    second["content"] = "kapital"
    second["embedding_text"] = "kapital"
    second["character_count"] = len(second["content"])
    rows.append(second)
    chunks_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    manifest_path = v2_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["child_count"] = len(rows)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    embedding_path = (
        v2_root / "embeddings" / "bge_m3" / manifest["corpus_version"] / "manifest.json"
    )
    embedding = json.loads(embedding_path.read_text())
    embedding["source_child_count"] = len(rows)
    embedding["embedded_chunk_count"] = len(rows)
    embedding_path.write_text(json.dumps(embedding), encoding="utf-8")

    retriever = BM25V2Retriever(v2_root=v2_root)

    assert retriever.corpus_size == len(rows)
    assert retriever.empty_token_document_count == 1
