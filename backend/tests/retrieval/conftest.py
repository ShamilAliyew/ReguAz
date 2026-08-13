from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.reguaz.services.chunking import (
    ApproximateTokenizer,
    ChunkingConfig,
    ChunkingPipeline,
)
from backend.reguaz.services.chunking.models import Manifest
from backend.reguaz.services.embeddings.v2_models import EmbeddingManifest


@pytest.fixture
def v2_root(tmp_path: Path) -> Path:
    source = tmp_path / "cleaned_documents" / "laws" / "Qanun.md"
    source.parent.mkdir(parents=True)
    source.write_text(
        "<!-- PAGE: 1 -->\n"
        "Maddə 1. Kapital tələbi\n"
        "1.1. Bank minimum kapital normativini qorumalıdır.\n"
        "1.2. Likvidlik göstəricisi hər gün hesablanır.\n"
        "1.3. Risk hesabatı aylıq təqdim edilir.",
        encoding="utf-8",
    )
    root = tmp_path / "data" / "processed" / "v2"
    ChunkingPipeline(
        ChunkingConfig(input_root=source.parents[1], output_root=root),
        ApproximateTokenizer(),
    ).run()
    manifest = Manifest.model_validate_json(
        (root / "manifest.json").read_text(encoding="utf-8")
    )
    embedding = EmbeddingManifest(
        corpus_version=manifest.corpus_version,
        source_chunk_manifest=str(root / "manifest.json"),
        source_chunk_manifest_sha256="0" * 64,
        source_child_count=manifest.child_count,
        embedded_chunk_count=manifest.child_count,
        model_id="BAAI/bge-m3",
        model_revision="test-revision",
        model_library_version="1.4.0",
        transformers_version="5.0.0",
        torch_version="2.0.0",
        vector_modes=["dense", "sparse"],
        dense_dimension=1024,
        dense_normalized=True,
        max_length=512,
        batch_size=4,
        shard_size=256,
        shard_count=1,
        shard_sha256={"shards/part-00000.jsonl": "0" * 64},
        category_allowlist=["laws"],
        build_timestamp=datetime.now(timezone.utc),
    )
    target = root / "embeddings" / "bge_m3" / manifest.corpus_version
    target.mkdir(parents=True)
    (target / "manifest.json").write_text(
        json.dumps(embedding.model_dump(mode="json")), encoding="utf-8"
    )
    return root


def first_child_payload(root: Path) -> dict[str, object]:
    chunks_path = next((root / "documents").glob("*/chunks.jsonl"))
    child = json.loads(chunks_path.read_text(encoding="utf-8").splitlines()[0])
    document = json.loads(
        (chunks_path.parent / "document.json").read_text(encoding="utf-8")
    )
    child.pop("embedding_text", None)
    child.pop("atomic_units", None)
    child["category"] = document["category"]
    child["corpus_version"] = json.loads(
        (root / "manifest.json").read_text(encoding="utf-8")
    )["corpus_version"]
    child["embedding_model_id"] = "BAAI/bge-m3"
    child["embedding_model_revision"] = "test-revision"
    return child
