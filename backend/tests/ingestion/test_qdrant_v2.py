from __future__ import annotations

from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from backend.reguaz.services.ingestion import qdrant_v2 as qdrant_v2_module
from backend.reguaz.services.embeddings.v2_pipeline import (
    EmbeddingSettings,
    V2EmbeddingPipeline,
)
from backend.reguaz.services.ingestion.qdrant_v2 import (
    QdrantV2IngestionPipeline,
    QdrantV2Settings,
)
from backend.tests.embeddings.test_v2_embedding_pipeline import (
    FakeDenseSparseEncoder,
    build_corpus,
)


def test_qdrant_v2_dense_sparse_ingestion_and_alias(tmp_path: Path) -> None:
    v2_root = build_corpus(tmp_path)
    embedding_pipeline = V2EmbeddingPipeline(
        EmbeddingSettings(
            input_root=v2_root,
            output_root=v2_root / "embeddings",
            batch_size=2,
            shard_size=2,
        ),
        FakeDenseSparseEncoder(),
    )
    embedding_manifest = embedding_pipeline.run()
    settings = QdrantV2Settings(
        v2_root=v2_root,
        embeddings_root=v2_root / "embeddings",
        qdrant_path=v2_root / "qdrant",
        upload_batch_size=2,
    )
    report = QdrantV2IngestionPipeline(settings).run()
    assert report.status == "success"
    assert report.point_count == embedding_manifest.embedded_chunk_count
    assert report.dense_smoke_result_count > 0
    assert report.sparse_smoke_result_count > 0

    client = QdrantClient(path=str(v2_root / "qdrant"))
    try:
        aliases = {
            alias.alias_name: alias.collection_name
            for alias in client.get_aliases().aliases
        }
        assert aliases["reguaz_v2_current"] == report.physical_collection
        assert client.count("reguaz_v2_current", exact=True).count == report.point_count
        info = client.get_collection(report.physical_collection)
        assert info.config.metadata["corpus_version"] == report.corpus_version
    finally:
        client.close()


def test_qdrant_v1_sentinel_is_outside_v2_storage(tmp_path: Path) -> None:
    v2_root = build_corpus(tmp_path)
    sentinel = tmp_path / "data" / "qdrant" / "v1.keep"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("v1", encoding="utf-8")
    assert (v2_root / "qdrant").resolve() != sentinel.parent.resolve()
    assert sentinel.read_text(encoding="utf-8") == "v1"


def test_remote_qdrant_ingestion_client_uses_url_and_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    v2_root = build_corpus(tmp_path)
    V2EmbeddingPipeline(
        EmbeddingSettings(
            input_root=v2_root,
            output_root=v2_root / "embeddings",
            batch_size=2,
            shard_size=2,
        ),
        FakeDenseSparseEncoder(),
    ).run()
    captured: dict[str, object] = {}
    fake_client = object()

    def fake_factory(**kwargs: object) -> object:
        captured.update(kwargs)
        return fake_client

    monkeypatch.setattr(qdrant_v2_module, "QdrantClient", fake_factory)
    pipeline = QdrantV2IngestionPipeline(
        QdrantV2Settings(
            v2_root=v2_root,
            embeddings_root=v2_root / "embeddings",
            qdrant_url="https://example.qdrant.io:6333",
            qdrant_api_key="test-secret",
        )
    )

    assert pipeline._create_client(v2_root / "qdrant") is fake_client
    assert captured == {
        "url": "https://example.qdrant.io:6333",
        "api_key": "test-secret",
        "timeout": 60.0,
    }


def test_remote_qdrant_ingestion_rejects_invalid_url() -> None:
    with pytest.raises(ValueError, match="must use http"):
        QdrantV2Settings(qdrant_url="example.qdrant.io:6333")


def test_remote_qdrant_ingestion_rejects_hidden_api_key_character() -> None:
    with pytest.raises(ValueError, match=r"QDRANT_API_KEY.*U\+2028"):
        QdrantV2Settings(
            qdrant_url="https://example.qdrant.io:6333",
            qdrant_api_key="token\u2028value",
        )


def test_remote_qdrant_ingestion_rejects_markdown_url() -> None:
    with pytest.raises(ValueError, match="plain URL"):
        QdrantV2Settings(
            qdrant_url="[https://example.qdrant.io](https://example.qdrant.io)",
            qdrant_api_key="test-secret",
        )
