from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient

from backend.reguaz.services.embeddings.v2_pipeline import EmbeddingSettings, V2EmbeddingPipeline
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
        aliases = {alias.alias_name: alias.collection_name for alias in client.get_aliases().aliases}
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
