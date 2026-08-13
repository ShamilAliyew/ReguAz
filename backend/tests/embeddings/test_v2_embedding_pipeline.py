from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.reguaz.services.chunking import ApproximateTokenizer, ChunkingConfig, ChunkingPipeline
from backend.reguaz.services.embeddings.bge_m3_v2 import (
    BgeM3DenseSparseEncoder,
    DenseSparseBatch,
)
from backend.reguaz.services.embeddings.v2_models import EmbeddingRecord, SparseEmbedding
from backend.reguaz.services.embeddings.v2_pipeline import (
    EmbeddingSettings,
    V2CorpusReader,
    V2EmbeddingPipeline,
    read_embedding_records,
)


class FakeDenseSparseEncoder:
    model_id = "BAAI/bge-m3"
    model_revision = "test-revision"
    dense_dimension = 4
    normalized = True

    def encode_documents(
        self, texts: list[str], *, batch_size: int, max_length: int
    ) -> DenseSparseBatch:
        assert batch_size > 0 and max_length == 512
        dense = [[1.0, 0.0, 0.0, 0.0] for _ in texts]
        sparse = []
        for text in texts:
            token = int(hashlib.sha256(text.encode()).hexdigest()[:6], 16)
            sparse.append(SparseEmbedding(indices=[token], values=[1.0]))
        return DenseSparseBatch(dense=dense, sparse=sparse)


def build_corpus(tmp_path: Path, category: str = "laws") -> Path:
    input_root = tmp_path / "cleaned_documents"
    source = input_root / category / "Qanun.md"
    source.parent.mkdir(parents=True)
    source.write_text(
        "<!-- PAGE: 1 -->\nMaddə 1. Başlıq\n1.1. Birinci hüquqi mətn.\n1.2. İkinci hüquqi mətn.",
        encoding="utf-8",
    )
    v2_root = tmp_path / "data" / "processed" / "v2"
    ChunkingPipeline(
        ChunkingConfig(input_root=input_root, output_root=v2_root),
        ApproximateTokenizer(),
    ).run()
    return v2_root


def test_embedding_pipeline_writes_valid_dense_sparse_shards(tmp_path: Path) -> None:
    v2_root = build_corpus(tmp_path)
    output = v2_root / "embeddings"
    pipeline = V2EmbeddingPipeline(
        EmbeddingSettings(
            input_root=v2_root,
            output_root=output,
            batch_size=2,
            shard_size=1,
        ),
        FakeDenseSparseEncoder(),
    )
    manifest = pipeline.run()
    assert manifest.embedded_chunk_count > 0
    assert manifest.vector_modes == ["dense", "sparse"]
    assert manifest.dense_dimension == 4
    records = [
        record
        for shard in sorted(pipeline.final_directory.glob("shards/*.jsonl"))
        for record in read_embedding_records(shard)
    ]
    assert len(records) == manifest.embedded_chunk_count
    assert all(isinstance(record, EmbeddingRecord) for record in records)
    assert all(record.dense == [1.0, 0.0, 0.0, 0.0] for record in records)
    assert all(record.sparse.indices and record.sparse.values for record in records)
    assert pipeline.run().shard_sha256 == manifest.shard_sha256


def test_embedding_resume_rejects_stale_shard(tmp_path: Path) -> None:
    v2_root = build_corpus(tmp_path)
    pipeline = V2EmbeddingPipeline(
        EmbeddingSettings(input_root=v2_root, output_root=v2_root / "embeddings", shard_size=1),
        FakeDenseSparseEncoder(),
    )
    manifest = pipeline.run()
    shard = pipeline.final_directory / next(iter(manifest.shard_sha256))
    payload = json.loads(shard.read_text().splitlines()[0])
    payload["embedding_input_sha256"] = "0" * 64
    shard.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        pipeline.run()


def test_category_must_come_from_allowlisted_document_folder(tmp_path: Path) -> None:
    v2_root = build_corpus(tmp_path, category="not_allowed")
    with pytest.raises(ValueError, match="unsupported category"):
        V2CorpusReader(v2_root).read_all()


def test_sparse_embedding_may_be_empty_but_must_remain_aligned() -> None:
    assert SparseEmbedding(indices=[], values=[]).indices == []
    with pytest.raises(ValueError, match="aligned"):
        SparseEmbedding(indices=[1], values=[])


def test_bge_m3_encoder_passes_resolved_snapshot_to_flagembedding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import FlagEmbedding
    import huggingface_hub
    import transformers

    (tmp_path / "sparse_linear.pt").write_bytes(b"sparse")
    (tmp_path / "colbert_linear.pt").write_bytes(b"colbert")
    captured: dict[str, object] = {}

    class FakeModel:
        def __init__(self, model_path: str, **kwargs: object) -> None:
            captured["model_path"] = model_path
            captured["kwargs"] = kwargs

    monkeypatch.setattr(FlagEmbedding, "BGEM3FlagModel", FakeModel)
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda **kwargs: str(tmp_path),
    )
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(_commit_hash="pinned-revision"),
    )

    encoder = BgeM3DenseSparseEncoder(device="cpu", local_files_only=True)

    assert encoder.model_revision == "pinned-revision"
    assert captured["model_path"] == str(tmp_path)


def test_bge_m3_encoder_rejects_snapshot_without_auxiliary_heads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub
    import transformers

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda **kwargs: str(tmp_path),
    )
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(_commit_hash="pinned-revision"),
    )

    with pytest.raises(RuntimeError, match="auxiliary heads are missing"):
        BgeM3DenseSparseEncoder(device="cpu", local_files_only=True)
