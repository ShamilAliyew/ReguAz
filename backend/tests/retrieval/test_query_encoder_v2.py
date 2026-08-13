from __future__ import annotations

import pytest

from backend.reguaz.services.embeddings.bge_m3_v2 import BgeM3DenseSparseEncoder


class FakeBgeModel:
    def __init__(self, *, empty_sparse: bool = False) -> None:
        self.empty_sparse = empty_sparse
        self.query_calls: list[tuple[list[str], dict[str, object]]] = []
        self.corpus_calls: list[tuple[list[str], dict[str, object]]] = []

    def _output(self, texts: list[str]) -> dict[str, object]:
        dense = [[1.0] + [0.0] * 1023 for _ in texts]
        sparse = [{} if self.empty_sparse else {"17": 0.75} for _ in texts]
        return {"dense_vecs": dense, "lexical_weights": sparse}

    def encode_queries(self, texts: list[str], **kwargs: object) -> dict[str, object]:
        self.query_calls.append((texts, kwargs))
        return self._output(texts)

    def encode_corpus(self, texts: list[str], **kwargs: object) -> dict[str, object]:
        self.corpus_calls.append((texts, kwargs))
        return self._output(texts)


def make_encoder(model: FakeBgeModel) -> BgeM3DenseSparseEncoder:
    encoder = object.__new__(BgeM3DenseSparseEncoder)
    encoder.model_revision = "test-revision"
    encoder._model = model
    return encoder


def test_query_encoder_uses_one_dense_sparse_query_call() -> None:
    model = FakeBgeModel()
    encoder = make_encoder(model)

    dense, sparse = encoder.encode_query("kapital normativi")

    assert len(model.query_calls) == 1
    assert model.query_calls[0][0] == ["kapital normativi"]
    assert model.corpus_calls == []
    assert len(dense) == 1024
    assert sum(value * value for value in dense) == pytest.approx(1.0)
    assert sparse.indices == [17]
    assert sparse.values == [0.75]


def test_query_encoder_rejects_empty_query() -> None:
    encoder = make_encoder(FakeBgeModel())
    with pytest.raises(ValueError, match="non-empty"):
        encoder.encode_query("   ")


def test_query_encoder_rejects_empty_sparse_output() -> None:
    encoder = make_encoder(FakeBgeModel(empty_sparse=True))
    with pytest.raises(ValueError, match="empty sparse"):
        encoder.encode_query("likvidlik")
