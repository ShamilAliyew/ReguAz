from __future__ import annotations

from types import SimpleNamespace

import torch

from backend.reguaz.retrieval import reranker as reranker_module
from backend.reguaz.retrieval.reranker import CrossEncoderReranker


class FakeEmbeddings(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(4, 2)
        self.register_buffer("position_ids", torch.full((4,), 99), persistent=False)


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(max_position_embeddings=4)
        self.new = SimpleNamespace(embeddings=FakeEmbeddings())

    @property
    def device(self) -> torch.device:
        return self.anchor.device


class FakeCrossEncoder:
    last_kwargs: dict[str, object] = {}

    def __init__(self, model_name: str, **kwargs: object) -> None:
        self.model = FakeModel()
        self.last_kwargs = {"model_name": model_name, **kwargs}
        FakeCrossEncoder.last_kwargs = self.last_kwargs

    def predict(self, pairs: list[list[str]], **kwargs: object) -> list[float]:
        return [float(index) for index, _ in enumerate(pairs)]


def test_alibaba_remote_code_is_pinned_and_position_buffer_is_repaired(
    monkeypatch,
) -> None:
    monkeypatch.setattr(reranker_module, "CrossEncoder", FakeCrossEncoder)
    reranker = CrossEncoderReranker(
        model_name="Alibaba-NLP/gte-multilingual-reranker-base",
        revision="model-revision",
        trust_remote_code=True,
        code_revision="code-revision",
        local_files_only=True,
        device="cpu",
        show_progress_bar=False,
    )

    assert FakeCrossEncoder.last_kwargs["revision"] == "model-revision"
    assert FakeCrossEncoder.last_kwargs["trust_remote_code"] is True
    assert FakeCrossEncoder.last_kwargs["model_kwargs"] == {
        "code_revision": "code-revision"
    }
    assert FakeCrossEncoder.last_kwargs["processor_kwargs"] == {
        "code_revision": "code-revision"
    }
    assert FakeCrossEncoder.last_kwargs["config_kwargs"] == {
        "code_revision": "code-revision"
    }
    assert torch.equal(
        reranker._model.model.new.embeddings.position_ids,
        torch.arange(4),
    )
    assert reranker.rerank("sual", ["bir", "iki"]) == [0.0, 1.0]
