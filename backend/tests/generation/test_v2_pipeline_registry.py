from __future__ import annotations

from typing import Any

import pytest

from backend.reguaz.services.generation import v2_pipeline_registry as registry_module
from backend.reguaz.services.generation.v2_pipeline_registry import (
    ModelUnavailableError,
    V2GenerationPipelineRegistry,
)


class FakeLLM:
    provider = "groq"
    model_id = "openai/gpt-oss-20b"
    model_path = model_id
    device = "remote:groq"
    max_tokens = 256

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakePipeline:
    def __init__(self, **kwargs: Any) -> None:
        self.llm = kwargs["llm"]

    def generate(self, question: str) -> dict[str, str]:
        return {"question": question, "model": self.llm.model_id}


def test_registry_lazily_reuses_model_and_closes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-only")
    created: list[FakeLLM] = []

    def create(*, model_type: str) -> FakeLLM:
        assert model_type == "groq_gpt_oss_20b"
        model = FakeLLM()
        created.append(model)
        return model

    monkeypatch.setattr(registry_module.LLMFactory, "create", create)
    monkeypatch.setattr(registry_module, "V2GenerationPipeline", FakePipeline)
    registry = V2GenerationPipelineRegistry(
        retriever=object(),
        artifacts=object(),
        default_model="groq_gpt_oss_20b",
        generation_settings={},
    )
    assert created == []
    first = registry.generate("sual")
    second = registry.generate_for_model("ikinci", "groq_gpt_oss_20b")
    assert first["model"] == "openai/gpt-oss-20b"
    assert second["question"] == "ikinci"
    assert len(created) == 1
    assert registry.catalog()[1]["loaded"] is True
    registry.close()
    registry.close()
    assert created[0].closed is True


def test_registry_rejects_unconfigured_remote_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-only")
    monkeypatch.setattr(registry_module.LLMFactory, "create", lambda **_: FakeLLM())
    monkeypatch.setattr(registry_module, "V2GenerationPipeline", FakePipeline)
    registry = V2GenerationPipelineRegistry(
        retriever=object(),
        artifacts=object(),
        default_model="groq_gpt_oss_20b",
        generation_settings={},
    )
    registry.get_pipeline("groq_gpt_oss_20b")
    monkeypatch.delenv("GROQ_API_KEY")
    with pytest.raises(ModelUnavailableError, match="GROQ_API_KEY"):
        registry.get_pipeline("groq_gpt_oss_120b")
    registry.close()
