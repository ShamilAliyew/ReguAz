from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.reguaz.services.generation.nvidia_gpt_service import (
    NvidiaGptOssService,
)


class FakeCompletions:
    def __init__(self, content: str | None = '{"status":"answered"}') -> None:
        self.content = content
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        message = SimpleNamespace(
            content=self.content,
            reasoning_content="private reasoning must not leave provider",
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=101, completion_tokens=23),
        )


class FakeClient:
    def __init__(self, content: str | None = '{"status":"answered"}') -> None:
        self.completions = FakeCompletions(content)
        self.chat = SimpleNamespace(completions=self.completions)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def service(client: FakeClient) -> NvidiaGptOssService:
    return NvidiaGptOssService(
        model_id="openai/gpt-oss-120b",
        base_url="https://integrate.api.nvidia.com/v1",
        context_window=131072,
        temperature=1.0,
        max_tokens=4096,
        top_p=1.0,
        timeout_seconds=30,
        client=client,
    )


def test_structured_generation_uses_contract_and_hides_reasoning() -> None:
    client = FakeClient()
    model = service(client)
    result = model.generate_structured("JSON qaytar", {"type": "object"})
    assert result["text"] == '{"status":"answered"}'
    assert result["prompt_tokens"] == 101
    assert result["completion_tokens"] == 23
    assert "reasoning" not in result
    call = client.completions.calls[0]
    assert call["model"] == "openai/gpt-oss-120b"
    assert call["temperature"] == 1.0
    assert call["top_p"] == 1.0
    assert call["max_tokens"] == 4096
    assert call["stream"] is False


def test_empty_final_content_fails_closed_and_close_releases_client() -> None:
    client = FakeClient(None)
    model = service(client)
    with pytest.raises(RuntimeError, match="no final content"):
        model.generate_structured("JSON qaytar", {"type": "object"})
    model.close()
    assert client.closed is True


def test_missing_api_key_rejected_without_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="NVIDIA_API_KEY"):
        NvidiaGptOssService(
            model_id="openai/gpt-oss-120b",
            base_url="https://integrate.api.nvidia.com/v1",
            context_window=131072,
            temperature=1.0,
            max_tokens=4096,
            top_p=1.0,
            timeout_seconds=30,
        )
