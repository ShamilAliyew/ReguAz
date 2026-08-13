from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.reguaz.services.generation.groq_service import GroqGptOssService


class FakeCompletions:
    def __init__(self, content: str | None = '{"status":"answered"}') -> None:
        self.content = content
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=self.content,
                        reasoning="private reasoning",
                    )
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=120,
                completion_tokens=30,
                prompt_tokens_details=SimpleNamespace(cached_tokens=64),
                queue_time=0.01,
                prompt_time=0.02,
                completion_time=0.03,
                total_time=0.06,
            ),
        )


class FakeClient:
    def __init__(self, content: str | None = '{"status":"answered"}') -> None:
        self.completions = FakeCompletions(content)
        self.chat = SimpleNamespace(completions=self.completions)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def service(client: FakeClient) -> GroqGptOssService:
    return GroqGptOssService(
        model_id="openai/gpt-oss-20b",
        base_url="https://api.groq.com/openai/v1",
        context_window=131072,
        temperature=0.2,
        max_tokens=2048,
        top_p=0.9,
        timeout_seconds=30,
        reasoning_effort="low",
        client=client,
    )


def test_groq_uses_strict_schema_and_hides_reasoning() -> None:
    client = FakeClient()
    model = service(client)
    schema = {
        "type": "object",
        "properties": {"status": {"type": "string"}},
        "required": ["status"],
        "additionalProperties": False,
    }
    result = model.generate_structured("JSON qaytar", schema)
    call = client.completions.calls[0]
    response_format = call["response_format"]
    assert isinstance(response_format, dict)
    assert response_format["json_schema"]["strict"] is True
    assert call["reasoning_effort"] == "low"
    assert call["extra_body"] == {"include_reasoning": False}
    assert "reasoning" not in result
    assert result["cached_prompt_tokens"] == 64
    assert result["provider_total_ms"] == pytest.approx(60.0)


def test_groq_rejects_unknown_model_and_missing_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Unsupported Groq"):
        GroqGptOssService(
            model_id="untrusted/model",
            base_url="https://api.groq.com/openai/v1",
            context_window=131072,
            temperature=0.2,
            max_tokens=2048,
            top_p=0.9,
            timeout_seconds=30,
        )
    with pytest.raises(EnvironmentError, match="GROQ_API_KEY"):
        GroqGptOssService(
            model_id="openai/gpt-oss-120b",
            base_url="https://api.groq.com/openai/v1",
            context_window=131072,
            temperature=0.2,
            max_tokens=2048,
            top_p=0.9,
            timeout_seconds=30,
        )


def test_groq_empty_response_fails_closed_and_client_closes() -> None:
    client = FakeClient(None)
    model = service(client)
    with pytest.raises(RuntimeError, match="no final content"):
        model.generate_structured("JSON qaytar", {"type": "object"})
    model.close()
    assert client.closed is True
