from __future__ import annotations

from typing import Any

import httpx
import pytest

from backend.reguaz.services.speech.openrouter_tts import (
    OpenRouterTTSService,
    SpeechInputError,
    SpeechProviderError,
    SpeechRateLimitError,
)
from backend.reguaz.services.speech.text_normalizer import prepare_speech_text


class FakeAsyncClient:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        self.calls.append({"url": url, **kwargs})
        return self.responses.pop(0)


def mp3_response(*, status: int = 200, generation_id: str = "gen-1") -> httpx.Response:
    return httpx.Response(
        status,
        content=b"ID3" + b"\x00" * 32,
        headers={
            "content-type": "audio/mpeg",
            "x-generation-id": generation_id,
        },
    )


@pytest.mark.asyncio
async def test_speech_request_is_private_mp3_and_normalized() -> None:
    client = FakeAsyncClient([mp3_response()])
    service = OpenRouterTTSService(api_key="secret", client=client)

    result = await service.synthesize("**AMB** 12.3 üzrə 25% tələb edir. [1]")

    call = client.calls[0]
    assert call["url"] == "https://openrouter.ai/api/v1/audio/speech"
    assert call["headers"]["Authorization"] == "Bearer secret"
    assert call["json"]["response_format"] == "mp3"
    assert call["json"]["provider"] == {"zdr": True, "data_collection": "deny"}
    assert "voice" not in call["json"]
    assert "Azərbaycan Mərkəzi Bankı" in call["json"]["input"]
    assert "on iki nöqtə üç" in call["json"]["input"]
    assert "iyirmi beş faiz" in call["json"]["input"]
    assert "mənbə" not in call["json"]["input"]
    assert "[1]" not in call["json"]["input"]
    assert result.content_type == "audio/mpeg"
    assert result.generation_id == "gen-1"


@pytest.mark.asyncio
async def test_verified_voice_is_sent_only_when_configured() -> None:
    client = FakeAsyncClient([mp3_response()])
    service = OpenRouterTTSService(
        api_key="secret", client=client, voice="verified-fish-voice"
    )

    await service.synthesize("Azərbaycan dilində cavab")

    assert client.calls[0]["json"]["voice"] == "verified-fish-voice"


@pytest.mark.asyncio
async def test_rate_limit_preserves_retry_after() -> None:
    client = FakeAsyncClient(
        [httpx.Response(429, headers={"retry-after": "12"}, json={"error": "busy"})]
    )
    service = OpenRouterTTSService(api_key="secret", client=client)

    with pytest.raises(SpeechRateLimitError) as error:
        await service.synthesize("Qısa cavab")

    assert error.value.retry_after_seconds == 12
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_transient_error_retries_once_before_audio_starts() -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    client = FakeAsyncClient([httpx.Response(503), mp3_response()])
    service = OpenRouterTTSService(
        api_key="secret", client=client, max_retries=1, sleeper=fake_sleep
    )

    await service.synthesize("Qısa cavab")

    assert len(client.calls) == 2
    assert sleeps == [0.25]


@pytest.mark.asyncio
async def test_non_audio_or_invalid_mp3_is_rejected() -> None:
    client = FakeAsyncClient(
        [
            httpx.Response(
                200,
                content=b'{"error":"bad"}',
                headers={"content-type": "application/json"},
            )
        ]
    )
    service = OpenRouterTTSService(api_key="secret", client=client)

    with pytest.raises(SpeechProviderError, match="non-MP3"):
        await service.synthesize("Qısa cavab")


@pytest.mark.asyncio
async def test_input_limit_applies_after_normalization() -> None:
    client = FakeAsyncClient([mp3_response()])
    service = OpenRouterTTSService(
        api_key="secret", client=client, max_input_characters=5
    )

    with pytest.raises(SpeechInputError, match="exceeds"):
        await service.synthesize("altı simvol")
    assert not client.calls


def test_text_normalization_preserves_azerbaijani_and_removes_markdown() -> None:
    value = prepare_speech_text(
        "## Başlıq\n- Əmanət [qaydası](https://example.test) 1/5 nisbətini göstərir."
    )

    assert value == "Başlıq Əmanət qaydası bir bölü beş nisbətini göstərir."


def test_text_normalization_removes_adjacent_and_grouped_citations() -> None:
    value = prepare_speech_text(
        "Bank tələbi yerinə yetirməlidir [3][1]. Limit də tətbiq olunur [2, 4]."
    )

    assert value == "Bank tələbi yerinə yetirməlidir. Limit də tətbiq olunur."
