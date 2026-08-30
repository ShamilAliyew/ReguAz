"""OpenRouter text-to-speech adapter with strict binary validation."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from collections.abc import Mapping
from typing import Any, Awaitable, Callable

import httpx

from backend.reguaz.services.speech.text_normalizer import prepare_speech_text
from backend.reguaz.utils.logger import get_logger


logger = get_logger(__name__, "tts.log")

FISH_AUDIO_S21_FREE = "fish-audio/s2.1-pro-free:free"
SUPPORTED_TTS_MODELS = frozenset({FISH_AUDIO_S21_FREE})
_RETRYABLE_STATUSES = frozenset({502, 503, 504})
_MP3_CONTENT_TYPES = frozenset({"audio/mpeg", "audio/mp3"})


class SpeechInputError(ValueError):
    """The requested text cannot be safely synthesized."""


class SpeechInputTooLongError(SpeechInputError):
    """The normalized text exceeds the configured provider-safe limit."""


class SpeechProviderError(RuntimeError):
    """OpenRouter returned an invalid or unavailable audio response."""

    def __init__(self, message: str, *, upstream_status: int | None = None) -> None:
        super().__init__(message)
        self.upstream_status = upstream_status


class SpeechRateLimitError(SpeechProviderError):
    """OpenRouter rejected the request due to a rate limit."""

    def __init__(self, retry_after_seconds: int | None = None) -> None:
        super().__init__("TTS provider rate limit reached", upstream_status=429)
        self.retry_after_seconds = retry_after_seconds


@dataclass(frozen=True, slots=True)
class SpeechAudio:
    content: bytes
    content_type: str
    generation_id: str | None
    model_id: str
    elapsed_ms: float
    normalized_characters: int


@dataclass(frozen=True, slots=True)
class _RawSpeechResponse:
    status_code: int
    headers: Mapping[str, str]
    content: bytes


class OpenRouterTTSService:
    """Generate MP3 speech without exposing provider credentials to the UI."""

    provider = "openrouter"
    output_format = "mp3"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        model_id: str = FISH_AUDIO_S21_FREE,
        voice: str | None = None,
        site_url: str | None = None,
        app_title: str | None = "ReguAZ",
        timeout_seconds: float = 60.0,
        max_input_characters: int = 5_000,
        max_audio_bytes: int = 15 * 1024 * 1024,
        max_concurrency: int = 2,
        max_retries: int = 1,
        client: Any | None = None,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if not api_key.strip() and client is None:
            raise EnvironmentError("OPENROUTER_API_KEY is not configured")
        if not base_url.startswith("https://"):
            raise ValueError("OpenRouter TTS base URL must use HTTPS")
        if model_id not in SUPPORTED_TTS_MODELS:
            raise ValueError(f"unsupported OpenRouter TTS model: {model_id}")
        if timeout_seconds <= 0:
            raise ValueError("TTS timeout must be positive")
        if max_input_characters < 1 or max_audio_bytes < 1:
            raise ValueError("TTS input/audio limits must be positive")
        if max_concurrency < 1 or max_retries < 0:
            raise ValueError("TTS concurrency/retry settings are invalid")

        self.model_id = model_id
        self.voice = voice.strip() if voice and voice.strip() else None
        self.max_input_characters = max_input_characters
        self.max_audio_bytes = max_audio_bytes
        self.max_retries = max_retries
        self._endpoint = f"{base_url.rstrip('/')}/audio/speech"
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
        }
        if site_url:
            self._headers["HTTP-Referer"] = site_url
        if app_title:
            self._headers["X-OpenRouter-Title"] = app_title
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 10.0)),
            limits=httpx.Limits(max_connections=max_concurrency + 2),
            follow_redirects=False,
        )
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._sleeper = sleeper

    async def synthesize(self, text: str) -> SpeechAudio:
        try:
            normalized = prepare_speech_text(text)
        except ValueError as exc:
            raise SpeechInputError(str(exc)) from exc
        if len(normalized) > self.max_input_characters:
            raise SpeechInputTooLongError(
                f"speech text exceeds {self.max_input_characters} characters"
            )

        payload: dict[str, Any] = {
            "model": self.model_id,
            "input": normalized,
            "response_format": self.output_format,
            "provider": {"zdr": True, "data_collection": "deny"},
        }
        # OpenRouter's generic speech contract requires a voice, while current
        # Fish Audio metadata exposes no supported voice IDs. Omit it by default
        # and allow a verified provider-specific value through configuration.
        if self.voice is not None:
            payload["voice"] = self.voice

        started = time.perf_counter()
        async with self._semaphore:
            response = await self._post_with_bounded_retry(payload)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
        content = bytes(response.content)

        if content_type not in _MP3_CONTENT_TYPES:
            logger.warning(
                "TTS response rejected status=%s content_type=%s",
                response.status_code,
                content_type or "missing",
            )
            raise SpeechProviderError("TTS provider returned a non-MP3 response")
        if not content or not _looks_like_mp3(content):
            raise SpeechProviderError("TTS provider returned invalid MP3 audio")
        if len(content) > self.max_audio_bytes:
            raise SpeechProviderError("TTS provider audio exceeds the configured limit")

        generation_id = response.headers.get("x-generation-id")
        logger.info(
            "TTS complete model=%s chars=%d bytes=%d latency_ms=%.1f generation_id=%s",
            self.model_id,
            len(normalized),
            len(content),
            elapsed_ms,
            generation_id or "missing",
        )
        return SpeechAudio(
            content=content,
            content_type="audio/mpeg",
            generation_id=generation_id,
            model_id=self.model_id,
            elapsed_ms=elapsed_ms,
            normalized_characters=len(normalized),
        )

    async def _post_with_bounded_retry(self, payload: dict[str, Any]) -> Any:
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._send_once(payload)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt >= self.max_retries:
                    logger.warning(
                        "TTS transport failed category=%s", type(exc).__name__
                    )
                    raise SpeechProviderError(
                        "TTS provider is temporarily unavailable"
                    ) from exc
                await self._sleeper(0.25 * (2**attempt))
                continue

            if response.status_code == 429:
                raise SpeechRateLimitError(
                    _parse_retry_after(response.headers.get("retry-after"))
                )
            if (
                response.status_code in _RETRYABLE_STATUSES
                and attempt < self.max_retries
            ):
                await self._sleeper(0.25 * (2**attempt))
                continue
            if response.status_code < 200 or response.status_code >= 300:
                logger.warning("TTS upstream error status=%d", response.status_code)
                raise SpeechProviderError(
                    "TTS provider rejected the request",
                    upstream_status=response.status_code,
                )
            return response
        raise SpeechProviderError("TTS provider is temporarily unavailable")

    async def _send_once(self, payload: dict[str, Any]) -> _RawSpeechResponse:
        stream = getattr(self._client, "stream", None)
        if callable(stream):
            async with stream(
                "POST",
                self._endpoint,
                headers=self._headers,
                json=payload,
            ) as response:
                content = bytearray()
                if 200 <= response.status_code < 300:
                    async for chunk in response.aiter_bytes():
                        if len(content) + len(chunk) > self.max_audio_bytes:
                            raise SpeechProviderError(
                                "TTS provider audio exceeds the configured limit"
                            )
                        content.extend(chunk)
                return _RawSpeechResponse(
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    content=bytes(content),
                )

        # Injectable lightweight fakes may implement only post(). Production
        # always uses the streaming branch above.
        response = await self._client.post(
            self._endpoint,
            headers=self._headers,
            json=payload,
        )
        content = bytes(response.content)
        if len(content) > self.max_audio_bytes:
            raise SpeechProviderError("TTS provider audio exceeds the configured limit")
        return _RawSpeechResponse(
            status_code=response.status_code,
            headers=dict(response.headers),
            content=content,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _looks_like_mp3(content: bytes) -> bool:
    if content.startswith(b"ID3"):
        return True
    return len(content) >= 2 and content[0] == 0xFF and (content[1] & 0xE0) == 0xE0


def _parse_retry_after(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return max(0, int(float(value)))
    except ValueError:
        try:
            delta = parsedate_to_datetime(value).timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            return None
        return max(0, int(delta))
