"""Groq-hosted GPT-OSS generation service."""

from __future__ import annotations

import os
import time
from typing import Any, Literal

from openai import OpenAI

from backend.reguaz.services.generation.base_llm import BaseLLM
from backend.reguaz.utils.logger import get_logger


logger = get_logger(__name__, "llm_generation.log")

GROQ_GPT_OSS_MODELS = frozenset(
    {
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
    }
)


class GroqGptOssService(BaseLLM):
    """Remote GPT-OSS provider using strict JSON-schema output."""

    provider = "groq"
    runtime = "groq_openai_compatible"
    device = "remote:groq"

    def __init__(
        self,
        *,
        model_id: str,
        base_url: str,
        context_window: int,
        temperature: float,
        max_tokens: int,
        top_p: float,
        timeout_seconds: float,
        reasoning_effort: Literal["low", "medium", "high"] = "low",
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        if model_id not in GROQ_GPT_OSS_MODELS:
            raise ValueError(f"Unsupported Groq GPT-OSS model: {model_id}")
        if not base_url.strip() or not base_url.startswith("https://"):
            raise ValueError("Groq API base must be an HTTPS URL")
        if context_window < 1 or max_tokens < 1 or max_tokens >= context_window:
            raise ValueError("Groq context/output token configuration is invalid")
        if not 0 < temperature <= 2 or not 0 <= top_p <= 1:
            raise ValueError("Groq sampling configuration is invalid")
        if timeout_seconds <= 0:
            raise ValueError("Groq timeout must be positive")
        if reasoning_effort not in {"low", "medium", "high"}:
            raise ValueError("Groq reasoning effort is invalid")
        resolved_key = api_key or os.getenv("GROQ_API_KEY")
        if client is None and not resolved_key:
            raise EnvironmentError("GROQ_API_KEY is not configured")

        self.model_id = model_id
        self.model_path = model_id
        self.base_url = base_url.rstrip("/")
        self.context_window = context_window
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.timeout_seconds = timeout_seconds
        self.reasoning_effort = reasoning_effort
        self._client = client or OpenAI(
            base_url=self.base_url,
            api_key=resolved_key,
            timeout=timeout_seconds,
            max_retries=2,
        )
        logger.info(
            "Groq GPT-OSS service ready model=%s timeout_s=%.1f reasoning=%s",
            self.model_id,
            self.timeout_seconds,
            self.reasoning_effort,
        )

    def count_tokens(self, text: str) -> int:
        # No provider tokenize endpoint. UTF-8 bytes are a conservative bound.
        return len(text.encode("utf-8")) if text else 0

    def count_chat_tokens(self, user_content: str) -> int:
        if not user_content.strip():
            raise ValueError("chat user content must not be empty")
        return self.count_tokens(user_content) + 64

    def get_prompt_budget(self) -> int:
        return self.context_window - self.max_tokens - 256

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        return str(self._complete(prompt, self.max_tokens, None)["text"])

    def generate_structured(
        self,
        user_content: str,
        schema: dict[str, Any],
        *,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        if not schema:
            raise ValueError("structured generation schema must not be empty")
        output_tokens = max_tokens or self.max_tokens
        if self.count_chat_tokens(user_content) + output_tokens > self.context_window:
            raise ValueError("rendered chat prompt exceeds configured context window")
        return self._complete(user_content, output_tokens, schema)

    def _complete(
        self,
        user_content: str,
        max_tokens: int,
        schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        request: dict[str, Any] = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": user_content}],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": max_tokens,
            "reasoning_effort": self.reasoning_effort,
            "stream": False,
            "extra_body": {"include_reasoning": False},
        }
        if schema is not None:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "reguaz_v2_answer",
                    "strict": True,
                    "schema": schema,
                },
            }
        try:
            completion = self._client.chat.completions.create(**request)
        except Exception as exc:
            logger.error("Groq request failed error_category=%s", type(exc).__name__)
            raise RuntimeError("Groq request failed") from exc

        if not completion.choices:
            raise RuntimeError("Groq response has no choices")
        message = completion.choices[0].message
        content = (message.content or "").strip()
        if not content:
            raise RuntimeError("Groq response has no final content")
        usage = completion.usage
        prompt_tokens = (
            int(usage.prompt_tokens)
            if usage is not None
            else self.count_chat_tokens(user_content)
        )
        completion_tokens = int(usage.completion_tokens) if usage is not None else 0
        details = getattr(usage, "prompt_tokens_details", None)
        cached_tokens = int(getattr(details, "cached_tokens", 0) or 0)
        return {
            "text": content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_prompt_tokens": cached_tokens,
            "generation_ms": (time.perf_counter() - started) * 1000.0,
            "provider_queue_ms": _seconds_to_ms(getattr(usage, "queue_time", None)),
            "provider_prompt_ms": _seconds_to_ms(getattr(usage, "prompt_time", None)),
            "provider_completion_ms": _seconds_to_ms(
                getattr(usage, "completion_time", None)
            ),
            "provider_total_ms": _seconds_to_ms(getattr(usage, "total_time", None)),
        }

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


def _seconds_to_ms(value: Any) -> float | None:
    return float(value) * 1000.0 if value is not None else None
