"""OpenAI-compatible NVIDIA NIM generation service for GPT-OSS."""

from __future__ import annotations

import os
import time
from typing import Any

from openai import OpenAI

from backend.reguaz.services.generation.base_llm import BaseLLM
from backend.reguaz.utils.logger import get_logger


logger = get_logger(__name__, "llm_generation.log")


class NvidiaGptOssService(BaseLLM):
    """Remote GPT-OSS provider with bounded latency and no reasoning disclosure."""

    provider = "nvidia_gpt_oss"
    runtime = "nvidia_nim_openai_compatible"
    device = "remote:nvidia"

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
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        if not model_id.strip():
            raise ValueError("NVIDIA model ID must not be empty")
        if context_window < 1 or max_tokens < 1 or max_tokens >= context_window:
            raise ValueError("NVIDIA context/output token configuration is invalid")
        if temperature < 0 or not 0 <= top_p <= 1:
            raise ValueError("NVIDIA sampling configuration is invalid")
        if timeout_seconds <= 0:
            raise ValueError("NVIDIA timeout must be positive")
        resolved_key = api_key or os.getenv("NVIDIA_API_KEY")
        if client is None and not resolved_key:
            raise EnvironmentError("NVIDIA_API_KEY is not configured")

        self.model_id = model_id
        self.model_path = model_id
        self.base_url = base_url.rstrip("/")
        self.context_window = context_window
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.timeout_seconds = timeout_seconds
        self._client = client or OpenAI(
            base_url=self.base_url,
            api_key=resolved_key,
            timeout=timeout_seconds,
            max_retries=0,
        )
        logger.info(
            "NVIDIA GPT service ready model=%s timeout_s=%.1f",
            self.model_id,
            self.timeout_seconds,
        )

    def count_tokens(self, text: str) -> int:
        # NVIDIA exposes authoritative usage only after inference and no public
        # tokenize endpoint. UTF-8 byte count is a safe, offline upper bound.
        return len(text.encode("utf-8")) if text else 0

    def count_chat_tokens(self, user_content: str) -> int:
        if not user_content.strip():
            raise ValueError("chat user content must not be empty")
        # Harmony chat framing plus generation header. Conservative preflight;
        # authoritative prompt usage is taken from NVIDIA response metadata.
        return self.count_tokens(user_content) + 64

    def get_prompt_budget(self) -> int:
        return self.context_window - self.max_tokens - 256

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        return str(self._complete(prompt, self.max_tokens)["text"])

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
        return self._complete(user_content, output_tokens)

    def _complete(self, user_content: str, max_tokens: int) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            completion = self._client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": user_content}],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=max_tokens,
                stream=False,
            )
        except Exception as exc:
            logger.error(
                "NVIDIA GPT request failed error_category=%s", type(exc).__name__
            )
            raise RuntimeError("NVIDIA GPT request failed") from exc

        if not completion.choices:
            raise RuntimeError("NVIDIA GPT response has no choices")
        message = completion.choices[0].message
        content = (message.content or "").strip()
        if not content:
            raise RuntimeError("NVIDIA GPT response has no final content")
        usage = completion.usage
        prompt_tokens = (
            int(usage.prompt_tokens)
            if usage is not None
            else self.count_chat_tokens(user_content)
        )
        completion_tokens = int(usage.completion_tokens) if usage is not None else 0
        return {
            "text": content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "generation_ms": (time.perf_counter() - started) * 1000.0,
        }

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()
