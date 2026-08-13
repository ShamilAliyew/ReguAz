"""Lazy, process-scoped V2 generation pipelines selected per request."""

from __future__ import annotations

import os
import threading
from typing import Any

from backend.reguaz import config
from backend.reguaz.services.generation.generation_pipeline_v2 import (
    V2GenerationPipeline,
)
from backend.reguaz.services.generation.llm_factory import LLMFactory
from backend.reguaz.services.generation.v2_models import V2GenerationSettings


MODEL_LABELS = {
    "gemma": "Gemma 3 4B (lokal)",
    "groq_gpt_oss_20b": "Groq GPT-OSS 20B (sürətli)",
    "groq_gpt_oss_120b": "Groq GPT-OSS 120B",
    "nvidia_gpt_oss": "NVIDIA GPT-OSS 120B",
}


class ModelUnavailableError(RuntimeError):
    """Requested model is known but unavailable in current process config."""


class V2GenerationPipelineRegistry:
    """Create each selected LLM/pipeline once and reuse it across requests."""

    closes_llms = True

    def __init__(
        self,
        *,
        retriever: Any,
        artifacts: Any,
        default_model: str,
        generation_settings: dict[str, Any],
    ) -> None:
        if default_model not in MODEL_LABELS:
            raise ValueError(f"Unsupported default LLM model: {default_model}")
        self.retriever = retriever
        self.artifacts = artifacts
        self.default_model = default_model
        self.generation_settings = dict(generation_settings)
        self._pipelines: dict[str, V2GenerationPipeline] = {}
        self._llms: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._closed = False

    @property
    def default_llm(self) -> Any | None:
        return self._llms.get(self.default_model)

    def generate(self, question: str) -> dict[str, Any]:
        return self.generate_for_model(question, self.default_model)

    def generate_for_model(self, question: str, model_type: str) -> dict[str, Any]:
        return self.get_pipeline(model_type).generate(question)

    def get_pipeline(self, model_type: str) -> V2GenerationPipeline:
        if model_type not in MODEL_LABELS:
            raise ValueError(f"Unsupported LLM model: {model_type}")
        with self._lock:
            if self._closed:
                raise RuntimeError("generation pipeline registry is closed")
            existing = self._pipelines.get(model_type)
            if existing is not None:
                return existing
            available, reason = self._availability(model_type)
            if not available:
                raise ModelUnavailableError(reason or "Model unavailable")
            try:
                llm = LLMFactory.create(model_type=model_type)
            except Exception as exc:
                raise ModelUnavailableError(
                    f"{MODEL_LABELS[model_type]} başladılmadı ({type(exc).__name__})"
                ) from exc
            settings = V2GenerationSettings(
                **self.generation_settings,
                reserved_output_tokens=llm.max_tokens,
            )
            pipeline = V2GenerationPipeline(
                retriever=self.retriever,
                artifacts=self.artifacts,
                llm=llm,
                settings=settings,
            )
            self._llms[model_type] = llm
            self._pipelines[model_type] = pipeline
            return pipeline

    def catalog(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for model_type, label in MODEL_LABELS.items():
            available, reason = self._availability(model_type)
            llm = self._llms.get(model_type)
            items.append(
                {
                    "id": model_type,
                    "label": label,
                    "available": available,
                    "reason": reason,
                    "loaded": llm is not None,
                    "provider": getattr(llm, "provider", _provider(model_type)),
                    "model_id": getattr(llm, "model_id", _model_id(model_type)),
                    "device": getattr(llm, "device", _device(model_type)),
                    "is_default": model_type == self.default_model,
                }
            )
        return items

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            llms = list(self._llms.values())
            self._llms.clear()
            self._pipelines.clear()
        for llm in llms:
            close = getattr(llm, "close", None)
            if callable(close):
                close()

    @staticmethod
    def _availability(model_type: str) -> tuple[bool, str | None]:
        if model_type == "gemma":
            if config.LLM_MODEL_PATH.exists():
                return True, None
            return False, "Lokal Gemma model faylı tapılmadı"
        if model_type.startswith("groq_"):
            if os.getenv("GROQ_API_KEY"):
                return True, None
            return False, "GROQ_API_KEY konfiqurasiya edilməyib"
        if model_type == "nvidia_gpt_oss":
            if os.getenv("NVIDIA_API_KEY"):
                return True, None
            return False, "NVIDIA_API_KEY konfiqurasiya edilməyib"
        return False, "Dəstəklənməyən model"


def _provider(model_type: str) -> str:
    if model_type.startswith("groq_"):
        return "groq"
    if model_type == "nvidia_gpt_oss":
        return "nvidia_gpt_oss"
    return "gemma"


def _model_id(model_type: str) -> str:
    return {
        "gemma": config.LLM_MODEL_PATH.name,
        "groq_gpt_oss_20b": config.GROQ_GPT_OSS_20B_MODEL,
        "groq_gpt_oss_120b": config.GROQ_GPT_OSS_120B_MODEL,
        "nvidia_gpt_oss": config.NVIDIA_GPT_OSS_MODEL,
    }[model_type]


def _device(model_type: str) -> str:
    if model_type.startswith("groq_"):
        return "remote:groq"
    if model_type == "nvidia_gpt_oss":
        return "remote:nvidia"
    return "local"
