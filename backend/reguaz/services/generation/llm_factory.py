"""Factory for instantiating LLM services."""

from __future__ import annotations

from typing import Any

from backend.reguaz import config
from backend.reguaz.services.generation.base_llm import BaseLLM
from backend.reguaz.services.generation.gemma_service import GemmaService
from backend.reguaz.services.generation.groq_service import GroqGptOssService
from backend.reguaz.services.generation.nvidia_gpt_service import NvidiaGptOssService
from backend.reguaz.utils.logger import get_logger

logger = get_logger(__name__, "llm_generation.log")


class LLMFactory:
    """
    Factory for creating LLM providers.

    Adheres to the Open/Closed Principle by allowing registration
    of new models without modifying the core creation logic.
    """

    _REGISTRY: dict[str, type[BaseLLM]] = {
        "gemma": GemmaService,
        "nvidia_gpt_oss": NvidiaGptOssService,
        "groq_gpt_oss_20b": GroqGptOssService,
        "groq_gpt_oss_120b": GroqGptOssService,
    }

    @classmethod
    def register(cls, name: str, service_class: type[BaseLLM]) -> None:
        """
        Register a new LLM service class.

        Parameters
        ----------
        name : str
            The identifier for the LLM type.
        service_class : type[BaseLLM]
            The class implementing the BaseLLM interface.

        Raises
        ------
        ValueError
            If the name is empty or service_class is not a subclass of BaseLLM.
        """
        if not name or not name.strip():
            raise ValueError("LLM provider name cannot be empty.")
        if not issubclass(service_class, BaseLLM):
            raise ValueError(f"{service_class.__name__} must inherit from BaseLLM.")

        cls._REGISTRY[name.lower()] = service_class
        logger.info(f"Registered new LLM provider: {name}")

    @classmethod
    def create(cls, model_type: str | None = None, **kwargs: Any) -> BaseLLM:
        """
        Create an instance of the requested LLM service.

        If `model_type` is not provided, defaults to the config setting.
        Any missing configuration parameters will be pulled from `config.py`.

        Parameters
        ----------
        model_type : str | None
            The identifier of the LLM to instantiate.
        **kwargs : Any
            Overrides for the model configuration parameters.

        Returns
        -------
        BaseLLM
            An instantiated object conforming to the BaseLLM interface.

        Raises
        ------
        ValueError
            If the requested model type is unknown.
        """
        model_type = model_type or config.DEFAULT_LLM_TYPE
        model_type = model_type.lower()

        if model_type not in cls._REGISTRY:
            supported = list(cls._REGISTRY.keys())
            logger.error(f"Unknown LLM type: '{model_type}'. Supported: {supported}")
            raise ValueError(
                f"Unsupported LLM type: '{model_type}'. Supported: {supported}"
            )

        service_class = cls._REGISTRY[model_type]

        params = cls._get_default_config(model_type)
        params.update(kwargs)

        logger.info(f"Instantiating LLM provider: {model_type}")
        return service_class(**params)

    @staticmethod
    def _get_default_config(model_type: str) -> dict[str, Any]:
        """
        Retrieve the default configuration for a given model type.

        Parameters
        ----------
        model_type : str
            The identifier of the LLM.

        Returns
        -------
        dict[str, Any]
            Dictionary containing the default parameters.
        """
        if model_type == "nvidia_gpt_oss":
            return {
                "model_id": config.NVIDIA_GPT_OSS_MODEL,
                "base_url": config.NVIDIA_API_BASE,
                "context_window": config.NVIDIA_GPT_OSS_CONTEXT_WINDOW,
                "temperature": 1.0,
                "max_tokens": config.NVIDIA_GPT_OSS_MAX_TOKENS,
                "top_p": 1.0,
                "timeout_seconds": config.NVIDIA_GPT_OSS_TIMEOUT_SECONDS,
            }

        if model_type in {"groq_gpt_oss_20b", "groq_gpt_oss_120b"}:
            model_id = (
                config.GROQ_GPT_OSS_20B_MODEL
                if model_type == "groq_gpt_oss_20b"
                else config.GROQ_GPT_OSS_120B_MODEL
            )
            return {
                "model_id": model_id,
                "base_url": config.GROQ_API_BASE,
                "context_window": config.GROQ_GPT_OSS_CONTEXT_WINDOW,
                "temperature": 0.2,
                "max_tokens": config.GROQ_GPT_OSS_MAX_TOKENS,
                "top_p": 0.9,
                "timeout_seconds": config.GROQ_GPT_OSS_TIMEOUT_SECONDS,
                "reasoning_effort": config.GROQ_GPT_OSS_REASONING_EFFORT,
            }

        base_config: dict[str, Any] = {
            "context_window": config.LLM_CONTEXT_WINDOW,
            "temperature": config.LLM_TEMPERATURE,
            "max_tokens": config.LLM_MAX_TOKENS,
            "top_p": config.LLM_TOP_P,
            "top_k": config.LLM_TOP_K,
            "repeat_penalty": config.LLM_REPEAT_PENALTY,
            "seed": config.LLM_SEED,
            "gpu_layers": config.LLM_GPU_LAYERS,
        }

        if model_type == "gemma":
            base_config["model_path"] = config.LLM_MODEL_PATH

        # Default configs for future models like gemini, openai, claude
        # can be appended here or moved to specialized config builders if needed.

        return base_config
