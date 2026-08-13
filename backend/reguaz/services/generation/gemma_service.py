"""Gemma LLM implementation using llama.cpp."""

from __future__ import annotations

import time
import json
from pathlib import Path
import threading
from typing import Any

from llama_cpp import Llama, LlamaGrammar, llama_cpp
from llama_cpp.llama_chat_format import Jinja2ChatFormatter

from backend.reguaz import config
from backend.reguaz.services.generation.base_llm import BaseLLM
from backend.reguaz.utils.logger import get_logger

logger = get_logger(__name__, "llm_generation.log")


class GemmaService(BaseLLM):
    """
    LLM generation service using Gemma models via llama.cpp.

    This service is designed for Apple Silicon natively by using
    hardware acceleration (Metal) via llama-cpp-python.
    """

    provider = "gemma"
    runtime = "llama.cpp"

    def __init__(
        self,
        model_path: str | Path,
        context_window: int,
        temperature: float,
        max_tokens: int,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
        seed: int,
        gpu_layers: int = -1,
    ) -> None:
        """
        Initialize the Gemma service and load the model.

        Parameters
        ----------
        model_path : str | Path
            Path to the GGUF model file.
        context_window : int
            Maximum context length for generation.
        temperature : float
            Sampling temperature (0.0 to 1.0 is typical).
        max_tokens : int
            Maximum number of tokens to generate.
        top_p : float
            Nucleus sampling probability.
        top_k : int
            Top-K sampling value.
        repeat_penalty : float
            Penalty factor for repeated tokens.
        seed : int
            Random seed for reproducibility.

        Raises
        ------
        ValueError
            If validation of any argument fails.
        RuntimeError
            If model loading fails.
        """
        self.model_path = Path(model_path)
        self.context_window = context_window
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.repeat_penalty = repeat_penalty
        self.seed = seed
        self.gpu_layers = gpu_layers

        self._validate_config()
        self._llm: Llama | None = None
        self._chat_formatter: Jinja2ChatFormatter | None = None
        self._generation_lock = threading.Lock()
        self.device = "cpu"
        self._load_model()

    def _validate_config(self) -> None:
        """
        Validate all constructor parameters before model loading.

        Raises
        ------
        ValueError
            If any parameter is invalid.
        """
        if not self.model_path.exists() or not self.model_path.is_file():
            raise ValueError(
                f"Model path does not exist or is not a file: {self.model_path}"
            )
        if self.context_window < 1:
            raise ValueError("context_window must be at least 1.")
        if self.temperature < 0.0:
            raise ValueError("temperature cannot be negative.")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1.")
        if not (0.0 <= self.top_p <= 1.0):
            raise ValueError("top_p must be between 0.0 and 1.0.")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if self.repeat_penalty < 1.0:
            raise ValueError("repeat_penalty should be >= 1.0.")
        if self.gpu_layers < -1:
            raise ValueError("gpu_layers must be -1, 0, or a positive integer.")

    def _load_model(self) -> None:
        """
        Load the llama.cpp model into memory.

        Raises
        ------
        RuntimeError
            If the model fails to load.
        """
        config_str = (
            f"context_window={self.context_window}, temperature={self.temperature}, "
            f"max_tokens={self.max_tokens}, top_p={self.top_p}, top_k={self.top_k}, "
            f"repeat_penalty={self.repeat_penalty}, seed={self.seed}"
        )
        logger.info(
            f"Loading Gemma model from {self.model_path} with config: {config_str}"
        )
        start_time = time.perf_counter()

        try:
            used_gpu_layers = self.gpu_layers
            try:
                self._llm = self._create_llama(used_gpu_layers)
            except Exception as accelerated_error:
                if used_gpu_layers == 0:
                    raise
                logger.warning(
                    "Gemma accelerator initialization failed; retrying on CPU "
                    "error_category=%s",
                    type(accelerated_error).__name__,
                )
                self._llm = None
                used_gpu_layers = 0
                self._llm = self._create_llama(used_gpu_layers)
            self._configure_verified_chat_template()
            system_info = llama_cpp.llama_print_system_info().decode(
                "utf-8", errors="replace"
            )
            self.device = (
                "metal" if used_gpu_layers != 0 and "MTL" in system_info else "cpu"
            )
            elapsed = time.perf_counter() - start_time
            logger.info(
                "Gemma model loaded successfully in %.4f seconds; device=%s; chat_format=%s",
                elapsed,
                self.device,
                self._llm.chat_format,
            )
        except Exception as e:
            logger.error("Failed to load Gemma model", exc_info=True)
            raise RuntimeError(f"Gemma model loading failed: {e}") from e

    def _create_llama(self, gpu_layers: int) -> Llama:
        return Llama(
            model_path=str(self.model_path),
            n_ctx=self.context_window,
            n_gpu_layers=gpu_layers,
            seed=self.seed,
            verbose=False,
        )

    def _configure_verified_chat_template(self) -> None:
        if self._llm is None:
            raise RuntimeError("Gemma model is not loaded")
        metadata = self._llm.metadata
        template = metadata.get("tokenizer.chat_template")
        if not isinstance(template, str) or not template.strip():
            raise RuntimeError("GGUF tokenizer.chat_template metadata is required")
        bos_id = int(metadata.get("tokenizer.ggml.bos_token_id", 2))
        eos_id = int(metadata.get("tokenizer.ggml.eos_token_id", 1))
        bos_token = self._llm.detokenize([bos_id], special=True).decode(
            "utf-8", errors="strict"
        )
        eos_token = self._llm.detokenize([eos_id], special=True).decode(
            "utf-8", errors="strict"
        )
        if not bos_token or not eos_token:
            raise RuntimeError("GGUF chat template special tokens are invalid")
        self._chat_formatter = Jinja2ChatFormatter(
            template=template,
            bos_token=bos_token,
            eos_token=eos_token,
            add_generation_prompt=True,
            stop_token_ids=[eos_id],
        )
        self._llm.chat_handler = self._chat_formatter.to_chat_handler()

    def render_chat_prompt(self, user_content: str) -> str:
        if not user_content.strip():
            raise ValueError("chat user content must not be empty")
        if self._chat_formatter is None:
            raise RuntimeError("verified GGUF chat template is unavailable")
        formatted = self._chat_formatter(
            messages=[{"role": "user", "content": user_content}]
        )
        return formatted.prompt

    def count_chat_tokens(self, user_content: str) -> int:
        if self._llm is None:
            raise RuntimeError("Model is not loaded. Cannot count chat tokens.")
        prompt = self.render_chat_prompt(user_content)
        return len(
            self._llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
        )

    def generate_structured(
        self,
        user_content: str,
        schema: dict[str, Any],
        *,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        if self._llm is None:
            raise RuntimeError("Model is not loaded. Cannot generate.")
        prompt_tokens = self.count_chat_tokens(user_content)
        output_tokens = max_tokens or self.max_tokens
        if prompt_tokens + output_tokens > self.context_window:
            raise ValueError("rendered chat prompt exceeds configured context window")
        grammar = LlamaGrammar.from_json_schema(
            json.dumps(schema, ensure_ascii=False), verbose=False
        )
        started = time.perf_counter()
        try:
            with self._generation_lock:
                response = self._llm.create_chat_completion(
                    messages=[{"role": "user", "content": user_content}],
                    max_tokens=output_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    repeat_penalty=self.repeat_penalty,
                    seed=self.seed,
                    grammar=grammar,
                )
            text = self._extract_chat_text(response)
        except Exception as exc:
            logger.error("Structured generation failed", exc_info=True)
            raise RuntimeError("Structured model generation failed") from exc
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        return {
            "text": text,
            "prompt_tokens": int(usage.get("prompt_tokens", prompt_tokens)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "generation_ms": (time.perf_counter() - started) * 1000.0,
        }

    @staticmethod
    def _extract_chat_text(response: Any) -> str:
        if not isinstance(response, dict):
            raise ValueError("chat response must be a dictionary")
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("chat response has no choices")
        message = choices[0].get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ValueError("chat response has no message content")
        content = message["content"].strip()
        if not content:
            raise ValueError("chat response content is empty")
        return content

    def count_tokens(self, text: str) -> int:
        """
        Count tokens using llama.cpp tokenizer.
        """
        if self._llm is None or not text:
            return 0
        return len(self._llm.tokenize(text.encode("utf-8")))

    def get_prompt_budget(self) -> int:
        """
        Returns the token budget exclusively reserved for the context portion of the prompt.
        """
        return self.context_window - self.max_tokens - config.PROMPT_RESERVED_TOKENS

    def generate(self, prompt: str) -> str:
        """
        Generate text based on the provided prompt.

        Parameters
        ----------
        prompt : str
            The user prompt to send to the model.

        Returns
        -------
        str
            The text generated by the model.

        Raises
        ------
        ValueError
            If the prompt is empty.
        RuntimeError
            If generation fails or returns an invalid response format.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        if self._llm is None:
            raise RuntimeError("Model is not loaded. Cannot generate.")

        logger.info("Generation started")
        start_time = time.perf_counter()

        prompt_tokens = self.count_tokens(prompt)
        total_requested = prompt_tokens + self.max_tokens

        logger.info(f"Context window: {self.context_window}")
        logger.info(f"Prompt tokens: {prompt_tokens}")
        logger.info(f"Generation tokens: {self.max_tokens}")
        logger.info(f"Requested total: {total_requested}")

        if total_requested > self.context_window:
            excess = total_requested - self.context_window
            logger.warning(f"Prompt exceeds context window by {excess} tokens!")

        try:
            response = self._llm(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repeat_penalty=self.repeat_penalty,
                echo=False,
            )

            answer = self._extract_text(response)
            elapsed = time.perf_counter() - start_time
            logger.info(f"Generation completed in {elapsed:.4f} seconds.")
            return answer

        except ValueError as ve:
            logger.error(f"Validation error during generation: {ve}")
            raise RuntimeError(f"Invalid model response: {ve}") from ve
        except Exception as e:
            logger.error("Generation failed", exc_info=True)
            raise RuntimeError(f"Text generation failed: {e}") from e

    def _extract_text(self, response: Any) -> str:
        """
        Extract the generated text safely from the llama.cpp response dictionary.

        Parameters
        ----------
        response : Any
            The raw response returned by the llama.cpp call.

        Returns
        -------
        str
            The extracted generated text.

        Raises
        ------
        ValueError
            If the response format is unexpected or missing required keys.
        """
        if not isinstance(response, dict):
            raise ValueError(f"Expected dict response, got {type(response)}")

        choices = response.get("choices")
        if not choices or not isinstance(choices, list) or len(choices) == 0:
            raise ValueError("Response missing 'choices' or 'choices' is empty.")

        first_choice = choices[0]
        if not isinstance(first_choice, dict) or "text" not in first_choice:
            raise ValueError("First choice is missing the 'text' key.")

        text = first_choice["text"]
        if not isinstance(text, str):
            raise ValueError(f"Expected string for 'text', got {type(text)}")

        return text.strip()

    def close(self) -> None:
        """
        Release model resources safely.
        """
        if self._llm is not None:
            logger.info("Closing Gemma model and releasing resources.")
            try:
                if hasattr(self._llm, "close") and callable(self._llm.close):
                    self._llm.close()
            except Exception as e:
                logger.warning(f"Error while closing model: {e}")
            finally:
                self._llm = None
                self._chat_formatter = None
