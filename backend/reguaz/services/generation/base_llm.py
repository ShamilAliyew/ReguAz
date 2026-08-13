from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """
    Abstract base class for all LLM implementations.

    Every concrete LLM provider (Gemma, Gemini, OpenAI, Claude, etc.)
    must implement this interface so that the RAG pipeline remains
    independent of the underlying model.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a response from the language model.

        Parameters
        ----------
        prompt : str
            Fully constructed prompt to send to the model.

        Returns
        -------
        str
            Generated response.
        """
        raise NotImplementedError

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text using the model's tokenizer.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        int
            The token count.
        """
        raise NotImplementedError

    @abstractmethod
    def get_prompt_budget(self) -> int:
        """
        Returns the token budget exclusively reserved for the context portion of the prompt.
        
        Returns
        -------
        int
            The available token budget for context chunks.
        """
        raise NotImplementedError