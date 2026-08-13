from __future__ import annotations

import re
from typing import Protocol


class TokenCounter(Protocol):
    tokenizer_id: str
    approximate: bool

    def count(self, text: str) -> int: ...


class BgeM3Tokenizer:
    tokenizer_id = "BAAI/bge-m3"
    approximate = False

    def __init__(self, *, local_files_only: bool = False) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "BAAI/bge-m3 tokenization requires transformers (normally installed "
                "through sentence-transformers). Install project dependencies or pass "
                "--allow-approx-tokenizer explicitly."
            ) from exc
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_id, local_files_only=local_files_only
            )
        except Exception as exc:
            raise RuntimeError(
                "BAAI/bge-m3 tokenizer is unavailable. Ensure it is cached or network "
                "access is available; alternatively pass --allow-approx-tokenizer."
            ) from exc

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=True))


class ApproximateTokenizer:
    """Explicit deterministic test/offline tokenizer; never an implicit fallback."""

    tokenizer_id = "reguaz/approx-wordpunct-v1"
    approximate = True
    _tokens = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def count(self, text: str) -> int:
        return len(self._tokens.findall(text))
