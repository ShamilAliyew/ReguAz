"""Deterministic preparation of Azerbaijani answers for speech synthesis."""

from __future__ import annotations

import re
import unicodedata

from backend.reguaz.services.speech.azerbaijani_numbers import verbalize_numbers


_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_MARKDOWN_IMAGE_RE = re.compile(r"!\[([^]]*)]\([^)]+\)")
_MARKDOWN_LINK_RE = re.compile(r"\[([^]]+)]\([^)]+\)")
_CITATION_RE = re.compile(
    r"(?:\s*\[\s*\d+(?:\s*[,;]\s*\d+)*\s*\])+"
)
_HEADING_OR_LIST_RE = re.compile(r"(?m)^\s*(?:#{1,6}|[-*+] |\d+[.)] )\s*")
_SPACE_BEFORE_PUNCTUATION_RE = re.compile(r"\s+([,.;:!?])")
_WHITESPACE_RE = re.compile(r"\s+")

_ABBREVIATIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bAMB\b", re.IGNORECASE), "Azərbaycan Mərkəzi Bankı"),
    (re.compile(r"\bAZN\b", re.IGNORECASE), "manat"),
)


def prepare_speech_text(text: str) -> str:
    """Convert rendered Markdown into conservative, speakable Azerbaijani text.

    The transformation removes presentation-only syntax while preserving the
    legal meaning and verbalizing numeric expressions in Azerbaijani. Visual
    citation markers are deliberately
    removed because source metadata remains available in the UI and should not
    interrupt spoken answers. Dates, legal locators, ordinals, decimals,
    ratios, grouped amounts and large integers use separate deterministic
    rules so the TTS provider never has to infer their language.
    """

    if not isinstance(text, str) or not text.strip():
        raise ValueError("speech text must not be empty")

    value = unicodedata.normalize("NFC", text)
    value = _FENCED_CODE_RE.sub(" ", value)
    value = _MARKDOWN_IMAGE_RE.sub(r" \1 ", value)
    value = _MARKDOWN_LINK_RE.sub(r" \1 ", value)
    value = _CITATION_RE.sub(" ", value)
    value = _HEADING_OR_LIST_RE.sub("", value)
    value = value.replace("`", " ").replace("**", "").replace("__", "")
    value = value.replace("№", " nömrə ").replace("%", " faiz ")
    value = value.replace("−", "-")

    for pattern, replacement in _ABBREVIATIONS:
        value = pattern.sub(replacement, value)

    value = verbalize_numbers(value)
    value = _SPACE_BEFORE_PUNCTUATION_RE.sub(r"\1", value)
    value = _WHITESPACE_RE.sub(" ", value).strip()
    if not value:
        raise ValueError("speech text is empty after normalization")
    return value
