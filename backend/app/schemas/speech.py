"""Pydantic contracts for optional answer speech synthesis."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SpeechRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=8_000,
        description="Final assistant answer to synthesize. Markdown is normalized server-side.",
    )


class SpeechStatusResponse(BaseModel):
    available: bool
    provider: str = "openrouter"
    model_id: str
    output_format: str = "mp3"
    reason: str | None = None
