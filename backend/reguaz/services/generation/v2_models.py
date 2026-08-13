from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


EvidenceRole = Literal["seed", "expanded"]
GenerationStatus = Literal["answered", "insufficient_evidence", "conflicting_evidence"]


class EvidencePath(StrictModel):
    seed_chunk_id: str
    relation_type: str | None = None
    relation_direction: Literal["outbound", "inbound", "structural"] | None = None
    expansion_depth: int = Field(default=0, ge=0, le=1)
    resolution_confidence: float | None = Field(default=None, ge=0, le=1)
    selection_reason: str


class PositionSelector(StrictModel):
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def ordered(self) -> "PositionSelector":
        if self.end < self.start:
            raise ValueError("selector end precedes start")
        return self


class QuoteSelector(StrictModel):
    exact: str = Field(min_length=1)
    prefix: str = ""
    suffix: str = ""


class SourceSelector(StrictModel):
    representation_id: str
    representation_type: Literal["cleaned_markdown"] = "cleaned_markdown"
    document_version_id: str
    source_sha256: str
    page: int | None = Field(default=None, ge=1)
    position: PositionSelector
    page_position: PositionSelector | None = None
    quote: QuoteSelector
    normalization_version: Literal["cleaned_markdown_utf8_codepoint_v1"] = (
        "cleaned_markdown_utf8_codepoint_v1"
    )


class EvidenceItem(StrictModel):
    evidence_id: str = ""
    role: EvidenceRole
    relation_type: str | None = None
    seed_chunk_id: str
    artifact_type: Literal["child", "parent"]
    chunk_id: str | None = None
    logical_chunk_id: str | None = None
    parent_chunk_id: str | None = None
    document_id: str
    document_version_id: str
    document_title: str
    canonical_locator: str
    hierarchy: dict[str, Any] = Field(default_factory=dict)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    content: str = Field(min_length=1)
    content_sha256: str
    evidence_text_sha256: str
    source_representation: Literal["cleaned_markdown"] = "cleaned_markdown"
    source_sha256: str
    source_spans: list[dict[str, Any]] = Field(default_factory=list)
    selectors: list[SourceSelector] = Field(default_factory=list)
    selection_reason: str
    priority: int = Field(ge=1)
    final_rank: int | None = Field(default=None, ge=1)
    reranker_score: float | None = None
    quality_flags: list[str] = Field(default_factory=list)
    provenance: list[EvidencePath] = Field(min_length=1)


class ExpansionDiagnostic(StrictModel):
    seed_chunk_id: str
    relation_type: str
    target_id: str | None = None
    decision: Literal["included", "excluded"]
    reason: str


class AnswerBlock(StrictModel):
    text: str = Field(min_length=1, max_length=2000)
    evidence_ids: list[str] = Field(min_length=1, max_length=8)


class StructuredGeneration(StrictModel):
    status: GenerationStatus
    answer_blocks: list[AnswerBlock] = Field(default_factory=list, max_length=12)
    limitations: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def status_consistency(self) -> "StructuredGeneration":
        if self.status == "answered" and not self.answer_blocks:
            raise ValueError("answered status requires answer blocks")
        if self.status != "answered" and self.answer_blocks:
            raise ValueError("non-answered status cannot contain answer blocks")
        return self


STRUCTURED_GENERATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["answered", "insufficient_evidence", "conflicting_evidence"],
        },
        "answer_blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["text", "evidence_ids"],
                "additionalProperties": False,
            },
        },
        "limitations": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["status", "answer_blocks", "limitations"],
    "additionalProperties": False,
}


class V2GenerationSettings(StrictModel):
    expansion_max_depth: Literal[1] = 1
    expansion_per_seed_cap: int = Field(default=4, ge=0, le=20)
    expansion_global_cap: int = Field(default=16, ge=0, le=100)
    reference_confidence_min: float = Field(default=0.8, ge=0, le=1)
    parent_excerpt_char_limit: int = Field(default=700, ge=100, le=4000)
    context_safety_margin_tokens: int = Field(default=256, ge=32, le=2048)
    reserved_output_tokens: int = Field(default=512, ge=64, le=4096)
    max_generation_retries: int = Field(default=1, ge=0, le=1)
