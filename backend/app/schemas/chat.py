"""
app/schemas/chat.py

Pydantic request and response models for:
  POST /api/v1/chat

Field mapping from GenerationPipeline output
---------------------------------------------
GenerationPipeline.generate() returns:
  {
    "question": str,
    "answer":   str,
    "sources":  list[dict],   # hydrated chunk metadata dicts from chunk_lookup
    "metrics":  {
        "retrieval_time":   float,
        "prompt_build_time": float,
        "generation_time":  float,
        "total_time":       float,
    }
  }

Each source dict contains the exact fields written by the chunker:
  chunk_id, document_id, title, category, chapter, article,
  section, subsection, page_start, page_end, source_file, text
"""

from __future__ import annotations

from enum import Enum

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MessageRole(str, Enum):
    """Valid roles in a conversation history turn."""

    user = "user"
    assistant = "assistant"


class StreamEventType(str, Enum):
    """Discriminator for Server-Sent Event payloads."""

    sources = "sources"
    token = "token"
    done = "done"
    error = "error"


LLMModelId = Literal[
    "gemma",
    "nvidia_gpt_oss",
    "groq_gpt_oss_20b",
    "groq_gpt_oss_120b",
]


# ---------------------------------------------------------------------------
# Nested models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """A single turn in a conversation history (for future multi-turn support)."""

    role: MessageRole
    content: str = Field(..., min_length=1, max_length=8000)


class SourceDocument(BaseModel):
    """
    A single retrieved regulatory chunk returned alongside the answer.

    ``citation`` matches the ``[N]`` markers the LLM embeds in the answer text.
    ``chunk_preview`` is a 300-character truncation of the full chunk text, used
    to render a readable snippet in the UI without sending the entire text in
    every chat response (the full text is loaded on demand via GET /chunks/{id}).
    """

    citation: int = Field(
        ..., ge=1, description="1-based index matching [N] in the answer"
    )
    chunk_id: str
    document_id: str | None = Field(None, description="Parent document identifier")
    document_name: str = Field(
        ..., description="Human-readable title of the source regulation"
    )
    category: str = Field(..., description="Regulation category slug, e.g. 'aml_kyc'")
    chapter: str | None = None
    article: str | None = None
    page: int | None = Field(
        None, description="Starting page of this chunk in the source PDF"
    )
    chunk_preview: str = Field(
        ...,
        description="First 300 characters of chunk text for inline display in the UI",
    )

    # Retrieval scoring (optional — populated once Phase 2 wires scores through)
    rerank_score: float | None = Field(
        None, description="Cross-Encoder relevance score"
    )
    rrf_score: float | None = Field(None, description="Reciprocal Rank Fusion score")
    semantic_rank: int | None = Field(None, description="Rank in dense semantic search")
    bm25_rank: int | None = Field(None, description="Rank in BM25 keyword search")
    canonical_locator: str | None = None
    role: Literal["seed", "expanded"] | None = None
    relation_type: str | None = None
    selectors: list[dict[str, Any]] = Field(default_factory=list)


class MetricsResponse(BaseModel):
    """
    Wall-clock timing diagnostics from the RAG pipeline.

    Mapped directly from the ``metrics`` dict in GenerationPipeline.generate():
      retrieval_time  ← metrics["retrieval_time"]
      generation_time ← metrics["generation_time"]
      total_time      ← metrics["total_time"]

    ``prompt_build_time`` is intentionally omitted from the public response
    (it is an internal implementation detail, not meaningful to the end user).
    """

    retrieval_time: float = Field(
        ..., description="Seconds spent in HybridQdrantRetriever"
    )
    generation_time: float = Field(..., description="Seconds spent in LLM generation")
    total_time: float = Field(..., description="Total end-to-end pipeline seconds")
    retrieval_ms: float | None = None
    relation_expansion_ms: float | None = None
    evidence_selection_ms: float | None = None
    context_budget_ms: float | None = None
    prompt_build_ms: float | None = None
    generation_ms: float | None = None
    provider_queue_ms: float | None = None
    provider_prompt_ms: float | None = None
    provider_completion_ms: float | None = None
    provider_total_ms: float | None = None
    structured_output_parsing_ms: float | None = None
    evidence_id_validation_ms: float | None = None
    citation_resolution_ms: float | None = None
    api_mapping_ms: float | None = None
    total_ms: float | None = None


class AnswerBlockResponse(BaseModel):
    text: str
    evidence_ids: list[str] = Field(default_factory=list)
    citation_numbers: list[int] = Field(default_factory=list)


class CitationResponse(BaseModel):
    citation: int = Field(ge=1)
    evidence_id: str
    role: Literal["seed", "expanded"]
    relation_type: str | None = None
    chunk_id: str | None = None
    logical_chunk_id: str | None = None
    parent_chunk_id: str | None = None
    document_id: str
    document_version_id: str
    document_title: str
    category: str = "unknown"
    canonical_locator: str
    hierarchy: dict[str, Any] = Field(default_factory=dict)
    page_start: int | None = None
    page_end: int | None = None
    exact_quotes: list[str] = Field(default_factory=list)
    selectors: list[dict[str, Any]] = Field(default_factory=list)
    content_sha256: str
    source_sha256: str
    source_validated: bool
    claim_support_status: Literal["not_evaluated"] = "not_evaluated"
    provenance: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Request body for POST /api/v1/chat."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Regulatory question. Azerbaijani is the primary language.",
        examples=["Bankın minimum nizamnamə kapitalı nə qədərdir?"],
    )
    session_id: str | None = Field(
        None,
        description="Optional client-supplied session identifier. Reserved for future session tracking.",
    )
    llm_model: LLMModelId | None = Field(
        None,
        description="Optional V2 generation model. Backend default is used when omitted.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "Bankın minimum nizamnamə kapitalı nə qədərdir?",
                    "session_id": None,
                    "llm_model": "gemma",
                }
            ]
        }
    }


class LLMModelOption(BaseModel):
    id: LLMModelId
    label: str
    available: bool
    reason: str | None = None
    loaded: bool
    provider: str
    model_id: str
    device: str
    is_default: bool


# ---------------------------------------------------------------------------
# Response — synchronous
# ---------------------------------------------------------------------------


class ChatResponse(BaseModel):
    """
    Response body for POST /api/v1/chat.

    ``answer`` contains inline citation markers ``[1]``, ``[2]`` that
    correspond 1-to-1 with entries in ``sources`` by ``citation`` index.
    The frontend replaces these markers with interactive citation badges.
    """

    session_id: str = Field(..., description="UUID v4 for this specific request")
    question: str = Field(..., description="The original question as submitted")
    answer: str = Field(
        ...,
        description=(
            "Model-generated answer grounded strictly in retrieved context. "
            "Contains [N] citation markers referencing sources[N-1]."
        ),
    )
    sources: list[SourceDocument] = Field(
        ...,
        description="Source chunks used to produce the answer, sorted by citation index",
    )
    metrics: MetricsResponse
    status: Literal["answered", "insufficient_evidence", "conflicting_evidence"] = (
        "answered"
    )
    answer_blocks: list[AnswerBlockResponse] = Field(default_factory=list)
    citations: list[CitationResponse] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    pipeline_version: Literal["v1", "v2"] = "v1"
    model: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Streaming event models (POST /api/v1/chat/stream — future)
# ---------------------------------------------------------------------------


class StreamSourcesEvent(BaseModel):
    """First SSE event: emitted after retrieval, before tokens start flowing."""

    type: StreamEventType = StreamEventType.sources
    session_id: str
    data: list[SourceDocument]


class StreamTokenEvent(BaseModel):
    """One SSE event per generated token."""

    type: StreamEventType = StreamEventType.token
    data: str


class StreamDoneEvent(BaseModel):
    """Final SSE event carrying pipeline metrics."""

    type: StreamEventType = StreamEventType.done
    data: MetricsResponse


class StreamErrorEvent(BaseModel):
    """Error SSE event sent if generation fails mid-stream."""

    type: StreamEventType = StreamEventType.error
    data: str
