from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Quality(StrictModel):
    parser_confidence: float = Field(ge=0, le=1)
    quality_flags: list[str] = Field(default_factory=list)
    normalizations: list[str] = Field(default_factory=list)


class SourceSpan(StrictModel):
    page: int | None = Field(default=None, ge=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    bbox: None = None
    extraction_method: Literal["existing_cleaned_markdown"] = "existing_cleaned_markdown"

    @model_validator(mode="after")
    def ordered(self) -> "SourceSpan":
        if self.char_end < self.char_start or self.line_end < self.line_start:
            raise ValueError("source span ends before it starts")
        return self


class HierarchyLabel(StrictModel):
    number: str | None = None
    title: str | None = None


class Hierarchy(StrictModel):
    division: HierarchyLabel | None = None
    chapter: HierarchyLabel | None = None
    article: HierarchyLabel | None = None
    section: HierarchyLabel | None = None
    clause: str | None = None
    subclause: str | None = None
    item: str | None = None
    annex: HierarchyLabel | None = None
    heading_path: list[str] = Field(default_factory=list)


class AtomicUnit(StrictModel):
    logical_unit_id: str
    canonical_locator: str
    source_spans: list[SourceSpan]


class Reference(StrictModel):
    raw_reference: str
    target_locator: str
    target_logical_chunk_id: str | None = None
    target_chunk_id: str | None = None
    resolved: bool
    confidence: float | None = Field(default=None, ge=0, le=1)


class Relations(StrictModel):
    parent: str
    previous_sibling: str | None = None
    next_sibling: str | None = None
    children: list[str] = Field(default_factory=list)
    references: list[Reference] = Field(default_factory=list)
    referenced_by: list[Reference] = Field(default_factory=list)
    annex_parent: str | None = None
    table_parent: str | None = None
    approved_document: str | None = None


class TableMetadata(StrictModel):
    table_id: str
    title: str | None = None
    column_headers: list[str]
    row_start: int = Field(ge=1)
    row_end: int = Field(ge=1)
    total_rows: int = Field(ge=1)


class DocumentMetadata(StrictModel):
    schema_version: Literal["2.0"] = "2.0"
    document_id: str
    document_version_id: str
    title: str
    short_title: str
    document_type: str
    authority: str | None = None
    act_number: str | None = None
    language: Literal["az"] = "az"
    category: str
    source_file: str
    source_relative_path: str
    source_url: str | None = None
    source_sha256: str
    publication_date: str | None = None
    effective_from: str | None = None
    effective_to: str | None = None
    status: Literal["unknown", "active", "repealed"] = "unknown"
    retrieved_at: str | None = None
    parser_name: Literal["reguaz-structure-parser"] = "reguaz-structure-parser"
    parser_version: Literal["2.0.0"] = "2.0.0"
    chunker_name: Literal["adaptive-structure-aware-hierarchical"] = (
        "adaptive-structure-aware-hierarchical"
    )
    chunker_version: Literal["2.0.0"] = "2.0.0"
    duplicate_of_document_id: str | None = None
    is_canonical_document: bool = True


class ChildChunk(StrictModel):
    schema_version: Literal["2.0"] = "2.0"
    chunk_id: str
    logical_chunk_id: str
    document_id: str
    document_version_id: str
    parent_chunk_id: str
    article_root_id: str
    chunk_type: str
    node_type: str
    language: Literal["az"] = "az"
    document_title: str
    short_title: str
    source_url: str | None = None
    source_file: str
    hierarchy: Hierarchy
    canonical_locator: str
    atomic_unit_ids: list[str]
    atomic_units: list[AtomicUnit]
    content: str = Field(min_length=1)
    embedding_text: str = Field(min_length=1)
    parent_lead_in: str | None = None
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    source_spans: list[SourceSpan]
    relations: Relations
    split_index: int = Field(ge=1)
    split_count: int = Field(ge=1)
    is_continuation: bool
    continuation_of: str | None = None
    embedding_token_count: int = Field(ge=1)
    tokenizer_id: str
    character_count: int = Field(ge=1)
    sentence_count: int = Field(ge=1)
    chunking_strategy: Literal["adaptive_structure_aware_hierarchical"] = (
        "adaptive_structure_aware_hierarchical"
    )
    packing_reason: str
    content_sha256: str
    effective_from: str | None = None
    effective_to: str | None = None
    status: Literal["unknown", "active", "repealed"] = "unknown"
    is_current: bool = True
    quality: Quality
    table: TableMetadata | None = None

    @model_validator(mode="after")
    def continuation_consistency(self) -> "ChildChunk":
        if self.is_continuation != (self.split_index > 1):
            raise ValueError("is_continuation conflicts with split_index")
        if self.is_continuation and not self.continuation_of:
            raise ValueError("continuation_of is required for continuations")
        return self


class ParentChunk(StrictModel):
    schema_version: Literal["2.0"] = "2.0"
    parent_chunk_id: str
    article_root_id: str
    document_id: str
    document_version_id: str
    document_title: str
    parent_type: str
    hierarchy: Hierarchy
    canonical_locator: str
    title: str | None = None
    content: str = Field(min_length=1)
    parent_lead_in: str | None = None
    child_chunk_ids: list[str] = Field(min_length=1)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    source_spans: list[SourceSpan]
    token_count: int = Field(ge=1)
    content_sha256: str
    quality: Quality


class QualityReport(StrictModel):
    document_id: str
    status: Literal["success", "failed"]
    profile: str
    profiles_detected: list[str]
    parent_count: int = Field(ge=0)
    child_count: int = Field(ge=0)
    atomic_unit_count: int = Field(ge=0)
    table_count: int = Field(ge=0)
    unclassified_node_count: int = Field(ge=0)
    unresolved_reference_count: int = Field(ge=0)
    duplicate_chunk_count: int = Field(ge=0)
    min_tokens: int = Field(ge=0)
    median_tokens: int = Field(ge=0)
    p95_tokens: int = Field(ge=0)
    max_tokens: int = Field(ge=0)
    quality_flags: list[str]
    errors: list[str]


class Manifest(StrictModel):
    schema_version: Literal["2.0"] = "2.0"
    corpus_version: str
    chunker_name: str
    chunker_version: str
    parser_name: str
    parser_version: str
    tokenizer_id: str
    approximate_tokenizer: bool
    chunk_size_configuration: dict[str, int]
    input_root: str
    output_root: str
    discovered_document_count: int
    successful_document_count: int
    failed_document_count: int
    duplicate_document_count: int
    parent_count: int
    child_count: int
    chunk_type_distribution: dict[str, int]
    document_profile_distribution: dict[str, int]
    token_distribution: dict[str, int]
    unresolved_reference_count: int
    unclassified_structure_count: int
    quality_flag_distribution: dict[str, int]
    source_manifest: list[dict[str, str]]
    failures: list[dict[str, str]]
    build_timestamp: datetime
