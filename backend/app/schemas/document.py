"""
app/schemas/document.py

Pydantic schemas for the Document API endpoints.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentMetadataResponse(BaseModel):
    """
    Metadata information for a single document.
    """

    document_id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    category: str = Field(..., description="Category slug of the regulation")
    total_pages: Optional[int] = Field(None, description="Total pages in the PDF")
    total_chunks: Optional[int] = Field(
        None, description="Total database chunks generated"
    )
    language: Optional[str] = Field(None, description="Language code of the document")
    parser: Optional[str] = Field(None, description="Parser used to extract text")
    publication_date: Optional[str] = Field(
        None,
        description="Official publication date of the regulation (if available)",
    )
    status: str = Field(
        "active", description="Regulatory status of the document: 'active' | 'archived'"
    )
    related_articles: List[str] = Field(
        default_factory=list,
        description="Key articles or sections found inside this document",
    )
    document_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Generic unstructured metadata key-values",
    )
    document_version_id: Optional[str] = None
    source_representation: Optional[str] = None
    source_sha256: Optional[str] = None


class ArticleInfo(BaseModel):
    """
    Information about an article or section found on a document page.
    """

    chapter: Optional[str] = Field(None, description="Chapter heading")
    article: Optional[str] = Field(None, description="Article number or heading")
    section: Optional[str] = Field(None, description="Section heading")
    chunk_id: str = Field(..., description="Reference chunk ID covering this text")


class DocumentPageResponse(BaseModel):
    """
    Content and metadata for a single document page.
    """

    document_id: str = Field(..., description="ID of the document")
    page_number: int = Field(..., description="Page index (1-based)")
    page_content: str = Field(..., description="Text content extracted from the page")
    article_information: List[ArticleInfo] = Field(
        default_factory=list,
        description="List of regulatory articles found on this page",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional page-level metadata (e.g., rendering properties)",
    )
    document_version_id: Optional[str] = None
    representation_id: Optional[str] = None
    source_sha256: Optional[str] = None
    page_start_offset: Optional[int] = Field(default=None, ge=0)


class DocumentHighlightResponse(BaseModel):
    """
    Coordination details for navigating to and highlighting a chunk in the UI.
    """

    document_id: str = Field(..., description="Parent document identifier")
    page: int = Field(..., description="Page number of the chunk (1-based)")
    article: Optional[str] = Field(None, description="Article label covering the chunk")
    chunk_id: str = Field(..., description="The queried chunk ID")
    chunk_start: Optional[int] = Field(
        None,
        description="Exact character start offset of the chunk (None for future enhancement)",
    )
    chunk_end: Optional[int] = Field(
        None,
        description="Exact character end offset of the chunk (None for future enhancement)",
    )
    highlighted_text: str = Field(
        ..., description="Raw text of the chunk to be highlighted"
    )
    offset_status: str = Field(
        "future_enhancement",
        description="Detailed offset indexing status: 'supported' | 'future_enhancement'",
    )
    selectors: List[Dict[str, Any]] = Field(default_factory=list)
    warning: Optional[str] = None
