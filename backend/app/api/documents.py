"""
app/api/documents.py

API routes for the document and page viewer.
Keep routers extremely thin, delegating all logic to DocumentService.
"""

from __future__ import annotations

from typing import List

# pyrefly: ignore [missing-import]
from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.app.core.dependencies import get_document_service
from backend.app.schemas.document import (
    DocumentHighlightResponse,
    DocumentMetadataResponse,
    DocumentPageResponse,
)
from backend.app.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get(
    "",
    response_model=List[DocumentMetadataResponse],
    summary="List all indexed regulatory documents",
)
def get_all_documents(
    document_service: DocumentService = Depends(get_document_service),
) -> List[DocumentMetadataResponse]:
    """
    Retrieve all document metadata cards indexed in the system.
    """
    return document_service.get_all_documents()


@router.get(
    "/highlight",
    response_model=DocumentHighlightResponse,
    summary="Get navigate and highlight markers for a chunk",
)
def get_highlight(
    document_id: str = Query(..., description="ID of the parent document"),
    chunk_id: str = Query(..., description="ID of the chunk to highlight"),
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentHighlightResponse:
    """Resolve authoritative cleaned-document selectors for a chunk."""
    highlight = document_service.get_highlight(document_id, chunk_id)
    if not highlight:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chunk '{chunk_id}' for document '{document_id}' not found.",
        )
    return highlight


@router.get(
    "/{document_id}",
    response_model=DocumentMetadataResponse,
    summary="Get document metadata catalog card",
)
def get_document_metadata(
    document_id: str,
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentMetadataResponse:
    """
    Retrieve metadata card for a specific regulation by its document_id.
    """
    metadata = document_service.get_document_metadata(document_id)
    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )
    return metadata


@router.get(
    "/{document_id}/page/{page_number}",
    response_model=DocumentPageResponse,
    summary="Get document page text and sections",
)
def get_document_page(
    document_id: str,
    page_number: int,
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentPageResponse:
    """
    Retrieve text and regulatory section markings for a specific 1-based page number.
    """
    if page_number < 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Page number must be a positive integer >= 1.",
        )

    page = document_service.get_document_page(document_id, page_number)
    if not page:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Page {page_number} of document '{document_id}' not found or content is empty.",
        )
    return page
