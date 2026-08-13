"""
app/api/chat.py

API routes for the chat service.
Keep routers extremely thin, delegating all mapping and processing logic to ChatService.
"""

from __future__ import annotations

from typing import Any

# pyrefly: ignore [missing-import]
from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.core.dependencies import get_generation_pipeline
from backend.app.schemas.chat import ChatRequest, ChatResponse, LLMModelOption
from backend.app.services.chat_service import ChatService
from backend.reguaz.services.generation.v2_pipeline_registry import (
    ModelUnavailableError,
)

router = APIRouter()


@router.post(
    "/chat", response_model=ChatResponse, summary="Send a question to the RAG pipeline"
)
def chat(
    request: ChatRequest,
    pipeline: Any = Depends(get_generation_pipeline),
) -> ChatResponse:
    """
    Submit a question in Azerbaijani to the banking regulations assistant.

    The request is validated, passed to the ChatService for execution in the
    GenerationPipeline, and formatted to return inline citations and retrieval metrics.
    """
    service = ChatService(pipeline)
    try:
        return service.process_query(request)
    except ModelUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "model_unavailable", "message": str(exc)},
        ) from exc


@router.get(
    "/llm-models",
    response_model=list[LLMModelOption],
    summary="List selectable generation models",
)
def llm_models(
    pipeline: Any = Depends(get_generation_pipeline),
) -> list[LLMModelOption]:
    catalog = getattr(pipeline, "catalog", None)
    if not callable(catalog):
        return [
            LLMModelOption(
                id="gemma",
                label="Gemma (lokal)",
                available=True,
                loaded=True,
                provider="gemma",
                model_id="legacy-default",
                device="local",
                is_default=True,
            )
        ]
    return [LLMModelOption.model_validate(item) for item in catalog()]
