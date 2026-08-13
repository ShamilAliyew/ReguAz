"""
app/core/dependencies.py

FastAPI dependency injection functions.

Every shared resource (GenerationPipeline, chunk_lookup, retriever) is accessed
through these functions.  Route handlers declare them via ``Depends()``; FastAPI
resolves them per-request by reading from ``app.state.reguaz``.

Design rules
------------
- Dependencies NEVER construct singletons.  Construction happens exclusively
  inside the lifespan context manager in ``app/core/lifespan.py``.
- If a singleton is not yet loaded (e.g., during the Phase 1 skeleton phase),
  the dependency raises HTTP 503 with a clear message.
- Type annotations use ``Any`` as a placeholder until Phase 2 wires the real
  types.  They will be tightened once the concrete classes are imported.
"""
from __future__ import annotations

from typing import Any

# pyrefly: ignore [missing-import]
from fastapi import Depends, HTTPException, Request, status

# pyrefly: ignore [missing-import]
from backend.app.core.lifespan import AppState


# ---------------------------------------------------------------------------
# Internal: raw state access
# ---------------------------------------------------------------------------

def _get_state(request: Request) -> AppState:
    """
    Return the ``AppState`` object attached to the running application.

    This is the single point of contact between HTTP requests and the
    application-level singletons.  All other dependency functions call this.
    """
    return request.app.state.reguaz  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Public dependencies
# ---------------------------------------------------------------------------

def get_generation_pipeline(state: AppState = Depends(_get_state)) -> Any:
    """
    Return the initialised ``GenerationPipeline`` singleton.

    Raises
    ------
    HTTPException(503)
        If the pipeline has not been loaded yet (startup still in progress,
        or Phase 2 not yet implemented).
    """
    if state.generation_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "service_unavailable",
                "message": "GenerationPipeline is not yet initialised. Service is starting up.",
            },
        )
    return state.generation_pipeline


def get_chunk_lookup(state: AppState = Depends(_get_state)) -> dict[str, dict[str, Any]]:
    """
    Return the in-memory ``chunk_id → chunk_metadata`` lookup dict.

    Raises
    ------
    HTTPException(503)
        If the chunk lookup has not been built yet.
    """
    if not state.chunk_lookup:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "service_unavailable",
                "message": "Chunk lookup is not yet loaded. Service is starting up.",
            },
        )
    return state.chunk_lookup


def get_retriever(state: AppState = Depends(_get_state)) -> Any:
    """
    Return the initialised ``HybridQdrantRetriever`` singleton.

    Raises
    ------
    HTTPException(503)
        If the retriever has not been loaded yet.
    """
    if state.retriever is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "service_unavailable",
                "message": "Retriever is not yet initialised. Service is starting up.",
            },
        )
    return state.retriever


def get_document_index(state: AppState = Depends(_get_state)) -> list[dict[str, Any]]:
    """
    Return the pre-built document metadata index list.

    Raises
    ------
    HTTPException(503)
        If the document index has not been built yet.
    """
    if not state.document_index:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "service_unavailable",
                "message": "Document index is not yet loaded. Service is starting up.",
            },
        )
    return state.document_index


def get_app_state(state: AppState = Depends(_get_state)) -> AppState:
    """
    Return the raw ``AppState`` for endpoints that need multiple fields
    (e.g., the health check endpoint).
    """
    return state


def get_document_service(state: AppState = Depends(_get_state)) -> Any:
    """
    Return the initialised ``DocumentService`` singleton.

    Raises
    ------
    HTTPException(503)
        If the document service has not been loaded yet.
    """
    if state.document_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "service_unavailable",
                "message": "Document service is not yet initialised. Service is starting up.",
            },
        )
    return state.document_service
