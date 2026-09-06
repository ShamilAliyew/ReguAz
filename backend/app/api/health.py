"""
app/api/health.py

API routes for application health checks.
"""

from __future__ import annotations

# pyrefly: ignore [missing-import]
from fastapi import APIRouter, Depends

from backend.app.core.config import get_settings
from backend.app.core.dependencies import get_app_state
from backend.app.core.lifespan import AppState
from backend.app.schemas.health import HealthResponse

router = APIRouter(tags=["status"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Get application health check status",
)
def get_health(
    state: AppState = Depends(get_app_state),
) -> HealthResponse:
    """
    Returns the loading state of all model files and system databases.
    Used by load balancers and deployment probes to verify service readiness.
    """
    settings = get_settings()

    pipeline_loaded = state.generation_pipeline is not None
    document_service_loaded = state.document_service is not None
    catalog = getattr(state.generation_pipeline, "catalog", None)
    model_catalog = catalog() if callable(catalog) else []
    llm_loaded = state.llm is not None or any(
        bool(item.get("loaded")) for item in model_catalog
    )
    model_available = llm_loaded or any(
        bool(item.get("available")) for item in model_catalog
    )
    loaded_devices = [
        str(item.get("device", "unknown"))
        for item in model_catalog
        if item.get("loaded")
    ]

    # Check if all crucial components are loaded
    all_loaded = pipeline_loaded and document_service_loaded and model_available
    status_str = "healthy" if all_loaded else "unhealthy"

    return HealthResponse(
        status=status_str,
        version=settings.APP_VERSION,
        pipeline_loaded=pipeline_loaded,
        document_service_loaded=document_service_loaded,
        llm_loaded=llm_loaded,
        chunk_lookup_size=len(state.chunk_lookup) if state.chunk_lookup else 0,
        document_index_size=len(state.document_index) if state.document_index else 0,
        pipeline_version=state.pipeline_version,
        generation_device=(
            str(getattr(state.llm, "device", "unknown"))
            if state.llm is not None
            else ",".join(loaded_devices) or "lazy"
        ),
        auth_enabled=settings.AUTH_ENABLED,
        auth_database_ready=state.auth_database is not None,
        qdrant_mode=str(getattr(state.retriever, "qdrant_mode", "unknown")),
    )
