"""
app/schemas/health.py

Pydantic schemas for the Health API endpoints.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """
    Health check response payload representing application availability.
    """
    status: str = Field(..., description="System status indicator: 'healthy' | 'unhealthy'")
    version: str = Field(..., description="API system version")
    pipeline_loaded: bool = Field(
        ...,
        description="Whether the core RAG GenerationPipeline is successfully loaded",
    )
    document_service_loaded: bool = Field(
        ...,
        description="Whether the DocumentService catalog manager is successfully initialized",
    )
    llm_loaded: bool = Field(
        ...,
        description="Whether the Gemma LLM model instance is loaded and initialized in memory",
    )
    chunk_lookup_size: int = Field(
        0,
        description="Total database chunks cached in the memory lookup",
    )
    document_index_size: int = Field(
        0,
        description="Total regulatory documents currently cataloged",
    )
    pipeline_version: str = "v1"
    generation_device: str = "unknown"
    auth_enabled: bool = False
    auth_database_ready: bool = False
    qdrant_mode: str = "unknown"
