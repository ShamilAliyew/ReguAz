from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI

from backend.app.core.config import Settings, get_settings
from backend.reguaz import config as reguaz_config


logger = logging.getLogger(__name__)


@dataclass
class AppState:
    chunk_lookup: dict[str, dict[str, Any]] = field(default_factory=dict)
    retriever: Any | None = None
    llm: Any | None = None
    generation_pipeline: Any | None = None
    document_index: list[dict[str, Any]] = field(default_factory=list)
    document_service: Any | None = None
    artifact_store: Any | None = None
    speech_service: Any | None = None
    auth_database: Any | None = None
    pipeline_version: str = "v1"
    is_ready: bool = False
    startup_time_s: float | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    started = time.perf_counter()
    state = AppState(pipeline_version=settings.REGUAZ_PIPELINE_VERSION)
    logger.info(
        "ReguAZ startup pipeline=%s environment=%s",
        state.pipeline_version,
        settings.APP_ENV,
    )
    try:
        _load_auth(state, settings)
        if state.pipeline_version == "v2":
            _load_v2(state, settings)
        elif state.pipeline_version == "v1":
            _load_v1(state, settings)
        else:
            raise ValueError(f"unsupported pipeline version: {state.pipeline_version}")
        _load_speech(state, settings)
        state.is_ready = True
        state.startup_time_s = time.perf_counter() - started
        app.state.reguaz = state
        logger.info(
            "ReguAZ startup complete pipeline=%s startup_s=%.3f",
            state.pipeline_version,
            state.startup_time_s,
        )
        yield
    finally:
        if state.auth_database is not None:
            try:
                state.auth_database.close()
            except Exception:
                logger.warning("auth database shutdown failed", exc_info=True)
        if state.speech_service is not None:
            try:
                await state.speech_service.aclose()
            except Exception:
                logger.warning("speech service shutdown failed", exc_info=True)
        pipeline_closes_llms = False
        if state.generation_pipeline is not None and hasattr(
            state.generation_pipeline, "close"
        ):
            try:
                state.generation_pipeline.close()
                pipeline_closes_llms = bool(
                    getattr(state.generation_pipeline, "closes_llms", False)
                )
            except Exception:
                logger.warning("generation pipeline shutdown failed", exc_info=True)
        if state.retriever is not None and hasattr(state.retriever, "close"):
            try:
                state.retriever.close()
            except Exception:
                logger.warning("retriever shutdown failed", exc_info=True)
        if (
            not pipeline_closes_llms
            and state.llm is not None
            and hasattr(state.llm, "close")
        ):
            try:
                state.llm.close()
            except Exception:
                logger.warning("LLM shutdown failed", exc_info=True)
        state.is_ready = False


def _load_auth(state: AppState, settings: Settings) -> None:
    if not settings.AUTH_ENABLED:
        logger.info("authentication is disabled for this environment")
        return
    if settings.DATABASE_URL is None:
        raise RuntimeError("AUTH_ENABLED requires DATABASE_URL")

    from backend.app.auth.database import AuthDatabase

    database_url = settings.DATABASE_URL.get_secret_value().strip()
    if not database_url:
        raise RuntimeError("AUTH_ENABLED requires a non-empty DATABASE_URL")
    state.auth_database = AuthDatabase(database_url)
    state.auth_database.initialize()
    logger.info("authentication database initialized")


def _load_speech(state: AppState, settings: Settings) -> None:
    """Load optional TTS without making core RAG startup depend on its key."""

    api_key = (
        settings.OPENROUTER_API_KEY.get_secret_value().strip()
        if settings.OPENROUTER_API_KEY is not None
        else ""
    )
    if not settings.OPENROUTER_TTS_ENABLED or not api_key:
        logger.info("answer speech is disabled or OPENROUTER_API_KEY is not configured")
        return
    from backend.reguaz.services.speech.openrouter_tts import OpenRouterTTSService

    state.speech_service = OpenRouterTTSService(
        api_key=api_key,
        base_url=settings.OPENROUTER_API_BASE,
        model_id=settings.OPENROUTER_TTS_MODEL,
        voice=settings.OPENROUTER_TTS_VOICE,
        site_url=settings.OPENROUTER_SITE_URL,
        app_title=settings.OPENROUTER_APP_TITLE,
        timeout_seconds=settings.OPENROUTER_TTS_TIMEOUT_SECONDS,
        max_input_characters=settings.OPENROUTER_TTS_MAX_INPUT_CHARACTERS,
        max_audio_bytes=settings.OPENROUTER_TTS_MAX_AUDIO_BYTES,
        max_concurrency=settings.OPENROUTER_TTS_MAX_CONCURRENCY,
        max_retries=settings.OPENROUTER_TTS_MAX_RETRIES,
    )


def _load_v2(state: AppState, settings: Settings) -> None:
    from backend.app.services.document_service import V2DocumentService
    from backend.reguaz.retrieval.hybrid_v2 import HybridV2Retriever
    from backend.reguaz.retrieval.v2_artifacts import V2ArtifactStore
    from backend.reguaz.services.generation.v2_pipeline_registry import (
        V2GenerationPipelineRegistry,
    )

    v2_root = _repo_path(settings.V2_ROOT)
    qdrant_path = _repo_path(settings.V2_QDRANT_DIR)
    state.artifact_store = V2ArtifactStore(v2_root=v2_root)
    state.chunk_lookup = {
        str(item["chunk_id"]): item for item in state.artifact_store.iter_children()
    }
    state.document_service = V2DocumentService(state.artifact_store)
    state.document_index = list(state.artifact_store.iter_document_metadata())
    state.retriever = HybridV2Retriever(
        v2_root=v2_root,
        qdrant_path=qdrant_path,
        qdrant_url=settings.QDRANT_URL,
        qdrant_api_key=(
            settings.QDRANT_API_KEY.get_secret_value()
            if settings.QDRANT_API_KEY is not None
            else None
        ),
        qdrant_timeout_seconds=settings.QDRANT_TIMEOUT_SECONDS,
        alias=settings.V2_QDRANT_ALIAS,
        local_files_only=settings.V2_LOCAL_FILES_ONLY,
        artifact_store=state.artifact_store,
    )
    state.generation_pipeline = V2GenerationPipelineRegistry(
        retriever=state.retriever,
        artifacts=state.artifact_store,
        default_model=settings.LLM_TYPE,
        generation_settings={
            "expansion_per_seed_cap": settings.V2_EXPANSION_PER_SEED_CAP,
            "expansion_global_cap": settings.V2_EXPANSION_GLOBAL_CAP,
            "reference_confidence_min": settings.V2_REFERENCE_CONFIDENCE_MIN,
            "parent_excerpt_char_limit": settings.V2_PARENT_EXCERPT_CHAR_LIMIT,
            "context_safety_margin_tokens": settings.V2_CONTEXT_SAFETY_MARGIN_TOKENS,
        },
    )
    # V2 providers are loaded lazily on first selection. This keeps startup
    # usable when one optional remote/local provider is unavailable.
    state.llm = None


def _load_v1(state: AppState, settings: Settings) -> None:
    from backend.app.services.document_service import DocumentService
    from backend.reguaz.retrieval.hybrid_qdrant import HybridQdrantRetriever
    from backend.reguaz.services.chunks.chunk_reader import ChunkReader
    from backend.reguaz.services.generation.generation_pipeline import (
        GenerationPipeline,
    )
    from backend.reguaz.services.generation.llm_factory import LLMFactory

    chunks_path = _repo_path(settings.CHUNKS_DIR)
    metadata_path = _repo_path(settings.METADATA_DIR)
    state.chunk_lookup = ChunkReader().build_lookup(chunks_path)
    state.document_service = DocumentService(
        metadata_dir=metadata_path,
        cleaned_docs_dir=metadata_path.parent / "cleaned_documents",
        chunk_lookup=state.chunk_lookup,
    )
    state.document_index = list(state.document_service._doc_metadata_map.values())
    qdrant_path = _repo_path(settings.QDRANT_DIR)
    if not qdrant_path.exists():
        qdrant_path = reguaz_config.QDRANT_PATH
    state.retriever = HybridQdrantRetriever(
        model_name=settings.EMBEDDING_MODEL,
        qdrant_dir=str(qdrant_path),
        chunks_dir=str(chunks_path),
        top_k_semantic=settings.TOP_K_SEMANTIC,
        top_k_bm25=settings.TOP_K_BM25,
        rerank_top_k=settings.RERANK_TOP_K,
        final_top_k=settings.DEFAULT_TOP_K,
        rrf_k=settings.RRF_K,
        reranker_model=settings.RERANKER_MODEL,
    )
    state.llm = LLMFactory.create(model_type=settings.LLM_TYPE)
    state.generation_pipeline = GenerationPipeline(
        retriever=state.retriever,
        chunk_lookup=state.chunk_lookup,
        llm=state.llm,
        top_k=settings.DEFAULT_TOP_K,
    )


def _repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return reguaz_config.BACKEND_ROOT.parent / path
