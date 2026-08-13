"""
app/core/config.py

Centralised application settings using pydantic-settings.

All values are read from environment variables or a .env file located at the
project root.  Calling get_settings() returns a cached singleton so the .env
file is only parsed once per process.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """
    Application-wide configuration.

    Fields are populated (in order of priority) from:
      1. Environment variables (highest priority)
      2. .env file at the project root
      3. Default values defined here (lowest priority)
    """

    model_config = SettingsConfigDict(
        env_file=REPOSITORY_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ───────────────────────────────────────────────────────────
    APP_ENV: str = "development"
    APP_NAME: str = "ReguAZ API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = (
        "AI-powered Azerbaijani banking regulatory intelligence platform. "
        "Hybrid retrieval (Qdrant + BM25) with cross-encoder reranking and "
        "local LLM generation via llama.cpp."
    )
    DEBUG: bool = False
    REGUAZ_PIPELINE_VERSION: Literal["v1", "v2"] = "v2"

    # ── API ───────────────────────────────────────────────────────────────────
    API_V1_PREFIX: str = "/api/v1"

    # ── CORS ──────────────────────────────────────────────────────────────────
    # In development the React Vite dev server typically runs on 5173.
    CORS_ORIGINS: List[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = False
    CORS_ALLOW_METHODS: List[str] = ["GET", "POST", "OPTIONS"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]

    QDRANT_DIR: str = "qdrant_data"
    CHUNKS_DIR: str = "data/processed/chunks"
    METADATA_DIR: str = "data/processed/metadata"
    V2_ROOT: str = "data/processed/v2"
    V2_QDRANT_DIR: str = "data/processed/v2/qdrant"
    V2_QDRANT_ALIAS: str = "reguaz_v2_current"
    V2_LOCAL_FILES_ONLY: bool = True

    # ── Retrieval ─────────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "bge_m3"
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    TOP_K_SEMANTIC: int = 20
    TOP_K_BM25: int = 20
    RERANK_TOP_K: int = 15
    DEFAULT_TOP_K: int = 5
    RRF_K: int = 60

    # ── LLM ───────────────────────────────────────────────────────────────────
    LLM_TYPE: Literal[
        "gemma",
        "nvidia_gpt_oss",
        "groq_gpt_oss_20b",
        "groq_gpt_oss_120b",
    ] = "gemma"
    V2_EXPANSION_PER_SEED_CAP: int = 4
    V2_EXPANSION_GLOBAL_CAP: int = 16
    V2_REFERENCE_CONFIDENCE_MIN: float = 0.8
    V2_PARENT_EXCERPT_CHAR_LIMIT: int = 700
    V2_CONTEXT_SAFETY_MARGIN_TOKENS: int = 256

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the application settings singleton.

    The instance is constructed once and cached for the lifetime of the process.
    Use ``get_settings.cache_clear()`` in tests to reset the cache between runs.
    """
    return Settings()
