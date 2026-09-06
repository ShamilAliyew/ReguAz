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

from pydantic import SecretStr
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
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["GET", "POST", "OPTIONS"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    CORS_EXPOSE_HEADERS: List[str] = [
        "X-Generation-Id",
        "X-TTS-Model",
        "X-TTS-Latency-Ms",
        "X-TTS-Characters",
        "Server-Timing",
    ]

    QDRANT_DIR: str = "qdrant_data"
    CHUNKS_DIR: str = "data/processed/chunks"
    METADATA_DIR: str = "data/processed/metadata"
    V2_ROOT: str = "data/processed/v2"
    V2_QDRANT_DIR: str = "data/processed/v2/qdrant"
    V2_QDRANT_ALIAS: str = "reguaz_v2_current"
    QDRANT_URL: str | None = None
    QDRANT_API_KEY: SecretStr | None = None
    QDRANT_TIMEOUT_SECONDS: float = 30.0
    V2_LOCAL_FILES_ONLY: bool = True

    # ── Authentication ────────────────────────────────────────────────────────
    AUTH_ENABLED: bool = False
    DATABASE_URL: SecretStr | None = None
    AUTH_COOKIE_NAME: str = "reguaz_session"
    AUTH_COOKIE_SECURE: bool = False
    AUTH_SESSION_TTL_DAYS: int = 7

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

    # ── Optional answer speech ────────────────────────────────────────────────
    OPENROUTER_TTS_ENABLED: bool = True
    OPENROUTER_API_KEY: SecretStr | None = None
    OPENROUTER_API_BASE: str = "https://openrouter.ai/api/v1"
    OPENROUTER_TTS_MODEL: Literal["fish-audio/s2.1-pro-free:free"] = (
        "fish-audio/s2.1-pro-free:free"
    )
    # Public Fish Audio Azerbaijani narrator voice. Keeping a reference ID
    # avoids provider-selected voice changes between requests.
    OPENROUTER_TTS_VOICE: str | None = "1db26754d7d84db39b8463c322c8d162"
    OPENROUTER_SITE_URL: str | None = None
    OPENROUTER_APP_TITLE: str = "ReguAZ"
    OPENROUTER_TTS_TIMEOUT_SECONDS: float = 60.0
    OPENROUTER_TTS_MAX_INPUT_CHARACTERS: int = 5_000
    OPENROUTER_TTS_MAX_AUDIO_BYTES: int = 15 * 1024 * 1024
    OPENROUTER_TTS_MAX_CONCURRENCY: int = 2
    OPENROUTER_TTS_MAX_RETRIES: int = 1

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
