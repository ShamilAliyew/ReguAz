# =============================================================================
# ReguAZ Backend — Dockerfile
#
# Build context: repo root  (docker build -f backend/Dockerfile .)
# Render settings:
#   Root Directory  : (leave empty / repo root)
#   Dockerfile Path : backend/Dockerfile
#
# Container layout
#   /app/              ← WORKDIR during dependency install
#     backend/         ← Full backend/ tree (app/, reguaz/, pyproject.toml …)
#     data/            ← processed chunks, metadata, qdrant collection
#
# Entry-point (same as local development):
#   uvicorn app.main:app
# launched from /app/backend so Python resolves `app` package correctly.
# main.py inserts /app (parent of /app/backend) into sys.path so that
# `from backend.app…` and `from backend.reguaz…` absolute imports work.
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Base image
#    python:3.12-slim satisfies requires-python = ">=3.12,<3.15" and keeps the
#    image lean.  The `slim` variant still ships gcc/g++ via build-essential
#    which is required to compile llama-cpp-python and several ML libraries.
# -----------------------------------------------------------------------------
FROM python:3.12-slim

# -----------------------------------------------------------------------------
# 2. Environment hardening
#    PYTHONDONTWRITEBYTECODE — avoids writing .pyc files (cleaner image).
#    PYTHONUNBUFFERED       — stdout/stderr flow directly to Render logs.
#    POETRY_* flags         — configure Poetry to behave headlessly and install
#                             packages into the global site-packages so that
#                             `uvicorn` is on PATH without activating a venv.
# -----------------------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.4.1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# -----------------------------------------------------------------------------
# 3. System dependencies
#    build-essential  — C/C++ toolchain for llama-cpp-python, tokenizers …
#    curl             — used by Poetry installer and health checks.
#    git              — some sentence-transformers / HF Hub installs need it.
#    libgomp1         — OpenMP runtime required by llama.cpp / torch.
#    After install the apt cache is removed to minimise layer size.
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# 4. Install Poetry
#    Pinned to POETRY_VERSION for reproducible builds.
# -----------------------------------------------------------------------------
RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

# -----------------------------------------------------------------------------
# 5. Working directory
#    /app is the container root; backend/ and data/ will live here.
# -----------------------------------------------------------------------------
WORKDIR /app

# -----------------------------------------------------------------------------
# 6. Dependency installation (layer-cached)
#    Copy only the lock files first so Docker can cache this expensive layer.
#    If pyproject.toml / poetry.lock do not change, pip install is skipped on
#    subsequent builds even when source code changes.
#
#    --only main  — skips dev, test, and frontend dependency groups.
#    --no-root    — the project is not an installable package; skip metadata.
# -----------------------------------------------------------------------------
COPY backend/pyproject.toml backend/poetry.lock ./backend/
RUN poetry install --no-root --no-ansi \
        --directory /app/backend \
    && rm -rf $POETRY_CACHE_DIR

# -----------------------------------------------------------------------------
# 7. Application source code
#    Copies the full backend/ tree (app/, reguaz/, __init__.py …).
#    The .dockerignore at backend/.dockerignore excludes models, logs, cache …
# -----------------------------------------------------------------------------
COPY backend/ ./backend/

# -----------------------------------------------------------------------------
# 8. Processed data & Qdrant collection
#    The retriever requires:
#      data/processed/chunks/   — JSONL chunk files
#      data/processed/metadata/ — document metadata
#      data/qdrant/             — local Qdrant vector collection
#
#    reguaz/config/settings.py resolves these relative to BACKEND_ROOT.parent
#    (i.e. /app) so placing data/ under /app matches both the fallback logic
#    in settings.py and lifespan.py.
#
#    GGUF model files (reguaz/models/*.gguf) are excluded via .dockerignore.
#    LLM inference is handled outside the container via Cloudflare Tunnel.
# -----------------------------------------------------------------------------
COPY data/processed/ ./data/processed/
COPY data/qdrant/    ./data/qdrant/

# -----------------------------------------------------------------------------
# 9. Runtime working directory
#    Switch to /app/backend so `uvicorn app.main:app` resolves the `app`
#    package from the current directory — identical to local development where
#    you run `uvicorn app.main:app` from inside the backend/ folder.
#    main.py adds /app (parent) to sys.path enabling `from backend.*` imports.
# -----------------------------------------------------------------------------
WORKDIR /app/backend

# -----------------------------------------------------------------------------
# 10. Port
#     Render injects PORT env var at runtime; we expose 8000 as the default.
#     The CMD below reads $PORT with a fallback so the image works everywhere.
# -----------------------------------------------------------------------------
EXPOSE 8000

# -----------------------------------------------------------------------------
# 11. Startup command
#     --host 0.0.0.0  — required for Render to route external traffic in.
#     --port          — Render sets $PORT; fallback to 8000 for local Docker.
#     No --reload     — production mode; reload adds latency and is unsafe.
#     workers=1       — llama.cpp model is not fork-safe; single worker only.
# -----------------------------------------------------------------------------
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
