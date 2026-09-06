# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv \
    && pip install "poetry==2.4.1" "poetry-plugin-export==1.9.0"

WORKDIR /app

# Dependency files are copied before source code so this expensive layer is
# reused whenever only application code changes.
COPY backend/pyproject.toml backend/poetry.lock ./backend/
# Render uses remote Groq/NVIDIA providers. Exclude the optional local
# llama.cpp runtime: compiling it from source exceeds Render's build RAM.
RUN --mount=type=cache,target=/root/.cache/pip \
    poetry export \
        --directory /app/backend \
        --only main \
        --without-hashes \
        --format requirements.txt \
        --output /tmp/requirements.txt \
    && grep -Eiv '^(torch|triton|nvidia-|cuda-|llama-cpp-python)' \
        /tmp/requirements.txt > /tmp/requirements-cpu.txt \
    && pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.12.1+cpu" \
    && pip install --requirement /tmp/requirements-cpu.txt


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    PATH="/opt/venv/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY backend/ ./backend/

# These are immutable, hash-validated retrieval contracts. Dense/sparse shards,
# embedded Qdrant storage and local GGUF files remain excluded by .dockerignore.
COPY data/processed/v2/manifest.json ./data/processed/v2/manifest.json
COPY data/processed/v2/documents/ ./data/processed/v2/documents/
COPY data/processed/v2/embeddings/ ./data/processed/v2/embeddings/
COPY data/processed/cleaned_documents/ ./data/processed/cleaned_documents/

# Qdrant storage, embedding shards, Hugging Face snapshots and optional local
# GGUF models are deliberately not baked into the portable image.
EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
