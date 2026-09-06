#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export HOST_UID="${HOST_UID:-$(id -u)}"
export HOST_GID="${HOST_GID:-$(id -g)}"
export HF_CACHE_DIR="${HF_CACHE_DIR:-${HOME}/.cache/huggingface}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: Docker is not installed or is not available on PATH." >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Error: Docker daemon is not running. Start Docker Desktop and retry." >&2
  exit 1
fi

if [[ ! -f "${REPOSITORY_ROOT}/.env" ]]; then
  echo "Error: ${REPOSITORY_ROOT}/.env is missing." >&2
  echo "Create it with: cp .env.example .env" >&2
  exit 1
fi

V2_ROOT="${REPOSITORY_ROOT}/data/processed/v2"
if [[ ! -f "${V2_ROOT}/manifest.json" || ! -d "${V2_ROOT}/qdrant" ]]; then
  echo "Error: required V2 manifest or Qdrant storage is missing under ${V2_ROOT}." >&2
  echo "Restore/build the ignored local V2 artefacts before starting Docker." >&2
  exit 1
fi

CLEANED_DOCUMENTS_ROOT="${REPOSITORY_ROOT}/data/processed/cleaned_documents"
if [[ ! -d "${CLEANED_DOCUMENTS_ROOT}" ]]; then
  echo "Error: authoritative cleaned documents are missing under ${CLEANED_DOCUMENTS_ROOT}." >&2
  echo "They are required for V2 evidence and citation validation." >&2
  exit 1
fi

HF_HUB_DIR="${HF_CACHE_DIR}/hub"
if [[ ! -d "${HF_HUB_DIR}/models--BAAI--bge-m3" ]]; then
  echo "Error: cached BAAI/bge-m3 snapshot was not found in ${HF_HUB_DIR}." >&2
  echo "Set HF_CACHE_DIR to the Hugging Face cache containing the V2 models." >&2
  exit 1
fi

if [[ ! -d "${HF_HUB_DIR}/models--BAAI--bge-reranker-v2-m3" ]]; then
  echo "Error: cached BAAI/bge-reranker-v2-m3 snapshot was not found in ${HF_HUB_DIR}." >&2
  echo "Set HF_CACHE_DIR to the Hugging Face cache containing the V2 models." >&2
  exit 1
fi

cd -- "${REPOSITORY_ROOT}"

docker compose config --quiet
docker compose up --build --detach

echo "ReguAZ containers started."
echo "UI:      http://127.0.0.1:${FRONTEND_PORT:-3000}"
echo "Backend: http://127.0.0.1:${BACKEND_PORT:-8000}"
echo "Follow startup: docker compose logs -f backend"
