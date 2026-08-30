#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"

if ! command -v poetry >/dev/null 2>&1; then
  echo "Error: Poetry is not installed or is not available on PATH." >&2
  echo "Install it from https://python-poetry.org/docs/#installation" >&2
  exit 1
fi

if [[ ! -f "${REPOSITORY_ROOT}/.env" ]]; then
  echo "Warning: ${REPOSITORY_ROOT}/.env is missing; optional remote LLM/TTS providers will be unavailable." >&2
  echo "Create it from .env.example before testing those providers." >&2
fi

cd -- "${REPOSITORY_ROOT}"

echo "Starting ReguAZ backend at http://${BACKEND_HOST}:${BACKEND_PORT}"
exec poetry run uvicorn backend.app.main:app \
  --host "${BACKEND_HOST}" \
  --port "${BACKEND_PORT}" \
  --reload
