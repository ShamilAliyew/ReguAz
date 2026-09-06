#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
POSTGRES_PORT="${POSTGRES_PORT:-5433}"

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

# The browser authentication flow requires PostgreSQL. For native development,
# bring up only the Compose database and inject a loopback connection URL into
# this backend process. Set REGUAZ_BOOTSTRAP_AUTH_DATABASE=false when supplying
# DATABASE_URL through the shell or root .env for an external database.
if [[ "${REGUAZ_BOOTSTRAP_AUTH_DATABASE:-true}" == "true" ]]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "Error: Docker is required to start the local authentication database." >&2
    echo "Start Docker Desktop, or set DATABASE_URL and REGUAZ_BOOTSTRAP_AUTH_DATABASE=false." >&2
    exit 1
  fi
  if ! docker info >/dev/null 2>&1; then
    echo "Error: Docker daemon is not running; the authentication database cannot start." >&2
    echo "Start Docker Desktop and retry." >&2
    exit 1
  fi

  export POSTGRES_PORT
  export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-reguaz_local_only}"
  docker compose up --detach --wait postgres
  export AUTH_ENABLED="true"
  export AUTH_COOKIE_SECURE="false"
  export DATABASE_URL="postgresql+psycopg://reguaz:${POSTGRES_PASSWORD}@127.0.0.1:${POSTGRES_PORT}/reguaz"
  echo "Authentication database ready at 127.0.0.1:${POSTGRES_PORT}"
fi

echo "Starting ReguAZ backend at http://${BACKEND_HOST}:${BACKEND_PORT}"
exec poetry run uvicorn backend.app.main:app \
  --host "${BACKEND_HOST}" \
  --port "${BACKEND_PORT}" \
  --reload
