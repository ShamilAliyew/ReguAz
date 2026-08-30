#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
FRONTEND_DIR="${REPOSITORY_ROOT}/frontend"

FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

if ! command -v npm >/dev/null 2>&1; then
  echo "Error: npm is not installed or is not available on PATH." >&2
  exit 1
fi

if [[ ! -d "${FRONTEND_DIR}/node_modules" ]]; then
  echo "Error: frontend dependencies are missing." >&2
  echo "Run: cd \"${FRONTEND_DIR}\" && npm ci" >&2
  exit 1
fi

cd -- "${FRONTEND_DIR}"

echo "Starting ReguAZ frontend at http://${FRONTEND_HOST}:${FRONTEND_PORT}"
exec npm run dev -- \
  --host "${FRONTEND_HOST}" \
  --port "${FRONTEND_PORT}"
