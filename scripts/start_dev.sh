#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

backend_pid=""
frontend_pid=""

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM

  if [[ -n "${backend_pid}" ]] && kill -0 "${backend_pid}" 2>/dev/null; then
    kill "${backend_pid}" 2>/dev/null || true
  fi
  if [[ -n "${frontend_pid}" ]] && kill -0 "${frontend_pid}" 2>/dev/null; then
    kill "${frontend_pid}" 2>/dev/null || true
  fi

  [[ -z "${backend_pid}" ]] || wait "${backend_pid}" 2>/dev/null || true
  [[ -z "${frontend_pid}" ]] || wait "${frontend_pid}" 2>/dev/null || true
  exit "${exit_code}"
}

trap cleanup EXIT INT TERM

"${SCRIPT_DIR}/start_backend.sh" &
backend_pid=$!

"${SCRIPT_DIR}/start_frontend.sh" &
frontend_pid=$!

echo "ReguAZ development services started. Press Ctrl+C to stop both."

# macOS ships Bash 3.2, which does not support `wait -n`. Polling both child
# processes keeps this launcher portable and tears down the peer if one exits.
while kill -0 "${backend_pid}" 2>/dev/null && kill -0 "${frontend_pid}" 2>/dev/null; do
  sleep 1
done

if ! kill -0 "${backend_pid}" 2>/dev/null; then
  wait "${backend_pid}"
else
  wait "${frontend_pid}"
fi
