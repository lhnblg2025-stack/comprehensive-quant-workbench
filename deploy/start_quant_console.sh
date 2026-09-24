#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$ROOT/start_quant.sh"
if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "http://127.0.0.1:${QUANT_WEB_PORT:-8600}/" >/dev/null 2>&1 || true
fi
