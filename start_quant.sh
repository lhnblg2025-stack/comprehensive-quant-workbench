#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec "${QUANT_PYTHON:-python3}" -m quant_web.server --port "${QUANT_WEB_PORT:-8600}"
