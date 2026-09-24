#!/usr/bin/env bash
set -u
ROOT="${QUANT_WORKSPACE:-.}"
cd "$ROOT"
mkdir -p generated/logs
LOG="generated/logs/pre_market_$(date +%Y%m%d).log"
python3 -m quant_system.analysis_core.pipeline --pre-market >> "$LOG" 2>&1
rc=$?
printf '[%s] pre_market_rc=%s\n' "$(date '+%F %T')" "$rc" >> "$LOG"
exit "$rc"

