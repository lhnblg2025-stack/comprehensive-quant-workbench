#!/usr/bin/env bash
set -u
ROOT="${PROJECT_ROOT}"
cd "$ROOT"
mkdir -p generated/logs
LOG="generated/logs/multi_system_review_$(date +%Y%m%d).log"
timeout 900 python3 -m quant_system.analysis_core.multi_agent >> "$LOG" 2>&1
rc=$?
printf '[%s] multi_system_review_rc=%s\n' "$(date '+%F %T')" "$rc" >> "$LOG"
exit "$rc"

