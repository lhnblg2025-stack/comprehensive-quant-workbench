#!/usr/bin/env bash
set -u
ROOT="${QUANT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${QUANT_PYTHON:-$(command -v python3)}"
cd "$ROOT" || exit 2
mkdir -p generated/logs generated/locks generated/runtime_status
LOG="generated/logs/intraday_guard_$(date +%Y%m%d).log"
STATUS="generated/runtime_status/intraday.json"
LOCK="generated/locks/intraday_full_market.lock"
HM="$(date +%H:%M)"
MINUTE=$((10#$(date +%M)))

write_status() {
  local status="$1" exit_code="$2" phase="$3"
  STATUS_TMP="${STATUS}.tmp.$$"
  python3 - "$STATUS_TMP" "$status" "$exit_code" "$phase" "$snapshot_rc" "$etf_rc" "$sector_flow_rc" "$history_rc" "$full_rc" "$risk_rc" <<'PY'
import json, os, sys
from datetime import datetime, timezone, timedelta
path, status, exit_code, phase = sys.argv[1:5]
keys = ("snapshot_rc", "etf_rc", "sector_flow_rc", "history_rc", "full_rc", "risk_rc")
values = [int(x) for x in sys.argv[5:11]]
data = {"schema": "intraday_cron/v1", "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "status": status, "exit_code": int(exit_code), "phase": phase,
        "stages": dict(zip(keys, values))}
tmp = path
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
os.replace(tmp, path)
PY
}

snapshot_rc=0; etf_rc=0; sector_flow_rc=0; history_rc=0; full_rc=0; risk_rc=0
exec 9>"$LOCK"
if ! flock -n 9; then
  printf '[%s] previous intraday job still running; skipped\n' "$(date '+%F %T')" >> "$LOG"
  exit 75
fi

if [[ ! -x "$PYTHON" ]]; then
  printf '[%s] invalid python runtime: %s\n' "$(date '+%F %T')" "$PYTHON" >> "$LOG"
  write_status blocked 2 startup
  exit 2
fi

if [ "$HM" = "09:25" ]; then
  timeout 120 "$PYTHON" scripts/auction_decision.py >> "$LOG" 2>&1
  rc=$?
  printf '[%s] phase=auction_report rc=%s\n' "$(date '+%F %T')" "$rc" >> "$LOG"
  write_status "$([ "$rc" -eq 0 ] && echo ok || echo failed)" "$rc" auction_report
  exit "$rc"
fi

if (( MINUTE % 5 != 0 )); then
  printf '[%s] phase=idle minute=%s; next collection boundary pending\n' "$(date '+%F %T')" "$MINUTE" >> "$LOG"
  write_status idle 0 idle
  exit 0
fi

timeout 90 "$PYTHON" scripts/realtime_snapshot.py >> "$LOG" 2>&1; snapshot_rc=$?
timeout 90 "$PYTHON" scripts/realtime_etf_snapshot.py >> "$LOG" 2>&1; etf_rc=$?
timeout 120 "$PYTHON" scripts/backfill_etf_sector_history.py --only sector >> "$LOG" 2>&1; sector_flow_rc=$?
timeout 120 "$PYTHON" scripts/build_intraday_flow_history.py --date "$(date +%F)" >> "$LOG" 2>&1; history_rc=$?

# Quotes are a hard dependency for all analysis. Other collectors are degraded,
# but remain visible in the machine-readable status.
if [ "$snapshot_rc" -ne 0 ]; then
  printf '[%s] phase=collection snapshot_rc=%s; analysis blocked\n' "$(date '+%F %T')" "$snapshot_rc" >> "$LOG"
  write_status blocked "$snapshot_rc" collection
  exit "$snapshot_rc"
fi

if (( MINUTE % 10 == 0 )); then
  timeout 600 "$PYTHON" -m quant_system.analysis_core.intraday_guard --full-decision >> "$LOG" 2>&1; full_rc=$?
  printf '[%s] phase=full_short_term_decision rc=%s\n' "$(date '+%F %T')" "$full_rc" >> "$LOG"
fi
if (( MINUTE % 30 == 0 )); then
  timeout 180 "$PYTHON" -m quant_system.analysis_core.intraday_guard --check --push-alerts >> "$LOG" 2>&1; risk_rc=$?
  printf '[%s] phase=risk_warning rc=%s\n' "$(date '+%F %T')" "$risk_rc" >> "$LOG"
fi

"$PYTHON" -u "$ROOT/scripts/write_data_base_status.py" >> "$LOG" 2>&1 || true
printf '[%s] phase=collection snapshot_rc=%s etf_rc=%s sector_flow_rc=%s history_rc=%s full_rc=%s risk_rc=%s\n' "$(date '+%F %T')" "$snapshot_rc" "$etf_rc" "$sector_flow_rc" "$history_rc" "$full_rc" "$risk_rc" >> "$LOG"

# Full decision and risk checks are hard failures; auxiliary collectors degrade.
final_rc=0
final_status=ok
if [ "$full_rc" -ne 0 ] || [ "$risk_rc" -ne 0 ]; then
  final_rc="${full_rc:-0}"
  [ "$final_rc" -eq 0 ] && final_rc="$risk_rc"
  final_status=failed
elif [ "$etf_rc" -ne 0 ] || [ "$sector_flow_rc" -ne 0 ] || [ "$history_rc" -ne 0 ]; then
  final_status=degraded
fi
write_status "$final_status" "$final_rc" collection
exit "$final_rc"
