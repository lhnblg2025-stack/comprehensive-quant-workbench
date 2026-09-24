#!/usr/bin/env bash
# Read-only unattended-service acceptance check for local or cloud hosts.
# This script never starts, restarts, or mutates services.
set -euo pipefail

BASE_URL="${QUANT_VERIFY_URL:-http://127.0.0.1:8600}"
API_KEY="${QUANT_WEB_API_KEY:-}"
APP_ROOT="${QUANT_APP_ROOT:-/opt/quant}"
CURL=(curl -fsS --max-time 5)
[[ -z "$API_KEY" ]] || CURL+=(-H "X-API-Key: $API_KEY")

failures=0
check() {
  local name="$1"; shift
  if "$@" >/dev/null 2>&1; then
    printf 'PASS %s\n' "$name"
  else
    printf 'FAIL %s\n' "$name" >&2
    failures=$((failures + 1))
  fi
}

check livez "${CURL[@]}" "$BASE_URL/api/livez"
check readyz "${CURL[@]}" "$BASE_URL/api/readyz"
check version "${CURL[@]}" "$BASE_URL/api/version"
check auth_status "${CURL[@]}" "$BASE_URL/api/auth_status"

if command -v pgrep >/dev/null 2>&1; then
  count="$(pgrep -fc 'quant_web/server.py' || true)"
  if [[ "$count" -eq 1 ]]; then
    printf 'PASS single_instance pid_count=%s\n' "$count"
  else
    printf 'FAIL single_instance pid_count=%s\n' "$count" >&2
    failures=$((failures + 1))
  fi
fi

if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files quant-web.service >/dev/null 2>&1; then
  check quant_web_active systemctl is-active --quiet quant-web.service
  check after_close_timer systemctl is-enabled --quiet quant-after-close.timer
  check data_update_timer systemctl is-enabled --quiet quant-data-update.timer
  check resident_recovery_timer systemctl is-enabled --quiet quant-resident-recovery.timer
  check resident_digest_timer systemctl is-enabled --quiet quant-resident-digest.timer
  if [[ -L "$APP_ROOT/current" && -f "$APP_ROOT/current/VERSION" ]]; then
    printf 'PASS current_release %s\n' "$(readlink -f "$APP_ROOT/current")"
  else
    printf 'FAIL current_release %s/current\n' "$APP_ROOT" >&2
    failures=$((failures + 1))
  fi
fi

if [[ "$failures" -ne 0 ]]; then
  printf 'RESULT failed failures=%s\n' "$failures" >&2
  exit 1
fi
printf 'RESULT ok\n'
