#!/usr/bin/env bash
# 盘后本地数据更新：收盘全市场快照批量落日线/估值，再生成特征与温度。
set -u
ROOT="${QUANT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${QUANT_PYTHON:-$(command -v python3 || true)}"
LOG_DIR="${QUANT_LOG_DIR:-$ROOT/generated/logs}"
RUN_DIR="${QUANT_RUN_DIR:-$ROOT/generated/run}"
SUMMARY="${QUANT_AFTER_CLOSE_LOG:-$LOG_DIR/after_close_update.log}"
LOCK="${QUANT_AFTER_CLOSE_LOCK:-$RUN_DIR/after_close.lock}"
mkdir -p "$LOG_DIR" "$RUN_DIR"

if [ -z "$PY" ] || [ ! -x "$PY" ]; then
    echo "[$(date '+%F %T')] 无可用 python3，更新中止" >> "$SUMMARY"
    exit 1
fi
if ! DAY="$($PY -c 'from quant_system.market_clock import latest_completed_trading_day; print(latest_completed_trading_day().isoformat())')"; then
    echo "[$(date '+%F %T')] 交易日计算失败，更新中止" >> "$SUMMARY"
    exit 1
fi
if [[ ! "$DAY" =~ ^20[0-9]{2}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "[$(date '+%F %T')] 非法交易日: $DAY" >> "$SUMMARY"
    exit 1
fi
DAY_COMPACT="${DAY//-/}"

exec 9>"$LOCK"
if ! flock -n 9; then
    echo "[$(date '+%F %T')] 已有盘后更新持锁运行，本次跳过" >> "$SUMMARY"
    exit 75
fi

echo "[$(date '+%F %T')] 盘后本地更新启动" >> "$SUMMARY"

run_parallel() {
    name="$1"
    limit="$2"
    log="$3"
    shift 3
    timeout --preserve-status "$limit" "$@" > "$log" 2>&1 &
    pid=$!
    eval "${name}_pid=$pid"
    echo "  $name PID=$pid -> $log" >> "$SUMMARY"
}

run_parallel close_snapshot 1800 "$LOG_DIR/update_close_snapshot_${DAY_COMPACT}.log" \
    "$PY" -u "$ROOT/scripts/update_daily_from_close_snapshot.py" --date "$DAY"
run_parallel index 900 "$LOG_DIR/update_index_${DAY_COMPACT}.log" \
    "$PY" -u "$ROOT/scripts/fetch_index_daily.py"

rc=0
for name in close_snapshot index; do
    eval "pid=\${${name}_pid}"
    wait "$pid"
    step_rc=$?
    echo "[$(date '+%F %T')] $name 完成 rc=$step_rc" >> "$SUMMARY"
    if [ "$step_rc" -ne 0 ] && [ "$rc" -eq 0 ]; then rc="$step_rc"; fi
done

if [ "$rc" -ne 0 ]; then
    echo "[$(date '+%F %T')] 基础数据更新失败 rc=$rc，下游构建停止" >> "$SUMMARY"
    exit "$rc"
fi

run_step() {
    name="$1"
    limit="$2"
    log="$3"
    shift 3
    timeout --preserve-status "$limit" "$@" >> "$log" 2>&1
    step_rc=$?
    echo "[$(date '+%F %T')] $name 完成 rc=$step_rc -> $log" >> "$SUMMARY"
    return "$step_rc"
}

run_step feature_store 1800 "$LOG_DIR/feature_store_${DAY_COMPACT}.log" \
    "$PY" -u "$ROOT/scripts/build_feature_store.py" --latest || exit $?
run_step temperature_replay 1800 "$LOG_DIR/temperature_replay_${DAY_COMPACT}.log" \
    "$PY" -u "$ROOT/quant_system/temperature_replay.py" --rebuild || exit $?
run_step position_map 300 "$LOG_DIR/temperature_replay_${DAY_COMPACT}.log" \
    "$PY" -u "$ROOT/quant_system/calibrate_position_map.py" || exit $?

echo "[$(date '+%F %T')] 盘后本地更新全部完成" >> "$SUMMARY"
exit 0
