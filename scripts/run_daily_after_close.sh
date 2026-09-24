#!/usr/bin/env bash
# 盘后 15:30 总入口：本地数据更新 → 盘后全链路（供 systemd 用户 timer 调用）。
# 也可手动: bash scripts/run_daily_after_close.sh
set -u
ROOT="${QUANT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${QUANT_PYTHON:-$(command -v python3 || true)}"
LOG_DIR="${QUANT_LOG_DIR:-$ROOT/generated/logs}"
RUN_DIR="${QUANT_RUN_DIR:-$ROOT/generated/run}"
LOG="${QUANT_DAILY_AFTER_CLOSE_LOG:-$LOG_DIR/daily_after_close.log}"
LOCK="${QUANT_DAILY_AFTER_CLOSE_LOCK:-$RUN_DIR/daily_after_close.lock}"
mkdir -p "$LOG_DIR" "$RUN_DIR"
if [ -z "$PY" ] || [ ! -x "$PY" ]; then
    echo "[$(date '+%F %T')] 无可用 python3，盘后入口中止" >> "$LOG"
    exit 1
fi
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "[$(date '+%F %T')] 盘后总入口已有实例运行，本次跳过" >> "$LOG"
    exit 75
fi
echo "[$(date '+%F %T')] 盘后总入口启动" >> "$LOG"

# 1. 本地数据更新（K线/估值/指数），成功后释放 after_close.lock
bash "$ROOT/scripts/after_close_update.sh" >> "$LOG" 2>&1
rc1=$?
echo "[$(date '+%F %T')] 本地数据更新完成 rc=$rc1" >> "$LOG"
if [ "$rc1" -ne 0 ]; then
    echo "[$(date '+%F %T')] 本地数据更新失败，停止报告链" >> "$LOG"
    exit "$rc1"
fi

# 2. 数据基座统一滚动更新：逐域重试，单域失败不丢失其他域状态。
"$PY" "$ROOT/scripts/rolling_data_update.py" --domains kline,industry,industry_chain --retries 2 >> "$LOG" 2>&1
rc_update=$?
echo "[$(date '+%F %T')] 数据基座滚动更新完成 rc=$rc_update" >> "$LOG"
if [ "$rc_update" -ne 0 ]; then
    echo "[$(date '+%F %T')] 数据基座滚动更新失败，停止报告链" >> "$LOG"
    exit "$rc_update"
fi

# 3. 盘后全链路（cloud_pull → 分析 → 研报 → 归档 → 飞书推送）
bash "$ROOT/scripts/run_after_close_pipeline.sh" >> "$LOG" 2>&1
rc2=$?
echo "[$(date '+%F %T')] 盘后全链路完成 rc=$rc2" >> "$LOG"
# 报告链结果优先；基座部分失败已写入 data_update_status.json。
exit "$rc2"
