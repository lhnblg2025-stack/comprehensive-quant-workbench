#!/usr/bin/env bash
# 云端数据回传兜底：19:30 由 systemd timer 触发，覆盖 15:30 主链 cloud_pull
# 因时间过早（早于云端 18:35/18:45 爬虫产出）而漏掉的当日回传。
# 与主链共享 flock，避免与仍运行中的盘后全链路重复回传。
set -u
ROOT="${QUANT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${QUANT_PYTHON:-$(command -v python3 || true)}"
LOG_DIR="${QUANT_LOG_DIR:-$ROOT/generated/logs}"
RUN_DIR="${QUANT_RUN_DIR:-$ROOT/generated/run}"
LOG="${QUANT_CLOUD_FALLBACK_LOG:-$LOG_DIR/cloud_pull_fallback.log}"
LOCK="${QUANT_CLOUD_FALLBACK_LOCK:-$RUN_DIR/cloud_pull.lock}"
mkdir -p "$LOG_DIR" "$RUN_DIR"
if [ -z "$PY" ] || [ ! -x "$PY" ]; then
  echo "[$(date '+%F %T')] 无可用 python3，云端回传中止" >> "$LOG"
  exit 1
fi

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[$(date '+%F %T')] 云端回传已在运行，本次跳过" >> "$LOG"
  exit 75
fi

echo "[$(date '+%F %T')] 云端回传兜底启动" >> "$LOG"
bash "$ROOT/scripts/run_cloud_data_cron.sh" >> "$LOG" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "[$(date '+%F %T')] 云端回传失败，跳过派生重建 rc=$rc" >> "$LOG"
  exit "$rc"
fi
# 回传只是第一步：必须重建短线派生层，否则页面仍消费旧 fusion/fund_forces。
if ! DAY="$($PY -c 'from quant_system.market_clock import latest_completed_trading_day; print(latest_completed_trading_day().isoformat())')"; then
  echo "[$(date '+%F %T')] 交易日计算失败，派生重建中止" >> "$LOG"
  exit 1
fi
if [[ ! "$DAY" =~ ^20[0-9]{2}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "[$(date '+%F %T')] 非法交易日: $DAY" >> "$LOG"
  exit 1
fi
run_step() {
  name="$1"; shift
  timeout --preserve-status 1800 "$@" >> "$LOG" 2>&1
  step_rc=$?
  echo "[$(date '+%F %T')] $name 完成 rc=$step_rc" >> "$LOG"
  if [ "$step_rc" -ne 0 ] && [ "$rc" -eq 0 ]; then rc="$step_rc"; fi
}
# 龙虎榜本地直连补齐当日原始记录，资金合力必须先有同日输入。
run_step leaderboard "$PY" -u "$ROOT/scripts/update_lhb_daily.py" --date "$DAY"
run_step social_sentiment "$PY" -u -c 'from quant_system.analysis_core.social_sentiment import collect_all, store; store(collect_all())'
# IMA 每日增量提取：只处理新增/变更媒体，结果进入统一研报基座。
run_step ima_extract "$PY" -u "$ROOT/scripts/extract_ima_media.py"
run_step fund_forces "$PY" -u -m quant_system.analysis_core.fund_forces --build
run_step fusion "$PY" -u -m quant_system.analysis_core.fusion --today
run_step research_flow "$PY" -u -m quant_system.analysis_core.research_flow --date "$DAY"
run_step research_fusion "$PY" -u "$ROOT/scripts/research_fusion_snapshot.py" --date "$DAY"
run_step decision_snapshot "$PY" -u "$ROOT/scripts/unified_decision_snapshot.py" --mode after_close --date "$DAY"
run_step data_base_status "$PY" -u "$ROOT/scripts/write_data_base_status.py"
echo "[$(date '+%F %T')] 云端回传兜底及派生重建完成 rc=$rc day=$DAY" >> "$LOG"
exit "$rc"
