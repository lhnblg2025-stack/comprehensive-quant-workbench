#!/usr/bin/env bash
# 唯一盘后编排入口：数据完成 -> 契约校验 -> 分析 -> 报告 -> 推送。
set -uo pipefail
ROOT="${QUANT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${QUANT_PYTHON:-$(command -v python3)}"

if [[ ! -x "$PY" ]]; then
  echo "python runtime is not executable: $PY" >&2
  exit 2
fi
DAY="${QUANT_BUSINESS_DAY:-}"
if [[ -z "$DAY" ]]; then
  DAY="$($PY - <<'PY'
from quant_system.market_clock import latest_completed_trading_day
print(latest_completed_trading_day().isoformat())
PY
)"
fi
if [[ ! "$DAY" =~ ^20[0-9]{2}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "invalid completed trading day: $DAY" >&2
  exit 2
fi
RUN_ID="${QUANT_RUN_ID:-after-close-${DAY}-$(date +%Y%m%dT%H%M%S)-$$}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "invalid run id: $RUN_ID" >&2
  exit 2
fi
export QUANT_RUN_ID="$RUN_ID" QUANT_AS_OF="$DAY" QUANT_ROOT="$ROOT" QUANT_PYTHON="$PY"
LOG_DIR="$ROOT/generated/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/after_close_pipeline_${DAY}.log"
STATUS_DIR="$ROOT/generated/runtime_status"
STATUS_FILE="$STATUS_DIR/after_close_shell.json"
MANIFEST_DIR="$ROOT/generated/runs/$RUN_ID"
MANIFEST_FILE="$MANIFEST_DIR/manifest.json"
mkdir -p "$STATUS_DIR" "$MANIFEST_DIR"
write_status() {
  local state="$1"; local stage="${2:-}"; local rc="${3:-}"
  local tmp="$STATUS_FILE.$$.$RANDOM.tmp"
  QUANT_STATUS_STATE="$state" QUANT_STATUS_STAGE="$stage" QUANT_STATUS_RC="$rc" \
    "$PY" - "$STATUS_FILE" "$MANIFEST_FILE" "$RUN_ID" "$DAY" "$tmp" <<'PY'
import json, os, sys
from datetime import datetime, timezone, timedelta
status_path, manifest_path, run_id, day, tmp = sys.argv[1:]
payload = {"schema": "after_close_shell/v1", "run_id": run_id, "data_as_of": day,
           "status": os.environ["QUANT_STATUS_STATE"], "current_stage": os.environ.get("QUANT_STATUS_STAGE") or None,
           "returncode": int(os.environ["QUANT_STATUS_RC"] or 0),
           "updated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")}
for path, temp in ((status_path, tmp), (manifest_path, manifest_path + f".{os.getpid()}.tmp")):
    with open(temp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2); fh.write("\n")
    os.replace(temp, path)
PY
}
CURRENT_STAGE="startup"
FINISHED=0
on_exit() {
  local rc=$?
  if [[ "$FINISHED" -eq 0 ]]; then
    write_status failed "$CURRENT_STAGE" "$rc" || true
  fi
}
trap on_exit EXIT INT TERM
write_status running "$CURRENT_STAGE" 0 || { echo "cannot persist startup status" >&2; exit 2; }
LOCK_DIR="${QUANT_LOCK_DIR:-${ROOT}/generated/locks}"
LOCK="$LOCK_DIR/after_close_pipeline.lock"
LOCAL_LOCK="$LOCK_DIR/after_close.lock"
mkdir -p "$LOCK_DIR"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[$(date '+%F %T')] 盘后全链路已有实例运行，当前执行明确跳过" >> "$LOG"
  exit 75
fi

run_step() {
  name="$1"; limit="$2"; shift 2
  CURRENT_STAGE="$name"
  write_status running "$name" 0 || return 2
  echo "[$(date '+%F %T')] START $name timeout=${limit}s run_id=$RUN_ID" >> "$LOG"
  timeout --preserve-status "$limit" "$@" >> "$LOG" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] END $name rc=$rc" >> "$LOG"
  if [[ "$rc" -ne 0 ]]; then
    write_status failed "$name" "$rc"
  else
    write_status running "$name.complete" 0
  fi
  return "$rc"
}

# 等待17:05本地更新释放锁；没有本地任务时立即通过。
exec 8>"$LOCAL_LOCK"
if ! flock -w 10800 8; then
  echo "[$(date '+%F %T')] 等待本地数据更新超时" >> "$LOG"
  exit 75
fi
flock -u 8

run_step cloud_pull 3600 bash "$ROOT/scripts/run_cloud_data_cron.sh" || exit $?
run_step index_refresh 900 "$PY" -u "$ROOT/scripts/fetch_index_daily.py" || exit $?
run_step etf_state_refresh 300 "$PY" -u "$ROOT/scripts/fetch_etf_state.py" || exit $?
# ETF历史/规模/申赎代理增量回填；外部源失败不阻断日报，保留已有仓库。
run_step etf_history_backfill 1800 "$PY" -u "$ROOT/scripts/backfill_etf_sector_history.py" --only etf --start "$($PY -c 'from datetime import datetime; print(datetime.now().replace(year=datetime.now().year-2).strftime("%Y%m%d"))')"
rc_etf_history=$?
if [ "$rc_etf_history" -ne 0 ]; then
  echo "[$(date '+%F %T')] WARN etf_history_backfill rc=$rc_etf_history，保留旧ETF历史" >> "$LOG"
fi
run_step official_capital_events 120 "$PY" -u "$ROOT/scripts/fetch_official_capital_events.py" || exit $?
# 龙虎榜为盘后决策关键数据；交易日无数据必须失败，避免静默沿用旧历史。
run_step lhb_refresh 300 "$PY" -u "$ROOT/scripts/update_lhb_daily.py" --date "$DAY" || exit $?
# 统一市场舆情滚动：股吧/微博/百度/B站/雪球/新闻，失败隔离，不阻断行情报告。
run_step social_sentiment 600 "$PY" -u -c 'from quant_system.analysis_core.social_sentiment import collect_all, store; data=collect_all(); print(store(data))'
rc_social=$?
if [ "$rc_social" -ne 0 ]; then
  echo "[$(date '+%F %T')] WARN social_sentiment rc=$rc_social，保留上次舆情快照" >> "$LOG"
fi

# 关键日频数据必须至少到最新交易日；禁止旧数据继续生成新日期报告。
run_step freshness_guard 120 "$PY" -u "$ROOT/scripts/freshness_guard.py" "$DAY" || exit $?
# 海外快照在数据发布门禁前刷新；源失败由海外评分自身 fail-closed。
run_step overseas_release_snapshot 180 "$PY" -u "$ROOT/scripts/overseas_collector.py" --save
rc_overseas_snapshot=$?
if [ "$rc_overseas_snapshot" -ne 0 ]; then
  echo "[$(date '+%F %T')] WARN overseas_release_snapshot rc=$rc_overseas_snapshot" >> "$LOG"
fi
# 所有正式消费者只允许读取同日、同一哈希集合的数据发布版本。
run_step data_release_gate 300 "$PY" -u -m quant_system.data_release "$DAY"
rc_release=$?
if [ "$rc_release" -ne 0 ]; then
  # One bounded domain-repair pass, then a hard revalidation.
  run_step release_repair 1800 "$PY" -u "$ROOT/scripts/release_repair.py" "$DAY" market_index benchmark_csi300 valuation financial lhb_hist zt_history overseas || exit $?
  run_step data_release_recheck 300 "$PY" -u -m quant_system.data_release "$DAY" || exit $?
fi

run_step short_term_pipeline 3600 "$PY" -u -m quant_system.analysis_core.pipeline --daily || exit $?
run_step multi_agent 1200 "$PY" -u -m quant_system.analysis_core.multi_agent || exit $?
# 因子衰减/盘中风格漂移产物：有快照则计算，无快照降级但不阻断日报。
run_step factor_decay 300 "$PY" -u -m quant_system.analysis_core.intraday_factor_decay --date "$DAY"
rc_factor_decay=$?
run_step factor_oos_refresh 900 "$PY" -u "$ROOT/scripts/ic_oos_report.py" --n 150 --seed 42 --forward 5 || exit $?
run_step factor_quality_registry 120 "$PY" -u "$ROOT/scripts/build_factor_quality_registry.py" || exit $?
run_step factor_revalidation 120 "$PY" -u "$ROOT/scripts/factor_revalidation.py" || exit $?
# 唯一生产因子组合：仅输出绑定同日release的目标权重。
run_step production_targets 300 "$PY" -u "$ROOT/scripts/export_production_targets.py" --date "$DAY"
rc_targets=$?
if [ "$rc_targets" -ne 0 ]; then
  echo "[$(date '+%F %T')] WARN production targets unavailable; no paper order target emitted" >> "$LOG"
fi
run_step production_pipeline 1800 "$PY" -u -m quant_system.production_pipeline --date "$DAY"
rc_production_pipeline=$?
if [ "$rc_production_pipeline" -ne 0 ]; then
  echo "[$(date '+%F %T')] WARN production pipeline audit unavailable" >> "$LOG"
fi
# 核心因子截面策略真实回测（OOS方向校准后），仅作策略研究审计。
run_step factor_strategy_backtest 600 "$PY" -u "$ROOT/scripts/factor_strategy_backtest.py" --n 150 || exit $?
run_step industry_chain_history 120 "$PY" -u "$ROOT/scripts/collect_industry_chain_history.py" --date "$DAY" || exit $?
if [ "$rc_factor_decay" -ne 0 ]; then
  echo "[$(date '+%F %T')] WARN factor_decay rc=$rc_factor_decay，保留上一轮因子健康状态" >> "$LOG"
fi
run_step fusion_review 1200 "$PY" -u "$ROOT/scripts/daily_review_chain.py" --date "$DAY" || exit $?

# IMA仅作为外部研报输入，不在盘后主链展示或重复解析，避免慢和空转。
# 资金/ETF/行业融合快照只读本地仓库；失败告警但不阻断 HTML 报告。
run_step intraday_flow_history 180 "$PY" -u "$ROOT/scripts/build_intraday_flow_history.py" --date "$DAY"
rc_intraday_history=$?
if [ "$rc_intraday_history" -ne 0 ]; then
  echo "[$(date '+%F %T')] WARN intraday_flow_history rc=$rc_intraday_history，报告明确降级分时图" >> "$LOG"
fi
run_step research_fusion_snapshot 300 "$PY" -u "$ROOT/scripts/research_fusion_snapshot.py" --date "$DAY" || exit $?
run_step decision_snapshot 300 "$PY" -u "$ROOT/scripts/unified_decision_snapshot.py" --mode after_close --date "$DAY" || exit $?
# 将同日统一决策写入融合研报及其飞书正文，禁止旧模板单独拼接结论。
REPORT_ROOT="${QUANT_REPORT_ROOT:-$ROOT/../Desktop/研报共享}"
FUSION_SOURCE="$REPORT_ROOT/A股融合研报_${DAY}.html"
if [ -f "$FUSION_SOURCE" ]; then
  run_step fusion_report_sync 180 "$PY" -u "$ROOT/scripts/sync_fusion_workbench.py" --snapshot "$ROOT/generated/decision_snapshot_after_close_${DAY}.json" --source "$FUSION_SOURCE" || exit $?
else
  echo "[$(date '+%F %T')] WARN fusion report source missing: $FUSION_SOURCE" >> "$LOG"
fi
run_step html_report 600 "$PY" -u "$ROOT/scripts/html_report_generator.py" "$DAY" || exit $?

# 产物归档、契约校验和投递必须全部成功才标记盘后链完成。
run_step archive_and_backup 300 "$PY" -u "$ROOT/scripts/archive_report.py" "$DAY" || exit $?
run_step artifact_guard 60 "$PY" -u "$ROOT/scripts/artifact_guard.py" "$DAY" || exit $?
run_step feishu_push 180 "$PY" -u "$ROOT/scripts/push_after_close_report.py" --date "$DAY" || exit $?
write_status succeeded complete 0
FINISHED=1
echo "[$(date '+%F %T')] 全链路完成 day=$DAY run_id=$RUN_ID" >> "$LOG"
exit 0
