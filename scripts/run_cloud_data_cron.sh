#!/usr/bin/env bash
# 云数据分阶段编排：任何阶段失败都保留真实退出码，不让长串联吞错。
set -u
ROOT="${QUANT_WORKSPACE:-.}"
cd "$ROOT"
mkdir -p generated/logs
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="generated/logs/cloud_data_${STAMP}.log"
rc=0

run_step() {
  name="$1"
  limit="$2"
  shift 2
  printf '[%s] START %s timeout=%ss\n' "$(date '+%F %T')" "$name" "$limit" >> "$LOG"
  timeout --preserve-status "$limit" "$@" >> "$LOG" 2>&1
  step_rc=$?
  printf '[%s] END %s rc=%s\n' "$(date '+%F %T')" "$name" "$step_rc" >> "$LOG"
  if [ "$step_rc" -ne 0 ] && [ "$rc" -eq 0 ]; then rc="$step_rc"; fi
}

run_step pull_cninfo 240 python3 scripts/pull_cloud_data.py --only cninfo
run_step pull_hot_rank 240 python3 scripts/pull_cloud_data.py --only hot_rank
run_step pull_ths_concept 240 python3 scripts/pull_cloud_data.py --only ths_concept
# 2026-08-24 修复: 补齐决策核心数据源。此前只拉4源，导致 fusion/zt_daily_stats/theme_cycle
# 停 08-21，指数决策与涨停天梯全用旧数据（=“数据不自动回滚”根因之一）。
run_step pull_fusion 240 python3 scripts/pull_cloud_data.py --only fusion
run_step pull_zt_daily_stats 240 python3 scripts/pull_cloud_data.py --only zt_daily_stats
run_step pull_theme_cycle 240 python3 scripts/pull_cloud_data.py --only theme_cycle
run_step pull_zt_pool 300 python3 scripts/pull_cloud_data.py --only zt_pool
run_step pull_lhb 300 python3 scripts/pull_cloud_data.py --only lhb
run_step pull_fund_flow 300 python3 scripts/pull_cloud_data.py --only fund_flow
run_step pull_zt_pool_em_daily 240 python3 scripts/pull_cloud_data.py --only zt_pool_em_daily
printf '[%s] SUMMARY rc=%s log=%s\n' "$(date '+%F %T')" "$rc" "$LOG" >> "$LOG"
exit "$rc"
