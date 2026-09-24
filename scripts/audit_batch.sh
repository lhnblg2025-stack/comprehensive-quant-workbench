#!/bin/bash
# audit_batch.sh — 批量派发 codex+deepseek 逐行审计
# W2.5 安全: DEEPSEEK_API_KEY 不再硬编码（原 sk-* 已入 git 历史，需轮换），运行前从环境注入
cd ${PROJECT_ROOT}
mkdir -p /tmp/audit

PROMPT_TEMPLATE='逐行审计以下文件，找出真实bug。要求：1)逐行阅读，重点找 逻辑错误/前视偏差/数据对齐/索引错位/除零/异常吞噬/硬编码日期/边界条件/并发写文件/结果可疑(有输出但错误)；2)输出格式：文件:行号|级别(Critical/Major/Minor)|问题描述|修复建议；3)只报真实问题并给证据，不要泛泛建议；4)中文回答；5)开头列出每个文件的函数清单便于核对。文件列表：'

BATCH_A="quant_system/analysis_core/battle_map.py quant_system/analysis_core/resonance_scorer.py quant_system/analysis_core/industry_graph.py quant_system/analysis_core/order_dispatcher.py quant_system/analysis_core/capital_allocator.py"
BATCH_B="quant_system/analysis_core/intraday_guard.py quant_system/analysis_core/self_healer.py quant_system/analysis_core/behavior_audit.py quant_system/analysis_core/audit_trail.py quant_system/analysis_core/calibration.py quant_system/analysis_core/regime_classifier.py quant_system/analysis_core/macro_veto.py quant_system/analysis_core/regime_drift_detector.py quant_system/analysis_core/pipeline.py"
BATCH_C="quant_system/analysis_core/zt_pool_history.py quant_system/analysis_core/ladder.py quant_system/analysis_core/emotion_cycle.py quant_system/analysis_core/theme_cycle.py quant_system/analysis_core/fund_forces.py quant_system/analysis_core/scenario.py quant_system/analysis_core/decision_card.py quant_system/analysis_core/fusion.py quant_system/analysis_core/predictions.py"
BATCH_D="quant_system/analysis_core/broker_gaming.py quant_system/analysis_core/short_term_extra.py quant_system/analysis_core/social_sentiment.py quant_system/analysis_core/knowledge_rag.py quant_system/analysis_core/company_analysis.py quant_system/analysis_core/valuation.py quant_system/analysis_core/data_sources.py quant_system/analysis_core/daily_report.py quant_system/analysis_core/report_render.py"
BATCH_E="quant_web/server.py quant_web/handlers/v11.py quant_web/stock_analysis.py quant_web/health_watchdog.py quant_platform/openclaw_api.py"
BATCH_F="scripts/a_share_daily_report.py"
BATCH_G="quant_system/data_store.py quant_system/backtest_engine.py quant_system/factor_model.py quant_system/market_regime.py quant_system/execution.py quant_system/performance.py"

run_batch() {
  local name="$1"; shift
  local files="$*"
  local out="/tmp/audit/${name}.log"
  nohup codex exec --sandbox read-only "${PROMPT_TEMPLATE} ${files}" > "$out" 2>&1 &
  echo "${name} PID $! → $out"
}

case "$1" in
  A) run_batch A $BATCH_A ;;
  B) run_batch B $BATCH_B ;;
  C) run_batch C $BATCH_C ;;
  D) run_batch D $BATCH_D ;;
  E) run_batch E $BATCH_E ;;
  F) run_batch F $BATCH_F ;;
  G) run_batch G $BATCH_G ;;
  all) run_batch A $BATCH_A; run_batch B $BATCH_B; run_batch C $BATCH_C
       run_batch D $BATCH_D; run_batch E $BATCH_E; run_batch F $BATCH_F; run_batch G $BATCH_G ;;
esac
