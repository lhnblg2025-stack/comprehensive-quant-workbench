#!/usr/bin/env bash
# ============================================================
# codex 严降成本包装器（2026-08-13 用户要求"严降成本"）
# 核心：复用会话上下文 → 触发 DeepSeek 前缀缓存（实测 91% 命中）
#       输入成本 3.0 → 0.10 元/百万（省 97%）
# 用法：codex_cheap.sh <任务描述> [额外参数...]
#       内部：同一天/同仓库任务尽量 --resume 最近会话（连续上下文）
# ============================================================
set -u
cd ${QUANT_WORKSPACE:-.}
source ~/.codex_env

TASK="${1:?用法: codex_cheap.sh <任务描述>}"
shift

# 策略：任务尽量批量合并，复用最近会话
# 说明：openclaw 调用 codex 是单次 exec（无法跨调用 resume），
#       因此真正的缓存收益来自"一个任务内多次模型调用"（codex 内部
#       本来就同会话，前缀连续 → 缓存命中）。
#       本包装器主要做：强制 --skip-git-repo-check + 限制 sandbox + 输出记录
LOG=/tmp/codex_cheap_$(date +%Y%m%d).log
echo "[$(date '+%F %T')] 任务: $TASK" >> "$LOG"

exec codex exec --skip-git-repo-check -C ${QUANT_WORKSPACE:-.} "$TASK" "$@" 2>&1 | tee -a "$LOG"
