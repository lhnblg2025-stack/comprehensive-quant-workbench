#!/usr/bin/env bash
set -u
ROOT="${PROJECT_ROOT}"
cd "$ROOT"
mkdir -p generated/logs
LOG="generated/logs/data_health_$(date +%Y%m%d).log"
python3 scripts/stamp_data_freshness.py --json >> "$LOG" 2>&1
stamp_rc=$?
digest_rc=0
if [ "$stamp_rc" -eq 0 ]; then
  python3 scripts/data_health_digest.py >> "$LOG" 2>&1
  digest_rc=$?
else
  echo "[$(date '+%F %T')] freshness stamp失败，跳过digest，避免发布旧摘要" >> "$LOG"
  digest_rc=75
fi
printf '[%s] stamp_rc=%s digest_rc=%s\n' "$(date '+%F %T')" "$stamp_rc" "$digest_rc" >> "$LOG"
# digest 的 1/2 是数据风险等级，不是任务执行异常。报告和前端会读取
# generated/data_health_*.md 展示真实降级；只有 stamp/digest 自身崩溃才让 cron 失败。
if [ "$stamp_rc" -ne 0 ]; then exit "$stamp_rc"; fi
if [ "$digest_rc" -gt 2 ]; then exit "$digest_rc"; fi
exit 0

