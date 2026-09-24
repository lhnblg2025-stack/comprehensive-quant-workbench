#!/usr/bin/env bash
# 交易日复盘编排 wrapper：保留每一步退出码，避免尾部 echo 覆盖失败。
set -u
ROOT="${PROJECT_ROOT}"
cd "$ROOT"
mkdir -p generated/logs
/usr/bin/env python3 scripts/data_contract_registry.py --check >> generated/logs/data_contract.log 2>&1
gate_rc=$?
/usr/bin/env python3 scripts/daily_review_chain.py >> generated/logs/review_push.log 2>&1
review_rc=$?
printf '[%s] data_contract_rc=%s review_rc=%s\n' "$(date '+%F %T')" "$gate_rc" "$review_rc" >> generated/logs/review_push.log
# 契约门禁仅作为降级记录，日报生成必须保留真实退出码。
exit "$review_rc"
