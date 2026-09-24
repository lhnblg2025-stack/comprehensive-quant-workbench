"""
self_healer — 系统自愈监控（V11 不分析股票，只分析系统自身）

监控项:
  1. 关键数据文件大小突变（vs data_health_state 快照）
  2. zt_daily_stats 关键列 NaN 占比（因子完整性代理）
  3. 最近日报/作战地图是否生成、是否过旧
  4. 涨停池历史表规模（全量 vs 被覆盖）
  5. 后台重建任务是否完成（zt_pool_history 行数阈值）

动作: 异常 → 写 generated/self_heal_{date}.json + 输出建议
      (保守模式标记: 决策层只输出观察不输出买卖建议)

用法:
  python3 -m quant_system.analysis_core.self_healer [--push]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.config import (  # noqa: E402
    MARKET_DIR, ZT_HISTORY, ZT_DAILY_STATS,
)
from quant_system.analysis_core.common import today  # noqa: E402

CST = timezone(timedelta(hours=8))
STATE_FILE = ROOT / "generated" / "data_health_state.json"
ZT_FULL_THRESHOLD = 100_000   # 全量涨停池历史行数阈值


def check(push: bool = False) -> dict:
    issues: list[dict] = []
    t = datetime.now(CST).isoformat()

    # ── 1. 文件大小突变 ──
    if STATE_FILE.exists():
        try:
            old = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            old = {}
        for name, path in [("zt_daily_stats", ZT_DAILY_STATS),
                           ("zt_pool_history", ZT_HISTORY),
                           ("fund_forces", MARKET_DIR / "fund_forces.parquet")]:
            if not path.exists():
                continue
            key = f"size::{name}"
            prev = old.get(key)
            if prev:
                chg = (path.stat().st_size - prev) / prev
                if abs(chg) > 0.35:
                    issues.append({"item": f"size::{name}", "level": "🔴",
                                   "msg": f"文件大小突变 {chg:+.0%}，疑似静默丢失/覆盖"})

    # ── 2. 天梯表关键列 NaN 占比 ──
    if ZT_DAILY_STATS.exists():
        df = pd.read_parquet(ZT_DAILY_STATS)
        for col in ["zt_cnt", "max_board", "premium", "zb_rate"]:
            if col in df.columns:
                nan_ratio = df[col].isna().mean()
                if nan_ratio > 0.3:
                    issues.append({"item": f"nan::{col}", "level": "🟠",
                                   "msg": f"{col} NaN 占比 {nan_ratio:.0%}，因子失效风险"})
    else:
        issues.append({"item": "zt_daily_stats", "level": "🔴",
                       "msg": "天梯表文件缺失，核心情绪数据不可用"})

    # ── 3. 产出物新鲜度 ──
    for name, pat in [("battle_map", "generated/battle_map_*.json"),
                      ("daily_report", "generated/short_term_daily.md")]:
        files = sorted(ROOT.glob(pat))
        if not files:
            issues.append({"item": name, "level": "🟠", "msg": "无产出文件"})
        else:
            mt = datetime.fromtimestamp(files[-1].stat().st_mtime)
            if datetime.now() - mt > timedelta(days=5):
                issues.append({"item": name, "level": "🟡",
                               "msg": f"最新产出 {mt.date()}，超过5天未更新"})

    # ── 4. 涨停池历史规模 ──
    if ZT_HISTORY.exists():
        rows = len(pd.read_parquet(ZT_HISTORY, columns=["date"]))
        if rows < ZT_FULL_THRESHOLD:
            issues.append({"item": "zt_pool_history", "level": "🟠",
                           "msg": f"仅 {rows} 行(<{ZT_FULL_THRESHOLD})，历史被覆盖，需全量重建"})
    else:
        issues.append({"item": "zt_pool_history", "level": "🔴",
                       "msg": "涨停池历史文件缺失，无法校验历史完整性"})

    # ── 汇总 ──
    level_order = {"🔴": 3, "🟠": 2, "🟡": 1}
    worst = max((level_order.get(i["level"], 0) for i in issues), default=0)
    mode = "conservative" if worst >= 3 else ("normal" if worst == 0 else "watch")
    res = {
        "date": today(), "checked_at": t,
        "issues": issues, "worst_level": worst,
        "decision_mode": mode,
        "note": "conservative=只输出观察不输出买卖建议; watch=提示人工; normal=正常",
    }
    out = ROOT / "generated" / f"self_heal_{today()}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    if push and issues:
        try:
            from quant_system.report_delivery import push_to_feishu
            push_to_feishu("\n".join(f"{i['level']} {i['msg']}" for i in issues),
                           title="🔧 系统自愈告警")
        except Exception as e:
            print(f"[self_healer] 推送失败: {e}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="系统自愈监控")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()
    r = check(push=args.push)
    print(f"决策模式: {r['decision_mode']} | 问题数: {len(r['issues'])}")
    for i in r["issues"]:
        print(f"  {i['level']} {i['msg']}")
