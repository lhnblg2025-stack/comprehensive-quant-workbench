#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全基座新鲜度门控（2026-08-21 —— 目标点名的"建全基座新鲜度门控维度"）

扫描 data_warehouse 全部目录 → 每域最新日期/滞后天数 → freshness 评级:
  fresh(≤2日) / stale(3-5日) / expired(≥6日)
汇入决策链: 陈旧基座标注降权（decision 侧提示），前端看板全景展示。

小函数:
 1. scan_all()      → 全目录新鲜度表
 2. gate_scores()   → 各域 fresh/stale/expired 统计 + 陈旧清单
 3. freshness_md()  → Markdown 段（全基座门控图）
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DW = ROOT / "data_warehouse"
CST = timezone(timedelta(hours=8))

# 关键基座的日期列名（不同结构差异）。上榜日=龙虎榜, created_at=预测账本,
# 日期/date/trade_date/报告期/ts 覆盖绝大多数。时间列（首次封板时间等）不参与。
_DATE_COLS = ["date", "日期", "trade_date", "报告期", "ts", "上榜日", "created_at", "交易日", "交易日期"]


def _date_from_filename(name: str):
    """文件名中的 YYYYMMDD（如 lhb_20260812.parquet / ztpool_20260814.parquet）。

    某些基座把业务日期只写在文件名、文件内没有日期列（涨停池历史即如此），
    不能用 mtime 冒充行情日期，但文件名日期是真实业务日期，可以安全使用。
    """
    import re as _re
    match = _re.search(r"(20\d{2})(\d{2})(\d{2})", name)
    if not match:
        return None
    year, month, day = match.groups()
    try:
        # 与 pd.to_datetime 一致，返回 naive datetime，避免混用 aware/naive 比较。
        return datetime(int(year), int(month), int(day))
    except ValueError:
        return None


def _dir_latest(d: str) -> tuple[str, int]:
    """扫描目录全部 parquet 的业务日期；无日期列不得用 mtime 冒充行情日期。"""
    import pandas as pd
    dp = DW / d
    if not dp.exists():
        return "", -1
    files = list(dp.glob("*.parquet"))
    if not files:
        return "", -1
    # 财务域可有五千余个股票文件。新鲜度只需最近业务日期，按 mtime
    # 取最近写入的候选并只读取 schema/日期列，避免逐个全表扫描。
    files = sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)[:128]
    latest_ts = None
    for f in files:
        # 1) 文件内日期列优先（业务日期在列里）。
        try:
            try:
                import pyarrow.parquet as pq
                schema = pq.read_schema(f).names
            except Exception:
                schema = pd.read_parquet(f, columns=[]).columns
            candidates = [c for c in schema if any(k in str(c).lower() for k in _DATE_COLS)]
            if candidates:
                df = pd.read_parquet(f, columns=[candidates[0]])
                values = pd.to_datetime(df[candidates[0]], errors="coerce").dropna()
                if not values.empty:
                    ts = values.max()
                    if latest_ts is None or ts > latest_ts:
                        latest_ts = ts
                    continue
        except Exception:
            pass
        # 2) 无日期列 → 文件名日期兜底（仅限文件名显式带 YYYYMMDD 的业务日期）。
        ts = _date_from_filename(f.name)
        if ts is not None and (latest_ts is None or ts > latest_ts):
            latest_ts = ts
    if latest_ts is None:
        return "", -1
    # 未来日期防护：真实行情不可能晚于今天；出现未来时间戳通常是采集端
    # 时区/时钟 bug。这里钳制到"今天"，避免把污染数据报成"最新 0 滞后"。
    if hasattr(latest_ts, "tzinfo") and latest_ts.tzinfo is not None:
        latest_ts = latest_ts.tz_localize(None)
    today = datetime.now(CST).replace(tzinfo=None)
    if latest_ts > today:
        latest_ts = today
    # 日频数据按A股交易日计算滞后，周末/节假日不凭自然日误报 stale。
    try:
        from quant_system.market_clock import get_trade_calendar
        calendar = sorted(get_trade_calendar())
        latest_day = latest_ts.strftime("%Y-%m-%d")
        today_day = datetime.now(CST).strftime("%Y-%m-%d")
        lag = max(0, sum(latest_day < x <= today_day for x in calendar))
    except Exception:
        today = datetime.now(CST).replace(tzinfo=None)
        lag = max(0, (today - latest_ts).days)
    return latest_ts.strftime("%Y-%m-%d"), lag


def scan_all() -> dict:
    """扫描全部数据基座目录 → {目录: {latest, lag, level}}。"""
    import os
    out = {}
    for d in sorted(os.listdir(DW)):
        if not (DW / d).is_dir():
            continue
        latest, lag = _dir_latest(d)
        if lag < 0:
            # 缺数据与陈旧数据分开表达；None 不会被 UI 误读成“滞后-1天”。
            out[d] = {"latest": "", "lag": None, "level": "no_data", "cycle": "?"}
            continue
        # 区分周期: 季频(财报/持仓150d) / 月频(宏观60d) / 日频(2d)
        _QUARTERLY = {"financial", "quarterly", "valuation", "cninfo"}
        _MONTHLY = {"macro", "stock", "oneoff"}
        if d in _QUARTERLY:
            thr_fresh, thr_stale, cyc = 150, 200, "季度"
        elif d in _MONTHLY:
            thr_fresh, thr_stale, cyc = 60, 90, "月"
        else:
            thr_fresh, thr_stale, cyc = 2, 5, "日"
        level = "fresh" if lag <= thr_fresh else ("stale" if lag <= thr_stale else "expired")
        out[d] = {"latest": latest, "lag": lag, "level": level, "cycle": cyc}
    return out


def gate_scores() -> dict:
    """门控汇总: 各评级计数 + 陈旧(非fresh)清单。"""
    scan = scan_all()
    fresh = sum(1 for v in scan.values() if v["level"] == "fresh")
    stale = sum(1 for v in scan.values() if v["level"] == "stale")
    expired = sum(1 for v in scan.values() if v["level"] == "expired")
    no_data = sum(1 for v in scan.values() if v["level"] == "no_data")
    def _items(level: str) -> list[dict]:
        return [{"dir": k, "lag": v["lag"], "latest": v["latest"], "cycle": v["cycle"]}
                for k, v in scan.items() if v["level"] == level]

    stale_items = sorted(_items("stale"), key=lambda x: -(x["lag"] or 0))
    expired_items = sorted(_items("expired"), key=lambda x: -(x["lag"] or 0))
    missing_items = _items("no_data")
    # 综合指数 0-100（全 fresh=100；no_data 计 0 分）
    total = len(scan) or 1
    score = round((fresh / total) * 100)
    return {
        "total": len(scan), "fresh": fresh, "stale": stale, "expired": expired,
        "no_data": no_data, "score": score,
        "stale_items": stale_items, "expired_items": expired_items,
        "missing_items": missing_items,
        "not_fresh": (expired_items + stale_items + missing_items)[:10],
        "scan": scan,
    }


def freshness_md() -> str:
    """全基座门控 → Markdown。"""
    g = gate_scores()
    L = ["## 🗃️ 数据基座新鲜度门控"]
    L.append(f"- 覆盖 **{g['total']}** 基座: fresh {g['fresh']} · stale {g['stale']} · expired {g['expired']} | 健康度 **{g['score']}/100**")
    nf = g.get("not_fresh") or []
    if g["total"] == 0:
        L.append("- ⚠️ 无任何基座数据（data_warehouse 目录缺失/空），不得判「新鲜」")
    elif nf:
        L.append("- ⚠️ 陈旧基座（决策降权）:")
        for x in nf[:8]:
            L.append(f"  - {x['dir']}: 最新 {x['latest']}（滞后 {x['lag']} 天）")
    else:
        L.append("- 全部基座新鲜 ✅")
    return "\n".join(L)


if __name__ == "__main__":
    import sys as _s
    try:
        _s.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    print(freshness_md())