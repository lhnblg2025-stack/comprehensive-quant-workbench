#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""盘中监控 · 轻量版（2026-08-21 全流程线①）

原则: 只读现有本地数据(涨停池/指数/资金)，全部小函数化、秒级、
     任何单一数据缺失自动降级，不阻塞不挂死。

小函数:
 1. index_heat()        → 主要指数红绿/涨跌幅快照（读 index_daily parquet 最新两日）
 2. zt_heat()           → 涨停强度（涨停/炸板/连板/最高板，读 zt_pool_em_daily）
 3. mainline_flash()    → 主线异动（涨停行业 Top + 连板龙头）
 4. fund_flash()        → 资金快照（fund_forces 最新合力）
 5. risk_flash()        → 风险提示（炸板率高/跌停多/断层）
 6. monitor_snapshot()  → 六合一快照（供前端轮询 /api/intraday）

所有函数 import 超时保护: each 独立 try/except + 短 read (tail N)。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
CST = timezone(timedelta(hours=8))

MK = ROOT / "data_warehouse" / "market"


def _completed_trade_day() -> str:
    try:
        from quant_system.market_clock import latest_completed_trading_day
        return latest_completed_trading_day().isoformat()
    except Exception:
        return datetime.now(CST).strftime("%Y-%m-%d")


def _latest_day_frame(df, column: str = "date", cutoff: str | None = None):
    """Select latest completed trading day, never after an explicit replay date."""
    import pandas as pd
    if df is None or df.empty or column not in df.columns:
        return df, None
    parsed = pd.to_datetime(df[column], errors="coerce")
    day = parsed.dt.strftime("%Y-%m-%d")
    cutoff = cutoff or _completed_trade_day()
    eligible = df[day.notna() & (day <= cutoff)].copy()
    if eligible.empty:
        return eligible, None
    eligible["_day"] = pd.to_datetime(eligible[column], errors="coerce").dt.strftime("%Y-%m-%d")
    latest = eligible["_day"].max()
    return eligible[eligible["_day"] == latest].drop(columns=["_day"]), latest


def _read_parquet(rel: str, tail: int | None = None):
    """极简 parquet 读（尾部 n 行）。"""
    import pandas as pd
    p = MK / rel
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if tail and len(df) > tail:
        df = df.tail(tail)
    return df


def _idx_heat_per(df, name: str, close_col: str = "close"):
    try:
        c = df[close_col].astype(float)
        last, prev = c.iloc[-1], c.iloc[-2]
        chg = (last / prev - 1) * 100
        return {"name": name, "last": round(float(last), 2), "chg_pct": round(float(chg), 2),
                "dir": "up" if chg > 0 else ("down" if chg < 0 else "flat")}
    except Exception:  # noqa: BLE001
        return None


def _cutoff_frame(df, cutoff: str | None = None, column: str = "date"):
    if df is None or df.empty or not cutoff or column not in df.columns:
        return df
    import pandas as pd
    parsed = pd.to_datetime(df[column], errors="coerce")
    return df[parsed.dt.strftime("%Y-%m-%d") <= cutoff].copy()


def index_heat(cutoff: str | None = None) -> list[dict]:
    """主要指数红绿（沪深300/中证500/创业板/科创50/上证50/中证1000）。"""
    out = []
    for nm in ("沪深300", "中证500", "创业板指", "科创50", "上证50", "中证1000"):
        try:
            df = _cutoff_frame(_read_parquet(f"index_daily_{nm}.parquet"), cutoff)
            h = _idx_heat_per(df, nm) if df is not None and len(df) >= 2 else None
            if h:
                out.append(h)
        except Exception:  # noqa: BLE001
            continue
    return out


def zt_heat(cutoff: str | None = None) -> dict:
    """涨停强度：涨停/炸板/跌停/连板/最高板。"""
    try:
        df = _read_parquet("zt_pool_em_daily.parquet")
        if df is None:
            return {"error": "涨停池缺失"}
        day, latest_day = _latest_day_frame(df, cutoff=cutoff)
        if day.empty:

            return {"error": "涨停池当日空"}
        import pandas as pd
        is_zt = day["is_zt"].fillna(False).astype(bool) if "is_zt" in day.columns else (day.get("pct_chg", 0) > 9.8 if "pct_chg" in day.columns else pd.Series(True, index=day.index))
        zt_n = int(is_zt.sum())
        zb_n = int((~is_zt).sum()) if "is_zt" in day.columns else 0
        dt_n = int((day["is_dt"].fillna(False)).sum()) if "is_dt" in day.columns else 0
        bc = day["board_count"].fillna(0) if "board_count" in day.columns else pd.Series(0, index=day.index)
        max_b = int(bc.max()) if len(bc) else 0
        lian = int((bc >= 2).sum())
        tt = zt_n + zb_n
        zb_rate = zb_n / tt if tt else 0
        return {"zt": zt_n, "zb": zb_n, "dt": dt_n, "max_board": max_b,
                "lianban": lian, "zb_rate": round(zb_rate, 2),
                "date": latest_day}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:80]}


def mainline_flash(top_n: int = 6, cutoff: str | None = None) -> list[dict]:
    """主线异动：概念/题材扩散 Top；申万行业只作归属信息。"""
    try:
        import pandas as pd
        zt = _read_parquet("zt_pool_em_daily.parquet")
        members_path = ROOT / "data_warehouse" / "classification" / "concept_member.parquet"
        boards_path = ROOT / "data_warehouse" / "classification" / "concept_board.parquet"
        if zt is None or zt.empty or not members_path.exists():
            return []
        members = pd.read_parquet(members_path, columns=["concept", "concept_name", "code"])
        board_names = {}
        if boards_path.exists():
            boards = pd.read_parquet(boards_path, columns=["board_code", "board_name"])
            board_names = dict(zip(boards["board_code"].astype(str), boards["board_name"].astype(str)))
        day, latest_day = _latest_day_frame(zt, cutoff=cutoff)
        if "is_zt" in day.columns:
            day = day[day["is_zt"].fillna(False).astype(bool)]
        member_map = {}
        for row in members.to_dict("records"):
            code = str(row.get("code") or "").zfill(6)
            concept_id = str(row.get("concept") or "")
            label = board_names.get(concept_id, str(row.get("concept_name") or "")).strip()
            if code and label:
                member_map.setdefault(code, set()).add(label)
        grouped = {}
        for row in day.to_dict("records"):
            code = str(row.get("code") or "").zfill(6)
            for concept in member_map.get(code, set()):
                item = grouped.setdefault(concept, {"name": concept, "concept": concept,
                                                     "taxonomy": "concept", "zt": 0,
                                                     "max_board": 0, "codes": set()})
                if code not in item["codes"]:
                    item["codes"].add(code)
                    item["zt"] += 1
                item["max_board"] = max(item["max_board"], int(float(row.get("board_count") or 0)))
        out = []
        for item in grouped.values():
            item.pop("codes", None)
            if item["zt"] >= 2:
                item["level"] = "主线候选"
                out.append(item)
        return sorted(out, key=lambda x: (-x["max_board"], -x["zt"]))[:top_n]
    except Exception:  # noqa: BLE001
        return []


def fund_flash(cutoff: str | None = None) -> dict:
    """资金快照：按最近完成交易日选择 fund_forces 合力。"""
    try:
        df = _read_parquet("fund_forces.parquet")
        if df is None or not len(df):
            return {"error": "资金数据缺失"}
        day, latest_day = _latest_day_frame(df, cutoff=cutoff)
        if day.empty:
            return {"error": "资金数据无已完成交易日记录"}
        r = day.sort_values("date").iloc[-1]
        fi = float(r.get("force_index", 0) or 0)
        youzi = float(r.get("youzi_net", 0) or 0)
        return {"force_index": round(fi), "youzi_net_yi": round(youzi / 1e8, 1),
                "date": latest_day}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:60]}


def risk_flash(cutoff: str | None = None) -> list[str]:
    """风险提示（轻量规则）。"""
    risks = []
    zh = zt_heat(cutoff=cutoff)
    if "error" not in zh:
        if zh.get("zb_rate", 0) >= 0.35:
            risks.append(f"炸板率{zh['zb_rate']:.0%} 偏高，追涨谨慎")
        if zh.get("dt", 0) > zh.get("zt", 0):
            risks.append(f"跌停{zh['dt']}>涨停{zh['zt']}，情绪恶化")
        if zh.get("max_board", 9) <= 2:
            risks.append("最高板≤2，连板高度受限")
    return risks


def monitor_snapshot(as_of: str | None = None) -> dict:
    """六合一快照；历史回放时所有本地源都截断到 as_of。"""
    from datetime import datetime as _dt
    cutoff = as_of or _completed_trade_day()
    now = datetime.now(CST)
    idx = index_heat(cutoff=cutoff)
    zh = zt_heat(cutoff=cutoff)
    snapshot = {
        "ts": now.strftime("%H:%M:%S"),
        "date": cutoff,
        "data_as_of": cutoff,
        "indices": idx,
        "zt": zh,
        "mainlines": mainline_flash(cutoff=cutoff),
        "fund": fund_flash(cutoff=cutoff),
        "risks": risk_flash(cutoff=cutoff),
        "up_n": sum(1 for i in idx if i and i.get("dir") == "up"),
        "down_n": sum(1 for i in idx if i and i.get("dir") == "down"),
    }
    # Keep an append-only intraday trail; downstream decay needs real windows,
    # while the API response remains unchanged for existing callers.
    out_dir = ROOT / "generated" / "intraday_snapshots"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{now:%Y-%m-%d}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(snapshot, ensure_ascii=False) + "\\n")
    except OSError:
        # Monitoring must still serve the current snapshot if persistence fails.
        pass
    return snapshot


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    s = monitor_snapshot()
    print(json.dumps(s, ensure_ascii=False, indent=1))