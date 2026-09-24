"""
sector.py — QuantV6 行业板块
新浪行业 spot（49行业带涨跌幅）主源，板块成分降级。
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from quant_system.market_forecast._support.common.logger import get_logger
from quant_system.market_forecast._support.data.cache import get, put
from quant_system.market_forecast._support.data.fetcher import safe_call

log = get_logger("qv6.sector")


def get_sector_spot() -> pd.DataFrame:
    """新浪行业板块行情。列: label/板块/公司家数/涨跌幅/总成交额。失败返回空。"""
    df = safe_call("stock_sector_spot", indicator="新浪行业")
    return df if df is not None else pd.DataFrame()


def get_sector_ranking(top_n: int = 5, bottom_n: int = 5) -> dict[str, Any]:
    """
    行业强弱排名。返回 {top: [{n, p}], bottom: [{n, p}]}。
    """
    cached = get("sector_ranking", 600)
    if cached is not None:
        return cached

    out: dict[str, Any] = {"top": [], "bottom": []}
    df = get_sector_spot()
    if df is None or len(df) == 0:
        return out
    try:
        pct_col = "涨跌幅" if "涨跌幅" in df.columns else None
        name_col = "板块" if "板块" in df.columns else df.columns[1]
        if not pct_col:
            return out
        df[pct_col] = pd.to_numeric(df[pct_col], errors="coerce")
        df = df.dropna(subset=[pct_col])
        tp = df.nlargest(top_n, pct_col)
        bt = df.nsmallest(bottom_n, pct_col)
        out["top"] = [{"n": str(r[name_col]), "p": round(float(r[pct_col]), 2)}
                      for _, r in tp.iterrows()]
        out["bottom"] = [{"n": str(r[name_col]), "p": round(float(r[pct_col]), 2)}
                         for _, r in bt.iterrows()]
        put("sector_ranking", out, 600)
    except Exception as e:
        log.warning(f"行业排名解析失败: {str(e)[:80]}")
    return out


def get_sector_members(sector_name: str) -> list[str]:
    """板块成分股（东财，失败返回空列表）。"""
    df = safe_call("stock_board_industry_cons_em", symbol=sector_name)
    if df is None or len(df) == 0:
        return []
    code_col = "代码" if "代码" in df.columns else df.columns[0]
    return [str(c).zfill(6) for c in df[code_col].tolist()]


def strongest_sectors(n: int = 3) -> list[str]:
    """最强行业名称列表。"""
    r = get_sector_ranking(top_n=n)
    return [s["n"] for s in r.get("top", [])]
