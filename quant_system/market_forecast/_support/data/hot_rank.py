"""
hot_rank.py — QuantV6 热股热搜
百度热搜（稳定）主源，东财热榜（反爬时）降级。
输出等权涨跌幅 + top 名单。
"""
from __future__ import annotations

from typing import Any


from quant_system.market_forecast._support.common.logger import get_logger
from quant_system.market_forecast._support.data.cache import get, put
from quant_system.market_forecast._support.data.fetcher import safe_call

log = get_logger("qv6.hot")


def get_hot_stocks_baidu() -> list[dict]:
    """百度股市热搜（A股）。返回 [{name, pct, heat}]。"""
    df = safe_call("stock_hot_search_baidu", symbol="A股")
    if df is None or len(df) == 0:
        return []
    out = []
    for _, r in df.iterrows():
        try:
            name_col = "名称/代码" if "名称/代码" in df.columns else df.columns[0]
            pct_col = "涨跌幅" if "涨跌幅" in df.columns else None
            heat_col = "综合热度" if "综合热度" in df.columns else None
            pct_raw = str(r.get(pct_col, "0")) if pct_col else "0"
            pct = float(pct_raw.replace("%", "").replace("+", ""))
            out.append({
                "name": str(r.get(name_col, "")),
                "pct": pct,
                "heat": int(r.get(heat_col, 0)) if heat_col else 0,
            })
        except Exception as e:
            log.error(f"[hot_rank] 操作失败: {e}", exc_info=True)
            continue
    return out


def get_hot_stocks_em() -> list[dict]:
    """东财人气榜（可能反爬）。返回 [{code, name, pct}]。"""
    df = safe_call("stock_hot_rank_em")
    if df is None or len(df) == 0:
        return []
    out = []
    for _, r in df.head(20).iterrows():
        try:
            out.append({
                "code": str(r.get("代码", "")),
                "name": str(r.get("名称", "")),
                "pct": float(r.get("涨跌幅", 0) or 0),
            })
        except Exception as e:
            log.error(f"[hot_rank] 操作失败: {e}", exc_info=True)
            continue
    return out


def get_hot_summary() -> dict[str, Any]:
    """
    热股汇总。返回 {equal_weight_return, top_names, source, count}。
    百度失败降级东财；两者都失败返回空 dict。
    """
    cached = get("hot_summary", 600)
    if cached is not None:
        return cached

    out: dict[str, Any] = {}
    items = get_hot_stocks_baidu()
    source = "baidu"
    if not items:
        em = get_hot_stocks_em()
        if em:
            items = [{"name": e["name"], "pct": e["pct"]} for e in em]
            source = "em"

    if items:
        pcts = [i["pct"] for i in items if i["pct"] is not None]
        out = {
            "equal_weight_return": round(sum(pcts) / len(pcts), 2) if pcts else None,
            "top_names": [i["name"] for i in items[:10]],
            "source": source,
            "count": len(items),
        }
        put("hot_summary", out, 600)
    return out


def hot_equal_weight_return() -> float | None:
    """热股等权涨跌幅（情绪指标之一）。"""
    s = get_hot_summary()
    return s.get("equal_weight_return")
