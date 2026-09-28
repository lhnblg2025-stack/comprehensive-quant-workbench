"""
sentiment_raw.py — QuantV6 情绪原始数据聚合
乐咕市场活跃度：上涨/下跌/平盘/涨停/跌停/活跃度，一次调用全拿到。
"""
from __future__ import annotations

from typing import Any


from quant_system.market_forecast._support.common.logger import get_logger
from quant_system.market_forecast._support.data.cache import get, put
from quant_system.market_forecast._support.data.fetcher import safe_call

log = get_logger("qv6.sentiment")


def get_market_activity() -> dict[str, Any]:
    """
    乐咕市场活跃度快照。
    返回 {rise, fall, flat, total, limit_up, limit_down, suspension, activity, stat_date}。
    失败返回空 dict。
    """
    cached = get("market_activity", 30)
    if cached is not None:
        return cached

    act = safe_call("stock_market_activity_legu")
    out: dict[str, Any] = {}
    if act is not None and len(act):
        try:
            kv = dict(zip(act["item"], act["value"]))
            rise = int(kv.get("上涨", 0))
            fall = int(kv.get("下跌", 0))
            flat = int(kv.get("平盘", 0))
            act_raw = str(kv.get("活跃度", "0")).replace("%", "").strip()
            out = {
                "rise": rise,
                "fall": fall,
                "flat": flat,
                "total": rise + fall + flat,
                "limit_up": int(kv.get("涨停", 0)),
                "limit_down": int(kv.get("跌停", 0)),
                "suspension": int(kv.get("停牌", 0)),
                "activity": float(act_raw) if act_raw else 0.0,
                "stat_date": str(kv.get("统计日期", "")),
                "ok": True,
            }
            put("market_activity", out, 30)
        except Exception as e:
            log.warning(f"乐咕活跃度解析失败: {str(e)[:80]}")
    return out


def get_limit_pool_summary() -> dict:
    """涨跌停家数（乐咕优先，东财池补充列表）。数据缺失时返回 ok=False + None 计数。"""
    act = get_market_activity()
    if not act.get("ok") or act.get("total", 0) <= 0:
        return {"limit_up": None, "limit_down": None, "ok": False}
    return {
        "limit_up": act.get("limit_up", 0),
        "limit_down": act.get("limit_down", 0),
        "ok": True,
    }


def get_limit_up_list() -> list[dict]:
    """涨停池列表（东财，失败返回空列表）。"""
    from quant_system.market_forecast._support.data.fetcher import safe_call as _sc
    df = _sc("stock_zt_pool_em", date=_today())
    if df is None or len(df) == 0:
        return []
    out = []
    for _, r in df.head(50).iterrows():
        try:
            out.append({
                "code": str(r.get("代码", "")),
                "name": str(r.get("名称", "")),
                "price": float(r.get("最新价", 0)),
                "pct": float(r.get("涨跌幅", 0)),
            })
        except Exception as e:
            log.error(f"[sentiment_raw] 操作失败: {e}", exc_info=True)
            continue
    return out


def get_limit_down_list() -> list[dict]:
    from quant_system.market_forecast._support.data.fetcher import safe_call as _sc
    df = _sc("stock_zt_pool_dtgc_em", date=_today())
    if df is None or len(df) == 0:
        return []
    out = []
    for _, r in df.head(30).iterrows():
        try:
            out.append({"code": str(r.get("代码", "")), "name": str(r.get("名称", ""))})
        except Exception as e:
            log.error(f"[sentiment_raw] 操作失败: {e}", exc_info=True)
            continue
    return out


def rise_ratio() -> float | None:
    """上涨家数占比 0~1；数据缺失返回 None（不得用 0.5 冒充真实市场）。"""
    a = get_market_activity()
    total = a.get("total", 0)
    if not a.get("ok") or total <= 0:
        return None
    return a.get("rise", 0) / total


def heat_flag() -> bool | None:
    """情绪过热标记：涨停>100 或 上涨比>85%；数据缺失返回 None。"""
    from quant_system.market_forecast._support.common.constants import HEAT_LIMIT_UP, HEAT_RISE_RATIO
    a = get_market_activity()
    if not a.get("ok") or a.get("total", 0) <= 0:
        return None
    return a.get("limit_up", 0) >= HEAT_LIMIT_UP or (a.get("rise", 0) / a["total"]) >= HEAT_RISE_RATIO


def _today() -> str:
    from quant_system.market_forecast._support.common.time_utils import fmt_date
    return fmt_date().replace("-", "")
