"""
v5_api — V5 深度模块 API 路由 (统一入口)

所有 V5 后端接口通过 /api/v5/<module> 访问。
server.py 只需加一行转发到此模块。
"""

from __future__ import annotations
import logging

import time as _time
import traceback
from datetime import datetime, timezone, timedelta
from typing import Any, Callable

CST = timezone(timedelta(hours=8))

# ── 模块缓存 ──
_cache: dict[str, tuple[float, Any]] = {}


def _cached(key: str, ttl: int, factory: Callable) -> dict:
    now = _time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] <= ttl:
        return hit[1]
    try:
        data = factory()
        _cache[key] = (now, data)
        return data
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


# ═══════════════════════════════════════
#  sentiment_factory
# ═══════════════════════════════════════

def handle_sentiment() -> dict:
    """合成情绪指数。"""
    def _fetch():
        from quant_system.sentiment_factory.composite import CompositeSentiment
        return CompositeSentiment().compute()
    return _cached("v5_sentiment", 300, _fetch)


def handle_sentiment_price() -> dict:
    def _fetch():
        from quant_system.sentiment_factory.price_sentiment import PriceSentiment
        return PriceSentiment().compute()
    return _cached("v5_sentiment_price", 300, _fetch)


def handle_sentiment_volume() -> dict:
    def _fetch():
        from quant_system.sentiment_factory.volume_sentiment import VolumeSentiment
        return VolumeSentiment().compute()
    return _cached("v5_sentiment_volume", 300, _fetch)


def handle_sentiment_funding() -> dict:
    def _fetch():
        from quant_system.sentiment_factory.funding_sentiment import FundingSentiment
        return FundingSentiment().compute()
    return _cached("v5_sentiment_funding", 600, _fetch)


def handle_sentiment_divergence() -> dict:
    def _fetch():
        from quant_system.sentiment_factory.sentiment_divergence import SentimentDivergence
        return SentimentDivergence().compute()
    return _cached("v5_sentiment_div", 300, _fetch)


# ═══════════════════════════════════════
#  market_depth
# ═══════════════════════════════════════

def handle_breadth() -> dict:
    def _fetch():
        from quant_system.market_depth.breadth_thrust import BreadthThrust
        return BreadthThrust().compute()
    return _cached("v5_breadth", 300, _fetch)


def handle_volume_divergence() -> dict:
    def _fetch():
        from quant_system.market_depth.volume_divergence import VolumeDivergence
        return VolumeDivergence().compute("sh000300")
    return _cached("v5_vol_div", 300, _fetch)


def handle_limit_up() -> dict:
    def _fetch():
        from quant_system.market_depth.limit_up_depth import LimitUpDepth
        return LimitUpDepth().compute()
    return _cached("v5_limit_up", 300, _fetch)


def handle_leader_laggard() -> dict:
    def _fetch():
        from quant_system.market_depth.leader_laggard import LeaderLaggard
        return LeaderLaggard().compute()
    return _cached("v5_leader", 600, _fetch)


# ═══════════════════════════════════════
#  cross_market
# ═══════════════════════════════════════

def handle_cross_market() -> dict:
    def _fetch():
        from quant_system.cross_market.composite import CrossMarketComposite
        return CrossMarketComposite().verify()
    return _cached("v5_cross", 600, _fetch)


def handle_bond_equity() -> dict:
    def _fetch():
        from quant_system.cross_market.bond_equity import BondEquity
        return BondEquity().compute()
    return _cached("v5_bond_eq", 1800, _fetch)


def handle_north_deep() -> dict:
    def _fetch():
        from quant_system.cross_market.north_flow_deep import NorthFlowDeep
        return NorthFlowDeep().compute()
    return _cached("v5_north_deep", 600, _fetch)


def handle_margin_deep() -> dict:
    def _fetch():
        from quant_system.cross_market.margin_deep import MarginDeep
        return MarginDeep().compute()
    return _cached("v5_margin_deep", 600, _fetch)


# ═══════════════════════════════════════
#  seasonality
# ═══════════════════════════════════════

def handle_seasonality_month() -> dict:
    def _fetch():
        from quant_system.seasonality.month_effect import MonthEffect
        return MonthEffect().compute(years=10)
    return _cached("v5_season_month", 3600, _fetch)


def handle_seasonality_holiday() -> dict:
    def _fetch():
        from quant_system.seasonality.holiday_effect import HolidayEffect
        return HolidayEffect().compute()
    return _cached("v5_season_holiday", 3600, _fetch)


def handle_seasonality_pattern() -> dict:
    def _fetch():
        from quant_system.seasonality.annual_pattern import AnnualPattern
        return AnnualPattern().compute()
    return _cached("v5_season_pattern", 3600, _fetch)


def handle_seasonality_weekday() -> dict:
    def _fetch():
        from quant_system.seasonality.weekday_effect import WeekdayEffect
        return WeekdayEffect().compute()
    return _cached("v5_season_wday", 3600, _fetch)


# ═══════════════════════════════════════
#  combined dashboard
# ═══════════════════════════════════════

def handle_v5_dashboard() -> dict:
    """V5 综合仪表盘 — 一次调用返回所有关键指标。

    覆盖: sentiment(4) / market_depth(4, 含 volume_divergence) /
    cross_market(1) / seasonality(4)。各处理器内部已通过 `_cached` 兜底异常,
    此处 try/except 仅为防御性保底, 单个模块失败不影响其余模块。
    """
    result = {
        "timestamp": datetime.now(CST).isoformat(),
        "sentiment": {},
        "market_depth": {},
        "cross_market": {},
        "seasonality": {},
    }
    try:
        result["sentiment"] = handle_sentiment()
    except Exception as e:
        logging.getLogger(__name__).error(f"[v5_api] 操作失败: {e}", exc_info=True)
    try:
        # P2-Q25-fix(M295): market_depth 补齐 volume_divergence 路由,
        # 与 docstring 承诺一致。
        result["market_depth"] = {
            "breadth": handle_breadth(),
            "volume_divergence": handle_volume_divergence(),
            "limit_up": handle_limit_up(),
            "leader_laggard": handle_leader_laggard(),
        }
    except Exception as e:
        logging.getLogger(__name__).error(f"[v5_api] 操作失败: {e}", exc_info=True)
    try:
        result["cross_market"] = handle_cross_market()
    except Exception as e:
        logging.getLogger(__name__).error(f"[v5_api] 操作失败: {e}", exc_info=True)
    try:
        # P2-Q25-fix(M295): 填充 seasonality 四个处理器(此前初始化 {} 后从未赋值,
        # docstring 声称"一次调用返回所有关键指标"与实际不符)。
        result["seasonality"] = {
            "month": handle_seasonality_month(),
            "holiday": handle_seasonality_holiday(),
            "pattern": handle_seasonality_pattern(),
            "weekday": handle_seasonality_weekday(),
        }
    except Exception as e:
        logging.getLogger(__name__).error(f"[v5_api] 操作失败: {e}", exc_info=True)
    return result


# ═══════════════════════════════════════
#  Router
# ═══════════════════════════════════════

ROUTES: dict[str, Callable] = {
    # sentiment_factory
    "sentiment": handle_sentiment,
    "sentiment/price": handle_sentiment_price,
    "sentiment/volume": handle_sentiment_volume,
    "sentiment/funding": handle_sentiment_funding,
    "sentiment/divergence": handle_sentiment_divergence,
    # market_depth
    "depth/breadth": handle_breadth,
    "depth/volume_divergence": handle_volume_divergence,
    "depth/limit_up": handle_limit_up,
    "depth/leader_laggard": handle_leader_laggard,
    # cross_market
    "cross": handle_cross_market,
    "cross/bond_equity": handle_bond_equity,
    "cross/north_deep": handle_north_deep,
    "cross/margin_deep": handle_margin_deep,
    # seasonality
    "season/month": handle_seasonality_month,
    "season/holiday": handle_seasonality_holiday,
    "season/pattern": handle_seasonality_pattern,
    "season/weekday": handle_seasonality_weekday,
    # combined
    "dashboard": handle_v5_dashboard,
}


def route(api_path: str) -> dict | None:
    """主入口：server.py 调用此函数。

    Args:
        api_path: 如 "/api/v5/sentiment" 或 "/api/v5/dashboard"

    Returns:
        dict (含 ok + data 字段) 或 None (路由不存在)
    """
    # 提取路由键: /api/v5/sentiment → sentiment
    parts = api_path.strip("/").split("/")
    if len(parts) < 3 or parts[0] != "api" or parts[1] != "v5":
        return None

    route_key = "/".join(parts[2:])  # e.g. "sentiment/price"
    handler = ROUTES.get(route_key)
    if handler is None:
        return None

    try:
        data = handler()
        return {"ok": True, "data": data}
    except Exception as e:
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}
