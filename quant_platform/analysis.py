#!/usr/bin/env python3
"""platform.analysis — 统一分析服务层（骨架收口 P1）

因子扫描 / 极端预警 / 复盘 / AI 决策的统一入口。
内部路由到 quant_system 各引擎（主）与 quant_v6（V8 体系）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))


# ── 因子 / 极端预警 ────────────────────────────────

def factor_scan(symbol: str, **kw) -> dict:
    """多因子极端扫描。返回 {score, level, extremes, factors, ...}"""
    sys.path.insert(0, str(WORKSPACE / "scripts"))
    from factor_extreme_alert import compute_factors
    return compute_factors(symbol, **kw)


def extreme_alerts(symbols: list[str], **kw) -> list[dict]:
    """批量极端预警。"""
    sys.path.insert(0, str(WORKSPACE / "scripts"))
    from factor_extreme_alert import compute_factors
    out = []
    for s in symbols:
        try:
            r = compute_factors(s, **kw)
            if r.get("level") in ("危险", "警示"):
                out.append({"symbol": s, **r})
        except Exception:
            continue
    return out


# ── 个股全景档案（数据融合）────────────────────────

def stock_profile(symbol: str) -> dict:
    """全景档案：行情+估值+财务+资金+事件融合。"""
    sys.path.insert(0, str(WORKSPACE / "quant_web"))
    from stock_analysis import build_stock_profile
    return build_stock_profile(symbol)


def ai_advice(symbol: str, model: str = "deepseek/deepseek-v4-flash") -> dict:
    """AI 决策建议。"""
    sys.path.insert(0, str(WORKSPACE / "quant_web"))
    from stock_analysis import _make_advice, build_stock_profile
    profile = build_stock_profile(symbol)
    return _make_advice(profile)


# ── 复盘 ──────────────────────────────────────────

def daily_review(brief: bool = False) -> str:
    """当日复盘（涨停/连板/龙虎榜/行业）。"""
    from quant_system.close_review import generate_review
    return generate_review(brief=brief)


# ── 市场监控 ──────────────────────────────────────

def market_scan(watchlist: list[str] | None = None) -> list[dict]:
    """盘中扫描（价格/量比/52周位置预警）。"""
    from quant_system.watchlist import fetch_quotes
    return fetch_quotes(watchlist or None)


# ── 宏观快照 ──────────────────────────────────────

def macro_snapshot() -> dict:
    """PMI/M2/SHIBOR/LPR 宏观快照。"""
    sys.path.insert(0, str(WORKSPACE / "quant_web"))
    from stock_analysis import macro_snapshot as _ms
    return _ms()
