"""
sentiment.py — QuantV6 情绪因子库
换手异动/连板高度/资金流类因子。输入 {code: DataFrame} + 可选情绪表，输出截面 Series。
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。
"""
from __future__ import annotations

import pandas as pd


def turnover_anomaly(data: dict[str, pd.DataFrame], window: int = 20) -> pd.Series:
    """换手率异动：当日量 / 20日均量（情绪活跃度，方向=1 温和放量偏好）。
    D4收敛登记: 独特因子保留
    """
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < window + 1 or "volume" not in df.columns:
            continue
        v = df["volume"].astype(float)
        avg = v.tail(window + 1).head(window).mean()
        out[sym] = float(v.iloc[-1] / avg) if avg else 0.0
    return pd.Series(out, dtype=float)


def limit_up_streak(data: dict[str, pd.DataFrame], limit_map: dict | None = None) -> pd.Series:
    """连板高度（来自涨跌停池映射：{code: 连板数}）。
    D4收敛登记: 独特因子保留
    """
    limit_map = limit_map or {}
    return pd.Series({sym: float(limit_map.get(sym, 0)) for sym in data}, dtype=float)


def money_flow_strength(data: dict[str, pd.DataFrame], flow_map: dict | None = None) -> pd.Series:
    """主力资金净流入（{code: 净流入额}，方向=1）。
    D4收敛登记: 独特因子保留
    """
    flow_map = flow_map or {}
    return pd.Series({sym: float(flow_map.get(sym, 0.0)) for sym in data}, dtype=float)


def margin_buy_ratio(data: dict[str, pd.DataFrame], margin_map: dict | None = None) -> pd.Series:
    """融资买入占比（{code: 占比}，方向=1 温和杠杆偏好）。
    D4收敛登记: 独特因子保留
    """
    margin_map = margin_map or {}
    return pd.Series({sym: float(margin_map.get(sym, 0.0)) for sym in data}, dtype=float)


def northbound_change(data: dict[str, pd.DataFrame], north_map: dict | None = None) -> pd.Series:
    """北向持仓变化（{code: 变化%}，方向=1）。
    D4收敛登记: 独特因子保留
    """
    north_map = north_map or {}
    return pd.Series({sym: float(north_map.get(sym, 0.0)) for sym in data}, dtype=float)


def price_volume_heat(data: dict[str, pd.DataFrame], window: int = 5) -> pd.Series:
    """量价热度：近5日涨幅 × 量比（动量+量能共振）。
    D4收敛登记: 独特因子保留
    """
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < window + 5 or "close" not in df.columns:
            continue
        close = df["close"].astype(float)
        mom = float(close.iloc[-1] / close.iloc[-window - 1] - 1) * 100
        v = df["volume"].astype(float) if "volume" in df.columns else None
        ratio = 1.0
        if v is not None:
            avg = v.tail(window + 1).head(window).mean()
            ratio = float(v.iloc[-1] / avg) if avg else 1.0
        out[sym] = mom * min(ratio, 3.0)
    return pd.Series(out, dtype=float)
