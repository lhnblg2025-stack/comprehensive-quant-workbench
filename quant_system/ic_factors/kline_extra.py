#!/usr/bin/env python3
"""kline_extra.py — K线横截面因子扩充 (V12.3 因子扩充第二批)

参考: WorldQuant 101 Alpha 精简(日线可算部分) + finhack/HFT 微观结构近似 +
intro_quant_finance 风险调整动量。

输入: _prep(df) 的 dict {close, open, high, low, volume, amount, turnover}(date-index)
输出: dict[str, pd.Series] 每个因子一条序列(与 _zoo_series_all 同构, 并入其输出)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def kline_extra_series(s: dict[str, pd.Series]) -> dict[str, pd.Series]:
    c = s["close"]
    o = s.get("open", c)
    h = s.get("high", c)
    l = s.get("low", c)
    v = s.get("volume", pd.Series(np.nan, index=c.index))
    a = s.get("amount", pd.Series(np.nan, index=c.index))
    r = c.pct_change()
    out: dict[str, pd.Series] = {}

    # ── 量价结构 ──
    # vwap_bias: 价偏离 VWAP(成交额/成交量近似)
    vwap = (a / v.replace(0, np.nan)) if a.notna().any() else pd.Series(np.nan, index=c.index)
    out["vwap_bias"] = (c / vwap - 1) * 100
    # amt_trend: 成交额 20 日趋势(当前/20日均量-1)
    amt20 = a.rolling(20).mean()
    out["amt_trend"] = (a / amt20 - 1) * 100
    # volume_std_20: 成交量 20 日波动(变异系数)
    v20 = v.rolling(20)
    out["volume_std_20"] = v20.std() / v20.mean().replace(0, np.nan)
    # high_low_range_20: 20 日振幅均值(ATR 简化)
    out["high_low_range_20"] = ((h - l) / c).rolling(20).mean() * 100
    # gap_up_5d: 5 日跳空高开次数
    gap = (o / c.shift(1) - 1) > 0.005
    out["gap_up_5d"] = gap.rolling(5).sum()

    # ── 动量增强 ──
    # mom_risk_adj: 风险调整动量(mom20 / vol20)
    vol20 = r.rolling(20).std()
    out["mom_risk_adj"] = (c.pct_change(20)) / vol20.replace(0, np.nan)
    # res_mom_approx: 残差动量近似(动量偏离自身60日均值)
    mom20 = c.pct_change(20)
    out["res_mom_approx"] = mom20 - mom20.rolling(60).mean()
    # wma_mom: 加权移动平均动量(近端权重大)
    w = np.arange(1, 21, dtype=float)
    wma_now = c.rolling(20).apply(lambda x: float(np.dot(x, w) / w.sum()), raw=True)
    wma_prev = c.shift(5).rolling(20).apply(lambda x: float(np.dot(x, w) / w.sum()), raw=True)
    out["wma_mom"] = (wma_now / wma_prev - 1) * 100

    # ── 收益分布 ──
    out["skew_20"] = r.rolling(20).skew()
    out["kurt_20"] = r.rolling(20).kurt()

    # ── 微观结构近似 (HFT 借鉴) ──
    # ofi_approx: 主动买卖近似(收盘-开盘符号 5 日累计)
    out["ofi_approx"] = np.sign(c - o).rolling(5).sum()
    # amihud_5d: Amihud 非流动性 5 日均值(|ret|/amount)
    amihud = r.abs() / a.replace(0, np.nan)
    out["amihud_5d"] = amihud.rolling(5).mean()
    # roll_spread: Roll 价差近似 2*sqrt(-cov(dc, dc_prev)) (负协方差时)
    dc = c.diff()
    cov_neg = (dc * dc.shift(1)).rolling(20).mean()
    out["roll_spread"] = np.where(cov_neg < 0, 2 * np.sqrt(-cov_neg), np.nan)
    out["roll_spread"] = pd.Series(out["roll_spread"], index=c.index)

    # ── Alpha101 精简 ──
    # alpha_rank_cm: close/ma20 偏离(rank 简化)
    out["alpha_rank_cm"] = (c / c.rolling(20).mean() - 1) * 100
    # alpha_corr_cv: corr(close, volume, 10) 滚动相关
    out["alpha_corr_cv"] = c.rolling(10).corr(v)
    # alpha_tsrank_20: close 在 20 日窗口的百分位(ts_rank)
    def _tsrank(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if len(x) < 2 or not np.isfinite(x).all():
            return np.nan
        return float((x < x[-1]).mean())
    out["alpha_tsrank_20"] = c.rolling(20).apply(_tsrank, raw=True)
    # alpha_delta_vol_5: volume 5 日变化
    out["alpha_delta_vol_5"] = v.pct_change(5) * 100
    # candle_strength: K线强度((c-o)/(h-l) 20日均值)
    rng = (h - l).replace(0, np.nan)
    out["candle_strength"] = ((c - o) / rng).rolling(20).mean()

    return out


# 新增因子清单(供 ZOO_FACTOR_NAMES 扩展)
KLINE_EXTRA_FACTORS = [
    "vwap_bias", "amt_trend", "volume_std_20", "high_low_range_20", "gap_up_5d",
    "mom_risk_adj", "res_mom_approx", "wma_mom", "skew_20", "kurt_20",
    "ofi_approx", "amihud_5d", "roll_spread",
    "alpha_rank_cm", "alpha_corr_cv", "alpha_tsrank_20", "alpha_delta_vol_5",
    "candle_strength",
]


# ── V12.3 审计: 将 K线横截面因子注册进 registry 元数据 ──
# 之前这些因子仅出现在 ic_vectorized ZOO_FACTOR_NAMES / IC CSV,
# 却不在 registry, 导致方向校准 validate_calibrate_direction 报「not_in_registry」。
# handler 接收 data dict（含 "series" = _prep 输出），经 kline_extra_series 计算；
# 输入不满足(无 series)时返回空序列, 仅元数据(方向/分类)供校准与复合层消费。
# 注: 各因子定义于 kline_extra_series(s: dict[str, pd.Series]) 中, s["close"] 必须存在。
_KLINE_EXTRA_CATEGORY = {
    "vwap_bias": "liquidity", "amt_trend": "liquidity", "volume_std_20": "volatility",
    "high_low_range_20": "volatility", "gap_up_5d": "sentiment",
    "mom_risk_adj": "momentum", "res_mom_approx": "momentum", "wma_mom": "momentum",
    "skew_20": "volatility", "kurt_20": "volatility",
    "ofi_approx": "flow", "amihud_5d": "liquidity", "roll_spread": "liquidity",
    "alpha_rank_cm": "momentum", "alpha_corr_cv": "liquidity",
    "alpha_tsrank_20": "momentum", "alpha_delta_vol_5": "volatility",
    "candle_strength": "sentiment",
}
_KLINE_EXTRA_DESCRIPTION = {
    "vwap_bias": "价偏离VWAP(%)", "amt_trend": "成交额20日趋势(%)",
    "volume_std_20": "成交量20日变异系数", "high_low_range_20": "20日振幅均值(ATR简化)",
    "gap_up_5d": "5日跳空高开次数", "mom_risk_adj": "风险调整动量(mom/vol)",
    "res_mom_approx": "残差动量近似(动量偏离60日均)", "wma_mom": "加权移动平均动量",
    "skew_20": "20日收益偏度", "kurt_20": "20日收益峰度",
    "ofi_approx": "主动买卖累计5日", "amihud_5d": "Amihud非流动性5日均",
    "roll_spread": "Roll价差近似", "alpha_rank_cm": "close/ma20偏离",
    "alpha_corr_cv": "close-volume 10日相关",
    "alpha_tsrank_20": "close 20日百分位(ts_rank)", "alpha_delta_vol_5": "volume 5日变化",
    "candle_strength": "K线强度(c-o)/(h-l) 20日均",
}


def _make_kline_extra_handler(name: str):
    """生成注册用 handler: 从 series dict 计算指定 kline_extra 因子序列。"""
    def _h(data, **kw):  # noqa: ANN001 - 兼容 registry.handler(data, **params)
        try:
            s = data.get("series")
            if s is None or "close" not in s:
                return pd.Series(dtype=float)
            out = kline_extra_series(s)
            return out.get(name, pd.Series(dtype=float))
        except Exception:  # noqa: BLE001 - 单因子失败返回空序列, 不阻断
            return pd.Series(dtype=float)
    return _h


def _register_kline_extra() -> None:
    from quant_system.ic_factors.registry import get_factor, register_factor
    for name in KLINE_EXTRA_FACTORS:
        if get_factor(name) is not None:
            continue
        decorator = register_factor(
            name=name,
            category=_KLINE_EXTRA_CATEGORY.get(name, "others"),
            description=_KLINE_EXTRA_DESCRIPTION.get(name, "K线横截面因子"),
            direction=1,
            data_deps=["kline"],
        )
        decorator(_make_kline_extra_handler(name))  # 应用装饰器完成实际注册


_register_kline_extra()
