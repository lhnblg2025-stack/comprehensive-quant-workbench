"""
technical_v7.py — V7.0 技术面因子批量扩充（K线 → 25+ 因子）
=============================================================
输入: data["kline"] = {code: DataFrame(含 close/high/low/volume/amount)}
      与 zoo.py 同款结构（date index 或 date 列均可）
覆盖: 多周期动量/均线偏离/RSI/KDJ/MACD/量价/新高新低/波动
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 异名异口径的均保留独立实现，不强迁。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant_system.market_forecast._support.common.logger import get_logger
from quant_system.ic_factors.registry import register_factor

log = get_logger("qv6.factor.technical_v7")


def _klines(data: dict) -> dict:
    k = data.get("kline") or {}
    return k if isinstance(k, dict) else {}


def _series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(dtype=float)


def _last_pct(v: pd.Series, n: int) -> float | None:
    """n 日收益率。"""
    v = v.dropna()
    if len(v) <= n or v.iloc[-1 - n] == 0:
        return None
    return float(v.iloc[-1] / v.iloc[-1 - n] - 1)


def _cross_section(data: dict, fn) -> pd.Series:
    out = {}
    for code, df in _klines(data).items():
        if df is None or df.empty:
            continue
        v = fn(df)
        if v is not None and np.isfinite(v):
            out[code] = v
    return pd.Series(out, dtype=float)


# ── 动量（多周期）────────────────────────────────────────
@register_factor(name="tech_mom_10", category="momentum",
                 data_deps=["kline"], description="10日动量", direction=1)
def tech_mom_10(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 异名异口径-不强迁保留"""
    return _cross_section(data, lambda df: _last_pct(_series(df, "close"), 10))


@register_factor(name="tech_mom_20", category="momentum",
                 data_deps=["kline"], description="20日动量", direction=1)
def tech_mom_20(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 异名异口径-不强迁保留"""
    return _cross_section(data, lambda df: _last_pct(_series(df, "close"), 20))


@register_factor(name="tech_mom_60", category="momentum",
                 data_deps=["kline"], description="60日动量", direction=1)
def tech_mom_60(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 异名异口径-不强迁保留"""
    return _cross_section(data, lambda df: _last_pct(_series(df, "close"), 60))


@register_factor(name="tech_mom_120", category="momentum",
                 data_deps=["kline"], description="120日动量", direction=1)
def tech_mom_120(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 异名异口径-不强迁保留"""
    return _cross_section(data, lambda df: _last_pct(_series(df, "close"), 120))


@register_factor(name="tech_mom_3_10", category="momentum",
                 data_deps=["kline"], description="3日动量(短线加速度)", direction=1)
def tech_mom_3_10(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 异名异口径-不强迁保留"""
    return _cross_section(data, lambda df: _last_pct(_series(df, "close"), 3))


# ── 均线偏离 ─────────────────────────────────────────────
def _ma_dev(df: pd.DataFrame, n: int) -> float | None:
    c = _series(df, "close").dropna()
    if len(c) < n + 5 or c.iloc[-1] == 0:
        return None
    ma = c.iloc[-n:].mean()
    return float(c.iloc[-1] / ma - 1)


@register_factor(name="tech_ma5_dev", category="momentum",
                 data_deps=["kline"], description="收盘价偏离MA5", direction=1)
def tech_ma5_dev(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _cross_section(data, lambda df: _ma_dev(df, 5))


@register_factor(name="tech_ma20_dev", category="momentum",
                 data_deps=["kline"], description="收盘价偏离MA20", direction=1)
def tech_ma20_dev(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _cross_section(data, lambda df: _ma_dev(df, 20))


@register_factor(name="tech_ma60_dev", category="momentum",
                 data_deps=["kline"], description="收盘价偏离MA60", direction=1)
def tech_ma60_dev(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _cross_section(data, lambda df: _ma_dev(df, 60))


@register_factor(name="tech_ma_bull", category="momentum",
                 data_deps=["kline"], description="均线多头排列强度(MA5>MA20>MA60)", direction=1)
def tech_ma_bull(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    def fn(df: pd.DataFrame) -> float | None:
        c = _series(df, "close").dropna()
        if len(c) < 65:
            return None
        ma5, ma20, ma60 = c.iloc[-5:].mean(), c.iloc[-20:].mean(), c.iloc[-60:].mean()
        return float((ma5 > ma20) + (ma20 > ma60) + (c.iloc[-1] > ma5))
    return _cross_section(data, fn)


# ── RSI ──────────────────────────────────────────────────
def _rsi(df: pd.DataFrame, n: int = 14) -> float | None:
    c = _series(df, "close").dropna()
    if len(c) < n + 2:
        return None
    diff = c.diff().dropna()
    gain = diff.clip(lower=0).tail(n).mean()
    loss = (-diff.clip(upper=0)).tail(n).mean()
    # 除零保护：连续上涨(loss=0)→超买极值100；连续下跌(gain=0)→超卖极值0；
    # 无波动(gain=loss=0)→中性50。显式分支避免 RuntimeWarning: divide by zero，
    # 最终 clip 兜底保证返回值恒在 [0, 100]，无 inf/NaN。
    if loss == 0:
        return 100.0 if gain > 0 else 50.0
    if gain == 0:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = gain / loss
    rsi = 100 - 100 / (1 + rs)
    return float(np.clip(rsi, 0.0, 100.0))


@register_factor(name="tech_rsi14", category="reversal",
                 data_deps=["kline"], description="RSI14(超买超卖)", direction=1,
                 health_rules={"min": 0.0, "max": 100.0})
def tech_rsi14(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 异名异口径-不强迁保留"""
    return _cross_section(data, lambda df: _rsi(df, 14))


@register_factor(name="tech_rsi6", category="reversal",
                 data_deps=["kline"], description="RSI6(短线超买超卖)", direction=1,
                 health_rules={"min": 0.0, "max": 100.0})
def tech_rsi6(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 异名异口径-不强迁保留"""
    return _cross_section(data, lambda df: _rsi(df, 6))


# ── 量价 ─────────────────────────────────────────────────
@register_factor(name="tech_vol_ratio", category="liquidity",
                 data_deps=["kline"], description="量比(最新量/5日均量)", direction=1)
def tech_vol_ratio(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    def fn(df: pd.DataFrame) -> float | None:
        v = _series(df, "volume").dropna()
        if len(v) < 6 or v.iloc[-6] == 0:
            return None
        avg5 = v.iloc[-6:-1].mean()
        return float(v.iloc[-1] / avg5) if avg5 > 0 else None
    return _cross_section(data, fn)


@register_factor(name="tech_amount_trend", category="liquidity",
                 data_deps=["kline"], description="成交额5日趋势(放大为1)", direction=1)
def tech_amount_trend(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    def fn(df: pd.DataFrame) -> float | None:
        a = _series(df, "amount").dropna()
        if len(a) < 11 or a.iloc[-11] == 0:
            return None
        return float(a.iloc[-5:].mean() / a.iloc[-11:-5].mean() - 1)
    return _cross_section(data, fn)


@register_factor(name="tech_price_vol_div", category="liquidity",
                 data_deps=["kline"], description="量价背离(价升量缩=危险, 反向)", direction=-1)
def tech_price_vol_div(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    def fn(df: pd.DataFrame) -> float | None:
        c = _series(df, "close").dropna()
        v = _series(df, "volume").dropna()
        if len(c) < 6 or len(v) < 6:
            return None
        pc = c.iloc[-1] / c.iloc[-6] - 1
        pv = v.iloc[-1] / v.iloc[-6] - 1
        return float(pc - pv)  # 价涨量缩 → 正值 → 反向因子
    return _cross_section(data, fn)


@register_factor(name="tech_turnover_z", category="liquidity",
                 data_deps=["kline"], description="换手率Z(相对自身20日, 高=活跃)", direction=1)
def tech_turnover_z(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    def fn(df: pd.DataFrame) -> float | None:
        t = _series(df, "turnover").dropna()
        if len(t) < 25:
            return None
        mu, sd = t.iloc[-20:].mean(), t.iloc[-20:].std()
        return float((t.iloc[-1] - mu) / sd) if sd > 1e-9 else None
    return _cross_section(data, fn)


# ── 新高新低 ─────────────────────────────────────────────
@register_factor(name="tech_high_20d", category="momentum",
                 data_deps=["kline"], description="创20日新高(突破)", direction=1)
def tech_high_20d(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    def fn(df: pd.DataFrame) -> float | None:
        h = _series(df, "high").dropna()
        if len(h) < 25:
            return None
        return float(h.iloc[-1] >= h.iloc[-21:-1].max())
    return _cross_section(data, fn)


@register_factor(name="tech_low_20d", category="reversal",
                 data_deps=["kline"], description="创20日新低(破位, 反向)", direction=-1)
def tech_low_20d(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    def fn(df: pd.DataFrame) -> float | None:
        lo = _series(df, "low").dropna()
        if len(lo) < 25:
            return None
        return float(lo.iloc[-1] <= lo.iloc[-21:-1].min())
    return _cross_section(data, fn)


@register_factor(name="tech_close_pos", category="momentum",
                 data_deps=["kline"], description="收盘价在20日区间位置(近高位=强)", direction=1)
def tech_close_pos(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    def fn(df: pd.DataFrame) -> float | None:
        c = _series(df, "close").dropna()
        if len(c) < 25:
            return None
        lo, hi = c.iloc[-20:].min(), c.iloc[-20:].max()
        return float((c.iloc[-1] - lo) / (hi - lo)) if hi > lo else 0.5
    return _cross_section(data, fn)


# ── 波动 ─────────────────────────────────────────────────
@register_factor(name="tech_atr_pct", category="volatility",
                 data_deps=["kline"], description="ATR/价格(真实波幅率)", direction=-1)
def tech_atr_pct(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    def fn(df: pd.DataFrame) -> float | None:
        h = _series(df, "high").dropna()
        l = _series(df, "low").dropna()
        c = _series(df, "close").dropna()
        n = min(len(h), len(l), len(c))
        if n < 15:
            return None
        h, l, c = h.iloc[-n:], l.iloc[-n:], c.iloc[-n:]
        pc = c.shift(1)
        tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1).dropna()
        atr = tr.iloc[-14:].mean()
        return float(atr / c.iloc[-1]) if c.iloc[-1] > 0 else None
    return _cross_section(data, fn)


@register_factor(name="tech_vol20", category="volatility",
                 data_deps=["kline"], description="20日收益波动率", direction=-1)
def tech_vol20(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 异名异口径-不强迁保留"""
    def fn(df: pd.DataFrame) -> float | None:
        c = _series(df, "close").dropna()
        if len(c) < 25:
            return None
        r = c.pct_change().dropna().iloc[-20:]
        return float(r.std()) if len(r) >= 10 else None
    return _cross_section(data, fn)


@register_factor(name="tech_vol_skew", category="volatility",
                 data_deps=["kline"], description="收益偏度(正偏=右尾机会)", direction=1)
def tech_vol_skew(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    def fn(df: pd.DataFrame) -> float | None:
        c = _series(df, "close").dropna()
        if len(c) < 65:
            return None
        r = c.pct_change().dropna().iloc[-60:]
        return float(r.skew()) if len(r) >= 30 else None
    return _cross_section(data, fn)


@register_factor(name="tech_drawdown_60", category="volatility",
                 data_deps=["kline"], description="60日最大回撤(反向)", direction=-1)
def tech_drawdown_60(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 异名异口径-不强迁保留"""
    def fn(df: pd.DataFrame) -> float | None:
        c = _series(df, "close").dropna()
        if len(c) < 65:
            return None
        w = c.iloc[-60:]
        return float((w / w.cummax() - 1).min())
    return _cross_section(data, fn)


# ── KDJ / MACD 变体 ──────────────────────────────────────
def _kdj_j(df: pd.DataFrame) -> float | None:
    h = _series(df, "high").dropna()
    l = _series(df, "low").dropna()
    c = _series(df, "close").dropna()
    n = min(len(h), len(l), len(c))
    if n < 20:
        return None
    h, l, c = h.iloc[-n:], l.iloc[-n:], c.iloc[-n:]
    low9, high9 = l.rolling(9).min(), h.rolling(9).max()
    rsv = (c - low9) / (high9 - low9).replace(0, np.nan) * 100
    k = rsv.ewm(com=2).mean()
    d = k.ewm(com=2).mean()
    j = 3 * k - 2 * d
    v = j.dropna()
    return float(v.iloc[-1]) if not v.empty else None


@register_factor(name="tech_kdj_j", category="reversal",
                 data_deps=["kline"], description="KDJ的J值(超买超卖)", direction=1)
def tech_kdj_j(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _cross_section(data, _kdj_j)


@register_factor(name="tech_macd_hist", category="momentum",
                 data_deps=["kline"], description="MACD柱(动能)", direction=1)
def tech_macd_hist(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 异名异口径-不强迁保留"""
    def fn(df: pd.DataFrame) -> float | None:
        c = _series(df, "close").dropna()
        if len(c) < 40:
            return None
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        v = (dif - dea).dropna()
        return float(v.iloc[-1]) if not v.empty else None
    return _cross_section(data, fn)


@register_factor(name="tech_macd_gold", category="momentum",
                 data_deps=["kline"], description="MACD金叉状态(近3日DIF上穿DEA)", direction=1)
def tech_macd_gold(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 异名异口径-不强迁保留"""
    def fn(df: pd.DataFrame) -> float | None:
        c = _series(df, "close").dropna()
        if len(c) < 40:
            return None
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        cross = (dif > dea) & (dif.shift(1) <= dea.shift(1))
        return float(bool(cross.iloc[-3:].any()))
    return _cross_section(data, fn)


# ── 缺口 / 形态 ──────────────────────────────────────────
@register_factor(name="tech_gap_up", category="momentum",
                 data_deps=["kline"], description="今日跳空高开幅度", direction=1)
def tech_gap_up(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    def fn(df: pd.DataFrame) -> float | None:
        o = _series(df, "open").dropna()
        c = _series(df, "close").dropna()
        if len(o) < 3 or len(c) < 3 or c.iloc[-2] == 0:
            return None
        return float(o.iloc[-1] / c.iloc[-2] - 1)
    return _cross_section(data, fn)


@register_factor(name="tech_candle_body", category="momentum",
                 data_deps=["kline"], description="阳线实体强度(今收-今开)/昨收", direction=1)
def tech_candle_body(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    def fn(df: pd.DataFrame) -> float | None:
        o = _series(df, "open").dropna()
        c = _series(df, "close").dropna()
        if len(o) < 1 or len(c) < 2 or c.iloc[-2] == 0:
            return None
        return float((c.iloc[-1] - o.iloc[-1]) / c.iloc[-2])
    return _cross_section(data, fn)


@register_factor(name="tech_upper_shadow", category="volatility",
                 data_deps=["kline"], description="上影线比率(抛压, 反向)", direction=-1)
def tech_upper_shadow(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    def fn(df: pd.DataFrame) -> float | None:
        h = _series(df, "high").dropna()
        c = _series(df, "close").dropna()
        o = _series(df, "open").dropna()
        if len(h) < 1 or h.iloc[-1] <= 0:
            return None
        body = max(c.iloc[-1], o.iloc[-1])
        return float((h.iloc[-1] - body) / h.iloc[-1])
    return _cross_section(data, fn)


# ══════════════════════════════════════════════════════════
# 第二批：技术指标变体（布林/威廉/CCI/OBV/DMI/动量/风险调整）
# ══════════════════════════════════════════════════════════

@register_factor(name="tech_boll_pos", category="volatility",
                 data_deps=["kline"], description="布林带位置(20,2, 0下轨/1上轨)", direction=1)
def tech_boll_pos(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 异名异口径-不强迁保留"""
    def fn(df: pd.DataFrame) -> float | None:
        c = _series(df, "close").dropna()
        if len(c) < 25:
            return None
        w = c.iloc[-20:]
        mid, sd = w.mean(), w.std()
        if sd < 1e-9:
            return 0.5
        up, lo = mid + 2 * sd, mid - 2 * sd
        return float((c.iloc[-1] - lo) / (up - lo))
    return _cross_section(data, fn)


@register_factor(name="tech_williams_r", category="reversal",
                 data_deps=["kline"], description="威廉%R(近14日, -100~0, 高=超买)", direction=-1)
def tech_williams_r(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    def fn(df: pd.DataFrame) -> float | None:
        h = _series(df, "high").dropna()
        l = _series(df, "low").dropna()
        c = _series(df, "close").dropna()
        n = min(len(h), len(l), len(c))
        if n < 15:
            return None
        hh, ll = h.iloc[-14:].max(), l.iloc[-14:].min()
        if hh == ll:
            return -50.0
        return float((hh - c.iloc[-1]) / (hh - ll) * -100)
    return _cross_section(data, fn)


@register_factor(name="tech_cci20", category="reversal",
                 data_deps=["kline"], description="CCI20(超买超卖)", direction=1)
def tech_cci20(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 异名异口径-不强迁保留"""
    def fn(df: pd.DataFrame) -> float | None:
        h = _series(df, "high").dropna()
        l = _series(df, "low").dropna()
        c = _series(df, "close").dropna()
        n = min(len(h), len(l), len(c))
        if n < 25:
            return None
        tp = (h.iloc[-20:] + l.iloc[-20:] + c.iloc[-20:]) / 3
        ma = tp.mean()
        md = (tp - ma).abs().mean()
        if md < 1e-9:
            return 0.0
        return float((tp.iloc[-1] - ma) / (0.015 * md))
    return _cross_section(data, fn)


@register_factor(name="tech_obv_slope", category="liquidity",
                 data_deps=["kline"], description="OBV 20日斜率(量能累积方向)", direction=1)
def tech_obv_slope(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    def fn(df: pd.DataFrame) -> float | None:
        c = _series(df, "close").dropna()
        v = _series(df, "volume").dropna()
        n = min(len(c), len(v))
        if n < 25:
            return None
        c, v = c.iloc[-25:], v.iloc[-25:]
        obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
        x = np.arange(len(obv))
        slope = np.polyfit(x, obv.values, 1)[0]
        return float(slope)
    return _cross_section(data, fn)


@register_factor(name="tech_dmi_plus", category="momentum",
                 data_deps=["kline"], description="DMI+方向指标(14日, 多头强度)", direction=1)
def tech_dmi_plus(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    def fn(df: pd.DataFrame) -> float | None:
        h = _series(df, "high").dropna()
        l = _series(df, "low").dropna()
        c = _series(df, "close").dropna()
        n = min(len(h), len(l), len(c))
        if n < 30:
            return None
        h, l, c = h.iloc[-30:], l.iloc[-30:], c.iloc[-30:]
        up = (h.diff()).clip(lower=0)
        dn = (-l.diff()).clip(lower=0)
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        pdi = 100 * up.rolling(14).mean() / atr.replace(0, np.nan)
        v = pdi.dropna()
        return float(v.iloc[-1]) if not v.empty else None
    return _cross_section(data, fn)


@register_factor(name="tech_ret_ytd", category="momentum",
                 data_deps=["kline"], description="年初至今动量", direction=1)
def tech_ret_ytd(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    def fn(df: pd.DataFrame) -> float | None:
        c = _series(df, "close").dropna()
        if c.empty:
            return None
        if "date" not in df.columns:
            return None
        dates = pd.to_datetime(df["date"], errors="coerce")
        if dates.isna().all():
            return None
        year = dates.iloc[0].year
        mask = dates >= pd.Timestamp(year=year, month=1, day=1)
        idx = c.index[mask]
        if len(idx) < 2 or c.loc[idx[0]] == 0:
            return None
        return float(c.loc[idx[-1]] / c.loc[idx[0]] - 1)
    return _cross_section(data, fn)


@register_factor(name="tech_high_52w_pos", category="momentum",
                 data_deps=["kline"], description="收盘在52周区间位置(近高位=强)", direction=1)
def tech_high_52w_pos(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    def fn(df: pd.DataFrame) -> float | None:
        c = _series(df, "close").dropna()
        if len(c) < 60:
            return None
        w = c.iloc[-250:] if len(c) >= 250 else c
        lo, hi = w.min(), w.max()
        return float((c.iloc[-1] - lo) / (hi - lo)) if hi > lo else 0.5
    return _cross_section(data, fn)


@register_factor(name="tech_consec_up", category="momentum",
                 data_deps=["kline"], description="连续上涨天数(连阳)", direction=1)
def tech_consec_up(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    def fn(df: pd.DataFrame) -> float | None:
        c = _series(df, "close").dropna()
        if len(c) < 3:
            return None
        ret = c.diff().dropna()
        cnt = 0
        for r in ret.iloc[::-1]:
            if r > 0:
                cnt += 1
            else:
                break
        return float(cnt)
    return _cross_section(data, fn)


@register_factor(name="tech_gap_fill", category="reversal",
                 data_deps=["kline"], description="跳空缺口回补(昨日高开缺口今日回补, 反向)", direction=-1)
def tech_gap_fill(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    def fn(df: pd.DataFrame) -> float | None:
        o = _series(df, "open").dropna()
        c = _series(df, "close").dropna()
        l = _series(df, "low").dropna()
        if len(o) < 3 or len(l) < 3 or len(c) < 3:
            return None
        gap_up = o.iloc[-1] > c.iloc[-2] * 1.005  # 今日跳空高开>0.5%
        if gap_up and l.iloc[-1] <= c.iloc[-2]:   # 但日内回补缺口
            return 1.0
        return 0.0
    return _cross_section(data, fn)


@register_factor(name="tech_ma20_slope", category="momentum",
                 data_deps=["kline"], description="MA20斜率(5日变化, 趋势强度)", direction=1)
def tech_ma20_slope(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    def fn(df: pd.DataFrame) -> float | None:
        c = _series(df, "close").dropna()
        if len(c) < 30:
            return None
        ma = c.rolling(20).mean().dropna()
        if len(ma) < 10 or ma.iloc[-6] == 0:
            return None
        return float(ma.iloc[-1] / ma.iloc[-6] - 1)
    return _cross_section(data, fn)


@register_factor(name="tech_sharpe_60", category="volatility",
                 data_deps=["kline"], description="60日夏普(收益/波动, 风险调整后收益)", direction=1)
def tech_sharpe_60(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    def fn(df: pd.DataFrame) -> float | None:
        c = _series(df, "close").dropna()
        if len(c) < 65:
            return None
        r = c.pct_change().dropna().iloc[-60:]
        sd = r.std()
        if sd < 1e-9 or r.mean() == 0:
            return 0.0
        return float(r.mean() / sd * np.sqrt(252))
    return _cross_section(data, fn)


@register_factor(name="tech_sortino_60", category="volatility",
                 data_deps=["kline"], description="60日索提诺(下行波动修正夏普)", direction=1)
def tech_sortino_60(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    def fn(df: pd.DataFrame) -> float | None:
        c = _series(df, "close").dropna()
        if len(c) < 65:
            return None
        r = c.pct_change().dropna().iloc[-60:]
        downside = r[r < 0].std()
        if downside < 1e-9:
            return 0.0 if r.mean() <= 0 else 10.0
        return float(r.mean() / downside * np.sqrt(252))
    return _cross_section(data, fn)


@register_factor(name="tech_calmar_60", category="volatility",
                 data_deps=["kline"], description="60日卡玛(年化收益/最大回撤)", direction=1)
def tech_calmar_60(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    def fn(df: pd.DataFrame) -> float | None:
        c = _series(df, "close").dropna()
        if len(c) < 65:
            return None
        w = c.iloc[-60:]
        ret = w.iloc[-1] / w.iloc[0] - 1
        mdd = (w / w.cummax() - 1).min()
        if mdd >= -1e-9:
            return 10.0 if ret > 0 else 0.0
        return float(ret / abs(mdd))
    return _cross_section(data, fn)


@register_factor(name="tech_vol_ma20", category="liquidity",
                 data_deps=["kline"], description="量能比(最新量/20日均量)", direction=1)
def tech_vol_ma20(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    def fn(df: pd.DataFrame) -> float | None:
        v = _series(df, "volume").dropna()
        if len(v) < 25 or v.iloc[-20:].mean() == 0:
            return None
        return float(v.iloc[-1] / v.iloc[-20:].mean())
    return _cross_section(data, fn)


@register_factor(name="tech_mom_accel", category="momentum",
                 data_deps=["kline"], description="动量加速度(20日动量-60日动量, 短线加速)", direction=1)
def tech_mom_accel(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    def fn(df: pd.DataFrame) -> float | None:
        c = _series(df, "close").dropna()
        m20, m60 = _last_pct(c, 20), _last_pct(c, 60)
        if m20 is None or m60 is None:
            return None
        return float(m20 - m60)
    return _cross_section(data, fn)


@register_factor(name="tech_ret_week", category="momentum",
                 data_deps=["kline"], description="5日周动量", direction=1)
def tech_ret_week(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _cross_section(data, lambda df: _last_pct(_series(df, "close"), 5))


@register_factor(name="tech_ret_250", category="momentum",
                 data_deps=["kline"], description="250日年动量", direction=1)
def tech_ret_250(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _cross_section(data, lambda df: _last_pct(_series(df, "close"), 250))


@register_factor(name="tech_up_down_ratio", category="momentum",
                 data_deps=["kline"], description="20日涨跌日比(>1=多头主导)", direction=1)
def tech_up_down_ratio(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    def fn(df: pd.DataFrame) -> float | None:
        c = _series(df, "close").dropna()
        if len(c) < 25:
            return None
        r = c.pct_change().dropna().iloc[-20:]
        up = (r > 0).sum()
        dn = (r < 0).sum()
        if dn == 0:
            return 20.0 if up > 0 else 0.0
        return float(up / dn)
    return _cross_section(data, fn)


# ══════════════════════════════════════════════════════════
# 第三批：横截面技术变体（RSI差/布林宽/突破/年线/量能异动）
# ══════════════════════════════════════════════════════════

@register_factor(name="tech_rsi_diff", category="reversal",
                 data_deps=["kline"], description="RSI20-RSI50(短线强度差)", direction=1)
def tech_rsi_diff(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    def fn(df: pd.DataFrame) -> float | None:
        c = _series(df, "close").dropna()
        if len(c) < 55:
            return None
        r20 = _rsi(df, 20)
        r50 = _rsi(df, 50)
        if r20 is None or r50 is None:
            return None
        return float(r20 - r50)
    return _cross_section(data, fn)


@register_factor(name="tech_boll_width", category="volatility",
                 data_deps=["kline"], description="布林带宽度(20,2, 波动扩张度)", direction=1)
def tech_boll_width(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    def fn(df: pd.DataFrame) -> float | None:
        c = _series(df, "close").dropna()
        if len(c) < 25 or c.iloc[-1] == 0:
            return None
        w = c.iloc[-20:]
        sd = w.std()
        return float(sd / w.mean() * 4)
    return _cross_section(data, fn)


@register_factor(name="tech_ret_20_60", category="momentum",
                 data_deps=["kline"], description="20日/60日动量比(>1=短强于长)", direction=1)
def tech_ret_20_60(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    def fn(df: pd.DataFrame) -> float | None:
        c = _series(df, "close").dropna()
        m20, m60 = _last_pct(c, 20), _last_pct(c, 60)
        if m20 is None or m60 is None or m60 == 0:
            return None
        return float(m20 / abs(m60))
    return _cross_section(data, fn)


@register_factor(name="tech_ma_cross_5_20", category="momentum",
                 data_deps=["kline"], description="MA5/MA20金叉状态(近3日)", direction=1)
def tech_ma_cross_5_20(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 异名异口径-不强迁保留"""
    def fn(df: pd.DataFrame) -> float | None:
        c = _series(df, "close").dropna()
        if len(c) < 25:
            return None
        ma5 = c.rolling(5).mean()
        ma20 = c.rolling(20).mean()
        cross = (ma5 > ma20) & (ma5.shift(1) <= ma20.shift(1))
        return float(bool(cross.iloc[-3:].any()))
    return _cross_section(data, fn)


@register_factor(name="tech_close_ma250", category="momentum",
                 data_deps=["kline"], description="收盘/年线偏离(牛熊分界)", direction=1)
def tech_close_ma250(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    def fn(df: pd.DataFrame) -> float | None:
        c = _series(df, "close").dropna()
        if len(c) < 260 or c.iloc[-250:].mean() == 0:
            return None
        ma = c.iloc[-250:].mean()
        return float(c.iloc[-1] / ma - 1)
    return _cross_section(data, fn)


@register_factor(name="tech_breakout_20", category="momentum",
                 data_deps=["kline"], description="突破20日高点幅度(正=突破)", direction=1)
def tech_breakout_20(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    def fn(df: pd.DataFrame) -> float | None:
        h = _series(df, "high").dropna()
        c = _series(df, "close").dropna()
        if len(h) < 25 or len(c) < 25:
            return None
        prev_hi = h.iloc[-21:-1].max()
        if prev_hi == 0:
            return None
        return float(c.iloc[-1] / prev_hi - 1)
    return _cross_section(data, fn)


@register_factor(name="tech_amount_z", category="liquidity",
                 data_deps=["kline"], description="成交额Z值(相对20日, 放量异动)", direction=1)
def tech_amount_z(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    def fn(df: pd.DataFrame) -> float | None:
        a = _series(df, "amount").dropna()
        if len(a) < 25:
            return None
        mu, sd = a.iloc[-20:].mean(), a.iloc[-20:].std()
        if sd < 1e-9:
            return 0.0
        return float((a.iloc[-1] - mu) / sd)
    return _cross_section(data, fn)


@register_factor(name="tech_vol_std20", category="liquidity",
                 data_deps=["kline"], description="量能波动(20日量std/均量, 异动度)", direction=1)
def tech_vol_std20(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    def fn(df: pd.DataFrame) -> float | None:
        v = _series(df, "volume").dropna()
        if len(v) < 25 or v.iloc[-20:].mean() == 0:
            return None
        return float(v.iloc[-20:].std() / v.iloc[-20:].mean())
    return _cross_section(data, fn)


@register_factor(name="tech_hl_range10", category="volatility",
                 data_deps=["kline"], description="10日高低振幅(波动活跃度)", direction=-1)
def tech_hl_range10(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    def fn(df: pd.DataFrame) -> float | None:
        h = _series(df, "high").dropna()
        l = _series(df, "low").dropna()
        if len(h) < 10 or len(l) < 10 or h.iloc[-10:].mean() == 0:
            return None
        return float((h.iloc[-10:].max() - l.iloc[-10:].min()) / h.iloc[-10:].mean())
    return _cross_section(data, fn)


@register_factor(name="tech_obv_trend", category="liquidity",
                 data_deps=["kline"], description="OBV趋势(20日均值/60日均值-1)", direction=1)
def tech_obv_trend(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    def fn(df: pd.DataFrame) -> float | None:
        c = _series(df, "close").dropna()
        v = _series(df, "volume").dropna()
        n = min(len(c), len(v))
        if n < 65:
            return None
        c, v = c.iloc[-65:], v.iloc[-65:]
        obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
        m20, m60 = obv.iloc[-20:].mean(), obv.iloc[-60:].mean()
        if abs(m60) < 1e-9:
            return 0.0
        return float(m20 / abs(m60))
    return _cross_section(data, fn)


@register_factor(name="tech_vol_contract", category="volatility",
                 data_deps=["kline"], description="波动率收缩(近期std/60日std, 低=蓄势)", direction=-1)
def tech_vol_contract(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    def fn(df: pd.DataFrame) -> float | None:
        c = _series(df, "close").dropna()
        if len(c) < 65:
            return None
        r = c.pct_change().dropna()
        sd5 = r.iloc[-5:].std()
        sd60 = r.iloc[-60:].std()
        if sd60 < 1e-12:
            return 1.0
        return float(sd5 / sd60)
    return _cross_section(data, fn)


@register_factor(name="tech_cum_ret_10", category="momentum",
                 data_deps=["kline"], description="10日累计收益(原始动量)", direction=1)
def tech_cum_ret_10(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _cross_section(data, lambda df: _last_pct(_series(df, "close"), 10))
