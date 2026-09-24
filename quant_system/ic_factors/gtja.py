"""
gtja.py — 国泰君安 GTJA Alpha 因子移植（2026-08-14）
====================================================
来源: eGTAlpha (dengyishuo/eGTAlpha, MIT) — 国泰君安 191 个短周期价量 Alpha
因子的公开公式。本模块移植其中日线可实现的 20 个到本地 ic_factors 风格。

注意:
  - 原版为分钟级因子，本地日线近似（vwap 用典型价 (H+L+C)/3 近似）
  - 方向 direction=1 注册，实际方向由 IC 校准链覆盖（validate_calibrate_direction）
  - 数据: {code: DataFrame}，df 列含 open/high/low/close/volume
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_ALPHA_EPS = 1e-8


def _s(func):
    """装饰器: dict 结果 → Series（与 price.py _series 同风格）。"""
    import functools

    @functools.wraps(func)
    def wrapper(data: dict[str, pd.DataFrame], *args, **kwargs) -> pd.Series:
        out = func(data, *args, **kwargs)
        return out if isinstance(out, pd.Series) else pd.Series(out, dtype=float)
    return wrapper


def _typ(df: pd.DataFrame) -> pd.Series:
    """典型价 (H+L+C)/3 近似 VWAP。"""
    return (df["high"] + df["low"] + df["close"]) / 3.0


def _delta(s: pd.Series, n: int) -> pd.Series:
    return s - s.shift(n)


def _delay(s: pd.Series, n: int) -> pd.Series:
    return s.shift(n)


def _tsrank(s: pd.Series, n: int) -> pd.Series:
    """时间序列排名 (0-1)。"""
    return s.rolling(n).rank(pct=True)


def _roll_corr(a: pd.Series, b: pd.Series, n: int) -> pd.Series:
    return a.rolling(n).corr(b)


def _roll_max(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).max()


def _roll_min(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).min()


def _roll_sum(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).sum()


def _cross_rank(s: pd.Series) -> pd.Series:
    """截面排名（最后一期）。"""
    return s.rank(pct=True)


# ══════════════════════════════════════════════════════════════════
# 因子实现（函数名 gtja_NNN，输入 data dict，输出 Series）
# ══════════════════════════════════════════════════════════════════

@_s
def gtja_001(data, window: int = 6) -> pd.Series:
    """-CORR(RANK(DELTA(LOG(VOL),1)), RANK((CLOSE-OPEN)/OPEN), 6) — 量价秩相关。"""
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < window + 3 or "volume" not in df.columns:
            continue
        v = df["volume"].astype(float).replace(0, np.nan)
        r1 = _tsrank(_delta(np.log(v), 1), window)
        r2 = _tsrank((df["close"] - df["open"]) / (df["open"] + _ALPHA_EPS), window)
        c = _roll_corr(r1, r2, window)
        if c.notna().any():
            out[sym] = float(-c.dropna().iloc[-1])
    return out


@_s
def gtja_002(data, window: int = 1) -> pd.Series:
    """-DELTA(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW), 1) — 价格位置变化。"""
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < window + 2:
            continue
        pos = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (
            df["high"] - df["low"] + _ALPHA_EPS)
        d = _delta(pos, window)
        if d.notna().any():
            out[sym] = float(-d.dropna().iloc[-1])
    return out


@_s
def gtja_003(data, window: int = 6) -> pd.Series:
    """ROLLSUM(close==delay(close,1)?0:close-(close>delay?min(low,delay):max(high,delay)), 6)"""
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < window + 2:
            continue
        close, low, high = df["close"], df["low"], df["high"]
        prev = _delay(close, 1)
        term = pd.Series(0.0, index=df.index)
        m = prev.notna()
        term[m] = close[m] - np.where(close[m] > prev[m],
                                      np.minimum(low[m], prev[m]),
                                      np.maximum(high[m], prev[m]))
        term[close == prev] = 0.0
        s = _roll_sum(term, window)
        if s.notna().any():
            out[sym] = float(s.dropna().iloc[-1])
    return out


@_s
def gtja_005(data, window: int = 5) -> pd.Series:
    """-TSMAX(CORR(TSRANK(VOL,5),TSRANK(HIGH,5),5),3) — 量价排名相关。"""
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < window + 3 or "volume" not in df.columns:
            continue
        rv = _tsrank(df["volume"].astype(float), window)
        rh = _tsrank(df["high"].astype(float), window)
        c = _roll_corr(rv, rh, window)
        m = _roll_max(c, 3)
        if m.notna().any():
            out[sym] = float(-m.dropna().iloc[-1])
    return out


@_s
def gtja_006(data, window: int = 4) -> pd.Series:
    """-RANK(SIGN(DELTA(OPEN*0.85+HIGH*0.15, 4))) — 开盘位置符号。"""
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < window + 2:
            continue
        x = df["open"] * 0.85 + df["high"] * 0.15
        d = _delta(x, window).fillna(0.0)
        out[sym] = float(-_cross_rank(np.sign(d)).iloc[-1])
    return out


@_s
def gtja_007(data, window: int = 3) -> pd.Series:
    """RANK(MAX(VWAP-CLOSE,3)+RANK(MIN(VWAP-CLOSE,3))*RANK(DELTA(VOL,3)))"""
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < window + 2 or "volume" not in df.columns:
            continue
        vwap = _typ(df)
        diff = vwap - df["close"]
        mx = _roll_max(diff, window).fillna(0.0)
        mn = _roll_min(diff, window).fillna(0.0)
        dv = _delta(df["volume"].astype(float), window).fillna(0.0)
        x = mx + _cross_rank(mn) * _cross_rank(dv)
        out[sym] = float(_cross_rank(x).iloc[-1])
    return out


@_s
def gtja_008(data, window: int = 4) -> pd.Series:
    """RANK(DELTA(((HIGH+LOW)/2*0.2+VWAP*0.8), 4)*-1)"""
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < window + 2:
            continue
        vwap = _typ(df)
        x = (df["high"] + df["low"]) / 2 * 0.2 + vwap * 0.8
        d = -_delta(x, window).fillna(0.0)
        out[sym] = float(_cross_rank(d).iloc[-1])
    return out


@_s
def gtja_009(data, window: int = 7) -> pd.Series:
    """SMA((H+L)/2-(DELAY(H,1)+DELAY(L,1))/2)*(H-L)/VOL, 7, 2) 近似: ewm alpha=2/7"""
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < window + 2 or "volume" not in df.columns:
            continue
        x = ((df["high"] + df["low"]) / 2
             - (_delay(df["high"], 1) + _delay(df["low"], 1)) / 2) * (
            df["high"] - df["low"]) / (df["volume"].astype(float) + _ALPHA_EPS)
        sm = x.ewm(alpha=2 / window, adjust=False).mean()
        if sm.notna().any():
            out[sym] = float(sm.dropna().iloc[-1])
    return out


@_s
def gtja_011(data, window: int = 6) -> pd.Series:
    """SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOL, 6) — 量价位置加权。"""
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < window + 1 or "volume" not in df.columns:
            continue
        x = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (
            df["high"] - df["low"] + _ALPHA_EPS) * df["volume"].astype(float)
        s = _roll_sum(x, window)
        if s.notna().any():
            out[sym] = float(s.dropna().iloc[-1])
    return out


@_s
def gtja_012(data, window: int = 1) -> pd.Series:
    """RANK(OPEN-VWAP)*RANK(CLOSE-VWAP) — 价格相对典型价位置。"""
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < 5:
            continue
        vwap = _typ(df)
        x = _cross_rank(df["open"] - vwap) * _cross_rank(df["close"] - vwap)
        out[sym] = float(x.iloc[-1])
    return out


@_s
def gtja_013(data, window: int = 1) -> pd.Series:
    """SQRT(HIGH*LOW)-VWAP — 波动区间中心偏离。"""
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < 2:
            continue
        x = np.sqrt(df["high"] * df["low"]) - _typ(df)
        if x.notna().any():
            out[sym] = float(x.dropna().iloc[-1])
    return out


@_s
def gtja_014(data, window: int = 5) -> pd.Series:
    """CLOSE-DELAY(CLOSE,5) — 5日收盘变动。"""
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < window + 1:
            continue
        d = _delta(df["close"].astype(float), window)
        if d.notna().any():
            out[sym] = float(d.dropna().iloc[-1])
    return out


@_s
def gtja_015(data, window: int = 1) -> pd.Series:
    """OPEN/DELAY(CLOSE,1)-1 — 开盘跳空。"""
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < 3:
            continue
        prev = _delay(df["close"].astype(float), 1)
        x = df["open"] / (prev + _ALPHA_EPS) - 1
        if x.notna().any():
            out[sym] = float(x.dropna().iloc[-1])
    return out


@_s
def gtja_016(data, window: int = 5) -> pd.Series:
    """-MAX(CORR(RANK(VOL),RANK(VWAP),5),5) — 量价秩相关极值。"""
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < window + 3 or "volume" not in df.columns:
            continue
        rv = _tsrank(df["volume"].astype(float), window)
        rw = _tsrank(_typ(df), window)
        c = _roll_corr(rv, rw, window)
        m = _roll_max(c, window)
        if m.notna().any():
            out[sym] = float(-m.dropna().iloc[-1])
    return out


@_s
def gtja_017(data, window: int = 15) -> pd.Series:
    """RANK(VWAP-MAX(VWAP,15))^DELTA(CLOSE,5) — 典型价回撤强度。"""
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < max(window, 5) + 2:
            continue
        vwap = _typ(df)
        dd = _cross_rank(vwap - _roll_max(vwap, window).fillna(vwap))
        dc = _delta(df["close"].astype(float), 5).fillna(0.0)
        x = np.sign(dc) * np.power(np.abs(dd), 1.0)
        if x.notna().any():
            out[sym] = float(x.dropna().iloc[-1])
    return out


@_s
def gtja_018(data, window: int = 5) -> pd.Series:
    """CLOSE/DELAY(CLOSE,5)-1 — 5日动量。"""
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < window + 1:
            continue
        prev = _delay(df["close"].astype(float), window)
        x = df["close"] / (prev + _ALPHA_EPS) - 1
        if x.notna().any():
            out[sym] = float(x.dropna().iloc[-1])
    return out


@_s
def gtja_019(data, window: int = 5) -> pd.Series:
    """close<delay(close,5)? (c-d)/d : (c-d)/c — 下跌用基期、上涨用当期。"""
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < window + 1:
            continue
        close = df["close"].astype(float)
        prev = _delay(close, window)
        d = close - prev
        x = d / (prev + _ALPHA_EPS)
        up = prev.notna() & (close >= prev) & (d != 0)
        x[up] = d[up] / close[up]
        if x.notna().any():
            out[sym] = float(x.dropna().iloc[-1])
    return out


@_s
def gtja_020(data, window: int = 6) -> pd.Series:
    """(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100 — 6日动量百分比。"""
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < window + 1:
            continue
        prev = _delay(df["close"].astype(float), window)
        x = (df["close"] - prev) / (prev + _ALPHA_EPS) * 100
        if x.notna().any():
            out[sym] = float(x.dropna().iloc[-1])
    return out


# ══════════════════════════════════════════════════════════════════
# 注册表（供 zoo.import_from_zoo / registry.autodiscover 消费）
# ══════════════════════════════════════════════════════════════════

GTJA_FACTORS: dict[str, tuple] = {
    "gtja_001": (gtja_001, "GTJA#001 量价秩相关", 1),
    "gtja_002": (gtja_002, "GTJA#002 价格位置变化", 1),
    "gtja_003": (gtja_003, "GTJA#003 真实区间和", 1),
    "gtja_005": (gtja_005, "GTJA#005 量价排名相关", 1),
    "gtja_006": (gtja_006, "GTJA#006 开盘位置符号", 1),
    "gtja_007": (gtja_007, "GTJA#007 典型价偏离", 1),
    "gtja_008": (gtja_008, "GTJA#008 加权价变动", 1),
    "gtja_009": (gtja_009, "GTJA#009 价格区间动量", 1),
    "gtja_011": (gtja_011, "GTJA#011 量价位置加权", 1),
    "gtja_012": (gtja_012, "GTJA#012 价格相对典型价", 1),
    "gtja_013": (gtja_013, "GTJA#013 波动中心偏离", 1),
    "gtja_014": (gtja_014, "GTJA#014 5日收盘变动", 1),
    "gtja_015": (gtja_015, "GTJA#015 开盘跳空", 1),
    "gtja_016": (gtja_016, "GTJA#016 量价秩相关极值", 1),
    "gtja_017": (gtja_017, "GTJA#017 典型价回撤强度", 1),
    "gtja_018": (gtja_018, "GTJA#018 5日动量", 1),
    "gtja_019": (gtja_019, "GTJA#019 双向动量", 1),
    "gtja_020": (gtja_020, "GTJA#020 6日动量百分比", 1),
}


def register_into(reg: dict, register_fn) -> int:
    """把 GTJA 因子注册进目标注册表（zoo._REGISTRY 或 ic registry）。

    reg:        目标 _REGISTRY dict
    register_fn: 注册函数（zoo.register_factor 装饰器 / Factor 构造）
    返回注册数。幂等: 已存在的同名因子不覆盖。
    """
    n = 0
    for name, (func, desc, direction) in GTJA_FACTORS.items():
        if name in reg:
            continue
        register_fn(name, func, direction=direction, description=desc)
        n += 1
    return n
