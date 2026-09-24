from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd

from .config import StrategyConfig


def _to_date_str(value) -> str:
    """把 date 字段（Timestamp/datetime/str/其他）统一为 'YYYY-MM-DD' 字符串。"""
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    s = str(value)
    return s[:10]


def add_technical_indicators(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    out = df.copy().sort_values("date").reset_index(drop=True)
    close = out["close"]
    high = out.get("high", close)
    low = out.get("low", close)
    volume = out.get("volume", pd.Series(0, index=out.index))

    out["ma_fast"] = close.rolling(config.fast_ma).mean().clip(lower=0.01)
    out["ma_slow"] = close.rolling(config.slow_ma).mean().clip(lower=0.01)
    out["ma_trend"] = close.rolling(config.trend_ma).mean().clip(lower=0.01)
    out["ma_long_trend"] = close.rolling(config.long_trend_ma).mean().clip(lower=0.01)
    out["ema_12"] = close.ewm(span=12, adjust=False).mean()
    out["ema_26"] = close.ewm(span=26, adjust=False).mean()
    out["macd_dif"] = out["ema_12"] - out["ema_26"]
    out["macd_dea"] = out["macd_dif"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = (out["macd_dif"] - out["macd_dea"]) * 2

    out["rsi_6"] = _rsi(close, 6)
    out["rsi_14"] = _rsi(close, 14)

    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    out["boll_mid"] = mid
    out["boll_upper"] = mid + 2 * std
    out["boll_lower"] = mid - 2 * std
    out["boll_width_pct"] = (out["boll_upper"] - out["boll_lower"]) / mid * 100

    out["atr_14"] = _atr(high, low, close, 14)
    out["atr_pct"] = out["atr_14"] / close * 100

    out["volume_ma"] = volume.rolling(config.volume_ma).mean()
    out["volume_ratio"] = volume / out["volume_ma"].replace(0, float("nan"))
    out["amount_ma20"] = out["amount"].rolling(20).mean() if "amount" in out.columns else pd.Series(0.0, index=out.index)
    delta = close.diff().fillna(0)
    out["obv"] = ((delta.gt(0).astype(int) - delta.lt(0).astype(int)) * volume.fillna(0)).cumsum()

    low_n = low.rolling(9).min()
    high_n = high.rolling(9).max()
    rsv = (close - low_n) / (high_n - low_n).replace(0, math.nan) * 100
    out["kdj_k"] = rsv.ewm(com=2, adjust=False).mean()
    out["kdj_d"] = out["kdj_k"].ewm(com=2, adjust=False).mean()
    out["kdj_j"] = 3 * out["kdj_k"] - 2 * out["kdj_d"]

    out["pct_5"] = close.pct_change(5) * 100
    out["pct_20"] = close.pct_change(20) * 100
    out["pct_60"] = close.pct_change(60) * 100
    # V4.1 fix: shift(1) to avoid lookahead — current close should not be included
    # when computing 60-day extrema for signal generation
    out["drawdown_60_pct"] = (close / close.rolling(60).max().shift(1) - 1) * 100
    out["highest_since_60"] = close.rolling(60).max().shift(1)
    out["lowest_since_60"] = close.rolling(60).min().shift(1)
    out["trend_score"] = _trend_score(out)
    out["momentum_score"] = _momentum_score(out)
    out["volume_score"] = _volume_score(out)
    out["risk_score"] = _risk_score(out)
    out["composite_score"] = (
        out["trend_score"] * 0.35
        + out["momentum_score"] * 0.25
        + out["volume_score"] * 0.2
        + (100 - out["risk_score"]) * 0.2
    ).round(1)

    out["patterns"] = detect_candlestick_patterns(out)
    out["rsrs_beta"] = _rsrs(out)
    beta_mean = out["rsrs_beta"].rolling(300).mean()
    beta_std = out["rsrs_beta"].rolling(300).std()
    out["rsrs_zscore"] = np.where(
        (beta_std > 1e-10) & beta_std.notna(),
        (out["rsrs_beta"] - beta_mean) / beta_std,
        np.nan,
    )

    # ── CCI (Commodity Channel Index) ──
    tp = (high + low + close) / 3                     # 典型价格
    tp_ma = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: abs(x - x.mean()).mean(), raw=True)
    out["cci_20"] = (tp - tp_ma) / (tp_mad.replace(0, np.nan) * 0.015)

    # 小时级别加权均线（日线视角的短期 EMA）
    out["ema_5"] = close.ewm(span=5, adjust=False).mean()
    out["ema_10"] = close.ewm(span=10, adjust=False).mean()
    out["ema_21"] = close.ewm(span=21, adjust=False).mean()

    return out


def detect_candlestick_patterns(df: pd.DataFrame) -> pd.Series:
    """检测 K 线形态，每个交易日返回形态 ID 列表（逗号分隔）。"""
    try:
        return detect_candlestick_patterns_vec(df)
    except Exception:
        return pd.Series("", index=df.index)


def detect_candlestick_patterns_vec(df: pd.DataFrame) -> pd.Series:
    """Vectorized K-line pattern detection; returns comma-separated pattern IDs per row."""
    out = pd.Series("", index=df.index, dtype="object")
    o, c, h, l = df["open"], df["close"], df["high"], df["low"]
    body = (c - o).abs()
    upper = h - o.where(o < c, c)
    lower = o.where(o > c, c) - l
    total = (h - l).replace(0, np.nan)
    body_pct = (body / total).replace([np.inf, -np.inf], np.nan)
    upper_pct = (upper / total).replace([np.inf, -np.inf], np.nan)
    lower_pct = (lower / total).replace([np.inf, -np.inf], np.nan)

    small_body = body_pct < 0.1
    valid_body = body_pct.notna() & (body >= 1e-6)
    po, pc, pb = o.shift(1), c.shift(1), body.shift(1)
    prev_red = pc < po
    prev_green = pc > po
    curr_green = c > o
    curr_red = c < o

    conditions = [
        (valid_body & small_body & (upper_pct < 0.3) & (lower_pct < 0.3), "doji"),
        (valid_body & small_body & (lower_pct > 0.6) & (upper_pct < 0.1), "hammer"),
        (valid_body & small_body & (upper_pct > 0.6) & (lower_pct < 0.1), "shooting_star"),
        (valid_body & (pb >= 1e-6) & curr_green & prev_red & (c > po) & (o < pc), "bullish_engulf"),
        (valid_body & (pb >= 1e-6) & curr_red & prev_green & (o > pc) & (c < po), "bearish_engulf"),
        (valid_body & (pb >= 1e-6) & curr_green & prev_red & (c > (po + pc) / 2) & (o < pc) & (c < po), "piercing"),
        (valid_body & (pb >= 1e-6) & curr_red & prev_green & (o > pc) & (c < (po + pc) / 2), "dark_cloud"),
        (valid_body & (pb > 0) & (body < pb * 0.3), "harami"),
        (valid_body & valid_body.shift(1, fill_value=False) & valid_body.shift(2, fill_value=False)
         & (c.shift(2) > o.shift(2)) & (c.shift(1) > o.shift(1)) & curr_green
         & (c.shift(1) > c.shift(2)) & (c > c.shift(1)), "three_white"),
        (valid_body & valid_body.shift(1, fill_value=False) & valid_body.shift(2, fill_value=False)
         & (c.shift(2) < o.shift(2)) & (c.shift(1) < o.shift(1)) & curr_red
         & (c.shift(1) < c.shift(2)) & (c < c.shift(1)), "three_black"),
        ((c.shift(2) < o.shift(2)) & ((body.shift(1) / total.shift(1)) < 0.3) & curr_green
         & (c > (o.shift(2) + c.shift(2)) / 2), "morning_star"),
        ((c.shift(2) > o.shift(2)) & ((body.shift(1) / total.shift(1)) < 0.3) & curr_red
         & (c < (o.shift(2) + c.shift(2)) / 2), "evening_star"),
    ]

    for cond, tag in conditions:
        mask = cond.fillna(False)
        out = out.mask(mask, out.where(out.eq(""), out + ",") + tag)
    return out


def _rsrs(df: pd.DataFrame, window: int = 18) -> pd.Series:
    """RSRS (阻力支撑相对强度) 择时指标。
    出自光大证券研报, 使用 OLS 回归斜率衡量阻力/支撑强度。
    beta = cov(high, low) / var(low), 滚动 window 天。
    """
    high = df["high"]
    low = df["low"]
    # Rolling regression beta using covariance/variance
    cov = high.rolling(window).cov(low)
    var_low = low.rolling(window).var().clip(lower=1e-10)
    beta = cov / var_low
    # P2-Q28-fix(M359): 预热期（前 window-1 行）保持 NaN 而非 fillna(0.0)。
    # 原 0 填充会污染滚动 300 日均值/标准差，使 rsrs_zscore 预热期失真。
    return beta


def latest_indicator_snapshot(symbol: str, df: pd.DataFrame, config: StrategyConfig) -> dict:
    data = add_technical_indicators(df, config).dropna(subset=["ma_slow", "ma_trend"])
    if data.empty:
        return {"symbol": symbol, "status": "insufficient_data", "rows": len(df)}
    row = data.iloc[-1]
    keys = [
        "close",
        "pct_chg",
        "pct_5",
        "pct_20",
        "pct_60",
        "ma_fast",
        "ma_slow",
        "ma_trend",
        "ma_long_trend",
        "macd_dif",
        "macd_dea",
        "macd_hist",
        "rsi_6",
        "rsi_14",
        "kdj_k",
        "kdj_d",
        "kdj_j",
        "boll_upper",
        "boll_mid",
        "boll_lower",
        "boll_width_pct",
        "atr_14",
        "atr_pct",
        "volume_ratio",
        "turnover",
        "drawdown_60_pct",
        "trend_score",
        "momentum_score",
        "volume_score",
        "risk_score",
        "composite_score",
        "rsrs_beta",
        "rsrs_zscore",
    ]
    # P2-Q28-fix(M363): 兼容字符串日期与 datetime/timestamp（原 row["date"].date()
    # 传入字符串日期会 AttributeError）。
    snapshot = {"symbol": symbol, "date": _to_date_str(row["date"])}
    for key in keys:
        if key in row and pd.notna(row[key]):
            snapshot[key] = round(float(row[key]), 3)
    # 均线保留 0.01 精度
    for key in ["ma_fast", "ma_slow", "ma_trend", "ma_long_trend", "stop_loss", "take_profit_ref", "trailing_stop_ref"]:
        if key in snapshot and snapshot.get(key) is not None:
            snapshot[key] = round(float(snapshot[key]), 2)
    return snapshot


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, math.nan)
    return 100 - 100 / (1 + rs)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean()


def _trend_score(out: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=out.index)
    score += (out["close"] > out["ma_fast"]).astype(float) * 20
    score += (out["ma_fast"] > out["ma_slow"]).astype(float) * 25
    score += (out["ma_slow"] > out["ma_trend"]).astype(float) * 25
    score += (out["ma_trend"] > out["ma_long_trend"]).astype(float) * 20
    # P2-Q28-fix(M358): 去掉多余 shift(1)。highest_since_60 在 add_technical_indicators
    # L60 已做 shift(1) 防前视，此处再 shift 会令"创60日新高"加分晚一天，且与
    # drawdown_60_pct（L59 单次 shift）口径不一致。
    score += (out["close"] >= out["highest_since_60"]).astype(float) * 10
    return score.clip(0, 100)


def _momentum_score(out: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=out.index)
    score += out["pct_20"].clip(-20, 20).fillna(0) + 20
    score += (out["macd_hist"] > 0).astype(float) * 20
    score += out["rsi_14"].sub(30).clip(0, 40).fillna(0)
    score += (out["kdj_k"] > out["kdj_d"]).astype(float) * 20
    return score.clip(0, 100)


def _volume_score(out: pd.DataFrame) -> pd.Series:
    ratio = out["volume_ratio"].replace([float("inf"), -float("inf")], math.nan).fillna(1.0)
    score = ratio.clip(0, 2.5) / 2.5 * 70
    score += (out["obv"] > out["obv"].rolling(20).mean()).astype(float) * 30
    return score.clip(0, 100)


def _risk_score(out: pd.DataFrame) -> pd.Series:
    score = pd.Series(20.0, index=out.index)
    score += out["atr_pct"].clip(0, 12).fillna(0) * 4
    score += out["drawdown_60_pct"].abs().clip(0, 40).fillna(0)
    score += (out["rsi_14"] > 80).astype(float) * 15
    score += (out["close"] < out["ma_slow"]).astype(float) * 20
    return score.clip(0, 100)


# --------------------------------------------------------------------------
# Intraday (minute-level) indicators — lighter version
# --------------------------------------------------------------------------

def add_intraday_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add lightweight technical indicators for minute-level K-lines."""
    out = df.copy().sort_values("date").reset_index(drop=True)
    close = out["close"]
    high = out.get("high", close)
    low = out.get("low", close)

    # 均线系
    for p in [5, 10, 20, 60]:
        out[f"ma_{p}"] = close.rolling(p).mean()

    # EMA 系（小时级别的EMA 5/10/21 = ~半日/1日/1周）
    out["ema_5"] = close.ewm(span=5, adjust=False).mean()
    out["ema_10"] = close.ewm(span=10, adjust=False).mean()
    out["ema_21"] = close.ewm(span=21, adjust=False).mean()

    out["ema_12"] = close.ewm(span=12, adjust=False).mean()
    out["ema_26"] = close.ewm(span=26, adjust=False).mean()
    out["macd_dif"] = out["ema_12"] - out["ema_26"]
    out["macd_dea"] = out["macd_dif"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = (out["macd_dif"] - out["macd_dea"]) * 2

    out["rsi_6"] = _rsi(close, 6)
    out["rsi_14"] = _rsi(close, 14)

    # CCI (hourly级别用 10 周期)
    tp = (high + low + close) / 3
    tp_ma = tp.rolling(10).mean()
    tp_mad = tp.rolling(10).apply(lambda x: abs(x - x.mean()).mean(), raw=True)
    out["cci_10"] = (tp - tp_ma) / (tp_mad.replace(0, float("nan")) * 0.015)

    # Volume MA
    if "volume" in out.columns:
        out["volume_ma"] = out["volume"].rolling(20).mean()
        out["volume_ratio"] = out["volume"] / out["volume_ma"].replace(0, float("nan"))
    else:
        out["volume_ratio"] = 1.0

    # ATR
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(14).mean()
    out["atr"] = atr
    out["atr_pct"] = (atr / close * 100).clip(0, 20)

    # MA交叉信号
    out["ma_cross"] = 0
    out.loc[out["ema_5"] > out["ema_21"], "ma_cross"] = 1   # 金叉做多
    out.loc[out["ema_5"] < out["ema_21"], "ma_cross"] = -1  # 死叉做空

    # 均线多头排列/空头排列
    out["ma_bull"] = ((out["ma_5"] > out["ma_10"]) &
                       (out["ma_10"] > out["ma_20"]) &
                       (out["ma_20"] > out["ma_60"])).astype(int)
    out["ma_bear"] = ((out["ma_5"] < out["ma_10"]) &
                       (out["ma_10"] < out["ma_20"]) &
                       (out["ma_20"] < out["ma_60"])).astype(int)

    return out


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return MACD DIF, DEA, and histogram for a close series."""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    return dif, dea, (dif - dea) * 2


def _kdj(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return KDJ K, D, and J lines."""
    low_n = low.rolling(window).min()
    high_n = high.rolling(window).max()
    rsv = (close - low_n) / (high_n - low_n).replace(0, np.nan) * 100
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    return k, d, 3 * k - 2 * d


def _boll(close: pd.Series, window: int = 20, mult: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return Bollinger middle, upper, and lower bands."""
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    return mid, mid + mult * std, mid - mult * std


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Return Wilder ADX for trend strength."""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)
    atr = _atr(high, low, close, window).replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1 / window, adjust=False).mean()


def add_minute_indicators(df_5min: pd.DataFrame, df_30min: pd.DataFrame, df_60min: pd.DataFrame) -> pd.DataFrame:
    """Compute and combine 5/30/60-minute technical indicators with min5/min30/min60 prefixes.

    P2-Q28-fix(M360): 原实现把不同长度的分钟帧按位置索引 concat —— 短帧尾部行对
    长帧是 NaN、不同频率行不对齐（时间错乱）。现按时间戳向后对齐（merge_asof,
    direction='backward'）：每个时间点取各帧在该时刻或之前最近一根 K 线的指标值，
    保证同一行的 min5/min30/min60 列属于同一时刻。本函数当前无外部调用方，
    作为潜在工具保留，契约见 docstring。
    """
    try:
        m5 = df_5min.copy().sort_values("date").reset_index(drop=True)
        c5 = m5["close"]
        volume5 = m5.get("volume", pd.Series(0.0, index=m5.index))
        m5["ema5"] = c5.ewm(span=5, adjust=False).mean()
        m5["ema10"] = c5.ewm(span=10, adjust=False).mean()
        m5["rsi14"] = _rsi(c5, 14)
        m5["vol_ratio"] = volume5 / volume5.rolling(20).mean().replace(0, np.nan)
        m5["pct_5bars"] = c5.pct_change(5) * 100

        m30 = df_30min.copy().sort_values("date").reset_index(drop=True)
        c30 = m30["close"]
        h30, l30 = m30.get("high", c30), m30.get("low", c30)
        m30["ema5"] = c30.ewm(span=5, adjust=False).mean()
        m30["ema10"] = c30.ewm(span=10, adjust=False).mean()
        m30["macd_dif"], m30["macd_dea"], m30["macd_hist"] = _macd(c30)
        m30["kdj_k"], m30["kdj_d"], m30["kdj_j"] = _kdj(h30, l30, c30)
        m30["boll_mid"], m30["boll_upper"], m30["boll_lower"] = _boll(c30)
        m30["atr14"] = _atr(h30, l30, c30, 14)

        m60 = df_60min.copy().sort_values("date").reset_index(drop=True)
        c60 = m60["close"]
        h60, l60 = m60.get("high", c60), m60.get("low", c60)
        m60["ema5"] = c60.ewm(span=5, adjust=False).mean()
        m60["ema10"] = c60.ewm(span=10, adjust=False).mean()
        m60["macd_dif"], m60["macd_dea"], m60["macd_hist"] = _macd(c60)
        m60["rsi14"] = _rsi(c60, 14)
        m60["boll_mid"], m60["boll_upper"], m60["boll_lower"] = _boll(c60)
        m60["kdj_k"], m60["kdj_d"], m60["kdj_j"] = _kdj(h60, l60, c60)
        tp = (h60 + l60 + c60) / 3
        tp_ma = tp.rolling(20).mean()
        tp_mad = tp.rolling(20).apply(lambda x: abs(x - x.mean()).mean(), raw=True)
        m60["cci20"] = (tp - tp_ma) / (tp_mad.replace(0, np.nan) * 0.015)
        m60["adx14"] = _adx(h60, l60, c60, 14)

        # 以 5 分钟帧的时间轴为基准（最细粒度）
        base = pd.to_datetime(m5["date"]).drop_duplicates().sort_values().rename("_ts")
        base_df = pd.DataFrame({"_ts": base})

        def _align_and_prefix(m: pd.DataFrame, prefix: str, cols: list[str]) -> pd.DataFrame:
            sub = m[["date"] + cols].copy()
            sub["_ts"] = pd.to_datetime(sub["date"])
            sub = sub.drop_duplicates("_ts").sort_values("_ts")
            aligned = pd.merge_asof(base_df, sub, on="_ts", direction="backward")
            aligned = aligned.set_index("_ts")[cols]
            aligned.columns = [f"{prefix}_{c}" for c in cols]
            return aligned

        frames = [
            _align_and_prefix(m5, "min5", ["ema5", "ema10", "rsi14", "vol_ratio", "pct_5bars"]),
            _align_and_prefix(m30, "min30", ["ema5", "ema10", "macd_dif", "macd_dea", "macd_hist", "kdj_k", "kdj_d", "kdj_j", "boll_mid", "boll_upper", "boll_lower", "atr14"]),
            _align_and_prefix(m60, "min60", ["ema5", "ema10", "macd_dif", "macd_dea", "macd_hist", "rsi14", "boll_mid", "boll_upper", "boll_lower", "kdj_k", "kdj_d", "kdj_j", "cci20", "adx14"]),
        ]
        return pd.concat(frames, axis=1)
    except Exception:
        return pd.DataFrame(index=df_5min.index if df_5min is not None else None)


def detect_minute_level_divergence(df_60min: pd.DataFrame) -> dict:
    """Detect 60-minute price/MACD DIF divergence and return type plus 0-100 strength."""
    try:
        df = df_60min.copy().sort_values("date").reset_index(drop=True)
        close = df["close"]
        dif = df["macd_dif"] if "macd_dif" in df.columns else _macd(close)[0]
        if len(df) < 10 or close.dropna().empty or dif.dropna().empty:
            return {"divergence_type": "none", "strength": 0}

        recent = min(20, len(df) // 2)
        prev_close = close.iloc[-recent * 2:-recent]
        curr_close = close.iloc[-recent:]
        prev_dif = dif.iloc[-recent * 2:-recent]
        curr_dif = dif.iloc[-recent:]
        if prev_close.empty or curr_close.empty:
            return {"divergence_type": "none", "strength": 0}

        # V11 审计修复（Medium）: 原实现背离方向错乱——
        # 顶背离比较 DIF 低点、底背离比较 DIF 高点，与经典 MACD 背离定义相反。
        # 正确语义：
        #   顶背离 = 价格创新高（curr>prev），但 DIF 高点走低（curr_dif.max() < prev_dif.max()）
        #   底背离 = 价格创新低（curr<prev），但 DIF 低点走高（curr_dif.min() > prev_dif.min()）
        price_high_gain = curr_close.max() - prev_close.max()
        dif_high_drop = prev_dif.max() - curr_dif.max()   # >0 表示 DIF 高点走低（顶背离）
        price_low_drop = prev_close.min() - curr_close.min()
        dif_low_rise = curr_dif.min() - prev_dif.min()    # >0 表示 DIF 低点走高（底背离）

        if price_high_gain > 0 and dif_high_drop > 0:
            price_strength = price_high_gain / max(abs(prev_close.max()), 1e-12)
            macd_strength = dif_high_drop / max(abs(prev_dif.max()), 1e-12)
            return {"divergence_type": "bearish", "strength": int(np.clip((price_strength + macd_strength) * 500, 1, 100))}
        if price_low_drop > 0 and dif_low_rise > 0:
            price_strength = price_low_drop / max(abs(prev_close.min()), 1e-12)
            macd_strength = dif_low_rise / max(abs(prev_dif.min()), 1e-12)
            return {"divergence_type": "bullish", "strength": int(np.clip((price_strength + macd_strength) * 500, 1, 100))}
        return {"divergence_type": "none", "strength": 0}
    except Exception:
        return {"divergence_type": "none", "strength": 0}


def compute_ma_slope(ma_series: pd.Series, lookback: int = 20) -> pd.Series:
    """Compute rolling linear-regression MA slope angle in degrees, clipped to [-90, 90]."""
    lookback = max(int(lookback), 2)
    x = np.arange(lookback, dtype=float)
    x_centered = x - x.mean()
    denom = float(np.sum(x_centered ** 2))

    def _angle(values: np.ndarray) -> float:
        if np.isnan(values).any() or denom <= 0:
            return np.nan
        slope = float(np.sum(x_centered * (values - values.mean())) / denom)
        return float(np.degrees(np.arctan(slope)))

    return ma_series.rolling(lookback).apply(_angle, raw=True).clip(-90, 90)


def add_advanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add advanced price/volume indicators including Chandelier, Keltner, PVT, TWAP/VWAP, and reversal strength.

    P2-Q28-fix(M374): 本函数当前无调用方（死代码），修正两处语义错误后保留：
      1) vwap 在 0 成交量日产生 NaN，现用前值填充（ffill），避免下游 NaN 传播；
      2) 移除原 herfindahl 列——它用单只股票市值的滚动占比计算"集中度"，
         而 HHI 是跨截面的概念，单票时序无法计算，原值语义错误且无调用方消费。
    """
    try:
        out = df.copy().sort_values("date").reset_index(drop=True)
        close = out["close"]
        high = out.get("high", close)
        low = out.get("low", close)
        open_ = out.get("open", close)
        volume = out.get("volume", pd.Series(0.0, index=out.index)).replace(0, np.nan)
        atr14 = out["atr_14"] if "atr_14" in out.columns else _atr(high, low, close, 14)

        out["chandelier_long_stop"] = high.rolling(22).max() - atr14 * 3
        out["chandelier_short_stop"] = low.rolling(22).min() + atr14 * 3
        out["keltner_mid"] = close.ewm(span=20, adjust=False).mean()
        out["keltner_upper"] = out["keltner_mid"] + atr14 * 1.5
        out["keltner_lower"] = out["keltner_mid"] - atr14 * 1.5
        delta_close = close.pct_change().fillna(0.0)
        out["pvt"] = (delta_close * volume.fillna(0.0)).cumsum()
        out["twap"] = (open_ + high + low + close) / 4
        out["twap_deviation_pct"] = (close / out["twap"].replace(0, np.nan) - 1) * 100
        if "amount" in out.columns:
            # P2-Q28-fix(M374): 0 成交量日 amount/volume=NaN，ffill 用前值填充；
            # 前导 NaN（序列开头即无量）无前值可填，用收盘价兜底，消除下游 NaN 传播。
            out["vwap"] = (out["amount"] / volume).ffill().fillna(close)
            out["vwap_deviation_pct"] = (close / out["vwap"].replace(0, np.nan) - 1) * 100
        else:
            out["vwap"] = np.nan
            out["vwap_deviation_pct"] = np.nan

        # V4.1 fix: shift(1) to avoid lookahead in 60-day high/low
        low60 = close.rolling(60).min().shift(1)
        high60 = close.rolling(60).max().shift(1)
        position = (close - low60) / (high60 - low60).replace(0, np.nan)
        out["reversal_strength"] = ((1 - position) * 100).clip(0, 100)
        out["reversal_ratio_60"] = (low60 / high60.replace(0, np.nan)).clip(0, 1)
        return out
    except Exception:
        return df.copy()
