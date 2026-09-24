# 已融合: 原 quant_v6/strategy/intraday_signal.py 移植为 quant_system 自包含实现（2026-08-08）
# -*- coding: utf-8 -*-
"""
intraday_signal.py — 分钟线盘中信号 (V6.1, T1: 分钟线盘中决策闭环)

问题背景: 原 opportunity_monitor 盘中扫描用日 K 指标判定, 日 K 收盘才定型,
盘中用日 K 判断买卖点毫无意义。用户要求: 定时监控必须用分时(分钟线)检测,
收盘后才发日 K。

本模块:
    1. 从分钟 K (5/15/30/60) 计算盘中信号: 分时RSI/CCI/MACD/均线支撑/量比/价格异动
    2. 统一信号接口, 输出与日 K 扫描同构的 {signals, signal_reasons, signal_count}
    3. 纯 pandas/numpy 实现, 不依赖 quant_system 内部结构

数据源: 新浪分钟线 fetch_sina_intraday (Vultr 实测可用; 东财被风控)。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# 指标计算 (纯函数)
# ════════════════════════════════════════════════════════════════

def _sma(vals: np.ndarray, n: int) -> np.ndarray:
    """简单移动平均 (返回与输入等长, 前 n-1 个为 NaN)。"""
    if len(vals) < n:
        return np.full(len(vals), np.nan)
    out = np.full(len(vals), np.nan)
    cum = np.cumsum(vals)
    out[n - 1:] = (cum[n - 1:] - np.concatenate([[0], cum[:-n]])) / n
    return out


def _ema(vals: np.ndarray, n: int) -> np.ndarray:
    """指数移动平均 (返回与输入等长)。"""
    if len(vals) == 0:
        return vals
    alpha = 2.0 / (n + 1)
    out = np.empty(len(vals))
    out[0] = vals[0]
    for i in range(1, len(vals)):
        out[i] = alpha * vals[i] + (1 - alpha) * out[i - 1]
    return out


def _rsi(closes: np.ndarray, n: int = 14) -> float:
    """RSI 值 (最后一点)。"""
    if len(closes) < n + 1:
        return 50.0
    delta = np.diff(closes)
    gain = np.clip(delta, 0, None)
    loss = np.clip(-delta, 0, None)
    avg_gain = gain[-n:].mean()
    avg_loss = loss[-n:].mean()
    if avg_loss < 1e-12:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def _cci(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, n: int = 20) -> float:
    """CCI 值 (最后一点)。"""
    if len(closes) < n:
        return 0.0
    tp = (highs + lows + closes) / 3.0
    last_tp = tp[-n:]
    mean_tp = last_tp.mean()
    md = np.abs(last_tp - mean_tp).mean()
    if md < 1e-12:
        return 0.0
    return float((tp[-1] - mean_tp) / (0.015 * md))


def _macd(closes: np.ndarray) -> tuple[float, float, float]:
    """MACD: (DIF, DEA, HIST) 最后一点。"""
    if len(closes) < 35:
        return 0.0, 0.0, 0.0
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    dif = ema12 - ema26
    dea = _ema(dif, 9)
    hist = (dif[-1] - dea[-1]) * 2
    return float(dif[-1]), float(dea[-1]), float(hist)


def _vol_ratio(volumes: np.ndarray, n: int = 20) -> float:
    """量比: 最后一根 vs 前 n 根均值。"""
    if len(volumes) < n + 1:
        return 0.0
    avg = volumes[-n - 1:-1].mean()
    if avg < 1e-9:
        return 0.0
    return float(volumes[-1] / avg)


# ════════════════════════════════════════════════════════════════
# 分钟 K → 信号
# ════════════════════════════════════════════════════════════════

def compute_intraday_signals(df: pd.DataFrame, symbol: str = "",
                             price: Optional[float] = None) -> dict[str, Any]:
    """从分钟 K 线 DataFrame 计算盘中信号。

    Args:
        df: 分钟 K, 至少含 open/high/low/close/volume 列, 按时间升序。
        symbol: 股票代码 (仅用于标识)。
        price: 当前实时价 (可选, 优先用实时价判断; 缺省用最后一根 close)。

    Returns:
        {
          "symbol": str, "signals": {信号名: 权重}, "signal_reasons": {信号名: 描述},
          "signal_count": 加权总分, "rsi": .., "cci": .., "macd_hist": ..,
          "ma5": .., "ma10": .., "vol_ratio": .., "bars": 分钟K根数, "period": ..
        }
        数据不足返回空 dict。
    """
    if df is None or len(df) < 30:
        return {}
    try:
        closes = df["close"].astype(float).values
        highs = df["high"].astype(float).values
        lows = df["low"].astype(float).values
        volumes = df["volume"].astype(float).values
    except Exception as exc:
        logger.warning("分钟K列解析失败 %s: %s", symbol, exc)
        return {}

    cur_price = float(price) if price and price > 0 else float(closes[-1])

    signals: dict[str, float] = {}
    reasons: dict[str, str] = {}

    # ── I1: 分时均线支撑 (MA5/MA10/MA20 分钟) ──
    ma5 = _sma(closes, 5)
    ma10 = _sma(closes, 10)
    ma20 = _sma(closes, 20)
    ma5_v = ma5[-1] if np.isfinite(ma5[-1]) else cur_price
    ma10_v = ma10[-1] if np.isfinite(ma10[-1]) else cur_price
    ma20_v = ma20[-1] if np.isfinite(ma20[-1]) else cur_price

    # V6.1-fix: 支撑判定收窄 (0.8%→0.4%), 且要求均线向上(斜率>0),
    # 否则 351/351 全命中无区分度。
    def _near(px: float, target: float, pct: float = 0.004) -> bool:
        if target <= 0:
            return False
        return abs(px / target - 1) <= pct

    ma5_up = len(closes) >= 6 and closes[-1] > closes[-6]
    ma10_up = len(closes) >= 11 and closes[-1] > closes[-11]
    ma20_up = len(closes) >= 21 and closes[-1] > closes[-21]

    if _near(cur_price, ma5_v) and ma5_up:
        signals["m5_support"] = 0.8
        reasons["m5_support"] = f"分时MA5=¥{ma5_v:.2f} 附近支撑(上行)"
    if _near(cur_price, ma10_v) and ma10_up:
        signals["m10_support"] = 1.0
        reasons["m10_support"] = f"分时MA10=¥{ma10_v:.2f} 附近支撑(上行)"
    if _near(cur_price, ma20_v) and ma20_up:
        signals["m20_support"] = 1.2
        reasons["m20_support"] = f"分时MA20=¥{ma20_v:.2f} 附近支撑(上行)"

    # 趋势向上才给支撑分 (均线斜率)
    if ma10_v > ma20_v and cur_price > ma10_v:
        signals["m_trend_up"] = 0.8
        reasons["m_trend_up"] = f"分时多头 MA10>MA20"
    # 均线向下(空头)时若价格反抽到 MA10 下方, 不给支撑分但可提示
    elif ma10_v < ma20_v:
        signals["m_trend_down"] = -0.8  # 负分压制
        reasons["m_trend_down"] = f"分时空头 MA10<MA20"

    # ── I2: 分时 RSI ──
    rsi = _rsi(closes, 14)
    if rsi < 30:
        signals["m_rsi_oversold"] = 1.2
        reasons["m_rsi_oversold"] = f"分时RSI={rsi:.0f} 超卖"
    elif rsi < 40:
        signals["m_rsi_near_oversold"] = 0.6
        reasons["m_rsi_near_oversold"] = f"分时RSI={rsi:.0f} 接近超卖"
    if len(closes) >= 15:
        rsi_prev = _rsi(closes[:-1], 14)
        if rsi_prev < 35 and rsi > rsi_prev + 5:
            signals["m_rsi_turn_up"] = 0.8
            reasons["m_rsi_turn_up"] = f"分时RSI回升 {rsi_prev:.0f}→{rsi:.0f}"

    # ── I3: 分时 CCI ──
    cci = _cci(highs, lows, closes, 20)
    if cci < -150:
        signals["m_cci_oversold"] = 1.0
        reasons["m_cci_oversold"] = f"分时CCI={cci:.0f} 超卖"
    elif cci < -100:
        signals["m_cci_near_oversold"] = 0.5
        reasons["m_cci_near_oversold"] = f"分时CCI={cci:.0f} 接近超卖"
    if cci > -50 and len(closes) >= 21:
        cci_prev = _cci(highs[:-1], lows[:-1], closes[:-1], 20)
        if cci_prev < -100 and cci > cci_prev:
            signals["m_cci_turn_up"] = 0.8
            reasons["m_cci_turn_up"] = f"分时CCI回升 {cci_prev:.0f}→{cci:.0f}"

    # ── I4: 分时 MACD ──
    dif, dea, hist = _macd(closes)
    # V11 审计修复（Medium）: 原实现 `hist > 0 and hist > 0` 完全冗余（恒等 hist>0）。
    # 意图是"红柱且动能增强"——修正为红柱且柱体较前值扩大（hist 加速）。
    hist_expanding = False
    if len(closes) >= 36:
        ema12_p = _ema(closes[:-1], 12)
        ema26_p = _ema(closes[:-1], 26)
        dif_p = ema12_p - ema26_p
        dea_p = _ema(dif_p, 9)
        hist_prev = (dif_p[-1] - dea_p[-1]) * 2 if len(dif_p) else 0.0
        hist_expanding = hist > hist_prev
    if hist > 0 and hist_expanding:
        signals["m_macd_positive"] = 0.7
        reasons["m_macd_positive"] = f"分时MACD红柱扩大 {hist:.5f}"
    if len(closes) >= 36:
        ema12 = _ema(closes, 12)
        ema26 = _ema(closes, 26)
        dif_list = ema12 - ema26
        dea_list = _ema(dif_list, 9)
        hist_prev = (dif_list[-2] - dea_list[-2]) * 2
        if hist_prev < 0 and hist > 0:
            signals["m_macd_cross"] = 1.3
            reasons["m_macd_cross"] = f"分时MACD金叉 {hist_prev:.5f}→{hist:.5f}"
        elif hist_prev < 0 and hist > hist_prev:
            signals["m_macd_shrink"] = 0.5
            reasons["m_macd_shrink"] = f"分时MACD绿柱缩脚 {hist_prev:.5f}→{hist:.5f}"

    # ── I5: 分时量比 ──
    vr = _vol_ratio(volumes, 20)
    if vr > 1.8:
        signals["m_vol_surge"] = 0.8
        reasons["m_vol_surge"] = f"分时量比 {vr:.1f} 放量"
    elif vr > 1.3:
        signals["m_vol_active"] = 0.4
        reasons["m_vol_active"] = f"分时量比 {vr:.1f}"

    # ── I6: 价格异动 (分时 V 型 / 日内位置) ──
    day_high = float(highs.max())
    day_low = float(lows.min())
    if day_high > day_low > 0:
        day_pos = (cur_price - day_low) / (day_high - day_low)
        if day_pos < 0.15:
            signals["m_day_low"] = 0.6
            reasons["m_day_low"] = f"日内位置 {day_pos:.0%} 接近最低"
        if day_pos > 0.85:
            signals["m_day_high"] = 0.5
            reasons["m_day_high"] = f"日内位置 {day_pos:.0%} 接近最高"
    # 尾盘拉升: 最近3根收盘价连续上行
    if len(closes) >= 4 and closes[-1] > closes[-2] > closes[-3] and closes[-3] >= closes[-4]:
        signals["m_late_rise"] = 0.9
        reasons["m_late_rise"] = "分时尾盘连拉3根"

    # 加权总分 (与日K _weighted_score 同构: 正分求和, 负分压制计入)
    total = sum(v for v in signals.values() if v > 0)
    neg = sum(v for v in signals.values() if v < 0)
    total += neg
    return {
        "symbol": symbol,
        "signals": signals,
        "signal_reasons": reasons,
        "signal_count": round(total, 2),
        "rsi": round(rsi, 1),
        "cci": round(cci, 1),
        "macd_hist": round(hist, 6),
        "ma5": round(ma5_v, 2),
        "ma10": round(ma10_v, 2),
        "ma20": round(ma20_v, 2),
        "vol_ratio": round(vr, 2),
        "day_pos": round(day_pos, 3) if day_high > day_low else None,
        "bars": int(len(closes)),
        "price": round(cur_price, 2),
    }


if __name__ == "__main__":  # pragma: no cover - 手工调试入口
    # 合成 60 根 5 分钟 K 验证
    np.random.seed(1)
    n = 60
    base = 10.0
    closes = base + np.cumsum(np.random.randn(n) * 0.01)
    highs = closes + 0.01
    lows = closes - 0.01
    volumes = np.random.randint(1000, 5000, n).astype(float)
    idx = pd.date_range("2026-08-04 09:35", periods=n, freq="5min")
    df = pd.DataFrame({"open": closes, "high": highs, "low": lows,
                       "close": closes, "volume": volumes}, index=idx)
    result = compute_intraday_signals(df, "600519")
    print("信号数:", len(result.get("signals", {})), "| 总分:", result.get("signal_count"))
    for k, v in result.get("signal_reasons", {}).items():
        print(f"  {k}: {v}")
