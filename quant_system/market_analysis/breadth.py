"""
market_analysis/breadth.py — 市场宽度分析
V4.1 feature | 深度重构

提供A/D线、McClellan摆动、均线上占比、新高新低、宽度推力等核心宽度指标。
"""

import numpy as np
import pandas as pd
from typing import Optional

try:
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


# ═══════════════════════════════════════════════════════
# 1. Advance/Decline Line (A/D线)
# ═══════════════════════════════════════════════════════

class AdvanceDeclineLine:
    """A/D线 — 市场宽度的核心指标，衡量上涨股票数量与下跌股票数量的累积差值
    
    A/D线（累积）比价格指数更早发出转折信号，是最常用的大盘宽度指标。
    顶背离：价格创新高但A/D线未创新高 → 下跌风险
    底背离：价格创新低但A/D线未创新低 → 上涨机会
    """

    def __init__(self):
        pass

    def ad_line(self, advances: pd.Series, declines: pd.Series) -> pd.Series:
        """净A/D线: advances - declines, 返回每日净差值
        
        Parameters
        ----------
        advances : pd.Series
            每日上涨家数，index为日期
        declines : pd.Series
            每日下跌家数，index为日期
        
        Returns
        -------
        pd.Series
            每日净上涨家数（上涨-下跌）
        """
        return advances - declines

    def cumulative_ad(self, advances: pd.Series, declines: pd.Series) -> pd.Series:
        """累积A/D线: 对净差值累积求和，是真正的A/D线
        
        Parameters
        ----------
        advances : pd.Series
            每日上涨家数
        declines : pd.Series
            每日下跌家数
        
        Returns
        -------
        pd.Series
            累积A/D线
        """
        net = advances - declines
        return net.cumsum()

    def ad_divergence(self, price_index: pd.Series, ad_series: pd.Series,
                      lookback: int = 20) -> dict:
        """检测A/D线与价格指数的背离
        
        顶背离：价格创N日新高但A/D未创N日新高
        底背离：价格创N日新低但A/D未创N日新低
        
        Parameters
        ----------
        price_index : pd.Series
            价格指数（如沪深300收盘价）
        ad_series : pd.Series
            累积A/D线
        lookback : int
            回溯窗口（默认20日）
        
        Returns
        -------
        dict
            has_bullish_divergence: 是否底背离（看多）
            has_bearish_divergence: 是否顶背离（看空）
            divergence_days: 背离持续天数
            divergence_strength: 背离强度(0-1)
        """
        common = price_index.index.intersection(ad_series.index)
        if len(common) < lookback + 5:
            return {"has_bullish_divergence": False, "has_bearish_divergence": False,
                    "divergence_days": 0, "divergence_strength": 0.0}

        price = price_index.loc[common]
        ad = ad_series.loc[common]

        # 近期最高/最低价
        price_high = price.rolling(lookback).max()
        price_low = price.rolling(lookback).min()
        ad_high = ad.rolling(lookback).max()
        ad_low = ad.rolling(lookback).min()

        latest_price = price.iloc[-1]
        latest_ad = ad.iloc[-1]
        recent_price_high = price_high.iloc[-1]
        recent_price_low = price_low.iloc[-1]
        recent_ad_high = ad_high.iloc[-1]
        recent_ad_low = ad_low.iloc[-1]

        # 顶背离：价格创N日新高但A/D未创新高
        bearish = (latest_price >= recent_price_high * 0.995 and
                   latest_ad < recent_ad_high * 0.98)

        # 底背离：价格创N日新低但A/D未创新低
        bullish = (latest_price <= recent_price_low * 1.005 and
                   latest_ad > recent_ad_low * 0.98)

        # 背离持续天数
        divergence_days = 0
        if bearish or bullish:
            for i in range(min(lookback, len(price) - 1), 0, -1):
                p = price.iloc[-i]
                a = ad.iloc[-i]
                if bearish:
                    if p >= price_high.iloc[-i] * 0.995 and a < ad_high.iloc[-i] * 0.98:
                        divergence_days += 1
                else:
                    if p <= price_low.iloc[-i] * 1.005 and a > ad_low.iloc[-i] * 0.98:
                        divergence_days += 1

        # 背离强度：基于价格与A/D的偏离程度
        strength = 0.0
        if bearish:
            price_ratio = (latest_price - recent_price_high * 0.995) / max(abs(recent_price_high), 1)
            ad_ratio = (recent_ad_high - latest_ad) / max(abs(recent_ad_high), 1)
            strength = min(1.0, max(0, (price_ratio + ad_ratio) / 2))
        elif bullish:
            price_ratio = (recent_price_low * 1.005 - latest_price) / max(abs(recent_price_low), 1)
            ad_ratio = (latest_ad - recent_ad_low * 0.98) / max(abs(recent_ad_low), 1)
            strength = min(1.0, max(0, (price_ratio + ad_ratio) / 2))

        return {
            "has_bullish_divergence": bool(bullish),
            "has_bearish_divergence": bool(bearish),
            "divergence_days": min(divergence_days, lookback),
            "divergence_strength": round(strength, 2),
        }

    def ad_ma_cross(self, ad_series: pd.Series, short: int = 20, long: int = 60) -> pd.Series:
        """A/D线均线交叉信号
        
        金叉(1): 短期均线上穿长期均线
        死叉(-1): 短期均线下穿长期均线
        无信号(0)
        
        Parameters
        ----------
        ad_series : pd.Series
            累积A/D线
        short : int
            短期均线周期（默认20）
        long : int
            长期均线周期（默认60）
        
        Returns
        -------
        pd.Series
            信号: 1=金叉, -1=死叉, 0=无
        """
        ma_short = ad_series.rolling(short).mean()
        ma_long = ad_series.rolling(long).mean()
        prev_diff = (ma_short.shift(1) - ma_long.shift(1))
        curr_diff = (ma_short - ma_long)
        cross = pd.Series(0, index=ad_series.index)
        cross[(prev_diff <= 0) & (curr_diff > 0)] = 1  # 金叉
        cross[(prev_diff >= 0) & (curr_diff < 0)] = -1  # 死叉
        return cross

    def ad_breakout(self, ad_series: pd.Series, period: int = 20,
                    threshold: float = 1.5) -> pd.Series:
        """A/D线突破信号：创N日新高且涨幅超过阈值标准差
        
        Parameters
        ----------
        ad_series : pd.Series
            累积A/D线
        period : int
            突破窗口（默认20）
        threshold : float
            标准差倍数阈值（默认1.5）
        
        Returns
        -------
        pd.Series
            突破强度信号（>0表示突破向上）
        """
        ad_high = ad_series.rolling(period).max()
        ad_std = ad_series.rolling(period).std()
        signal = (ad_series - ad_high.shift(1)) / ad_std.clip(lower=1e-12)
        return signal.clip(lower=0) * (signal > threshold)


# ═══════════════════════════════════════════════════════
# 2. McClellan Oscillator (麦氏摆动指标)
# ═══════════════════════════════════════════════════════

class McClellanOscillator:
    """McClellan摆动指标 — 基于涨跌家数净差值的EMA差值
    
    原理：上涨下跌家数的净差值做EMA19 - EMA39，判断市场短期动能。
    震荡区间通常为±100，超过+100为超买，低于-100为超卖。
    合指标(SI)：对摆动指标再取EMA19，判断中期趋势。
    """

    def __init__(self):
        pass

    def mcclellan_oscillator(self, advances: pd.Series, declines: pd.Series,
                              ema_short: int = 19, ema_long: int = 39) -> pd.Series:
        """McClellan摆动指标: EMA19(净差) - EMA39(净差)
        
        Parameters
        ----------
        advances : pd.Series
            上涨家数
        declines : pd.Series
            下跌家数
        ema_short : int
            短期EMA参数（默认19）
        ema_long : int
            长期EMA参数（默认39）
        
        Returns
        -------
        pd.Series
            McClellan摆动指标值
        """
        net = advances - declines
        ema_s = net.ewm(span=ema_short, adjust=False).mean()
        ema_l = net.ewm(span=ema_long, adjust=False).mean()
        return ema_s - ema_l

    def mcclellan_sum_index(self, advances: pd.Series, declines: pd.Series,
                             ema_short: int = 19, ema_long: int = 39) -> pd.Series:
        """McClellan合指标(SI): 对摆动指标再取EMA19
        
        合指标 > 0 表示中期上升动能，< 0 表示中期下降动能。
        """
        osc = self.mcclellan_oscillator(advances, declines, ema_short, ema_long)
        return osc.ewm(span=ema_short, adjust=False).mean()

    def signals(self, oscillator_values: pd.Series) -> dict:
        """基于McClellan摆动指标的买卖信号
        
        Returns
        -------
        dict
            regime: overbought(超买)/oversold(超卖)/neutral(中性)
            has_bullish_divergence: 底背离
            has_bearish_divergence: 顶背离
            zero_line_cross: 0轴穿越(up/down/none)
            extreme_score: 极端程度(0-100)
        """
        latest = oscillator_values.iloc[-1] if len(oscillator_values) > 0 else 0

        # 超买/超卖判断
        if latest > 100:
            regime = "overbought"
        elif latest < -100:
            regime = "oversold"
        else:
            regime = "neutral"

        # 0轴穿越
        prev = oscillator_values.iloc[-2] if len(oscillator_values) > 1 else 0
        zero_cross = "up" if prev <= 0 < latest else ("down" if prev >= 0 > latest else "none")

        # 极端程度
        extreme_score = min(100, max(0, abs(latest) / 2))

        # 背离检测（简化）
        bullish_div = False
        bearish_div = False
        if len(oscillator_values) > 60:
            recent_min = oscillator_values.iloc[-20:].min()
            older_min = oscillator_values.iloc[-60:-20].min()
            if older_min < -100 and recent_min > older_min * 0.7:
                bullish_div = True
            recent_max = oscillator_values.iloc[-20:].max()
            older_max = oscillator_values.iloc[-60:-20].max()
            if older_max > 100 and recent_max < older_max * 0.7:
                bearish_div = True

        return {
            "regime": regime,
            "value": latest,
            "has_bullish_divergence": bullish_div,
            "has_bearish_divergence": bearish_div,
            "zero_line_cross": zero_cross,
            "extreme_score": extreme_score,
        }

    def oscillator_histogram(self, oscillator: pd.Series, sum_index: pd.Series) -> pd.Series:
        """McClellan摆动直方图: 摆动指标 - 合指标
        
        正值表示短期动能强于中期趋势（加速），
        负值表示短期动能弱于中期趋势（减速）。
        """
        return oscillator - sum_index


# ═══════════════════════════════════════════════════════
# 3. Percent Above MA (均线上方占比)
# ═══════════════════════════════════════════════════════

class PercentAboveMA:
    """股票在均线之上的比例 — 最直观的宽度指标
    
    全市场股票在重要均线(20/60/144/300)之上的占比，
    可衡量市场整体的中期趋势健康度。
    极端值（>90%或<10%）通常预示反转。
    """

    def __init__(self):
        pass

    def pct_above_ma(self, close_panel: pd.DataFrame, period: int) -> pd.Series:
        """计算每日在MA(period)之上的股票比例
        
        Parameters
        ----------
        close_panel : pd.DataFrame
            多股票收盘价矩阵，每列一只股票，index为日期
        period : int
            均线周期（如20/60/144/300）
        
        Returns
        -------
        pd.Series
            每日在MA之上的比例（0-1）
        """
        ma = close_panel.rolling(period, min_periods=period // 2).mean()
        above = close_panel > ma
        # P2-Q22-fix(L252): 原 above.mean(axis=1) 将停牌/缺失(NaN)股票计入
        # "不在均线上方"分母，占比有偏。改为仅以有效(收盘价与MA均非NaN)股票为分母。
        valid = close_panel.notna() & ma.notna()
        pct = (above & valid).sum(axis=1) / valid.sum(axis=1).clip(lower=1)
        return pct

    def pct_above_ma_multi(self, close_panel: pd.DataFrame,
                           periods: list = None) -> pd.DataFrame:
        """多周期同时计算，返回DataFrame
        
        Parameters
        ----------
        close_panel : pd.DataFrame
            多股票收盘价矩阵
        periods : list
            均线周期列表，默认[20, 60, 144, 300]
        
        Returns
        -------
        pd.DataFrame
            每列为不同周期的一above比例
        """
        if periods is None:
            periods = [20, 60, 144, 300]
        result = {}
        for p in periods:
            col_name = f"MA{p}"
            try:
                result[col_name] = self.pct_above_ma(close_panel, p)
            except Exception:
                result[col_name] = pd.Series(dtype=float)
        return pd.DataFrame(result)

    def by_industry(self, close_panel: pd.DataFrame, industry_map: dict,
                    period: int = 144) -> pd.DataFrame:
        """按行业维度计算均线上方比例
        
        Parameters
        ----------
        close_panel : pd.DataFrame
            多股票收盘价矩阵（列：股票代码）
        industry_map : dict
            {股票代码: 申万一级行业名}
        period : int
            均线周期（默认144）
        
        Returns
        -------
        pd.DataFrame
            每行为日期，每列为行业
        """
        ma = close_panel.rolling(period, min_periods=period // 2).mean()
        above = (close_panel > ma).astype(float)
        industries = pd.Series(industry_map)
        result = {}
        for ind in industries.unique():
            stocks = industries[industries == ind].index
            cols = [c for c in above.columns if c in stocks]
            if cols:
                result[ind] = above[cols].mean(axis=1)
        return pd.DataFrame(result)

    def by_market_cap(self, close_panel: pd.DataFrame, cap_bins: dict,
                      period: int = 144) -> pd.DataFrame:
        """按市值维度计算均线上方比例
        
        Parameters
        ----------
        close_panel : pd.DataFrame
            多股票收盘价矩阵
        cap_bins : dict
            {股票代码: 市值标签}，如大盘/中盘/小盘/微盘
        period : int
            均线周期（默认144）
        
        Returns
        -------
        pd.DataFrame
            每列为不同市值类别
        """
        ma = close_panel.rolling(period, min_periods=period // 2).mean()
        above = (close_panel > ma).astype(float)
        cat_map = pd.Series(cap_bins)
        result = {}
        for cat in cat_map.unique():
            stocks = cat_map[cat_map == cat].index
            cols = [c for c in above.columns if c in stocks]
            if cols:
                result[cat] = above[cols].mean(axis=1)
        return pd.DataFrame(result)

    def z_score(self, series: pd.Series, lookback: int = 252) -> pd.Series:
        """计算历史z-score标准化，判断当前处于极端值
        
        z > 2 或 z < -2 表示处于历史极端位置，通常预示均值回归。
        
        Parameters
        ----------
        series : pd.Series
            均线上方比例序列
        lookback : int
            回溯窗口（默认252个交易日）
        
        Returns
        -------
        pd.Series
            z-score序列
        """
        mean = series.rolling(lookback, min_periods=60).mean()
        std = series.rolling(lookback, min_periods=60).std()
        return (series - mean) / std.clip(lower=1e-12)

    def cap_weighted(self, close_panel: pd.DataFrame, cap_series: pd.Series,
                     period: int = 144) -> pd.Series:
        """市值加权版本：大盘股权重更高的均线上方比例
        
        Parameters
        ----------
        close_panel : pd.DataFrame
            多股票收盘价矩阵
        cap_series : pd.Series
            各股票的市值序列（单位：亿），index为股票代码
        period : int
            均线周期
        
        Returns
        -------
        pd.Series
            市值加权均线上方比例
        """
        ma = close_panel.rolling(period, min_periods=period // 2).mean()
        above = (close_panel > ma).astype(float)
        weights = cap_series / cap_series.sum()
        common_cols = [c for c in above.columns if c in weights.index]
        if not common_cols:
            return pd.Series(dtype=float)
        weighted = above[common_cols].mul(weights[common_cols], axis=1)
        return weighted.sum(axis=1)

    def summary(self, above_ma_df: pd.DataFrame) -> str:
        """中文总结当前多周期宽度状况"""
        if above_ma_df.empty:
            return "暂无均线宽度数据"
        latest = above_ma_df.iloc[-1]
        lines = ["【均线上方占比（多周期）】"]
        for col, val in latest.items():
            status = "超买区" if val > 0.8 else ("超卖区" if val < 0.2 else "正常区间")
            lines.append(f"  {col}: {val*100:.1f}% — {status}")
        return "\n".join(lines)

    def extreme_signal(self, values: pd.Series, lookback: int = 252) -> dict:
        """极端信号判断"""
        z = self.z_score(values, lookback)
        latest_z = z.iloc[-1] if len(z) > 0 else 0
        latest_val = values.iloc[-1] if len(values) > 0 else 0.5
        signals = []
        if latest_z > 2.0:
            signals.append("超买信号（z>2）")
        elif latest_z < -2.0:
            signals.append("超卖信号（z<-2）")
        if latest_val > 0.9:
            signals.append("极端高位（>90%）")
        elif latest_val < 0.1:
            signals.append("极端低位（<10%）")
        return {
            "latest_value": latest_val,
            "latest_zscore": round(latest_z, 2),
            "signals": signals,
            "severity": "high" if len(signals) >= 2 else ("medium" if len(signals) >= 1 else "none"),
        }


# ═══════════════════════════════════════════════════════
# 4. New High / New Low (创新高新低)
# ═══════════════════════════════════════════════════════

class NewHighNewLow:
    """创新高新低分析
    
    创新高(New High)和创新低(New Low)的数量对比反映市场趋势的广度。
    当NH数量创新高但价格未创新高时 → 顶背离风险
    """

    def __init__(self):
        pass

    def nh_nl_count(self, high_panel: pd.DataFrame, low_panel: pd.DataFrame,
                    lookback: int = 252) -> pd.DataFrame:
        """每日创新高（N日）和新低（N日）的数量
        
        Parameters
        ----------
        high_panel : pd.DataFrame
            多股票最高价矩阵
        low_panel : pd.DataFrame
            多股票最低价矩阵
        lookback : int
            回溯窗口（默认252日,即年度新高新低）
        
        Returns
        -------
        pd.DataFrame
            columns: nh_count(创新高数), nl_count(创新低数)
        """
        # P2-Q22-fix(M241): 原用 == 精确相等判断创新高/新低；前复权价格多为浮点长小数，
        # == 易漏判新高新低。改为先 round(2)（A股两位小数报价）再比较。
        high_max = high_panel.rolling(lookback, min_periods=lookback // 2).max()
        low_min = low_panel.rolling(lookback, min_periods=lookback // 2).min()
        nh = (high_panel.round(2) == high_max.round(2)).astype(int)
        nl = (low_panel.round(2) == low_min.round(2)).astype(int)
        result = pd.DataFrame({
            "nh_count": nh.sum(axis=1),
            "nl_count": nl.sum(axis=1),
        })
        return result

    def nh_nl_ratio(self, nh_count: pd.Series, nl_count: pd.Series) -> pd.Series:
        """NH-NL比: nh / max(nl, 1), 对数压缩"""
        ratio = nh_count / nl_count.clip(lower=1)
        return np.log(ratio.clip(lower=0.01, upper=100))

    def nh_nl_oscillator(self, nh_count: pd.Series, nl_count: pd.Series,
                         ema_period: int = 20) -> pd.Series:
        """NH-NL摆动: EMA(净差值)"""
        net = (nh_count - nl_count).astype(float)
        return net.ewm(span=ema_period, adjust=False).mean()

    def by_exchange(self, high_panel: pd.DataFrame, low_panel: pd.DataFrame,
                    exchange_map: dict, lookback: int = 252) -> dict:
        """分交易所维度
        
        Returns
        -------
        dict
            {exchange_name: pd.DataFrame with nh_count, nl_count}
        """
        result = {}
        for exchange, codes in pd.Series(exchange_map).groupby(exchange_map):
            cols = [c for c in high_panel.columns if c in codes.index]
            if not cols:
                continue
            result[exchange] = self.nh_nl_count(
                high_panel[cols], low_panel[cols], lookback)
        return result

    def by_market_cap(self, high_panel: pd.DataFrame, low_panel: pd.DataFrame,
                      cap_bins: dict, lookback: int = 252) -> dict:
        """分市值维度"""
        result = {}
        for cat in set(cap_bins.values()):
            codes = [k for k, v in cap_bins.items() if v == cat]
            cols = [c for c in high_panel.columns if c in codes]
            if not cols:
                continue
            result[cat] = self.nh_nl_count(high_panel[cols], low_panel[cols], lookback)
        return result

    def extreme_signal(self, nh_count: pd.Series, nl_count: pd.Series) -> dict:
        """极端信号判断"""
        total = nh_count + nl_count
        if total.sum() == 0:
            return {"signal": "none", "description": "无数据"}
        nh_ratio = nh_count.iloc[-1] / max(total.iloc[-1], 1)
        nl_ratio = nl_count.iloc[-1] / max(total.iloc[-1], 1)
        signals = []
        if nh_ratio > 0.8:
            signals.append("创新高主导")
        if nl_ratio > 0.8:
            signals.append("创新低主导")
        return {
            "latest_nh": int(nh_count.iloc[-1]) if len(nh_count) > 0 else 0,
            "latest_nl": int(nl_count.iloc[-1]) if len(nl_count) > 0 else 0,
            "nh_ratio": round(nh_ratio, 2),
            "nl_ratio": round(nl_ratio, 2),
            "signals": signals,
        }

    def summary(self, nh_count: pd.Series, nl_count: pd.Series) -> str:
        """中文总结"""
        signal = self.extreme_signal(nh_count, nl_count)
        lines = ["【创新高新低】"]
        lines.append(f"  创新高: {signal['latest_nh']}家")
        lines.append(f"  创新低: {signal['latest_nl']}家")
        if signal['signals']:
            for s in signal['signals']:
                lines.append(f"  ⚠ {s}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════
# 5. Breadth Thrust (宽度推力信号)
# ═══════════════════════════════════════════════════════

class BreadthThrust:
    """宽度推力/爆发信号
    
    宽度推力：短期内大量股票同时上涨，表明市场动能强劲。
    Martin Pring定义：10个交易日内90%以上股票涨幅超过2%。
    """

    def __init__(self):
        pass

    def thrust_signal(self, advances_panel: pd.DataFrame, total_issues: int,
                      lookback: int = 10, threshold: float = 0.9) -> pd.Series:
        """宽度推力信号（Martin Pring 定义）

        P2-Q22-fix(M242): 原实现为"上涨家数占比 N 日均值≥0.9"，与类文档
        Pring 定义（"10个交易日内90%以上股票涨幅超过2%"）不符。现按 Pring
        定义重写：对每个交易日，取过去 lookback 日窗口，计算窗口内累计涨幅
        >2% 的股票占比，占比≥threshold 记 1。窗口数据不足（任一股缺历史）
        时输出 NaN，避免在无数据处给出虚假信号。
        注：advances_panel 语义由"1涨0跌标记"改为"按日涨跌幅(%)矩阵"。

        Parameters
        ----------
        advances_panel : pd.DataFrame
            各股票按日涨跌幅(%)矩阵（列=股票，index=日期）
        total_issues : int
            总股票数
        lookback : int
            窗口天数（默认10日）
        threshold : float
            窗口内涨幅>2%股票占比阈值（默认0.9，90%）

        Returns
        -------
        pd.Series
            推力信号，1表示触发（窗口不足为NaN）
        """
        if advances_panel is None or advances_panel.empty:
            return pd.Series(dtype=float)
        # 每只股票在 lookback 窗口内的累计涨幅（复利）
        window_ret = (1 + advances_panel / 100.0).rolling(lookback).apply(
            lambda x: float(np.prod(x) - 1), raw=True)
        above = (window_ret > 0.02).sum(axis=1)
        ratio = above / max(total_issues, 1)
        # 仅当所有股票在窗口内都有数据时才输出信号（避免部分窗口占比有偏）
        valid = advances_panel.rolling(lookback).count().ge(lookback).all(axis=1)
        signal = pd.Series(np.nan, index=advances_panel.index, dtype=float)
        signal[valid] = (ratio[valid] >= threshold).astype(float)
        return signal

    def mcclellan_oversold_bounce(self, mcclellan_osc: pd.Series,
                                   lookback: int = 3) -> pd.Series:
        """McClellan超卖反弹: 低于-100后连续回升"""
        oversold = mcclellan_osc < -100
        signal = pd.Series(0, index=mcclellan_osc.index)
        for i in range(lookback, len(mcclellan_osc)):
            if oversold.iloc[i - lookback]:
                if all(mcclellan_osc.iloc[i - j] > mcclellan_osc.iloc[i - j - 1] for j in range(lookback)):
                    signal.iloc[i] = 1
        return signal

    def ad_line_breakout(self, ad_series: pd.Series, lookback: int = 20) -> pd.Series:
        """A/D线突破创N日新高"""
        ad_high = ad_series.rolling(lookback).max()
        return (ad_series >= ad_high).astype(int)

    def breadth_thrust_index(self, advances: pd.Series, declines: pd.Series,
                              volume: pd.Series, lookback: int = 3) -> pd.Series:
        """综合宽度推力指数: A/D向上 + 成交量放大"""
        net = (advances - declines) / (advances + declines).clip(lower=1)
        vol_ratio = volume / volume.rolling(20).mean().clip(lower=1e-12)
        thrust = net.rolling(lookback).mean() * vol_ratio.clip(upper=5)
        return thrust


# ═══════════════════════════════════════════════════════
# 6. Breadth Report (宽度报告)
# ═══════════════════════════════════════════════════════

class BreadthReport:
    """宽度分析报告——整合所有宽度指标输出中文分析"""

    def __init__(self, ad_line: Optional[AdvanceDeclineLine] = None,
                 mcclellan: Optional[McClellanOscillator] = None,
                 above_ma: Optional[PercentAboveMA] = None,
                 nh_nl: Optional[NewHighNewLow] = None,
                 thrust: Optional[BreadthThrust] = None):
        self.ad = ad_line or AdvanceDeclineLine()
        self.mcc = mcclellan or McClellanOscillator()
        self.above = above_ma or PercentAboveMA()
        self.nhnl = nh_nl or NewHighNewLow()
        self.thrust = thrust or BreadthThrust()

    def full_report(self, advances: pd.Series, declines: pd.Series,
                    price_index: pd.Series, above_ma_df: pd.DataFrame,
                    nh_df: pd.DataFrame) -> str:
        """完整宽度报告
        
        Parameters
        ----------
        advances : pd.Series
            每日上涨家数
        declines : pd.Series
            每日下跌家数
        price_index : pd.Series
            价格指数
        above_ma_df : pd.DataFrame
            多周期均线上方占比
        nh_df : pd.DataFrame
            创新高新低计数（含nh_count, nl_count）
        
        Returns
        -------
        str
            中文宽度报告
        """
        # 计算各项指标
        cum_ad = self.ad.cumulative_ad(advances, declines)
        div = self.ad.ad_divergence(price_index, cum_ad)
        cross = self.ad.ad_ma_cross(cum_ad)
        mcc_osc = self.mcc.mcclellan_oscillator(advances, declines)
        mcc_sig = self.mcc.signals(mcc_osc)
        ad_ratio = advances.iloc[-1] / max(advances.iloc[-1] + declines.iloc[-1], 1) if len(advances) > 0 else 0.5

        lines = [
            "=" * 55,
            "【市场宽度分析报告】",
            "=" * 55,
            "",
            "【A/D线】",
            f"  累积A/D方向: {'向上' if len(cum_ad) > 1 and cum_ad.iloc[-1] > cum_ad.iloc[-20] else '向下' if len(cum_ad) > 1 else '—'}",
            f"  A/D均线交叉: {cross.iloc[-1] if len(cross) > 0 else 0}",
            f"  A/D线趋势: A/D线20日变化={cum_ad.diff(20).iloc[-1]:.0f}" if len(cum_ad) > 20 else "",
        ]

        if div["has_bullish_divergence"]:
            lines.append(f"  ⚠ 底背离: 价格新低但A/D未创新低（看多信号）强度{div['divergence_strength']}")
        if div["has_bearish_divergence"]:
            lines.append(f"  ⚠ 顶背离: 价格新高但A/D未创新高（看空信号）强度{div['divergence_strength']}")

        lines.extend([
            "",
            "【McClellan摆动】",
            f"  摆动值: {mcc_osc.iloc[-1]:.1f}" if len(mcc_osc) > 0 else "  摆动值: —",
            f"  状态: {mcc_sig['regime']}",
            f"  0轴穿越: {mcc_sig['zero_line_cross']}",
            f"  极端度: {mcc_sig['extreme_score']}/100",
            "",
            "【均线上方占比】",
        ])
        if not above_ma_df.empty:
            latest = above_ma_df.iloc[-1]
            for col in above_ma_df.columns:
                val = latest.get(col, 0)
                status = "强势" if val > 0.7 else ("弱势" if val < 0.3 else "中性")
                lines.append(f"  {col}: {val*100:.1f}% ({status})")

        lines.extend([
            "",
            "【创新高新低】",
            f"  创新高: {int(nh_df['nh_count'].iloc[-1]) if 'nh_count' in nh_df and len(nh_df) > 0 else 0}家",
            f"  创新低: {int(nh_df['nl_count'].iloc[-1]) if 'nl_count' in nh_df and len(nh_df) > 0 else 0}家",
            "",
            "【宽度综合分析】",
        ])

        total_issues = (advances.iloc[-1] + declines.iloc[-1]) if len(advances) > 0 else 1
        if ad_ratio > 0.7:
            lines.append("  ✅ 市场宽度良好，上涨占比显著高于下跌")
        elif ad_ratio < 0.3:
            lines.append("  ⚠ 市场宽度不佳，下跌占比显著高于上涨")
        else:
            lines.append("  ➡ 市场宽度中性，涨跌分化不明显")

        lines.append("")
        lines.append("=" * 55)
        return "\n".join(lines)

    def daily_summary(self, advances: pd.Series, declines: pd.Series) -> str:
        """每日快速宽度摘要"""
        if len(advances) < 1:
            return "暂无数据"
        total = advances.iloc[-1] + declines.iloc[-1]
        ad_ratio = advances.iloc[-1] / max(total, 1)
        if ad_ratio > 0.7:
            level = "宽度强劲"
        elif ad_ratio < 0.3:
            level = "宽度低迷"
        else:
            level = "宽度中性"
        return f"【宽度摘要】上涨{int(advances.iloc[-1])}家/下跌{int(declines.iloc[-1])}家/总{int(total)}家 | 涨跌比{ad_ratio:.2f} | {level}"

    def mechanism_explain(self, advances: pd.Series, declines: pd.Series,
                          price_index: pd.Series) -> str:
        """宽度指标的机制解释"""
        cum_ad = self.ad.cumulative_ad(advances, declines)
        div = self.ad.ad_divergence(price_index, cum_ad)
        mcc_osc = self.mcc.mcclellan_oscillator(advances, declines)
        latest_mcc = mcc_osc.iloc[-1] if len(mcc_osc) > 0 else 0

        lines = [
            "宽度指标机制解读：",
        ]

        # A/D线机制
        if div["has_bullish_divergence"]:
            lines.append("- A/D线底背离: 价格创新低但上涨家数累积线并未同步新低，说明下跌由少数权重股拉低，多数股票已有企稳迹象，这是经典的看多信号。")
        elif div["has_bearish_divergence"]:
            lines.append("- A/D线顶背离: 价格创新高但上涨家数累积线并未同步新高，说明上涨由少数权重股拉动，多数股票已开始走弱，这是经典的风险信号。")

        # McClellan机制
        if latest_mcc > 100:
            lines.append("- McClellan摆动超买: 短期上涨家数远高于长期均值，市场短期动能过强，可能面临短期回调，但强势市场可维持超买状态。")
        elif latest_mcc < -100:
            lines.append("- McClellan摆动超卖: 短期下跌家数远高于长期均值，市场短期恐慌过度，通常预示技术性反弹机会。")

        # 解读
        lines.append("")
        lines.append("宽度指标的核心逻辑: 宽度是市场内部结构的\"温度计\"。当宽度先于指数见顶/见底时，往往预示趋势可能发生转变。如果宽度持续向好但指数不涨，是\"蓄力\"还是\"诱多\"需要结合成交量判断。")

        return "\n".join(lines)


__all__ = [
    "AdvanceDeclineLine", "McClellanOscillator",
    "PercentAboveMA", "NewHighNewLow",
    "BreadthThrust", "BreadthReport",
]
