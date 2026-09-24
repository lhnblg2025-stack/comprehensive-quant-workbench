"""
market_microstructure.py — 市场微观结构分析
V4.1 feature

提供：订单簿分析、买卖价差、交易信号质量、市场深度评估
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

logger = __import__('logging').getLogger(__name__)


class BidAskAnalyzer:
    """买卖价差分析"""

    @staticmethod
    def quoted_spread(ask_price: pd.Series, bid_price: pd.Series) -> pd.Series:
        """报价价差 = Ask - Bid（绝对值）"""
        return ask_price - bid_price

    @staticmethod
    def relative_spread(ask_price: pd.Series, bid_price: pd.Series,
                        mid_price: Optional[pd.Series] = None) -> pd.Series:
        """相对价差 = (Ask - Bid) / Mid"""
        if mid_price is None:
            mid_price = (ask_price + bid_price) / 2
        return (ask_price - bid_price) / mid_price.clip(lower=1e-12)

    @staticmethod
    def effective_spread(trade_price: pd.Series, mid_price: pd.Series,
                          side: str = "buy") -> pd.Series:
        """有效价差 = 2 * |Trade - Mid| * sign"""
        diff = trade_price - mid_price
        if side == "buy":
            return 2 * diff
        return -2 * diff

    @staticmethod
    def realized_spread(trade_price: pd.Series, mid_price: pd.Series,
                         mid_after_n_minutes: pd.Series, side: str = "buy") -> pd.Series:
        """实现价差 = 2 * (Trade - Mid_after) * sign"""
        diff = trade_price - mid_after_n_minutes
        if side == "buy":
            return 2 * (trade_price - mid_after_n_minutes)
        return -2 * (trade_price - mid_after_n_minutes)

    @staticmethod
    def adverse_selection(realized_spread: pd.Series,
                           effective_spread: pd.Series) -> pd.Series:
        """逆向选择成本 = Effective - Realized"""
        return effective_spread - realized_spread

    @staticmethod
    def summary(close: pd.Series, high: pd.Series, low: pd.Series) -> dict:
        """价差估算（用高低价近似）"""
        # P1-Q28-fix: Roll (1984) 模型 S = 2*sqrt(-Cov(ΔP_t, ΔP_{t-1}))，
        # 必须用收盘价一阶差分的自协方差；原实现用 Δclose 与当日振幅(high-low)
        # 的协方差，且协方差为正时静默返回 0。现协方差>0（模型不适用）返回 NaN。
        dp = close.diff()
        cov_ = dp.cov(dp.shift(1))
        roll_spread = 2.0 * np.sqrt(-cov_) if (np.isfinite(cov_) and cov_ < 0) else np.nan

        # P1-Q28-fix: Corwin-Schultz (2012) α = E[ln(H/L)]，S = 2(e^α−1)/(1+e^α)；
        # 原实现用 np.exp(原始价差) 量纲错误（偏差约100倍）。
        alpha = np.log(high / low.clip(lower=1e-12)).mean()
        corwin_schultz = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))

        return {
            "roll_estimated_spread": round(roll_spread, 6) if not np.isnan(roll_spread) else float("nan"),
            "corwin_schultz_spread": round(corwin_schultz, 6),
        }


class OrderFlowAnalyzer:
    """订单流分析"""

    @staticmethod
    def order_imbalance(buy_volume: pd.Series, sell_volume: pd.Series) -> pd.Series:
        """订单失衡 = (Buy - Sell) / (Buy + Sell)"""
        total = buy_volume + sell_volume
        return (buy_volume - sell_volume) / total.clip(lower=1e-12)

    @staticmethod
    def vpin(volume: pd.Series, price_change: pd.Series,
             window: int = 20, n_buckets: int = 10) -> pd.Series:
        """VPIN (Volume-synchronized Probability of Informed Trading)

        参考: Easley, Lopez de Prado & O'Hara (2012), AFML Ch.18
        P1-Q28-fix: 修复输出与输入错位 + 买卖失衡口径：
          1) 用 bar 方向代理估算买卖量：涨→买、跌→卖、平→各半（tick 规则近似）；
          2) 按成交量桶计算 OI = |V_buy - V_sell| / (V_buy + V_sell)，
             替代原 |Σ收益|/桶成交量的非标准近似（AFML 的 E[|2v_B−1|]）；
          3) 桶级 OI 经滚动均值后回填到每个桶覆盖的原始 bar，输出与输入
             同长同索引，替代原先仅返回 n_buckets 个桶值的 RangeIndex 序列。
        """
        volume = volume.astype(float)
        price_change = price_change.fillna(0.0)
        direction = np.sign(price_change.to_numpy())
        buy_frac = (direction + 1.0) / 2.0      # 1 / 0 / 0.5
        buy_vol = volume.to_numpy() * buy_frac
        sell_vol = volume.to_numpy() - buy_vol

        cum_vol = volume.cumsum().to_numpy()
        total_vol = cum_vol[-1] if cum_vol.size else 0.0

        if n_buckets <= 0 or total_vol <= 0:
            return pd.Series(np.nan, index=volume.index, dtype=float)

        bucket_size = total_vol / n_buckets
        oi_buckets = np.full(n_buckets, np.nan)
        bucket_members: list[np.ndarray] = []
        for i in range(n_buckets):
            if i == n_buckets - 1:
                # 末桶包含尾部 bar（cum_vol == total_vol 的行）
                mask = (cum_vol >= i * bucket_size) & (cum_vol <= total_vol + 1e-12)
            else:
                mask = (cum_vol >= i * bucket_size) & (cum_vol < (i + 1) * bucket_size)
            idx = np.nonzero(mask)[0]
            bucket_members.append(idx)
            if idx.size == 0:
                continue
            v_buy = buy_vol[idx].sum()
            v_sell = sell_vol[idx].sum()
            oi_buckets[i] = abs(v_buy - v_sell) / max(v_buy + v_sell, 1e-8)

        # 桶级滚动均值（window 为桶窗口，与原实现 min(window, n_buckets) 语义一致；
        # min_periods=1 使 window>=n_buckets 时不再全 NaN，退化为全样本均值预热）
        vpin_bucket = pd.Series(oi_buckets).rolling(
            max(1, min(window, n_buckets)), min_periods=1).mean()

        # 回填到原始 bar 索引，保证与输入对齐
        out = pd.Series(np.nan, index=volume.index, dtype=float)
        for i, idx in enumerate(bucket_members):
            if idx.size:
                out.iloc[idx] = vpin_bucket.iloc[i]
        return out

    @staticmethod
    def trade_sign_ratio(buy_trades: int, sell_trades: int) -> float:
        """交易方向比"""
        total = buy_trades + sell_trades
        return (buy_trades - sell_trades) / max(total, 1)

    @staticmethod
    def amihud_illiquidity(returns: pd.Series, volume: pd.Series,
                            window: int = 20) -> pd.Series:
        """Amihud 非流动性指标"""
        illiq = returns.abs() / volume.clip(lower=1e4)
        return illiq.rolling(window).mean()


class MarketDepthAnalyzer:
    """市场深度分析"""

    @staticmethod
    def depth_at_price(ask_prices: list, ask_volumes: list,
                        bid_prices: list, bid_volumes: list,
                        levels: int = 5) -> dict:
        """指定档位深度"""
        return {
            "best_ask": ask_prices[0] if ask_prices else None,
            "best_bid": bid_prices[0] if bid_prices else None,
            "ask_volume_5": sum(ask_volumes[:levels]) if ask_volumes else 0,
            "bid_volume_5": sum(bid_volumes[:levels]) if bid_volumes else 0,
            "spread": ask_prices[0] - bid_prices[0] if ask_prices and bid_prices else None,
        }

    @staticmethod
    def market_impact_cost(order_value: float, total_volume: float,
                            avg_daily_volume: float, volatility: float,
                            participation: float = 0.1) -> float:
        """估算市场冲击成本 (Almgren-Chriss 简化版)

        P2-Q28-fix(L373): 原 adv_ratio = order_value(元)/avg_daily_volume(股)，
        量纲为"元/股"，是无量纲比值的错误用法。现改用成交量比
        total_volume(股)/avg_daily_volume(股)，与 Almgren-Chriss 参与率口径一致。
        """
        adv_ratio = total_volume / max(avg_daily_volume, 1e-8)
        perm = 0.1 * volatility * adv_ratio ** 0.3
        temp = volatility * participation ** 0.6
        return perm + temp

    @staticmethod
    def fill_probability(limit_price: float, current_price: float,
                          side: str = "buy", volatility: float = 0.02) -> float:
        """限价单成交概率估算"""
        from scipy.stats import norm
        if side == "buy":
            z = (limit_price - current_price) / (current_price * volatility)
        else:
            z = (current_price - limit_price) / (current_price * volatility)
        return norm.cdf(z)

    @staticmethod
    def optimal_limit_price(current_price: float, volatility: float,
                             risk_aversion: float = 1.0, side: str = "buy") -> float:
        """最优限价定价"""
        from scipy.stats import norm
        spread = current_price * volatility
        if side == "buy":
            discount = spread * norm.ppf(risk_aversion / (risk_aversion + 1))
            return current_price - discount
        else:
            premium = spread * norm.ppf(risk_aversion / (risk_aversion + 1))
            return current_price + premium


class TradeQualityAnalyzer:
    """交易质量分析"""

    @staticmethod
    def implementation_shortfall(arrival_price: float, execution_price: float,
                                  side: str = "buy", volume: int = 1) -> float:
        """实现缺口 = (Exec - Arrival) * sign * volume"""
        if side == "buy":
            return (execution_price - arrival_price) * volume
        return (arrival_price - execution_price) * volume

    @staticmethod
    def vwap_shortfall(trade_prices: list, trade_volumes: list,
                        benchmark_price: float, side: str = "buy") -> float:
        """VWAP 缺口"""
        total_val = sum(p * v for p, v in zip(trade_prices, trade_volumes))
        total_vol = sum(trade_volumes)
        vwap = total_val / max(total_vol, 1)
        if side == "buy":
            return vwap - benchmark_price
        return benchmark_price - vwap

    @staticmethod
    def participation_rate(trade_volume: float, market_volume: float) -> float:
        """参与率"""
        return trade_volume / max(market_volume, 1)

    @staticmethod
    def price_reversion_after_trade(price_before: float, price_after: float,
                                     side: str = "buy") -> float:
        """交易后价格回复度"""
        if side == "buy":
            return max(0, price_before - price_after) / max(price_before, 1e-12)
        return max(0, price_after - price_before) / max(price_before, 1e-12)

    @staticmethod
    def trade_timing_quality(trade_time: datetime, optimal_time: datetime,
                              volatility_profile: pd.Series | None = None) -> float:
        """交易时点质量评分

        P2-Q28-fix(L373): volatility_profile 原为死参数（从未使用）。
        现纳入评分：按序列平均日内波动率相对 2% 参考水平的比例调整系数
        （波动越低越有利于成交，加分；波动越高打折），结果仍限制在 [0,1]。
        """
        try:
            minute = trade_time.hour * 60 + trade_time.minute
            opt_minute = optimal_time.hour * 60 + optimal_time.minute
            diff = abs(minute - opt_minute)
            score = max(0, 1 - diff / 240)  # 240分钟=全日
            if volatility_profile is not None and len(volatility_profile) > 0:
                vols = np.asarray(volatility_profile, dtype=float)
                vols = vols[np.isfinite(vols)]
                if len(vols) > 0 and vols.mean() > 0:
                    vol_factor = float(np.clip(0.02 / vols.mean(), 0.5, 1.5))
                    score = min(1.0, score * vol_factor)
            return float(score)
        except Exception:
            return 0.5


class MicrostructureSummary:
    """市场微观结构汇总报告"""

    def __init__(self):
        self.bid_ask = BidAskAnalyzer()
        self.flow = OrderFlowAnalyzer()
        self.depth = MarketDepthAnalyzer()
        self.quality = TradeQualityAnalyzer()

    def generate_report(self, close: pd.Series, high: pd.Series,
                         low: pd.Series, volume: pd.Series) -> str:
        """生成微观结构分析报告"""
        # P2-Q28-fix(L372): 空 close 序列保护（原 close.index[0] 会 IndexError）
        if close is None or len(close) == 0:
            return "\n".join([
                "=" * 55,
                "市场微观结构报告 (V4.1 feature)",
                f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "=" * 55,
                "数据范围: 空序列，无可分析数据",
                "=" * 55,
            ])
        spread = self.bid_ask.summary(close, high, low)
        illiq = self.flow.amihud_illiquidity(close.pct_change(), volume)
        avg_illiq = illiq.mean() if not illiq.empty else 0

        lines = [
            "=" * 55,
            "市场微观结构报告 (V4.1 feature)",
            f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"数据范围: {str(close.index[0])[:10]} ~ {str(close.index[-1])[:10]}",
            "=" * 55,
            "",
            "【价差分析】",
            f"  Roll 估计价差: {spread['roll_estimated_spread']:.6f}",
            f"  Corwin-Schultz价差: {spread['corwin_schultz_spread']:.6f}",
            "",
            "【流动性】",
            f"  Amihud非流动性(均值): {avg_illiq:.8f}",
            f"  Amihud非流动性(中位数): {illiq.median():.8f}" if not illiq.empty else "",
            f"  日均成交量: {volume.mean():.0f}",
            "",
            "【波动性】",
            f"  收益率标准差: {close.pct_change().std() * 100:.2f}%",
            f"  日内振幅均值: {(high - low).mean() / close.mean() * 100:.2f}%",
        ]
        return "\n".join(lines)


__all__ = [
    "BidAskAnalyzer", "OrderFlowAnalyzer",
    "MarketDepthAnalyzer", "TradeQualityAnalyzer",
    "MicrostructureSummary",
]
