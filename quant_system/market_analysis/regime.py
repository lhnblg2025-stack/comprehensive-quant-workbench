"""
market_analysis/regime.py — 市场状态检测
V4.1 feature

市场状态检测是量化策略的核心。在不同状态下使用不同的策略参数。
包含HMM、趋势状态、波动率状态、流动性状态和综合市场状态。
"""

import numpy as np
import pandas as pd

try:
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


class HMMRegime:
    """市场状态检测（滚动均值/波动率阈值近似）

    P2-Q22-fix(L248): 类名 HMMRegime 具误导性——本实现并非真正的隐马尔可夫
    模型（无隐状态转移矩阵估计），而是滚动均值/波动率阈值近似。为不破坏
    既有调用方 API，保留类名，但：
      - 明确标注 is_true_hmm=False，调用方可据此判定；
      - 真 HMM 需引入 hmmlearn 依赖（当前不满足，未引入）。
    真 HMM 将市场收益序列分解为多个隐状态（通常2-4个），每个状态有不同
    均值和波动率，常用于识别牛/熊/震荡市。
    """

    def __init__(self, n_components: int = 3, n_iter: int = 100):
        self.n_components = n_components
        self.n_iter = n_iter
        self._means = None
        self._covars = None
        self._transmat = None
        self._probs = None
        # P2-Q22-fix(L248): 显式标记非真 HMM（滚动阈值近似），供调用方识别
        self.is_true_hmm = False

    def fit(self, returns: pd.Series) -> "HMMRegime":
        """HMM拟合——简化版（使用高斯混合近似HMM状态）
        
        注：完整HMM需要hmmlearn库。这里使用滚动窗口阈值法近似。
        
        Parameters
        ----------
        returns : pd.Series
            收益率序列
        
        Returns
        -------
        self
        """
        if len(returns) < 100:
            return self
        # 使用滚动均值和波动率分割状态
        roll_mean = returns.rolling(60).mean()
        roll_std = returns.rolling(60).std()
        z = (returns - roll_mean) / roll_std.clip(lower=1e-8)
        
        if self.n_components >= 3:
            # 3状态: 牛(高均值低波动), 熊(低均值高波动), 震荡(均值约0中波动)
            self._means = np.array([roll_mean.iloc[-1] * 2, roll_mean.iloc[-1] * -1, roll_mean.iloc[-1] * 0.5])
            self._covars = np.array([roll_std.iloc[-1], roll_std.iloc[-1] * 2, roll_std.iloc[-1]])
        else:
            self._means = np.array([roll_mean.iloc[-1], -roll_mean.iloc[-1]])
            self._covars = np.array([roll_std.iloc[-1], roll_std.iloc[-1]])
        
        self._transmat = np.ones((self.n_components, self.n_components)) / self.n_components
        return self

    def regime_probabilities(self, returns: pd.Series) -> pd.DataFrame:
        """各状态概率"""
        if self._means is None:
            self.fit(returns)
        n = len(returns)
        probs = np.zeros((n, self.n_components))
        for i in range(n):
            for j in range(self.n_components):
                probs[i, j] = np.exp(-0.5 * ((returns.iloc[i] - self._means[j]) / 
                                               max(self._covars[j], 1e-8)) ** 2)
        probs = probs / probs.sum(axis=1, keepdims=True).clip(min=1e-8)
        return pd.DataFrame(probs, index=returns.index, 
                          columns=[f"State_{i}" for i in range(self.n_components)])

    def current_regime(self, returns: pd.Series) -> int:
        """当前主导状态"""
        probs = self.regime_probabilities(returns)
        if probs.empty:
            return 0
        return int(probs.iloc[-1].idxmax().split("_")[1])

    def regime_characteristics(self) -> dict:
        """各状态特征"""
        if self._means is None:
            return {}
        chars = {}
        for i in range(self.n_components):
            chars[f"State_{i}"] = {
                "mean_return": round(self._means[i], 4),
                "volatility": round(self._covars[i], 4),
            }
        return chars


class TrendRegime:
    """趋势状态检测（基于用户硬规则：300MA/144MA为主，MA20/MA60辅助）"""

    def ma_slope(self, close: pd.Series, period: int = 144) -> float:
        """均线斜率（角度制）
        
        Parameters
        ----------
        close : pd.Series
            收盘价序列
        period : int
            均线周期
        
        Returns
        -------
        float
            均线角度（度）
        """
        if len(close) < period + 5:
            return 0.0
        ma = close.rolling(period).mean().dropna()
        if len(ma) < 20:
            return 0.0
        # 最近20日均线做线性回归求斜率
        x = np.arange(20)
        y = ma.iloc[-20:].values
        slope = np.polyfit(x, y, 1)[0]
        # 转换为角度（度）
        angle = np.degrees(np.arctan(slope / ma.iloc[-1]))
        return round(angle, 1)

    def price_vs_ma(self, close: pd.Series, ma_period: int = 144) -> float:
        """价格相对均线的偏离程度（%）"""
        if len(close) < ma_period:
            return 0.0
        ma = close.rolling(ma_period).mean().iloc[-1]
        return round((close.iloc[-1] / ma - 1) * 100, 2)

    def trend_strength(self, close: pd.Series, period: int = 60) -> float:
        """趋势强度评分(0-100)
        
        基于ADX方法的简化：趋势方向的一致性
        """
        if len(close) < period:
            return 0.0
        ret = close.pct_change()
        pos_ratio = (ret.iloc[-period:] > 0).sum() / period
        neg_ratio = (ret.iloc[-period:] < 0).sum() / period
        dominance = abs(pos_ratio - neg_ratio)
        return round(dominance * 100, 1)

    def trend_regime(self, close: pd.Series) -> str:
        """综合趋势状态"""
        slope_300 = self.ma_slope(close, 300)
        slope_144 = self.ma_slope(close, 144)
        price_vs_144 = self.price_vs_ma(close, 144)
        strength = self.trend_strength(close)

        if slope_300 > 5 and slope_144 > 3 and price_vs_144 > 0:
            return "上升趋势（强势）"
        elif slope_300 > 2 and slope_144 > 0:
            return "上升趋势（温和）"
        elif slope_300 < -5 and slope_144 < -3:
            return "下降趋势（弱势）"
        elif slope_300 < -2 and slope_144 < 0:
            return "下降趋势（温和）"
        elif abs(slope_144) < 3:
            return "震荡盘整"
        else:
            return "方向不明"


class VolatilityRegime:
    """波动率状态检测"""

    def vol_regime(self, returns: pd.Series, lookback: int = 252) -> str:
        """波动率状态"""
        if len(returns) < lookback:
            return "正常"
        current_vol = returns.iloc[-20:].std() * np.sqrt(252)  # 最近20日年化
        # P2-Q22-fix(L249): 原 returns.iloc[:lookback] 含最近20日，与 current_vol
        # 窗口重叠，历史基准被当前波动污染。现剔除最近20日再作为基准。
        hist_vol = returns.iloc[:lookback - 20].std() * np.sqrt(252)
        ratio = current_vol / max(hist_vol, 1e-8)
        if ratio > 1.5:
            return "高波动"
        elif ratio < 0.5:
            return "低波动"
        else:
            return "正常波动"

    def vol_percentile(self, returns: pd.Series, window: int = 20,
                        lookback: int = 252) -> float:
        """波动率历史分位数"""
        if len(returns) < lookback:
            return 0.5
        current = returns.iloc[-window:].std()
        hist = pd.Series([returns.iloc[i - window:i].std() 
                         for i in range(window, lookback, window)])
        return (hist <= current).mean()

    def term_structure(self, returns: pd.Series,
                        horizons: list = None) -> dict:
        """波动率期限结构"""
        if horizons is None:
            horizons = [5, 10, 20, 60, 120]
        vols = {}
        for h in horizons:
            if len(returns) >= h:
                vols[f"{h}d"] = round(returns.iloc[-h:].std() * np.sqrt(252) * 100, 2)
        return vols


class LiquidityRegime:
    """流动性状态检测"""

    def amihud_illiquidity(self, returns: pd.Series, volume: pd.Series,
                            window: int = 20) -> float:
        """Amihud非流动性指标：|收益| / 成交额（日均）
        
        数值越高表示越不流动性。
        """
        if len(returns) < window or len(volume) < window:
            return 0.0
        illiq = (returns.abs() / volume.clip(lower=1e-8)).rolling(window).mean()
        return float(illiq.iloc[-1]) if len(illiq) > 0 else 0.0

    def liquidity_regime(self, spread: float = None,
                          depth: float = None) -> str:
        """流动性状态

        P2-Q22-fix(L250): 原实现忽略 spread/depth 参数硬编码返回"正常"。
        现根据传入的相对价差 spread（比例，如0.001=10bp）与市场深度 depth
        （无量纲）估算流动性；未提供任何指标时显式返回"未评估"，
        不再以"正常"冒充结论。
        """
        if spread is None and depth is None:
            return "未评估(需提供spread/depth数据)"
        if spread is not None:
            s = abs(float(spread))
            if s > 0.005:      # >50bp 相对价差
                return "流动性差"
            if s < 0.001:      # <10bp 相对价差
                return "流动性好"
        if depth is not None and float(depth) <= 0:
            return "流动性差"
        return "流动性正常"


class CompositeRegime:
    """综合市场状态——整合趋势/波动率/流动性/情绪"""

    def __init__(self):
        self.trend = TrendRegime()
        self.vol = VolatilityRegime()
        self.liquidity = LiquidityRegime()

    def composite_state(self, close: pd.Series, returns: pd.Series) -> dict:
        """综合判断当前市场状态
        
        Returns
        -------
        dict
            state: 综合状态（牛市/熊市/平衡市/结构市）
            confidence: 置信度(0-1)
            details: 各维度细分
        """
        trend_state = self.trend.trend_regime(close)
        vol_state = self.vol.vol_regime(returns)
        
        if "上升" in trend_state and "正常" in vol_state:
            state = "牛市"
            confidence = 0.8
        elif "上升" in trend_state and "高波动" in vol_state:
            state = "结构市（上行震荡）"
            confidence = 0.6
        elif "下降" in trend_state:
            state = "熊市"
            confidence = 0.7
        else:
            state = "平衡市"
            confidence = 0.5

        return {
            "state": state,
            "confidence": confidence,
            "details": {
                "trend": trend_state,
                "volatility": vol_state,
            }
        }


class RegimeReport:
    """市场状态报告"""

    def __init__(self):
        self.hmm = HMMRegime()
        self.trend = TrendRegime()
        self.vol = VolatilityRegime()
        self.composite = CompositeRegime()

    def full_report(self, close: pd.Series, returns: pd.Series) -> str:
        """完整市场状态报告"""
        trend_state = self.trend.trend_regime(close)
        vol_state = self.vol.vol_regime(returns)
        composite = self.composite.composite_state(close, returns)

        lines = [
            "=" * 55,
            "【市场状态报告】",
            "=" * 55,
            "",
            "【综合状态】",
            f"  {composite['state']} (置信度: {composite['confidence']:.0%})",
            "",
            "【趋势状态】",
            f"  {trend_state}",
            f"  300MA斜率: {self.trend.ma_slope(close, 300):+.1f}°" if len(close) > 300 else "",
            f"  144MA斜率: {self.trend.ma_slope(close, 144):+.1f}°" if len(close) > 144 else "",
            f"  价格距144MA: {self.trend.price_vs_ma(close, 144):+.2f}%" if len(close) > 144 else "",
            f"  趋势强度: {self.trend.trend_strength(close)}/100",
            "",
            "【波动率状态】",
            f"  {vol_state}",
            "=" * 55,
        ]
        return "\n".join(lines)

    def mechanism_explain(self, close: pd.Series) -> str:
        """市场状态机制解释"""
        composite = self.composite.composite_state(close, close.pct_change().dropna())
        lines = [
            "市场状态机制解读：",
            f"  当前市场处于【{composite['state']}】状态。",
        ]
        if "牛市" in composite["state"]:
            lines.append("- 趋势向上，波动率正常。建议保持趋势跟踪策略，回调时加仓。")
            lines.append("- 关注趋势是否加速或减速。加速可能见顶，减速可能进入震荡。")
        elif "熊市" in composite["state"]:
            lines.append("- 趋势向下，建议降低仓位，以防御性策略为主。")
            lines.append("- 关注超卖信号和底部形态，等待趋势反转信号。")
        elif "震荡" in composite["state"]:
            lines.append("- 趋势不明朗，适合均值回归策略和波段操作。")
            lines.append("- 关注突破方向。向上突破确认后转为趋势策略，向下反之。")
        else:
            lines.append("- 市场无明确方向，建议均衡配置，等待方向选择。")

        return "\n".join(lines)


__all__ = [
    "HMMRegime", "TrendRegime", "VolatilityRegime",
    "LiquidityRegime", "CompositeRegime", "RegimeReport",
]
