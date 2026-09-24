"""
market_analysis/volume.py — 市场量能分析
V4.1 feature

成交量分析是判断市场动能的关键。量价配合是技术分析的基础。
核心维度：全市场成交额、量价关系、行业成交额分布、VWAP、OBV等。
"""

import pandas as pd


class VolumeAnalysis:
    """量能分析——判断市场成交量的状态和趋势
    
    量能是市场的燃料。没有成交量的上涨是虚涨，高成交量的下跌是恐慌。
    """

    def __init__(self):
        pass

    def total_market_volume(self, volume_series: pd.Series = None) -> float:
        """全市场成交额"""
        if volume_series is not None and len(volume_series) > 0:
            return float(volume_series.iloc[-1])
        return 0.0

    def sector_volume_top(self, sector_volumes: dict, n: int = 5) -> pd.DataFrame:
        """成交额前N行业
        
        Parameters
        ----------
        sector_volumes : dict
            {行业名: 成交额}
        n : int
            返回前N个
        
        Returns
        -------
        pd.DataFrame
        """
        series = pd.Series(sector_volumes).sort_values(ascending=False)
        return pd.DataFrame({
            "行业": series.index[:n],
            "成交额(亿)": series.values[:n],
        })

    def volume_pct_change(self, current: float, prev: float) -> float:
        """量能变化率(%)"""
        return (current / max(prev, 1e-8) - 1) * 100

    def volume_ma_ratios(self, volume_series: pd.Series,
                         periods: list = None) -> dict:
        """量能相对均线倍数
        
        Parameters
        ----------
        volume_series : pd.Series
            日度成交量/成交额序列
        periods : list
            均线周期列表，默认[5, 20, 60]
        
        Returns
        -------
        dict
            {f"MA{p}": 倍数}
        """
        if periods is None:
            periods = [5, 20, 60]
        latest = volume_series.iloc[-1] if len(volume_series) > 0 else 0
        ratios = {}
        for p in periods:
            # P2-Q22-fix(L251): 数据不足时返回 NaN，不再返回 latest/max(0,1)=latest
            # 原始值冒充"倍数"。latest 或 ma 非正/为 NaN 时同样返回 NaN。
            if len(volume_series) <= p or not pd.notna(latest):
                ratios[f"MA{p}"] = float("nan")
                continue
            ma = volume_series.rolling(p).mean().iloc[-1]
            if pd.notna(ma) and ma > 0:
                ratios[f"MA{p}"] = round(latest / ma, 2)
            else:
                ratios[f"MA{p}"] = float("nan")
        return ratios

    def volume_surge_detection(self, volume: pd.Series, ma_period: int = 20,
                                threshold: float = 2.0) -> pd.Series:
        """放量检测：成交量超过均量threshold倍
        
        Returns
        -------
        pd.Series
            布尔值，True表示放量
        """
        ma = volume.rolling(ma_period).mean()
        return (volume / ma.clip(lower=1e-8)) > threshold

    def volume_shrink_detection(self, volume: pd.Series, ma_period: int = 20,
                                 threshold: float = 0.5) -> pd.Series:
        """缩量检测：成交量低于均量threshold倍"""
        ma = volume.rolling(ma_period).mean()
        return (volume / ma.clip(lower=1e-8)) < threshold

    def volume_price_divergence(self, price_change: pd.Series,
                                 volume_change: pd.Series) -> dict:
        """量价关系判断
        
        四种状态：
        - 放量上涨: 正常走势，上涨有量能支持
        - 缩量上涨: 上攻乏力，可能需要回调
        - 放量下跌: 恐慌下跌，可能继续下跌
        - 缩量下跌: 惜售，可能接近底部
        
        Parameters
        ----------
        price_change : pd.Series
            涨跌幅序列（%）
        volume_change : pd.Series
            成交量变化率（相对于前一日，%）
        
        Returns
        -------
        dict
            regime, description
        """
        if len(price_change) < 1 or len(volume_change) < 1:
            return {"regime": "unknown", "description": "数据不足"}

        pc = price_change.iloc[-1]
        vc = volume_change.iloc[-1]

        is_up = pc > 0
        is_down = pc < 0
        vol_surge = vc > 20  # 成交量放大>20%
        vol_shrink = vc < -20  # 成交量缩小>20%

        if is_up and vol_surge:
            regime = "bullish_volume"
            desc = "放量上涨: 上涨有量能支持，趋势健康"
        elif is_up and vol_shrink:
            regime = "bearish_divergence"
            desc = "缩量上涨: 上涨乏力，量价背离，警惕回调"
        elif is_down and vol_surge:
            regime = "panic_selling"
            desc = "放量下跌: 恐慌抛售，短期可能继续下探"
        elif is_down and vol_shrink:
            regime = "bearish_shrink"
            desc = "缩量下跌: 惜售情绪，接近底部区域的可能性较大"
        elif is_up:
            regime = "normal_up"
            desc = "小幅上涨: 量能变化不大"
        elif is_down:
            regime = "normal_down"
            desc = "小幅下跌: 量能变化不大"
        else:
            regime = "flat"
            desc = "横盘: 方向不明确"

        return {
            "regime": regime,
            "description": desc,
            "price_change": round(pc, 2),
            "volume_change": round(vc, 2),
        }

    def volume_regime(self, volume: pd.Series, lookback: int = 252) -> str:
        """量能状态分类
        
        基于历史分位数判断当前量能水平状态。
        
        Returns
        -------
        str
            地量/缩量/正常/放量/天量
        """
        if len(volume) < 60:
            return "正常"
        latest = volume.iloc[-1]
        pct = (volume <= latest).sum() / max(len(volume), 1)
        # Q22-fix: 原实现把 pct<0.05 判"天量"、pct>0.95 判"地量"——标签完全颠倒
        # （pct 是历史中 <= 当日量能的比例：接近 1 说明当日是历史最高量=天量，
        # 接近 0 说明是历史最低量=地量）。已实测复现：末日最高量输出"地量"。
        if pct > 0.95:
            return "天量"
        elif pct > 0.8:
            return "放量"
        elif pct < 0.05:
            return "地量"
        elif pct < 0.2:
            return "缩量"
        else:
            return "正常"


class VWAPAnalysis:
    """成交量加权均价"""

    def vwap(self, price: pd.Series, volume: pd.Series) -> pd.Series:
        """VWAP = Σ(P_i * V_i) / Σ(V_i)"""
        cum_pv = (price * volume).cumsum()
        cum_v = volume.cumsum()
        return cum_pv / cum_v.clip(lower=1e-8)

    def vwap_deviation(self, close: pd.Series, vwap: pd.Series) -> pd.Series:
        """收盘价偏离VWAP的程度（%）"""
        return (close / vwap - 1) * 100

    def vwap_band(self, vwap_series: pd.Series, std_dev: float = 1.0) -> tuple:
        """VWAP上下轨"""
        std = vwap_series.rolling(20).std()
        return (vwap_series - std_dev * std, vwap_series + std_dev * std)


class OBVAnalysis:
    """OBV（On-Balance Volume）能量潮指标"""

    def obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """OBV计算：收盘涨则加成交量，跌则减成交量

        P2-Q22-fix(L251): 空序列时 direction.iloc[0] 抛 IndexError，增加保护。
        """
        if close is None or len(close) == 0:
            return pd.Series(dtype=float)
        direction = (close.diff() > 0).astype(int) * 2 - 1
        direction.iloc[0] = 0
        return (direction * volume).cumsum()

    def obv_ma(self, obv_series: pd.Series, period: int = 20) -> pd.Series:
        """OBV均线"""
        return obv_series.rolling(period).mean()

    def obv_divergence(self, price: pd.Series, obv_series: pd.Series,
                        lookback: int = 20) -> dict:
        """OBV与价格背离"""
        if len(price) < lookback or len(obv_series) < lookback:
            return {}
        common = price.index.intersection(obv_series.index)[-lookback:]
        if len(common) < 10:
            return {}
        p = price.loc[common]
        o = obv_series.loc[common]
        p_high = p.max()
        o_high = o.max()
        p_low = p.min()
        o_low = o.min()
        bearish = p.iloc[-1] >= p_high * 0.995 and o.iloc[-1] < o_high * 0.98
        bullish = p.iloc[-1] <= p_low * 1.005 and o.iloc[-1] > o_low * 0.98
        return {
            "bullish_divergence": bullish,
            "bearish_divergence": bearish,
        }

    def obv_breakout(self, obv_series: pd.Series, period: int = 20) -> pd.Series:
        """OBV突破N日新高"""
        return (obv_series >= obv_series.rolling(period).max()).astype(int)


class AccumulationDistribution:
    """A/D线（资金流向指标）"""

    def a_d_line(self, high: pd.Series, low: pd.Series,
                  close: pd.Series, volume: pd.Series) -> pd.Series:
        """A/D线: Multiplier * Volume 的累积
        
        Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
        """
        hl = high - low
        multiplier = ((close - low) - (high - close)) / hl.clip(lower=1e-8)
        return (multiplier * volume).cumsum()

    def money_flow_ratio(self, close: pd.Series, high: pd.Series,
                          low: pd.Series, volume: pd.Series) -> pd.Series:
        """资金流向比例"""
        a_d = self.a_d_line(high, low, close, volume)
        return a_d / volume.rolling(20).mean().clip(lower=1)

    def cmf(self, close: pd.Series, high: pd.Series, low: pd.Series,
             volume: pd.Series, period: int = 20) -> pd.Series:
        """Chaikin Money Flow"""
        multiplier = ((close - low) - (high - close)) / (high - low).clip(lower=1e-8)
        mf_volume = multiplier * volume
        return mf_volume.rolling(period).sum() / volume.rolling(period).sum().clip(lower=1)


class VPINAnalysis:
    """VPIN (Volume-synchronized Probability of Informed Trading)
    
    基于成交量分桶的知情交易概率指标。
    高VPIN表示知情交易者活跃，市场信息不对称程度高，短期反转概率大。
    """

    def vpin(self, trades: pd.DataFrame, bucket_volume: float = 500_000,
              n_buckets: int = 50) -> pd.Series:
        """计算VPIN
        
        Parameters
        ----------
        trades : pd.DataFrame
            逐笔交易数据，需含price, volume, direction(1买/-1卖)
        bucket_volume : float
            每个桶的成交量（单位：股/手）
        n_buckets : int
            用于计算VPIN的桶数
        
        Returns
        -------
        pd.Series
            VPIN序列
        """
        buckets = []
        v_balance = 0.0
        v_buy = 0.0
        v_sell = 0.0

        for _, trade in trades.iterrows():
            vol = trade.get("volume", 0)
            direction = trade.get("direction", 0)
            if direction > 0:
                v_buy += vol
            else:
                v_sell += vol
            v_balance += vol

            if v_balance >= bucket_volume:
                buckets.append(min(v_buy, v_sell) / max(v_balance, 1))
                v_balance = 0.0
                v_buy = 0.0
                v_sell = 0.0

        if len(buckets) < n_buckets:
            return pd.Series(dtype=float)
        bins = pd.Series(buckets).rolling(n_buckets).mean()
        return bins.dropna()

    def vpin_toxicity(self, vpin_series: pd.Series, threshold: float = 0.5) -> pd.Series:
        """VPIN毒性判断：高于阈值表示高毒性"""
        return (vpin_series > threshold).astype(int)


class VolumeReport:
    """量能分析报告"""

    def __init__(self):
        self.va = VolumeAnalysis()
        self.vwap = VWAPAnalysis()
        self.obv = OBVAnalysis()

    def full_report(self, volume_series: pd.Series,
                    price_change: pd.Series) -> str:
        """完整量能分析报告"""
        ratios = self.va.volume_ma_ratios(volume_series)
        vp_div = self.va.volume_price_divergence(price_change,
                                                   volume_series.pct_change() * 100)
        regime = self.va.volume_regime(volume_series)

        lines = [
            "=" * 55,
            "【量能分析报告】",
            "=" * 55,
            "",
            f"  全市场成交额: {volume_series.iloc[-1] if len(volume_series) > 0 else 0:.0f}亿",
            f"  量能状态: {regime}",
            f"  相对MA5: {ratios.get('MA5', 0):.2f}倍",
            f"  相对MA20: {ratios.get('MA20', 0):.2f}倍",
            f"  相对MA60: {ratios.get('MA60', 0):.2f}倍",
            "",
            "【量价关系】",
            f"  {vp_div['description']}",
            "",
            "【量能趋势】",
        ]
        if len(volume_series) > 20:
            vol_20d = volume_series.iloc[-20:]
            vol_prev = volume_series.iloc[-40:-20]
            trend = "放大" if vol_20d.mean() > vol_prev.mean() * 1.1 else ("缩小" if vol_20d.mean() < vol_prev.mean() * 0.9 else "持平")
            lines.append(f"  近20日量能较前20日: {trend} ({vol_20d.mean()/vol_prev.mean()*100-100:+.1f}%)")

        lines.extend([
            "",
            "=" * 55,
        ])
        return "\n".join(lines)

    def daily_brief(self, volume_series: pd.Series) -> str:
        """每日量能简报"""
        if len(volume_series) < 2:
            return "暂无量能数据"
        chg = (volume_series.iloc[-1] / volume_series.iloc[-2] - 1) * 100
        direction = "放大" if chg > 0 else "缩小"
        return f"【量能简报】成交额{volume_series.iloc[-1]:.0f}亿 ({direction}{abs(chg):.1f}%)"

    def mechanism_explain(self, volume_series: pd.Series,
                          price_change: pd.Series) -> str:
        """量能变化的机制解释"""
        vp = self.va.volume_price_divergence(price_change,
                                               volume_series.pct_change() * 100)
        lines = [
            "量能机制解读：",
        ]
        if vp["regime"] == "bullish_volume":
            lines.append("- 放量上涨说明买盘充裕，场外资金积极入场，市场信心强。如果这种状态持续，上升趋势有可靠支撑。")
        elif vp["regime"] == "bearish_divergence":
            lines.append("- 缩量上涨说明场内资金已充分介入，场外接力资金不足，是典型的量价背离信号。历史上这种情况往往跟随调整。")
        elif vp["regime"] == "panic_selling":
            lines.append("- 放量下跌说明恐慌盘涌出，这是最强烈的中期看空信号之一。需要等待恐慌情绪释放、成交量恢复正常后才可能见底。")
        elif vp["regime"] == "bearish_shrink":
            lines.append("- 缩量下跌说明主动卖盘减少，抛压减弱，是潜在的筑底信号。后续需要放量阳线确认底部。")
        return "\n".join(lines)


__all__ = [
    "VolumeAnalysis", "VWAPAnalysis", "OBVAnalysis",
    "AccumulationDistribution", "VPINAnalysis",
    "VolumeReport",
]
