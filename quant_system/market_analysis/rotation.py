"""
market_analysis/rotation.py — 行业轮动与风格分析
V4.1 feature

行业轮动是A股市场最重要的特征之一。不同阶段有不同领涨行业。
分析行业轮动速度、风格切换、产业链传导，为行业配置提供依据。
"""

import numpy as np
import pandas as pd


class SectorRotation:
    """行业轮动分析
    
    行业轮动速度反映市场热点的持续性。
    轮动过快说明市场缺乏主线，是短线特征；
    轮动缓慢说明有明显领涨主线，中期趋势健康发展。
    """

    def __init__(self):
        pass

    def sector_performance(self, sector_returns: dict) -> pd.Series:
        """行业涨跌幅排名
        
        Parameters
        ----------
        sector_returns : dict
            {行业名: 涨跌幅(%)}
        
        Returns
        -------
        pd.Series
            排序后的行业涨跌幅
        """
        return pd.Series(sector_returns).sort_values(ascending=False)

    def sector_rank_change(self, current_rank: pd.Series,
                           prev_rank: pd.Series) -> dict:
        """行业排名变化统计
        
        Returns
        -------
        dict
            avg_rank_change: 平均排名变化
            max_rank_change: 最大排名变化
            upgraded_sectors: 排名提升最多的行业
            downgraded_sectors: 排名下降最多的行业
        """
        common = current_rank.index.intersection(prev_rank.index)
        if len(common) == 0:
            return {}
        change = pd.Series({c: prev_rank[c] - current_rank[c] for c in common})
        return {
            "avg_rank_change": round(change.abs().mean(), 1),
            "max_rank_change": round(change.abs().max(), 1),
            "upgraded_sectors": change.nlargest(3).to_dict(),
            "downgraded_sectors": change.nsmallest(3).to_dict(),
        }

    def sector_winner_loser_ratio(self, sector_returns: dict,
                                   top_pct: float = 0.25) -> float:
        """强势/弱势行业比"""
        ret_series = pd.Series(sector_returns).sort_values(ascending=False)
        n = max(1, int(len(ret_series) * top_pct))
        winners = ret_series.head(n)
        losers = ret_series.tail(n)
        return winners.mean() / max(abs(losers.mean()), 0.01)

    def rotation_index(self, sector_returns_df: pd.DataFrame,
                        lookback: int = 20) -> float:
        """轮动速度指标
        
        基于行业排名变化的标准差。数值越高，轮动越快。
        
        Parameters
        ----------
        sector_returns_df : pd.DataFrame
            行业涨跌幅矩阵，列=行业，行=日期
        lookback : int
            回溯周期
        
        Returns
        -------
        float
            轮动速度指数
        """
        if len(sector_returns_df) < lookback:
            return 0.0
        recent = sector_returns_df.iloc[-lookback:]
        ranks = recent.rank(axis=1)
        rank_changes = ranks.diff().abs().sum(axis=1).dropna()
        return round(rank_changes.mean(), 1)

    def rotation_regime(self, index_value: float) -> str:
        """轮动状态"""
        if index_value > 15:
            return "快速轮动"
        elif index_value > 8:
            return "中速轮动"
        else:
            return "慢速轮动"

    def rotation_matrix(self, sector_returns_df: pd.DataFrame,
                         n_weeks: int = 12) -> pd.DataFrame:
        """行业相对强度矩阵
        
        计算每个行业在每周的相对强度排名。
        热力图数据格式。
        """
        if len(sector_returns_df) < n_weeks:
            return pd.DataFrame()
        recent = sector_returns_df.iloc[-n_weeks:]
        return recent.rank(axis=1)

    def sector_leadership(self, sector_returns_df: pd.DataFrame,
                           lookback: int = 60) -> dict:
        """领涨力评分
        
        基于以下维度评分(0-100)：
        1. 累计涨幅排名
        2. 涨幅持续性（正收益天数占比）
        3. 超额收益稳定性
        """
        if len(sector_returns_df) < lookback:
            return {}
        recent = sector_returns_df.iloc[-lookback:]
        cum_ret = (1 + recent / 100).prod() - 1
        pos_days = (recent > 0).sum() / len(recent)
        # 超额稳定性 = 日均超额收益 / 超额收益标准差
        excess = recent.sub(recent.mean(axis=1), axis=0)
        stability = excess.mean() / excess.std().clip(lower=0.001)

        score = (cum_ret.rank(pct=True) * 40 +
                 pos_days * 30 +
                 (stability.rank(pct=True) * 30))
        return score.nlargest(5).to_dict()


class StyleRotation:
    """风格轮动分析
    
    大盘/中盘/小盘/微盘风格切换是A股的重要特征。
    风格连续性指数反映当前市场的主线风格。
    """

    def style_performance(self, large: float, mid: float,
                           small: float, micro: float) -> dict:
        """风格涨跌幅"""
        return {
            "大盘": large,
            "中盘": mid,
            "小盘": small,
            "微盘": micro,
        }

    def style_diffusion_index(self, large_small_diff: pd.Series,
                               lookback: int = 20) -> float:
        """风格扩散指数
        
        大盘vs小盘的分化程度。数值越高，风格越极端。
        """
        if len(large_small_diff) < lookback:
            return 0.0
        return round(large_small_diff.iloc[-lookback:].abs().mean(), 1)

    def style_continuity(self, style_returns: pd.DataFrame,
                          lookback: int = 60) -> float:
        """风格持续性评分
        
        最近一段时期风格领先方向的一致性。
        1=高度一致持续, -1=频繁切换
        """
        if len(style_returns) < lookback:
            return 0.0
        large_vs_small = style_returns["大盘"] - style_returns["小盘"]
        pos_days = (large_vs_small > 0).sum()
        neg_days = (large_vs_small < 0).sum()
        return (pos_days - neg_days) / max(pos_days + neg_days, 1)

    def style_regime(self, large_small_diff: float) -> str:
        """风格状态"""
        if large_small_diff > 1.5:
            return "大盘显著跑赢"
        elif large_small_diff > 0.5:
            return "大盘略占优"
        elif large_small_diff < -1.5:
            return "小盘显著跑赢"
        elif large_small_diff < -0.5:
            return "小盘略占优"
        else:
            return "风格均衡"


class IndustryChainTransmission:
    """产业链传导分析
    
    上游（原材料）→中游（制造）→下游（消费）的传导关系。
    当上游涨价时，利润会沿产业链传导；下游需求走弱也会向上游传导。
    """

    def upstream_downstream_corr(self, upstream_ret: pd.Series,
                                  downstream_ret: pd.Series,
                                  lookback: int = 60) -> float:
        """上下游相关系数"""
        if len(upstream_ret) < lookback or len(downstream_ret) < lookback:
            return 0.0
        return upstream_ret.iloc[-lookback:].corr(downstream_ret.iloc[-lookback:])

    def supply_chain_impact(self, chain_map: dict,
                             trigger_industry: str,
                             impact: float) -> dict:
        """供应链冲击传导模拟
        
        Parameters
        ----------
        chain_map : dict
            {行业名: [受其影响的行业列表]}
        trigger_industry : str
            触发冲击的行业
        impact : float
            冲击幅度（%）
        
        Returns
        -------
        dict
            {行业名: 传导幅度}
        """
        result = {trigger_industry: impact}
        impacted = chain_map.get(trigger_industry, [])
        decay = 0.5  # 每次传导衰减50%
        for ind in impacted:
            result[ind] = impact * decay
        return result

    def industry_leading_lag(self, lag_industry: pd.Series,
                              lead_candidates: dict,
                              max_lag: int = 20) -> dict:
        """找出对某行业有领先关系的行业

        Parameters
        ----------
        lag_industry : pd.Series
            滞后行业收益率序列
        lead_candidates : dict
            {候选领先行业名: 收益率序列}
        max_lag : int
            最大滞后天数

        Returns
        -------
        dict
            {领先行业: (最佳滞后天数, 最大相关系数)}
        """
        best = {}
        for name, lead_series in lead_candidates.items():
            max_corr = 0
            best_lag = 0
            for lag in range(1, max_lag + 1):
                if len(lead_series) <= lag or len(lag_industry) <= lag:
                    continue
                lead = lead_series.iloc[:-lag] if lag > 0 else lead_series
                lag_val = lag_industry.iloc[lag:]
                # P2-Q22-fix(M240): 原位置切片 lead[:-lag] vs lag[lag:] 未校验
                # 两序列索引(日期)一致性，索引不一致时静默错配数据。改用
                # pd.concat 按索引内连接对齐后再计算相关系数，保证只使用共同日期。
                aligned = pd.concat([lead, lag_val], axis=1, join="inner").dropna()
                if len(aligned) < 5:
                    continue
                corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
                if abs(corr) > abs(max_corr):
                    max_corr = corr
                    best_lag = lag
            if best_lag > 0:
                best[name] = (best_lag, round(max_corr, 3))
        return dict(sorted(best.items(), key=lambda x: -abs(x[1][1]))[:5])

    def transmission_mechanism(self, chain_data: dict) -> str:
        """传导机制解释"""
        lines = ["产业链传导机制分析："]
        lines.append("- 上游→下游传导：上游原材料涨价会导致下游成本上升，利润从下游向上游转移")
        lines.append("- 下游→上游传导：下游需求走弱会减少上游订单，导致上游库存累积")
        lines.append("- 横向传导：同环节行业间存在替代竞争关系")
        return "\n".join(lines)


class RelativeStrength:
    """相对强度分析"""

    def rs_ratio(self, industry_ret: pd.Series,
                  market_ret: pd.Series) -> pd.Series:
        """RS比率 = 行业收益 / 市场收益"""
        return (1 + industry_ret / 100).cumprod() / (1 + market_ret / 100).cumprod() * 100

    def rs_momentum(self, rs_ratio: pd.Series, period: int = 20) -> pd.Series:
        """RS动量"""
        return rs_ratio.rolling(period).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == period else 0,
            raw=True
        )

    def rotation_heatmap(self, sector_returns_df: pd.DataFrame,
                          n_weeks: int = 12) -> pd.DataFrame:
        """行业轮动热力图数据"""
        if len(sector_returns_df) < n_weeks:
            return pd.DataFrame()
        recent = sector_returns_df.iloc[-n_weeks:]
        return recent.T


class RotationReport:
    """轮动分析报告"""

    def __init__(self):
        self.sr = SectorRotation()
        self.style = StyleRotation()

    def full_report(self, sector_returns: dict,
                    style_data: dict = None,
                    sector_returns_df: pd.DataFrame = None) -> str:
        """完整轮动报告

        P2-Q22-fix(M239): 原 rotation_idx=0 占位使轮动速度永远输出
        "慢速轮动/正常/主线清晰"，结论误导。现接受可选历史行业收益矩阵
        sector_returns_df（行=日期，列=行业，至少20行），有数据时调用
        self.sr.rotation_index 计算真实轮动指数；无数据时显式标注"数据不足"，
        不输出误导性结论。参数为新增可选，兼容既有调用方。
        """
        perf = self.sr.sector_performance(sector_returns)
        # P2-Q22-fix(M239): 真实轮动指数需历史行业收益矩阵，单日涨跌幅无法计算
        if sector_returns_df is not None and len(sector_returns_df) >= 20:
            rotation_idx = self.sr.rotation_index(sector_returns_df)
            rotation_note = ""
        else:
            rotation_idx = None
            rotation_note = "（未提供≥20日历史行业收益矩阵，轮动速度不计算）"

        lines = [
            "=" * 55,
            "【行业轮动报告】",
            "=" * 55,
            "",
            "【行业涨跌幅TOP5】",
        ]
        for name, ret in perf.head(5).items():
            lines.append(f"  {name}: {ret:+.2f}%")
        lines.extend(["", "【行业涨跌幅LAST5】"])
        for name, ret in perf.tail(5).items():
            lines.append(f"  {name}: {ret:+.2f}%")

        if rotation_idx is not None:
            lines.extend([
                "",
                "【轮动速度】",
                f"  轮动指数: {rotation_idx} ({self.sr.rotation_regime(rotation_idx)})",
            ])
        else:
            lines.extend([
                "",
                "【轮动速度】",
                f"  轮动指数: — {rotation_note}",
            ])

        if style_data:
            lines.extend(["", "【风格表现】"])
            for style_name, ret in style_data.items():
                lines.append(f"  {style_name}: {ret:+.2f}%")

        lines.extend(["", "【轮动解读】"])
        if rotation_idx is None:
            lines.append("  数据不足，未计算轮动速度（需≥20日历史行业收益矩阵）。")
        else:
            lines.append("  行业轮动速度" + ("较快" if rotation_idx > 10 else "正常") + "，")
            lines.append(
                "  市场以快轮动为主，缺乏持续性主线，建议均衡配置" if rotation_idx > 10
                else "  市场主线清晰，可重点关注领涨行业")

        lines.extend([
            "",
            "=" * 55,
        ])
        return "\n".join(lines)

    def daily_brief(self, sector_returns: dict) -> str:
        """每日轮动简报"""
        perf = self.sr.sector_performance(sector_returns)
        top = perf.index[0] if len(perf) > 0 else "—"
        last = perf.index[-1] if len(perf) > 0 else "—"
        return f"【轮动简报】领涨:{top}({perf.iloc[0]:+.1f}%) 垫底:{last}({perf.iloc[-1]:+.1f}%)"

    def mechanism_explain(self, sector_returns: dict,
                          macro_context: str = "") -> str:
        """轮动机制解释"""
        perf = self.sr.sector_performance(sector_returns)
        top3 = perf.head(3)
        bottom3 = perf.tail(3)
        lines = [
            "行业轮动机制解读：",
            f"  领涨行业: {', '.join(f'{n}({r:+.1f}%)' for n, r in top3.items())}",
            f"  领跌行业: {', '.join(f'{n}({r:+.1f}%)' for n, r in bottom3.items())}",
            "",
            "  行业表现的驱动逻辑：",
        ]
        # 启发式判断逻辑
        if "电子" in perf.index and perf["电子"] > 2:
            lines.append("  - 电子行业领涨：通常与AI/半导体周期、消费电子复苏有关")
        if "银行" in perf.index and perf["银行"] > 1:
            lines.append("  - 银行走强：反映市场风险偏好降低，资金向低估值板块防御")
        if "食品饮料" in perf.index and perf["食品饮料"] > 1:
            lines.append("  - 消费走强：可能与消费刺激政策、CPI回升有关")
        if "国防军工" in perf.index and perf["国防军工"] > 2:
            lines.append("  - 军工走强：可能受地缘政治事件催化")
        if "煤炭" in perf.index and perf["煤炭"] > 1:
            lines.append("  - 资源品走强：受全球大宗商品价格、通胀预期影响")

        return "\n".join(lines)


__all__ = [
    "SectorRotation", "StyleRotation",
    "IndustryChainTransmission", "RelativeStrength",
    "RotationReport",
]
