"""
monitor.py — 因子监控器（Factor Monitor）
V4.1 feature

提供因子的实时监控面板：IC 时序、衰减剖面、相关性热图、优秀差股票。
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。
"""

import pandas as pd
from datetime import datetime
from .evaluation import FactorEvaluator
from .registry import FactorCategory, get_default_registry

logger = __import__('logging').getLogger(__name__)


class FactorMonitor:
    """因子监控器"""

    def __init__(self):
        self._ic_history: dict[str, pd.Series] = {}
        self._evaluator = FactorEvaluator()
        self._registry = get_default_registry()

    def update_ic(self, factor_name: str, alpha: pd.Series, forward_return: pd.Series):
        """记录因子的最新 IC（同日多次更新保留最新值）"""
        ic = self._evaluator.rank_ic(alpha, forward_return)
        today = datetime.now().strftime("%Y-%m-%d")
        if factor_name not in self._ic_history:
            self._ic_history[factor_name] = pd.Series(dtype=float)
        series = self._ic_history[factor_name]
        # P2-Q5-fix (L450): 同日多次更新互相覆盖（Series 按日期 label 赋值会隐式覆盖）。
        #   显式剔除同 label 旧值后追加，明确"同日取最新"，避免歧义。
        series = series[series.index != today]
        self._ic_history[factor_name] = pd.concat(
            [series, pd.Series([ic], index=[today])]
        )

    def ic_time_series(self, factor_name: str, window: int = 60) -> pd.Series:
        """返回因子的 IC 时序（最近 window 期）"""
        if factor_name not in self._ic_history:
            return pd.Series(dtype=float)
        series = self._ic_history[factor_name]
        return series.iloc[-window:] if len(series) > window else series

    def factor_decay_profile(self, alpha_df: pd.DataFrame,
                             forward_return: pd.Series,
                             max_lag: int = 20) -> pd.Series:
        """因子衰减剖面"""
        return self._evaluator.factor_decay(alpha_df, forward_return, max_lag)

    def factor_correlation_report(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """因子相关性报告"""
        return self._evaluator.factor_correlation_matrix(factor_df)

    def top_bottom_stocks(self, factor_name: str, factor_df: pd.DataFrame,
                          n: int = 10) -> tuple:
        """展示因子的最高/最低暴露股票。

        P2-Q5-fix (L448): 返回顺序与文档对齐——(最高暴露组, 最低暴露组)。
        V5.4 返回 (最低组, 最高组)，与 docstring"最高/最低暴露"相反。
        """
        if factor_name not in factor_df.columns:
            return (pd.Series(), pd.Series())
        sorted_ = factor_df[factor_name].dropna().sort_values()
        return (sorted_.tail(n), sorted_.head(n))

    def factor_performance_summary(self, factor_df: pd.DataFrame,
                                   forward_return: pd.Series) -> pd.DataFrame:
        """所有因子的绩效汇总"""
        rows = []
        for col in factor_df.columns:
            ic = self._evaluator.rank_ic(factor_df[col], forward_return)
            spread = self._evaluator.spread_return(factor_df[col], forward_return)
            rows.append({"factor": col, "rank_ic": ic, "spread": spread})
        return pd.DataFrame(rows).sort_values("rank_ic", ascending=False)

    def generate_monitor_report(self, factor_df: pd.DataFrame,
                                forward_return: pd.Series) -> str:
        """生成文本监控报告"""
        summary = self.factor_performance_summary(factor_df, forward_return)
        top5 = summary.head(5)
        bot5 = summary.tail(5)

        lines = [
            "=" * 55,
            "因子监控报告 (V4.1 feature)",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"因子数量: {len(summary)}",
            "=" * 55,
            "",
            "【IC 排名 Top 5】",
        ]

        for _, row in top5.iterrows():
            lines.append(f"  {row['factor']:<25s} IC={row['rank_ic']:+.4f}  Spread={row['spread']:+.4f}")

        lines.extend(["", "【IC 排名 Bottom 5】"])
        for _, row in bot5.iterrows():
            lines.append(f"  {row['factor']:<25s} IC={row['rank_ic']:+.4f}  Spread={row['spread']:+.4f}")

        # 按类别汇总
        lines.extend(["", "【按类别汇总】"])
        for cat in FactorCategory:
            cat_factors = [f.name for f in self._registry.list_by_category(cat)
                          if f.name in summary["factor"].values]
            if not cat_factors:
                continue
            cat_summary = summary[summary["factor"].isin(cat_factors)]
            avg_ic = cat_summary["rank_ic"].mean()
            avg_spread = cat_summary["spread"].mean()
            lines.append(f"  {cat.value:<15s} IC={avg_ic:+.4f}  Spread={avg_spread:+.4f}  ({len(cat_factors)}个)")

        # IC 稳定性
        lines.extend(["", "【IC 稳定性（最近20期IC > 0 占比 > 60% 的因子）】"])
        stable = []
        for factor_name in factor_df.columns:
            ic_series = self.ic_time_series(factor_name, window=20)
            if len(ic_series) >= 10 and (ic_series > 0).mean() > 0.6:
                stable.append(factor_name)
        lines.append(f"  {', '.join(stable[:10])}" if stable else "  无")

        lines.append("")
        lines.append("=" * 55)
        return "\n".join(lines)

    def monitor_dashboard(self, factor_df: pd.DataFrame,
                          forward_return: pd.Series) -> dict:
        """返回面板数据（供 Web/JSON 展示用）"""
        summary = self.factor_performance_summary(factor_df, forward_return)
        return {
            "summary": summary.to_dict("records"),
            "n_factors": len(summary),
            "mean_ic": summary["rank_ic"].mean(),
            "best_factor": summary.iloc[0]["factor"] if len(summary) > 0 else "",
            "worst_factor": summary.iloc[-1]["factor"] if len(summary) > 0 else "",
            "timestamp": datetime.now().isoformat(),
        }
