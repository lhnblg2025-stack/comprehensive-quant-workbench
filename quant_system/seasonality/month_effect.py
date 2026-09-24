"""
month_effect — A股各月/各季度历史表现统计 (V5)

回答："历史上这个月是涨还是跌？"
不预测，只统计过去N年的客观规律。
"""

from __future__ import annotations

from datetime import timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CST = timezone(timedelta(hours=8))
ROOT = Path(__file__).resolve().parent.parent


# P2-Q23-fix(M255): 单样本自助显著性检验（均值≠0 的 95% CI + 双侧 p 值），
# 纯 numpy 实现。seasonality 各模块统一使用，避免"最好/最差月"无显著性
# 挖掘把噪声当规律。
def _bootstrap_test(rets: list[float], n_boot: int = 2000, seed: int = 7) -> dict[str, Any]:
    """单样本自助检验：均值是否显著区别于 0。

    Returns:
        {t_stat, p_value, ci_low, ci_high, significant}；样本<3 或方差为 0
        时返回全 None（不臆造显著性）。
    """
    empty = {"t_stat": None, "p_value": None, "ci_low": None,
             "ci_high": None, "significant": None}
    arr = np.asarray(rets, dtype=float)
    n = len(arr)
    if n < 3:
        return empty
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    if sd == 0 or not np.isfinite(sd):
        return empty
    se = sd / np.sqrt(n)
    t_stat = float(mean / se)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    ci_low, ci_high = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    tail = float(np.mean(means <= 0)) if mean > 0 else float(np.mean(means >= 0))
    p_value = float(min(1.0, 2.0 * tail))
    significant = bool(p_value < 0.05 and (ci_low > 0 or ci_high < 0))
    return {"t_stat": round(t_stat, 3), "p_value": round(p_value, 4),
            "ci_low": round(ci_low, 4), "ci_high": round(ci_high, 4),
            "significant": significant}


# P2-Q23-fix(M255): 多重比较校正说明。12 个月同时检验时，Bonferroni 单次
# 阈值 = alpha/12。
def _bonferroni_note(n_tests: int, alpha: float = 0.05) -> str:
    return (f"同时检验{n_tests}个假设,Bonferroni校正后单次显著性阈值"
            f"{alpha}/{n_tests}={alpha / n_tests:.4f}")


class MonthEffect:
    """各月历史表现统计。"""

    def __init__(self) -> None:
        self.cache: dict[str, Any] = {}

    def compute(self, years: int = 10) -> dict[str, Any]:
        """计算各月历史表现。

        Args:
            years: 回看年数

        Returns:
            {
                "monthly_stats": [{month, avg_return, win_rate, std, median, max, min, count}],
                "best_month": {month, avg_return},
                "worst_month": {month, avg_return},
                "quarterly_stats": [{quarter, avg_return, win_rate}],
            }
        """
        cache_key = f"month_{years}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        result: dict[str, Any] = {
            "monthly_stats": [],
            "best_month": {},
            "worst_month": {},
            "quarterly_stats": [],
        }

        try:
            import akshare as ak
            df = ak.stock_zh_index_daily(symbol="sh000001")
            if df is None or df.empty:
                return result

            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")

            # 限制年数
            cutoff = df["date"].max() - pd.DateOffset(years=years)
            df = df[df["date"] >= cutoff]

            # 计算日收益
            df["return"] = df["close"].pct_change()

            # 按月分组
            df["month"] = df["date"].dt.month
            df["year"] = df["date"].dt.year

            # 每月累计收益
            monthly = df.groupby(["year", "month"])["return"].apply(
                lambda x: (1 + x).prod() - 1
            ).reset_index()
            monthly.columns = ["year", "month", "return"]

            stats = []
            for m in range(1, 13):
                m_data = monthly[monthly["month"] == m]["return"].dropna().values
                if len(m_data) > 0:
                    # P2-Q23-fix(M255): 显著性检验（n=月数≈10，自助法不依赖正态假设）
                    sig = _bootstrap_test(m_data.tolist())
                    stats.append({
                        "month": m,
                        "avg_return": round(float(np.mean(m_data)), 4),
                        "win_rate": round(float(np.mean(m_data > 0)), 4),
                        "std": round(float(np.std(m_data, ddof=1)), 4),
                        "median": round(float(np.median(m_data)), 4),
                        "max": round(float(np.max(m_data)), 4),
                        "min": round(float(np.min(m_data)), 4),
                        "count": int(len(m_data)),
                        "p_value": sig["p_value"],
                        "significant": sig["significant"],
                        "ci_low": sig["ci_low"],
                        "ci_high": sig["ci_high"],
                    })

            result["monthly_stats"] = stats

            if stats:
                best = max(stats, key=lambda s: s["avg_return"])
                worst = min(stats, key=lambda s: s["avg_return"])
                result["best_month"] = {"month": best["month"],
                                         "avg_return": best["avg_return"]}
                result["worst_month"] = {"month": worst["month"],
                                          "avg_return": worst["avg_return"]}

            # 季度统计
            q_map = {1: "Q1", 2: "Q1", 3: "Q1", 4: "Q2", 5: "Q2", 6: "Q2",
                     7: "Q3", 8: "Q3", 9: "Q3", 10: "Q4", 11: "Q4", 12: "Q4"}
            monthly["quarter"] = monthly["month"].map(q_map)
            q_stats = monthly.groupby("quarter")["return"].agg(
                ["mean", lambda x: np.mean(x > 0)]
            ).reset_index()
            q_stats.columns = ["quarter", "avg_return", "win_rate"]

            result["quarterly_stats"] = [
                {
                    "quarter": r["quarter"],
                    "avg_return": round(float(r["avg_return"]), 4),
                    "win_rate": round(float(r["win_rate"]), 4),
                }
                for _, r in q_stats.iterrows()
            ]

            # 总结
            month_names = ["", "1月", "2月", "3月", "4月", "5月", "6月",
                          "7月", "8月", "9月", "10月", "11月", "12月"]
            # P2-Q23-fix(L270): 原 `if best["month"] and worst["month"]:` 在 stats
            # 为空时 best/worst 未定义 → NameError。改以 `if stats:` 为前置条件
            # （best/worst 仅在 stats 非空时定义）。同时注明 p 值，区分真实规律
            # 与偶然最值；并附 12 月多重比较 Bonferroni 提示。
            if stats:
                # 月份样本可能不足 3 个（p_value=None），summary 需防御 None
                bp = (f"p={best['p_value']:.3f}{'显著' if best['significant'] else '不显著'}"
                      if best.get("p_value") is not None else "样本不足,未检验")
                wp = (f"p={worst['p_value']:.3f}{'显著' if worst['significant'] else '不显著'}"
                      if worst.get("p_value") is not None else "样本不足,未检验")
                result["summary"] = (
                    f"近{years}年最好的是{month_names[best['month']]}(均涨{best['avg_return']:.1%},{bp}), "
                    f"最差的是{month_names[worst['month']]}(均涨{worst['avg_return']:.1%},{wp})"
                )
                result["bonferroni_note"] = _bonferroni_note(12)

        except Exception as e:
            result["error"] = str(e)

        self.cache[cache_key] = result
        return result


def main() -> None:
    me = MonthEffect()
    r = me.compute(years=10)
    print("═" * 55)
    print("  A股各月表现 (近10年)")
    print("═" * 55)
    for s in r.get("monthly_stats", []):
        bar = "█" * int(abs(s["avg_return"]) * 200) if s["avg_return"] != 0 else ""
        direction = "📈" if s["avg_return"] > 0 else "📉" if s["avg_return"] < 0 else "➡️"
        print(f"  {s['month']:>2}月 {direction} {s['avg_return']:>+7.2%}  "
              f"胜率{s['win_rate']:.0%}  {bar}")

    print()
    qs = r.get("quarterly_stats", [])
    if qs:
        print("  ── 季度 ──")
        for q in qs:
            print(f"  {q['quarter']}: {q['avg_return']:>+7.2%} 胜率{q['win_rate']:.0%}")

    print(f"\n  {r.get('summary', '')}")


if __name__ == "__main__":
    main()
