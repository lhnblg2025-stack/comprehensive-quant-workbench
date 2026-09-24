"""
annual_pattern — A股年度模式 (V5)

经典A股日历效应量化：
  1. 春季躁动 (1-3月): 年初信贷宽松 + 两会预期
  2. 五穷六绝 (5-6月): 年报季报结束后真空期
  3. 七翻身 (7-8月): 中报预期驱动
  4. 秋季行情 (9-11月): 三季报 + 年末博弈
  5. 年末行情 (12月): 排名博弈 + 来年布局

统计各窗口的历史表现, 给出胜率和平均收益。
"""

from __future__ import annotations

from datetime import timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CST = timezone(timedelta(hours=8))
ROOT = Path(__file__).resolve().parent.parent


# P2-Q23-fix(M255): 统计显著性检验——单样本自助法(bootstrap)置信区间与双侧
# p 值，纯 numpy 实现(无 scipy 依赖)。seasonality 各模块统一用此函数，避免
# "近10年最好/最差月"式的无显著性挖掘把噪声当规律；配合 Bonferroni 校正提示，
# 多窗口同时检验时单次阈值应降到 alpha/检验次数。
def _bootstrap_test(rets: list[float], n_boot: int = 2000, seed: int = 7) -> dict[str, Any]:
    """单样本自助检验：均值是否显著区别于 0。

    Returns:
        dict: {t_stat, p_value, ci_low, ci_high, significant}
        - p_value: 双侧自助 p 值（重采样均值越过 0 的比例 × 2，min=1.0）
        - significant: p<0.05 且 95% CI 不含 0
        - 样本<3 或方差为 0 时返回全 None（不臆造显著性）
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


# P2-Q23-fix(M255): 多重比较校正说明。n_tests 个窗口同时检验时，Bonferroni
# 单次阈值 = alpha/n_tests，用于提示"best/worst"型结论需打折解读。
def _bonferroni_note(n_tests: int, alpha: float = 0.05) -> str:
    return (f"同时检验{n_tests}个假设,Bonferroni校正后单次显著性阈值"
            f"{alpha}/{n_tests}={alpha / n_tests:.4f}")


class AnnualPattern:
    """年度模式分析。"""

    # 经典A股日历窗口（月份范围）
    PATTERNS = {
        "春季躁动": (1, 3),
        "五穷六绝": (5, 6),
        "七翻身": (7, 8),
        "秋季行情": (9, 11),
        "年末行情": (12, 12),
    }

    def __init__(self) -> None:
        self.cache: dict[str, Any] = {}

    def compute(self, years: int = 15) -> dict[str, Any]:
        """计算各年度窗口的历史表现。"""
        cache_key = f"annual_{years}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        result: dict[str, Any] = {}

        try:
            import akshare as ak
            df = ak.stock_zh_index_daily(symbol="sh000001")
            if df is None or df.empty:
                return result

            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
            cutoff = df["date"].max() - pd.DateOffset(years=years)
            df = df[df["date"] >= cutoff]
            df["return"] = df["close"].pct_change()
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month

            patterns = []
            for name, (start_m, end_m) in self.PATTERNS.items():
                window_data = df[(df["month"] >= start_m) & (df["month"] <= end_m)]

                # 每年窗口内累计收益
                # P2-Q23-fix(M256): 原 `yr["return"].sum()` 为日收益简单加和
                # （非复利），与 month_effect 的 (1+r).prod()-1 复利口径不一致，
                # 实测春季躁动 2019 偏差 1.85pp(22.09% vs 23.93%)。改复利口径；
                # prod 遇 NaN 会传播，先 dropna（sum 原本跳过 NaN，行为保持一致）。
                yearly_returns = {}
                for year in sorted(window_data["year"].unique()):
                    yr = window_data[window_data["year"] == year]
                    ret = (1.0 + yr["return"].dropna()).prod() - 1.0
                    yearly_returns[int(year)] = round(float(ret), 4)

                all_rets = list(yearly_returns.values())
                win_rate = np.mean(np.array(all_rets) > 0) if all_rets else 0

                # P2-Q23-fix(M255): 显著性检验（自助 95% CI + 双侧 p 值）。
                # 样本数=窗口年数（n≈10~15），远小于 t 检验的渐近要求，
                # 用自助法不依赖正态假设；p 值与 CI 直接写入结果，供调用方
                # 判断"最好/最差窗口"是否只是噪声。
                sig = _bootstrap_test(all_rets) if all_rets else {
                    "t_stat": None, "p_value": None, "ci_low": None,
                    "ci_high": None, "significant": None,
                }

                patterns.append({
                    "pattern": name,
                    "window": f"{start_m}月-{end_m}月",
                    "avg_return": round(float(np.mean(all_rets)), 4) if all_rets else 0,
                    "win_rate": round(float(win_rate), 4),
                    "std": round(float(np.std(all_rets, ddof=1)), 4) if len(all_rets) > 1 else 0,
                    "best_year": max(yearly_returns, key=yearly_returns.get) if yearly_returns else None,
                    "best_return": max(yearly_returns.values()) if yearly_returns else 0,
                    "worst_year": min(yearly_returns, key=yearly_returns.get) if yearly_returns else None,
                    "worst_return": min(yearly_returns.values()) if yearly_returns else 0,
                    "count": int(len(yearly_returns)),
                    "yearly_returns": yearly_returns,
                    "p_value": sig["p_value"],
                    "significant": sig["significant"],
                    "ci_low": sig["ci_low"],
                    "ci_high": sig["ci_high"],
                })

            result["patterns"] = patterns

            # 找最好和最差的窗口
            if patterns:
                best = max(patterns, key=lambda p: p["avg_return"])
                worst = min(patterns, key=lambda p: p["avg_return"])
                # P2-Q23-fix(M255): summary 注明 p 值，显著性与否一目了然；
                # 并附 Bonferroni 多重比较提示，避免把 5 个窗口中偶然的最值当规律。
                sig_txt = (f"(p={best['p_value']:.3f}{'显著' if best['significant'] else '不显著'})"
                           if best.get("p_value") is not None else "(样本不足,未检验)")
                sig_txt_w = (f"(p={worst['p_value']:.3f}{'显著' if worst['significant'] else '不显著'})"
                             if worst.get("p_value") is not None else "(样本不足,未检验)")
                result["summary"] = (
                    f"最强窗口'{best['pattern']}'({best['window']}, "
                    f"均涨{best['avg_return']:.1%}, 胜率{best['win_rate']:.0%}{sig_txt}), "
                    f"最弱窗口'{worst['pattern']}'({worst['window']}, "
                    f"均涨{worst['avg_return']:.1%}{sig_txt_w})"
                )
                result["bonferroni_note"] = _bonferroni_note(len(patterns))

        except Exception as e:
            result["error"] = str(e)

        self.cache[cache_key] = result
        return result


def main() -> None:
    ap = AnnualPattern()
    r = ap.compute()
    print("═" * 60)
    print("  A股年度日历效应")
    print("═" * 60)
    for p in r.get("patterns", []):
        arrow = "📈" if p["avg_return"] > 0 else "📉"
        bar_len = int(abs(p["avg_return"]) * 100)
        bar = "█" * min(bar_len, 30)
        print(f"\n  {arrow} {p['pattern']:<8} ({p['window']:<8})")
        print(f"     均涨: {p['avg_return']:>+7.2%}  胜率: {p['win_rate']:.0%}  "
              f"波动: {p['std']:.2%}  {bar}")
        print(f"     最好: {p['best_return']:.1%}({p.get('best_year', '?')})  "
              f"最差: {p['worst_return']:.1%}({p.get('worst_year', '?')})")
    print(f"\n  {r.get('summary', '')}")


if __name__ == "__main__":
    main()
