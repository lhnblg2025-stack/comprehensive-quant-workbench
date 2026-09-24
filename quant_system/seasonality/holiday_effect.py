"""
holiday_effect — A股节前/节后效应 (V5)

统计六大节日前后的市场表现：
  - 春节: "春节效应" 节前涨, 节后继续涨
  - 国庆: "国庆行情"
  - 五一/清明/端午/中秋: 小长假效应

数据源: akshare 上证指数日线 + 交易日历
"""

from __future__ import annotations

from datetime import timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CST = timezone(timedelta(hours=8))
ROOT = Path(__file__).resolve().parent.parent


# P2-Q23-fix(M255): 单样本自助显著性检验（均值≠0 的 95% CI + 双侧 p 值），
# 纯 numpy 实现。seasonality 各模块统一使用，避免无显著性挖掘把噪声当规律。
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


# P2-Q23-fix(M255): 多重比较校正说明（3 类节日 × 节前/节后 2 组 = 6 次检验）。
def _bonferroni_note(n_tests: int, alpha: float = 0.05) -> str:
    return (f"同时检验{n_tests}个假设,Bonferroni校正后单次显著性阈值"
            f"{alpha}/{n_tests}={alpha / n_tests:.4f}")


class HolidayEffect:
    """节日效应分析。"""

    def __init__(self) -> None:
        self.cache: dict[str, Any] = {}

        # 近10年春节日期（除夕/春节前最后一个交易日）
        self.spring_festival_dates = [
            "2017-01-27", "2018-02-15", "2019-02-04", "2020-01-24",
            "2021-02-11", "2022-01-31", "2023-01-20", "2024-02-09",
            "2025-01-28", "2026-02-16",
        ]

    def compute(self) -> dict[str, Any]:
        """计算各节日效应。"""
        if "holiday" in self.cache:
            return self.cache["holiday"]

        result: dict[str, Any] = {
            "holiday_effects": [],
            "summary": "",
        }

        try:
            import akshare as ak
            df = ak.stock_zh_index_daily(symbol="sh000001")
            if df is None or df.empty:
                return result

            df["date"] = pd.to_datetime(df["date"])
            df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
            df["return"] = df["close"].pct_change()

            # P1-Q23-fix(H04): 用交易日偏移替代日历日偏移。
            # 原实现按节假日日期做 ±N 日历日偏移，节后窗口整体落在休市期
            # （实测 春节/国庆 after 样本=0、五一=3），节前窗口因周末丢失
            # 20-30% 样本，春节参考日(除夕)10 年中仅 1 年为交易日。
            # 改为：参考日 = 节前最后一个交易日（交易日历中 ≤ 节假日日期者），
            #   节前5日 = 参考日及其前 4 个交易日（共 5 个交易日），
            #   节后5日 = 参考日之后第 1~5 个交易日。
            # 同时输出 before_n/after_n 并在样本不足时明确标注，避免把 0
            # 当作真实统计（原 after_avg/after_win_rate 恒为 0 却照常输出）。
            trade_dates = pd.DatetimeIndex(df["date"])

            holidays = {
                "春节": self.spring_festival_dates,
                "国庆": ["2017-09-29", "2018-09-28", "2019-09-30", "2020-09-30",
                        "2021-09-30", "2022-09-30", "2023-09-28", "2024-09-30",
                        "2025-09-30", "2026-09-30"],
                "五一": ["2017-04-28", "2018-04-27", "2019-04-30", "2020-04-30",
                        "2021-04-30", "2022-04-29", "2023-04-28", "2024-04-30",
                        "2025-04-30", "2026-04-30"],
            }

            effects = []
            for name, dates in holidays.items():
                before_returns: list[float] = []
                after_returns: list[float] = []
                before_total = 0
                after_total = 0
                count = 0

                for d in dates:
                    d_dt = pd.to_datetime(d)
                    # 参考日 = 节前最后一个交易日（≤ d 的最大交易日）
                    pos = int(trade_dates.searchsorted(d_dt, side="right")) - 1
                    if pos < 0:
                        continue
                    # P1-Q23-fix(H04): 参考日距节假日超过 10 天 → 数据未覆盖该年
                    # （如 2026 年国庆晚于最新行情日期，searchsorted 会误取到
                    # 数据末尾的交易日作为参考日），跳过该年，禁止用错误参考日
                    # 产生虚假统计。
                    ref = trade_dates[pos]
                    if (d_dt - ref).days > 10:
                        continue
                    count += 1

                    # 节前5日：参考日(节前最后交易日)及其前4个交易日
                    start = max(0, pos - 4)
                    for j in range(start, pos + 1):
                        before_total += 1
                        r = df["return"].iloc[j]
                        if not pd.isna(r):
                            before_returns.append(float(r))

                    # 节后5日：节后第1~5个交易日
                    for j in range(pos + 1, min(len(df), pos + 6)):
                        after_total += 1
                        r = df["return"].iloc[j]
                        if not pd.isna(r):
                            after_returns.append(float(r))

                # P1-Q23-fix(H04): 样本不足时明确标注（n 与期望数不符）
                note_parts: list[str] = []
                if count == 0:
                    note_parts.append("无有效参考日(数据覆盖不足)")
                else:
                    if count < len(dates):
                        note_parts.append(f"有效年份{count}/{len(dates)}")
                    if len(before_returns) < before_total:
                        note_parts.append(f"节前样本不足({len(before_returns)}/{before_total})")
                    if len(after_returns) < after_total:
                        note_parts.append(f"节后样本不足({len(after_returns)}/{after_total})")

                # P2-Q23-fix(M255): 显著性检验（n=节前/后5交易日×10年≈50，自助法）
                sig_before = _bootstrap_test(before_returns) if before_returns else None
                sig_after = _bootstrap_test(after_returns) if after_returns else None

                effects.append({
                    "holiday": name,
                    "years": count,
                    "before_n": len(before_returns),
                    "after_n": len(after_returns),
                    "before_avg": round(float(np.mean(before_returns)), 4) if before_returns else 0.0,
                    "before_win_rate": round(float(np.mean(np.array(before_returns) > 0)), 4) if before_returns else 0.0,
                    "after_avg": round(float(np.mean(after_returns)), 4) if after_returns else 0.0,
                    "after_win_rate": round(float(np.mean(np.array(after_returns) > 0)), 4) if after_returns else 0.0,
                    "before_p_value": sig_before["p_value"] if sig_before else None,
                    "before_significant": sig_before["significant"] if sig_before else None,
                    "after_p_value": sig_after["p_value"] if sig_after else None,
                    "after_significant": sig_after["significant"] if sig_after else None,
                    "note": "; ".join(note_parts) if note_parts else None,
                })

            result["holiday_effects"] = effects

            # 总结（含样本量与样本不足标注）
            summaries = []
            for e in effects:
                b_dir = "↑" if e["before_avg"] > 0 else "↓"
                a_dir = "↑" if e["after_avg"] > 0 else "↓"
                # P2-Q23-fix(M255): summary 注明 p 值（* 表示 p<0.05 显著）
                bp = (f"p={e['before_p_value']:.3f}{'*' if e['before_significant'] else ''}"
                      if e.get("before_p_value") is not None else "样本不足")
                ap = (f"p={e['after_p_value']:.3f}{'*' if e['after_significant'] else ''}"
                      if e.get("after_p_value") is not None else "样本不足")
                text = (
                    f"{e['holiday']}: 节前{b_dir}{abs(e['before_avg']):.2%}"
                    f"(胜率{e['before_win_rate']:.0%},n={e['before_n']},{bp}) "
                    f"节后{a_dir}{abs(e['after_avg']):.2%}"
                    f"(胜率{e['after_win_rate']:.0%},n={e['after_n']},{ap})"
                )
                if e.get("note"):
                    text += f"[{e['note']}]"
                summaries.append(text)
            result["summary"] = " | ".join(summaries)
            result["bonferroni_note"] = _bonferroni_note(len(effects) * 2)

        except Exception as e:
            result["error"] = str(e)

        self.cache["holiday"] = result
        return result


def main() -> None:
    he = HolidayEffect()
    r = he.compute()
    print("═" * 60)
    print("  节日效应")
    print("═" * 60)
    for h in r.get("holiday_effects", []):
        print(f"\n  {h['holiday']} (近{h['years']}年):")
        print(f"    节前5日均涨: {h['before_avg']:>+7.2%}  胜率: {h['before_win_rate']:.0%}  (n={h['before_n']})")
        print(f"    节后5日均涨: {h['after_avg']:>+7.2%}  胜率: {h['after_win_rate']:.0%}  (n={h['after_n']})")
        if h.get("note"):
            print(f"    ⚠️ {h['note']}")
    print(f"\n  {r.get('summary', '')}")


if __name__ == "__main__":
    main()
