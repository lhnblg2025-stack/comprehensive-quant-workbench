"""
policy_window — 政策窗口效应 (V5)

关键政策事件前后的市场表现：
  - 两会 (3月): 政府工作报告定调
  - 中央经济工作会议 (12月): 下一年经济定调
  - 政治局会议 (4/7/10/12月): 季度经济分析

数据源: 上证指数日线 + 硬编码近N年政策日期
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
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


# P2-Q23-fix(M255): 多重比较校正说明（两会/中央经济工作 2 类窗口 × 前后各 1 组）。
def _bonferroni_note(n_tests: int, alpha: float = 0.05) -> str:
    return (f"同时检验{n_tests}个假设,Bonferroni校正后单次显著性阈值"
            f"{alpha}/{n_tests}={alpha / n_tests:.4f}")


class PolicyWindow:
    """政策窗口效应分析。"""

    def __init__(self) -> None:
        self.cache: dict[str, Any] = {}

        # 近10年两会开幕日期
        self.lianghui_dates = [
            "2017-03-05", "2018-03-05", "2019-03-05", "2020-05-22",
            "2021-03-05", "2022-03-05", "2023-03-05", "2024-03-05",
            "2025-03-05", "2026-03-05",
        ]

        # 中央经济工作会议（12月中旬）
        self.cewg_dates = [
            "2017-12-18", "2018-12-19", "2019-12-10", "2020-12-16",
            "2021-12-08", "2022-12-15", "2023-12-11", "2024-12-12",
            "2025-12-15", "2026-12-15",
        ]

    def compute(self) -> dict[str, Any]:
        """计算政策窗口效应。"""
        if "policy" in self.cache:
            return self.cache["policy"]

        result: dict[str, Any] = {
            "policy_effects": [],
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
            trade_dates = pd.DatetimeIndex(df["date"])
            today = datetime.now(CST).date()

            windows = {
                "两会(3月)": self.lianghui_dates,
                "中央经济工作(12月)": self.cewg_dates,
            }

            effects = []
            for name, dates in windows.items():
                before_rets = []
                after_rets = []
                before_total = 0
                after_total = 0
                valid_years = 0
                skipped_future = 0

                for d in dates:
                    d_dt = pd.to_datetime(d)

                    # P2-Q23-fix(M260): 未来日期不参与统计。原实现把 2026-12-15
                    # (中央经济工作, 晚于今日) 也计入 valid_years，"近10年"虚报
                    # 为 10 而实际仅 9 个有效年份。先过滤，再按事件日就近取交易日。
                    if d_dt.date() > today:
                        skipped_future += 1
                        continue

                    # P2-Q23-fix(M260): 原 ±10 日历日偏移会因周末/节假日丢样本
                    # （非交易日无 return 可查）。改用交易日偏移：参考日=事件日
                    # 或事件日前最近交易日（searchsorted），前后各取 10 个交易日。
                    pos = int(trade_dates.searchsorted(d_dt, side="right")) - 1
                    if pos < 0:
                        continue
                    ref = trade_dates[pos]
                    # 事件日距最近交易日超 10 天 → 该年行情未覆盖，跳过，禁止
                    # 用错误参考日产生虚假统计（同 holiday_effect H04 处理）。
                    if (d_dt - ref).days > 10:
                        continue
                    valid_years += 1

                    # 前10个交易日（参考日之前，不含参考日）
                    for j in range(max(0, pos - 10), pos):
                        before_total += 1
                        r = df["return"].iloc[j]
                        if not pd.isna(r):
                            before_rets.append(float(r))

                    # 后10个交易日（参考日之后）
                    for j in range(pos + 1, min(len(df), pos + 11)):
                        after_total += 1
                        r = df["return"].iloc[j]
                        if not pd.isna(r):
                            after_rets.append(float(r))

                # P2-Q23-fix(M255): 显著性检验（n=10年×10交易日≈100，自助法）
                sig_before = _bootstrap_test(before_rets) if before_rets else None
                sig_after = _bootstrap_test(after_rets) if after_rets else None

                note_parts: list[str] = []
                if skipped_future:
                    note_parts.append(f"跳过未来年份{skipped_future}")
                if valid_years == 0:
                    note_parts.append("无有效参考年(数据覆盖不足)")
                if before_total and len(before_rets) < before_total:
                    note_parts.append(f"前窗样本不足({len(before_rets)}/{before_total})")
                if after_total and len(after_rets) < after_total:
                    note_parts.append(f"后窗样本不足({len(after_rets)}/{after_total})")

                effects.append({
                    "window": name,
                    "years": valid_years,
                    "before_n": len(before_rets),
                    "after_n": len(after_rets),
                    "before_10d_avg": round(float(np.mean(before_rets)), 4) if before_rets else 0,
                    "before_win_rate": round(float(np.mean(np.array(before_rets) > 0)), 4) if before_rets else 0,
                    "after_10d_avg": round(float(np.mean(after_rets)), 4) if after_rets else 0,
                    "after_win_rate": round(float(np.mean(np.array(after_rets) > 0)), 4) if after_rets else 0,
                    "before_p_value": sig_before["p_value"] if sig_before else None,
                    "before_significant": sig_before["significant"] if sig_before else None,
                    "after_p_value": sig_after["p_value"] if sig_after else None,
                    "after_significant": sig_after["significant"] if sig_after else None,
                    "note": "; ".join(note_parts) if note_parts else None,
                })

            result["policy_effects"] = effects
            summaries = []
            for e in effects:
                b = "↑" if e["before_10d_avg"] > 0 else "↓"
                a = "↑" if e["after_10d_avg"] > 0 else "↓"
                # P2-Q23-fix(M255): summary 注明前后窗 p 值，显著性与否一目了然
                bp = (f"(p={e['before_p_value']:.3f}{'*' if e['before_significant'] else ''})"
                      if e.get("before_p_value") is not None else "(样本不足)")
                ap = (f"(p={e['after_p_value']:.3f}{'*' if e['after_significant'] else ''})"
                      if e.get("after_p_value") is not None else "(样本不足)")
                text = f"{e['window']}: 前{b}{abs(e['before_10d_avg']):.2%}{bp} 后{a}{abs(e['after_10d_avg']):.2%}{ap}"
                if e.get("note"):
                    text += f"[{e['note']}]"
                summaries.append(text)
            result["summary"] = " | ".join(summaries)
            result["bonferroni_note"] = _bonferroni_note(len(effects) * 2)

        except Exception as e:
            result["error"] = str(e)

        self.cache["policy"] = result
        return result


def main() -> None:
    pw = PolicyWindow()
    r = pw.compute()
    print("═" * 55)
    print("  政策窗口效应")
    print("═" * 55)
    for w in r.get("policy_effects", []):
        print(f"\n  {w['window']} (近{w['years']}年):")
        print(f"    前10日均涨: {w['before_10d_avg']:>+7.2%} 胜率{w['before_win_rate']:.0%}")
        print(f"    后10日均涨: {w['after_10d_avg']:>+7.2%} 胜率{w['after_win_rate']:.0%}")
    print(f"\n  {r.get('summary', '')}")


if __name__ == "__main__":
    main()
