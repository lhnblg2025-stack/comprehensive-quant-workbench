"""Market breadth prediction helpers."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class BreadthSignal:
    rise_ratio: float = 0.5
    limit_up: int = 0
    limit_down: int = 0
    breadth_state: str = "neutral"
    score: float = 0.0
    detail: dict = field(default_factory=dict)


def detect_breadth(activity: dict | pd.DataFrame | None) -> BreadthSignal:
    if activity is None:
        return BreadthSignal()
    if isinstance(activity, pd.DataFrame):
        total = len(activity)
        pct_col = "pct_chg" if "pct_chg" in activity.columns else "涨跌幅" if "涨跌幅" in activity.columns else None
        if pct_col is None or total == 0:
            return BreadthSignal()
        pct = activity[pct_col].astype(float)
        # P2-5：涨跌停阈值分板块——主板 10%（ST 5% 无法从涨跌幅列区分，忽略）、
        # 创业板/科创板 20%、北交所 30%。有代码列时按代码逐行判断；
        # 无代码列时退化为 9.8 单阈值（仅主板口径，历史行为）。
        code_col = next((c for c in ("代码", "code", "股票代码") if c in activity.columns), None)
        if code_col is not None:
            def _is_limit(row) -> tuple[bool, bool]:
                c = str(row[code_col]).zfill(6)
                if c.startswith(("300", "301", "688", "689")):
                    thr = 19.8
                elif c.startswith(("4", "8", "920")):
                    thr = 29.8
                else:
                    thr = 9.8
                return bool(row[pct_col] >= thr), bool(row[pct_col] <= -thr)
            flags = activity.apply(_is_limit, axis=1)
            limit_up = int(flags.str[0].sum())
            limit_down = int(flags.str[1].sum())
        else:
            limit_up = int((pct >= 9.8).sum())
            limit_down = int((pct <= -9.8).sum())
        rise = int((pct > 0).sum())
    else:
        total = int(activity.get("total", 0))
        rise = int(activity.get("rise", activity.get("up", 0)))
        limit_up = int(activity.get("limit_up", 0))
        limit_down = int(activity.get("limit_down", 0))
    ratio = rise / total if total > 0 else 0.5
    score = (ratio - 0.5) * 2 + min(limit_up / 100, 0.5) - min(limit_down / 100, 0.5)
    if ratio >= 0.65 and limit_up > limit_down:
        state = "strong"
    elif ratio <= 0.35 or limit_down > limit_up * 2:
        state = "weak"
    else:
        state = "neutral"
    return BreadthSignal(round(ratio, 3), limit_up, limit_down, state, round(score, 3), {"total": total, "rise": rise})


def breadth_probability(signal: BreadthSignal) -> float:
    base = 0.5 + 0.18 * signal.score
    if signal.breadth_state == "strong":
        base += 0.05
    elif signal.breadth_state == "weak":
        base -= 0.05
    return round(max(0.05, min(0.95, base)), 3)
