"""
weekday_effect — A股周内效应 (V5)

经典研究：周一"周末效应"（一般是跌），周五效应（一般是涨）。
但统计结果随市场环境变化，应定期更新。
"""

from __future__ import annotations

from datetime import timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CST = timezone(timedelta(hours=8))
ROOT = Path(__file__).resolve().parent.parent


class WeekdayEffect:
    """周内效应分析。"""

    def __init__(self) -> None:
        self.cache: dict[str, Any] = {}

    def compute(self, years: int = 10) -> dict[str, Any]:
        """计算周内效应。"""
        cache_key = f"wday_{years}"
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
            df["weekday"] = df["date"].dt.weekday  # 0=Mon, 4=Fri

            day_names = {0: "周一", 1: "周二", 2: "周三", 3: "周四", 4: "周五"}
            stats = []

            for d in range(5):
                day_data = df[df["weekday"] == d]["return"].dropna().values
                if len(day_data) > 0:
                    stats.append({
                        "weekday": day_names[d],
                        "avg_return": round(float(np.mean(day_data)), 6),
                        "win_rate": round(float(np.mean(day_data > 0)), 4),
                        "std": round(float(np.std(day_data, ddof=1)), 4),
                        "median": round(float(np.median(day_data)), 6),
                        "count": int(len(day_data)),
                    })

            result["weekday_stats"] = stats

            if stats:
                best = max(stats, key=lambda s: s["avg_return"])
                worst = min(stats, key=lambda s: s["avg_return"])
                result["best_day"] = best["weekday"]
                result["worst_day"] = worst["weekday"]
                result["summary"] = (
                    f"近{years}年表现最好的是{best['weekday']}(均涨{best['avg_return']:.3%}), "
                    f"最差的是{worst['weekday']}(均跌{worst['avg_return']:.3%})"
                )

        except Exception as e:
            result["error"] = str(e)

        self.cache[cache_key] = result
        return result


def main() -> None:
    we = WeekdayEffect()
    r = we.compute()
    print("═" * 55)
    print("  周内效应 (近10年)")
    print("═" * 55)
    for s in r.get("weekday_stats", []):
        arrow = "📈" if s["avg_return"] > 0 else "📉"
        bar = "█" * int(abs(s["avg_return"]) * 5000)
        print(f"  {s['weekday']} {arrow} {s['avg_return']:>+8.3%}  "
              f"胜率{s['win_rate']:.0%}  {bar}")
    print(f"\n  {r.get('summary', '')}")


if __name__ == "__main__":
    main()
