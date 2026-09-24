"""
earnings_season — A股财报季效应 (V5)

财报密集披露期对市场的影响：
  1月：年报预告
  4月：年报+一季报
  7月：中报
  10月：三季报

财报季 vs 非财报季的市场表现差异。
"""

from __future__ import annotations

from datetime import timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CST = timezone(timedelta(hours=8))
ROOT = Path(__file__).resolve().parent.parent


class EarningsSeason:
    """财报季效应分析。"""

    def __init__(self) -> None:
        self.cache: dict[str, Any] = {}

    def compute(self, years: int = 10) -> dict[str, Any]:
        """计算财报季效应。"""
        cache_key = f"earnings_{years}"
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

            df["month"] = df["date"].dt.month
            df["year"] = df["date"].dt.year

            # 月收益
            monthly = df.groupby(["year", "month"])["return"].apply(
                lambda x: (1 + x).prod() - 1
            ).reset_index()

            # 财报季(1,4,7,10月) vs 非财报季
            earnings_months = [1, 4, 7, 10]
            monthly["is_earnings"] = monthly["month"].isin(earnings_months)

            earnings_returns = monthly[monthly["is_earnings"]]["return"].dropna().values
            non_earnings_returns = monthly[~monthly["is_earnings"]]["return"].dropna().values

            if len(earnings_returns) > 0 and len(non_earnings_returns) > 0:
                result = {
                    "earnings_season": {
                        "avg_return": round(float(np.mean(earnings_returns)), 4),
                        "win_rate": round(float(np.mean(earnings_returns > 0)), 4),
                        "std": round(float(np.std(earnings_returns, ddof=1)), 4),
                        "count": int(len(earnings_returns)),
                    },
                    "non_earnings_season": {
                        "avg_return": round(float(np.mean(non_earnings_returns)), 4),
                        "win_rate": round(float(np.mean(non_earnings_returns > 0)), 4),
                        "std": round(float(np.std(non_earnings_returns, ddof=1)), 4),
                        "count": int(len(non_earnings_returns)),
                    },
                    "earnings_detail": [],
                }

                # 各财报月详细
                for m in earnings_months:
                    m_data = monthly[monthly["month"] == m]["return"].dropna().values
                    if len(m_data) > 0:
                        month_names = {1: "1月(年报预告)", 4: "4月(年报一季报)",
                                      7: "7月(中报)", 10: "10月(三季报)"}
                        result["earnings_detail"].append({
                            "month": month_names.get(m, f"{m}月"),
                            "avg_return": round(float(np.mean(m_data)), 4),
                            "win_rate": round(float(np.mean(m_data > 0)), 4),
                            "count": int(len(m_data)),
                        })

        except Exception as e:
            result["error"] = str(e)

        self.cache[cache_key] = result
        return result


def main() -> None:
    es = EarningsSeason()
    r = es.compute()
    print("═" * 55)
    print("  财报季 vs 非财报季")
    print("═" * 55)
    es_data = r.get("earnings_season", {})
    nes = r.get("non_earnings_season", {})
    if es_data:
        print(f"  📊 财报季(1/4/7/10月):")
        print(f"     均涨: {es_data.get('avg_return', 0):>+7.2%}  "
              f"胜率: {es_data.get('win_rate', 0):.0%}  "
              f"波动: {es_data.get('std', 0):.2%}")
    if nes:
        print(f"  📊 非财报季:")
        print(f"     均涨: {nes.get('avg_return', 0):>+7.2%}  "
              f"胜率: {nes.get('win_rate', 0):.0%}  "
              f"波动: {nes.get('std', 0):.2%}")
    print()
    for d in r.get("earnings_detail", []):
        print(f"  {d['month']}: {d['avg_return']:>+7.2%} 胜率{d['win_rate']:.0%}")


if __name__ == "__main__":
    main()
