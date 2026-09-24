from __future__ import annotations

import pandas as pd

from quant_system.factor_backtest_runner import Config, run


def test_weekly_and_monthly_rebalance_are_supported(tmp_path):
    rows = []
    dates = pd.date_range("2020-01-01", periods=80, freq="B")
    for date in dates:
        for stock in range(12):
            rows.append({"date": date, "code": f"{stock:06d}", "close": 100 + stock + dates.get_loc(date), "mom20": stock})
    pd.DataFrame(rows).to_parquet(tmp_path / "panel.parquet")
    for rebalance in ("weekly", "monthly"):
        report = run(Config(str(tmp_path), str(tmp_path / rebalance), ("mom20",), min_stocks=10, rebalance=rebalance))
        assert report["results"][0]["long_only"]["full"]["observations"] > 0
