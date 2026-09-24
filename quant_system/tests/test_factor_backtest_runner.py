from pathlib import Path

import pandas as pd

from quant_system.factor_backtest_runner import Config, run


def test_factor_backtest_emits_oos_cost_and_benchmark_metrics(tmp_path: Path):
    rows = []
    for day in range(50):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day)
        for stock in range(40):
            rows.append({
                "code": f"{stock:06d}",
                "date": date.strftime("%Y%m%d"),
                "close": 100 + day + stock * 0.01,
                "mom20": stock + day / 100,
            })
    pd.DataFrame(rows).to_parquet(tmp_path / "snapshot.parquet")
    report = run(Config(str(tmp_path), str(tmp_path / "out"), ("mom20",), min_stocks=30))
    result = report["results"][0]
    assert result["long_only"]["oos"]["observations"] > 0
    assert "cost_x2_total_return" in result
    assert "vs_benchmark" in result
    assert "information_ratio" in result["vs_benchmark"]
    assert (tmp_path / "out" / "factor_backtest_report.json").exists()
