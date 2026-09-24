from __future__ import annotations

import pandas as pd

from quant_system.full_chain import FullChainConfig, run_full_chain


def test_full_chain_fail_closed_but_writes_all_research_artifacts(tmp_path):
    rows = []
    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    for date_index, date in enumerate(dates):
        for stock in range(12):
            close = 100 + date_index + stock * 0.1
            rows.append({"date": date, "code": f"{stock:06d}", "open": close, "high": close + 1, "low": close - 1, "close": close, "mom20": stock})
    source = tmp_path / "panel.parquet"
    pd.DataFrame(rows).to_parquet(source)
    output = tmp_path / "out"
    result = run_full_chain(FullChainConfig(str(source), str(output), ("mom20",), top_n=3, rebalance="weekly"))
    assert result["quality_gate"] == "HOLD"
    assert result["execution"]["status"] == "BLOCKED"
    assert result["store_rows"] > 0
    for name in ("candidate_library.parquet", "candidate_performance.parquet", "factor_diagnostics.json", "paper_plan.json", "candidate_store.db", "full_chain_manifest.json"):
        assert (output / name).exists()
