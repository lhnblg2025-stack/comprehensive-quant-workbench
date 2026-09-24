from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_system.historical_panel_builder import build_hfq_panel


def test_hfq_signal_and_raw_execution_tracks_are_separate(tmp_path: Path, monkeypatch):
    def fake_fetch(symbol, start, end, adjust):
        dates = pd.bdate_range("2024-01-01", periods=25)
        base = 500.0 if adjust == "hfq" else 10.0
        close = pd.Series(range(25), dtype=float) + base
        return pd.DataFrame({"date": dates, "open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0, "amount": 10000.0})

    monkeypatch.setattr("quant_system.sources_all.fetch_daily_unified", fake_fetch)
    result = build_hfq_panel(["000001"], tmp_path, start="2024-01-01", end="2024-02-10")
    manifest = json.loads((tmp_path / "panel_manifest.json").read_text())
    panel = pd.read_parquet(tmp_path / "hfq_panel.parquet")
    assert result["signal_adjust"] == "hfq"
    assert result["execution_adjust"] == "raw"
    assert manifest["sources"][0]["signal_adjust"] == "hfq"
    assert manifest["sources"][0]["execution_adjust"] == "raw"
    assert panel.loc[0, "close"] == panel.loc[0, "hfq_close"]
    assert panel.loc[0, "hfq_close"] != panel.loc[0, "raw_close"]
