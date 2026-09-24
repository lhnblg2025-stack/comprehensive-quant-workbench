from __future__ import annotations

import pandas as pd


def test_northbound_holdings_keeps_ownership_fields_and_filters_invalid(monkeypatch):
    import quant_system.north_flow as nf

    frame = pd.DataFrame([
        {"代码": "600519", "名称": "贵州茅台", "持股数量": 1000, "持股市值": 200000, "持股数量占A股百分比": 2.5, "持股市值变化-5日": 1200, "日期": "2026-08-25"},
        {"代码": "000001", "名称": "坏数据", "持股数量": -1, "持股市值": 100, "持股数量占A股百分比": 1, "日期": "2026-08-25"},
    ])
    class FakeAk:
        @staticmethod
        def stock_hsgt_hold_stock_em(market, indicator):
            assert market == "北向"
            assert indicator == "5日排行"
            return frame
    monkeypatch.setitem(__import__("sys").modules, "akshare", FakeAk)
    result = nf.fetch_north_holdings(top_n=20)
    assert result["data_contract"] == "northbound-holdings.v1"
    assert result["ok"] is True
    assert len(result["rows"]) == 1
    row = result["rows"][0]
    assert row["symbol"] == "600519"
    assert row["holding_pct"] == 2.5
    assert row["holding_change_5d"] == 1200
    assert "day_net" not in row
