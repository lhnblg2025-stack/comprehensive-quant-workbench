from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pandas as pd

from quant_platform import openclaw_api
from quant_system import data_pipeline
from quant_web import stock_analysis


def test_money_flow_standardizer_rejects_date_only_frame():
    frame = pd.DataFrame({"日期": ["2026-08-27", "2026-08-26"], "主力净流入-净额": [None, None]})
    assert data_pipeline._standardize_stock_money_flow(frame, "002714", 60) is None


def test_money_flow_standardizer_keeps_finite_main_net():
    frame = pd.DataFrame({"日期": ["2026-08-27", "2026-08-26"], "主力净流入-净额": [1200000, None]})
    result = data_pipeline._standardize_stock_money_flow(frame, "002714", 60)
    assert result is not None
    assert result["main_net"].notna().sum() == 1


def test_lhb_recent_filters_real_shangbang_date():
    frame = pd.DataFrame({
        "代码": ["600519", "000001"],
        "名称": ["贵州茅台", "平安银行"],
        "龙虎榜净买额": [100.0, 50.0],
        "上榜日": ["2026-08-27", "2026-07-01"],
    })
    with patch.object(openclaw_api._ds, "get_dataset", return_value=frame):
        result = openclaw_api.lhb_recent(5)
    assert result["status"] == "available"
    assert result["date_column"] == "上榜日"
    assert result["as_of"] == "2026-08-27"
    assert result["total_rows"] == 1
    assert result["top_net_buy"][0]["代码"] == "600519"


def test_openclaw_stock_overview_normalizes_numeric_code_and_reports_status(monkeypatch, tmp_path):
    class FakeStore:
        def warehouse_root(self):
            return tmp_path

        def get(self, code, days=120):
            return pd.DataFrame({
                "date": pd.to_datetime(["2026-08-26"]),
                "close": [40.0], "pct_chg": [1.0],
            })

        def get_dataset(self, name, files=None):
            if name == "sw_industry_map":
                return pd.DataFrame({"code": ["002714.0"], "industry": ["农林牧渔"]})
            return pd.DataFrame()

    monkeypatch.setattr(openclaw_api, "_ds", FakeStore())
    (tmp_path / "valuation").mkdir()
    pd.DataFrame({
        "date": ["2026-08-26"], "peTTM": [10.0], "pbMRQ": [2.0],
    }).to_parquet(tmp_path / "valuation" / "002714.parquet", index=False)
    (tmp_path / "financial").mkdir()
    pd.DataFrame({"日期": ["2026-06-30"], "净利润增长率(%)": [5.0]}).to_parquet(
        tmp_path / "financial" / "002714.parquet", index=False
    )

    result = openclaw_api.stock_overview("002714.0")
    assert result["code"] == "002714"
    assert result["valuation"]["pe_ttm"] == 10.0
    assert result["valuation"]["pb"] == 2.0
    assert result["industry"] == "农林牧渔"
    assert result["source_status"]["valuation"]["status"] == "available"
    assert result["source_status"]["financial"]["status"] == "available"


def test_float_market_value_detail_exposes_historical_share_date():
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2026-08-26", "2026-08-27"]),
        "close": [40.0, 41.0],
        "outstanding_share": [3_000_000_000, None],
    })
    value, as_of, status = stock_analysis._derive_float_mv_detail(frame)
    assert value == 1200.0
    assert as_of == "2026-08-26"
    assert status == "available"
