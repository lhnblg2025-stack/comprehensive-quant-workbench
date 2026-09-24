from __future__ import annotations

import pandas as pd

from quant_system.trade_state_contract import validate_trade_state


def _row(source: str = "vendor:daily-trade-state:v1") -> dict:
    return {
        "code": "000001", "date": "2024-01-02", "suspended": False,
        "suspension_reason": "", "st_flag": False,
        "limit_up_price": 11.0, "limit_down_price": 9.0,
        "limit_up_locked": False, "limit_down_locked": False,
        "source_document_id": source, "source_as_of": "2024-01-03",
    }


def test_authoritative_trade_state_schema_passes():
    assert validate_trade_state(pd.DataFrame([_row()]))["status"] == "PASS"


def test_derived_or_proxy_trade_state_is_rejected():
    result = validate_trade_state(pd.DataFrame([_row("derived_ohlcv_candidate_only")]))
    assert result["status"] == "BLOCK"
    assert "non_authoritative_source" in result["errors"]
