from __future__ import annotations

from quant_system.product_contract import PRODUCT_VERSION, validate_production_contract


def test_v01_production_contract_resolves():
    result = validate_production_contract()
    assert PRODUCT_VERSION == "v0.1"
    assert result["status"] == "PASS", result["errors"]
    assert result["components"]["execution_backtester"]["attribute"] == "BacktestEngine"
    assert "quant_system.factor_zoo" in result["research_only"]
