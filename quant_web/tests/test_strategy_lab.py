from __future__ import annotations

import pytest

from quant_web.handlers.strategy_lab import _attach_benchmark, _load_asset_panel, _research_data_gate, _research_years, _validate_source, default_source
from quant_web.handlers.strategy_templates import TEMPLATES


def test_default_strategy_contract_is_valid():
    _validate_source(default_source())


def test_strategy_requires_build_targets():
    with pytest.raises(ValueError, match="strategy_requires_build_targets"):
        _validate_source("def other(panel, params):\n    return {}\n")


def test_strategy_rejects_unsafe_imports_and_dynamic_execution():
    with pytest.raises(ValueError, match="strategy_import_not_allowed"):
        _validate_source("import os\ndef build_targets(panel, params):\n    return {}\n")
    with pytest.raises(ValueError, match="strategy_dynamic_execution_not_allowed"):
        _validate_source("def build_targets(panel, params):\n    return eval('1')\n")


def test_all_builtin_templates_are_valid():
    # Public release scope: exactly the three selected strategies.
    assert set(TEMPLATES) == {"rsi_reversal", "low_volatility", "momentum_12_1"}
    for template in TEMPLATES.values():
        _validate_source(template["source"])


def test_stock_universe_is_mainboard_only_and_excludes_st():
    panel, _, _ = _load_asset_panel("stock")
    assert panel["code"].str.startswith(("00", "60")).all()
    assert panel["code"].nunique() > 100


def test_research_modes_reject_two_year_fast_path():
    assert _research_years("research_5y") == 5
    assert _research_years("research_10y") == 10
    with pytest.raises(ValueError, match="research_mode"):
        _research_years("quick")


def test_stock_research_gate_preserves_fixed_mainboard_sample_and_discloses_blocker():
    panel, state, path = _load_asset_panel("stock")
    _, _, gate = _research_data_gate(panel, state, path, years=5)
    assert gate["symbols"] == panel["code"].nunique() == 300
    assert gate["coverage"]["minimum"] >= 0.8
    assert gate["classification"] == "research_only"
    assert gate["research_executable"] is True
    assert gate["execution_mode"] == "research_proxy"
    assert gate["production_status"] == "DATA_BLOCKED"
    assert "non_authoritative_historical_trade_state" in gate["production_blockers"]
    assert {"proxy_limit_rule", "unknown_st_state", "partial_lifecycle"}.issubset(gate["proxy_limitations"])


def test_etf_universe_and_benchmarks_are_distinct():
    import pandas as pd

    panel, _, _ = _load_asset_panel("etf")
    assert panel["code"].nunique() >= 10
    equity = pd.DataFrame({"date": pd.date_range("2025-01-02", periods=5, freq="B"), "total_value": [100, 101, 100, 103, 104]})
    stock_curve, stock_benchmark = _attach_benchmark(equity, "stock")
    etf_curve, etf_benchmark = _attach_benchmark(equity, "etf")
    assert stock_benchmark["components"] == ["上证指数", "深证成指"]
    assert etf_benchmark["components"] == ["沪深300"]
    assert {"strategy_nav", "benchmark_nav", "excess_nav", "drawdown"}.issubset(stock_curve.columns)
    assert len(etf_curve) == len(equity)
