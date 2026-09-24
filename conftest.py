"""Test gates for large runtime assets intentionally excluded from source control."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
ASSET_TESTS = {
    "tests/test_ic_factors_v7.py::TestWarehousePath::test_domain_dir_resolves_to_data_warehouse": ROOT / "data_warehouse/kline",
    "tests/test_market_history_funding.py": ROOT / "data_warehouse/market/market_margin_sh.parquet",
    "tests/test_market_level_factors.py::test_key_factors_have_real_data": ROOT / "data_warehouse/market/macro_monthly__pmi.parquet",
    "tests/test_stock_product_contract.py::test_profile_uses_source_date_not_request_time": ROOT / "data_warehouse/kline/002714.parquet",
    "tests/test_v11_audit_regressions.py::TestResonanceNoLookahead::test_seat_missing_weight_redistribution": ROOT / "data_warehouse/market/theme_cycle.parquet",
    "tests/test_v11_audit_regressions.py::TestScenarioSimilar::test_no_wrong_drop": ROOT / "data_warehouse/market/zt_daily_stats.parquet",
    "tests/test_wave_contracts.py::test_decision_assist_readiness_scope": ROOT / "generated/review_latest.json",
    "quant_system/tests/test_product_contract_v01.py::test_v01_production_contract_resolves": ROOT / "generated/factor_quality_registry.json",
}

def pytest_collection_modifyitems(config, items):
    for item in items:
        for prefix, asset in ASSET_TESTS.items():
            empty_dir = asset.is_dir() and not any(asset.glob("*.parquet"))
            if item.nodeid.startswith(prefix) and (not asset.exists() or empty_dir):
                item.add_marker(pytest.mark.skip(reason=f"recovery asset missing: {asset.relative_to(ROOT)}"))
                break
