"""Test gates for large runtime assets intentionally excluded from source control.

The public release ships source, not runtime data. Tests that need a populated
data warehouse or generated artefacts are skipped with an explicit reason
instead of failing, so a clean clone reports a truthful green/partial result.

Add an entry here whenever a test depends on an asset that is not in the repo.
"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
WAREHOUSE = ROOT / "data_warehouse"
GENERATED = ROOT / "generated"


def _has_parquet(directory: Path) -> bool:
    try:
        return directory.is_dir() and any(directory.rglob("*.parquet"))
    except OSError:
        return False


def _missing(asset: Path) -> bool:
    """True when the required asset is absent (or an empty parquet directory)."""
    if asset.is_dir():
        return not _has_parquet(asset)
    return not asset.exists()


# nodeid prefix -> asset that must exist for the test to run.
ASSET_TESTS = {
    # --- existing gates ---
    "tests/test_ic_factors_v7.py::TestWarehousePath::test_domain_dir_resolves_to_data_warehouse": WAREHOUSE / "kline",
    "tests/test_market_history_funding.py": WAREHOUSE / "market/market_margin_sh.parquet",
    "tests/test_market_level_factors.py::test_key_factors_have_real_data": WAREHOUSE / "market/macro_monthly__pmi.parquet",
    "tests/test_stock_product_contract.py::test_profile_uses_source_date_not_request_time": WAREHOUSE / "kline/002714.parquet",
    "tests/test_v11_audit_regressions.py::TestResonanceNoLookahead::test_seat_missing_weight_redistribution": WAREHOUSE / "market/theme_cycle.parquet",
    "tests/test_v11_audit_regressions.py::TestScenarioSimilar::test_no_wrong_drop": WAREHOUSE / "market/zt_daily_stats.parquet",
    "tests/test_wave_contracts.py::test_decision_assist_readiness_scope": GENERATED / "review_latest.json",
    "quant_system/tests/test_product_contract_v01.py::test_v01_production_contract_resolves": GENERATED / "factor_quality_registry.json",
    # --- gates added for the full declassified release ---
    # Whole-module gates: every test in these files reads the runtime warehouse.
    "tests/test_after_close_extra.py": WAREHOUSE,
    "tests/test_v3_new_modules.py": WAREHOUSE,
    "quant_system/tests/test_research_upgrade.py": WAREHOUSE / "kline_raw",
    "scripts/tests/test_freshness_audit_p22.py": WAREHOUSE / "data_freshness.json",
    # Specific asset-dependent tests.
    "tests/test_v11_audit_regressions.py::TestRegimeStale": WAREHOUSE,
    "tests/test_after_close_extra.py::TestPushReport::test_build_push_text_has_all_sections": GENERATED
    / "decision_snapshot_after_close_2026-08-26.json",
    "quant_web/tests/test_strategy_lab.py::test_stock_universe_is_mainboard_only_and_excludes_st": WAREHOUSE,
    "quant_web/tests/test_strategy_lab.py::test_stock_research_gate_preserves_fixed_mainboard_sample_and_discloses_blocker": WAREHOUSE,
    "quant_web/tests/test_strategy_lab.py::test_etf_universe_and_benchmarks_are_distinct": WAREHOUSE / "events/etf_history.parquet",
}

# nodeid prefix -> (required top-level module, install hint).
# These adapters are optional extras (see requirements.lock.txt); the tests are
# skipped when the dependency is missing instead of reported as broken.
OPTIONAL_DEPENDENCY_TESTS = (
    ("quant_system/tests/test_backtrader_adapter.py", "backtrader", "backtrader"),
    ("quant_system/tests/test_backtrader_execution_events.py", "backtrader", "backtrader"),
    ("quant_system/tests/test_backtest_crosscheck.py::test_bounded_crosscheck", "backtrader", "backtrader"),
    ("quant_system/tests/test_integrations.py::TestAlphalensAdapter", "alphalens", "alphalens-reloaded"),
    ("quant_system/tests/test_integrations.py::TestRiskfolioAdapter", "riskfolio", "Riskfolio-Lib"),
)


def pytest_collection_modifyitems(config, items):
    for item in items:
        for prefix, asset in ASSET_TESTS.items():
            if item.nodeid.startswith(prefix) and _missing(asset):
                item.add_marker(
                    pytest.mark.skip(reason=f"runtime asset missing: {asset.relative_to(ROOT)}")
                )
                break

    # Optional third-party adapters: the code degrades gracefully when the
    # package is absent, so the matching tests skip rather than error.
    for prefix, module, hint in OPTIONAL_DEPENDENCY_TESTS:
        if importlib.util.find_spec(module) is not None:
            continue
        for item in items:
            if item.nodeid.startswith(prefix):
                item.add_marker(
                    pytest.mark.skip(reason=f"optional dependency not installed: {hint}")
                )
