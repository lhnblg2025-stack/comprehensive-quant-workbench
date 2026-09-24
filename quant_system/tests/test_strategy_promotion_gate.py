from __future__ import annotations

from quant_system.strategy_promotion_gate import evaluate_candidate


def test_proxy_result_cannot_be_promoted_even_with_strong_metrics():
    result = evaluate_candidate({
        "return_basis": "raw_next_open_to_following_open",
        "oos_windows": 3,
        "oos_excess_annual": .1,
        "cost_x2_oos_total_return": .1,
        "consistent_quantiles": 2,
        "industry_cap_pass": True,
        "execution_reconciled": True,
    }, data_quality="deadline_pit_proxy_and_current_industry_proxy")
    assert result["status"] == "research_only"
    assert "data_quality_not_production_pit" in result["failures"]


def test_only_fully_validated_candidate_can_promote():
    result = evaluate_candidate({
        "return_basis": "raw_next_open_to_following_open",
        "oos_windows": 3,
        "oos_excess_annual": .1,
        "cost_x2_oos_total_return": .1,
        "consistent_quantiles": 2,
        "industry_cap_pass": True,
        "execution_reconciled": True,
    }, data_quality="actual_pubdate_and_historical_industry")
    assert result == {"status": "candidate", "failures": [], "promotable": True}
