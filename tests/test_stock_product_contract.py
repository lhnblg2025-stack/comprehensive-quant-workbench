import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant_web.stock_analysis import build_stock_profile, _score_profile


def test_missing_profile_is_not_neutral_advice():
    profile = build_stock_profile("999999")
    assert profile["available"] is False
    assert profile["data_status"] == "missing"
    assert profile["score"] == {}
    assert profile["ai_advice"] == {}


def test_partial_score_renormalizes_available_dimensions():
    score = _score_profile({"trend": {"above_ma20": True}, "liquidity": {"vol_ratio_5_20": 1.2}})
    assert score["total"] is not None
    assert score["coverage"]["available_dimensions"] == 2
    assert set(score["unavailable_dimensions"]) == {"valuation", "financial", "events"}
    assert abs(sum(score["weights"].values()) - 1) < 0.001


def test_profile_uses_source_date_not_request_time():
    profile = build_stock_profile("002714")
    assert profile["generated_at"]
    assert profile["data_as_of"] == profile["source_dates"]["quote"]
    assert profile["generated_at"][:10] >= profile["data_as_of"]
