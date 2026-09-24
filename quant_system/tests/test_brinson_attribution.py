"""test_brinson_attribution — 域Q Brinson 四维归因回归测试。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quant_system"))

from quant_system.analysis_core import brinson_attribution as ba


def known_portfolio(actual_return: float = 0.10, date: str = "2026-01-02") -> list[dict]:
    return [{
        "date": date,
        "actual_return": actual_return,
        "positions": [
            {"industry": "A", "weight": 0.8, "return": 0.15},
            {"industry": "B", "weight": 0.4, "return": 0.01},
        ],
    }]


def known_benchmark(date: str = "2026-01-02") -> list[dict]:
    return [{
        "date": date,
        "industries": [
            {"industry": "A", "weight": 0.5, "return": 0.10},
            {"industry": "B", "weight": 0.5, "return": -0.02},
        ],
    }]


def test_known_decomposition_values():
    r = ba.compute_attribution(known_portfolio(), known_benchmark(), date="2026-01-02")
    d = r["daily"][0]["contributions"]
    assert d["timing"] == pytest.approx(0.008)
    assert d["allocation"] == pytest.approx(0.024)
    assert d["selection"] == pytest.approx(0.052)
    assert d["friction"] == pytest.approx(-0.024)
    assert r["daily"][0]["theoretical_return"] == pytest.approx(0.124)
    assert r["daily"][0]["benchmark_return"] == pytest.approx(0.04)


def test_friction_is_actual_minus_theoretical():
    r = ba.compute_attribution(known_portfolio(actual_return=0.124), known_benchmark(),
                               date="2026-01-02")
    d = r["daily"][0]
    assert d["actual_return"] == pytest.approx(0.124)
    assert d["theoretical_return"] == pytest.approx(0.124)
    assert d["contributions"]["friction"] == pytest.approx(0.0)


def test_lock_allocator_after_three_negative_timing_days():
    days = ["2026-01-02", "2026-01-03", "2026-01-04"]
    portfolio = [{
        "date": d, "actual_return": 0.01,
        "positions": [{"industry": "A", "weight": 0.5, "return": 0.02}],
    } for d in days]
    benchmark = [{
        "date": d,
        "industries": [{"industry": "A", "weight": 1.0, "return": 0.02}],
    } for d in days]
    r = ba.compute_attribution(portfolio, benchmark, date="2026-01-04")
    assert r["daily"][-1]["timing_negative_streak"] == 3
    assert r["lock_allocator"] is True
    assert "锁定" in r["lock_allocator_advice"]


def test_lock_allocator_not_triggered_positive_timing():
    portfolio = [{
        "date": "2026-01-02", "actual_return": 0.03,
        "positions": [{"industry": "A", "weight": 1.5, "return": 0.02}],
    }]
    benchmark = [{
        "date": "2026-01-02",
        "industries": [{"industry": "A", "weight": 1.0, "return": 0.02}],
    }]
    r = ba.compute_attribution(portfolio, benchmark, date="2026-01-02")
    assert r["daily"][0]["contributions"]["timing"] > 0
    assert r["lock_allocator"] is False


def test_output_structure_keys():
    r = ba.compute_attribution(known_portfolio(), known_benchmark(), date="2026-01-02")
    assert r["schema"] == "brinson_attribution/v1"
    assert set(r["cumulative"]) == {"timing", "allocation", "selection", "friction"}
    assert isinstance(r["daily"], list)
    assert "lock_allocator" in r
    assert r["degraded"] is False


def test_degraded_no_portfolio():
    r = ba.compute_attribution([], known_benchmark(), date="2026-01-02")
    assert r["degraded"] is True
    assert r["reason"] == "no_portfolio_data"
    assert r["daily"] == []


def test_degraded_no_benchmark():
    r = ba.compute_attribution(known_portfolio(), [], date="2026-01-02")
    assert r["degraded"] is True
    assert r["reason"] == "no_benchmark_data"


def test_degraded_no_matched_dates():
    r = ba.compute_attribution(known_portfolio(date="2026-01-03"),
                               known_benchmark(date="2026-01-02"), date="2026-01-03")
    assert r["degraded"] is True
    assert r["reason"] == "no_matched_attribution_dates"


def test_industry_map_fallback():
    portfolio = [{
        "date": "2026-01-02", "actual_return": 0.01,
        "positions": [{"code": "600001", "weight": 0.5, "return": 0.02}],
    }]
    benchmark = [{
        "date": "2026-01-02",
        "industries": [{"industry": "A", "weight": 1.0, "return": 0.02}],
    }]
    r = ba.compute_attribution(portfolio, benchmark,
                               industry_map={"600001": "A"}, date="2026-01-02")
    assert r["degraded"] is False
    assert r["daily"][0]["contributions"]["selection"] == pytest.approx(0.0)


def test_unknown_industry_aggregation():
    portfolio = [{
        "date": "2026-01-02", "actual_return": 0.01,
        "positions": [{"weight": 0.5, "return": 0.02}],
    }]
    benchmark = [{
        "date": "2026-01-02",
        "industries": [{"industry": "A", "weight": 1.0, "return": 0.02}],
    }]
    r = ba.compute_attribution(portfolio, benchmark, date="2026-01-02")
    assert r["degraded"] is False
    assert len(r["daily"]) == 1


def test_benchmark_overall_return_fallback():
    benchmark = [{
        "date": "2026-01-02",
        "industries": [
            {"industry": "A", "weight": 0.5, "return": 0.10},
            {"industry": "B", "weight": 0.5, "return": -0.02},
        ],
    }]
    r = ba.compute_attribution(known_portfolio(), benchmark, date="2026-01-02")
    assert r["daily"][0]["benchmark_return"] == pytest.approx(0.04)


def test_dataframe_inputs():
    portfolio = pd.DataFrame([{
        "date": "2026-01-02", "industry": "A", "weight": 0.8, "return": 0.15,
        "actual_return": 0.10,
    }, {
        "date": "2026-01-02", "industry": "B", "weight": 0.4, "return": 0.01,
    }])
    benchmark = pd.DataFrame([{
        "date": "2026-01-02", "industry": "A", "weight": 0.5, "return": 0.10,
    }, {
        "date": "2026-01-02", "industry": "B", "weight": 0.5, "return": -0.02,
    }])
    r = ba.compute_attribution(portfolio, benchmark, date="2026-01-02")
    assert len(r["daily"]) == 1
    assert r["daily"][0]["contributions"]["allocation"] == pytest.approx(0.024)


def test_cumulative_series_sums_daily():
    portfolio = known_portfolio() + known_portfolio(actual_return=0.09, date="2026-01-03")
    benchmark = known_benchmark() + known_benchmark(date="2026-01-03")
    r = ba.compute_attribution(portfolio, benchmark, date="2026-01-03")
    timing = sum(d["contributions"]["timing"] for d in r["daily"])
    allocation = sum(d["contributions"]["allocation"] for d in r["daily"])
    selection = sum(d["contributions"]["selection"] for d in r["daily"])
    assert r["cumulative"]["timing"] == pytest.approx(timing)
    assert r["cumulative"]["allocation"] == pytest.approx(allocation)
    assert r["cumulative"]["selection"] == pytest.approx(selection)


def test_json_and_markdown_written(tmp_path):
    r = ba.run_attribution(known_portfolio(), known_benchmark(), date="2026-01-02",
                           out_dir=tmp_path)
    day = "20260102"
    json_path = tmp_path / "attribution" / day / "attribution_2026-01-02.json"
    md_path = tmp_path / "attribution" / day / "attribution_2026-01-02.md"
    assert json_path.exists() and md_path.exists()
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed["date"] == "2026-01-02"
    assert "Brinson 业绩归因" in md_path.read_text(encoding="utf-8")
    assert r["degraded"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
