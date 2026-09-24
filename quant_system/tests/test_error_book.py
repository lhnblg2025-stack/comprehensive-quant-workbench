"""test_error_book — 域Q 强看多暴跌错题本与失效环境反向扫描测试。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quant_system"))

from quant_system.analysis_core import error_book as eb


def _hist(dates=("2026-01-02",), conf=0.8, direction="看多"):
    return [{"date": d, "direction": direction, "confidence": conf} for d in dates]


def _rets(values=(-0.05,)):
    return {f"2026-01-{i + 3:02d}": v for i, v in enumerate(values)}


def test_strong_bullish_crash_recorded():
    r = eb.run_error_book(_hist(), _rets(), date="2026-01-02")
    assert r["n_cases"] == 1
    assert r["cases"][0]["date"] == "2026-01-02"
    assert r["cases"][0]["next_date"] == "2026-01-03"
    assert r["cases"][0]["next_return"] == -0.05


def test_confidence_below_threshold_skipped():
    r = eb.run_error_book(_hist(conf=0.69), _rets(), date="2026-01-02")
    assert r["n_cases"] == 0
    assert r["strong_bullish_count"] == 0


def test_bearish_high_confidence_skipped():
    r = eb.run_error_book(_hist(direction="看空"), _rets(), date="2026-01-02")
    assert r["n_cases"] == 0


def test_return_above_threshold_skipped():
    r = eb.run_error_book(_hist(), _rets(values=(-0.029,)), date="2026-01-02")
    assert r["n_cases"] == 0


def test_next_date_selected_in_order():
    returns = {"2026-01-04": -0.04, "2026-01-03": -0.05}
    r = eb.run_error_book(_hist(), returns, date="2026-01-02")
    assert r["n_cases"] == 1
    assert r["cases"][0]["next_date"] == "2026-01-03"
    assert r["cases"][0]["next_return"] == -0.05


def test_missing_next_return_not_recorded():
    r = eb.run_error_book([{"date": "2026-01-05", "direction": "看多", "confidence": 0.8}],
                          {"2026-01-04": -0.05}, date="2026-01-05")
    assert r["degraded"] is False
    assert r["n_cases"] == 0


def test_environment_features_from_lookup():
    hist = [{"date": "2026-01-02", "direction": "看多", "confidence": 0.8}]
    env = {"2026-01-02": {"涨跌家数比": 0.4, "北向方向": "流出"}}
    r = eb.run_error_book(hist, _rets(), environment=env, date="2026-01-02")
    assert r["cases"][0]["features"]["breadth_ratio"] == 0.4
    assert r["cases"][0]["features"]["northbound_direction"] == "流出"


def _three_cases():
    dates = ["2026-01-02", "2026-01-03", "2026-01-04"]
    returns = {"2026-01-03": -0.04, "2026-01-04": -0.05, "2026-01-05": -0.04}
    history = [
        {"date": dates[0], "direction": "看多", "confidence": 0.8,
         "features": {"涨跌家数比": 0.4, "北向方向": "流出"}},
        {"date": dates[1], "direction": "看多", "confidence": 0.9,
         "features": {"涨跌家数比": 0.4, "北向方向": "流出"}},
        {"date": dates[2], "direction": "看多", "confidence": 0.85,
         "features": {"涨跌家数比": 1.2, "北向方向": "流入"}},
    ]
    return history, returns


def test_feature_rule_generated_above_threshold():
    history, returns = _three_cases()
    r = eb.run_error_book(history, returns, date="2026-01-04", rule_threshold=0.6)
    north_rules = [rule for rule in r["rules"]
                   if any(c["feature"] == "northbound_direction" and c["value"] == "流出"
                          for c in rule["conditions"])]
    assert north_rules, "北向方向=流出 占比 2/3 > 0.6，应生成规则"
    assert north_rules[0]["ratio"] == pytest.approx(2 / 3)


def test_below_threshold_feature_has_no_rule():
    history, returns = _three_cases()
    r = eb.run_error_book(history, returns, date="2026-01-04", rule_threshold=0.6)
    inflow_rules = [rule for rule in r["rules"]
                    if any(c["feature"] == "northbound_direction" and c["value"] == "流入"
                           for c in rule["conditions"])]
    assert inflow_rules == []


def test_pairwise_rule_statement():
    history = [
        {"date": "2026-01-02", "direction": "看多", "confidence": 0.8,
         "features": {"涨跌家数比": 0.4, "炸板率": 0.5}},
        {"date": "2026-01-03", "direction": "看多", "confidence": 0.9,
         "features": {"涨跌家数比": 0.4, "炸板率": 0.5}},
    ]
    returns = {"2026-01-03": -0.05, "2026-01-04": -0.04}
    r = eb.run_error_book(history, returns, date="2026-01-03", rule_threshold=0.6)
    pair = next((rule for rule in r["rules"]
                 if {c["feature"] for c in rule["conditions"]} == {"breadth_ratio", "blast_rate"}),
                None)
    assert pair is not None
    assert "且" in pair["statement"]
    assert "失效概率 100.0%" in pair["statement"]


def test_output_structure():
    r = eb.run_error_book(_hist(), _rets(), date="2026-01-02")
    assert r["schema"] == "error_book/v1"
    assert r["degraded"] is False
    assert isinstance(r["cases"], list)
    assert isinstance(r["rules"], list)
    assert r["strong_bullish_count"] == 1


def test_json_and_markdown_written(tmp_path):
    r = eb.run_error_book(_hist(), _rets(), date="2026-01-02", out_dir=tmp_path)
    day = "20260102"
    json_path = tmp_path / "error_book" / day / "error_book_2026-01-02.json"
    md_path = tmp_path / "error_book" / day / "error_book_2026-01-02.md"
    assert json_path.exists() and md_path.exists()
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed["n_cases"] == 1
    assert "错题本" in md_path.read_text(encoding="utf-8")


def test_degraded_no_history():
    r = eb.run_error_book([], {"2026-01-03": -0.05}, date="2026-01-02")
    assert r["degraded"] is True
    assert r["reason"] == "no_history_data"
    assert r["cases"] == []


def test_degraded_no_returns():
    r = eb.run_error_book(_hist(), [], date="2026-01-02")
    assert r["degraded"] is True
    assert r["reason"] == "no_return_data"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
