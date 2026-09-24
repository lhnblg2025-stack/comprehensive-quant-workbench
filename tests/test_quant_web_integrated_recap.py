from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import quant_web.server as server
import quant_web.stock_analysis as stock_analysis
from quant_web.decision_contract import build_decision_gate


def _stub_handler():
    handler = server.QuantHandler.__new__(server.QuantHandler)
    handler._sent = []
    handler._send_json = lambda payload, status=200: handler._sent.append((status, payload))
    return handler


def test_market_recap_returns_single_auditable_view_model(monkeypatch, tmp_path):
    generated = tmp_path / "generated"
    generated.mkdir()
    snapshot = {
        "as_of": "2026-08-26",
        "generated_at": "2026-08-27T10:00:00+08:00",
        "market": {
            "temperature": 41,
            "emotion_stage": "修复",
            "force_index": 25,
            "breadth": 0.65,
            "zt": {"zt_cnt": 72, "dt_cnt": 5, "zb_cnt": 25, "max_board": 5,
                   "ladder_json": json.dumps({"1": 63, "5": 1})},
            "risk_flags": ["资金合力偏弱"],
        },
        "mainlines": [{"name": "通用设备", "score": 56, "zt": 4, "max_board": 2, "level": "主线"}],
        "opportunities": [{"code": "002714", "name": "牧原股份", "short_term_score": 47,
                           "trade_label": "观察", "trade_reasons": ["未匹配主线"]}],
    }
    (generated / "decision_snapshot_after_close_2026-08-26.json").write_text(
        json.dumps(snapshot, ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setattr(server, "ROOT", tmp_path)
    handler = _stub_handler()
    handler._handle_market_recap({"date": ["2026-08-26"]})
    status, payload = handler._sent[-1]
    assert status == 200
    assert payload["data_status"] == "available"
    assert payload["market"]["seal_rate"] == 74.2
    assert payload["mainlines"][0]["name"] == "通用设备"
    assert payload["candidates"][0]["code"] == "002714"


def test_financial_summary_discards_blank_rows(monkeypatch):
    frame = pd.DataFrame([
        {"日期": "2026-03-31", "净资产收益率(%)": -1.39},
        {"日期": None, "净资产收益率(%)": None},
    ])
    monkeypatch.setattr(stock_analysis, "_read_parquet", lambda rel: frame)
    cleaned = stock_analysis.load_financial("002714")
    result = stock_analysis._financial_summary(cleaned)
    assert result["report_date"] == "2026-03-31"
    assert result["roe"] == -1.39


def test_profile_score_exposes_missing_and_coverage_adjustment():
    score = stock_analysis._score_profile({"trend": {"above_ma20": True}, "liquidity": {"vol_ratio_5_20": 1.2}})
    assert score["raw_score"] == score["total"]
    assert score["coverage_adjusted_score"] < score["raw_score"]
    assert "dimension_contributions" in score
    assert "valuation" in score["unavailable_dimensions"]


def test_partial_commodity_can_participate_but_reduces_weight():
    profile = {
        "quote": {"close": 10},
        "score": {"total": 60, "coverage": {"coverage_ratio": 0.8}},
        "commodity": {"coverage": 0.5},
    }
    gate = build_decision_gate(profile, {}, {
        "profile": {"status": "available", "as_of": "2026-08-27"},
        "decision": {"status": "missing"},
        "stock_flow": {"status": "available", "as_of": "2026-08-27"},
        "flow": {"status": "available", "as_of": "2026-08-27"},
        "commodity": {"status": "partial", "as_of": "2026-08-27"},
        "research": {"status": "missing"},
        "news": {"status": "missing"},
    }, reference_date="2026-08-27")
    assert gate["source_status"]["commodity"]["is_decision_eligible"] is True
    assert gate["status"] == "review_only"
