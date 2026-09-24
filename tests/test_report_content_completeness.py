from __future__ import annotations

from scripts.report_contract import contract_from_daily_review, contract_from_stock_overview, validate_report_contract


def test_daily_review_contract_keeps_deep_and_long_block_content():
    values = {"level1": {"level2": {"level3": {"level4": {"signal": "keep-me"}}}}, "rows": [{"code": f"{i:06d}", "name": f"N{i}"} for i in range(24)]}
    report = contract_from_daily_review({"date": "2026-08-28", "blocks": {"deep": {"source": "fixture", "value": values}}})
    claims = {item["claim"]: item["value"] for item in report["evidence"]}
    assert any(value == "keep-me" for value in claims.values())
    assert any(item["claim"] == "deep.rows.count" and item["value"] == 24 for item in report["evidence"])
    assert report["detail"]["blocks"]["deep"]["value"]["rows"][-1]["code"] == "000023"
    assert not validate_report_contract(report)


def test_stock_contract_exposes_all_deep_evidence_without_reparsing():
    payload = {
        "symbol": "002714", "name": "测试股", "as_of": "2026-08-28",
        "source_status": {"profile": {"status": "available", "label": "画像", "as_of": "2026-08-28"}},
        "profile": {"quote": {"close": 10, "date": "2026-08-28"}, "score": {"total": 60}},
        "stock_flow": {"rows": [{"date": "2026-08-27", "main_net_yi": 1.2}, {"date": "2026-08-28", "main_net_yi": 2.3}]},
        "flow": {"rows": [{"date": "2026-08-28", "net": 4}]},
        "research": [{"title": "研报事实", "full_text": "不得丢失"}],
        "stock_news": {"items": [{"title": "新闻"}]},
        "score_hierarchy": {"layers": [{"name": "趋势", "score": 80}]},
        "ic_weight_explanation": {"weights": {"momentum": 0.4}},
        "commodity": {"items": [{"name": "猪价", "return_20d_pct": 3.2}]},
        "decision": {"decision": "observe"}, "market_context": {"m2": {"value": 1}},
        "evidence_summary": {"research_matches": 1},
        "decision_explanation": {"stance": "observe", "supporting_factors": ["证据待确认"]},
    }
    report = contract_from_stock_overview(payload)
    detail = report["detail"]
    assert detail["stock_flow"]["rows"][-1]["main_net_yi"] == 2.3
    assert detail["research"][0]["full_text"] == "不得丢失"
    assert detail["score_hierarchy"]["layers"][0]["name"] == "趋势"
    assert not validate_report_contract(report), validate_report_contract(report)
