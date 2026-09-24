# -*- coding: utf-8 -*-
"""股票级研报证据归因等级测试。"""
from scripts.unified_decision_snapshot import _research_evidence


def test_research_evidence_preserves_code_and_name_indexes():
    data = _research_evidence({
        "n_reports": 2,
        "parsed": [
            {"title": "代码研报", "codes": ["601678"], "companies": ["滨化股份(601678)"], "concepts": ["化工原料"], "rating": "strong", "target_price": 8.2, "catalysts": ["涨价"]},
            {"title": "简称研报", "codes": [], "companies": ["滨化股份"], "concepts": ["化工原料"], "rating": "neutral", "target_price": None, "catalysts": ["政策"]},
        ],
        "all_codes": ["601678"],
    })
    assert data["codes"]["601678"]["reports"] == 1
    assert data["names"]["滨化股份"]["reports"] == 2


def test_name_only_evidence_does_not_count_as_code_score():
    data = _research_evidence({
        "n_reports": 1,
        "parsed": [{"title": "简称研报", "codes": [], "companies": ["滨化股份"], "rating": "strong", "catalysts": ["订单"]}],
    })
    assert data["codes"] == {}
    assert data["names"]["滨化股份"]["decision_extracted_reports"] == 1
