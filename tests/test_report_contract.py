from __future__ import annotations

import json

import pytest

from scripts.report_contract import (
    contract_from_daily_review,
    make_report_contract,
    validate_report_contract,
    validate_and_write_contract,
)


def _valid_report():
    return make_report_contract(
        "weekly", "测试周报", "2026-08-28",
        period_start="2026-08-24", period_end="2026-08-28",
        subject={"id": "market", "name": "全市场", "kind": "market"},
        summary={"stance": "observe", "summary": "等待确认", "horizon": "weeks", "confidence": 0.5},
        sources=[{"id": "src-1", "name": "本地仓库", "kind": "warehouse", "status": "ok", "observed_at": "2026-08-28"}],
        evidence=[{"id": "ev-1", "claim": "指数周涨跌", "value": 0.01, "unit": "pct", "observed_at": "2026-08-28", "source_ref": "src-1", "quality": "primary"}],
        transmission=[{"from": "指数", "to": "风险预算", "mechanism": "趋势确认", "direction": "positive", "evidence_refs": ["ev-1"]}],
        risks=[{"description": "覆盖不足", "trigger": "数据缺失", "impact": "降级", "severity": "medium", "evidence_refs": ["ev-1"]}],
        actions=[{"action": "validate", "target": "市场", "condition": "同日数据确认", "invalidated_by": "日期错位", "evidence_refs": ["ev-1"]}],
        chapters=[{"title": "核心", "conclusion": "观察", "evidence": ["ev-1"], "implication": "等待", "next_check": "下周"}],
    )


def test_contract_has_subject_and_stable_ids():
    report = _valid_report()
    assert report["report_id"] == "weekly:market:2026-08-28"
    assert report["subject"]["kind"] == "market"
    assert not validate_report_contract(report)


def test_missing_source_is_degraded_and_traceable():
    report = make_report_contract(
        "daily_review", "日报", "2026-08-28",
        summary="观察", evidence=[{"claim": "宽度", "value": 0.5, "observed_at": "2026-08-28"}],
    )
    assert report["status"]["state"] == "failed"
    assert report["sources"][0]["id"] == "src-undisclosed"
    assert report["evidence"][0]["source_ref"] == "src-undisclosed"


def test_none_bucket_returns_problems_instead_of_throwing():
    report = _valid_report()
    report["actions"] = None
    problems = validate_report_contract(report)
    assert "actions_not_list" in problems


def test_bad_relationship_ref_is_rejected():
    report = _valid_report()
    report["transmission"][0]["evidence_refs"] = ["ev-nope"]
    assert "transmission_1_evidence_ref" in validate_report_contract(report)


def test_sidecar_validates_before_writing(tmp_path):
    report = _valid_report()
    json_path, md_path = validate_and_write_contract(tmp_path / "report.md", report)
    assert json.loads(json_path.read_text(encoding="utf-8"))["quality"]["validated"] is True
    assert "事实证据" in md_path.read_text(encoding="utf-8")
    broken = _valid_report()
    broken["subject"] = None
    with pytest.raises(ValueError, match="invalid research report contract"):
        validate_and_write_contract(tmp_path / "broken.md", broken)
    assert not (tmp_path / "broken.md.research.json").exists()


def test_daily_review_adapter_preserves_legacy_fields_and_links_evidence():
    report = contract_from_daily_review({
        "run_id": "run-1", "date": "2026-08-28",
        "health": {"degraded": ["breadth"]},
        "blocks": {"market": {"source": "local", "value": {"breadth": 0.6, "date": "2026-08-28"}}},
        "battle_map": {"action_card": "谨慎观望"},
        "decision": {"stance": "observe", "summary": "等待资金确认"},
    })
    assert report["report_type"] == "daily_review"
    assert report["subject"]["id"] == "ashare-market"
    assert report["data_gaps"]
    assert not validate_report_contract(report)
