from __future__ import annotations

import pandas as pd
import pytest

from quant_system.production_controls import (
    ProductionPreflight,
    UnavailableOMS,
    apply_pit_corporate_actions,
    build_pit_release,
    build_three_way_reconciliation,
    transition_approval,
    validate_corporate_actions,
    validate_trade_state_release,
)


def trade_state():
    return pd.DataFrame([{"code":"600000","date":"2024-01-02","suspended":False,"suspension_reason":"","st_flag":False,"limit_up_price":11.,"limit_down_price":9.,"limit_up_locked":False,"limit_down_locked":False,"source_document_id":"exchange:600000:20240102","source_as_of":"2024-01-02"}])


def test_pit_release_is_fail_closed_without_attestation(tmp_path):
    source = tmp_path / "state.parquet"; trade_state().to_parquet(source)
    release = build_pit_release("pit-1", "2024-01-02", "licensed-provider", [source], trade_state_authoritative=True, instrument_master_authoritative=True, corporate_actions_authoritative=True)
    assert release["validation"]["status"] == "BLOCK"
    assert "independent_release_attestation_missing" in release["validation"]["errors"]
    checked = validate_trade_state_release(trade_state(), release=release)
    assert checked["status"] == "BLOCK"


def test_corporate_actions_are_pit_bounded_and_applied(tmp_path):
    actions = pd.DataFrame([{"code":"600000","ex_date":"2024-01-03","action_type":"split","ratio":2.,"cash_per_share":0.,"source_document_id":"exchange:action:1","source_as_of":"2024-01-02"}])
    report = validate_corporate_actions(actions, release_as_of="2024-01-03")
    assert report["status"] == "PASS"
    panel = pd.DataFrame([{"code":"600000","date":"2024-01-04","raw_open":10.,"raw_high":11.,"raw_low":9.,"raw_close":10.}])
    adjusted, applied = apply_pit_corporate_actions(panel, actions, as_of="2024-01-03")
    assert adjusted.iloc[0].raw_close == 5.
    assert applied["applied_actions"] == 1


def test_three_way_reconciliation_blocks_mismatch():
    frame = lambda shares: pd.DataFrame([{"symbol":"600000","shares":shares}])
    result = build_three_way_reconciliation(frame(100), frame(90), frame(100))
    assert result["status"] == "BLOCK"
    assert result["summary"]["mismatched"] == 1


def test_preflight_and_approval_are_fail_closed():
    event = transition_approval("DRAFT", "REVIEW", actor="researcher", evidence={"manifest":"m1"})
    assert event["to"] == "REVIEW"
    with pytest.raises(ValueError): transition_approval("DRAFT", "APPROVED", actor="x", evidence={"x":1})
    check = ProductionPreflight({"validation":{"status":"BLOCK"}}, UnavailableOMS().health(), False, {"status":"BLOCK"}, {"status":"REVIEW"}).evaluate()
    assert check["status"] == "BLOCK"
    assert "oms_not_production_ready" in check["blockers"]
    assert UnavailableOMS().submit({})["status"] == "BLOCKED"


def test_research_release_freeze_detects_input_mutation(tmp_path):
    from quant_system.production_controls import freeze_research_release, verify_research_release
    source = tmp_path / "prices.csv"
    source.write_text("date,code,close\n2024-01-01,000001,10\n", encoding="utf-8")
    manifest = freeze_research_release(tmp_path / "release.json", [source], as_of="2024-01-01")
    assert manifest["status"] == "FROZEN"
    assert verify_research_release(manifest)["status"] == "PASS"
    source.write_text("date,code,close\n2024-01-01,000001,11\n", encoding="utf-8")
    assert verify_research_release(manifest)["status"] == "BLOCK"


def test_research_evidence_never_authorizes_live_execution():
    from quant_system.production_controls import build_research_evidence
    evidence = build_research_evidence(
        release={"status": "FROZEN", "release_id": "r1", "evidence_sha256": "h"},
        quality={"status": "PASS"}, pit={"status": "PASS"},
        corporate_actions={"status": "PASS"}, factors=["momentum"],
    )
    assert evidence["status"] == "PASS"
    assert evidence["live_execution"]["status"] == "BLOCK"
