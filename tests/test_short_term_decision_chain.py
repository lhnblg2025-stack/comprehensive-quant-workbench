from pathlib import Path
import json

from scripts import unified_decision_snapshot as uds
from scripts.html_report_generator import _load_review, _render_decision_audit, _render_decision_snapshot

ROOT = Path(__file__).resolve().parents[1]


def test_requested_review_date_never_falls_back(monkeypatch, tmp_path):
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "review_2026-08-25.json").write_text('{"date":"2026-08-25"}', encoding="utf-8")
    monkeypatch.setattr("scripts.html_report_generator.ROOT", tmp_path)
    assert _load_review("2026-08-26") == {}


def test_specific_theme_excludes_generic_market_labels():
    assert uds._is_specific_theme("机器人概念")
    assert uds._is_specific_theme("人工智能")
    assert not uds._is_specific_theme("融资融券")
    assert not uds._is_specific_theme("深股通")
    assert not uds._is_specific_theme("创业板综")


def test_short_score_blocks_chasing_and_weak_market():
    item = {"change_pct": 9.8, "amount": 5e8, "turnover": 8}
    score, reasons, blockers = uds._short_score(
        item,
        {"industry": "机器人概念", "score": 80},
        50.0,
        None,
        25.0,
        0.60,
    )
    assert score > 0
    assert any("接近涨停" in reason for reason in blockers)
    assert any("资金合力偏弱" in reason for reason in blockers)


def test_concept_flow_uses_trading_day_column_not_timestamp_day(monkeypatch, tmp_path):
    import pandas as pd
    monkeypatch.setattr(uds, "MARKET", tmp_path)
    frame = pd.DataFrame([
        {"date": "2026-08-28", "ts": "2026-08-28 14:50:00", "type": "概念", "name": "机器人", "main_net_yi": 8.0},
        {"date": "2026-08-28", "ts": "2026-08-29 06:40:00", "type": "概念", "name": "机器人", "main_net_yi": 9.0},
    ])
    frame.to_parquet(tmp_path / "concept_fund_flow_intraday.parquet")
    values, as_of = uds._load_flow_values("2026-08-28")
    assert as_of == "2026-08-28"
    assert values["机器人"] == 9.0


def test_strict_short_score_rejects_missing_flow():
    score, reasons, blockers = uds._short_score(
        {"change_pct": 4.0, "amount": 5e8, "turnover": 8},
        {"name": "机器人概念", "score": 80}, None, None, 50.0, 0.7, strict=True,
    )
    assert score is None
    assert not reasons
    assert any("严格模式不评分" in item for item in blockers)


def test_short_score_is_normalized_and_explained():
    score, _, _, breakdown = uds._short_score_detail(
        {"change_pct": 4.0, "amount": 5e8, "turnover": 8, "composite_conf": 5.0},
        {"name": "机器人概念", "structure_score": 80, "structure_status": "当日结构"},
        20.0,
        {"board_count": 2, "is_zt": True},
        50.0,
        0.7,
        stock_flow_value=3.0,
    )
    assert 0 <= score <= 100
    assert breakdown["_meta"]["max_raw_score"] == 135.0
    assert breakdown["_meta"]["normalized_score"] == score
    assert 0 <= score <= 100
    assert breakdown["行业资金"]["weight_pct"] == 13.3
    assert all("normalized_points" in value for key, value in breakdown.items() if key != "_meta")


def test_live_snapshot_shortlist_is_mainboard_and_bounded():
    path = ROOT / "generated" / "decision_snapshot_intraday_2026-08-26.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    selected = data.get("short_term_candidates") or []
    assert len(selected) <= 10
    assert data["counts"]["scanned"] >= data["counts"]["candidates"]
    for item in selected:
        assert item["board"] in ("沪主板", "深主板")
        assert item.get("mainline_match")
        assert (item.get("flow_match") or {}).get("net_yi", 0) > 0
        assert not item.get("hard_risk")
        assert item.get("short_term_reasons")
        assert (item.get("execution_strategy") or {}).get("entry")
        assert (item.get("execution_strategy") or {}).get("stop")


def test_report_renders_reasons_and_execution_strategy():
    snapshot = {
        "as_of": "2026-08-26",
        "counts": {"scanned": 5202, "candidates": 100, "short_term_selected": 1,
                   "execution_candidates": 0, "hard_risk_candidates": 0},
        "market": {"risk_flags": ["资金合力偏弱"]},
        "short_term_candidates": [{
            "code": "600000", "name": "样例", "industry": "银行", "board": "沪主板",
            "trade_label": "高分观察", "short_term_score": 72,
            "mainline_match": "银行", "flow_match": {"name": "银行", "net_yi": 10},
            "short_term_reasons": ["行业资金净流入+10亿"],
            "short_term_blockers": ["市场资金合力偏弱"],
            "regulatory_risk": {"tier1": 0},
            "execution_strategy": {"entry": "两轮确认后试探", "add": "回踩不破加仓",
                                   "stop": "行业资金转负退出", "avoid": "不追涨"},
        }],
    }
    html = _render_decision_snapshot(snapshot)
    assert "全市场短线扫描" in html
    assert "行业资金净流入" in html
    assert "两轮确认后试探" in html
    assert "行业资金转负退出" in html


def test_after_close_chain_rejects_cross_date_intraday(monkeypatch, tmp_path):
    gen = tmp_path / "generated"
    gen.mkdir()
    (gen / "after_close_extra_2026-08-26.json").write_text("{}", encoding="utf-8")
    (gen / "intraday_chain_2026-08-25.json").write_text(
        json.dumps({"n_scanned": 5200, "n_candidates": 1, "candidate_rows": [{"code": "600000"}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(uds, "GEN", gen)
    chain = uds._load_chain("after_close", "2026-08-26")
    assert not chain.get("candidate_rows")
    assert chain["intraday_coverage"]["status"] == "date_mismatch"
    assert chain["_source_dates"]["intraday_chain"] == "2026-08-25"


def test_after_close_chain_preserves_same_day_coverage(monkeypatch, tmp_path):
    gen = tmp_path / "generated"
    gen.mkdir()
    (gen / "after_close_extra_2026-08-26.json").write_text("{}", encoding="utf-8")
    payload = {
        "n_scanned": 5202,
        "n_candidates": 197,
        "candidate_rows": [{"code": "600000"}],
        "coverage": {"status": "complete", "snapshot_rows": 5202, "signal_evaluated": 5202,
                     "candidate_count": 197, "evidence_evaluated": 197, "ratio": 1.0},
    }
    (gen / "intraday_chain_2026-08-26.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(uds, "GEN", gen)
    chain = uds._load_chain("after_close", "2026-08-26")
    assert chain["n_scanned"] == 5202
    assert chain["n_candidates"] == 197
    assert chain["intraday_coverage"]["status"] == "complete"
    assert chain["candidate_rows"] == [{"code": "600000"}]


def test_decision_audit_renders_coverage_sources_and_candidates():
    snapshot = {
        "intraday_coverage": {"status": "complete", "snapshot_rows": 5202, "signal_evaluated": 5202,
                              "candidate_count": 197, "evidence_evaluated": 197, "ratio": 1.0},
        "source_chain": {"intraday_chain": True},
        "source_dates": {"intraday_chain": "2026-08-26"},
        "mainlines": [{"industry": "机器人", "source": "intraday", "score": 90}],
        "opportunities": [{"code": "600000", "name": "样例", "board": "沪主板", "industry": "机器人",
                           "short_term_score": 70, "short_term_reasons": ["资金流入"],
                           "short_term_blockers": ["等待确认"], "trade_label": "观察"}],
        "evidence": {"holders": {"items": []}, "regulatory": {"total_hits": 0, "codes": 0},
                     "factor_decay": {"status": "degraded", "reason": "insufficient_snapshot"}},
    }
    html = _render_decision_audit(snapshot)
    assert "扫描覆盖与来源日期" in html
    assert "5202" in html
    assert "全市场候选审计" in html
    assert "600000" in html
    assert "盘中覆盖降级" not in html
