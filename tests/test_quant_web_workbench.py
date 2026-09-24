from __future__ import annotations

from quant_web.api_catalog import API_CATALOG_CONTRACT, build_catalog
from quant_web.workbench import WORKBENCH_CONTRACT, project_snapshot


def test_api_catalog_is_stable_and_version_neutral():
    payload = build_catalog()
    assert payload["contract"] == API_CATALOG_CONTRACT
    assert payload["routing_policy"]["source_of_truth"]
    assert {item["id"] for item in payload["domains"]} >= {"decision", "market", "stock", "research", "strategy"}
    assert all("read_path" in item for item in payload["domains"])


def test_workbench_projects_small_truthful_observation_state():
    payload = project_snapshot({
        "ok": True,
        "schema_version": "decision-snapshot.v1",
        "mode": "after_close",
        "as_of": "2026-08-27",
        "generated_at": "2026-08-28T20:00:00+08:00",
        "degraded": True,
        "market": {
            "temperature": 16,
            "force_index": 25,
            "breadth": 0.5,
            "emotion_stage": "冰点",
            "risk_flags": ["行业资金缺失"],
            "zt": {"zt_cnt": 12, "dt_cnt": 0, "zb_cnt": 2, "max_board": 6},
        },
        "mainlines": [{"name": "信创", "zt": 3, "candidate_count": 21, "structure_score": 56}],
        "short_term_candidates": [],
        "observation_candidates": [{"symbol": "000017", "name": "深中华A", "trade_allowed": False}],
        "counts": {"scanned": 5202, "execution_candidates": 0, "observation_candidates": 10},
        "source_dates": {"fusion": "2026-08-27", "industry_fund_flow": None},
        "source_freshness": {"fusion": True, "industry_fund_flow": False},
        "source_chain": {"fusion": "fusion.parquet"},
        "date_mismatches": {"fund_forces": "2026-08-25"},
    })
    assert payload["contract"] == WORKBENCH_CONTRACT
    assert payload["state"] == "degraded"
    assert payload["candidate_mode"] == "observation"
    assert payload["counts"]["execution_candidates"] == 0
    assert next(item for item in payload["source_status"] if item["key"] == "industry_fund_flow")["status"] == "missing"
    assert len(payload["candidates"]) == 1
    assert payload["empty_policy"]["candidates"]
