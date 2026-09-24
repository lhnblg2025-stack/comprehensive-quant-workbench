from __future__ import annotations

import pandas as pd
import pytest

from quant_system.backtest_protocol import BacktestRequest
from quant_system.backtest_service import data_gate, execute, spec_hash


def panel() -> pd.DataFrame:
    rows = []
    for i, day in enumerate(pd.bdate_range("2024-01-02", periods=8)):
        for j, code in enumerate(("000001", "000002")):
            price = 10.0 + i * 0.1 + j
            rows.append({"date": day, "code": code, "raw_open": price, "raw_high": price * 1.01,
                         "raw_low": price * .99, "raw_close": price, "volume": 100000,
                         "amount": price * 100000, "score": float(j)})
    return pd.DataFrame(rows)


def state() -> pd.DataFrame:
    data = panel()[["date", "code"]].copy()
    data["suspended"] = False
    data["limit_up_locked"] = False
    data["limit_down_locked"] = False
    return data


def test_request_protocol_rejects_invalid_risk_and_hashes_costs():
    base = {"strategy": {"signal": "score"}, "costs": {"slippage_bps": 10}}
    request = BacktestRequest.from_dict(base)
    assert request.schema == "canonical-backtest/v1"
    changed = BacktestRequest.from_dict({**base, "costs": {"slippage_bps": 20}})
    assert spec_hash(request.to_dict()) != spec_hash(changed.to_dict())
    with pytest.raises(ValueError, match="max_position_weight"):
        BacktestRequest.from_dict({"strategy": {"signal": "score"}, "risk": {"max_position_weight": 2}})


def test_production_gate_fails_closed_without_quality_report(tmp_path):
    request = BacktestRequest.from_dict({"strategy": {"signal": "score"}, "mode": "paper"})
    result = execute(request, panel(), trade_state=state(), quality_path=tmp_path / "missing.json")
    assert result["ok"] is False
    assert result["status"] == "DATA_BLOCKED"
    assert "quality_report_missing" in result["gate"]["blockers"]


def test_research_executes_through_canonical_backtrader(tmp_path):
    pytest.importorskip("backtrader")
    request = BacktestRequest.from_dict({
        "strategy": {"signal": "score", "family": "trend", "quantile": .5, "rebalance_sessions": 2},
        "costs": {"slippage_bps": 0},
        "risk": {"max_position_weight": .5, "max_names": 2},
        "mode": "research",
    })
    run = execute(request, panel(), trade_state=state(), quality_path=tmp_path / "missing.json")
    assert run["ok"] is True
    assert run["canonical"] is True
    assert run["schema"] == "canonical-backtest/v1"
    assert run["result"].metadata["research_framework"] == "microsoft_qlib"
    assert run["result"].metadata["execution_engine"] == "backtrader"


def test_data_gate_detects_duplicate_and_invalid_execution_rows(tmp_path):
    broken = pd.concat([panel(), panel().iloc[[0]]], ignore_index=True)
    broken.loc[0, "raw_open"] = 0
    gate = data_gate(broken, quality_path=tmp_path / "missing.json")
    assert gate["status"] == "BLOCK"
    assert any(item.startswith("duplicate_date_code") for item in gate["blockers"])
    assert any(item.startswith("nonpositive_raw_open") for item in gate["blockers"])
