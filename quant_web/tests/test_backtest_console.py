from __future__ import annotations

from quant_web.handlers.backtest_console import _gate, _matrix_rows, _ml_summary, _registry_rows


def test_console_reads_real_research_state():
    gate = _gate()
    assert gate["status"] in {"PASS", "DATA_BLOCKED", "unknown"}
    rows = _matrix_rows()
    assert isinstance(rows, list)
    assert isinstance(_registry_rows(), list)
    assert isinstance(_ml_summary(), dict)


def test_console_matrix_rows_have_comparable_metrics():
    for row in _matrix_rows()[:5]:
        assert "factor" in row
        assert "oos_sharpe" in row
        assert "cost_x2_total_return" in row
