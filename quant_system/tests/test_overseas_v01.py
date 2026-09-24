from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from overseas_collector import assess_quotes  # noqa: E402


def _quote(label, date="2026-08-28", price=100, change=1):
    return {"label": label, "date": date, "price": price, "chg_pct": change}


def test_overseas_group_gate_passes_complete_snapshot():
    quotes = [
        _quote("纳指"), _quote("标普500"),
        _quote("伦敦金"), _quote("白银"),
        _quote("费城半导体"),
        _quote("恒生科技"), _quote("腾讯控股"),
        _quote("美元指数"), _quote("美10年债殖"),
    ]
    result = assess_quotes(quotes, as_of="2026-08-31")
    assert result["score_eligible"] is True
    assert result["coverage"]["failed_groups"] == []
    assert all(item["valid"] for item in result["quotes"])


def test_overseas_stale_and_missing_change_fail_closed():
    quotes = [
        _quote("纳指", date="2026-08-20"),
        _quote("标普500", change=None),
        _quote("伦敦金"), _quote("白银"),
    ]
    result = assess_quotes(quotes, as_of="2026-08-31")
    reasons = {item["invalid_reason"] for item in result["invalid_quotes"]}
    assert any(reason.startswith("stale:") for reason in reasons)
    assert "missing_change" in reasons
    assert result["score_eligible"] is False
    assert "us_equity" in result["coverage"]["failed_groups"]
