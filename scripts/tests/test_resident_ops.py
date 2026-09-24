from __future__ import annotations

from scripts import resident_ops


def test_digest_is_nonempty_without_external_delivery(monkeypatch):
    monkeypatch.setattr(resident_ops, "dashboard_snapshot", lambda: {
        "release": {"release_id": "r1", "expected_day": "2026-08-28", "warnings": []},
        "health": {"ok": True, "alert_count": 0, "alerts": []},
        "last_after_close": {"status": "succeeded", "attempts": []},
        "paper_execution": {"orders": 0, "fill_rate": 0, "mean_adverse_slippage_bps": None},
    })
    text = resident_ops.digest()
    assert "量化常驻节点日报" in text
    assert "r1" in text
