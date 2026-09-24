from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "quant_web" / "static"


def test_all_verified_deep_pages_remain_discoverable():
    index = (STATIC / "index.html").read_text(encoding="utf-8")
    for path in ("/v4.html", "/intraday.html", "/review.html", "/research_report.html", "/research_dashboard.html", "/market_weekly.html", "/research_images.html", "/knowledge_base.html"):
        assert path in index, path
    assert "m-north" in index
    assert "northboundHoldingsTable" in index
    assert "rtSignalAlerts" in index


def test_main_app_keeps_subnav_signals_and_visual_hooks():
    app = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "bindSubNavigation" in app
    assert "signals=true" in app
    assert "/api/alert" in app
    assert "visual:backtest" in app
    assert "visual:sector" in app
    assert "loadNorthboundHoldings" in app


def test_compatibility_assets_are_not_deleted():
    for name in ("v3-system.js", "v4_dashboard.html", "v5-init.js"):
        assert (STATIC / name).is_file(), name
