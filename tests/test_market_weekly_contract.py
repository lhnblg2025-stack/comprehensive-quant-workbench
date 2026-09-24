from __future__ import annotations

from scripts.market_weekly_report import build_market_weekly, render_markdown


def test_weekly_report_registers_real_missing_modules():
    report = build_market_weekly("2026-08-27")
    fields = {item["field"] for item in report["data_gaps"]}
    assert report["schema_version"] == "market_weekly.v2"
    assert {"market_flow", "market_margin", "futures_basis"} <= fields
    if report["industries"]["status"] == "missing":
        assert "industry" in fields
    else:
        assert report["industries"]["status"] == "available"
    assert report["market_flow"]["status"] in {"missing", "partial"}
    assert report["margin"]["status"] == "missing"


def test_weekly_markdown_explains_gap_impact_in_chinese():
    report = build_market_weekly("2026-08-27")
    text = render_markdown(report)
    assert "缺口影响" in text
    assert "本周不形成方向性结论" in text
    assert "市场资金" in text
