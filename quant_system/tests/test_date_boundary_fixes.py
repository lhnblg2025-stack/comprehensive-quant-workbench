from __future__ import annotations

from datetime import date
import json

from quant_system import market_clock
from quant_system.analysis_core import battle_map, pipeline, research_flow


def test_latest_completed_trading_day_converts_aware_timestamp(monkeypatch):
    monkeypatch.setattr(market_clock, "is_trading_day", lambda value=None: True)
    monkeypatch.setattr(market_clock, "prev_trading_day", lambda value, n=1: date(2026, 8, 27))
    monkeypatch.setattr(market_clock, "latest_trading_day", lambda value=None: date(2026, 8, 28))

    # 07:30 UTC is 15:30 in China, so the current trading day is completed.
    assert market_clock.latest_completed_trading_day("2026-08-28T07:30:00+00:00") == date(2026, 8, 28)


def test_battle_map_passes_request_date_to_social_sentiment(monkeypatch):
    seen = {}

    def fake_import(name, fromlist=()):
        assert name.endswith("social_sentiment")

        def composite(as_of=None):
            seen["as_of"] = as_of
            return {"market_sentiment": 0.1, "coverage": 1.0}

        return type("M", (), {"market_sentiment_composite": staticmethod(composite)})

    monkeypatch.setattr(battle_map, "__import__", fake_import, raising=False)
    result = battle_map._social_sentiment_for_date("2026-08-10")
    assert seen["as_of"] == "2026-08-10"
    assert result["market_sentiment"] == 0.1


def test_ima_index_maps_each_report_to_its_own_text(tmp_path, monkeypatch):
    export = tmp_path / "IMA资料导出" / "导出"
    base = export.parent / "研报库"
    base.mkdir(parents=True)
    rows = [
        {"title": "报告甲", "path": "报告甲.txt"},
        {"title": "报告乙", "path": "报告乙.txt"},
    ]
    (base / "_items.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows), encoding="utf-8")
    (base / "报告甲.txt").write_text("甲" * 30, encoding="utf-8")
    (base / "报告乙.txt").write_text("乙" * 30, encoding="utf-8")
    monkeypatch.setattr(research_flow, "IMA_EXPORT_DIR", export)

    reports = research_flow.load_ima_index_reports()
    by_title = {row["title"]: row["content"] for row in reports}
    assert by_title["报告甲"] == "甲" * 30
    assert by_title["报告乙"] == "乙" * 30


def test_pipeline_gate_rejects_future_core_dates(monkeypatch):
    expected = "2026-08-28"
    core_dates = {"zt_daily_stats": "2026-08-28", "fusion": "2026-08-29"}
    assert pipeline._validate_core_dates(core_dates, expected)["future"] == {"fusion": "2026-08-29"}
