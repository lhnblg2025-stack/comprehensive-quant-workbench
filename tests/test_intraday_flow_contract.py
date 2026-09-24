from scripts import build_intraday_flow_history as flow
from quant_web.handlers import intraday as intraday_handler
from quant_web.handlers import v11 as v11_handler


def test_intraday_handler_rejects_invalid_date(monkeypatch):
    sent = []
    intraday_handler.handler_intraday({"date": ["bad-date"]}, lambda payload, status=200: sent.append((status, payload)))
    assert sent[-1][1]["ok"] is False
    assert "YYYY-MM-DD" in sent[-1][1]["error"]


def test_intraday_handler_exposes_cross_date_chain_status(monkeypatch, tmp_path):
    monkeypatch.setattr(intraday_handler, "ROOT", tmp_path)
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "intraday_chain_2026-08-25.json").write_text(
        '{"time":"2026-08-25 15:00:00","n_scanned":1,"rows":[{"code":"600000"}],"coverage":{"status":"complete"}}', encoding="utf-8"
    )
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "intraday_monitor_light.py").write_text(
        "def monitor_snapshot(): return {}\n", encoding="utf-8"
    )
    sent = []
    intraday_handler.handler_intraday({"date": ["2026-08-26"]}, lambda payload, status=200: sent.append((status, payload)))
    assert sent[-1][1]["data"]["decision_chain"]["source_date"] == "2026-08-25"
    assert sent[-1][1]["data"]["decision_chain"]["date_mismatch"] is True


def test_intraday_payload_requires_market_and_etf_for_complete_day(monkeypatch, tmp_path):
    monkeypatch.setattr(flow, "GENERATED", tmp_path)
    monkeypatch.setattr(flow, "_snapshot_files", lambda _day: [])
    monkeypatch.setattr(flow, "_industry_map", lambda: __import__("pandas").DataFrame(columns=["code", "industry", "industry_code"]))
    monkeypatch.setattr(flow, "_build_etf_history", lambda _day: (
        [{"time": "09:35"}, {"time": "10:10"}, {"time": "13:10"}, {"time": "14:40"}],
        {"status": "available", "snapshot_count": 4},
    ))
    result = flow.build("2026-08-26")
    assert result["market_complete_day"] is False
    assert result["etf_complete_day"] is True
    assert result["complete_day"] is False
