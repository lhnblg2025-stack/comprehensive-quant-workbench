

def test_missing_requested_daily_report_does_not_fallback(monkeypatch, tmp_path):
    from quant_web import server
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "review_2026-08-28.json").write_text('{"date":"2026-08-28"}', encoding="utf-8")
    monkeypatch.setattr(server, "ROOT", tmp_path)
    handler = server.QuantHandler.__new__(server.QuantHandler)
    handler._sent = []
    handler._send_json = lambda payload, status=200: handler._sent.append((status, payload))
    handler._handle_research_report({"type": ["daily_review"], "date": ["2026-08-27"]})
    assert handler._sent[-1][0] == 404
    assert handler._sent[-1][1]["data_status"] == "missing"
    assert "未回退其他日期" in handler._sent[-1][1]["error"]
