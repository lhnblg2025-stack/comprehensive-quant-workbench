from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import quant_web.handlers.v11 as v11
import quant_web.server as server


def _stub_handler() -> server.QuantHandler:
    handler = server.QuantHandler.__new__(server.QuantHandler)
    handler._sent = []
    handler._send_json = lambda payload, status=200: handler._sent.append((status, payload))
    return handler


def test_stock_overview_exposes_code_attributed_news(monkeypatch, tmp_path):
    fake_news = {
        "ok": True,
        "coverage": "stock_symbol",
        "source_symbol": "600519",
        "source": "akshare.stock_news_em",
        "n": 1,
        "sentiment": 0.2,
        "bull": 1,
        "bear": 0,
        "neutral": 0,
        "records": [{"title": "贵州茅台发布重要公告", "date": "2026-08-25", "source": "财经源"}],
    }
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "build_stock_profile", lambda symbol: {
        "name": "贵州茅台", "quote": {"date": "2026-08-25"}, "signals": []
    })
    monkeypatch.setattr(server, "normalize_symbol", lambda value: str(value).zfill(6))
    monkeypatch.setattr(server, "resolve_symbol", lambda value: value)
    monkeypatch.setattr(
        "quant_system.analysis_core.social_sentiment.fetch_news",
        lambda symbol, limit=30: fake_news,
    )
    monkeypatch.setattr(
        "quant_system.analysis_core.social_sentiment.sentiment_score",
        lambda title: 0.2,
    )
    monkeypatch.setattr(
        "quant_system.data_pipeline.get_stock_money_flow_with_status",
        lambda symbol, days=60: (pd.DataFrame(), {"status": "provider_empty", "source": "test", "message": "test empty"}),
    )

    handler = _stub_handler()
    handler._handle_stock_overview({"symbol": ["600519"]})
    status, payload = handler._sent[-1]

    assert status == 200
    assert payload["ok"] is True
    news = payload["stock_news"]
    assert news["source_symbol"] == "600519"
    assert news["coverage"] == "stock_symbol"
    assert news["items"][0]["title"] == "贵州茅台发布重要公告"
    assert payload["evidence_summary"]["news_headlines"] == 1


def test_stock_overview_does_not_expose_unattributed_news(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "build_stock_profile", lambda symbol: {"quote": {}, "signals": []})
    monkeypatch.setattr(server, "normalize_symbol", lambda value: str(value).zfill(6))
    monkeypatch.setattr(server, "resolve_symbol", lambda value: value)
    monkeypatch.setattr(
        "quant_system.analysis_core.social_sentiment.fetch_news",
        lambda symbol, limit=30: {
            "ok": True, "coverage": "market_aggregate", "source_symbol": None,
            "records": [{"title": "市场聚合标题"}], "n": 1,
        },
    )
    monkeypatch.setattr(
        "quant_system.data_pipeline.get_stock_money_flow_with_status",
        lambda symbol, days=60: (pd.DataFrame(), {"status": "provider_empty", "source": "test", "message": "test empty"}),
    )

    handler = _stub_handler()
    handler._handle_stock_overview({"symbol": ["600519"]})
    _, payload = handler._sent[-1]

    assert payload["stock_news"]["status"] == "missing"
    assert payload["stock_news"]["items"] == []
    assert payload["source_status"]["news"]["status"] == "missing"


def test_etf_flow_filters_code_and_reports_share_proxy(monkeypatch, tmp_path):
    events = tmp_path / "data_warehouse" / "events"
    events.mkdir(parents=True)
    state = pd.DataFrame([
        {"code": "510300", "name": "沪深300ETF", "date": "2026-08-24", "price": 4.6, "shares": 2_000_000_000, "amount": 100_000_000},
        {"code": "510300", "name": "沪深300ETF", "date": "2026-08-25", "price": 4.7, "shares": 2_100_000_000, "amount": 120_000_000, "shares_delta": 100_000_000},
        {"code": "510050", "name": "上证50ETF", "date": "2026-08-25", "price": 2.8, "shares": 3_000_000_000, "amount": 900_000_000, "shares_delta": 50_000_000},
    ])
    state.to_parquet(events / "etf_state.parquet", index=False)
    history = state.copy()
    history["scale_yi"] = history["shares"] * history["price"] / 1e8
    history.to_parquet(events / "etf_history.parquet", index=False)
    (events / "official_capital_events.json").write_text(json.dumps([]), encoding="utf-8")

    import quant_system.analysis_core.common as common
    monkeypatch.setattr(common, "ROOT", tmp_path)
    sent = []
    v11.handler_v12_etf_flow({"code": ["510300"], "days": ["20"]}, lambda payload, status=200: sent.append((status, payload)))

    status, payload = sent[-1]
    assert status == 200
    assert payload["code"] == "510300"
    assert {row["code"] for row in payload["history"]} == {"510300"}
    assert payload["signals"]["latest_shares_delta_m"] == 100.0
    assert payload["signals"]["flow_direction"] == "net_subscription_proxy"
    assert payload["signals"]["official_flow_available"] is False
    assert payload["requested_days"] == 20
    assert payload["actual_days"] == 2


def test_etf_flow_rejects_unattributed_legacy_table(monkeypatch, tmp_path):
    events = tmp_path / "data_warehouse" / "events"
    events.mkdir(parents=True)
    pd.DataFrame([{"date": "2026-08-25", "close": 4.6, "amount": 100_000_000}]).to_parquet(events / "etf_flow.parquet", index=False)
    import quant_system.analysis_core.common as common
    monkeypatch.setattr(common, "ROOT", tmp_path)
    sent = []

    v11.handler_v12_etf_flow({"code": ["510300"]}, lambda payload, status=200: sent.append((status, payload)))

    status, payload = sent[-1]
    assert status == 200
    assert payload["ok"] is False
    assert payload["data_status"] == "unattributed"


def test_frontend_contains_decision_evidence_contract():
    root = Path(__file__).resolve().parents[1] / "quant_web"
    app = (root / "static" / "app.js").read_text(encoding="utf-8")
    html = (root / "static" / "index.html").read_text(encoding="utf-8")
    assert "stockNews.items" in app
    assert "data-stock-flow-mode" in app
    assert "main_net_yi" in app and "active_flow_proxy_yi" in app
    assert "official_flow_available" in app
    assert "String(date || '').match(/^20\\d{2}-\\d{2}-\\d{2}$/)" in app
    assert "|| sharedSymbol()" in app
    assert "002172" not in app
    assert "etfEventList" in html
    assert "份额变化 / 申赎方向代理" in html
    assert "commodityStatusLabel" in app
    assert "当前不以行业代理替代" in app
    assert "content.hidden = !hasQuote" in app
    assert "当前不以行业代理替代" in app
    assert "commodityDirectionLabel" in app
    assert "marketRecapFlowChart" in app
    assert "为什么入选" in app
    assert "资金来自哪里" in app
    assert "何时失效" in app
    assert "data.score ?? '-'" in app
    assert "data.total ?? (headlines.length || null)" in app
    assert 'data-sub="s-ai"' in html and 'data-sub="f-library"' in html and 'data-sub="m-history"' in html
    assert "var(--text-muted)" not in html
    assert "data-view=" + '"stock" data-sub="s-ai"' in html
    assert "data-view=" + '"factors" data-sub="f-library"' in html
    assert "data-view=" + '"market" data-sub="m-history"' in html
    css = (root / "static" / "styles.css").read_text(encoding="utf-8")
    assert "--text-muted" not in css
    assert "color: transparent !important" not in css
    assert "metric .value[style*=\"color\"]" in css
    assert '<section id="slip" class="view"' in html
    assert "table { min-width: 980px; }" in css
    assert "color: var(--muted) !important" in css
    assert "-webkit-text-fill-color: currentColor !important" in css
    assert ".skeleton.empty" in css
    assert "pointer-events: none" in css
    assert 'id="slip" class="view"' in html
    assert 'style="display:block;padding:8px 14px;color:#b0b7c3' not in html
    assert "Math.abs(Number(b[metric]" not in app
    assert "非主力净流入" in app
    assert "active_flow_proxy_yi:'成交额方向代理" in app
    viz = (root / "static" / "viz9.js").read_text(encoding="utf-8")
    assert "dom.innerHTML = '';" not in viz
    assert "面积=传入权重" in viz
