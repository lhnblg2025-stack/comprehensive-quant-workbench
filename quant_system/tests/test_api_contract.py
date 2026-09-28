"""quant_web/server.py 的 Web API 契约测试（P1b）。

不启动真实端口，直接构造 QuantHandler 并调用 do_GET/do_POST，捕获
_send_json 的 JSON payload 与 HTTP status，验证前端依赖的响应字段结构。
数据获取函数按需 monkeypatch 为合成小数据，避免依赖真实 data_warehouse。
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import quant_web.server as server


class _CaptureHandler(server.QuantHandler):
    """Socket-free handler：只捕获 JSON 响应，不写网络。"""

    def __init__(self, path: str, body: dict | None = None) -> None:
        self.path = path
        self.directory = str(server.STATIC_DIR)
        self.headers: dict[str, str] = {}
        self.wfile = io.BytesIO()
        self.rfile = io.BytesIO()
        self.sent: list[tuple[dict, int]] = []
        self.errors: list[int] = []
        if body is not None:
            raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.rfile = io.BytesIO(raw)
            self.headers["Content-Length"] = str(len(raw))

    def _send_json(self, data: dict, status: int = 200) -> None:
        # 与原 _send_json 使用同一序列化规则，并立即反解析，
        # 保证测试断言的是前端真正能拿到的 JSON 类型。
        payload = json.loads(json.dumps(server._json_safe(data), ensure_ascii=False))
        self.sent.append((payload, status))

    def send_response(self, code: int, message: str | None = None) -> None:
        pass

    def send_header(self, keyword: str, value: str) -> None:
        pass

    def end_headers(self) -> None:
        pass

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        self.errors.append(code)


def _get(path: str) -> tuple[dict, int]:
    handler = _CaptureHandler(path)
    handler.do_GET()
    assert len(handler.sent) == 1, f"expected one JSON response for {path}, got {handler.sent}"
    return handler.sent[0]


def _post(path: str, body: dict) -> tuple[dict, int]:
    handler = _CaptureHandler(path, body=body)
    handler.do_POST()
    assert len(handler.sent) == 1, f"expected one JSON response for {path}, got {handler.sent}"
    return handler.sent[0]


def test_market_contract() -> None:
    """GET /api/market 直接返回 load_market_state 的结构。

    实际字段不是 temperature/cycle，而是 trade_date/risk/risk_level/
    breadth/hot_metrics/source；数据源缺失时会返回明确的 source='missing'。
    """
    payload, status = _get("/api/market")
    assert status == 200
    assert isinstance(payload, dict)
    assert isinstance(payload.get("trade_date"), (str, type(None)))
    assert isinstance(payload.get("risk"), int)
    assert isinstance(payload.get("risk_level"), str)
    assert isinstance(payload.get("breadth"), dict)
    assert isinstance(payload.get("hot_metrics"), dict)
    source = payload.get("source")
    assert isinstance(source, str) and source
    # 契约允许两种状态：正常数据源路径 或 明确的降级字段。
    assert source == "missing" or "meta.json" in source


def test_config_contract() -> None:
    """GET /api/config 应返回默认策略与组合配置键。"""
    payload, status = _get("/api/config")
    assert status == 200
    assert isinstance(payload.get("strategy"), dict)
    assert isinstance(payload.get("portfolio"), dict)
    assert isinstance(payload["strategy"].get("fast_ma"), int)
    assert isinstance(payload["strategy"].get("slow_ma"), int)
    assert isinstance(payload["portfolio"].get("initial_cash"), (int, float))
    assert isinstance(payload["portfolio"].get("max_position_pct"), (int, float))


def test_sector_contract(monkeypatch) -> None:
    """GET /api/sector 使用合成板块映射，验证分组结构。"""
    monkeypatch.setattr(
        "quant_system.data.fetch_sector_map",
        lambda refresh=False: {"600519": "白酒"},
    )
    payload, status = _get("/api/sector?symbols=600519")
    assert status == 200
    assert payload["ok"] is True
    assert isinstance(payload.get("sectors"), dict)
    assert isinstance(payload.get("sector_names"), dict)
    assert payload["sectors"]["白酒"] == ["600519"]
    assert payload["sector_names"]["白酒"] == "白酒"
    assert payload.get("note") == ""


def test_realtime_contract(monkeypatch) -> None:
    """GET /api/realtime 返回 quotes 列表，元素包含 code/price 等字段。"""
    quote = SimpleNamespace(
        symbol="002714",
        price=42.5,
        change=-0.3,
        change_pct="-0.7",
        high=43.1,
        low=42.0,
        open=42.9,
        volume=123456,
        amount=5200000.0,
        amplitude=2.56,
        date="2026-08-13",
        time="15:00:00",
    )
    monkeypatch.setattr(server, "fetch_realtime", lambda symbols, timeout=10: {"002714": quote})
    payload, status = _get("/api/realtime?symbols=002714")
    assert status == 200
    assert payload["ok"] is True
    assert isinstance(payload.get("quotes"), list)
    assert payload.get("count") == 1
    assert isinstance(payload.get("signals"), dict)
    item = payload["quotes"][0]
    assert item["code"] == "002714"
    assert isinstance(item["price"], (int, float))
    assert isinstance(item["change"], (int, float))
    assert isinstance(item["change_pct"], (int, float))
    assert isinstance(item["volume"], (int, float))


def test_search_stock_contract(monkeypatch) -> None:
    """GET /api/search_stock?q=茅台 返回 results 列表。

    注意：当前路由使用 results 键，不是 data 键。
    """
    monkeypatch.setattr(
        server,
        "fuzzy_search_stocks",
        lambda query, limit=20: [
            {"name": "贵州茅台", "code": "600519", "match_type": "name_contains", "score": 95}
        ],
    )
    payload, status = _get("/api/search_stock?q=茅台")
    assert status == 200
    assert payload["ok"] is True
    assert isinstance(payload.get("results"), list)
    assert len(payload["results"]) == 1
    item = payload["results"][0]
    assert isinstance(item.get("name"), str)
    assert isinstance(item.get("code"), str)
    assert isinstance(item.get("match_type"), str)


def test_portfolio_contract(monkeypatch) -> None:
    """GET /api/portfolio 返回组合汇总结构。"""
    monkeypatch.setattr(
        server,
        "_load_portfolio",
        lambda: {"positions": {}, "updated_at": ""},
    )
    payload, status = _get("/api/portfolio")
    assert status == 200
    assert payload["ok"] is True
    assert isinstance(payload.get("portfolio"), dict)
    assert isinstance(payload["portfolio"].get("positions"), list)
    assert isinstance(payload.get("total_cost"), (int, float))
    assert isinstance(payload.get("total_value"), (int, float))
    assert isinstance(payload.get("pnl"), (int, float))
    assert isinstance(payload.get("pnl_pct"), (int, float))


def test_v11_feedback_invalid_vote_returns_400() -> None:
    """POST /api/v11/feedback 非法 vote 返回 400 + ok:false。"""
    payload, status = _post("/api/v11/feedback", {"vote": "invalid"})
    assert status == 400
    assert payload["ok"] is False
    assert isinstance(payload.get("error"), str)


def test_unknown_path_returns_404() -> None:
    """未知 GET 路径由 SimpleHTTPRequestHandler 返回 404。"""
    handler = _CaptureHandler("/api/definitely_not_a_route")
    handler.do_GET()
    assert handler.sent == []
    assert handler.errors == [404]


def test_health_alias_contract() -> None:
    """GET /api/health 当前未注册，按实际契约返回 404。

    健康检查的真实 JSON 路由是 /api/v1/health；状态/uptime 字段由
    /api/system/status 提供。这里把旧路径现状固定下来，避免前端误依赖。
    """
    handler = _CaptureHandler("/api/health")
    handler.do_GET()
    assert handler.sent == []
    assert handler.errors == [404]


def test_health_v1_contract() -> None:
    """当前 server 健康检查暴露为 /api/v1/health，而非 /api/health。"""
    payload, status = _get("/api/v1/health")
    assert status == 200
    assert payload["ok"] is True
    assert isinstance(payload.get("service"), str)
    assert isinstance(payload.get("time"), str)
    assert isinstance(payload.get("datasets"), list)
    assert isinstance(payload.get("tasks"), dict)
    assert isinstance(payload.get("freshness"), list)
    assert isinstance(payload.get("audit_log_bytes"), int)


def test_system_status_contract() -> None:
    """GET /api/system/status 提供前端状态卡片依赖的 ok/uptime 等字段。"""
    payload, status = _get("/api/system/status")
    assert status == 200
    assert payload["ok"] is True
    assert isinstance(payload.get("server_alive"), bool)
    assert isinstance(payload.get("server_time"), str)
    assert isinstance(payload.get("uptime"), str)
    assert isinstance(payload.get("cron_jobs"), list)


def test_market_degrades_cleanly_via_regime(monkeypatch) -> None:
    """load_market_state 抛异常时，market/regime handler 返回 ok:false + error。

    /api/market 本身当前没有 try/except，因此降级契约用有错误包装的
    /api/market/regime 验证；它同样直接调用 load_market_state。
    """
    monkeypatch.setattr(server, "_cache_get", lambda key, ttl: None)

    def boom() -> dict:
        raise RuntimeError("market data source down")

    monkeypatch.setattr(server, "load_market_state", boom)
    payload, status = _get("/api/market/regime")
    assert status == 500
    assert payload["ok"] is False
    assert isinstance(payload.get("error"), str)


def test_market_route_does_not_silently_fabricate_ok(monkeypatch) -> None:
    """GET /api/market 当前不包错误；异常应向上抛而不是伪造成功响应。

    这是现状契约：避免将来把崩溃静默成 {"ok": true} 的假成功。
    """
    def boom() -> dict:
        raise RuntimeError("market data source down")

    monkeypatch.setattr(server, "load_market_state", boom)
    handler = _CaptureHandler("/api/market")
    with pytest.raises(RuntimeError, match="market data source down"):
        handler.do_GET()
    assert handler.sent == []


# ── V12.3 决策链 + RAG 知识检索契约 ──

_CHAIN_INTRADAY_FIXTURE = {
    "time": "2026-01-01 10:00:00",
    "bias": {"tone": "进攻", "advance": 3200, "decline": 800,
             "breadth": 0.8, "index_pct": 0.62, "summary": "广度健康"},
    "rows": [
        {"symbol": "002714", "name": "牧原股份", "price": 42.5, "change_pct": 2.1,
         "pe": 9.0, "pb": 2.0, "dominant": "value", "dominant_label": "价值低估",
         "summary": "PE 低于行业中枢", "composite_conf": 0.85},
    ],
    "n_scanned": 12, "n_opportunities": 1,
}

_CHAIN_AFTER_CLOSE_FIXTURE = {
    "date": "2026-01-01",
    "lhb": {"date": "2026-01-01", "broker_top": [
                {"name": "招商证券深圳南山南油大道营业部", "buy": 4.68, "net": 1.2,
                 "stocks": "胜宏科技", "day": "2026-01-01"}],
            "stock_top": [
                {"code": "300476", "name": "胜宏科技", "net": 1.35, "pct": 9.98,
                 "reason": "日涨幅偏离值达7%", "fwd1": 3.2, "day": "2026-01-01"}]},
    "value_picks": [
        {"code": "600015", "name": "华夏银行", "price": 6.61, "pe": 3.9, "pb": 0.33,
         "roe": 18.7, "dist52w": 10.4, "score": 76.1},
    ],
    "built_at": "2026-01-01 18:55:00",
}


def _write_generated(tmp_path, name: str, payload: dict) -> None:
    gen = tmp_path / "generated"
    gen.mkdir(exist_ok=True)
    (gen / name).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_decision_chain_intraday_contract(monkeypatch, tmp_path) -> None:
    """GET /api/decision_chain?mode=intraday 返回 content.bias/rows 契约。

    依赖 generated/intraday_chain_{date}.json；用 tmp ROOT + 固定 date 隔离。
    """
    monkeypatch.setattr(server, "ROOT", tmp_path)
    _write_generated(tmp_path, "intraday_chain_2026-01-01.json", _CHAIN_INTRADAY_FIXTURE)
    payload, status = _get("/api/decision_chain?mode=intraday&date=2026-01-01")
    assert status == 200
    assert payload["ok"] is True
    data = payload["data"]
    assert data["mode"] == "intraday"
    assert data["date"] == "2026-01-01"
    content = data["content"]
    assert content["bias"]["tone"] == "进攻"
    assert content["bias"]["breadth"] == 0.8
    rows = content["rows"]
    assert isinstance(rows, list) and rows
    row = rows[0]
    assert row["symbol"] == "002714"
    assert row["dominant_label"] == "价值低估"
    assert isinstance(row["composite_conf"], float)


def test_decision_chain_after_close_contract(monkeypatch, tmp_path) -> None:
    """GET /api/decision_chain?mode=after_close 返回龙虎榜 + 低估池契约。"""
    monkeypatch.setattr(server, "ROOT", tmp_path)
    _write_generated(tmp_path, "after_close_extra_2026-01-01.json", _CHAIN_AFTER_CLOSE_FIXTURE)
    payload, status = _get("/api/decision_chain?mode=after_close&date=2026-01-01")
    assert status == 200
    assert payload["ok"] is True
    content = payload["data"]["content"]
    lhb = content["lhb"]
    assert lhb["broker_top"][0]["buy"] == 4.68
    st = lhb["stock_top"][0]
    assert st["code"] == "300476" and st["net"] == 1.35
    vp = content["value_picks"][0]
    assert vp["pe"] == 3.9 and vp["score"] == 76.1
    assert vp["dist52w"] == 10.4


def test_decision_chain_missing_artifact_contract(monkeypatch, tmp_path) -> None:
    """产物缺失时返回 ok:true + data.error（前端渲染为'产物缺失'）。"""
    monkeypatch.setattr(server, "ROOT", tmp_path)
    payload, status = _get("/api/decision_chain?mode=intraday&date=2026-01-02")
    assert status == 200
    assert payload["ok"] is True
    assert "产物缺失" in payload["data"].get("error", "")


def test_rag_search_contract(monkeypatch) -> None:
    """GET /api/rag_search?q= 返回 hits 列表（file/cat/score/text 契约）。"""
    monkeypatch.setattr(server, "_cache_get", lambda key, ttl: None)

    def fake_search(q, k=5):
        return [
            {"file": "skills/trading-mastery/游资心法/小鳄鱼/深度研读报告.md",
             "cat": "游资心法库(26位)", "score": 0.565,
             "text": "低吸操作规范：首板次日分时图调整后拐头向上时买入"},
        ]

    import quant_system.analysis_core as _core
    import quant_system.analysis_core.knowledge_rag  # noqa: F401 - 绑定包属性供 monkeypatch
    fake_rag = SimpleNamespace(search=fake_search)
    # from-import 优先查 sys.modules, getattr 路径查包属性 → 两处都替换
    monkeypatch.setitem(sys.modules, "quant_system.analysis_core.knowledge_rag", fake_rag)
    monkeypatch.setattr(_core, "knowledge_rag", fake_rag)
    payload, status = _get("/api/rag_search?q=低吸")
    assert status == 200
    assert payload["ok"] is True
    hits = payload.get("hits")
    assert isinstance(hits, list) and hits
    hit = hits[0]
    assert isinstance(hit["file"], str) and "trading-mastery" in hit["file"]
    assert isinstance(hit["cat"], str)
    assert isinstance(hit["score"], (int, float))
    assert "低吸" in hit["text"]


def test_rag_search_requires_query(monkeypatch) -> None:
    """GET /api/rag_search 缺 q 时返回 ok:false + error。"""
    monkeypatch.setattr(server, "_cache_get", lambda key, ttl: None)
    payload, status = _get("/api/rag_search")
    assert status == 200
    assert payload["ok"] is False
    assert "q 必填" in payload.get("error", "")


def test_stock_lens_contract(monkeypatch) -> None:
    """GET /api/stock_lens?code= 返回短线/中线双视角全景契约。"""
    monkeypatch.setattr(server, "_cache_get", lambda key, ttl: None)

    def fake_analyze(code, date=None):
        return {"code": code, "name": "测试", "as_of": "2026-08-14",
                "synth": {"verdict": "短线驱动(情绪)", "tone": "进攻偏谨慎",
                          "advice": "快进快出", "short_score": 78.0, "mid_score": 27.0},
                "short": {"score": 78.0, "parts": [{"key": "涨停基因", "score": 30, "max": 30,
                                                   "data": {"zt_30d": 3, "max_board": 3}}],
                          "reasons": ["近30日涨停3次"], "data_sources": {"a": "b"}},
                "mid": {"score": 27.0, "parts": [{"key": "融资盘", "score": 0, "max": 30,
                                                  "data": {"has_data": False}}],
                        "reasons": [], "data_sources": {"c": "d"}}}

    import quant_system.analysis_core as _core2
    import quant_system.analysis_core.stock_lens  # noqa: F401 - 绑定包属性供 monkeypatch
    fake_lens = SimpleNamespace(analyze=fake_analyze)
    monkeypatch.setitem(sys.modules, "quant_system.analysis_core.stock_lens", fake_lens)
    monkeypatch.setattr(_core2, "stock_lens", fake_lens)
    payload, status = _get("/api/stock_lens?code=002172")
    assert status == 200
    assert payload["ok"] is True
    d = payload["data"]
    assert d["code"] == "002172"
    assert d["synth"]["short_score"] == 78.0
    assert d["synth"]["verdict"] == "短线驱动(情绪)"
    assert d["short"]["parts"][0]["key"] == "涨停基因"
    assert isinstance(d["mid"]["parts"], list)


def test_stock_lens_requires_code(monkeypatch) -> None:
    """GET /api/stock_lens 缺 code 时返回 ok:false + error。"""
    monkeypatch.setattr(server, "_cache_get", lambda key, ttl: None)
    payload, status = _get("/api/stock_lens")
    assert status == 200
    assert payload["ok"] is False
    assert "code 必填" in payload.get("error", "")


def test_enrich_backtest_contract():
    """回测增强: 迷你曲线 → drawdown/monthly 生成, 不抛异常。"""
    eq = [{"date": "2024-01-02", "equity": 100000.0},
          {"date": "2024-02-01", "equity": 95000.0},
          {"date": "2024-03-01", "equity": 110000.0}]
    r = server._enrich_backtest_with_benchmark({"equity_curve": eq})
    dd = r.get("drawdown_curve") or []
    assert len(dd) == 3
    assert dd[0] == 0.0 and dd[1] < 0  # 首日0, 次日回撤
    assert isinstance(r.get("monthly_returns"), list)


def test_enrich_backtest_benchmark(monkeypatch, tmp_path):
    """本地沪深300指数文件 → benchmark_curve + metrics 生成。"""
    import pandas as _pd
    # 构造 30 个交易日的指数
    dates = _pd.date_range("2024-01-01", periods=30, freq="B")
    idx_df = _pd.DataFrame({"date": dates, "close": [3000 + i * 10 for i in range(30)]})
    market = tmp_path / "data_warehouse" / "market"
    market.mkdir(parents=True)
    idx_df.to_parquet(market / "index_daily_沪深300.parquet", index=False)
    monkeypatch.setattr(server, "ROOT", tmp_path)
    eq = [{"date": dates[0].strftime("%Y-%m-%d"), "equity": 100000.0},
          {"date": dates[-1].strftime("%Y-%m-%d"), "equity": 120000.0}]
    r = server._enrich_backtest_with_benchmark({"equity_curve": eq})
    bc = r.get("benchmark_curve") or []
    assert len(bc) >= 20
    assert bc[0]["nav"] == pytest.approx(1.0, abs=0.02)
    bm = r.get("benchmark_metrics") or {}
    assert set(bm) >= {"alpha", "beta", "information_ratio"}
