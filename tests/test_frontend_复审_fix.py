# -*- coding: utf-8 -*-
"""前端 quant_web 复审 P1 契约/字段断裂修复的回归测试（对应 6 项）。

覆盖:
  1. /api/news_sentiment 真正接 fetch_news，返回前端契约 total/headlines/sentiment，
     且数据来自新闻而非因子均值。
  2. /api/factor_ic 返回裸 factors/range/chart（前端 app.js:4011 读 d.factors）。
  3. /api/factor_combine 返回裸 method/ic_mean/icir/chart（前端 app.js:4041 读裸）。
  4. /api/reports?mode=read 顶层 content（前端 app.js:2664 读 d.content）。
  5. stock_analysis.build_stock_profile 市值由 kline 的 close*outstanding_share
     推导（不再读不存在的 total_mv 列，恒 0 已修）。
  6. macro_snapshot 取 M2 数量列(亿元) 而非 M0-环比增速，日期转正确格式。

说明：import quant_web.server 会拉起较重依赖（akshare/backtest 等），本机可用；
测试通过桩对象隔离网络/wsgi 调用。
"""
from __future__ import annotations

import types

import pytest

import quant_web.server as sw
import quant_web.stock_analysis as sa


# ---------------------------------------------------------------------------
# 工具：桩 handler（捕获 _send_json 的 payload，不真写 socket）
# ---------------------------------------------------------------------------

def _stub_handler() -> sw.QuantHandler:
    h = sw.QuantHandler.__new__(sw.QuantHandler)
    h._sent: list = []
    h._send_json = lambda payload, status=200: h._sent.append((status, payload))
    return h


@pytest.fixture
def stub_handler():
    return _stub_handler()


# ---------------------------------------------------------------------------
# 1. 新闻情绪
# ---------------------------------------------------------------------------

def test_news_sentiment_returns_total_and_headlines_from_news(monkeypatch):
    import quant_system.analysis_core.social_sentiment as _ss

    fake_news = {
        "ok": True, "source": "news_akshare", "n": 7,
        "bull": 3, "bear": 1, "neutral": 3, "bull_bear_ratio": 3.0,
        "sentiment": 0.4,
        "fetched": ["stock_news_em(600519:10)"],
        "top_bullish": ["标题A-利好", "标题B-利好", "标题C-利好"],
    }
    monkeypatch.setattr(_ss, "fetch_news", lambda symbol=None, limit=30: dict(fake_news, coverage="stock_symbol", source_symbol=symbol or "002714", records=[{"title": t} for t in fake_news["top_bullish"]]))

    h = _stub_handler()
    h._handle_news_sentiment({"symbol": ["002714"]})
    status, payload = h._sent[0]
    assert payload["ok"] is True
    data = payload["data"]
    # 契约字段必须存在
    assert "total" in data and data["total"] == 7           # 新闻条数，来自真实新闻
    assert "headlines" in data and data["headlines"]          # 新闻表，来自真实标题
    assert "sentiment" in data and data["sentiment"] == "positive"
    assert data["score"] == 0.4                               # 真实新闻情绪分
    # 来源必须是新闻，而不是 factor_model 冒充
    assert data["source"] == "news_akshare"
    assert all(hd["sentiment"] == "positive" for hd in data["headlines"])
    assert all(hd["title"].startswith("标题") for hd in data["headlines"])


def test_news_sentiment_ok_false_when_fetch_news_fails(monkeypatch):
    import quant_system.analysis_core.social_sentiment as _ss

    monkeypatch.setattr(_ss, "fetch_news", lambda symbol=None, limit=30: {"ok": False, "error": "新闻接口均不可用"})
    h = _stub_handler()
    h._handle_news_sentiment({"symbol": ["002714"]})
    status, payload = h._sent[0]
    assert payload["ok"] is False                      # 失败如实返回 ok:false，不冒充
    assert "error" in payload


# ---------------------------------------------------------------------------
# 2. factor_ic 裸返回
# ---------------------------------------------------------------------------

def test_factor_ic_returns_bare_keys(stub_handler, monkeypatch):
    rows = [{
        "name": "动量", "category": "Momentum", "ic_mean": 0.05, "icir": 0.4,
        "win_rate": 0.6, "n_days": 100, "coverage": 0.9, "grade": "A", "direction": 1,
    }]
    monkeypatch.setattr(stub_handler, "_load_ic_report", lambda: list(rows))
    stub_handler._handle_factor_ic({"start": ["20230101"], "end": [""]})
    status, payload = stub_handler._sent[0]
    assert payload["ok"] is True
    # 前端 app.js 读裸 d.factors / d.range / d.chart
    assert "factors" in payload and payload["factors"]
    assert "range" in payload
    assert "chart" in payload
    # 不应再嵌套在 data 下导致 d.factors 为 undefined
    assert "data" not in payload or "factors" not in payload.get("data", {})


def test_factor_ic_empty_returns_bare_factors(stub_handler, monkeypatch):
    monkeypatch.setattr(stub_handler, "_load_ic_report", lambda: [])
    stub_handler._handle_factor_ic({"start": ["20230101"], "end": [""]})
    status, payload = stub_handler._sent[0]
    assert payload["ok"] is True
    assert payload["factors"] == []
    assert payload["range"]
    assert payload["chart"] is None


# ---------------------------------------------------------------------------
# 3. factor_combine 裸返回
# ---------------------------------------------------------------------------

def test_factor_combine_returns_bare_keys(stub_handler, monkeypatch):
    rows = [{
        "name": f"F{i}", "category": "C", "ic_mean": 0.03 * (i + 1), "icir": 0.3,
        "win_rate": 0.6, "n_days": 100, "coverage": 0.9, "grade": "B", "direction": 1,
    } for i in range(5)]
    monkeypatch.setattr(stub_handler, "_load_ic_report", lambda: list(rows))
    stub_handler._handle_factor_combine({"method": ["equal"]})
    status, payload = stub_handler._sent[0]
    assert payload["ok"] is True
    # 前端 app.js 读裸 d.method / d.ic_mean / d.icir / d.chart
    assert "method" in payload and payload["method"] == "equal"
    assert "ic_mean" in payload
    assert "icir" in payload
    assert "chart" in payload
    assert "data" not in payload or "method" not in payload.get("data", {})


# ---------------------------------------------------------------------------
# 4. reports read 顶层 content
# ---------------------------------------------------------------------------

def test_reports_read_exposes_top_level_content(stub_handler, monkeypatch):
    monkeypatch.setattr(
        sw, "_read_report",
        lambda path: {"path": path, "name": "xx.md", "content": "报告正文正文"},
    )
    stub_handler._handle_reports({"mode": ["read"], "path": ["/fake/xx.md"]})
    status, payload = stub_handler._sent[0]
    assert payload["ok"] is True
    # 前端 readReport 读 d.content（顶层）
    assert payload["content"] == "报告正文正文"
    # 兼容旧嵌套 report.content
    assert payload["report"]["content"] == "报告正文正文"


# ---------------------------------------------------------------------------
# 5. 市值由 close*outstanding_share 推导（非 0）
# ---------------------------------------------------------------------------

def test_build_stock_profile_derives_float_mv_from_kline(monkeypatch):
    import pandas as pd
    from datetime import datetime

    # 假 kline：close * outstanding_share → 流通市值 > 0
    kline = pd.DataFrame({
        "date": [datetime(2026, 8, 12), datetime(2026, 8, 13)],
        "open": [40.0, 39.9], "high": [40.5, 40.4], "low": [39.8, 39.0],
        "close": [40.0, 39.3], "volume": [100, 120], "amount": [1e6, 1.2e6],
        "outstanding_share": [3.2e9, 3.2e9],
    })
    # 假 valuation，仅含真实 schema 列（无 total_mv/float_mv）。
    # 真实 load_valuation 会把 peTTM→pe_ttm / pbMRQ→pb / psTTM→ps 重命名。
    val = pd.DataFrame({
        "date": [datetime(2026, 8, 12), datetime(2026, 8, 13)],
        "pe_ttm": [24.0, 23.7], "pb": [2.7, 2.69],
        "ps": [1.7, 1.68], "pcf": [10.0, 9.9],
    })

    monkeypatch.setattr(sa, "load_kline", lambda code: kline)
    monkeypatch.setattr(sa, "load_valuation", lambda code: val)
    monkeypatch.setattr(sa, "load_financial", lambda code: None)
    monkeypatch.setattr(sa, "load_events", lambda code: [])
    monkeypatch.setattr(sa, "stock_name", lambda code: "测试股")

    p = sa.build_stock_profile("002714")
    v = p["valuation"]
    # 市值必须非 0 且来自 close*outstanding_share（约 39.3*3.2e9/1e8=1257.6 亿元量级）
    assert v["float_mv_yi"] and v["float_mv_yi"] > 0
    assert v["total_mv_yi"] == v["float_mv_yi"]
    assert 1000 <= v["total_mv_yi"] <= 1500, v["total_mv_yi"]
    # 老文档声称的 total_mv 列不存在，修复后不再读它（不会因缺列而报错/置 0）
    assert "close" not in val.columns


# ---------------------------------------------------------------------------
# 6. 宏观 M2 取 M2 数量列，非 M0-环比
# ---------------------------------------------------------------------------

def test_macro_snapshot_m2_takes_quantity_column_not_m0_mom(monkeypatch):
    import pandas as pd

    # 用真实列名，使测试与线上 schema 完全一致
    fake_m2 = pd.DataFrame({
        "月份": ["2026年06月份", "2026年05月份"],
        "货币和准货币(M2)-数量(亿元)": [3567108.43, 3536688.92],
        "货币和准货币(M2)-同比增长": [8.0, 8.6],
        "货币和准货币(M2)-环比增长": [0.860113, 0.177421],
        "货币(M1)-数量(亿元)": [1184775.53, 1148891.41],
        "货币(M1)-同比增长": [4.0, 5.5],
        "货币(M1)-环比增长": [3.123369, 0.266852],
        "流通中的现金(M0)-数量(亿元)": [147364.79, 146854.71],
        "流通中的现金(M0)-同比增长": [11.8, 11.9],
        "流通中的现金(M0)-环比增长": [0.347336, -0.422214],
    })
    monkeypatch.setattr(
        sa, "_read_parquet",
        lambda rel: fake_m2 if rel == "macro/m2_yearly.parquet" else None,
    )
    ms = sa.macro_snapshot()
    m2 = ms["m2"]
    # 取的是 M2 数量列（3567108.43 亿元），而非 M0-环比 0.347
    assert m2["value"] == 3567108.43
    assert m2["unit"] == "亿元"
    assert "M0" not in m2.get("column", "")
    assert "M2" in m2.get("column", "")
    # date 为正确的年月格式（"2026年06月份" → "2026-06"），不再是月份串
    assert m2["date"] == "2026-06"
