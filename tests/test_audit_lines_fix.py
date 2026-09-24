# -*- coding: utf-8 -*-
"""quant_web/audit_lines_2109-3206.md 审计结果回归测试。

覆盖已确认问题：
  1. /api/ml_signals 失败时不再用缓存价格伪造 ml_score，而是如实返回 ok:false/failures
  2. /api/v4/portfolio/optimize 使用模拟收益时必须带 synthetic=true 标记
  3. /api/v4/performance/factor_attribution 常量占位必须显式标注
  4. /api/v4/health 异常时必须返回 ok:false，不再吞错
"""
from __future__ import annotations

import pathlib

import quant_web.server as sw
import quant_system.ml_signals as ms
import quant_system.trade_db as td
import scripts.system_manager as sm


def _stub_handler() -> sw.QuantHandler:
    h = sw.QuantHandler.__new__(sw.QuantHandler)
    h._sent: list = []
    h._send_json = lambda payload, status=200: h._sent.append((status, payload))
    return h


def test_ml_signals_no_fabricated_cache_estimate(monkeypatch):
    """复数 ML 端点失败时不伪造 ml_score，返回 ok:false 和 failures。"""
    monkeypatch.setattr(ms, "predict_stock", lambda code: (_ for _ in ()).throw(RuntimeError("boom")))
    h = _stub_handler()
    h._handle_ml_signals({"symbol": ["600519"]})
    status, payload = h._sent[0]
    assert payload["ok"] is False
    assert "ML预测失败" in payload["error"]
    assert payload["failures"][0]["symbol"] == "600519"
    assert "ml_score" not in payload
    assert not any("ml_score" in row for row in payload.get("failures", []))


def test_ml_signals_preserves_real_rows_and_reports_failures(monkeypatch):
    """部分成功时不丢弃成功行，失败进 failures，仍返回 ok:true。"""
    monkeypatch.setattr(ms, "predict_stock", lambda code: {
        "symbol": code,
        "ml_score": 0.8,
        "recommendation": "强",
        "signals": ["真实模型输出"],
        "top_contributors": [],
    } if code == "600519" else (_ for _ in ()).throw(RuntimeError("fail")))
    h = _stub_handler()
    h._handle_ml_signals({"symbols": ["600519,000858"]})
    status, payload = h._sent[0]
    assert payload["ok"] is True
    assert len(payload["data"]["rows"]) == 1
    assert payload["data"]["rows"][0]["symbol"] == "600519"
    assert len(payload["data"]["failures"]) == 1
    assert payload["data"]["failures"][0]["symbol"] == "000858"


def test_v4_portfolio_optimize_synthetic_flag(monkeypatch, tmp_path):
    """随机收益回退必须带 synthetic=true，前端可区分演示/真实。"""
    monkeypatch.setattr(sw, "STATIC_DIR", tmp_path)
    h = _stub_handler()
    h._handle_v4_portfolio_optimize({"method": ["risk_parity"]})
    status, payload = h._sent[0]
    assert payload["ok"] is True
    assert payload["data"]["synthetic"] is True
    assert "模拟" in payload["data"]["fallback"]


def test_v4_factor_attribution_placeholder_flag():
    """常量归因必须显式标注 placeholder，不得冒充真实计算。"""
    h = _stub_handler()
    h._handle_v4_factor_attribution()
    status, payload = h._sent[0]
    assert payload["ok"] is True
    data = payload["data"]
    assert data["is_placeholder"] is True
    assert data["mode"] == "placeholder"
    assert "待接入真实因子归因计算" in data["note"]


def test_v4_health_returns_ok_false_on_error(monkeypatch):
    """健康检查异常时 ok:false，前端才能感知失败而不是渲染 unknown。"""
    monkeypatch.setattr(sm, "check_system_health", lambda: (_ for _ in ()).throw(RuntimeError("health fail")))
    h = _stub_handler()
    h._handle_v4_health()
    status, payload = h._sent[0]
    assert payload["ok"] is False
    assert "系统健康检查失败" in payload["error"]


def test_sector_rotation_ok_false_when_all_sources_fail(monkeypatch):
    """行业板块数据源全部失败时返回 ok:false，不把空数据包装成成功。"""
    import types
    import sys

    fake_ak = types.ModuleType("akshare")
    def _bad_ak(*args, **kwargs):
        raise RuntimeError("ak fail")
    fake_ak.stock_board_industry_summary_ths = _bad_ak
    monkeypatch.setitem(sys.modules, "akshare", fake_ak)

    import requests
    def _bad_http(*args, **kwargs):
        raise RuntimeError("http fail")
    monkeypatch.setattr(requests, "get", _bad_http)

    h = _stub_handler()
    h._handle_sector_rotation({})
    status, payload = h._sent[0]
    assert payload["ok"] is False
    assert "行业板块数据不可用" in payload["error"]
