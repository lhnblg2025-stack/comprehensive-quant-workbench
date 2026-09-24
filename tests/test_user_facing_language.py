from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "quant_web" / "static"


def test_default_product_surfaces_use_business_language():
    index = (STATIC / "index.html").read_text(encoding="utf-8")
    report = (STATIC / "research_report.html").read_text(encoding="utf-8")
    portfolio = (STATIC / "portfolio_dashboard.html").read_text(encoding="utf-8")
    images = (STATIC / "research_images.html").read_text(encoding="utf-8")
    research = (STATIC / "research_dashboard.html").read_text(encoding="utf-8")
    weekly = (STATIC / "market_weekly.html").read_text(encoding="utf-8")
    assert "V4 原版组合仪表盘" not in index
    assert "SSE 实时推送" not in index
    assert "DataStore.freshness" not in index
    assert "研究方法依据" in index
    assert "统一研究口径" in report
    assert "高级审计信息" in report
    assert "research_report.v1" not in report
    assert "查看完整报告" in report
    assert "VaR95" not in portfolio
    assert "HRP" not in portfolio
    assert "IMA 图片" not in images
    assert "搜索主题、图像文字或分析摘要" in images
    assert "V4" not in portfolio and "V4" not in research
    assert "IC / IF / IM / IH 基差" not in weekly
    assert "可信度" in research
    assert "判断依据" in research


def test_subnavigation_triggers_real_loaders():
    app = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "'st-backtest': 'runBacktest'" in app
    assert "'st-optimize': 'runOptimize'" in app
    assert "'o-chat': 'focusChatInput'" in app
    assert "'s-ai': 'loadStockAiAnalysis'" in app
