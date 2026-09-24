from __future__ import annotations

from pathlib import Path

from quant_web.api_catalog import build_catalog


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "quant_web" / "static"


# These are verified user journeys, not implementation-version names.  A refactor
# may change the renderer, but it may not silently remove a proven entry point.
LEGACY_AND_DEEP_PAGES = (
    "paper_trading.html",
    "trader.html",
    "intraday.html",
    "review.html",
    "v4.html",
    "report.html",
    "research_report.html",
    "market_weekly.html",
    "base_panorama.html",
    "research_images.html",
    "research_dashboard.html",
    "knowledge_base.html",
)


def test_verified_pages_and_compatibility_targets_remain_present():
    for page in LEGACY_AND_DEEP_PAGES:
        assert (STATIC / page).exists() or page in {"review.html", "v4.html", "report.html"}, page
    index = (STATIC / "index.html").read_text(encoding="utf-8")
    for href in (
        "/paper_trading.html", "/trader.html", "/intraday.html", "/review.html",
        "/v4.html", "/report.html", "/research_report.html", "/market_weekly.html",
        "/base_panorama.html", "/research_images.html", "/research_dashboard.html",
        "/knowledge_base.html",
    ):
        assert href in index, f"首页缺少已验证入口: {href}"
    for legacy_path in ("/api/v11/daily", "/api/v11/battle", "/api/v11/orders", "/api/v12/kline", "/api/v12/fund_flow", "/api/v3/fama_macbeth", "/api/v3/risk_model", "/api/v3/allocation", "/api/v5/"):
        assert legacy_path in (ROOT / "quant_web" / "server.py").read_text(encoding="utf-8")


def test_catalog_covers_deep_research_and_ops_capabilities():
    domains = {item["id"]: item for item in build_catalog()["domains"]}
    assert "/api/factor_ic" in domains["factors"]["deep_paths"]
    assert "/api/v4/portfolio/optimize" in domains["risk"]["deep_paths"]
    assert "/api/data_health" == domains["ops"]["read_path"]
    assert "/api/industry_chain_history" in domains["knowledge"]["deep_paths"]
    assert "/api/research_images" in domains["knowledge"]["deep_paths"]
    assert "v11/v12" in build_catalog()["routing_policy"]["legacy"]


def test_research_reader_exposes_all_contract_sections_and_raw_detail():
    page = (STATIC / "research_report.html").read_text(encoding="utf-8")
    for marker in ("章节结构", "资金与业务传导", "风险与行动条件", "证据链", "数据来源情况", "数据缺口"):
        assert marker in page
    assert "高级审计信息" in page and "分析依据" in page
    assert "report._detail=d.detail||null" in page
    for report_type in ("weekly", "daily_review", "stock", "industry", "commodity"):
        assert f'value="{report_type}"' in page


def test_home_workbench_keeps_density_and_truthfulness_signals():
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    for marker in ("统一决策工作台", "主线与资金", "候选与执行", "风险门控", "证据源状态", "数据日期与更新时间分别展示"):
        assert marker in page
    # Deep functions remain discoverable in the primary shell, not hidden by a
    # summary-only replacement.
    for marker in ("因子工坊", "回测", "组合", "数据基座", "研报图像证据"):
        assert marker in page
