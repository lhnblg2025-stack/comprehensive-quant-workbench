from __future__ import annotations

from pathlib import Path

from quant_web.api_catalog import build_catalog

ROOT = Path(__file__).resolve().parents[1]
SERVER = (ROOT / "quant_web" / "server.py").read_text(encoding="utf-8")
REPORT_PAGE = (ROOT / "quant_web" / "static" / "research_report.html").read_text(encoding="utf-8")


def test_v4_aliases_share_one_complete_workflow():
    assert 'parsed.path in ("/v4.html", "/v4_dashboard.html")' in SERVER or 'parsed.path in ("/v4.html", "/portfolio_dashboard.html")' in SERVER
    assert "portfolio_dashboard.html" in SERVER
    page = (ROOT / "quant_web" / "static" / "portfolio_dashboard.html").read_text(encoding="utf-8")
    for marker in ("组合优化", "组合风控检查", "绩效统计", "因子归因", "纸面订单记录"):
        assert marker in page


def test_unknown_html_is_not_silently_rendered_as_home():
    assert '"页面入口不存在"' in SERVER
    assert 'status=404' in SERVER
    unknown_block = SERVER[SERVER.index("# 合法静态页仍由 SimpleHTTPRequestHandler"):SERVER.index("    def _serve_index_html")]
    assert "static_target.is_file()" in unknown_block
    assert "self._serve_index_html()" not in unknown_block


def test_unified_reader_keeps_user_facing_report_only():
    assert "openOriginalReport" in REPORT_PAGE
    assert "encodeURIComponent(date)" in REPORT_PAGE
    assert "查看来源" not in REPORT_PAGE
    assert "rawDetail" not in REPORT_PAGE
    assert "data_warehouse" not in REPORT_PAGE


def test_catalog_declares_strategy_and_unique_domain_ids():
    ids = [item["id"] for item in build_catalog()["domains"]]
    assert "strategy" in ids
    assert len(ids) == len(set(ids))
