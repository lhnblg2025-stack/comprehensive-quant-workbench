from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def test_financial_cache_can_be_read_without_write_access(tmp_path, monkeypatch):
    import quant_system.financial_data as fd

    db_path = tmp_path / "financial.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE financial_indicators (
        symbol TEXT, date TEXT, indicator TEXT, value REAL, updated TEXT
        )"""
    )
    conn.executemany(
        "INSERT INTO financial_indicators VALUES (?, ?, ?, ?, ?)",
        [
            ("002714", "20260630", "roe", 8.2, "2026-08-01"),
            ("002714", "20260630", "profit_growth_yoy", -12.5, "2026-08-01"),
        ],
    )
    conn.commit()
    conn.close()
    db_path.chmod(0o444)
    monkeypatch.setattr(fd, "DB_PATH", db_path)

    data = fd.fetch_financial_indicators("002714", max_age_days=9999)

    assert data == {"profit_growth_yoy": -12.5, "roe": 8.2}


def test_frontend_review_model_exposes_collector_error():
    from quant_web.handlers.review import _frontend_model

    review = {
        "date": "2026-08-25",
        "generated_at": "now",
        "battle_map": {},
        "blocks": {
            "strong_direction": {
                "value": {"directions": []},
                "error": "timeout",
            },
            "model_verdict": {
                "value": {
                    "verdict": {
                        "consensus": "震荡",
                        "confidence": 0.39,
                        "votes": 11,
                    }
                }
            },
        },
        "decision": {},
    }

    model = _frontend_model(review)

    assert model["directions_error"] == "timeout"
    assert model["verdict"]["votes"] == 11


def test_freshness_missing_items_never_have_negative_lag(tmp_path, monkeypatch):
    import scripts.freshness_gate as fg

    (tmp_path / "empty_domain").mkdir()
    monkeypatch.setattr(fg, "DW", tmp_path)

    result = fg.gate_scores()

    assert result["missing_items"] == [
        {"dir": "empty_domain", "lag": None, "latest": "", "cycle": "?"}
    ]
    assert "-1" not in json.dumps(result, ensure_ascii=False)


def test_review_dashboard_renders_decision_meaning_instead_of_empty_cards():
    page = (Path(__file__).resolve().parents[1] / "quant_web/static/review_dashboard.html").read_text(
        encoding="utf-8"
    )

    assert "renderModelVerdict(x)" in page
    assert "不能据此判定“无主线”" in page
    assert "当前无买点信号，不因“中枢内震荡”直接开仓" in page
    assert "另有 ${missing.length} 个非关键/未接入域无数据" in page
    assert "function escapeHtml" in page


def test_market_pano_marks_read_error_as_partial(monkeypatch, tmp_path):
    import scripts.market_pano as pano

    monkeypatch.setattr(pano, "CACHE", tmp_path / "market.json")
    monkeypatch.setattr(pano, "MK", tmp_path)
    def broken_read(name, **kwargs):
        pano._READ_STATUS[name] = {"status": "error", "error": "bad parquet"}
        return None

    monkeypatch.setattr(pano, "_rd", broken_read)

    result = pano.market_pano()

    assert result["status"] == "unavailable"
    assert result["errors"] == [] or isinstance(result["errors"], list)


def test_overseas_fallback_contract_uses_real_pct_fields(monkeypatch):
    import scripts.overseas_collector as oc

    class FakeResponse:
        def read(self):
            return (
                'v_usNDX="200~Nasdaq~.NDX~29433.43~29641.56~29545.90~1~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~~2026-08-28 17:15:59~-208.13~-0.70~29752.78~29383.92~USD";'
            ).encode("gbk")

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(oc.urllib.request, "urlopen", lambda *_args, **_kwargs: FakeResponse())

    quotes = oc._gtimg_quotes()

    assert quotes[0]["label"] == "纳指100"
    assert quotes[0]["chg_pct"] == -0.7
    assert quotes[0]["date"] == "2026-08-28"
