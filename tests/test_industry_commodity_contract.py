from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from quant_system import industry_commodity as ic


def _series(latest: date, prices: list[float]) -> pd.DataFrame:
    dates = [pd.Timestamp(latest - timedelta(days=len(prices) - i - 1)) for i in range(len(prices))]
    return pd.DataFrame({"date": dates, "price": prices})


def test_muyuan_mapping_covers_product_and_feed_costs():
    taxonomy = ic.resolve_stock_taxonomy("002714")
    assert taxonomy["profile"] == "养殖业"
    specs = {row["tag"]: row for row in ic.INDUSTRY_COMMODITY_MAP["养殖业"]}
    assert set(specs) == {"lh", "meal", "corn", "soybean"}
    assert specs["lh"]["direction"] == 1
    assert specs["meal"]["direction"] == -1
    assert specs["corn"]["direction"] == -1
    assert ic.COMMODITY_SPECS["lh"]["contract"] == "LH0"
    assert ic.COMMODITY_SPECS["meal"]["contract"] == "M0"
    assert ic.COMMODITY_SPECS["corn"]["contract"] == "C0"


def test_soybean_is_display_only_when_meal_is_fresh(monkeypatch):
    latest = date(2026, 8, 27)
    prices = [100.0] * 20 + [110.0]
    monkeypatch.setattr(ic, "_load_series", lambda tag: _series(latest, prices))
    result = ic.build_industry_commodity_evidence("002714", reference_date=latest)
    items = {row["tag"]: row for row in result["items"]}
    assert result["status"] == "available"
    assert items["meal"]["included_in_score"] is True
    assert items["soybean"]["status"] == "substitute_only"
    assert items["soybean"]["included_in_score"] is False
    assert result["total_weight"] == 2.5
    assert result["available_weight"] == 2.5


def test_soybean_can_substitute_when_meal_is_missing(monkeypatch):
    latest = date(2026, 8, 27)
    prices = [100.0] * 20 + [110.0]
    monkeypatch.setattr(
        ic,
        "_load_series",
        lambda tag: None if tag == "meal" else _series(latest, prices),
    )
    result = ic.build_industry_commodity_evidence("002714", reference_date=latest)
    items = {row["tag"]: row for row in result["items"]}
    assert items["meal"]["status"] == "missing"
    assert items["soybean"]["status"] == "available"
    assert items["soybean"]["included_in_score"] is True
    assert "豆粕" in result["missing_items"]


def test_company_overrides_cover_key_commodity_stocks():
    assert ic.resolve_stock_taxonomy("000895")["profile"] == "肉制品"
    assert ic.resolve_stock_taxonomy("002460")["profile"] == "能源金属"
    assert ic.resolve_stock_taxonomy("002466")["profile"] == "能源金属"
    assert ic.resolve_stock_taxonomy("601899")["profile"] == "金铜资源"
    assert ic.resolve_stock_taxonomy("603993")["profile"] == "铜钼钴资源"
    assert ic.INDUSTRY_COMMODITY_MAP["肉制品"][0]["tag"] == "lh"
    assert ic.INDUSTRY_COMMODITY_MAP["能源金属"][0]["tag"] == "lithium"


def test_company_templates_do_not_fall_back_to_broad_proxies(monkeypatch):
    latest = date(2026, 8, 27)
    prices = [100.0] * 20 + [110.0]
    monkeypatch.setattr(ic, "_load_series", lambda tag: _series(latest, prices) if tag == "copper" else None)
    lithium = ic.build_industry_commodity_evidence("002460", reference_date=latest)
    assert {item["tag"] for item in lithium["items"]} == {"lithium"}
    metals = ic.build_industry_commodity_evidence("603993", reference_date=latest)
    assert {item["tag"] for item in metals["items"]} == {"copper", "molybdenum", "cobalt"}
    assert {item["status"] for item in metals["items"] if item["tag"] in {"molybdenum", "cobalt"}} == {"not_observable"}
    assert "铝" not in [item["name"] for item in metals["items"]]


def test_collector_uses_shared_contract_registry():
    from scripts import update_industry_commodities as collector
    assert set(collector.CONTRACTS) == set(ic.COLLECTOR_TAGS)
    assert all(collector.CONTRACTS[tag]["symbol"] == ic.COMMODITY_SPECS[tag]["contract"] for tag in collector.CONTRACTS)


def test_advice_contains_each_available_commodity_once():
    from quant_web.stock_analysis import _make_advice
    profile = {
        "code": "002714", "name": "牧原股份",
        "score": {"total": 60, "coverage": {"available_dimensions": 1, "total_dimensions": 6}},
        "trend": {}, "valuation": {}, "financial": {}, "liquidity": {}, "events": [],
        "commodity": {"status": "available", "score": 60, "coverage": 1.0, "items": [
            {"name": "豆粕", "status": "available", "direction": -1, "return_20d_pct": 5.0, "weight": 0.75},
        ]},
    }
    reasons = _make_advice(profile)["reasons"]
    assert sum("豆粕近20日" in reason for reason in reasons) == 1
