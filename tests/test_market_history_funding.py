from quant_web.market_history import get_market_history, get_market_history_catalog


def test_margin_market_history_is_exposed():
    result = get_market_history("margin_market", days=40)
    assert result["ok"] is True
    assert result["count"] > 1000
    assert {"sh_balance_yi", "sz_balance_yi", "close"}.issubset(result["columns"])
    assert result["start"] <= result["as_of"]


def test_market_fund_flow_history_is_exposed():
    result = get_market_history("market_fund_flow", days=40)
    assert result["ok"] is True
    assert result["count"] > 30
    assert {"main_net", "super_net", "close"}.issubset(result["columns"])


def test_history_catalog_lists_funding_assets_without_remote_fetch():
    catalog = get_market_history_catalog(refresh=True)
    items = {item["asset"]: item for item in catalog["items"]}
    assert catalog["catalog_refresh"] == "local_status_only"
    assert items["margin_market"]["status"] == "available"
    assert items["market_fund_flow"]["status"] == "available"
