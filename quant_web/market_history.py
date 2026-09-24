"""Unified historical market data acquisition for the web UI.

Every response carries source, status, coverage and an explicit error. The
module persists fetched rows under data_warehouse/market_history and never
turns an unavailable source into zeroes.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "data_warehouse" / "market_history"

ASSETS: dict[str, dict[str, Any]] = {
    "a_share_etf": {"name": "沪深300ETF", "symbol": "510300", "kind": "etf_daily", "path": "events/etf_history.parquet", "source": "akshare.fund_etf_hist_em(510300,daily)"},
    "margin_market": {"name": "沪深两融余额", "symbol": "MARGIN", "kind": "margin_pair", "source": "data_warehouse/market/market_margin_{sh,sz}.parquet"},
    "market_fund_flow": {"name": "全市场主力资金", "symbol": "ALL_A", "kind": "market_flow", "path": "market/stock_market_fund_flow.parquet", "source": "data_warehouse/market/stock_market_fund_flow.parquet"},
    "sp500": {"name": "标普500", "symbol": "us^GSPC", "kind": "global", "source": "Yahoo Finance chart"},
    "us_global": {"name": "美股全球观察", "symbol": "usSPY", "kind": "global", "source": "Yahoo Finance chart"},
    "futures_basis": {"name": "股指期货基差", "symbol": "IF", "kind": "local", "path": "market/futures_basis.parquet", "source": "data_warehouse/market/futures_basis.parquet"},
    "bond_futures": {"name": "十年期国债期货", "symbol": "T0", "kind": "local", "path": "market/bond_futures.parquet", "source": "data_warehouse/market/bond_futures.parquet"},
    "gold": {"name": "黄金期货", "symbol": "AU0", "kind": "local", "path": "market/commodity__gold.parquet", "source": "data_warehouse/market/commodity__gold.parquet"},
    "crude": {"name": "原油期货", "symbol": "SC0", "kind": "local", "path": "market/commodity__crude.parquet", "source": "data_warehouse/market/commodity__crude.parquet"},
    "silver": {"name": "白银期货", "symbol": "AG0", "kind": "futures", "source": "akshare.futures_main_sina"},
    "lithium_carbonate": {"name": "碳酸锂期货", "symbol": "LC0", "kind": "futures", "source": "akshare.futures_main_sina"},
    "hog": {"name": "生猪期货", "symbol": "LH0", "kind": "local", "path": "market/commodity__lh.parquet", "source": "data_warehouse/market/commodity__lh.parquet"},
}


def _cache_path(asset: str) -> Path:
    return STORE / f"{asset}.parquet"


def _manifest_path(asset: str) -> Path:
    return STORE / f"{asset}.json"


def _safe_rows(frame: pd.DataFrame, limit: int = 800) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    out = frame.copy()
    date_col = next((c for c in ("date", "日期", "trade_date") if c in out.columns), None)
    if date_col:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    out = out.tail(limit)
    return json.loads(out.where(pd.notna(out), None).to_json(orient="records", force_ascii=False))


def _standardize(frame: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = frame.copy()
    rename = {"日期": "date", "交易日期": "date", "收盘价": "close", "开盘价": "open", "最高价": "high", "最低价": "low",
              "成交量": "volume", "成交额": "amount", "收盘": "close", "开盘": "open", "最高": "high", "最低": "low"}
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    if date_col not in out.columns and "date" not in out.columns:
        return pd.DataFrame()
    if date_col != "date" and date_col in out.columns:
        out = out.rename(columns={date_col: "date"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    for col in ("open", "high", "low", "close", "volume", "amount", "ret"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "close" in out.columns:
        out["ret"] = out["close"].pct_change()
    return out.reset_index(drop=True)


def _read_margin_pair() -> pd.DataFrame:
    parts = []
    for market, name in (("sh", "sh_balance_yi"), ("sz", "sz_balance_yi")):
        path = ROOT / "data_warehouse" / "market" / f"market_margin_{market}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if frame.empty or "日期" not in frame.columns or "融资融券余额" not in frame.columns:
            continue
        item = frame[["日期", "融资融券余额"]].rename(columns={"日期": "date", "融资融券余额": name})
        item["date"] = pd.to_datetime(item["date"], errors="coerce")
        item[name] = pd.to_numeric(item[name], errors="coerce") / 1e8
        parts.append(item.dropna(subset=["date"]).drop_duplicates("date", keep="last"))
    if not parts:
        return pd.DataFrame()
    result = parts[0]
    for item in parts[1:]:
        result = result.merge(item, on="date", how="outer")
    result = result.sort_values("date")
    # 单边文件缺失时 get() 会返回整数 0，后续 .fillna 抛 AttributeError；
    # 缺失侧必须以 NaN 序列占位，汇总仍可计算且不伪装成 0 余额。
    import numpy as _np
    for col in ("sh_balance_yi", "sz_balance_yi"):
        if col not in result.columns:
            result[col] = _np.nan
    result["close"] = result["sh_balance_yi"].fillna(0) + result["sz_balance_yi"].fillna(0)
    result["margin_side_missing"] = result["sh_balance_yi"].isna() | result["sz_balance_yi"].isna()
    result["change_yi"] = result["close"].diff()
    return result.reset_index(drop=True)


def _read_market_fund_flow() -> pd.DataFrame:
    path = ROOT / "data_warehouse" / "market" / "stock_market_fund_flow.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    rename = {
        "日期": "date", "主力净流入-净额": "main_net", "主力净流入-净占比": "main_net_pct",
        "超大单净流入-净额": "super_net", "大单净流入-净额": "large_net",
        "中单净流入-净额": "medium_net", "小单净流入-净额": "small_net",
        "上证-收盘价": "sh_close", "上证-涨跌幅": "sh_pct", "深证-收盘价": "sz_close", "深证-涨跌幅": "sz_pct",
    }
    out = frame.rename(columns={key: value for key, value in rename.items() if key in frame.columns}).copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for column in ("main_net", "super_net", "large_net", "medium_net", "small_net"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce") / 1e8
    out["close"] = out.get("main_net")
    return out.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _read_local(asset: str, spec: dict[str, Any]) -> pd.DataFrame:
    path = ROOT / "data_warehouse" / spec["path"]
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if asset == "futures_basis" and "contract" in frame.columns:
        frame = frame[frame["contract"].astype(str) == str(spec["symbol"])]
    return _standardize(frame)


def _fetch(asset: str, spec: dict[str, Any], days: int) -> tuple[pd.DataFrame, str]:
    if spec["kind"] == "margin_pair":
        return _read_margin_pair(), spec["source"]
    if spec["kind"] == "market_flow":
        return _read_market_fund_flow(), spec["source"]
    if spec["kind"] == "local":
        return _read_local(asset, spec), spec["source"]
    cached = _cache_path(asset)
    if cached.exists():
        try:
            frame = _standardize(pd.read_parquet(cached))
            if len(frame) >= min(days, 30):
                return frame, spec.get("source", "local cache")
        except Exception:
            pass
    if spec["kind"] in ("global", "etf"):
        if spec["kind"] == "global":
            from quant_system.global_market import fetch_global_kline
            frame = fetch_global_kline(spec["symbol"], period="daily", count=max(days, 1200))
            source = spec["source"]
        else:
            import akshare as ak
            frame = ak.fund_etf_hist_em(symbol=spec["symbol"], period="daily", start_date="20100101", end_date=datetime.now().strftime("%Y%m%d"), adjust="")
            source = spec["source"]
        frame = _standardize(frame)
    elif spec["kind"] == "etf_daily":
        # 全量日线为主源；events/etf_history.parquet 稀疏采样只做兜底。
        frame = pd.DataFrame()
        source = spec["source"]
        try:
            import akshare as ak
            frame = ak.fund_etf_hist_em(symbol=spec["symbol"], period="daily", start_date="20100101", end_date=datetime.now().strftime("%Y%m%d"), adjust="")
        except Exception:
            frame = pd.DataFrame()
        if frame.empty:
            try:
                import akshare as ak
                prefix = "sh" if str(spec["symbol"]).startswith(("5", "6", "9")) else "sz"
                frame = ak.fund_etf_hist_sina(symbol=f"{prefix}{spec['symbol']}")
                source = "akshare.fund_etf_hist_sina"
            except Exception:
                frame = pd.DataFrame()
        frame = _standardize(frame)
        if frame.empty:
            frame = _read_local(asset, {"path": spec["path"], "symbol": spec["symbol"]})
            if "code" in frame.columns:
                frame = frame[frame["code"].astype(str).str.zfill(6) == str(spec["symbol"])]
            source = spec["path"]
    elif spec["kind"] == "local_or_futures":
        frame = _read_local(asset, spec)
        source = spec["source"]
    else:
        import akshare as ak
        frame = ak.futures_main_sina(symbol=spec["symbol"])
        source = spec["source"]
        frame = _standardize(frame)
    if frame.empty:
        return frame, source
    STORE.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cached, index=False)
    _manifest_path(asset).write_text(json.dumps({
        "asset": asset, "symbol": spec["symbol"], "source": source,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(frame), "start": str(frame["date"].min().date()), "end": str(frame["date"].max().date()),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return frame, source


def get_market_history(asset: str, days: int = 400, refresh: bool = False) -> dict[str, Any]:
    if asset not in ASSETS:
        return {"ok": False, "status": "invalid_asset", "error": f"不支持的市场资产: {asset}", "assets": ASSETS}
    spec = ASSETS[asset]
    try:
        if refresh and _cache_path(asset).exists():
            _cache_path(asset).unlink()
        frame, source = _fetch(asset, spec, max(20, min(days, 2000)))
        if frame.empty:
            return {"ok": False, "status": "missing", "asset": asset, "name": spec["name"], "symbol": spec["symbol"], "source": source, "rows": [], "error": "历史数据为空；请刷新采集或检查数据源"}
        rows = _safe_rows(frame, max(20, min(days, 800)))
        return {
            "ok": True, "status": "available", "asset": asset, "name": spec["name"], "symbol": spec["symbol"],
            "source": source, "as_of": str(frame["date"].max().date()), "start": str(frame["date"].min().date()),
            "count": len(frame), "display_count": len(rows), "rows": rows,
            "columns": list(frame.columns), "data_contract": "history.v1",
        }
    except Exception as exc:
        return {"ok": False, "status": "error", "asset": asset, "name": spec["name"], "symbol": spec["symbol"], "source": spec.get("source"), "rows": [], "error": str(exc)[:240]}


def get_market_history_catalog(refresh: bool = False) -> dict[str, Any]:
    """List local coverage only; bulk refresh belongs to explicit per-asset actions.

    A catalog refresh previously forced every remote source synchronously, blocking
    the UI for minutes and making a status action destructive.
    """
    items = []
    for asset, spec in ASSETS.items():
        cache = _cache_path(asset)
        manifest = _manifest_path(asset)
        frame = pd.DataFrame()
        error = None
        if cache.exists():
            try:
                frame = _standardize(pd.read_parquet(cache))
            except Exception as exc:
                error = f"缓存读取失败: {str(exc)[:160]}"
        if frame.empty and spec.get("kind") in {"local", "margin_pair", "market_flow"}:
            try:
                if spec.get("kind") == "margin_pair":
                    frame = _read_margin_pair()
                elif spec.get("kind") == "market_flow":
                    frame = _read_market_fund_flow()
                else:
                    frame = _read_local(asset, spec)
            except Exception as exc:
                error = f"本地数据读取失败: {str(exc)[:160]}"
        status = "available" if not frame.empty else ("error" if error else "missing")
        items.append({
            "asset": asset, "name": spec["name"], "symbol": spec["symbol"], "status": status,
            "source": spec.get("source", ""),
            "as_of": str(frame["date"].max().date()) if not frame.empty else None,
            "start": str(frame["date"].min().date()) if not frame.empty else None,
            "count": len(frame), "error": error,
            "cached": cache.exists(), "manifest": manifest.name if manifest.exists() else None,
        })
    return {"ok": True, "status": "available", "data_contract": "history.v1", "items": items,
            "catalog_refresh": "local_status_only", "note": "单资产抓取由各行按钮触发，目录刷新不执行远程批量抓取"}
