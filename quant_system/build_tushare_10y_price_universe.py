"""Build a clean, resumable 800-stock eight-year RAW/HFQ price universe via Tushare.

Only price, adjustment, liquidity and security-master facts are materialized
here. PIT financials and exchange trade-state history remain separate required
inputs and are never synthesized from this dataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from quant_system.research_protocol import DEFAULT_PROTOCOL


def _token() -> str:
    from scripts.secret_loader import get_secret
    token = get_secret("TUSHARE_API_KEY")
    if not token:
        raise RuntimeError("tushare_token_unavailable")
    return token


def _balanced(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    data = frame.copy()
    data["symbol"] = data["symbol"].astype(str).str.zfill(6)
    data = data[~data["symbol"].str.startswith(("4", "8"))].copy()
    data["bucket"] = "main"
    data.loc[data["symbol"].str.startswith("30"), "bucket"] = "growth"
    data.loc[data["symbol"].str.startswith("68"), "bucket"] = "star"
    quotas = {"main": int(count * .60), "growth": int(count * .25), "star": count - int(count * .60) - int(count * .25)}
    selected = [data[data["bucket"] == name].sort_values("symbol").head(quota) for name, quota in quotas.items()]
    out = pd.concat(selected, ignore_index=True)
    if len(out) < count:
        out = pd.concat([out, data[~data["symbol"].isin(out["symbol"])].sort_values("symbol").head(count - len(out))], ignore_index=True)
    return out.head(count).drop(columns="bucket")


def select_universe(pro, *, start: str, count: int) -> pd.DataFrame:
    fields = "ts_code,symbol,name,industry,market,list_date,delist_date"
    try:
        listed = pro.stock_basic(exchange="", list_status="L", fields=fields)
        delisted = pro.stock_basic(exchange="", list_status="D", fields=fields)
        all_stocks = pd.concat([listed, delisted], ignore_index=True).drop_duplicates("ts_code")
        all_stocks["list_date"] = pd.to_datetime(all_stocks["list_date"], errors="coerce")
        all_stocks["delist_date"] = pd.to_datetime(all_stocks["delist_date"], errors="coerce")
        eligible = all_stocks[all_stocks["list_date"].le(pd.Timestamp(start))].copy()
        selected = _balanced(eligible, count)
        selected["universe_source"] = "tushare_stock_basic_snapshot"
        return selected
    except Exception:
        # stock_basic is rate-limited on some tokens. Use only the local code
        # inventory to continue price collection; this fallback cannot unlock
        # listing/delisting PIT and is marked as such in the artifact.
        codes = [path.stem.zfill(6) for path in Path(__file__).resolve().parents[1].joinpath("data_warehouse", "kline").glob("*.parquet")]
        data = pd.DataFrame({"symbol": sorted(set(codes) - {"000300", "000905", "000852"})})
        data["ts_code"] = data["symbol"].map(lambda s: f"{s}.SH" if s.startswith(("5", "6", "9")) else f"{s}.SZ")
        data["name"] = pd.NA; data["industry"] = pd.NA; data["market"] = pd.NA; data["list_date"] = pd.NaT; data["delist_date"] = pd.NaT
        selected = _balanced(data, count)
        selected["universe_source"] = "code_fallback_no_listing_dates"
        return selected


def fetch_symbol(pro, ts_code: str, *, start: str, end: str, with_adjustment: bool = False) -> pd.DataFrame:
    daily = pro.daily(ts_code=ts_code, start_date=start.replace("-", ""), end_date=end.replace("-", ""))
    if daily.empty:
        return pd.DataFrame()
    out = daily.copy()
    out["turnover_rate"] = pd.NA
    out["turnover_rate_f"] = pd.NA
    if with_adjustment:
        adj = pro.adj_factor(ts_code=ts_code, start_date=start.replace("-", ""), end_date=end.replace("-", ""))
        if adj.empty:
            return pd.DataFrame()
        out = out.merge(adj, on=["ts_code", "trade_date"], how="inner")
    else:
        out["adj_factor"] = pd.NA
    out["date"] = pd.to_datetime(out["trade_date"])
    out["code"] = out["ts_code"].str.split(".").str[0].str.zfill(6)
    for col in ("open", "high", "low", "close", "vol", "amount", "adj_factor", "turnover_rate", "turnover_rate_f"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    adjustment_available = out["adj_factor"].notna().any()
    last_adj = out["adj_factor"].dropna().iloc[-1] if adjustment_available else None
    for raw, hfq in (("open", "hfq_open"), ("high", "hfq_high"), ("low", "hfq_low"), ("close", "hfq_close")):
        out[f"raw_{raw}"] = out[raw]
        out[hfq] = out[raw] * out["adj_factor"] / last_adj if adjustment_available else pd.NA
    out["price_adjustment_source"] = "tushare_adj_factor" if adjustment_available else "pending_tushare_adj_factor"
    out["volume"] = out["vol"] * 100.0
    out["amount"] = out["amount"] * 1000.0
    out["is_tradable"] = out["raw_open"].gt(0) & out["raw_close"].gt(0)
    out["is_suspended"] = pd.NA
    out["limit_up"] = pd.NA
    out["limit_down"] = pd.NA
    out["trade_state_source"] = "unverified_tushare_daily"
    return out[["date", "code", "open", "high", "low", "close", "raw_open", "raw_high", "raw_low", "raw_close", "hfq_open", "hfq_high", "hfq_low", "hfq_close", "volume", "amount", "turnover_rate", "turnover_rate_f", "adj_factor", "price_adjustment_source", "is_tradable", "is_suspended", "limit_up", "limit_down", "trade_state_source"]].sort_values("date")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-08-31")
    parser.add_argument("--count", type=int, default=DEFAULT_PROTOCOL.universe_size)
    parser.add_argument("--rate-limit", type=int, default=45, help="requests per minute before sleeping")
    parser.add_argument("--rate-sleep", type=float, default=61.0, help="seconds to sleep after a request batch")
    args = parser.parse_args(argv)
    if not DEFAULT_PROTOCOL.universe_size <= args.count <= 2000:
        raise ValueError("count must be between 800 and 2000")
    import tushare as ts
    root = Path(args.root); price_dir = root / "raw" / "prices"; price_dir.mkdir(parents=True, exist_ok=True)
    pro = ts.pro_api(_token())
    universe = select_universe(pro, start=args.start, count=args.count)
    universe.to_parquet(root / "security_master_source.parquet", index=False)
    failures = []
    requests_since_pause = 0
    for index, row in enumerate(universe.itertuples(index=False), 1):
        if requests_since_pause >= args.rate_limit:
            import time
            time.sleep(args.rate_sleep)
            requests_since_pause = 0
        path = price_dir / f"{row.symbol}.parquet"
        if not path.exists():
            try:
                # Training signals require HFQ; execution fields remain raw.
                data = fetch_symbol(pro, row.ts_code, start=args.start, end=args.end, with_adjustment=True)
                requests_since_pause += 1
                if data.empty:
                    failures.append({"ts_code": row.ts_code, "reason": "empty_daily_or_adjustment"})
                else:
                    data.to_parquet(path, index=False)
            except Exception as exc:
                failures.append({"ts_code": row.ts_code, "reason": f"{type(exc).__name__}:{str(exc)[:160]}"})
        if index % 10 == 0:
            print(json.dumps({"processed": index, "stored": len(list(price_dir.glob('*.parquet'))), "total": len(universe)}), flush=True)
    manifest = {"schema": "tushare_price_universe/v1", "start": args.start, "end": args.end, "requested": len(universe), "stored": len(list(price_dir.glob("*.parquet"))), "price_source": "tushare_daily+adj_factor", "trade_state": "unverified_tushare_daily", "failures": failures}
    (root / "price_collection_manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
