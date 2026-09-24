"""Build a 500-stock, 10-year auditable PIT dataset from Baostock.

The builder uses Baostock's public ``pubDate`` as announcement_date. Financial
requests are persisted incrementally so a stopped run resumes without inventing
availability dates. Industry classifications are annual historical snapshots,
with each snapshot effective on its query date until superseded.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_START = "2016-01-01"
DEFAULT_END = "2026-08-31"


def _bs_code(code: str) -> str:
    code = str(code).zfill(6)
    return f"sh.{code}" if code.startswith(("5", "6", "9")) else f"sz.{code}"


def _rows(query) -> list[list[str]]:
    rows: list[list[str]] = []
    while query.error_code == "0" and query.next():
        rows.append(query.get_row_data())
    return rows


def _balanced_codes(codes: list[str], count: int) -> list[str]:
    buckets: dict[str, list[str]] = {"main": [], "growth": [], "star": []}
    for code in sorted(set(codes)):
        if code in {"000300", "000905", "000852"}:
            continue
        bucket = "star" if code.startswith("68") else "growth" if code.startswith(("30", "39")) else "main"
        buckets[bucket].append(code)
    quotas = {"main": int(count * 0.60), "growth": int(count * 0.25), "star": count - int(count * 0.60) - int(count * 0.25)}
    selected: list[str] = []
    for bucket in ("main", "growth", "star"):
        selected.extend(buckets[bucket][:quotas[bucket]])
    remaining = [code for bucket in buckets.values() for code in bucket if code not in set(selected)]
    return (selected + remaining)[:count]


def select_codes(count: int, bs=None, as_of: str | None = None) -> list[str]:
    """Choose a deterministic board-balanced universe, using a source snapshot if needed."""
    local_codes = [path.stem.zfill(6) for path in sorted((ROOT / "data_warehouse" / "kline").glob("*.parquet"))]
    # A dated source universe comes first. A broad local cache may start years
    # after the campaign start and therefore cannot define a ten-year universe.
    if bs is None or not as_of:
        return _balanced_codes(local_codes, count)
    query = bs.query_all_stock(day=as_of)
    source_codes = []
    while query.error_code == "0" and query.next():
        row = dict(zip(query.fields, query.get_row_data()))
        code = str(row.get("code", "")).replace("sh.", "").replace("sz.", "")
        if len(code) == 6 and code[0] in {"0", "3", "6", "8"}:
            source_codes.append(code)
    return _balanced_codes(source_codes, count)


def fetch_prices(bs, code: str, start: str, end: str) -> pd.DataFrame:
    fields = "date,open,high,low,close,volume,amount,turn"
    raw = _rows(bs.query_history_k_data_plus(_bs_code(code), fields, start_date=start, end_date=end, frequency="d", adjustflag="3"))
    hfq = _rows(bs.query_history_k_data_plus(_bs_code(code), "date,open,high,low,close", start_date=start, end_date=end, frequency="d", adjustflag="1"))
    if not raw or not hfq:
        return pd.DataFrame()
    raw_frame = pd.DataFrame(raw, columns=fields.split(","))
    hfq_frame = pd.DataFrame(hfq, columns=["date", "hfq_open", "hfq_high", "hfq_low", "hfq_close"])
    frame = raw_frame.merge(hfq_frame, on="date", how="inner")
    for column in frame.columns:
        if column != "date":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["code"] = code
    frame["raw_open"] = frame["open"]
    frame["raw_high"] = frame["high"]
    frame["raw_low"] = frame["low"]
    frame["raw_close"] = frame["close"]
    frame["close"] = frame["hfq_close"]
    frame["is_tradable"] = frame["raw_open"].gt(0) & frame["raw_close"].gt(0)
    # Baostock daily bars do not provide historical suspension or limit-lock flags.
    # Keep the fields nullable and label the source so production research cannot
    # mistake placeholder flags for exchange-grade historical tradeability data.
    frame["is_suspended"] = pd.NA
    frame["limit_up"] = pd.NA
    frame["limit_down"] = pd.NA
    frame["trade_state_source"] = "unverified_baostock_daily_bar"
    return frame


def fetch_financials(bs, code: str, years: range) -> pd.DataFrame:
    specs = {
        "profit": ("query_profit_data", ["code", "pubDate", "statDate", "roe", "net_margin", "gross_margin", "net_profit", "eps_ttm", "revenue", "total_share", "liqa_share"]),
        "growth": ("query_growth_data", ["code", "pubDate", "statDate", "yoy_equity", "yoy_assets", "yoy_ni", "yoy_eps", "yoy_parent_ni"]),
        "balance": ("query_balance_data", ["code", "pubDate", "statDate", "current_ratio", "quick_ratio", "cash_ratio", "yoy_liability", "debt_to_assets", "asset_to_equity"]),
        "cash": ("query_cash_flow_data", ["code", "pubDate", "statDate", "cfo_to_assets", "ncfa_to_assets", "tangible_assets_to_assets", "ebit_to_interest", "cfo_to_revenue", "cfo_to_net_income", "cfo_to_growth"]),
    }
    parts: list[pd.DataFrame] = []
    for year in years:
        for quarter in (1, 2, 3, 4):
            frames = []
            for _, (method, columns) in specs.items():
                query = getattr(bs, method)(code=_bs_code(code), year=year, quarter=quarter)
                values = _rows(query)
                if values:
                    frames.append(pd.DataFrame(values, columns=columns[:len(values[0])]))
            if not frames:
                continue
            merged = frames[0]
            for frame in frames[1:]:
                keys = [key for key in ("code", "pubDate", "statDate") if key in merged.columns and key in frame.columns]
                merged = merged.merge(frame, on=keys, how="outer")
            parts.append(merged)
    if not parts:
        return pd.DataFrame()
    frame = pd.concat(parts, ignore_index=True)
    frame["code"] = frame["code"].astype(str).str.replace(r"^(sh|sz)\.", "", regex=True).str.zfill(6)
    frame = frame.rename(columns={"statDate": "report_period", "pubDate": "announcement_date"})
    for column in frame.columns:
        if column not in {"code", "report_period", "announcement_date"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["book_value_per_share"] = pd.NA
    frame["operating_cashflow_per_share"] = pd.NA
    frame["source_document_id"] = "baostock_financial_release"
    frame["source_as_of"] = "baostock_pubDate"
    return frame.dropna(subset=["report_period", "announcement_date"]).drop_duplicates(["code", "report_period", "announcement_date"], keep="last")


def fetch_industry_history(bs, snapshot_dates: list[str]) -> pd.DataFrame:
    parts = []
    for day in snapshot_dates:
        query = bs.query_stock_industry(date=day)
        values = _rows(query)
        if not values:
            continue
        frame = pd.DataFrame(values, columns=query.fields)
        frame = frame.rename(columns={"updateDate": "effective_date", "industryClassification": "industry_code"})
        frame["code"] = frame["code"].astype(str).str.replace(r"^(sh|sz)\.", "", regex=True).str.zfill(6)
        frame["source_as_of"] = "baostock_stock_industry"
        parts.append(frame[["code", "industry", "industry_code", "effective_date", "source_as_of"]])
    if not parts:
        return pd.DataFrame(columns=["code", "industry", "industry_code", "effective_date", "end_date", "source_as_of"])
    history = pd.concat(parts, ignore_index=True)
    history["effective_date"] = pd.to_datetime(history["effective_date"])
    history = history.sort_values(["code", "effective_date"]).drop_duplicates(["code", "effective_date"], keep="last")
    history["end_date"] = history.groupby("code")["effective_date"].shift(-1) - pd.Timedelta(days=1)
    return history


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "data_warehouse" / "research_panels" / "ashare_pit_500_1000"))
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--stage", choices=("all", "prices", "financials", "industries"), default="all")
    args = parser.parse_args()
    if not 500 <= args.count <= 1000:
        raise ValueError("count must be between 500 and 1000")
    import baostock as bs
    output = Path(args.output)
    raw_dir = output / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock_login:{login.error_msg}")
    try:
        codes = select_codes(args.count, bs=bs, as_of=args.start)
        if len(codes) < args.count:
            raise RuntimeError(f"source_universe_below_requested:requested={args.count}:available={len(codes)}")
        stage_results: dict[str, dict[str, object]] = {}
        if args.stage in {"all", "prices"}:
            price_dir = raw_dir / "prices"
            price_dir.mkdir(exist_ok=True)
            failures = []
            for index, code in enumerate(codes, 1):
                path = price_dir / f"{code}.parquet"
                if not path.exists():
                    try:
                        frame = fetch_prices(bs, code, args.start, args.end)
                        if frame.empty:
                            failures.append({"code": code, "reason": "empty_price_response"})
                        else:
                            frame.to_parquet(path, index=False)
                    except Exception as exc:
                        failures.append({"code": code, "reason": f"{type(exc).__name__}:{str(exc)[:120]}"})
                if index % 25 == 0:
                    print(json.dumps({"stage": "prices", "processed": index, "stored": len(list(price_dir.glob('*.parquet'))), "total": len(codes)}), flush=True)
                if args.sleep:
                    time.sleep(args.sleep)
            stage_results["prices"] = {"requested": len(codes), "stored": len(list(price_dir.glob("*.parquet"))), "failures": failures}
        if args.stage in {"all", "financials"}:
            financial_dir = raw_dir / "financials"
            financial_dir.mkdir(exist_ok=True)
            years = range(pd.Timestamp(args.start).year, pd.Timestamp(args.end).year + 1)
            failures = []
            for index, code in enumerate(codes, 1):
                path = financial_dir / f"{code}.parquet"
                if not path.exists():
                    try:
                        frame = fetch_financials(bs, code, years)
                        if frame.empty:
                            failures.append({"code": code, "reason": "empty_financial_response"})
                        else:
                            frame.to_parquet(path, index=False)
                    except Exception as exc:
                        failures.append({"code": code, "reason": f"{type(exc).__name__}:{str(exc)[:120]}"})
                if index % 10 == 0:
                    print(json.dumps({"stage": "financials", "processed": index, "stored": len(list(financial_dir.glob('*.parquet'))), "total": len(codes)}), flush=True)
                if args.sleep:
                    time.sleep(args.sleep)
            stage_results["financials"] = {"requested": len(codes), "stored": len(list(financial_dir.glob("*.parquet"))), "failures": failures}
        if args.stage in {"all", "industries"}:
            snapshot_dates = [f"{year}-01-04" for year in range(pd.Timestamp(args.start).year, pd.Timestamp(args.end).year + 1)]
            history = fetch_industry_history(bs, snapshot_dates)
            history[history["code"].isin(codes)].to_parquet(output / "sw_l1_industry_history.parquet", index=False)
        manifest = {"schema": "long_pit_source/v2", "provider": "baostock", "start": args.start, "end": args.end, "requested_codes": codes, "count": len(codes), "financial_availability": "provider_pubDate_plus_one_trading_session", "industry_history": "annual_baostock_snapshots", "trade_state": "unverified_baostock_daily_bar", "stages": stage_results}
        (output / "build_manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    finally:
        bs.logout()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
