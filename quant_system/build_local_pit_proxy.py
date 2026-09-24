"""Build a broad local PIT-proxy dataset from cached A-share data.

This uses statutory reporting deadlines, not fabricated actual announcement dates:
Q1 -> Apr 30, Q2 -> Aug 31, Q3 -> Oct 31, Q4 -> next Apr 30. Every record is
labelled ``deadline_pit_proxy`` and must not be promoted as actual pubDate PIT.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = {
    "摊薄每股收益(元)": "eps_ttm",
    "每股净资产_调整前(元)": "book_value_per_share",
    "每股经营性现金流(元)": "operating_cashflow_per_share",
    "净资产收益率(%)": "roe",
    "销售净利率(%)": "net_margin",
    "销售毛利率(%)": "gross_margin",
    "资产负债率(%)": "debt_to_assets",
    "流动比率": "current_ratio",
    "利息支付倍数": "ebit_to_interest",
    "经营现金净流量与净利润的比率(%)": "cfo_to_net_income",
    "经营现金净流量对销售收入比率(%)": "cfo_to_revenue",
    "资产的经营现金流量回报率(%)": "cfo_to_assets",
    "净利润增长率(%)": "yoy_ni",
    "主营业务收入增长率(%)": "yoy_eps",
    "净资产增长率(%)": "yoy_equity",
}


def _announcement_deadline(period: pd.Timestamp) -> pd.Timestamp:
    if period.month == 3:
        return pd.Timestamp(period.year, 4, 30)
    if period.month == 6:
        return pd.Timestamp(period.year, 8, 31)
    if period.month == 9:
        return pd.Timestamp(period.year, 10, 31)
    if period.month == 12:
        return pd.Timestamp(period.year + 1, 4, 30)
    raise ValueError(f"unsupported_report_period:{period}")


def _read_financial(code: str, path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "日期" not in frame.columns:
        return pd.DataFrame()
    frame["日期"] = pd.to_datetime(frame["日期"], errors="coerce")
    rows = []
    for _, row in frame.dropna(subset=["日期"]).iterrows():
        item = {"code": code, "report_period": row["日期"]}
        for source, target in METRICS.items():
            item[target] = pd.to_numeric(row.get(source), errors="coerce")
        rows.append(item)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["announcement_date"] = result["report_period"].map(_announcement_deadline)
    result["source_document_id"] = "local_financial_abstract_deadline_proxy"
    result["source_as_of"] = "deadline_pit_proxy"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-08-31")
    args = parser.parse_args()
    if args.count < 500:
        raise ValueError("count must be at least 500")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    kline_dir = ROOT / "data_warehouse" / "kline"
    selected = []
    for path in sorted(kline_dir.glob("*.parquet")):
        code = path.stem.zfill(6)
        if code == "000300" or not (ROOT / "data_warehouse" / "financial" / f"{code}.parquet").exists():
            continue
        candidate = pd.read_parquet(path, columns=["date"])
        dates = pd.to_datetime(candidate["date"], errors="coerce")
        if int(((dates >= args.start) & (dates <= args.end)).sum()) < 1200:
            continue
        selected.append(code)
        if len(selected) == args.count:
            break
    if len(selected) < args.count:
        raise ValueError(f"insufficient_eligible_symbols:{len(selected)}")
    frames = []
    price_frames = []
    for code in selected:
        kline = pd.read_parquet(kline_dir / f"{code}.parquet")
        kline["date"] = pd.to_datetime(kline["date"], errors="coerce")
        kline = kline[(kline["date"] >= args.start) & (kline["date"] <= args.end)].copy()
        if kline.empty:
            continue
        kline["code"] = code
        kline = kline.rename(columns={"open": "raw_open", "high": "raw_high", "low": "raw_low", "close": "raw_close"})
        kline["close"] = kline["raw_close"]
        kline["is_tradable"] = True
        kline["is_suspended"] = False
        kline["limit_up"] = False
        kline["limit_down"] = False
        price_frames.append(kline)
        fin_path = ROOT / "data_warehouse" / "financial" / f"{code}.parquet"
        if fin_path.exists():
            fin = _read_financial(code, fin_path)
            if not fin.empty:
                frames.append(fin)
    panel = pd.concat(price_frames, ignore_index=True).sort_values(["date", "code"])
    financials = pd.concat(frames, ignore_index=True).sort_values(["code", "announcement_date", "report_period"])
    panel.to_parquet(output / "price_panel.parquet", index=False)
    financials.to_parquet(output / "fundamental_releases.parquet", index=False)
    membership = panel.groupby("code", as_index=False)["date"].agg(start_date="min", end_date="max")
    membership["universe_id"] = "local_broad_500"
    membership["source_as_of"] = "observed_price_interval"
    membership.to_csv(output / "membership.csv", index=False)
    industry = pd.read_parquet(ROOT / "data_warehouse" / "market" / "sw_industry_map.parquet")
    industry["code"] = industry["code"].astype(str).str.zfill(6)
    industry = industry[industry["code"].isin(selected)][["code", "industry", "industry_code"]].copy()
    industry["effective_date"] = pd.Timestamp(args.start)
    industry["end_date"] = pd.NaT
    industry["source_as_of"] = "current_snapshot_proxy_not_historical"
    industry.to_parquet(output / "sw_l1_industry_history_proxy.parquet", index=False)
    bench = pd.read_parquet(ROOT / "data_warehouse" / "market" / "index_daily_沪深300.parquet")
    bench["date"] = pd.to_datetime(bench["date"])
    bench = bench[(bench["date"] >= panel["date"].min()) & (bench["date"] <= panel["date"].max())].sort_values("date")
    bench["return"] = pd.to_numeric(bench["close"], errors="coerce").pct_change()
    benchmark_returns = bench[["date", "return"]].dropna()
    # Strict benchmark alignment uses only dates with an observable CSI300 return.
    panel = panel[panel["date"].isin(benchmark_returns["date"])].copy()
    panel.to_parquet(output / "price_panel.parquet", index=False)
    membership = panel.groupby("code", as_index=False)["date"].agg(start_date="min", end_date="max")
    membership["universe_id"] = "local_broad_500"
    membership["source_as_of"] = "observed_price_interval"
    membership.to_csv(output / "membership.csv", index=False)
    benchmark_returns.to_csv(output / "benchmark_csi300.csv", index=False)
    report = {"schema": "local_pit_proxy/v1", "data_quality": "deadline_pit_proxy", "industry_quality": "current_snapshot_proxy_not_historical", "symbols": int(panel.code.nunique()), "price_rows": int(len(panel)), "financial_releases": int(len(financials)), "date_min": str(panel.date.min().date()), "date_max": str(panel.date.max().date()), "financial_columns": list(financials.columns)}
    (output / "proxy_build_report.json").write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
