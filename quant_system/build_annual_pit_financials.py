"""Build annual PIT releases from raw Sina statements and actual disclosure dates.

Mappings remain intentionally conservative. A missing or ambiguous source field
produces null, never a proxy value substituted from a different report period.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _value(frame: pd.DataFrame, names: tuple[str, ...]) -> pd.Series:
    candidates = [pd.to_numeric(frame[name], errors="coerce") for name in names if name in frame]
    if candidates:
        # Prefer the source field carrying actual observations; schemas sometimes
        # include an empty aggregate label beside the populated net-value field.
        return max(candidates, key=lambda value: int(value.notna().sum()))
    return pd.Series(pd.NA, index=frame.index, dtype="Float64")


def _annual_row(raw: pd.DataFrame, code: str) -> pd.DataFrame:
    parts = {name: group.copy() for name, group in raw.groupby("statement_type")}
    balance = parts.get("资产负债表", pd.DataFrame())
    income = parts.get("利润表", pd.DataFrame())
    cash = parts.get("现金流量表", pd.DataFrame())
    dates = set()
    for frame in parts.values():
        dates.update(pd.to_datetime(frame["report_period"], errors="coerce").dropna().tolist())
    rows = []
    for period in sorted(dates):
        if period.month != 12 or period.day != 31:
            continue
        b = balance[balance["report_period"] == period]
        i = income[income["report_period"] == period]
        c = cash[cash["report_period"] == period]
        row = {"code": code, "report_period": period}
        if len(b):
            base = b.iloc[0:1]
            row["total_assets"] = _value(base, ("资产", "资产总计", "负债及股东权益总计")).iloc[0]
            row["total_liabilities"] = _value(base, ("负债", "负债合计")).iloc[0]
            row["total_equity"] = _value(base, ("股东权益合计", "所有者权益合计", "归属于母公司股东权益合计")).iloc[0]
            row["shares_outstanding"] = _value(base, ("实收资本(或股本)", "股本", "实收资本")).iloc[0]
        if len(i):
            base = i.iloc[0:1]
            row["revenue"] = _value(base, ("营业收入", "营业总收入")).iloc[0]
            row["net_profit"] = _value(base, ("净利润", "归属于母公司股东的净利润")).iloc[0]
        if len(c):
            base = c.iloc[0:1]
            row["operating_cashflow"] = _value(base, ("经营活动产生的现金流量", "经营活动产生的现金流量净额")).iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args(argv)
    root = Path(args.root); raw_dir = root / "raw" / "annual_financials"
    pieces = []
    for path in sorted(raw_dir.glob("*.parquet")):
        raw = pd.read_parquet(path)
        if raw.empty or "report_period" not in raw:
            continue
        pieces.append(_annual_row(raw, path.stem.zfill(6)))
    annual = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    disclosures = pd.read_parquet(root / "annual_disclosure_dates.parquet")
    disclosures["code"] = disclosures["code"].astype(str).str.zfill(6)
    disclosures["report_period"] = pd.to_datetime(disclosures["report_period"])
    if annual.empty:
        releases = annual
    else:
        releases = annual.merge(disclosures[["code", "report_period", "announcement_date", "source_document_id", "source_as_of"]], on=["code", "report_period"], how="left")
        releases = releases.dropna(subset=["announcement_date"])
        releases["debt_to_assets"] = releases["total_liabilities"] / releases["total_assets"]
        releases["roe"] = releases["net_profit"] / releases["total_equity"]
        releases["cfo_to_assets"] = releases["operating_cashflow"] / releases["total_assets"]
        releases["cfo_to_revenue"] = releases["operating_cashflow"] / releases["revenue"]
        releases["cfo_to_net_income"] = releases["operating_cashflow"] / releases["net_profit"]
        releases["net_margin"] = releases["net_profit"] / releases["revenue"]
        shares = pd.to_numeric(releases.get("shares_outstanding"), errors="coerce").where(lambda value: value > 0)
        releases["book_value_per_share"] = releases["total_equity"] / shares
        releases["eps_ttm"] = releases["net_profit"] / shares
        releases["operating_cashflow_per_share"] = releases["operating_cashflow"] / shares
    releases.to_parquet(root / "annual_fundamental_releases.parquet", index=False)
    coverage = {column: int(releases[column].notna().sum()) for column in ("total_assets", "total_liabilities", "total_equity", "revenue", "net_profit", "operating_cashflow", "debt_to_assets", "net_margin", "cfo_to_net_income") if column in releases}
    report = {"schema": "annual_pit_financial_releases/v1", "status": "annual_pit_only", "raw_symbols": len(pieces), "releases": int(len(releases)), "symbols": int(releases.code.nunique()) if len(releases) else 0, "announcement_coverage": int(releases.announcement_date.notna().sum()) if len(releases) else 0, "field_coverage": coverage, "limitations": ["annual_only", "quarterly_releases_not_collected", "requires_historical_trade_state_and_corporate_actions_for_promotion"]}
    (root / "annual_pit_financial_report.json").write_text(json.dumps(report, ensure_ascii=True, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
