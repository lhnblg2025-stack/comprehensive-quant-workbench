"""Materialize official SW historical stock classifications as effective intervals."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pandas as pd

URL = "https://www.swsresearch.com/swindex/pdf/SwClass2021/StockClassifyUse_stock.xls"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args(argv)
    import requests
    # The official endpoint currently presents a certificate chain unavailable in
    # this runtime. Record that explicit transport exception in the manifest.
    response = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60, verify=False)
    response.raise_for_status()
    raw = pd.read_excel(io.BytesIO(response.content), header=0)
    raw = raw.rename(columns={"股票代码": "code", "计入日期": "effective_date", "行业代码": "industry", "更新日期": "source_as_of"})
    raw["code"] = raw["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    raw["effective_date"] = pd.to_datetime(raw["effective_date"], errors="coerce")
    raw["source_as_of"] = pd.to_datetime(raw["source_as_of"], errors="coerce")
    raw["industry_code"] = raw["industry"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    # The official workbook uses a six-digit detailed classification. Research
    # neutralization requires the historical SW level-1 parent, not tiny leaf groups.
    raw["industry"] = raw["industry_code"].str[:2]
    raw = raw.dropna(subset=["code", "effective_date", "industry"]).sort_values(["code", "effective_date", "source_as_of"])
    raw = raw.drop_duplicates(["code", "effective_date"], keep="last")
    raw["end_date"] = raw.groupby("code")["effective_date"].shift(-1) - pd.Timedelta(days=1)
    universe = {path.stem.zfill(6) for path in (Path(args.root) / "raw" / "prices").glob("*.parquet")}
    history = raw[raw["code"].isin(universe)].copy()
    output = Path(args.root) / "sw_l1_industry_history.parquet"
    history.to_parquet(output, index=False)
    report = {"schema": "sw_historical_industry/v1", "source_url": URL, "transport": "official_https_verify_false_due_to_runtime_ca_chain", "source_rows": int(len(raw)), "rows": int(len(history)), "symbols": int(history.code.nunique()), "effective_min": str(history.effective_date.min().date()), "effective_max": str(history.effective_date.max().date()), "missing_universe_symbols": sorted(universe - set(history.code))}
    (Path(args.root) / "sw_historical_industry_report.json").write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
