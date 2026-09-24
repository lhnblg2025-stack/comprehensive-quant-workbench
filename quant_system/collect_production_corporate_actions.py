"""Collect and normalize implemented corporate actions and delisting references."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _actions(code: str, frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if frame.empty:
        return pd.DataFrame()
    for row in frame.to_dict("records"):
        ex_date = pd.to_datetime(row.get("除权除息日"), errors="coerce")
        status = str(row.get("方案进度", ""))
        if pd.isna(ex_date) or "实施" not in status:
            continue
        cash_per_10 = pd.to_numeric(row.get("现金分红-现金分红比例"), errors="coerce")
        stock_per_10 = pd.to_numeric(row.get("送转股份-送转总比例"), errors="coerce")
        source_id = f"eastmoney_fhps:{code}:{str(row.get('报告期', ''))}"
        if pd.notna(cash_per_10) and cash_per_10 > 0:
            rows.append({"code": code, "ex_date": ex_date, "action_type": "dividend", "cash_per_share": float(cash_per_10) / 10.0, "ratio": 1.0, "source_document_id": source_id, "source_as_of": row.get("最新公告日期")})
        if pd.notna(stock_per_10) and stock_per_10 > 0:
            rows.append({"code": code, "ex_date": ex_date, "action_type": "split", "cash_per_share": 0.0, "ratio": 1.0 + float(stock_per_10) / 10.0, "source_document_id": source_id, "source_as_of": row.get("最新公告日期")})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args(argv)
    import akshare as ak
    root = Path(args.root); cached = root / "raw" / "corporate_actions"; cached.mkdir(parents=True, exist_ok=True)
    codes = sorted(path.stem.zfill(6) for path in (root / "raw" / "prices").glob("*.parquet"))
    failures = []; parts = []
    for index, code in enumerate(codes, 1):
        path = cached / f"{code}.parquet"
        try:
            if path.exists():
                normalized = pd.read_parquet(path)
            else:
                normalized = _actions(code, ak.stock_fhps_detail_em(symbol=code))
                normalized.to_parquet(path, index=False)
            if not normalized.empty:
                parts.append(normalized)
        except Exception as exc:
            failures.append({"code": code, "reason": f"{type(exc).__name__}:{str(exc)[:160]}"})
        if index % 25 == 0:
            print(json.dumps({"processed": index, "cached": len(list(cached.glob('*.parquet'))), "total": len(codes)}), flush=True)
    actions = pd.concat(parts, ignore_index=True).drop_duplicates(["code", "ex_date", "action_type"], keep="last") if parts else pd.DataFrame(columns=["code", "ex_date", "action_type", "cash_per_share", "ratio"])
    actions.to_parquet(root / "corporate_actions.parquet", index=False)
    delisted = []
    for getter, code_col, date_col, exchange in ((ak.stock_info_sh_delist, "公司代码", "暂停上市日期", "SH"), (ak.stock_info_sz_delist, "证券代码", "终止上市日期", "SZ")):
        try:
            frame = getter().rename(columns={code_col: "code", date_col: "delist_date"})
            frame["code"] = frame["code"].astype(str).str.zfill(6); frame["delist_date"] = pd.to_datetime(frame["delist_date"], errors="coerce"); frame["exchange"] = exchange
            delisted.append(frame[["code", "delist_date", "exchange"]])
        except Exception as exc:
            failures.append({"source": f"{exchange}_delist", "reason": f"{type(exc).__name__}:{str(exc)[:160]}"})
    delisting = pd.concat(delisted, ignore_index=True) if delisted else pd.DataFrame(columns=["code", "delist_date", "exchange"])
    delisting.to_parquet(root / "delisting_reference.parquet", index=False)
    report = {"schema": "production_corporate_actions/v1", "actions": int(len(actions)), "action_symbols": int(actions.code.nunique()) if len(actions) else 0, "delisting_rows": int(len(delisting)), "failures": failures}
    (root / "corporate_actions_report.json").write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
