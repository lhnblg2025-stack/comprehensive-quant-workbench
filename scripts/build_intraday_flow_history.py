#!/usr/bin/env python3
"""Build auditable intraday market and industry flow-proxy histories.

The source is the existing five-minute full-market quote snapshots. Quote
snapshots do not contain true order-flow direction, so ``active_flow_proxy_yi``
is explicitly a signed turnover proxy, never labelled as actual net inflow.
ETF intraday data remains unavailable until a dedicated ETF snapshot collector
is installed.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WAREHOUSE = ROOT / "data_warehouse"
GENERATED = ROOT / "generated"
CST = timezone(timedelta(hours=8))


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        try:
            return None if value != value else value
        except Exception:
            return value
    if hasattr(value, "item"):
        try:
            return _json_value(value.item())
        except Exception:
            pass
    return str(value)


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False, default=_json_value)
            handle.write("\n")
        Path(temp_name).replace(path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _snapshot_files(day: str) -> list[Path]:
    directory = WAREHOUSE / "realtime_snapshot" / day.replace("-", "")
    files = []
    for path in directory.glob("*.parquet"):
        stamp = path.stem[:6]
        if len(stamp) == 6 and ("093000" <= stamp <= "113500" or "125500" <= stamp <= "150500"):
            files.append(path)
    return sorted(files)


def _industry_map() -> pd.DataFrame:
    path = WAREHOUSE / "market" / "sw_industry_map.parquet"
    frame = pd.read_parquet(path, columns=["code", "industry", "industry_code"])
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    return frame.drop_duplicates("code", keep="last")


def _build_etf_history(day: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    directory = WAREHOUSE / "realtime_etf_snapshot" / day.replace("-", "")
    files = sorted(path for path in directory.glob("*.parquet") if len(path.stem) >= 6)
    snapshots: list[pd.DataFrame] = []
    latest_amount: pd.Series | None = None
    for path in files:
        stamp = path.stem[:6]
        if not ("093000" <= stamp <= "113500" or "125500" <= stamp <= "150500"):
            continue
        frame = pd.read_parquet(path)
        if not {"code6", "name", "amount_wan", "pct_chg"}.issubset(frame.columns):
            continue
        frame = frame.copy()
        frame["amount_wan"] = pd.to_numeric(frame["amount_wan"], errors="coerce").fillna(0).clip(lower=0)
        frame["pct_chg"] = pd.to_numeric(frame["pct_chg"], errors="coerce")
        frame["_time"] = f"{stamp[:2]}:{stamp[2:4]}"
        snapshots.append(frame)
        latest_amount = frame.set_index("code6")["amount_wan"]
    if not snapshots or latest_amount is None:
        return [], {"status": "unavailable", "reason": "dedicated intraday ETF snapshots unavailable"}
    top_codes = set(latest_amount.sort_values(ascending=False).head(40).index.astype(str))
    rows: list[dict[str, Any]] = []
    previous: dict[str, float] = {}
    for frame in snapshots:
        for item in frame[frame["code6"].astype(str).isin(top_codes)].to_dict("records"):
            code = str(item["code6"])
            amount = float(item.get("amount_wan") or 0)
            delta = max(0.0, amount - previous.get(code, 0.0))
            pct = float(item.get("pct_chg") or 0)
            rows.append({
                "timestamp": f"{day} {item['_time']}:00", "time": item["_time"],
                "code": code, "name": str(item.get("name") or code),
                "pct_chg": round(pct, 3), "amount_yi": round(amount / 10000, 3),
                "amount_delta_yi": round(delta / 10000, 3),
                "active_flow_proxy_yi": round(delta * (1 if pct > 0 else (-1 if pct < 0 else 0)) / 10000, 3),
            })
            previous[code] = amount
    return rows, {"status": "available", "snapshot_count": len(snapshots), "tracked_top_amount": len(top_codes)}


def build(day: str) -> dict[str, Any]:
    files = _snapshot_files(day)
    mapping = _industry_map()
    market_rows: list[dict[str, Any]] = []
    industry_rows: list[dict[str, Any]] = []
    previous_amount: dict[str, float] = {}

    for path in files:
        frame = pd.read_parquet(path)
        required = {"code", "pct_chg", "amount_wan"}
        if not required.issubset(frame.columns):
            continue
        frame = frame.copy()
        frame["code"] = frame["code"].astype(str).str[-6:].str.zfill(6)
        frame["pct_chg"] = pd.to_numeric(frame["pct_chg"], errors="coerce")
        frame["amount_wan"] = pd.to_numeric(frame["amount_wan"], errors="coerce").fillna(0).clip(lower=0)
        current = frame.set_index("code")["amount_wan"]
        if previous_amount:
            prior = current.index.to_series().map(previous_amount).fillna(0)
            frame["amount_delta_wan"] = (current.values - prior.values).clip(min=0)
        else:
            frame["amount_delta_wan"] = current.values
        previous_amount = current.to_dict()
        frame["signed_delta_wan"] = frame["amount_delta_wan"] * frame["pct_chg"].fillna(0).apply(
            lambda value: 1 if value > 0 else (-1 if value < 0 else 0)
        )
        frame = frame.merge(mapping, on="code", how="left")
        timestamp = f"{day} {path.stem[:2]}:{path.stem[2:4]}:{path.stem[4:6]}"
        valid_pct = frame["pct_chg"].dropna()
        market_rows.append({
            "timestamp": timestamp,
            "time": timestamp[11:16],
            "name": "全市场",
            "stock_count": int(len(frame)),
            "advance": int((valid_pct > 0).sum()),
            "decline": int((valid_pct < 0).sum()),
            "breadth": round(float((valid_pct > 0).mean()), 4) if len(valid_pct) else None,
            "amount_yi": round(float(frame["amount_wan"].sum()) / 10000, 2),
            "amount_delta_yi": round(float(frame["amount_delta_wan"].sum()) / 10000, 2),
            "active_flow_proxy_yi": round(float(frame["signed_delta_wan"].sum()) / 10000, 2),
        })
        grouped = frame.dropna(subset=["industry"]).groupby(["industry", "industry_code"], dropna=False)
        for (industry, industry_code), batch in grouped:
            pct = batch["pct_chg"].dropna()
            industry_rows.append({
                "timestamp": timestamp,
                "time": timestamp[11:16],
                "name": str(industry),
                "industry_code": str(industry_code),
                "stock_count": int(len(batch)),
                "avg_pct": round(float(pct.mean()), 3) if len(pct) else None,
                "breadth": round(float((pct > 0).mean()), 4) if len(pct) else None,
                "amount_yi": round(float(batch["amount_wan"].sum()) / 10000, 3),
                "amount_delta_yi": round(float(batch["amount_delta_wan"].sum()) / 10000, 3),
                "active_flow_proxy_yi": round(float(batch["signed_delta_wan"].sum()) / 10000, 3),
            })

    etf_rows, etf_status = _build_etf_history(day)
    def _periods(times: list[str]) -> dict[str, bool]:
        return {
            "open": any("09:30" <= value <= "10:00" for value in times),
            "morning": any("10:00" < value <= "11:35" for value in times),
            "afternoon": any("13:00" <= value <= "14:30" for value in times),
            "close": any("14:30" < value <= "15:05" for value in times),
        }
    periods = _periods([row["time"] for row in market_rows])
    etf_periods = _periods([row["time"] for row in etf_rows])
    market_complete = bool(market_rows) and all(periods.values())
    etf_complete = etf_status.get("status") == "available" and all(etf_periods.values())
    payload = {
        "schema_version": "intraday-flow-history.v1",
        "date": day,
        "generated_at": datetime.now(CST).isoformat(),
        "status": "available" if market_rows else "unavailable",
        "source": f"data_warehouse/realtime_snapshot/{day.replace('-', '')}/*.parquet",
        "snapshot_count": len(market_rows),
        "periods": periods,
        "etf_periods": etf_periods,
        "market_complete_day": market_complete,
        "etf_complete_day": etf_complete,
        "complete_day": market_complete and etf_complete,
        "methodology": {
            "amount": "腾讯全市场快照累计成交额；相邻快照差分后负值截为0",
            "active_flow_proxy_yi": "成交额增量乘个股涨跌方向后汇总，仅为主动资金代理，不是真实主力净流入",
            "breadth": "上涨股票数/有效涨跌幅股票数",
            "etf": "独立腾讯ETF批量快照；成交额增量乘涨跌方向仍为代理，不是真实主力净流入",
        },
        "market": market_rows,
        "industry": industry_rows,
        "etf": {**etf_status, "rows": etf_rows},
    }
    output = GENERATED / f"intraday_flow_history_{day}.json"
    _atomic_write(output, payload)
    try:
        payload["output_path"] = output.relative_to(ROOT).as_posix()
    except ValueError:
        payload["output_path"] = str(output)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="聚合五分钟快照为盘中市场/行业时序")
    parser.add_argument("--date", default=datetime.now(CST).strftime("%Y-%m-%d"))
    args = parser.parse_args()
    result = build(args.date)
    print(json.dumps({
        "ok": result["status"] == "available",
        "date": result["date"],
        "snapshots": result["snapshot_count"],
        "complete_day": result["complete_day"],
        "output": result.get("output_path"),
    }, ensure_ascii=False))
    return 0 if result["status"] == "available" else 1


if __name__ == "__main__":
    raise SystemExit(main())
