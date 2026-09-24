#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Refresh shareholder-count history for the user's focused watchlist.

The all-market gdhs snapshot is a slow upstream dataset. This targeted job
tries the per-stock history endpoint and records success or the exact source
failure, so downstream code can use fresh focused data without pretending the
whole market snapshot is current.
"""
from __future__ import annotations
import json
from datetime import date, datetime
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WATCH = ROOT / "config" / "watchlist.json"
OUT = ROOT / "generated" / "watchlist_holders.json"


def serial(v):
    if isinstance(v, (date, datetime, pd.Timestamp)):
        return v.isoformat()
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    return v.item() if hasattr(v, "item") else v


def f10_holder(code: str) -> dict | None:
    """Eastmoney F10 shareholder research (gdrs) — direct fallback.

    AkShare's wrapper reads a different payload key for some symbols and can
    raise inside the library; the raw endpoint returns HOLDER_TOTAL_NUM and
    TOTAL_NUM_RATIO for the latest report period.
    """
    import requests
    prefix = "SH" if code.startswith(("6", "9")) else ("BJ" if code.startswith(("4", "8")) else "SZ")
    url = f"https://emweb.securities.eastmoney.com/PC_HSF10/ShareholderResearch/PageAjax?code={prefix}{code}"
    r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0",
                                               "Referer": "https://emweb.securities.eastmoney.com/"})
    r.raise_for_status()
    gdrs = (r.json().get("gdrs") or [])
    if not gdrs:
        return None
    row = gdrs[0]
    return {
        "as_of": serial(row.get("END_DATE")),
        "holder_count": serial(row.get("HOLDER_TOTAL_NUM")),
        "change_pct": serial(row.get("TOTAL_NUM_RATIO")),
    }


def main() -> int:
    import akshare as ak
    cfg = json.loads(WATCH.read_text(encoding="utf-8"))
    codes = [str(x).zfill(6) for x in cfg.get("watch", [])]
    results = []
    for code in codes:
        item = {"code": code, "status": "error", "source": "akshare.stock_zh_a_gdhs_detail_em"}
        try:
            frame = ak.stock_zh_a_gdhs_detail_em(symbol=code)
            if frame is None or frame.empty:
                item["error"] = "source returned empty"
            else:
                date_col = "股东户数统计截止日"
                frame = frame.sort_values(date_col) if date_col in frame.columns else frame
                row = frame.iloc[-1]
                item.update({"status": "available", "as_of": serial(row.get(date_col)),
                             "name": serial(row.get("名称")),
                             "holder_count": serial(row.get("股东户数-本次")),
                             "change_pct": serial(row.get("股东户数-增减比例")),
                             "announcement_date": serial(row.get("股东户数公告日期")),
                             "rows": len(frame)})
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {str(exc)[:240]}"
        # F10 comparison pass: keep the freshest observation across sources.
        fallback = None
        try:
            fallback = f10_holder(code)
        except Exception as fexc:
            if item.get("status") != "available":
                item["error"] += f" | f10: {type(fexc).__name__}: {str(fexc)[:120]}"
        if fallback and fallback.get("as_of"):
            try:
                current = str(item.get("as_of", ""))[:10]
                fresh = str(fallback["as_of"])[:10]
                if item.get("status") != "available" or fresh > current:
                    item.update({"status": "available", "source": "eastmoney_f10_gdrs", **fallback})
            except Exception:
                pass
        if item.get("status") != "available":
            results.append(item)
            continue
        results.append(item)
    out = {"generated_at": datetime.now().astimezone().isoformat(), "items": results,
           "available": sum(x["status"] == "available" for x in results), "total": len(results)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out["available"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
