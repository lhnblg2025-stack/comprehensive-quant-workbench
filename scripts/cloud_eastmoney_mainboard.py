"""Run on the domestic cloud host only: Eastmoney main-board backfill."""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import pandas as pd
import requests

PREFIXES = ("000", "001", "002", "003", "600", "601", "603", "605")

def is_mainboard(code: str) -> bool:
    return str(code).zfill(6).startswith(PREFIXES)

def fetch(code: str, start: str, end: str, session: requests.Session, retries: int = 5) -> pd.DataFrame:
    code = str(code).zfill(6)
    market = 1 if code.startswith(("6", "9")) else 0
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "klt": "101", "fqt": 0,
        "secid": f"{market}.{code}",
        "beg": pd.Timestamp(start).strftime("%Y%m%d"),
        "end": pd.Timestamp(end).strftime("%Y%m%d"),
        "_": time.time_ns(),
    }
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/", "Accept": "application/json,text/plain,*/*"}
    last_error = None
    for attempt in range(retries):
        try:
            res = session.get("https://push2his.eastmoney.com/api/qt/stock/kline/get", params=params, headers=headers, timeout=30)
            if res.status_code in (403, 429, 430, 502, 503, 504):
                raise requests.HTTPError(f"retryable_status:{res.status_code}")
            res.raise_for_status()
            payload = res.json()
            rows = ((payload.get("data") or {}).get("klines") or [])
            out = []
            for row in rows:
                f = row.split(",")
                if len(f) >= 7:
                    out.append({"date": f[0], "open": f[1], "close": f[2], "high": f[3], "low": f[4],
                                "volume": f[5], "amount": f[6], "pct_chg": f[8] if len(f)>8 else None,
                                "turnover": f[10] if len(f)>10 else None})
            frame = pd.DataFrame(out)
            if frame.empty:
                return frame
            for col in ("open", "close", "high", "low", "volume", "amount", "pct_chg", "turnover"):
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
            frame["date"] = pd.to_datetime(frame["date"])
            frame["code"] = code
            frame["source"] = "cloud_eastmoney"
            frame["adjust"] = "raw"
            return frame.sort_values("date").drop_duplicates("date").reset_index(drop=True)
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            time.sleep(min(60.0, 2.0 ** attempt + 0.5))
    raise RuntimeError(f"eastmoney_retries_exhausted:{code}:{last_error}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2017-12-31")
    ap.add_argument("--interval", type=float, default=0.8)
    ap.add_argument("--max-codes", type=int, default=None)
    args = ap.parse_args()
    codes = [str(x).zfill(6) for x in Path(args.codes).read_text(encoding="utf-8").splitlines() if is_mainboard(x)]
    if args.max_codes is not None:
        codes = codes[: args.max_codes]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    records = []
    session = requests.Session()
    session.headers.update({"Connection": "keep-alive"})
    for i, code in enumerate(codes, 1):
        target = out / f"{code}.parquet"
        rec = {"code": code, "status": "FAILED"}
        try:
            if target.exists() and len(pd.read_parquet(target)) >= 350:
                rec.update(status="EXISTING_VALID", rows=len(pd.read_parquet(target)))
            else:
                frame = fetch(code, args.start, args.end, session)
                if len(frame) < 350:
                    rec.update(status="INVALID", rows=len(frame))
                else:
                    tmp = target.with_suffix(".parquet.part")
                    frame.to_parquet(tmp, index=False)
                    tmp.replace(target)
                    rec.update(status="DOWNLOADED", rows=len(frame))
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}:{exc}"
        records.append(rec)
        if i % 25 == 0:
            (out / "manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(max(args.interval, 2.0))
    (out / "manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"requested": len(codes), "downloaded": sum(r["status"]=="DOWNLOADED" for r in records), "existing": sum(r["status"]=="EXISTING_VALID" for r in records), "invalid": sum(r["status"]=="INVALID" for r in records), "failed": sum(r["status"]=="FAILED" for r in records)}, ensure_ascii=False))

if __name__ == "__main__": main()
