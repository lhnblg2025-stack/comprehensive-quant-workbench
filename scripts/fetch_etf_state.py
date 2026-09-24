#!/usr/bin/env python3
"""抓取ETF实时行情、最新份额、折溢价与主力净流入并按交易日落盘。"""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_warehouse" / "events" / "etf_state.parquet"
SCALE = ROOT / "data_warehouse" / "events" / "etf_exchange_scale.parquet"


def main() -> int:
    import akshare as ak
    spot = ak.fund_etf_spot_em()
    if spot is None or spot.empty:
        raise RuntimeError("ETF实时列表为空")
    cols = {
        "代码":"code", "名称":"name", "最新价":"price", "IOPV实时估值":"iopv",
        "基金折价率":"discount_pct", "涨跌幅":"change_pct", "成交量":"volume",
        "成交额":"amount", "主力净流入-净额":"main_net", "最新份额":"shares",
        "流通市值":"float_mv", "总市值":"total_mv", "数据日期":"date", "更新时间":"updated_at",
    }
    keep = [c for c in cols if c in spot.columns]
    df = spot[keep].rename(columns=cols).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["code", "date"])
    for c in set(df.columns) - {"code", "name", "date", "updated_at"}:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if OUT.exists():
        old = pd.read_parquet(OUT)
        df = pd.concat([old, df], ignore_index=True)
    df = df.drop_duplicates(["date", "code"], keep="last").sort_values(["date", "code"])
    df["shares_delta"] = df.groupby("code")["shares"].diff()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, OUT)

    # 交易所规模/份额快照，作为独立申赎/份额趋势底座。
    frames = []
    for fn, market in ((getattr(ak, "fund_etf_scale_sse", None), "SSE"), (getattr(ak, "fund_etf_scale_szse", None), "SZSE")):
        if fn is None:
            continue
        try:
            x = fn()
            if x is not None and not x.empty:
                x["market"] = market
                frames.append(x)
        except Exception as exc:
            print(f"scale {market} failed: {exc}")
    if frames:
        scale = pd.concat(frames, ignore_index=True)
        scale.to_parquet(SCALE, index=False)
    latest = df["date"].max()
    print(f"ETF state saved: {len(df)} rows, as_of={latest}, codes={df[df['date']==latest]['code'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
