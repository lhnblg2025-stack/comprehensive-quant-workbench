#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ETF历史与行业资金增量采集。

- ETF历史：按代码抓取日行情，合并已有份额/交易所规模快照，增量保存。
- 行业资金：抓取当前行业资金排名，按时间戳追加，供全天分时交互图。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data_warehouse" / "events"
MARKET = ROOT / "data_warehouse" / "market"
ETF_HIST = EVENTS / "etf_history.parquet"
ETF_STATE = EVENTS / "etf_state.parquet"
ETF_SCALE = EVENTS / "etf_exchange_scale.parquet"
SECTOR_INTRADAY = MARKET / "sector_fund_flow_intraday.parquet"
INDUSTRY_FLOW = MARKET / "industry_fund_flow.parquet"
CONCEPT_FLOW = MARKET / "concept_fund_flow_intraday.parquet"
CST = timezone(timedelta(hours=8))
DEFAULT_CODES = ["510300", "510050", "510500", "159915", "159919", "512100", "512880", "512000", "512480", "512760", "512690", "515790", "516160", "588000", "588080", "513100", "513500", "513050", "513120", "518880"]


def _atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _history_one(ak, code: str, start: str) -> pd.DataFrame:
    frames = []
    try:
        df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=start, end_date=datetime.now().strftime("%Y%m%d"), adjust="")
        if df is not None and not df.empty:
            rename = {"日期":"date", "开盘":"open", "收盘":"price", "最高":"high", "最低":"low", "成交量":"volume", "成交额":"amount", "涨跌幅":"change_pct", "换手率":"turnover"}
            keep = [c for c in rename if c in df.columns]
            x = df[keep].rename(columns=rename).copy()
            x["code"] = code
            frames.append(x)
    except Exception as exc:
        print(f"ETF {code} eastmoney failed: {exc}")
    if not frames:
        try:
            prefix = "sh" if code.startswith(("5", "6")) else "sz"
            df = ak.fund_etf_hist_sina(symbol=prefix + code)
            if df is not None and not df.empty:
                rename = {"date":"date", "open":"open", "close":"price", "high":"high", "low":"low", "volume":"volume", "amount":"amount"}
                x = df[[c for c in rename if c in df.columns]].rename(columns=rename).copy()
                x["code"] = code
                frames.append(x)
        except Exception as exc:
            print(f"ETF {code} sina failed: {exc}")
    if not frames:
        return pd.DataFrame()
    x = pd.concat(frames, ignore_index=True)
    x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()
    x = x.dropna(subset=["date"])
    for c in set(x.columns) - {"code", "date"}:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x


def collect_etf(codes: list[str], start: str) -> dict:
    import akshare as ak
    parts = []
    for idx, code in enumerate(codes):
        x = _history_one(ak, code, start)
        if not x.empty:
            parts.append(x)
        if idx + 1 < len(codes):
            time.sleep(0.35)
    if not parts:
        return {"ok": False, "error": "ETF历史接口均无数据"}
    hist = pd.concat(parts, ignore_index=True)
    # 网络失败回退表可能带有上一轮 merge 的重复列，先按列名去重再纵向合并。
    hist = hist.loc[:, ~hist.columns.duplicated(keep="last")]
    if ETF_HIST.exists():
        old_hist = pd.read_parquet(ETF_HIST)
        old_hist = old_hist.loc[:, ~old_hist.columns.duplicated(keep="last")]
        hist = pd.concat([old_hist, hist], ignore_index=True)
    hist = hist.loc[:, ~hist.columns.duplicated(keep="last")]
    hist["code"] = hist["code"].astype(str).str.zfill(6)
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.normalize()
    hist = hist.drop_duplicates(["code", "date"], keep="last")

    if ETF_STATE.exists():
        state = pd.read_parquet(ETF_STATE).copy()
        state["code"] = state["code"].astype(str).str.zfill(6)
        state["date"] = pd.to_datetime(state["date"], errors="coerce").dt.normalize()
        state = state.sort_values(["code", "date"]).drop_duplicates(["code", "date"], keep="last")
        cols = [c for c in ("code", "date", "name", "shares", "shares_delta", "main_net", "discount_pct", "total_mv") if c in state]
        # 幂等化：清除历史产物里上一轮 merge 残留的同名/后缀列，避免重复列名
        # （ETF_HIST 已在上一轮持久化了 name/shares/... 与 *_state 两套列）。
        overlap = [c for c in cols if c not in ("code", "date")]
        drop_cols = [c for c in overlap + [f"{c}_state" for c in overlap] if c in hist.columns]
        if drop_cols:
            hist = hist.drop(columns=drop_cols)
        hist = hist.merge(state[cols], on=["code", "date"], how="left", suffixes=("", "_state"))
    hist["scale_yi"] = hist.get("total_mv")
    if "scale_yi" in hist:
        hist["scale_yi"] = pd.to_numeric(hist["scale_yi"], errors="coerce") / 1e8
    if "shares" in hist:
        calculated = pd.to_numeric(hist["shares"], errors="coerce") * pd.to_numeric(hist["price"], errors="coerce") / 1e8
        hist["scale_yi"] = hist["scale_yi"].fillna(calculated)
    hist = hist.sort_values(["code", "date"])
    if "shares" in hist:
        derived = hist.groupby("code")["shares"].diff()
        hist["shares_delta"] = pd.to_numeric(hist.get("shares_delta"), errors="coerce").fillna(derived)
    _atomic(hist, ETF_HIST)
    return {"ok": True, "rows": len(hist), "codes": int(hist["code"].nunique()), "start": str(hist["date"].min().date()), "end": str(hist["date"].max().date()), "path": str(ETF_HIST)}


def collect_sector() -> dict:
    import akshare as ak
    now = datetime.now(CST)
    # The source is a trading-day snapshot. Keep the explicit trading date
    # separate from collection time so post-midnight jobs do not relabel it.
    trade_day = now.date()
    try:
        from quant_system.market_clock import latest_completed_trading_day
        trade_day = latest_completed_trading_day(now)
    except Exception:
        pass
    frames = []
    saved = {}
    errors = {}
    for sector_type in ("行业资金流", "概念资金流"):
        last_error = ""
        df = None
        for attempt in range(3):
            try:
                df = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type=sector_type)
                if df is not None and not df.empty:
                    break
                last_error = "接口返回空数据"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {str(exc)[:120]}"
            if attempt < 2:
                time.sleep(attempt + 1)
        if df is None or df.empty:
            errors[sector_type] = last_error or "接口返回空数据"
            print(f"sector {sector_type} failed: {errors[sector_type]}")
            continue
        name_col = next((c for c in ("名称", "行业名称", "板块名称") if c in df.columns), None)
        net_col = next((c for c in ("今日主力净流入-净额", "主力净流入-净额", "今日主力净流入") if c in df.columns), None)
        if not name_col or not net_col:
            errors[sector_type] = f"返回字段不完整：{list(df.columns)[:12]}"
            continue
        x = pd.DataFrame({"ts": now, "date": trade_day, "type": sector_type.replace("资金流", ""), "name": df[name_col].astype(str), "main_net": pd.to_numeric(df[net_col], errors="coerce")})
        x["main_net_yi"] = x["main_net"] / 1e8
        frames.append(x)
        target = INDUSTRY_FLOW if sector_type == "行业资金流" else CONCEPT_FLOW
        old = pd.read_parquet(target) if target.exists() else pd.DataFrame()
        merged = pd.concat([old, x], ignore_index=True)
        merged["ts"] = pd.to_datetime(merged["ts"], errors="coerce", utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
        merged = merged.drop_duplicates(["ts", "type", "name"], keep="last").sort_values(["ts", "type", "name"])
        cutoff = pd.Timestamp(now.replace(tzinfo=None)) - pd.Timedelta(days=30)
        _atomic(merged[merged["ts"] >= cutoff], target)
        saved[sector_type] = {"rows": len(merged), "path": target.name}
    if not frames:
        return {"ok": False, "error": "行业/概念资金源无数据", "details": errors}
    out = pd.concat(frames, ignore_index=True)
    if SECTOR_INTRADAY.exists():
        out = pd.concat([pd.read_parquet(SECTOR_INTRADAY), out], ignore_index=True)
    out["ts"] = pd.to_datetime(out["ts"], errors="coerce", utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    out = out.drop_duplicates(["ts", "type", "name"], keep="last").sort_values(["ts", "type", "name"])
    cutoff = pd.Timestamp(now.replace(tzinfo=None)) - pd.Timedelta(days=30)
    out = out[out["ts"] >= cutoff]
    _atomic(out, SECTOR_INTRADAY)
    return {"ok": True, "rows": len(out), "latest": now.isoformat(), "saved": saved, "errors": errors}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default=",".join(DEFAULT_CODES))
    ap.add_argument("--start", default=(datetime.now().replace(year=datetime.now().year - 2)).strftime("%Y%m%d"))
    ap.add_argument("--only", choices=("all", "etf", "sector"), default="all")
    args = ap.parse_args()
    codes = [x.strip().zfill(6) for x in args.codes.split(",") if x.strip()]
    result = {}
    if args.only in ("all", "etf"):
        result["etf"] = collect_etf(codes, args.start)
    if args.only in ("all", "sector"):
        result["sector"] = collect_sector()
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0 if any(x.get("ok") for x in result.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
