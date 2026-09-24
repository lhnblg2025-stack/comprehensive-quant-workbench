#!/usr/bin/env python3
"""增量更新 K线到最新交易日（腾讯源，稳定免限流）
用法: python3 scripts/update_kline_tencent.py [--days 10] [--symbol 000001] [--limit N]
只更新本地已有股票的最后 days 个交易日，避免全量重抓。
腾讯接口: https://web.ifzq.gtimg.cn/appstock/app/fqkline/get
返回: [日期, 开, 收, 高, 低, 量]（前复权 qfq）
"""

from __future__ import annotations
import logging

import io, sys, time, warnings

# V12.3 审计: 仅 main 运行时包装 stdout(utf-8), 否则会影响被当作库导入的调用方
# (此前模块级 wrap 会让 pytest 等宿主进程的 stdout 被 TextIOWrapper 替换, 破坏捕获)
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
KL = ROOT / "data_warehouse" / "kline"
URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"}


def tencent_symbol(sym: str) -> str:
    return ("bj" if sym.startswith(("4", "8")) or sym.startswith("920")
            else "sh" if sym.startswith(("5", "6", "9")) else "sz") + sym


def fetch_tencent(sym: str, start: str, end: str, count: int = 120) -> pd.DataFrame | None:
    """腾讯日K（前复权）"""
    ts = tencent_symbol(sym)
    params = {"param": f"{ts},day,{start},{end},{count},qfq"}
    r = requests.get(URL, params=params, headers=HEADERS, timeout=15)
    j = r.json()
    d = j.get("data", {}).get(ts, {})
    rows = d.get("qfqday") or d.get("day") or []
    if not rows:
        return None
    # 容错：行可能 6 或 7+ 列（部分含额外字段），只取前 6 列
    rows = [r[:6] for r in rows]
    df = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # 审计 2026-08-16：腾讯接口成交量单位为“手”，全系统统一为“股”
    if "volume" in df.columns:
        df["volume"] = df["volume"] * 100.0
    return df


def latest_expected_trade_date(now=None) -> str:
    """Return the latest expected A-share trading date, not the calendar date."""
    from datetime import date
    current = now.date() if hasattr(now, "date") else (now or datetime.now(timezone(timedelta(hours=8)))).date()
    try:
        from quant_system.market_clock import is_trading_day
        for offset in range(0, 15):
            candidate = current - timedelta(days=offset)
            if is_trading_day(candidate):
                return candidate.strftime("%Y-%m-%d")
    except Exception:
        pass
    while current.weekday() >= 5:
        current -= timedelta(days=1)
    return current.strftime("%Y-%m-%d")


def load_existing(rel: Path) -> pd.DataFrame:
    if rel.exists():
        try:
            return pd.read_parquet(rel)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def fetch_combined(sym: str, start: str, end: str) -> pd.DataFrame | None:
    """腾讯优先（稳定快速，量价齐全），东财兜底（补 amount/turnover 字段）。
    注：腾讯无 amount/turnover，缺失时后续由旧数据延续或 NaN（因子引擎跳过不误报）。
    """
    # 1) 腾讯（快速稳定）
    try:
        df = fetch_tencent(sym, start, end)
        if df is not None and not df.empty:
            return df
    except Exception as e:
        logging.getLogger(__name__).error(f"[update_kline_tencent] 操作失败: {e}", exc_info=True)
    # 2) 东财兜底
    try:
        import akshare as ak
        df = ak.stock_zh_a_hist(symbol=sym, period="daily",
                                start_date=start.replace("-", ""), end_date=end.replace("-", ""), adjust="qfq")
        if df is not None and len(df):
            df = df.rename(columns={
                "日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
                "最低": "low", "成交量": "volume", "成交额": "amount",
                "涨跌幅": "pct_chg", "换手率": "turnover",
            })
            df = df[["date", "open", "close", "high", "low", "volume", "amount", "pct_chg", "turnover"]]
            df["date"] = pd.to_datetime(df["date"])
            return df
    except Exception as e:
        logging.getLogger(__name__).error(f"[update_kline_tencent] 操作失败: {e}", exc_info=True)
    return None


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from quant_platform.runtime import task_start, task_end
        task_start("kline_update")
    except Exception as e:
        logging.getLogger(__name__).error(f"[update_kline_tencent] 操作失败: {e}", exc_info=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=15, help="增量窗口（自然日）")
    ap.add_argument("--limit", type=int, default=0, help="0=全部，>0=只处理前N只（测试）")
    ap.add_argument("--symbol", default=None)
    args = ap.parse_args()

    _CST = timezone(timedelta(hours=8))
    now_cst = datetime.now(_CST)
    today = now_cst.strftime("%Y-%m-%d")
    expected_trade_date = latest_expected_trade_date(now_cst)
    start = (now_cst - timedelta(days=args.days)).strftime("%Y-%m-%d")

    if args.symbol:
        symbols = [args.symbol]
    else:
        symbols = [f.stem for f in sorted(KL.glob("*.parquet")) if f.stem.isdigit() and len(f.stem) == 6]
        symbols.sort()
        # 断点续传：跳过已含最新交易日的
        if not args.limit:
            latest_ok = []
            for f in sorted(KL.glob("*.parquet")):
                if f.stem.isdigit() and len(f.stem) == 6:
                    try:
                        df = pd.read_parquet(f, columns=["date"])
                        if str(pd.to_datetime(df["date"], errors="coerce").max())[:10] >= expected_trade_date:
                            latest_ok.append(f.stem)
                    except Exception as e:
                        logging.getLogger(__name__).error(f"[update_kline_tencent] 操作失败: {e}", exc_info=True)
            if latest_ok:
                latest_ok = set(latest_ok)
                symbols = [s for s in symbols if s not in latest_ok]
                print(f"断点续传: 跳过已最新 {len(latest_ok)} 只", flush=True)
        if args.limit:
            symbols = symbols[:args.limit]
    print(f"待更新: {len(symbols)} 只（窗口 {start}~{today}，目标交易日 {expected_trade_date}）", flush=True)

    ok, fail, updated_rows = 0, [], 0
    t0 = time.time()
    s = requests.Session()

    def work_one(sym: str) -> tuple[str, int, str | None]:
        """返回 (symbol, 新增行数, 错误信息或None)"""
        try:
            df = fetch_combined(sym, start, today)
            if df is None or df.empty:
                return sym, 0, None
            old = load_existing(KL / f"{sym}.parquet")
            if not old.empty:
                old["date"] = pd.to_datetime(old["date"])
                merged = pd.concat([old[~old["date"].isin(df["date"])], df]).sort_values("date").drop_duplicates("date")
            else:
                merged = df
            for c in ["amount", "turnover", "outstanding_share"]:
                if c in old.columns and c not in merged.columns:
                    merged[c] = pd.NA
            # V12.3 审计 P1-1: 腾讯源追加行缺 amount/turnover/pct_chg(全NaN), 原"列名判断"
            # 因 concat 对列取并集恒不成立, 兜底从不触发 → 最新K线被污染(信号/止损/因子输入)。
            # 改为按"值"判断, 补 pct_chg=close.pct_change 显式 fillna(0), amount/turnover 补0。
            if "pct_chg" in merged.columns:
                if merged["pct_chg"].isna().any():
                    merged["pct_chg"] = merged["close"].pct_change() * 100.0
            else:
                merged["pct_chg"] = merged["close"].pct_change() * 100.0
            # W2.5 修复: 缺失保持 NaN（不得把"缺失"记成"0 成交/0 换手/0 涨幅"）
            # pct_chg 首行天然 NaN（无前收），amount/turnover 缺失列已置 pd.NA，均保持 NaN
            # （原 fillna(0.0) 已移除，避免污染 K线/因子/止损输入）
            merged = merged.sort_values("date")
            import os as _os
            _out = KL / f"{sym}.parquet"
            _tmp = _out.with_suffix(".parquet.tmp")
            merged.to_parquet(_tmp, index=False)
            _os.replace(_tmp, _out)
            return sym, len(df), None
        except Exception as e:
            return sym, 0, str(e)[:80]

    done = 0
    for sym in symbols:
        s_, n, err = work_one(sym)
        done += 1
        if err:
            fail.append((sym, err))
        else:
            ok += 1
            updated_rows += n
        if done % 300 == 0:
            print(f"  ... {done}/{len(symbols)} ok={ok} fail={len(fail)} {time.time()-t0:.0f}s", flush=True)
    print(f"\n完成: 更新 {ok} 只, 新增 {updated_rows} 行K线, 失败 {len(fail)}, 耗时 {time.time()-t0:.0f}s", flush=True)
    for s_, e in fail[:15]:
        print(f"  FAIL {s_}: {e}", flush=True)
    try:
        from quant_platform.runtime import task_end
        task_end("kline_update", ok=(len(fail) == 0), detail=f"ok={ok} rows={updated_rows} fail={len(fail)}")
    except Exception as e:
        logging.getLogger(__name__).error(f"[update_kline_tencent] 操作失败: {e}", exc_info=True)
    # 审计 2026-08-16：有失败返回非零，定时任务可感知部分失败
    return 1 if len(fail) else 0


if __name__ == "__main__":
    sys.exit(main())
