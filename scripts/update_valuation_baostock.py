#!/usr/bin/env python3
"""估值增量更新（baostock 源）
- 对 data_warehouse/valuation/{code}.parquet 增量追加最新交易日估值
- baostock 一次 login 循环查询，串行稳定
- 断点续传：跳过已含最新交易日的
- 用法: python3 update_valuation_baostock.py [--symbol 000001] [--limit 100] [--days 30]
"""
import logging
import argparse
import io
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
VAL = ROOT / "data_warehouse" / "valuation"
FIELDS = "date,peTTM,pbMRQ,psTTM,pcfNcfTTM"


def bs_code(sym: str) -> str:
    if sym.startswith("920"):
        return "bj." + sym
    if sym.startswith(("4", "8")):
        return "bj." + sym
    if sym.startswith(("5", "6", "9")):
        return "sh." + sym
    return "sz." + sym


def main() -> int:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    try:
        from quant_platform.runtime import task_start
        task_start("valuation_update")
    except Exception as e:
        logging.getLogger(__name__).error(f"[update_valuation_baostock] 操作失败: {e}", exc_info=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", type=str, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--days", type=int, default=30)
    args = ap.parse_args()

    import baostock as bs

    today = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d")
    start = (pd.Timestamp.now(tz="Asia/Shanghai") - pd.Timedelta(days=args.days)).strftime("%Y-%m-%d")

    if args.symbol:
        symbols = [args.symbol]
    else:
        symbols = [f.stem for f in sorted(VAL.glob("*.parquet")) if f.stem.isdigit() and len(f.stem) == 6]
        symbols.sort()
        # 断点续传：跳过已含最新交易日的
        latest_ok = []
        for f in sorted(VAL.glob("*.parquet")):
            if f.stem.isdigit() and len(f.stem) == 6:
                try:
                    df = pd.read_parquet(f, columns=["date"])
                    if str(df["date"].max())[:10] >= today:
                        latest_ok.append(f.stem)
                except Exception as e:
                    logging.getLogger(__name__).error(f"[update_valuation_baostock] 操作失败: {e}", exc_info=True)
        if latest_ok:
            latest_ok = set(latest_ok)
            symbols = [s for s in symbols if s not in latest_ok]
            print(f"断点续传: 跳过已最新 {len(latest_ok)} 只", flush=True)
        if args.limit:
            symbols = symbols[:args.limit]

    print(f"待更新: {len(symbols)} 只（窗口 {start}~{today}）", flush=True)

    lg = bs.login()
    if lg.error_code != "0":
        print(f"baostock login 失败: {lg.error_msg}", flush=True)
        return 1
    t0 = time.time()
    ok = fail = updated_rows = 0
    fails = []
    try:
        for i, sym in enumerate(symbols):
            try:
                rs = None
                for _attempt in range(3):
                    rs = bs.query_history_k_data_plus(
                        bs_code(sym), FIELDS,
                        start_date=start, end_date=today,
                        frequency="d", adjustflag="2")
                    if rs is not None and getattr(rs, "error_code", None) == "0":
                        break
                    time.sleep(1 + _attempt)
                if rs is None or getattr(rs, "error_code", None) != "0":
                    raise RuntimeError("baostock 查询失败")
                rows = []
                # V2 (2026-08-07): rs.next() 阻塞无超时 → 服务器挂起时进程卡死。
                # 用 signal.alarm 给整段读行加 60s 硬超时（Linux/Unix 生效）。
                # Windows 无 SIGALRM，退化无超时，避免 import 即抛 AttributeError。
                import signal as _sig

                def _timeout_handler(*_a):
                    raise TimeoutError("baostock rs.next() 超时 60s")

                _old_handler = None
                if hasattr(_sig, "SIGALRM"):
                    _old_handler = _sig.signal(_sig.SIGALRM, _timeout_handler)
                    _sig.alarm(60)
                try:
                    while (rs.error_code == "0") and rs.next():
                        rows.append(rs.get_row_data())
                finally:
                    if hasattr(_sig, "SIGALRM"):
                        _sig.alarm(0)
                        _sig.signal(_sig.SIGALRM, _old_handler)
                if not rows:
                    ok += 1
                else:
                    new = pd.DataFrame(rows, columns=rs.fields)
                    for c in ["peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]:
                        new[c] = pd.to_numeric(new[c], errors="coerce")
                    new["date"] = pd.to_datetime(new["date"])
                    new = new[FIELDS.split(",")].dropna(subset=["date"])
                    old = pd.read_parquet(VAL / f"{sym}.parquet")
                    if not old.empty:
                        old["date"] = pd.to_datetime(old["date"])
                        merged = pd.concat([old[~old["date"].isin(new["date"])], new]).sort_values("date").drop_duplicates("date")
                    else:
                        merged = new
                    import os as _os
                    _out = VAL / f"{sym}.parquet"
                    _tmp = _out.with_suffix(".parquet.tmp")
                    merged.to_parquet(_tmp, index=False)
                    _os.replace(_tmp, _out)
                    ok += 1
                    updated_rows += len(new)
            except Exception as e:  # noqa: BLE001
                fail += 1
                fails.append((sym, str(e)[:80]))
            if (i + 1) % 500 == 0:
                print(f"  ... {i+1}/{len(symbols)} ok={ok} fail={fail} {time.time()-t0:.0f}s", flush=True)
    finally:
        bs.logout()
    print(f"\n完成: 更新 {ok} 只, 新增 {updated_rows} 行估值, 失败 {fail}, 耗时 {time.time()-t0:.0f}s", flush=True)
    for s_, e in fails[:15]:
        print(f"  FAIL {s_}: {e}", flush=True)
    try:
        from quant_platform.runtime import task_end
        task_end("valuation_update", ok=(fail == 0), detail=f"ok={ok} rows={updated_rows} fail={fail}")
    except Exception as e:
        logging.getLogger(__name__).error(f"[update_valuation_baostock] 操作失败: {e}", exc_info=True)
    # 审计 2026-08-16：有失败返回非零，定时任务可感知部分失败
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
