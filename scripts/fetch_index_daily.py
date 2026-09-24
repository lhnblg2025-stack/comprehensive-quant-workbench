#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""指数日线落盘 — data_warehouse/market/index_daily_{name}.parquet

抓取主要指数日线（B2 数据补齐）:
  - 主源: akshare stock_zh_index_daily（新浪，symbol: sh000016 上证50 / sh000300
    沪深300 / sz399006 创业板指 / sh000688 科创50 / sh000905 中证500 / sh000852 中证1000）
  - 兜底: akshare stock_zh_index_daily_em（东财，同代码需 sh/sz 前缀，否则返回空表）
落盘列: date/open/high/low/close/volume（date 为 date 类型，UTC+8）
幂等增量: 已有文件按 date 合并去重（keep=last）；--force 全量重抓覆盖。
输出: 每个指数新鲜度摘要（as_of / 行数 / 新增）。

用法:
  python3 scripts/fetch_index_daily.py
  python3 scripts/fetch_index_daily.py --force
  python3 scripts/fetch_index_daily.py --index 沪深300
"""
from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data_warehouse" / "market"
COLS = ["date", "open", "high", "low", "close", "volume"]

# (显示名, 新浪 symbol)；东财代码见 EM_CODES（代码不带前缀，抓取时补 sh/sz）
# 2026-08-24 修复: 补上证指数/深证成指 —— 决策引擎判定“上证跌破MA300”依赖这两个主指数，
# 缺失导致长期趋势门控永远不触发。
INDEX_SPECS = [
    ("上证指数", "sh000001"),
    ("深证成指", "sz399001"),
    ("上证50",   "sh000016"),
    ("沪深300",  "sh000300"),
    ("创业板指", "sz399006"),
    ("科创50",   "sh000688"),
    ("中证500",  "sh000905"),
    ("中证1000", "sh000852"),
    ("红利指数", "sh000015"),
]
EM_CODES = {
    "上证指数": "000001",
    "深证成指": "399001",
    "上证50": "000016",
    "沪深300": "000300",
    "创业板指": "399006",
    "科创50": "000688",
    "中证500": "000905",
    "中证1000": "000852",
    "红利指数": "000015",
}


def _em_symbol(code: str) -> str:
    """东财 stock_zh_index_daily_em 需市场前缀：3 开头深市(sz)，其余沪市(sh)。"""
    return f"sz{code}" if code.startswith("3") else f"sh{code}"


def _fetch_sina(symbol: str) -> pd.DataFrame | None:
    """新浪源 stock_zh_index_daily（date/open/high/low/close/volume 小写列）。"""
    import akshare as ak

    old = socket.getdefaulttimeout()
    socket.setdefaulttimeout(15)
    try:
        df = ak.stock_zh_index_daily(symbol=symbol)
    finally:
        socket.setdefaulttimeout(old)
    return df if df is not None and not df.empty else None


def _fetch_em(symbol: str) -> pd.DataFrame | None:
    """东财兜底 stock_zh_index_daily_em（列 date/open/close/high/low/volume/amount）。"""
    import akshare as ak

    old = socket.getdefaulttimeout()
    socket.setdefaulttimeout(15)
    try:
        df = ak.stock_zh_index_daily_em(symbol=symbol)
    finally:
        socket.setdefaulttimeout(old)
    return df if df is not None and not df.empty else None


def normalize(df: pd.DataFrame) -> pd.DataFrame | None:
    """统一列/类型：date(date) + OHLC/volume(float)，按 date 去重升序。"""
    if df is None or df.empty:
        return None
    if not set(COLS).issubset(df.columns):
        return None
    out = df[COLS].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date", "close"])
    for c in COLS[1:]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["volume"] = out["volume"].fillna(0.0)
    out = out.drop_duplicates("date", keep="last").sort_values("date")
    return out.reset_index(drop=True)


def merge_and_save(name: str, new_df: pd.DataFrame, force: bool
                   ) -> tuple[pd.DataFrame, int]:
    """增量合并落盘：已有文件按 date 合并（keep=last）；--force 全量覆盖。

    Returns: (最终 DataFrame, 新增行数)
    """
    p = OUT_DIR / f"index_daily_{name}.parquet"
    if force or not p.exists():
        final = new_df
        added = len(new_df)
    else:
        old = normalize(pd.read_parquet(p))
        if old is None or old.empty:
            final = new_df
            added = len(new_df)
        else:
            old_dates = set(old["date"])
            merged = pd.concat([old, new_df], ignore_index=True)
            final = (merged.drop_duplicates("date", keep="last")
                     .sort_values("date").reset_index(drop=True))
            added = int((~final["date"].isin(old_dates)).sum())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import os as _os
    _tmp = p.with_suffix(".parquet.tmp")
    final.to_parquet(_tmp, index=False)
    _os.replace(_tmp, p)
    return final, added


def main() -> int:
    ap = argparse.ArgumentParser(description="指数日线落盘（新浪源 + 东财兜底，增量合并）")
    ap.add_argument("--force", action="store_true", help="全量重抓覆盖（默认增量合并）")
    ap.add_argument("--index", default=None, help="仅抓指定指数（显示名，如 沪深300）")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results: list[tuple[str, pd.Timestamp | None, int, int, str]] = []

    for name, sina_sym in INDEX_SPECS:
        if args.index and name != args.index:
            continue
        df = None
        source = ""
        err = ""
        try:
            df = _fetch_sina(sina_sym)
            if df is not None:
                source = "sina"
        except Exception as e:
            err = f"sina: {type(e).__name__} {str(e)[:80]}"
        if df is None:
            try:
                df = _fetch_em(_em_symbol(EM_CODES[name]))
                if df is not None:
                    source = "em" if not err else f"em(sina失败:{err[:40]})"
            except Exception as e:
                err = (err + "; " if err else "") + f"em: {type(e).__name__} {str(e)[:80]}"
        if df is None:
            results.append((name, None, 0, 0, "失败: " + (err or "空数据")))
            print(f"[失败] {name:<6} 抓取不可用 — {err or '空数据'}", flush=True)
            continue
        try:
            norm = normalize(df)
            if norm is None or norm.empty:
                results.append((name, None, 0, 0, "失败: 空数据"))
                print(f"[失败] {name:<6} 空数据", flush=True)
                continue
            final, added = merge_and_save(name, norm, args.force)
            asof = final["date"].max()
            results.append((name, asof, len(final), added, source))
            print(f"[落盘] {name:<6} 源={source:<8} 新增={added:<5} 总行数={len(final):<6} "
                  f"截至={asof.date()} -> {OUT_DIR / f'index_daily_{name}.parquet'}", flush=True)
        except Exception as e:
            results.append((name, None, 0, 0, "失败: " + str(e)[:120]))
            print(f"[失败] {name:<6} 落盘异常 — {str(e)[:120]}", flush=True)
        time.sleep(0.3)  # 新浪/东财限流保护

    print("\n[新鲜度摘要]")
    print(f"{'指数':<8}{'数据截至':<14}{'行数':<8}{'新增':<6}状态")
    for name, asof, n, added, status in results:
        if asof is None:
            print(f"{name:<8}{'—':<14}{'—':<8}{'—':<6}{status}")
        else:
            print(f"{name:<8}{str(asof.date()):<14}{n:<8}{added:<6}{status}")
    ok = sum(1 for r in results if r[1] is not None)
    print(f"\n成功 {ok}/{len(results)}；耗时 {time.time() - t0:.1f}s", flush=True)

    # 2026-08-21 审计修复: 主文件 index_daily.parquet 同步（消费方 rs_strength/leader_follower
    # 读主文件，迁移后只写分文件导致主文件停更 08-14）。写完分文件后把沪深300合并进主文件。
    try:
        from pathlib import Path as _P
        import os as _os2
        _m = _P(__file__).resolve().parent.parent / "data_warehouse" / "market"
        _src = _m / "index_daily_沪深300.parquet"
        _main = _m / "index_daily.parquet"
        if _src.exists():
            _nd = normalize(pd.read_parquet(_src))
            if _nd is not None and not _nd.empty:
                if _main.exists():
                    _od = normalize(pd.read_parquet(_main))
                    _all = (pd.concat([_od, _nd], ignore_index=True)
                            .drop_duplicates("date", keep="last").sort_values("date"))
                else:
                    _all = _nd
                _tmp = _main.with_suffix(".parquet.tmp")
                _all.to_parquet(_tmp, index=False)
                _os2.replace(_tmp, _main)
                print(f"[主文件同步] index_daily.parquet 最新={_all['date'].max().date()} "
                      f"总行数={len(_all)}", flush=True)
            else:
                print("[主文件同步] 沪深300 分文件为空，跳过", flush=True)
        else:
            print("[主文件同步] 无 index_daily_沪深300.parquet，跳过", flush=True)
    except Exception as _e:
        print(f"[主文件同步] 失败: {_e}", flush=True)
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
