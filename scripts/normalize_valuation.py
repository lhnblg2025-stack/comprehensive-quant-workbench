#!/usr/bin/env python3
"""估值文件规范化（骨架收口 P1-2）

问题：data_warehouse/valuation/ 下三种 schema 混存：
  A. 新格式 baostock:  date/peTTM/pbMRQ/psTTM/pcfNcfTTM   (5列)
  B. 旧格式乐咕:      date/close/total_mv/.../pe_ttm/pb/ps/pcf  (10列)
  C. 混合:            date/.../pe_ttm/.../peTTM/pbMRQ/...      (14列, 新数据只写新列)

统一为：date/peTTM/pbMRQ/psTTM/pcfNcfTTM（5列），
新格式优先，旧格式回退，无数据填 NaN。保留 close/total_mv/float_mv 为附加列（可选 --keep-extra）。

用法: python3 normalize_valuation.py [--keep-extra] [--dry-run]
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
VAL = ROOT / "data_warehouse" / "valuation"

STD = ["date", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]
EXTRA = ["close", "total_mv", "float_mv"]


def normalize_one(path: Path, keep_extra: bool) -> tuple[str, str]:
    df = pd.read_parquet(path)
    orig_cols = list(df.columns)
    if set(STD) <= set(orig_cols) and not ({"pe_ttm", "pb", "ps", "pcf"} & set(orig_cols)):
        # 已是纯新格式（且无旧列并存）
        return path.stem, "skip"
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")
    out = pd.DataFrame({"date": df["date"]})
    # 新格式优先，旧格式回退
    for new, old in (("peTTM", "pe_ttm"), ("pbMRQ", "pb"), ("psTTM", "ps"), ("pcfNcfTTM", "pcf")):
        col = new if new in df.columns else (old if old in df.columns else None)
        out[new] = df[col].astype(float) if col else pd.NA
    if keep_extra:
        for c in EXTRA:
            if c in df.columns:
                out[c] = df[c]
    out = out.dropna(subset=STD[1:], how="all")  # 全空行删除
    out.to_parquet(path, index=False)
    return path.stem, "fixed"


def main() -> int:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-extra", action="store_true", help="保留 close/total_mv/float_mv")
    ap.add_argument("--dry-run", action="store_true", help="只统计不写")
    args = ap.parse_args()

    files = sorted(VAL.glob("*.parquet"))
    t0 = time.time()
    n_fixed = n_skip = 0
    for f in files:
        try:
            stem, status = normalize_one(f, args.keep_extra)
            if status == "fixed":
                n_fixed += 1
            else:
                n_skip += 1
            if args.dry_run and status == "fixed":
                print(f"  [DRY] {stem}: 需规范化")
        except Exception as e:  # noqa: BLE001
            print(f"  ERR {f.stem}: {str(e)[:80]}")
    print(f"\n完成: 共{len(files)}个, 规范化{n_fixed}, 已标准{n_skip}, 耗时{time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
