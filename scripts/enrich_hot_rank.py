#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""enrich_hot_rank — 热榜涨跌幅补全（qt.gtimg.cn 批量查询）

背景: 腾讯云爬虫 crawler_hot_rank.py 只落 code/rank/rank_change/hist_rank 四列，
无涨跌幅。本脚本读 hot_rank/hot_rank_*.parquet（默认 data_warehouse，最新 N 个文件），
对缺 pct_chg 列的文件用腾讯行情 qt.gtimg.cn 批量查询补全（幂等）:
  代码格式 SZ301308/SH600664 → 提取6位 → sz/sh 前缀 → 每批 20 只 →
  GBK 响应按 '~' 切分取 parts[32]（涨跌幅 %）。
已有 pct_chg 列 → 直接跳过；任何失败 → 静默保留原文件（不写回）。

用法:
  python3 scripts/enrich_hot_rank.py                 # 最新 3 个文件
  python3 scripts/enrich_hot_rank.py --files 1       # 只处理最新 1 个
  HOT_RANK_DIR=/tmp python3 scripts/enrich_hot_rank.py   # 目录覆盖（自查用）
  QT_GTIMG_BASE=http://127.0.0.1:PORT/q= python3 ... # 行情端点覆盖（测试用）
"""
from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "data_warehouse" / "hot_rank"
QT_BASE = os.environ.get("QT_GTIMG_BASE", "http://qt.gtimg.cn/q=")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
BATCH = 20          # 每批 20 只
TIMEOUT = 10
PCT_COL = "pct_chg"


def _six_digits(raw) -> str | None:
    """任意格式 → 6 位纯数字；无法归一化返回 None。"""
    m = re.search(r"(\d{6})", str(raw or ""))
    return m.group(1) if m else None


def _qt_symbol(raw) -> str | None:
    """'SZ301308'/'SH600664'/'301308' → 'sz301308'/'sh600664'（qt 行情代码）。"""
    six = _six_digits(raw)
    if six is None:
        return None
    s = str(raw).strip().upper()
    if s.startswith("SZ"):
        return f"sz{six}"
    if s.startswith("SH"):
        return f"sh{six}"
    if s.startswith("BJ"):
        return f"bj{six}"
    if six[0] in ("6", "9"):
        return f"sh{six}"
    if six[0] in ("0", "3"):
        return f"sz{six}"
    if six[0] in ("4", "8"):
        return f"bj{six}"
    return None


def _fetch_pct_chg(symbols: list[str]) -> dict[str, float]:
    """批量查询 qt.gtimg.cn → {symbol: pct_chg}；请求失败抛异常（由调用方降级）。"""
    url = QT_BASE + ",".join(symbols)
    r = requests.get(url, headers={"User-Agent": UA, "Referer": "https://gu.qq.com/"},
                     timeout=TIMEOUT)
    r.raise_for_status()
    text = r.content.decode("gbk", errors="replace")
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip().removeprefix("v_")
        if '"' not in line:
            continue
        payload = line.split('"', 1)[1].rsplit('"', 1)[0]
        parts = payload.split("~")
        if len(parts) > 32 and parts[32].strip():
            try:
                out[key] = float(parts[32])
            except ValueError:
                logging.getLogger("enrich_hot_rank").warning(
                    f"[enrich_hot_rank] 涨跌幅解析失败 key={key} raw={parts[32]!r}")
    return out


def enrich_one(path: Path) -> dict:
    """单文件补涨跌幅：缺 pct_chg 列才补（幂等）；失败静默保留原文件。"""
    res = {"file": path.name}
    try:
        df = pd.read_parquet(path)
    except Exception as e:  # noqa: BLE001
        res.update({"status": "error", "n": 0, "filled": 0,
                    "note": f"读取失败 {type(e).__name__}: {str(e)[:80]}（保留原文件）"})
        return res
    if df is None or len(df) == 0 or "code" not in df.columns:
        res.update({"status": "skip", "n": int(len(df)) if df is not None else 0,
                    "filled": 0, "note": "空表或缺 code 列"})
        return res
    if PCT_COL in df.columns:
        res.update({"status": "skip", "n": int(len(df)),
                    "filled": int(df[PCT_COL].notna().sum()),
                    "note": "已有 pct_chg 列（幂等跳过）"})
        return res
    symbols = [_qt_symbol(c) for c in df["code"].tolist()]
    pct_map: dict[str, float] = {}
    for i in range(0, len(symbols), BATCH):
        batch = symbols[i:i + BATCH]
        try:
            pct_map.update(_fetch_pct_chg(batch))
        except Exception as e:  # noqa: BLE001
            res.update({"status": "error", "n": int(len(df)), "filled": 0,
                        "note": f"行情请求失败（批次 {i // BATCH + 1}）"
                                f"{type(e).__name__}: {str(e)[:60]}（保留原文件）"})
            return res
    df = df.reset_index(drop=True)
    df["_sym"] = symbols
    df[PCT_COL] = pd.to_numeric(df["_sym"].map(pct_map), errors="coerce")
    df = df.drop(columns=["_sym"])
    try:
        df.to_parquet(path, index=False)
    except Exception as e:  # noqa: BLE001
        res.update({"status": "error", "n": int(len(df)), "filled": 0,
                    "note": f"写回失败 {type(e).__name__}: {str(e)[:80]}（保留原文件）"})
        return res
    res.update({"status": "ok", "n": int(len(df)),
                "filled": int(df[PCT_COL].notna().sum()),
                "note": ""})
    return res


def enrich_hot_rank(data_dir: Path | None = None, files: int = 3) -> dict:
    """目录级入口：最新 files 个 hot_rank_*.parquet 补涨跌幅（异常不抛出）。"""
    try:
        data_dir = data_dir or Path(os.environ.get("HOT_RANK_DIR", DEFAULT_DIR))
        paths = sorted(data_dir.glob("hot_rank_*.parquet"))
        if not paths:
            return {"dir": str(data_dir), "files": [], "ok": 0, "skipped": 0, "failed": 0}
        results = [enrich_one(p) for p in paths[-max(1, files):]]
        return {"dir": str(data_dir), "files": results,
                "ok": sum(1 for r in results if r["status"] == "ok"),
                "skipped": sum(1 for r in results if r["status"] == "skip"),
                "failed": sum(1 for r in results if r["status"] == "error")}
    except Exception as e:  # noqa: BLE001
        return {"dir": str(data_dir), "files": [], "ok": 0, "skipped": 0,
                "failed": 0, "note": f"{type(e).__name__}: {str(e)[:100]}"}


def enrich_latest(files: int = 3) -> dict:
    """pull_cloud_data 内嵌调用入口（mod.enrich_latest()）：默认目录最新 files 个文件补涨跌幅。"""
    summary = enrich_hot_rank(None, files)
    for r in summary["files"]:
        note = f" {r['note']}" if r.get("note") else ""
        print(f"[enrich_hot_rank:{r['file']}] {r['status']}（n={r.get('n')}, "
              f"filled={r.get('filled')}）{note}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="热榜涨跌幅补全（qt.gtimg.cn 批量查询，幂等）")
    ap.add_argument("--files", type=int, default=int(os.environ.get("HOT_RANK_FILES", "3")),
                    help="处理最新 N 个文件（默认 3）")
    ap.add_argument("--dir", default=None,
                    help="覆盖 hot_rank 目录（默认 $HOT_RANK_DIR 或 data_warehouse/hot_rank）")
    args = ap.parse_args()

    if args.dir:
        summary = enrich_hot_rank(Path(args.dir), args.files)
    else:
        summary = enrich_latest(args.files)
    print(f"[enrich_hot_rank] 目录 {summary['dir']} | "
          f"ok {summary['ok']} / skip {summary['skipped']} / failed {summary['failed']}")


if __name__ == "__main__":
    main()
