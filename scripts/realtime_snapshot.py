#!/usr/bin/env python3
"""盘中实时数据爬取（腾讯批量接口，每5分钟由 cron 触发）
- 拉全市场 A股实时报价（含价格/涨跌幅/成交量/成交额/换手率/量比/52周位置等）
- 保存为带时间戳的 parquet 快照: data_warehouse/realtime_snapshot/YYYYMMDD/HHMMSS.parquet
- 只保存最近 N 个快照（默认 60 个 = 5分钟×5小时），防磁盘膨胀
- 非交易时段静默退出

用法: python3 realtime_snapshot.py [--keep 60]
"""
from __future__ import annotations
import logging

import argparse
import io
import os
import sys
import time as _time
import urllib.request
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data_warehouse" / "realtime_snapshot"
CST = timezone(timedelta(hours=8))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    from quant_system.utils import now_cst as _now_cst_impl
except ImportError:
    _now_cst_impl = None

# CPU 节流（用户硬规则：后台轮询必须用 CpuThrottle）
try:
    from quant_system.cpu_throttle import CpuThrottle

    _throttle = CpuThrottle(min_interval=0.3)
except Exception:  # noqa: BLE001
    _throttle = None

BATCH = 500  # 腾讯单请求最大安全批数（414 错误上限）
URL = "https://qt.gtimg.cn/q="


def now_cst() -> datetime:
    """D1 收敛: 转发 quant_system.utils.now_cst（import 失败时回退原实现）。"""
    if _now_cst_impl is not None:
        return _now_cst_impl()
    return datetime.now(CST)


def is_trading_minute(allow_after: bool = True) -> bool:
    """交易时段：9:30-11:30 / 13:00-15:00（含14:57-15:00集合竞价尾段）。
    allow_after: 是否允许 15:00-15:05 的收盘确认窗口。"""
    t = now_cst()
    if t.weekday() >= 5:
        return False
    hm = t.hour * 100 + t.minute
    if 930 <= hm <= 1130:
        return True
    if 1300 <= hm <= 1500:
        return True
    if allow_after and 1501 <= hm <= 1505:
        return True
    return False


def load_symbols() -> list[str]:
    """从 kline 目录取全部 A股代码（含北交所），按交易所前缀分组。"""
    syms = sorted(
        f.stem for f in (ROOT / "data_warehouse" / "kline").glob("*.parquet")
        if f.stem.isdigit() and len(f.stem) == 6
    )
    out = []
    for s in syms:
        if s.startswith(("6", "9")):
            out.append("sh" + s)
        elif s.startswith(("4", "8")):
            out.append("bj" + s)
        else:
            out.append("sz" + s)
    return out


def safe_float(x: str) -> float | None:
    """D1/D组收敛: 异口径保留。

    与 quant_system.utils.safe_float（默认清洗 %/,/中文单位、过滤 inf/bool 的富语义）
    不同：本地为无清洗简单版（nan→None、inf 放行），故保留原实现。
    """
    try:
        v = float(x)
        return v if v == v else None  # nan → None
    except Exception:  # noqa: BLE001
        return None


def fetch_all(symbols: list[str]) -> pd.DataFrame:
    """分 batch 拉腾讯实时行情，返回 DataFrame。"""
    rows: list[dict] = []
    errors = 0
    for i in range(0, len(symbols), BATCH):
        batch = symbols[i : i + BATCH]
        url = URL + ",".join(batch)
        text = ""
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    text = resp.read().decode("gbk", "ignore")
                break
            except Exception:  # noqa: BLE001
                errors += 1
                if attempt < 2:
                    _time.sleep(1 + attempt)
        if not text:
            continue
        for item in text.split(";\n"):
            item = item.strip().rstrip(";")
            if '="' not in item:
                continue
            key = item.split("=", 1)[0].strip()
            if not key.startswith("v_"):
                continue
            code_full = key[2:]  # sh600000 / sz000001 / bj...
            quote = item.split('="', 1)[1].rstrip('"')
            p = quote.split("~")
            if len(p) < 50:
                continue
            rows.append({
                "code": code_full,
                "name": p[1],
                "price": safe_float(p[3]),
                "pre_close": safe_float(p[4]),
                "open": safe_float(p[5]),
                "volume_lot": safe_float(p[6]),
                "amount_wan": safe_float(p[37]),
                "high": safe_float(p[33]),
                "low": safe_float(p[34]),
                "pct_chg": safe_float(p[32]),
                "turnover": safe_float(p[38]),
                "pe_ttm": safe_float(p[39]),
                "pb": safe_float(p[46]),
                "ts": p[30],
            })
    df = pd.DataFrame(rows)
    return df, errors


def main() -> int:
    if hasattr(os, "nice"):  # Windows 无 os.nice（POSIX-only），跳过降优先级
        os.nice(10)  # 用户硬规则：后台进程降优先级
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")

    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=60, help="保留最近N个快照")
    ap.add_argument("--force", action="store_true", help="非交易时段也执行（测试用）")
    args = ap.parse_args()

    if not is_trading_minute() and not args.force:
        print(f"[{now_cst():%H:%M:%S}] 非交易时段，退出", flush=True)
        return 0
    symbols = load_symbols()
    t0 = _time.time()
    df, errors = fetch_all(symbols)
    if df.empty:
        print(f"[{now_cst():%H:%M:%S}] 无数据（errors={errors}），退出", flush=True)
        return 1

    # 保存带时间戳快照
    day_dir = OUT / now_cst().strftime("%Y%m%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    stamp = now_cst().strftime("%H%M%S")
    path = day_dir / f"{stamp}.parquet"
    df.to_parquet(path, index=False)

    # 清理旧快照（保留最近 N 个）
    snaps = sorted(day_dir.glob("*.parquet"))
    for f in snaps[:-args.keep]:
        try:
            f.unlink()
        except Exception as e:  # noqa: BLE001
            logging.getLogger(__name__).error(f"[realtime_snapshot] 操作失败: {e}", exc_info=True)

    print(
        f"[{now_cst():%H:%M:%S}] 快照 {len(df)}只 -> {path.name} "
        f"({_time.time()-t0:.1f}s, errors={errors})",
        flush=True,
    )
    try:
        from quant_platform.runtime import task_start, task_end
        task_start("realtime_snapshot")
        task_end("realtime_snapshot", ok=True, detail=f"{len(df)}只 {stamp}")
    except Exception as e:
        logging.getLogger(__name__).error(f"[realtime_snapshot] 操作失败: {e}", exc_info=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
