#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""稳定滚动更新器：逐域更新、重试、断点记录、目标日期验收。

本脚本只负责调度现有采集器，不在页面请求中抓取数据。
失败域保留旧产物，并在 generated/data_update_status.json 标记 failed/stale。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "generated"
STATUS = GEN / "data_update_status.json"
LOCK = Path(os.environ.get("QUANT_DATA_LOCK", os.environ.get("QUANT_LOCK_DIR", str(GEN / "locks")))) / "rolling_data_update.lock"
CST = timezone(timedelta(hours=8))


def _target_day() -> str:
    try:
        from quant_system.market_clock import latest_completed_trading_day
        return latest_completed_trading_day().isoformat()
    except Exception:
        now = datetime.now(CST).date()
        while now.weekday() >= 5:
            now -= timedelta(days=1)
        return now.isoformat()


def _load() -> dict:
    try:
        return json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {"schema": "data_update_status/v1", "domains": {}}
    except (OSError, ValueError):
        return {"schema": "data_update_status/v1", "domains": {}}


def _latest_for(domain: str) -> str | None:
    import pandas as pd
    candidates = {
        "kline": ROOT / "data_warehouse" / "kline" / "002714.parquet",
        "kline_hfq": ROOT / "data_warehouse" / "kline_hfq" / "002714.parquet",
        "industry": ROOT / "data_warehouse" / "industry" / "sw_first_hist.parquet",
        "financial": ROOT / "data_warehouse" / "financial" / "002714.parquet",
        "patterns": GEN / "pattern_report_latest.md",
        "industry_chain": ROOT / "data_warehouse" / "market" / "commodity__crude.parquet",
    }
    path = candidates.get(domain)
    if not path or not path.exists():
        if domain == "patterns":
            files = sorted(GEN.glob("pattern_report_*.md"))
            path = files[-1] if files else None
        elif domain == "industry_chain":
            files = sorted(GEN.glob("industry_chain_history_*.json"))
            path = files[-1] if files else None
    if not path or not path.exists():
        return None
    try:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
            for col in ("date", "日期", "报告期", "trade_date"):
                if col in df.columns:
                    values = pd.to_datetime(df[col], errors="coerce").dropna()
                    return values.max().strftime("%Y-%m-%d") if not values.empty else None
            # 财务宽表的报告期列
            dates = [str(c) for c in df.columns if str(c).isdigit() and len(str(c)) == 8]
            return max(dates)[:4] + "-" + max(dates)[4:6] + "-" + max(dates)[6:] if dates else None
        import re
        match = re.search(r"20\d{2}[-_]\d{2}[-_]\d{2}", path.stem)
        return match.group(0).replace("_", "-") if match else None
    except Exception:
        return None


TASKS = {
    "kline": ([sys.executable, str(ROOT / "scripts/update_kline_tencent.py"), "--days", "15"], 1800),
    # 当前采集器的真实产物是前复权；后复权不能用前复权复制冒充。
    "kline_hfq": (None, 0),
    "industry": ([sys.executable, str(ROOT / "scripts/update_industry_sw.py"), "--only", "info,cons,spot,hist"], 2400),
    "financial": ([sys.executable, str(ROOT / "scripts/update_financial_quarterly.py"), "--pool", "watch"], 1800),
    "industry_chain": ([sys.executable, str(ROOT / "scripts/update_industry_commodities.py")], 600),
    "patterns": ([sys.executable, "-m", "quant_system.analysis_core.pattern_engine", "--report"], 1800),
}


def _run_domain(domain: str, command: list[str] | None, timeout: int, retries: int) -> dict:
    if command is None:
        return {"status": "not_supported", "attempt": 0, "latest": _latest_for(domain),
                "error": "当前没有后复权采集器，不能用前复权数据冒充"}
    last_error = ""
    for attempt in range(1, retries + 1):
        started = time.time()
        try:
            proc = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout)
            if proc.returncode == 0:
                return {"status": "ok", "attempt": attempt, "duration": round(time.time() - started, 1),
                        "latest": _latest_for(domain), "stdout": proc.stdout[-500:]}
            last_error = (proc.stderr or proc.stdout or f"exit {proc.returncode}")[-500:]
        except subprocess.TimeoutExpired:
            last_error = f"第{attempt}次超时({timeout}秒)"
        if attempt < retries:
            time.sleep(min(30, 3 * attempt))
    return {"status": "failed", "attempt": retries, "latest": _latest_for(domain), "error": last_error}


def main() -> int:
    ap = argparse.ArgumentParser(description="统一滚动更新数据域")
    ap.add_argument("--domains", default=",".join(TASKS), help="逗号分隔的数据域")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    domains = [x.strip() for x in args.domains.split(",") if x.strip()]
    unknown = [x for x in domains if x not in TASKS]
    if unknown:
        ap.error(f"未知数据域: {','.join(unknown)}")
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w") as lock_file:
        import fcntl
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"status": "skipped", "reason": "已有更新实例"}, ensure_ascii=False))
            return 75
        target = _target_day()
        state = _load()
        state.update({"schema": "data_update_status/v1", "target_day": target,
                      "started_at": datetime.now(CST).isoformat(timespec="seconds"),
                      "finished_at": None})
        state.setdefault("domains", {})
        results = {}
        for domain in domains:
            command, timeout = TASKS[domain]
            results[domain] = {"status": "planned", "command": command, "timeout": timeout} if args.dry_run else _run_domain(domain, command, timeout, max(1, args.retries))
            state["domains"][domain] = {**results[domain], "updated_at": datetime.now(CST).isoformat(timespec="seconds"), "target_day": target}
            if not args.dry_run:
                STATUS.parent.mkdir(parents=True, exist_ok=True)
                STATUS.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        # 未选中的域不继承旧的 failed/planned，避免把历史状态误报为本次每日失败。
        selected = set(domains)
        if not args.dry_run:
            if "financial" not in selected:
                # 季度数据虽低频，但断点任务的最后结果必须进入统一状态，不能用旧
                # 文件报告掩盖刚完成的真实批量更新。
                fin_cp = ROOT / "generated" / "financial_update_checkpoint.json"
                fin_result = {}
                try:
                    fin_result = json.loads(fin_cp.read_text(encoding="utf-8")).get("last_result") or {}
                except (OSError, ValueError):
                    pass
                state["domains"]["financial"] = {
                    "status": "ok" if fin_result else "deferred",
                    "schedule": "低频季度更新（断点续传）",
                    "latest": _latest_for("financial"),
                    "target_day": target,
                    "last_result": fin_result or None,
                }
            if "patterns" not in selected:
                state["domains"]["patterns"] = {"status": "deferred", "schedule": "低频形态扫描", "latest": _latest_for("patterns"), "target_day": target}
            if "kline_hfq" not in selected:
                state["domains"]["kline_hfq"] = {"status": "not_supported", "latest": _latest_for("kline_hfq"), "target_day": target, "error": "当前没有后复权采集器"}
        state["finished_at"] = datetime.now(CST).isoformat(timespec="seconds")
        if not args.dry_run:
            STATUS.parent.mkdir(parents=True, exist_ok=True)
            STATUS.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    degraded = [domain for domain, result in results.items()
                if result.get("status") in {"failed", "not_supported"}]
    print(json.dumps({"status": "degraded" if degraded else "ok", "target_day": target,
                      "results": results, "degraded_domains": degraded,
                      "status_file": str(STATUS)}, ensure_ascii=False))
    return 2 if degraded else 0


if __name__ == "__main__":
    raise SystemExit(main())
