#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""无人值守盘后编排：失败即阻断依赖下游，并写机器可读状态。"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CST = timezone(timedelta(hours=8))
GEN = ROOT / "generated"


def _status_dir() -> Path:
    return GEN / "runtime_status"


def _run_stage(name: str, command: list[str], timeout: int, *, env: dict[str, str] | None = None) -> dict:
    started = time.time()
    try:
        proc = subprocess.run(command, cwd=str(ROOT), text=True, capture_output=True,
                              timeout=timeout, env=env)
        return {"name": name, "status": "ok" if proc.returncode == 0 else "failed",
                "returncode": proc.returncode, "duration_seconds": round(time.time() - started, 2),
                "stdout_tail": proc.stdout[-1200:], "stderr_tail": proc.stderr[-1200:]}
    except subprocess.TimeoutExpired as exc:
        return {"name": name, "status": "timeout", "returncode": None,
                "duration_seconds": round(time.time() - started, 2),
                "stdout_tail": str(exc.stdout or "")[-1200:], "stderr_tail": str(exc.stderr or "")[-1200:]}
    except OSError as exc:
        return {"name": name, "status": "failed", "returncode": None,
                "duration_seconds": round(time.time() - started, 2),
                "stdout_tail": "", "stderr_tail": str(exc)[-1200:]}


def _target_date(value: str | None) -> str:
    if value:
        try:
            parsed = datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError as exc:
            raise SystemExit(f"invalid --date: {value}") from exc
        if parsed.isoformat() != value:
            raise SystemExit(f"invalid --date: {value}")
        return value
    try:
        from quant_system.market_clock import latest_completed_trading_day
        return latest_completed_trading_day().isoformat()
    except Exception as exc:
        raise SystemExit(f"cannot determine completed trading day: {exc}") from exc


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _write_status(status: dict) -> Path:
    path = _status_dir() / "after_close.json"
    _atomic_write(path, json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    return path


def _write_manifest(run_dir: Path, manifest: dict) -> Path:
    path = run_dir / "manifest.json"
    _atomic_write(path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="统一盘后数据与决策编排")
    parser.add_argument("--date", help="分析日期 YYYY-MM-DD")
    parser.add_argument("--run-id", help="外层编排生成的运行标识")
    parser.add_argument("--push", action="store_true", help="显式请求外部投递")
    parser.add_argument("--skip-weekly", action="store_true")
    args = parser.parse_args()
    date = _target_date(args.date)
    from quant_system.market_clock import is_trading_day
    if not is_trading_day(date):
        raise SystemExit(f"not a trading day: {date}")

    run_id = args.run_id or f"after-close-{date}-{datetime.now(CST).strftime('%H%M%S')}-{uuid.uuid4().hex[:8]}"
    if not run_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for ch in run_id):
        raise SystemExit(f"invalid --run-id: {run_id}")
    run_dir = GEN / "runs" / run_id
    stages: list[dict] = []
    blocked: list[str] = []
    started_at = datetime.now(CST).isoformat(timespec="seconds")
    child_env = os.environ.copy()
    child_env.update({"QUANT_RUN_ID": run_id, "QUANT_AS_OF": date, "QUANT_ROOT": str(ROOT)})

    manifest = {
        "schema": "after_close_orchestrator/v3", "run_id": run_id,
        "requested_date": date, "started_at": started_at, "generated_at": started_at,
        "external_delivery_requested": args.push, "status": "running", "exit_code": None,
        "current_stage": None, "stages": stages, "blocked_downstream": blocked,
        "artifacts": {"review_json": str(GEN / f"review_{date}.json"),
                      "review_markdown": str(GEN / f"review_{date}.md"),
                      "weekly_json": None if args.skip_weekly else str(GEN / f"market_weekly_{date}.json"),
                      "weekly_markdown": None if args.skip_weekly else str(GEN / f"market_weekly_{date}.md")},
    }
    _write_manifest(run_dir, manifest)
    _write_status(manifest)

    def checkpoint() -> None:
        manifest["generated_at"] = datetime.now(CST).isoformat(timespec="seconds")
        _write_manifest(run_dir, manifest)
        _write_status(manifest)

    def required(name: str, command: list[str], timeout: int) -> bool:
        manifest["current_stage"] = name
        checkpoint()
        result = _run_stage(name, command, timeout, env=child_env)
        stages.append(result)
        manifest["current_stage"] = None
        checkpoint()
        return result["status"] == "ok"

    review_cmd = [sys.executable, str(ROOT / "scripts" / "daily_review_chain.py"), "--date", date]
    if args.push:
        review_cmd.append("--push")
    if not required("统一决策复盘", review_cmd, 1800):
        blocked.extend(["统一决策快照", "详细HTML报告", "全市场周报"])
    elif not required("统一决策快照", [sys.executable, str(ROOT / "scripts" / "unified_decision_snapshot.py"), "--mode", "after_close", "--date", date], 600):
        blocked.extend(["详细HTML报告", "全市场周报"])
    elif not required("详细HTML报告", [sys.executable, str(ROOT / "scripts" / "html_report_generator.py"), date], 900):
        blocked.append("全市场周报")

    if not args.skip_weekly and not blocked:
        weekly_json = GEN / f"market_weekly_{date}.json"
        if not required("全市场周报", [sys.executable, str(ROOT / "scripts" / "market_weekly_report.py"), "--as-of", date, "--out", str(weekly_json)], 900):
            blocked.append("全市场周报")
    elif not args.skip_weekly and "全市场周报" not in blocked:
        blocked.append("全市场周报")

    failed = [s for s in stages if s["status"] != "ok"]
    status = "blocked" if blocked else ("failed" if failed else "ok")
    manifest.update({
        "status": status, "exit_code": 0 if status == "ok" else 2,
        "current_stage": None, "finished_at": datetime.now(CST).isoformat(timespec="seconds"),
    })
    checkpoint()
    path = run_dir / "manifest.json"
    print(json.dumps({"run_id": run_id, "status": status, "exit_code": manifest["exit_code"],
                      "manifest": str(path), "status_file": str(_status_dir() / "after_close.json")}, ensure_ascii=False))
    return manifest["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
