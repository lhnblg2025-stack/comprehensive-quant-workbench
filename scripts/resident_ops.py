#!/usr/bin/env python3
"""Resident-node operations: retry policy, run status, catch-up, and Feishu digest."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("QUANT_ROOT", str(Path(__file__).resolve().parents[1]))).resolve()
RUN_DIR = ROOT / "generated" / "runtime_status"
CST = timezone(timedelta(hours=8))


def _now() -> datetime:
    return datetime.now(CST)


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def status_path(day: str) -> Path:
    return RUN_DIR / f"after_close_{day}.json"


def run_pipeline(day: str, *, retries: int = 2, timeout: int = 10_800) -> dict[str, Any]:
    """Execute the existing after-close pipeline with bounded retries and receipts."""
    day = str(day)
    previous = _read(status_path(day))
    if previous.get("status") == "succeeded":
        return previous
    attempts = list(previous.get("attempts") or [])
    configured = os.environ.get("QUANT_RESIDENT_COMMAND", "").strip()
    if configured:
        command = ["cmd", "/c", configured] if os.name == "nt" else ["bash", "-lc", configured]
    else:
        if os.name == "nt":
            command = ["cmd.exe", "/c", str(ROOT / "run_after_close_unified.bat")]
        else:
            command = ["bash", str(ROOT / "scripts" / "run_after_close_pipeline.sh")]
    for attempt in range(len(attempts) + 1, retries + 2):
        started = _now()
        proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=timeout,
                              env={**os.environ, "QUANT_BUSINESS_DAY": day})
        record = {
            "attempt": attempt, "started_at": started.isoformat(timespec="seconds"),
            "finished_at": _now().isoformat(timespec="seconds"), "returncode": proc.returncode,
            "output_tail": ((proc.stdout or "") + (proc.stderr or ""))[-4000:],
        }
        attempts.append(record)
        payload = {"schema": "quant-resident-run/v1", "day": day,
                   "status": "succeeded" if proc.returncode == 0 else "retrying",
                   "attempts": attempts, "updated_at": _now().isoformat(timespec="seconds")}
        if proc.returncode == 0:
            payload["status"] = "succeeded"; _write(status_path(day), payload); return payload
        _write(status_path(day), payload)
    payload["status"] = "failed"; _write(status_path(day), payload)
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from alert_dedup import get_deduper, make_sig
        from feishu_sender import send_markdown
        signature = make_sig("resident_after_close", f"{day}:{attempts[-1].get('returncode')}")
        if get_deduper().should_send("resident_after_close", signature, level="danger"):
            send_markdown(f"# 盘后链失败\n\n- 数据日：{day}\n- 已重试：{len(attempts)} 次\n- 最后阶段输出：`{attempts[-1].get('output_tail','')[-800:]}`", title="量化常驻节点告警")
    except Exception:
        pass
    return payload


def latest_release() -> dict[str, Any]:
    return _read(ROOT / "generated" / "data_releases" / "latest.json")


def latest_health() -> dict[str, Any]:
    return _read(ROOT / "generated" / "health_guardian" / "alerts_latest.json")


def paper_summary() -> dict[str, Any]:
    try:
        from quant_system.trade_db import paper_execution_stats
        return paper_execution_stats()
    except Exception as exc:
        return {"error": str(exc)[:160]}


def dashboard_snapshot() -> dict[str, Any]:
    runs = sorted(RUN_DIR.glob("after_close_*.json")) if RUN_DIR.exists() else []
    last_run = _read(runs[-1]) if runs else {}
    return {
        "schema": "quant-resident-dashboard/v1", "generated_at": _now().isoformat(timespec="seconds"),
        "release": latest_release(), "health": latest_health(), "last_after_close": last_run,
        "paper_execution": paper_summary(),
    }


def digest() -> str:
    state = dashboard_snapshot(); release = state.get("release") or {}; health = state.get("health") or {}
    run = state.get("last_after_close") or {}; paper = state.get("paper_execution") or {}
    warnings = release.get("warnings") or []
    lines = [f"# 量化常驻节点日报 · {_now().strftime('%Y-%m-%d %H:%M')}", "",
             f"- 数据发布：**{release.get('release_id', '未生成')}** | {release.get('expected_day', '-')} | {'正常' if release else '缺失'}",
             f"- 发布警告：{'、'.join(warnings) if warnings else '无'}",
             f"- 盘后链：**{run.get('status', '未运行')}** | 尝试 {len(run.get('attempts') or [])} 次",
             f"- 健康：{'正常' if health.get('ok') else '异常'} | 告警 {health.get('alert_count', '-')}",
             f"- 纸面执行：订单 {paper.get('orders', 0)} | 成交率 {float(paper.get('fill_rate') or 0):.1%} | 不利滑点 {paper.get('mean_adverse_slippage_bps', '-')}"]
    if health.get("alerts"):
        lines.extend(["", "## 需要关注", *[f"- {item}" for item in health["alerts"][:6]]])
    lines.extend(["", "该消息为运行与决策入口摘要；异常已单独去重告警，纸面成交请通过控制台或CSV回填。"])
    return "\n".join(lines)


def send_digest() -> bool:
    from feishu_sender import send_markdown
    return bool(send_markdown(digest(), title="量化常驻节点日报"))


def catch_up() -> dict[str, Any]:
    """Run the latest completed market session once after host restart/outage."""
    from quant_system.market_clock import latest_completed_trading_day
    return run_pipeline(latest_completed_trading_day().isoformat())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run"); run.add_argument("--day", required=True); run.add_argument("--retries", type=int, default=2)
    sub.add_parser("catch-up"); sub.add_parser("snapshot"); sub.add_parser("digest")
    args = parser.parse_args()
    if args.command == "run": out = run_pipeline(args.day, retries=args.retries)
    elif args.command == "catch-up": out = catch_up()
    elif args.command == "snapshot": out = dashboard_snapshot()
    else:
        ok = send_digest(); out = {"sent": ok}
    print(json.dumps(out, ensure_ascii=False, indent=2) if isinstance(out, dict) else out)
    return 0 if not isinstance(out, dict) or out.get("status") != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
