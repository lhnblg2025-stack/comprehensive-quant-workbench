#!/usr/bin/env python3
"""交易日盘后最小编排器：以 manifest 固化数据、报告与投递状态。

该入口默认只生成报告；生产调度应显式传入 ``--deliver``，避免“生成成功”
被误认为“已经送达”。每一步都写入同一个 run manifest，失败以非零退出。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "generated"
RUNS = GENERATED / "runs"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    output = ((p.stdout or "") + (p.stderr or "")).strip()
    return p.returncode, output[-4000:]


def main() -> int:
    ap = argparse.ArgumentParser(description="交易日盘后报告与投递编排")
    ap.add_argument("--date", default=None, help="分析交易日 YYYY-MM-DD/YYYYMMDD")
    ap.add_argument("--deliver", action="store_true", help="实际投递；默认仅生成")
    ap.add_argument("--channels", default="feishu", help="投递渠道，逗号分隔")
    ap.add_argument("--no-finalize", action="store_true", help="仅用于本地验证：传给日报生成器跳过投递收尾")
    args = ap.parse_args()
    date = (args.date or "").replace("-", "") or None
    run_id = f"{datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    manifest = {
        "run_id": run_id,
        "run_date": datetime.now().astimezone().date().isoformat(),
        "data_as_of": date,
        "execution_mode": "paper",
        "delivery_requested": args.deliver,
        "started_at": _now(),
        "steps": [],
    }
    path = RUNS / run_id / "manifest.json"
    _write(path, manifest)

    cmd = [sys.executable, str(ROOT / "scripts" / "a_share_daily_report.py")]
    if date:
        cmd += ["--date", date]
    if args.deliver:
        cmd += ["--deliver", "--channels", args.channels]
    if args.no_finalize:
        cmd += ["--no-finalize"]
    rc, output = _run(cmd)
    step = {"name": "daily_report", "status": "ok" if rc == 0 else "failed", "exit_code": rc,
            "delivery": "requested" if args.deliver else "skipped", "finished_at": _now(), "output_tail": output}
    manifest["steps"].append(step)
    manifest["status"] = "ok" if rc == 0 else "failed"
    manifest["finished_at"] = _now()
    _write(path, manifest)
    print(json.dumps({"manifest": str(path), "run_id": run_id, "status": manifest["status"],
                      "delivery": step["delivery"]}, ensure_ascii=False))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
