#!/usr/bin/env python3
"""platform.runtime — 运行闭环：任务状态机 / 审计 / 健康（骨架收口 P3）

统一任务状态记录：所有常驻任务（cron 触发）通过本模块登记状态，
状态写入 logs/task_state.json，供前端 /api/v1/health 和人工检查。

用法:
  from quant_platform.runtime import task_start, task_end, task_status
  task_start("kline_update")
  ...
  task_end("kline_update", ok=True, detail="5206只 08-07")
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = ROOT / "logs" / "task_state.json"
CST = timezone(timedelta(hours=8))

MAX_TASKS = 100  # 保留最近 N 条记录


def _load() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {"tasks": []}
    return {"tasks": []}


def _save(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _now() -> str:
    return datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S")


def task_start(name: str) -> None:
    """任务开始：登记 running 状态。"""
    state = _load()
    tasks = state["tasks"]
    tasks = [t for t in tasks if t.get("name") != name]  # 同名替换
    tasks.append({"name": name, "status": "running", "start": _now(), "end": None, "ok": None, "detail": ""})
    state["tasks"] = tasks[-MAX_TASKS:]
    _save(state)


def task_end(name: str, ok: bool, detail: str = "") -> None:
    """任务结束：更新状态为 done/failed。"""
    state = _load()
    for t in state["tasks"]:
        if t.get("name") == name and t.get("status") == "running":
            t["status"] = "done" if ok else "failed"
            t["end"] = _now()
            t["ok"] = bool(ok)
            t["detail"] = detail
            break
    _save(state)


def task_status(name: str | None = None) -> dict:
    """查询任务状态。name=None 返回全部。"""
    state = _load()
    tasks = state["tasks"]
    if name:
        tasks = [t for t in tasks if t.get("name") == name]
    return {"tasks": tasks, "count": len(tasks), "updated": _now()}


def running_tasks() -> list[str]:
    """当前 running 状态的任务名。"""
    return [t["name"] for t in _load()["tasks"] if t.get("status") == "running"]
