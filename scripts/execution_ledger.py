#!/usr/bin/env python3
"""execution_ledger.py — 投递回执与技能执行登记的统一账本。

统一决策工作台生产闭环要求每一笔投递、每一次技能执行都有可追溯登记：
  - 投递回执(delivery)  → generated/delivery_receipts/{date}_{kind}_*.json 单笔回执
                          + generated/delivery_receipts/index.json 统一索引
  - 技能执行(skill_run) → generated/skill_runs/{execution_id}.json 单次载荷
                          + generated/skill_runs/index.json 统一索引

两个索引都是追加式账本：record_* 幂等（同 id 覆盖），原子写（tmp+rename），
不做全量重读。这样工作台 /api/workbench 或运维页可以不扫目录而直接读索引。

用法（其它脚本 import）:
    from execution_ledger import record_delivery, record_skill_run, ledger_summary
"""
from __future__ import annotations

import json
import fcntl
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "generated"
DELIVERY_DIR = GEN / "delivery_receipts"
SKILL_RUN_DIR = GEN / "skill_runs"

CST = timezone(timedelta(hours=8))

DELIVERY_INDEX = DELIVERY_DIR / "index.json"
SKILL_RUN_INDEX = SKILL_RUN_DIR / "index.json"
LEDGER_LOCK = GEN / ".execution_ledger.lock"


@contextmanager
def _ledger_lock():
    LEDGER_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_LOCK.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _now() -> str:
    return datetime.now(CST).isoformat(timespec="seconds")


def _load_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema": "execution_ledger/v1", "entries": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"schema": "execution_ledger/v1", "entries": {}}
        data.setdefault("schema", "execution_ledger/v1")
        data.setdefault("entries", {})
        return data
    except (OSError, ValueError):
        return {"schema": "execution_ledger/v1", "entries": {}}


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _save_index(path: Path, index: dict[str, Any]) -> None:
    _atomic_write(path, index)


def record_delivery(*, delivery_id: str, record: dict[str, Any]) -> Path:
    """登记一笔投递回执到统一索引（追加/覆盖同 delivery_id）。"""
    with _ledger_lock():
        DELIVERY_DIR.mkdir(parents=True, exist_ok=True)
        index = _load_index(DELIVERY_INDEX)
        entry = dict(record)
        entry.setdefault("delivery_id", delivery_id)
        entry.setdefault("recorded_at", _now())
        index["entries"][delivery_id] = entry
        index["updated_at"] = _now()
        index["count"] = len(index["entries"])
        _save_index(DELIVERY_INDEX, index)
        return DELIVERY_INDEX


def record_skill_run(*, execution_id: str, record: dict[str, Any]) -> Path:
    """登记一次技能执行到统一索引（追加/覆盖同 execution_id）。"""
    with _ledger_lock():
        SKILL_RUN_DIR.mkdir(parents=True, exist_ok=True)
        index = _load_index(SKILL_RUN_INDEX)
        entry = dict(record)
        entry.setdefault("execution_id", execution_id)
        entry.setdefault("recorded_at", _now())
        index["entries"][execution_id] = entry
        index["updated_at"] = _now()
        index["count"] = len(index["entries"])
        _save_index(SKILL_RUN_INDEX, index)
        return SKILL_RUN_INDEX


def ledger_summary() -> dict[str, Any]:
    """工作台/运维页读取的统一账本摘要。"""
    delivery = _load_index(DELIVERY_INDEX)
    skb = _load_index(SKILL_RUN_INDEX)
    latest_delivery = None
    if delivery.get("entries"):
        # 按 recorded_at 倒序取最近一笔有意义的回执
        ordered = sorted(delivery["entries"].items(),
                         key=lambda kv: kv[1].get("recorded_at") or "", reverse=True)
        latest_delivery = ordered[0][1] if ordered else None
    return {
        "schema": "execution_ledger/v1",
        "delivery": {
            "count": delivery.get("count", len(delivery.get("entries", {}))),
            "index": str(DELIVERY_INDEX),
            "latest": latest_delivery,
        },
        "skill_runs": {
            "count": skb.get("count", len(skb.get("entries", {}))),
            "index": str(SKILL_RUN_INDEX),
        },
        "generated_at": _now(),
    }


if __name__ == "__main__":
    print(json.dumps(ledger_summary(), ensure_ascii=False, indent=2))
