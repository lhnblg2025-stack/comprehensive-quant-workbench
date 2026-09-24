#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一 IMA 研报链路：枚举 -> 元数据 -> 正文/媒体 -> NLP -> research_flow。

该入口不把主题搜索当作全量抓取。默认分页遍历知识库目录，保存原始元数据和失败台账；
媒体下载受限时可断点继续，不会清空已有结果。实际研报解析仍由 research_flow 负责。
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from export_ima import call, list_kb
except ModuleNotFoundError:
    # Support both ``python scripts/ima_research_pipeline.py`` and
    # ``import scripts.ima_research_pipeline`` invocation styles.
    from scripts.export_ima import call, list_kb


def _checked_call(api_path: str, body: dict) -> dict:
    response = call(api_path, body)
    if isinstance(response, dict) and response.get("code") not in (None, 0, "0"):
        raise RuntimeError(f"IMA API {api_path} 返回错误 {response.get('code')}: {response.get('msg') or response}")
    return response

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data_warehouse" / "ima_export" / "unified"
MANIFEST = "manifest.json"
ITEMS = "items.jsonl"
FAILURES = "failures.jsonl"
FOLDERS = "folders.jsonl"
STATE = "state.json"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
            if row.get("media_id"):
                out.add(str(row["media_id"]))
        except json.JSONDecodeError:
            continue
    return out


# 业务优先级：先做当天盘面和产业线索，再做券商研报，财联社脱水放最后。
PRIORITY_TOKENS = (
    "七、每日复盘",
    "十一、调研会议",
    "九、题材概念",
    "二、高盛",
    "八、大摩",
    "十、中金",
    "一：彭博",
    "三、红宝书",
    "五、财联社",
)
DATE_RE = re.compile(r"(20\d{2})[年./_-]?(\d{1,2})[月./_-]?(\d{1,2})")

def _date_from_text(text: str) -> str | None:
    match = DATE_RE.search(str(text or ""))
    if not match:
        return None
    return f"{match.group(1)}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"



def _priority(path: str) -> int:
    text = str(path or "")
    for index, token in enumerate(PRIORITY_TOKENS):
        if token in text:
            return index
    return len(PRIORITY_TOKENS)


def enumerate_kb(kb_id: str, out: Path, page_size: int = 50, sleep: float = 1.0, max_requests: int = 0, priority: bool = True) -> dict:
    """递归分页枚举所有目录，持久化队列避免重扫成功目录。"""
    out.mkdir(parents=True, exist_ok=True)
    items_path, folders_path, state_path = out / ITEMS, out / FOLDERS, out / STATE
    seen = _load_ids(items_path)
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"queue": [{"folder_id": None, "folder_path": "", "cursor": ""}], "done": []}
    queue = list(state.get("queue") or [])
    if priority:
        queue.sort(key=lambda task: (_priority(task.get("folder_path")), str(task.get("folder_path") or "")))
    done = set(state.get("done") or [])
    stats = {"folders": 0, "media": 0, "pages": 0, "duplicates": 0, "errors": 0, "requests": 0}
    while queue and (not max_requests or stats["requests"] < max_requests):
        task = queue[0]
        folder_id, folder_path, cursor = task.get("folder_id"), task.get("folder_path", ""), task.get("cursor", "")
        key = folder_id or "__root__"
        if key in done:
            queue.pop(0)
            continue
        body = {"knowledge_base_id": kb_id, "cursor": cursor, "limit": page_size}
        if folder_id:
            body["folder_id"] = folder_id
        try:
            response = _checked_call("openapi/wiki/v1/get_knowledge_list", body)
            stats["requests"] += 1
            data = response.get("data") or {}
            page = data.get("knowledge_list") or []
            stats["pages"] += 1
        except Exception as exc:  # noqa: BLE001
            stats["requests"] += 1
            stats["errors"] += 1
            error_text = str(exc)[:1000]
            _append_jsonl(out / FAILURES, {"stage": "enumerate", "folder_id": folder_id, "cursor": cursor, "error": error_text, "action": "resume from state.json after cooldown"})
            state["queue"] = queue
            state["done"] = sorted(done)
            _write_json(state_path, state)
            if "200001" in error_text or "频率超限" in error_text:
                time.sleep(max(3.0, sleep * 3) + random.random())
            queue.append(queue.pop(0))
            if stats["errors"] >= 3:
                break
            continue
        child_tasks = []
        for raw in page:
            mid = str(raw.get("media_id") or "")
            if not mid:
                continue
            kind = "folder" if raw.get("media_type") == 99 or mid.startswith("folder_") else "media"
            full_path = f"{folder_path}/{raw.get('title') or mid}"
            record = {**raw, "media_id": mid, "kind": kind, "knowledge_base_id": kb_id,
                      "folder_path": folder_path, "full_path": full_path,
                      "priority_bucket": _priority(folder_path),
                      "date_hint": _date_from_text(full_path),
                      "enumerated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            if mid in seen:
                stats["duplicates"] += 1
            else:
                _append_jsonl(items_path, record)
                seen.add(mid)
                stats["folders" if kind == "folder" else "media"] += 1
            if kind == "folder" and mid not in done:
                child_tasks.append({"folder_id": mid, "folder_path": f"{folder_path}/{raw.get('title') or mid}", "cursor": ""})
        next_cursor = str(data.get("next_cursor") or "")
        if next_cursor and not data.get("is_end"):
            task["cursor"] = next_cursor
        else:
            done.add(key)
            queue.pop(0)
            queue.extend(child_tasks)
            if priority:
                queue.sort(key=lambda pending: (_priority(pending.get("folder_path")), str(pending.get("folder_path") or "")))
        state["queue"], state["done"] = queue, sorted(done)
        _write_json(state_path, state)
        time.sleep(max(0.0, sleep))
    return stats


def build_manifest(out: Path, kb_id: str, stats: dict, expected_count: int | None = None) -> dict:
    rows = []
    if (out / ITEMS).exists():
        rows = [json.loads(line) for line in (out / ITEMS).read_text(encoding="utf-8").splitlines() if line.strip()]
    total_media = sum(1 for x in rows if x.get("kind") == "media")
    errors = int(stats.get("errors") or 0)
    state = json.loads((out / STATE).read_text(encoding="utf-8")) if (out / STATE).exists() else {}
    pending = len(state.get("queue") or [])
    manifest = {
        "schema": "ima-research-pipeline-v1",
        "knowledge_base_id": kb_id,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "enumeration": stats,
        "total_items": len(rows),
        "total_media": total_media,
        "expected_count": expected_count,
        "pending_folders": pending,
        "status": "partial" if errors or pending or (expected_count and len(rows) < expected_count) else "enumerated",
        "complete": not errors and not pending and (not expected_count or len(rows) >= expected_count),
        "next": "按 state.json 继续枚举剩余目录" if errors or pending else "download media/body, then run research_flow",
    }
    _write_json(out / MANIFEST, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="统一 IMA 研报全量枚举入口")
    parser.add_argument("--list-kb", action="store_true")
    parser.add_argument("--kb", default="")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--max-requests", type=int, default=0, help="本次最多请求数，0=不限")
    parser.add_argument("--expected-count", type=int, default=39808, help="知识库详情显示的参考总数；仅用于完整性对账")
    parser.add_argument("--no-priority", action="store_true", help="按发现顺序处理，不启用重要目录优先")
    parser.add_argument("--today", default=datetime.now().strftime("%Y-%m-%d"), help="当天日期，用于标记 date_hint")
    args = parser.parse_args()
    if args.list_kb:
        result = _checked_call("openapi/wiki/v1/search_knowledge_base", {"query": "", "cursor": "", "limit": 20})
        print(json.dumps(result.get("data", {}).get("info_list", []), ensure_ascii=False, indent=2))
        return 0
    if not args.kb:
        parser.error("需要 --kb，或使用 --list-kb")
    out = Path(args.out)
    try:
        stats = enumerate_kb(args.kb, out, max(1, min(args.page_size, 50)), max(0, args.sleep), max(0, args.max_requests), not args.no_priority)
        manifest = build_manifest(out, args.kb, stats, args.expected_count)
        manifest["today"] = args.today
        manifest["priority_order"] = list(PRIORITY_TOKENS)
        _write_json(Path(args.out) / MANIFEST, manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001
        failure = {"schema": "ima-research-pipeline-v1", "knowledge_base_id": args.kb, "status": "blocked", "blocked_stage": "api_auth_or_skill_version", "error": str(exc)[:1000], "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        _write_json(Path(args.out) / MANIFEST, failure)
        _append_jsonl(Path(args.out) / FAILURES, failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
