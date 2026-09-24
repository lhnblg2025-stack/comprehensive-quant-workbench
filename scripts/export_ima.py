#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导出 IMA 知识库：目录树 + 媒体元数据 + 笔记文本 + 会议纪要定位。

用法:
  python3 scripts/export_ima.py --list-kb
  python3 scripts/export_ima.py --kb "<kb_id>" --out data_warehouse/ima_export/研报库 --max-items 200
  python3 scripts/export_ima.py --kb "<kb_id>" --search "会议纪要"
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMA_SKILL_DIR = ROOT / ".archive" / "skills-20260820" / "ima-skills"
IMA_API = IMA_SKILL_DIR / "ima_api.cjs"
CLIENT_FILE = Path.home() / ".config" / "ima" / "client_id"
KEY_FILE = Path.home() / ".config" / "ima" / "api_key"


def _opts() -> str:
    cid = CLIENT_FILE.read_text().strip()
    key = KEY_FILE.read_text().strip()
    return json.dumps({"clientId": cid, "apiKey": key})


def call(api_path: str, body: dict) -> dict:
    raw = json.dumps(body, ensure_ascii=False)
    env = dict(os.environ)
    r = subprocess.run(
        ["node", str(IMA_API), api_path, raw, _opts()],
        capture_output=True, text=True, timeout=60, env=env,
    )
    if r.returncode != 0:
        raise RuntimeError(f"IMA API {api_path} 失败: {r.stderr[:300]}")
    return json.loads(r.stdout)


def list_kb() -> list[dict]:
    d = call("openapi/wiki/v1/search_knowledge_base", {"query": "", "cursor": "", "limit": 20})
    return d.get("data", {}).get("info_list", [])


def walk_kb(kb_id: str, folder_id: str | None, out_dir: Path,
            max_items: int, depth: int = 0) -> int:
    """递归导出知识库目录与媒体清单，返回处理条目数。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    body: dict = {"knowledge_base_id": kb_id, "cursor": "", "limit": 20}
    if folder_id:
        body["folder_id"] = folder_id
    count = 0
    cursor = ""
    while True:
        body["cursor"] = cursor
        try:
            d = call("openapi/wiki/v1/get_knowledge_list", body)
        except Exception as e:
            print(f"  [err] {e}", file=sys.stderr)
            break
        data = d.get("data", {})
        items = data.get("knowledge_list", [])
        for item in items:
            count += 1
            if max_items and count > max_items:
                return count
            title = item.get("title", "")
            media_id = item.get("media_id", "")
            mtype = item.get("media_type")
            rec = {"title": title, "media_id": media_id, "media_type": mtype,
                   "parent": folder_id or "", "tags": item.get("tags", [])}
            # 文件夹递归
            if mtype == 99 or str(media_id).startswith("folder_"):
                rec["kind"] = "folder"
                (out_dir / "_folders.jsonl").open("a", encoding="utf-8").write(
                    json.dumps(rec, ensure_ascii=False) + "\n")
                sub = out_dir / _safe_name(title)
                count = walk_kb(kb_id, media_id, sub, max_items - count if max_items else 0, depth + 1)
            else:
                rec["kind"] = "media"
                (out_dir / "_items.jsonl").open("a", encoding="utf-8").write(
                    json.dumps(rec, ensure_ascii=False) + "\n")
        if data.get("is_end"):
            break
        cursor = data.get("next_cursor", "")
        if not cursor:
            break
    return count


def search_kb(kb_id: str, query: str, limit: int = 20) -> list[dict]:
    d = call("openapi/wiki/v1/search_knowledge", {
        "query": query, "knowledge_base_id": kb_id, "cursor": ""})
    return d.get("data", {}).get("info_list", [])


def get_media_text(kb_id: str, media_id: str) -> dict:
    """获取媒体原文；笔记类型返回文本。"""
    d = call("openapi/wiki/v1/get_media_info", {"media_id": media_id})
    return d


def _safe_name(s: str) -> str:
    import re
    return re.sub(r'[\\/:*?"<>|]', "_", str(s))[:80]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-kb", action="store_true")
    ap.add_argument("--kb", default="")
    ap.add_argument("--out", default=str(ROOT / "data_warehouse" / "ima_export"))
    ap.add_argument("--max-items", type=int, default=200)
    ap.add_argument("--search", default="")
    ap.add_argument("--search-kb", default="")
    args = ap.parse_args()

    if args.list_kb:
        for kb in list_kb():
            print(f"{kb['kb_id']}\t{kb['kb_name']}\t内容{int(kb.get('content_count',0))}")
        return 0

    if not args.kb:
        print("需要 --kb <id> 或 --list-kb", file=sys.stderr)
        return 1

    if args.search:
        hits = search_kb(args.search_kb or args.kb, args.search)
        print(json.dumps(hits, ensure_ascii=False, indent=2))
        return 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # 清空旧的 jsonl
    for f in out.glob("*.jsonl"):
        f.unlink()
    print(f"导出知识库 {args.kb} → {out} (max={args.max_items})")
    n = walk_kb(args.kb, None, out, args.max_items)
    print(f"完成，处理 {n} 条（含递归）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
