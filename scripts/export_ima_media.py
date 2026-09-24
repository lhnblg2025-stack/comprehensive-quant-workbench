#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按搜索词导出 IMA 媒体原文到本地（供研报/会议纪要 OCR/NLP 使用）。

用法:
  python3 scripts/export_ima_media.py --kb "-ppx..." --query "会议纪要" --out data_warehouse/ima_export/media --limit 30
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
IMA_API = ROOT / ".archive" / "skills-20260820" / "ima-skills" / "ima_api.cjs"
CLIENT_FILE = Path.home() / ".config" / "ima" / "client_id"
KEY_FILE = Path.home() / ".config" / "ima" / "api_key"


def _opts() -> str:
    return json.dumps({
        "clientId": CLIENT_FILE.read_text().strip(),
        "apiKey": KEY_FILE.read_text().strip(),
    })


def call(api_path: str, body: dict) -> dict:
    r = subprocess.run(
        ["node", str(IMA_API), api_path, json.dumps(body, ensure_ascii=False), _opts()],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr[:300])
    return json.loads(r.stdout)


def search_kb(kb_id: str, query: str) -> list[dict]:
    d = call("openapi/wiki/v1/search_knowledge", {"query": query, "knowledge_base_id": kb_id, "cursor": ""})
    return d.get("data", {}).get("info_list", [])


def get_media_info(media_id: str) -> dict:
    d = call("openapi/wiki/v1/get_media_info", {"media_id": media_id})
    return d.get("data", {})


def list_folder(kb_id: str, folder_id: str, limit: int = 500) -> list[dict]:
    """分页列出文件夹下的所有条目（含子文件夹与媒体）。"""
    out: list[dict] = []
    cursor = ""
    while True:
        body = {"knowledge_base_id": kb_id, "cursor": cursor, "limit": 20}
        if folder_id:
            body["folder_id"] = folder_id
        d = call("openapi/wiki/v1/get_knowledge_list", body)
        data = d.get("data", {})
        out.extend(data.get("knowledge_list", []))
        if data.get("is_end") or len(out) >= limit:
            break
        cursor = data.get("next_cursor", "")
        if not cursor:
            break
    return out


def download(url: str, headers: dict, dest: Path, timeout: int = 60) -> bool:
    try:
        with requests.get(url, headers=headers, timeout=timeout, stream=True) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"  [下载失败] {dest.name}: {e}", file=sys.stderr)
        return False


def ext_for_media_type(t: int) -> str:
    return {
        1: "pdf", 2: "doc", 3: "docx", 4: "xls", 5: "xlsx", 6: "ppt",
        7: "pptx", 8: "md", 9: "png", 10: "jpg", 11: "note", 12: "url", 13: "txt",
    }.get(int(t or 0), "bin")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True)
    ap.add_argument("--folder", default="", help="指定文件夹 media_id(如 folder_xxx)，遍历该目录下载媒体")
    ap.add_argument("--query", default="会议纪要")
    ap.add_argument("--out", default=str(ROOT / "data_warehouse" / "ima_export" / "media"))
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.jsonl"

    if args.folder:
        hits = list_folder(args.kb, args.folder, limit=args.limit)
        print(f"文件夹 {args.folder} 条目 {len(hits)} 条，下载前 {args.limit} 条")
    else:
        hits = search_kb(args.kb, args.query)
        print(f"搜索 '{args.query}' 命中 {len(hits)} 条，下载前 {args.limit} 条")
    ok = 0
    for i, h in enumerate(hits[: args.limit]):
        mid = h.get("media_id", "")
        title = str(h.get("title", f"ima_{i}"))
        if not mid:
            continue
        safe = "".join(c for c in title if c not in '\\/:*?"<>|')[:80] or f"ima_{i}"
        try:
            info = get_media_info(mid)
        except Exception as e:
            print(f"  [err] {title}: {e}", file=sys.stderr)
            continue
        url = (info.get("url_info") or {}).get("url")
        headers = (info.get("url_info") or {}).get("headers", {})
        if not url:
            print(f"  [无url] {title}")
            continue
        ext = ext_for_media_type(info.get("media_type"))
        dest = out / f"{safe}.{ext}"
        if download(url, headers, dest):
            ok += 1
            with meta_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "title": title, "media_id": mid, "media_type": info.get("media_type"),
                    "path": str(dest), "ext": ext, "query": args.query,
                }, ensure_ascii=False) + "\n")
        time.sleep(0.2)
    print(f"完成：成功下载 {ok}/{min(len(hits), args.limit)} → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
