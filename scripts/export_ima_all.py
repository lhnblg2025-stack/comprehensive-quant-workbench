#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""递归导出 IMA 知识库全量媒体（PDF/txt/docx/图片）到本地。

按文件夹层级遍历整个知识库，下载所有可下载的媒体文件，元数据写入 meta.jsonl。
会议纪要/研报 PDF 直接下载；图片保留给 OCR。

用法:
  python3 scripts/export_ima_all.py --kb "<kb_id>" --out data_warehouse/ima_export/全量 --limit 0
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

# media_type → 扩展名
MEDIA_EXT = {
    1: "pdf", 2: "doc", 3: "docx", 4: "xls", 5: "xlsx", 6: "ppt",
    7: "pptx", 8: "md", 9: "png", 10: "jpg", 11: "note", 12: "url", 13: "txt",
}
# 需要下载的媒体类型（文本/PDF/文档/图片均可；跳过 12 url 和 99 folder）
DOWNLOADABLE = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13}


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


def list_folder(kb_id: str, folder_id: str | None, limit: int = 1000) -> list[dict]:
    out: list[dict] = []
    cursor = ""
    while True:
        body = {"knowledge_base_id": kb_id, "cursor": cursor, "limit": 20}
        if folder_id:
            body["folder_id"] = folder_id
        try:
            d = call("openapi/wiki/v1/get_knowledge_list", body)
        except Exception as e:
            print(f"  [list_err] folder={folder_id}: {e}", file=sys.stderr)
            break
        data = d.get("data", {})
        items = data.get("knowledge_list", [])
        out.extend(items)
        if data.get("is_end") or len(out) >= limit:
            break
        cursor = data.get("next_cursor", "")
        if not cursor:
            break
    return out


def get_media_info(media_id: str) -> dict:
    d = call("openapi/wiki/v1/get_media_info", {"media_id": media_id})
    return d.get("data", {})


def download(url: str, headers: dict, dest: Path, timeout: int = 60) -> bool:
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, headers=headers, timeout=timeout, stream=True) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"  [下载失败] {dest.name}: {e}", file=sys.stderr)
        return False


def safe_name(s: str) -> str:
    import re
    return re.sub(r'[\\/:*?"<>|]', "_", str(s))[:90] or "untitled"


def walk(kb_id: str, folder_id: str | None, out_dir: Path, meta_f,
         stats: dict, depth: int = 0) -> None:
    """递归遍历并下载。folder_id=None 表示根目录。"""
    items = list_folder(kb_id, folder_id, limit=2000)
    folders = [x for x in items if x.get("media_type") == 99 or str(x.get("media_id", "")).startswith("folder_")]
    medias = [x for x in items if x not in folders]

    for folder in folders:
        fid = folder.get("media_id")
        title = safe_name(folder.get("title", "folder"))
        sub = out_dir / title
        # 深度保护防止无限递归
        if depth < 8:
            walk(kb_id, fid, sub, meta_f, stats, depth + 1)

    for m in medias:
        mtype = m.get("media_type")
        mid = m.get("media_id")
        title = str(m.get("title", "media"))
        if mtype not in DOWNLOADABLE or not mid:
            stats["skipped"] += 1
            continue
        ext = MEDIA_EXT.get(int(mtype or 0), "bin")
        dest = out_dir / f"{safe_name(title)}.{ext}"
        if dest.exists() and dest.stat().st_size > 0:
            stats["already"] += 1
            continue
        try:
            info = get_media_info(mid)
        except Exception as e:
            print(f"  [info_err] {title}: {e}", file=sys.stderr)
            stats["err"] += 1
            continue
        url = (info.get("url_info") or {}).get("url")
        headers = (info.get("url_info") or {}).get("headers", {})
        if not url:
            stats["no_url"] += 1
            continue
        if download(url, headers, dest):
            stats["ok"] += 1
            meta_f.write(json.dumps({
                "title": title, "media_id": mid, "media_type": mtype,
                "ext": ext, "path": str(dest.relative_to(ROOT)),
                "folder": str(out_dir.relative_to(ROOT.parent)) if out_dir != ROOT else "",
            }, ensure_ascii=False) + "\n")
        else:
            stats["err"] += 1
        time.sleep(0.1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True)
    ap.add_argument("--out", default=str(ROOT / "data_warehouse" / "ima_export" / "全量"))
    ap.add_argument("--limit", type=int, default=0, help="0=不限")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta_f = (out / "meta.jsonl").open("a", encoding="utf-8")
    stats = {"ok": 0, "err": 0, "no_url": 0, "skipped": 0, "already": 0}
    print(f"递归导出知识库 → {out}")
    try:
        walk(args.kb, None, out, meta_f, stats)
    finally:
        meta_f.close()
    print(f"完成: 下载{stats['ok']} 已有{stats['already']} 无url{stats['no_url']} 跳过{stats['skipped']} 错误{stats['err']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
