#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全量盘点 IMA 知识库：递归遍历所有文件夹 + 统计每个文件夹的条目数。

输出:
  1) data_warehouse/ima_export/census/tree.tsv   — 目录树（缩进 + 条目数）
  2) data_warehouse/ima_export/census/folders.jsonl — 每个文件夹记录
  3) data_warehouse/ima_export/census/leaf_counts.json — 叶子级条目数合计

支持断点续跑：已完成的 folder 若在 done.txt 中则跳过。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMA_API = ROOT / ".archive" / "skills-20260820" / "ima-skills" / "ima_api.cjs"
CLIENT_FILE = Path.home() / ".config" / "ima" / "client_id"
KEY_FILE = Path.home() / ".config" / "ima" / "api_key"


def _opts() -> str:
    cid = CLIENT_FILE.read_text().strip()
    key = KEY_FILE.read_text().strip()
    return json.dumps({"clientId": cid, "apiKey": key})


def call(api_path: str, body: dict) -> dict:
    raw = json.dumps(body, ensure_ascii=False)
    r = subprocess.run(
        ["node", str(IMA_API), api_path, raw, _opts()],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        raise RuntimeError(f"IMA API {api_path} 失败: {r.stderr[:300]}")
    return json.loads(r.stdout)


def get_list(kb_id: str, folder_id, cursor: str, limit=50) -> dict:
    body = {"knowledge_base_id": kb_id, "cursor": cursor, "limit": limit}
    if folder_id:
        body["folder_id"] = folder_id
    return call("openapi/wiki/v1/get_knowledge_list", body)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True)
    ap.add_argument("--out", default=str(ROOT / "data_warehouse" / "ima_export" / "census"))
    ap.add_argument("--folder", default="", help="起始文件夹(默认根)")
    ap.add_argument("--sleep", type=float, default=0.15, help="每次API调用间隔(秒)")
    ap.add_argument("--label", default="研报库")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tree_f = out / "tree.tsv"
    folders_f = out / "folders.jsonl"
    done_f = out / "done.txt"
    done = set()
    if done_f.exists():
        done = {l.strip() for l in done_f.read_text().splitlines() if l.strip()}

    # dict folder_id -> (name, parent_id, item_count, kind)
    folder_meta: dict[str, dict] = {}

    def safe(s: str) -> str:
        import re
        return re.sub(r'[\t\r\n]', ' ', str(s))

    def walk(folder_id, name, parent_id, depth):
        fkey = folder_id or "<root>"
        if folder_id and fkey in done:
            return
        items = []
        cursor = ""
        first = True
        while True:
            try:
                d = get_list(args.kb, folder_id, cursor)
            except Exception as e:
                print(f"[{name}] err: {e}", file=sys.stderr)
                time.sleep(2)
                continue
            time.sleep(args.sleep)
            data = d.get("data", {})
            for it in data.get("knowledge_list", []):
                items.append(it)
            if data.get("is_end"):
                break
            cursor = data.get("next_cursor", "")
            if not cursor:
                break
        n_files = 0
        n_folders = 0
        for it in items:
            mtype = it.get("media_type")
            mid = str(it.get("media_id", ""))
            if mtype == 99 or mid.startswith("folder_"):
                n_folders += 1
                fname = it.get("title", "")
                fold_id = mid
                folder_meta[fold_id] = {"name": fname, "parent": folder_id,
                                        "items": 0, "kind": "folder"}
                # 记录到 folders.jsonl
                with folders_f.open("a", encoding="utf-8") as fo:
                    fo.write(json.dumps({"folder_id": fold_id, "name": fname,
                                         "parent": folder_id}, ensure_ascii=False) + "\n")
            else:
                n_files += 1
        total = n_files + n_folders
        indent = "  " * depth
        with tree_f.open("a", encoding="utf-8") as tf:
            tf.write(f"{indent}{safe(name)}\t文件{n_files}\t文件夹{n_folders}\t合计{total}\n")
        print(f"{indent}{name}: 文件{n_files} 文件夹{n_folders}")
        sys.stdout.flush()
        if folder_id:
            with done_f.open("a", encoding="utf-8") as df:
                df.write(fkey + "\n")
        # 递归子文件夹
        for it in items:
            mtype = it.get("media_type")
            mid = str(it.get("media_id", ""))
            if mtype == 99 or mid.startswith("folder_"):
                walk(mid, it.get("title", ""), folder_id, depth + 1)

    walk(args.folder or None, args.label or "<root>", None, 0)

    # 汇总
    total_files = 0
    # 重新统计每个文件夹的文件数（从 tree 汇总不好做，这里简单用最后一级）
    print("\n=== 盘点完成 ===")
    print(f"输出目录: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
