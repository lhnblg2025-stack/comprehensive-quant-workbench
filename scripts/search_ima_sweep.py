#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 search_knowledge（知识库内搜索，不受浏览/下载 220021 配额限制）枚举全库内容。

用法:
  python3 scripts/search_ima_sweep.py --kb "-ppxYIh-...=" --queries "M3期货:期货,早报" "M5复盘:复盘,涨停" \
      [--per-query 200] [--sleep 0.15] [--out data_warehouse/ima_export/search_index]

输出:
  <out>/sweep.jsonl        — 每条命中: query/media_id/title/parent_folder_id/media_type/highlight_content
  <out>/sweep.tsv          — 去重汇总表 (media_id, title, folder, query 来源)
  <out>/progress.json      — 断点进度（query -> next_cursor / done）
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

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
    d = json.loads(r.stdout)
    if isinstance(d, dict) and d.get("code") not in (0, None):
        raise RuntimeError(f"{api_path}: {d.get('msg', d)}")
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True)
    ap.add_argument("--queries", nargs="*", default=[],
                    help="形如 M3期货:期货,早报 的 模块:关键词列表")
    ap.add_argument("--per-query", type=int, default=200, help="每查询最多翻页条数")
    ap.add_argument("--sleep", type=float, default=0.15)
    ap.add_argument("--out", default=str(ROOT / "data_warehouse" / "ima_export" / "search_index"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jl = out / "sweep.jsonl"
    tsv = out / "sweep.tsv"
    prog_f = out / "progress.json"

    queries = []
    for q in args.queries:
        if ":" in q:
            mod, kws = q.split(":", 1)
            queries.append((mod, [k for k in kws.split(",") if k.strip()]))
        else:
            queries.append((q, [q]))
    if not queries:
        print("没有查询，示例: --queries 'M3期货:期货,早报'", file=sys.stderr)
        return 2

    progress = {}
    if prog_f.exists():
        progress = json.loads(prog_f.read_text(encoding="utf-8"))

    seen = set()
    if tsv.exists():
        # tsv 无表头，不跳首行（避免漏掉第一个 media_id 造成重复）
        for line in tsv.read_text(encoding="utf-8").splitlines():
            cols = line.split("\t")
            if len(cols) >= 2:
                seen.add(cols[0])

    total_new = 0
    for mod, kws in queries:
        for kw in kws:
            key = f"{mod}:{kw}"
            state = progress.get(key, {})
            cursor = state.get("next_cursor", "")
            if state.get("done"):
                print(f"[skip] {key} 已完成"); continue
            got = 0
            while got < args.per_query:
                body = {"query": kw, "cursor": cursor, "knowledge_base_id": args.kb}
                try:
                    d = call("openapi/wiki/v1/search_knowledge", body)
                except RuntimeError as e:
                    if "220021" in str(e) or "RATELIMIT" in str(e):
                        print(f"[STOP] {key} 命中限流: {e}")
                        progress[key] = {"next_cursor": cursor}
                        prog_f.write_text(json.dumps(progress, ensure_ascii=False, indent=1), encoding="utf-8")
                        print(f"\n=== 汇总: 新增 {total_new} 条 ===")
                        return 3
                    print(f"[err] {key}: {e}", file=sys.stderr)
                    time.sleep(2)
                    continue
                data = d.get("data", {})
                items = data.get("info_list", [])
                if not items:
                    break
                with jl.open("a", encoding="utf-8") as f:
                    for it in items:
                        mid = it.get("media_id", "")
                        if not mid or mid in seen:
                            continue
                        seen.add(mid)
                        rec = {"query": key, "media_id": mid, "title": it.get("title", ""),
                               "parent_folder_id": it.get("parent_folder_id", ""),
                               "media_type": it.get("media_type", ""),
                               "highlight_content": it.get("highlight_content", "")}
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        tsv_f = tsv.open("a", encoding="utf-8")
                        tsv_f.write(f"{mid}\t{it.get('title','')}\t{it.get('parent_folder_id','')}\t{key}\n")
                        tsv_f.close()
                        total_new += 1
                got += len(items)
                time.sleep(args.sleep)
                if data.get("is_end"):
                    break
                cursor = data.get("next_cursor", "")
                if not cursor:
                    break
                progress[key] = {"next_cursor": cursor}
            progress[key] = {"done": True}
            prog_f.write_text(json.dumps(progress, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"[ok] {key}: {got} 条扫描，新增 {total_new} 累计")
    print(f"\n=== 完成: 新增 {total_new} 条命中 ===")
    print(f"输出: {jl} / {tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
