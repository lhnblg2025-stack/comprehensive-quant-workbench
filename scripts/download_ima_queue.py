#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按优先级队列下载 IMA 媒体原文（search_knowledge 枚举出的全库清单）。

用法:
  python3 scripts/download_ima_queue.py --limit 30 [--min-priority 0] [--sleep 0.3]

行为:
- 读 data_warehouse/ima_export/search_index/queue.jsonl（media_id/title/type/priority）
- 跳过 _downloaded.tsv 已下（全局台账 data_warehouse/ima_export/media/_downloaded.tsv）
- 按 (priority, title) 顺序下载到 data_warehouse/ima_export/media/{模块}/ 下
- 每条成功写入 _downloaded.tsv + 更新 queue.jsonl 的 downloaded 标记
- 命中 220021 限流 -> 记录进度返回码 3
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 云端兼容: 环境变量覆盖 node 路径/API脚本(本机 .archive, 云端 C:/quant/scripts)
_NODE_CMD = os.environ.get("IMA_NODE_CMD", "node")
_IMA_ALT = os.environ.get("IMA_API_JS")
IMA_API = Path(_IMA_ALT) if _IMA_ALT else (ROOT / ".archive" / "skills-20260820" / "ima-skills" / "ima_api.cjs")
CLIENT_FILE = Path.home() / ".config" / "ima" / "client_id"
KEY_FILE = Path.home() / ".config" / "ima" / "api_key"
QUEUE = ROOT / "data_warehouse" / "ima_export" / "search_index" / "queue.jsonl"
MEDIA_ROOT = ROOT / "data_warehouse" / "ima_export" / "media"
DONE_TSV = MEDIA_ROOT / "_downloaded.tsv"

MODULE_DIR = {
    "M1": "M1新闻舆情", "M2": "M2内外资研报", "M3": "M3期货早报",
    "M4": "M4产业链题材", "M5": "M5短线复盘", "M6": "M6会议纪要",
    "M7": "M7闭门会宏观", "M8": "M8规则库",
}
EXT = {1: "pdf", 2: "html", 3: "docx", 4: "ppt", 5: "xlsx", 6: "html",
       7: "md", 8: "md", 9: "png", 10: "jpg", 11: "note", 12: "txt",
       13: "txt", 14: "xmind", 15: "mp3", 16: "mp4", 20: "html", 21: "epub"}


def _opts() -> str:
    return json.dumps({"clientId": CLIENT_FILE.read_text().strip(),
                       "apiKey": KEY_FILE.read_text().strip()})


def call(api_path: str, body: dict) -> dict:
    r = subprocess.run([_NODE_CMD, str(IMA_API), api_path, json.dumps(body, ensure_ascii=False), _opts()],
                       capture_output=True, timeout=60)
    # 云端/本机都按 UTF-8 解码(node 输出 UTF-8; 避免 Windows gbk 崩)
    out = r.stdout.decode("utf-8", errors="replace")
    err = r.stderr.decode("utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(err[:300])
    return json.loads(out)


def get_media_info(media_id: str) -> dict:
    d = call("openapi/wiki/v1/get_media_info", {"media_id": media_id})
    if isinstance(d, dict) and "code" in d and str(d.get("code")) == "220021":
        raise RuntimeError("RATELIMIT_220021")
    return d.get("data", {})


def download(url: str, headers: dict, dest: Path, timeout: int = 120) -> bool:
    import requests
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


def load_done() -> set:
    if not DONE_TSV.exists():
        return set()
    return {l.split("\t")[1] for l in DONE_TSV.read_text(encoding="utf-8").splitlines() if "\t" in l}


def append_done(mid: str, title: str, pathstr: str):
    with DONE_TSV.open("a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{mid}\t{title}\t{pathstr}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--min-priority", type=int, default=0, help="只处理 priority>=此值（0最高优先）")
    ap.add_argument("--sleep", type=float, default=0.3)
    args = ap.parse_args()

    if not QUEUE.exists():
        print(f"[ERR] 队列不存在: {QUEUE}"); return 2
    done = load_done()
    all_items = [json.loads(l) for l in QUEUE.read_text(encoding="utf-8").splitlines() if l.strip()]
    for q in all_items:
        if q.get("media_id") in done:
            q["downloaded"] = True
    items = [q for q in all_items if q["priority"] >= args.min_priority and not q.get("downloaded")]
    todo = [q for q in items if q["media_id"] not in done]
    print(f"队列 {len(items)} 待下，跳过已下 {len(items)-len(todo)}")

    ok = 0; fail = 0
    for q in todo[: args.limit]:
        mid = q["media_id"]; title = q["title"]; mtype = q.get("media_type", "")
        mod = q.get("query", "M2").split(":")[0]
        ddir = MEDIA_ROOT / MODULE_DIR.get(mod, "其他")
        ddir.mkdir(parents=True, exist_ok=True)
        safe = "".join(c for c in title if c not in '\\/:*?"<>|').strip() or f"ima_{mid}"
        if not Path(safe).suffix:
            safe = f"{safe}.{EXT.get(int(mtype or 0), 'bin')}"
        dest = ddir / safe
        if dest.exists() and dest.stat().st_size > 0:
            done.add(mid); append_done(mid, title, str(dest)); q["downloaded"] = True
            continue
        try:
            info = get_media_info(mid)
        except RuntimeError as e:
            if "RATELIMIT" in str(e):
                print(f"[STOP] 220021 限流于 {title}。进度已记录。成功 {ok} 失败 {fail}")
                _save(queue=all_items)
                return 3
            print(f"  [err] {title}: {e}", file=sys.stderr); fail += 1; continue
        url = (info.get("url_info") or {}).get("url")
        headers = (info.get("url_info") or {}).get("headers", {})
        if not url:
            print(f"  [无url] {title}"); fail += 1; continue
        if download(url, headers, dest):
            done.add(mid); append_done(mid, title, str(dest)); q["downloaded"] = True; ok += 1
            print(f"  [OK] {mod}/{dest.name}")
        else:
            fail += 1
        time.sleep(args.sleep)
    _save(queue=all_items)
    print(f"\n=== 完成: 成功 {ok}, 失败 {fail}, 剩余 {len(todo)-ok-fail} ===")
    return 0


def _save(queue: list):
    with QUEUE.open("w", encoding="utf-8") as f:
        for q in queue:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
