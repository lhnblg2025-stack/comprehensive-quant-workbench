#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 _items.jsonl 台账批量下载 IMA 媒体原文(PDF 优先)到对应日目录。

用法:
  python3 scripts/download_ima_pdfs.py --kb "-ppx..." \
      --month "研报库/一：彭博Bloomberg路透社投研报告/一、彭博Bloomberg研报🌏/2026年/8月" \
      --sleep 0.3 [--limit N] [--reverse]

行为:
- 遍历 month 下每个日目录(非8位数字目录跳过)，读 _items.jsonl
- 对每条 media 调 get_media_info 拿 url/headers，下载到该日目录下(文件名=原 title, 不重命名)
- 已下载记录到 <month>/_downloaded.tsv(按 media_id 去重)
- 命中 220021 限流 -> 记录进度并停止返回码 3
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMA_API = ROOT / ".archive" / "skills-20260820" / "ima-skills" / "ima_api.cjs"
CLIENT_FILE = Path.home() / ".config" / "ima" / "client_id"
KEY_FILE = Path.home() / ".config" / "ima" / "api_key"
GLOBAL_DONE_TSV = ROOT / "data_warehouse" / "ima_export" / "media" / "_downloaded.tsv"

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

def get_media_info(media_id: str) -> dict:
    d = call("openapi/wiki/v1/get_media_info", {"media_id": media_id})
    if isinstance(d, dict) and "code" in d and str(d.get("code")) == "220021":
        raise RuntimeError("RATELIMIT_220021")
    return d.get("data", {})

def download(url: str, headers: dict, dest: Path, timeout: int = 90) -> bool:
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

def ext_for_media_type(t):
    return {1:"pdf",2:"html",3:"docx",4:"ppt",5:"xlsx",6:"html",7:"md",8:"md",
            9:"png",10:"jpg",11:"note",12:"txt",13:"txt",14:"xmind",15:"mp3",
            16:"mp4",20:"html",21:"epub"}.get(int(t or 0),"bin")

def load_done(tsv: Path) -> set:
    if not tsv.exists():
        return set()
    s=set()
    for line in tsv.read_text(encoding="utf-8").splitlines():
        cols=line.split("\t")
        if len(cols)>=2: s.add(cols[1])
    return s

def append_done(tsv: Path, mid, title, pathstr, ext):
    tsv.parent.mkdir(parents=True, exist_ok=True)
    with tsv.open("a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{mid}\t{title}\t{pathstr}\t{ext}\n")

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--kb", required=True)
    ap.add_argument("--month", required=True, help="月份目录，含多个 8位数字 日目录")
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reverse", action="store_true", help="日目录倒序处理")
    args=ap.parse_args()

    month=Path(args.month)
    if not month.exists():
        print(f"[ERR] 月份目录不存在: {month}"); return 2
    tsv=month/"_downloaded.tsv"
    done=load_done(tsv)
    global_done=load_done(GLOBAL_DONE_TSV)
    done |= global_done

    day_dirs=[d for d in month.iterdir() if d.is_dir() and d.name.isdigit() and len(d.name)==8]
    day_dirs.sort(reverse=args.reverse)
    if not day_dirs:
        print(f"[INFO] {month} 无 8位数字 日目录"); return 0

    total_ok=0; total_skip=0; total_fail=0
    for d in day_dirs:
        jf=d/"_items.jsonl"
        if not jf.exists():
            print(f"[SKIP] {d.name}: 无 _items.jsonl"); continue
        items=[json.loads(l) for l in jf.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"\n=== {d.name}: {len(items)} items ===")
        for it in items:
            mid=it.get("media_id","")
            title=str(it.get("title",f"ima_{mid}"))
            mtype=it.get("media_type")
            if not mid:
                total_fail+=1; continue
            if mid in done:
                total_skip+=1; continue
            safe="".join(c for c in title if c not in '\\/:*?"<>|').strip() or f"ima_{mid}"
            # 若 title 已带扩展名则直接用，否则按 media_type 补
            if not Path(safe).suffix:
                safe=f"{safe}.{ext_for_media_type(mtype)}"
            dest=d/safe
            if dest.exists() and dest.stat().st_size>0:
                if mid not in load_done(tsv):
                    append_done(tsv,mid,title,str(dest),dest.suffix)
                if mid not in global_done:
                    append_done(GLOBAL_DONE_TSV,mid,title,str(dest),dest.suffix)
                    global_done.add(mid)
                done.add(mid)
                total_skip+=1; continue
            try:
                info=get_media_info(mid)
            except RuntimeError as e:
                if "RATELIMIT" in str(e):
                    print(f"[STOP] 220021 限流于 {d.name}/{title}. 进度已写入 {tsv}")
                    print(f"\n=== 汇总: 下载成功 {total_ok}, 跳过 {total_skip}, 失败 {total_fail} ===")
                    return 3
                print(f"  [err] {title}: {e}", file=sys.stderr)
                total_fail+=1; continue
            url=(info.get("url_info") or {}).get("url")
            headers=(info.get("url_info") or {}).get("headers", {})
            if not url:
                print(f"  [无url] {title}")
                total_fail+=1; continue
            if download(url, headers, dest):
                done.add(mid); append_done(tsv,mid,title,str(dest),dest.suffix)
                if mid not in global_done:
                    append_done(GLOBAL_DONE_TSV,mid,title,str(dest),dest.suffix)
                    global_done.add(mid)
                total_ok+=1
                print(f"  [OK] {dest.name}")
            else:
                total_fail+=1
            time.sleep(args.sleep)
    print(f"\n=== 完成 {month}: 下载成功 {total_ok}, 跳过(已下载) {total_skip}, 失败 {total_fail} ===")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
