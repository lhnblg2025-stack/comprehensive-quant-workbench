#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量把文本文件/JSONL 内容保存为有道云笔记（调用 youdaonote-clip）。

用法:
  python3 scripts/save_to_youdao.py --dir data_warehouse/ima_export/media/会议纪要/extracted --prefix "IMA会议纪要-"
  python3 scripts/save_to_youdao.py --jsonl data_warehouse/ima_export/media/会议纪要/extracted.jsonl --content-key path
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLIP_MJS = ROOT / "skills" / "youdaonote-clip" / "clip-note.mjs"
SSE_URL = "https://open.mail.163.com/api/ynote/mcp/sse"
API_KEY = os.environ.get("YOUDAONOTE_API_KEY", "").strip()
TIMEOUT = int(os.environ.get("YOUDAONOTE_TIMEOUT", "120"))
SAVED_LEDGER = ROOT / "data_warehouse" / "ima_export" / "media" / "youdao_saved.jsonl"


def load_saved_titles() -> set:
    if not SAVED_LEDGER.exists():
        return set()
    s = set()
    for line in SAVED_LEDGER.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                s.add(json.loads(line).get("title", ""))
            except Exception as e:
                logging.getLogger("save_to_youdao").warning(
                    f"[save_to_youdao] 账本坏行跳过: {line[:60]!r} ({type(e).__name__})")
    return s


def record_saved(title: str, src: str):
    with SAVED_LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"title": title, "src": str(src),
                            "time": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False) + "\n")


def _redact_secret(s: str) -> str:
    return s.replace(API_KEY, "<YOUDAONOTE_API_KEY>") if API_KEY else s


def save_note(title: str, content: str, retries: int = 2) -> bool:
    """调用有道云 createNote。失败返回 False。SSE 断连自动重试。"""
    if not API_KEY:
        print("  [错误] 缺少 YOUDAONOTE_API_KEY，拒绝使用源码兜底密钥", file=sys.stderr)
        return False
    # 2026-08-21 审计: 内容里行首 "-"/"--"/"- " 等会被 clip-note 的 minimist 当成选项参数，
    # 导致 "Option '--content' argument is ambiguous"。对这类行加全角引导，避免歧义。
    safe_lines = []
    for ln in content.splitlines():
        if ln.startswith("---") or (ln.strip() and ln.strip()[0] in "-+="):
            safe_lines.append("　" + ln)
        else:
            safe_lines.append(ln)
    content = "\n".join(safe_lines)
    for attempt in range(1, retries + 1):
        tmp_name = ""
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as tmp:
                tmp.write(content)
                tmp_name = tmp.name
            env = dict(os.environ)
            env["YOUDAONOTE_API_KEY"] = API_KEY
            r = subprocess.run(
                ["node", str(CLIP_MJS), "--create-note", "--title", title[:100],
                 "--content-file", tmp_name, "--sse-url", SSE_URL],
                capture_output=True, text=True, timeout=TIMEOUT, env=env,
            )
            if r.returncode == 0 and "成功" in r.stdout:
                return True
            msg = _redact_secret(f"{r.stdout[:150]}{r.stderr[:150]}")
            print(f"  [失败-{attempt}/{retries}] {title}: {msg}", file=sys.stderr)
        except subprocess.TimeoutExpired:
            print(f"  [超时-{attempt}/{retries}] {title}", file=sys.stderr)
        except Exception as e:
            print(f"  [错误-{attempt}/{retries}] {title}: {e}", file=sys.stderr)
        finally:
            if tmp_name:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
        if attempt < retries:
            time.sleep(3)  # 等待 SSE 重连
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="", help="目录下所有 .txt 保存")
    ap.add_argument("--jsonl", default="", help="jsonl 列表，用 --path-key 取文件路径")
    ap.add_argument("--path-key", default="extracted")
    ap.add_argument("--prefix", default="IMA导出-")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--sleep", type=float, default=0.6)
    args = ap.parse_args()

    files: list[Path] = []
    titles: dict[Path, str] = {}
    if args.dir:
        d = Path(args.dir)
        for f in sorted(d.glob("*.txt")):
            files.append(f)
            titles[f] = f.stem
    elif args.jsonl:
        with open(args.jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                p = obj.get(args.path_key) or obj.get("path")
                if p and Path(p).exists():
                    fp = Path(p)
                    files.append(fp)
                    titles[fp] = obj.get("title") or fp.stem
    else:
        print("需要 --dir 或 --jsonl", file=sys.stderr)
        return 1

    ok = 0
    saved_titles = load_saved_titles()
    for i, fp in enumerate(files[: args.limit]):
        try:
            content = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"  [读取失败] {fp}: {e}", file=sys.stderr)
            continue
        title = f"{args.prefix}{titles.get(fp, fp.stem)}"
        if title in saved_titles:
            print(f"({i+1}/{min(len(files), args.limit)}) 跳过(已存) {title}")
            continue
        print(f"({i+1}/{min(len(files), args.limit)}) 保存 {title}")
        if save_note(title, content):
            ok += 1
            record_saved(title, fp)
        time.sleep(args.sleep)
    print(f"\n完成：成功 {ok}/{min(len(files), args.limit)}")
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
