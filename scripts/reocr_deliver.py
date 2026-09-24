#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OCR 完成后处理：清理 extracted 临时文件 + 把新 OCR 的会议纪要文本存有道云。

用法:
  python3 scripts/reocr_deliver.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "data_warehouse" / "ima_export" / "media" / "extracted"


def clean_junk() -> int:
    """删除 extracted 下临时/垃圾文件：_up_*.png、双重后缀 .png.png.txt 空、.png.png 图源残留。"""
    n = 0
    for p in EXT.rglob("*.png"):
        if p.name.startswith("_up_") or p.name.startswith("_block_"):
            p.unlink(missing_ok=True); n += 1
    for p in EXT.rglob("*"):
        if not p.is_file():
            continue
        # 双重后缀 .png.png.txt（空或残留）
        if p.name.endswith(".png.png.txt") and p.stat().st_size < 50:
            p.unlink(missing_ok=True); n += 1
    return n


def deliver_notes(prefix: str = "IMA会议纪要-") -> int:
    """把 extracted 下新的会议纪要 txt 存有道云（跳过已有台账）。"""
    # 复用 save_to_youdao 的保存逻辑
    sys.path.insert(0, str(ROOT / "scripts"))
    from save_to_youdao import save_note, load_saved_titles, record_saved
    saved = load_saved_titles()
    ok = 0
    # 只处理这 4 张新闻会议纪要图（标题带 美联储/金价/沃什/沉默应对）
    keys = ("美联储会议纪要", "沃什的会议纪要", "金价回落")
    for txt in sorted(EXT.glob("*.txt")):
        name = txt.name
        if not any(k in name for k in keys):
            continue
        # 去掉双重后缀得到干净标题
        stem = txt.name
        while Path(stem).suffix and Path(Path(stem).stem).suffix:
            stem = Path(stem).stem
        if stem.endswith(".txt"):
            stem = stem[:-4]
        content = txt.read_text(encoding="utf-8", errors="ignore").strip()
        if len(content) < 30:
            print(f"  [跳过-文本过短] {stem} ({len(content)}字)")
            continue
        title = f"{prefix}{stem}"
        if title in saved:
            print(f"  [跳过-已存] {title}")
            continue
        if save_note(title, content):
            ok += 1
            record_saved(title, txt)
            print(f"  [存有道云] {title}")
    return ok


def main() -> int:
    cleaned = clean_junk()
    print(f"清理临时文件 {cleaned} 个")
    ok = deliver_notes()
    print(f"\n有道云保存 {ok} 条")
    return 0 if ok >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
