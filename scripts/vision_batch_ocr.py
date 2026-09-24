#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""云端批量视觉 OCR（2026-08-22 —— 替换 reocr_ima_media 的 tesseract 依赖）

云端无 tesseract 二进制 → 改 gpt-5.5(PRO key) 视觉通道:
  - 遍历 data_warehouse/ima_export/media 下未识别图片(.png/.jpg 无同名 .txt)
  - vision_ocr_ima.vision_ocr 逐张识别(带3次重试)
  - 产出 {img}.txt

用法:
  python3 scripts/vision_batch_ocr.py [--limit N] [--regen]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEDIA = ROOT / "data_warehouse" / "ima_export" / "media"
sys.path.insert(0, str(ROOT / "scripts"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass


def scan_pending(limit: int = 0) -> list[Path]:
    """未识别图片(无同名 .txt 或空 txt)。"""
    cands = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for p in MEDIA.rglob(ext):
            txt = p.with_suffix(p.suffix + ".txt")
            if not txt.exists() or txt.stat().st_size == 0:
                cands.append(p)
    if limit:
        cands = cands[:limit]
    return cands


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--regen", action="store_true", help="强制重识别(即使已有txt)")
    args = ap.parse_args()

    import vision_ocr_ima as vo  # noqa: PLC0415  (PRO key 优先已改)

    cands = scan_pending(args.limit)
    if args.regen:
        cands = list(MEDIA.rglob("*.png"))[:args.limit or 999]
    print(f"[vision_batch] 待识别: {len(cands)}", flush=True)
    ok = fail = 0
    for img in cands:
        txt = img.with_suffix(img.suffix + ".txt")
        if txt.exists() and txt.stat().st_size > 0 and not args.regen:
            ok += 1
            continue
        for attempt in range(3):
            try:
                t0 = time.time()
                text = vo.vision_ocr(str(img))
                txt.write_text(text, encoding="utf-8")
                print(f"  OK {os.path.basename(str(img))[:40]} {time.time()-t0:.0f}s len={len(text)}", flush=True)
                ok += 1
                break
            except Exception as e:  # noqa: BLE001
                print(f"    retry {attempt+1} {os.path.basename(str(img))[:30]}: {str(e)[:50]}", flush=True)
                time.sleep(4)
        else:
            print(f"  FAIL {os.path.basename(str(img))[:40]}", flush=True)
            fail += 1
    print(f"[vision_batch] DONE ok={ok} fail={fail}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:
        print(f'[vision_batch] 异常{str(e)[:80]}, exit 0');
        sys.exit(0)