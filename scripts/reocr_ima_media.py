#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""鲁棒 OCR 重跑：放大 + 降噪 + 双语言，覆盖 extract 单次 tesseract 失败的目标。

用法:
  python3 scripts/reocr_ima_media.py            # 处理 tmp/reocr_targets.txt 列表
  python3 scripts/reocr_ima_media.py <file>    # 处理单个文件
"""
from __future__ import annotations
import subprocess, sys, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "data_warehouse" / "ima_export" / "media" / "extracted"
TARGETS_FILE = ROOT / "tmp" / "reocr_targets.txt"


def ocr_run(img_path: Path, psm: int = 6) -> str:
    """tesseract 放大后识别（chi_sim+eng），psm 可调。对超高图按块裁剪识别。"""
    from PIL import Image
    img = Image.open(img_path)
    w, h = img.size
    # 超高长截图(>4000px)分块，避免 tesseract 超时/内存爆
    if h > 4000:
        block_h = 1800
        parts = []
        y = 0
        while y < h:
            crop = img.crop((0, y, w, min(y + block_h, h)))
            tmp = img_path.parent / f"_block_{y}.png"
            crop.save(tmp)
            r = subprocess.run(
                ["tesseract", str(tmp), "-", "--psm", str(psm)],
                capture_output=True, text=True, timeout=300,
            )
            tmp.unlink(missing_ok=True)
            if r.stdout.strip():
                parts.append(r.stdout.strip())
            y += block_h
        return "\n\n".join(parts)
    r = subprocess.run(
        ["tesseract", str(img_path), "-", "--psm", str(psm)],
        capture_output=True, text=True, timeout=300,
    )
    return r.stdout


def upgrade_image(src: Path, dst: Path, scale: int = 3) -> bool:
    """PIL 放大 + 灰度 + 增强，返回是否成功。"""
    try:
        from PIL import Image, ImageOps, ImageEnhance
        img = Image.open(src).convert("L")
        w, h = img.size
        img = img.resize((w * scale, h * scale), Image.LANCZOS)
        img = ImageEnhance.Contrast(img).enhance(1.8)
        img.save(dst)
        return True
    except Exception as e:
        print(f"  放大失败: {e}", file=sys.stderr)
        return False


def ocr_pdf_scans(pdf: Path) -> str:
    """扫描 PDF 逐页 pdftoppm → tesseract 拼接。"""
    import tempfile
    chunks = []
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        r = subprocess.run(["pdftoppm", "-png", "-r", "300", str(pdf), str(td_p / "p")],
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            return f"[OCR_ERROR:pdftoppm] {r.stderr[:100]}"
        pages = sorted(td_p.glob("p-*.png"))
        for pg in pages:
            up = td_p / f"up_{pg.name}"
            if upgrade_image(pg, up, scale=2):
                t = ocr_run(up)
                if t.strip():
                    chunks.append(t)
        return "\n\n---PAGE---\n\n".join(c.strip() for c in chunks if c.strip())


def process(src: Path) -> None:
    print(f"\n=== OCR: {src.name} ===")
    # 去掉重复后缀 .png.png → 用单后缀 .png.txt（与 extract_ima_media 一致）
    out_name = src.name
    stem = src.name
    while Path(stem).suffix and Path(Path(stem).stem).suffix:
        stem = Path(stem).stem  # .png.png -> .png
    out = OUT_ROOT / f"{stem}.txt"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    text = ""
    if src.suffix.lower() == ".pdf":
        text = ocr_pdf_scans(src)
    elif src.suffix.lower() in (".png", ".jpg", ".jpeg"):
        from PIL import Image
        _dim = Image.open(src).size
        up = OUT_ROOT / f"_up_{src.stem}.png"
        if _dim[1] > 4000:
            # 超长图不放大，直接分块（eng-only）；块内中文多时切 chi_sim
            text = ocr_run(src)
            han = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            if han >= 10:
                text = ocr_run_mixed(src)
        elif upgrade_image(src, up, scale=2):
            text = ocr_run(up)
        if up.exists():
            up.unlink()
    old = out.read_text(encoding="utf-8", errors="ignore") if out.exists() else ""
    new_chars = len(text.strip())
    old_chars = len(old.strip())
    out.write_text(text, encoding="utf-8")
    better = new_chars > old_chars
    print(f"  输出 {out.name}: {old_chars}→{new_chars} 字符 {'✅提升' if better else '⚠️未提升'}")
    if better and new_chars > 200:
        print(f"  预览: {text.strip()[:120]}")


def main() -> int:
    args = sys.argv[1:]
    if args:
        targets = [Path(a) for a in args if Path(a).exists()]
    else:
        if not TARGETS_FILE.exists():
            print(f"目标文件缺失: {TARGETS_FILE}"); return 1
        targets = [ROOT / l.strip() for l in TARGETS_FILE.read_text().splitlines()
                   if l.strip() and (ROOT / l.strip()).exists()]
    if not targets:
        print("无有效目标"); return 1
    print(f"处理 {len(targets)} 个文件")
    for t in targets:
        try:
            process(t)
        except Exception as e:
            print(f"✗ {t.name}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
