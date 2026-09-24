#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量提取 IMA 导出媒体文本：PDF/txt/docx(仅解压xml兜底)/图片OCR。

输出到 data_warehouse/ima_export/media/{子目录}/extracted/*.txt 并生成 extracted.jsonl
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


_VISION_MODEL = None  # 例如 "qwen/qwen3.5-plus"（input 含 image）


def ocr_with_vision(path: Path) -> tuple[str, bool]:
    """可选的视觉模型 OCR 后端（OpenClaw image 模型）。

    当前 opencode go 的 gpt-5.6-luna 网关不支持图片输入(403/地区限制)，
    因此默认不启用；若配置了支持 image 的模型(如 qwen3.5-plus)可在此调用。
    返回 (text, ok)。
    """
    global _VISION_MODEL
    if not _VISION_MODEL:
        return ("", False)
    try:
        import base64, json, urllib.request
        import os
        # 读取 openclaw.json 找该 provider 的 baseUrl/apiKey（示例实现）
        cfg = json.load(open(os.path.expanduser("~/.openclaw/openclaw.json"), encoding="utf-8"))
        prov = cfg["models"]["providers"]
        prov_name, _, model_id = _VISION_MODEL.partition("/")
        p = prov.get(prov_name, {})
        base = p.get("baseUrl", "").rstrip("/") + "/chat/completions"
        apikey = p.get("apiKey", "")
        b64 = base64.b64encode(path.read_bytes()).decode()
        ext = path.suffix.lower().lstrip(".") or "png"
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "webp": "image/webp", "bmp": "image/bmp"}.get(ext, "image/png")
        payload = {"model": model_id, "messages": [{"role": "user", "content": [
            {"type": "text", "text": "请 OCR 这张研报/新闻/概念股截图，原样输出文字，保留表格信息。"},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]}]}
        req = urllib.request.Request(base, data=json.dumps(payload).encode(),
                                     headers={"Authorization": f"Bearer {apikey}",
                                              "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read().decode())
        text = d["choices"][0]["message"].get("content", "")
        return text, bool(text.strip())
    except Exception as e:
        return f"[VISION_ERR] {e}", False


def extract_file(path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    stem = path.stem
    text = ""
    ok = False
    method = ""
    if ext == ".txt":
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            ok = True; method = "txt"
        except Exception as e:
            text = f"[ERR] {e}"
    elif ext == ".pdf":
        r = subprocess.run(["pdftotext", "-layout", str(path), "-"],
                           capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            text = r.stdout
            ok = True; method = "pdftotext"
        else:
            text = f"[ERR] {r.stderr[:200]}"
    elif ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        try:
            import sys as _s
            # 可选：视觉模型后端（默认未启用，_VISION_MODEL 为空则用 tesseract）
            if _VISION_MODEL:
                vt, vok = ocr_with_vision(path)
                if vok:
                    text = vt; ok = True; method = "vision"
                    out = out_dir / f"{stem}.txt"
                    out.write_text(text, encoding="utf-8")
                    return {"path": str(path), "extracted": str(out), "ok": True,
                            "method": "vision", "chars": len(text)}
            _s.path.insert(0, str(ROOT))
            from PIL import Image
            from quant_system.ocr_util import ocr_image, is_ocr_error
            # 大图先降采样到最长边 ≤2000px，显著加速 tesseract 且保留文字可读性
            img = Image.open(path)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            w, h = img.size
            max_side = 2000
            if max(w, h) > max_side:
                scale = max_side / float(max(w, h))
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            text = ocr_image(img)
            if is_ocr_error(text):
                text = f"[OCR_ERROR] {text}"
            else:
                ok = True; method = "ocr_resized"
        except Exception as e:
            text = f"[ERR] {e}"
    elif ext in (".docx",):
        # 无 python-docx 时用 unzip 提取 document.xml 并粗略去标签
        try:
            import zipfile, re
            with zipfile.ZipFile(path) as z:
                xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
            # 段落分隔
            xml = xml.replace("</w:p>", "\n")
            text = re.sub(r"<[^>]+>", "", xml)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if text:
                ok = True; method = "docx_xml"
            else:
                text = "[ERR] docx 无文本"
        except Exception as e:
            text = f"[ERR] {e}"
    elif ext in (".doc", ".xls", ".xlsx", ".ppt", ".pptx"):
        text = f"[SKIP] 暂不支持 {ext} 提取，保留原始文件"
    else:
        text = f"[SKIP] 未知类型 {ext}"

    out = out_dir / f"{stem}.txt"
    out.write_text(text, encoding="utf-8")
    return {"path": str(path), "extracted": str(out), "ok": ok, "method": method,
            "chars": len(text)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "data_warehouse" / "ima_export" / "media"))
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()
    root = Path(args.dir)
    out_root = root / "extracted"
    out_root.mkdir(parents=True, exist_ok=True)
    records = []
    for f in sorted(root.rglob("*")):
        if f.suffix.lower() not in (".pdf", ".txt", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".docx"):
            continue
        if "extracted" in f.parts or f.name == "meta.jsonl":
            continue
        # 增量提取：目标文本存在且不早于源文件时跳过，避免每天重复 OCR/PDF 解析。
        out_file = out_root / f"{f.stem}.txt"
        if out_file.exists() and out_file.stat().st_mtime >= f.stat().st_mtime:
            rec = {"path": str(f), "extracted": str(out_file), "ok": True,
                   "method": "cached", "chars": len(out_file.read_text(encoding="utf-8", errors="ignore"))}
            records.append(rec)
            continue
        rec = extract_file(f, out_root)
        records.append(rec)
        print(f"{'✅' if rec['ok'] else '⚠️'} {rec['path']} ({rec['method']}, {rec['chars']}字)")
    out_json = Path(args.out_json) if args.out_json else root / "extracted.jsonl"
    with out_json.open("w", encoding="utf-8") as fo:
        for r in records:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n完成：{len(records)} 个文件 → {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
