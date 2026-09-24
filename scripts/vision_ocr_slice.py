#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""超长图竖向切片 + 视觉 OCR 拼接（解决超高图压缩后文字过小/请求体过大的问题）。

用法:
  python3 scripts/vision_ocr_slice.py <图片路径> [切片高度px=7000]
写入 extracted/<原始名单后缀>.txt
"""
from __future__ import annotations
import base64, io, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "data_warehouse" / "ima_export" / "media" / "extracted"
PROVIDER = "chatgpt-plus"
MODEL = "gpt-5.5"
MAX_TOKENS = 8000


def get_credentials():
    cfg = json.load(open(os.path.expanduser("~/.openclaw/openclaw.json")))
    p = cfg["models"]["providers"][PROVIDER]
    return p["baseUrl"].rstrip("/") + "/chat/completions", p["apiKey"]


def vision_ocr_b64(url, key, mime, data_b64, retries: int = 3) -> str:
    import urllib.request
    import urllib.error
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "请完整 OCR 这张新闻截图。逐段原样输出正文文字（中英都保留、标题日期段落、图表说明），只输出截图里的文字，不要总结省略。"},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data_b64}"}},
        ]}],
        "max_tokens": MAX_TOKENS,
    }
    last = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.loads(r.read().decode())
            return d["choices"][0]["message"]["content"]
        except Exception as e:
            last = e
            print(f"    重试{attempt}/{retries}: {e}", flush=True)
            time.sleep(3 * attempt)
    raise last


def slice_ocr(src: Path, slice_h: int = 6000) -> str:
    from PIL import Image
    url, key = get_credentials()
    img = Image.open(src).convert("RGB")
    w, h = img.size
    print(f"  {w}x{h} 分 {max(1, -(-h // slice_h))} 段", flush=True)
    parts = []
    y = 0
    seg = 1
    while y < h:
        crop = img.crop((0, y, w, min(y + slice_h, h)))
        buf = io.BytesIO()
        crop.save(buf, "JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        print(f"  段{seg}: OCR ...", flush=True)
        try:
            t = vision_ocr_b64(url, key, "image/jpeg", b64)
        except Exception as e:
            print(f"  段{seg} 失败: {e}", flush=True)
            t = ""
        if t.strip():
            parts.append(t.strip())
            print(f"  段{seg} ok {len(t)}字", flush=True)
        seg += 1
        y += slice_h
        time.sleep(1)
    return "\n\n---\n\n".join(parts)


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: python scripts/vision_ocr_slice.py <图片> [slice_h]"); return 1
    src = Path(sys.argv[1])
    slice_h = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    stem = src.name
    while Path(stem).suffix and Path(Path(stem).stem).suffix:
        stem = Path(stem).stem
    out = OUT_ROOT / f"{stem}.txt"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    text = slice_ocr(src, slice_h)
    out.write_text(text, encoding="utf-8")
    print(f"✅ {out.name} {len(text)}字 耗时{time.time()-t0:.0f}s")
    return 0 if len(text.strip()) > 100 else 1


if __name__ == "__main__":
    raise SystemExit(main())
