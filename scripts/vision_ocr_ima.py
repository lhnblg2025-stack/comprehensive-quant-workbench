#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""视觉模型 OCR：有界 JPEG payload、长图纵向切片与可重试请求。"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "data_warehouse" / "ima_export" / "media" / "extracted"
TARGETS = ROOT / "tmp" / "reocr_targets.txt"
# Prefer a provider whose configured model explicitly supports image input.
# Override with IMA_VISION_PROVIDER / IMA_VISION_MODEL when deploying elsewhere.
PROVIDER = os.environ.get("IMA_VISION_PROVIDER", "image-model")
MODEL = os.environ.get("IMA_VISION_MODEL", "gpt-image-2")
MAX_TOKENS = 8000
MAX_WIDTH = int(os.environ.get("VISION_MAX_WIDTH", "1400"))
MAX_HEIGHT = int(os.environ.get("VISION_MAX_HEIGHT", "2200"))
JPEG_QUALITY = int(os.environ.get("VISION_JPEG_QUALITY", "76"))
CHUNK_OVERLAP = int(os.environ.get("VISION_CHUNK_OVERLAP", "80"))


@dataclass(frozen=True)
class ImageChunk:
    index: int
    total: int
    y_start: int
    y_end: int
    mime: str
    data: bytes

    @property
    def payload_bytes(self) -> int:
        return len(self.data)


def get_credentials():
    """环境变量优先，否则读取现有 openclaw provider 配置。"""
    key = os.environ.get("TSYJZZZ_PRO_API_KEY") or os.environ.get("TSYJZZZ_PLUS_API_KEY")
    if key:
        return "https://your-llm-relay.example.com/v1/chat/completions", key
    with open(os.path.expanduser("~/.openclaw/openclaw.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    p = cfg["models"]["providers"][PROVIDER]
    return p["baseUrl"].rstrip("/") + "/chat/completions", p["apiKey"]


def _prepare_image(img_path: Path, limit_height: bool = False):
    from PIL import Image
    img = Image.open(img_path).convert("RGB")
    scale = min(1.0, MAX_WIDTH / img.width)
    if limit_height:
        scale = min(scale, MAX_HEIGHT / img.height)
    if scale < 1:
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)
    return img


def _encode_image(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=JPEG_QUALITY, optimize=True)
    return buf.getvalue()


def _image_b64(img_path: Path) -> tuple[str, bytes]:
    """编码为有界 JPEG；最长边受 MAX_WIDTH/MAX_HEIGHT 约束。"""
    return "image/jpeg", _encode_image(_prepare_image(Path(img_path), limit_height=True))


def image_chunks(img_path: Path, max_height: int = MAX_HEIGHT,
                 overlap: int = CHUNK_OVERLAP) -> list[ImageChunk]:
    """将图片按缩放后的纵向坐标切片，块间保留 overlap 像素。"""
    img = _prepare_image(Path(img_path))
    if img.height <= max_height:
        return [ImageChunk(0, 1, 0, img.height, "image/jpeg", _encode_image(img))]
    step = max(1, max_height - max(0, min(overlap, max_height - 1)))
    ranges = []
    start = 0
    while start < img.height:
        end = min(img.height, start + max_height)
        ranges.append((start, end))
        if end == img.height:
            break
        start += step
    total = len(ranges)
    return [ImageChunk(i, total, start, end, "image/jpeg", _encode_image(img.crop((0, start, img.width, end))))
            for i, (start, end) in enumerate(ranges)]


def _request_vision(url: str, key: str, mime: str, data: bytes, prompt: str,
                    timeout: int = 600, retries: int = 3, chunk_index: int | None = None) -> str:
    import urllib.error
    import urllib.request
    b64 = base64.b64encode(data).decode("ascii")
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]}],
        "max_tokens": MAX_TOKENS,
    }
    raw = json.dumps(payload).encode()
    label = f"chunk index={chunk_index}" if chunk_index is not None else "chunk index=0"
    last = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            req = urllib.request.Request(url, data=raw, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                status = getattr(response, "status", response.getcode())
                body = response.read().decode("utf-8", errors="replace")
            if status != 200:
                raise RuntimeError(f"HTTP {status}: {body[:300]}")
            parsed = json.loads(body)
            return parsed["choices"][0]["message"]["content"]
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError, RuntimeError) as exc:
            last = exc
            detail = f"payload_bytes={len(raw)} {label} attempt={attempt}/{max(1, retries)}: {exc}"
            if attempt < max(1, retries):
                print(f"[vision] retrying {detail}", file=sys.stderr, flush=True)
                time.sleep(2 ** (attempt - 1))
            else:
                raise RuntimeError(f"vision request failed: {detail}") from exc
    raise RuntimeError(f"vision request failed: payload_bytes={len(raw)} {label}: {last}")


_OCR_PROMPT = "请完整 OCR 这张截图。逐段原样输出正文文字（标题日期、段落、图表说明都保留）；中文原样、英文保留；只输出截图里出现的文字，不要添加、不要总结、不要省略。"


def vision_ocr(img_path: Path, retries: int = 3, timeout: int = 600) -> str:
    """OCR 短图单请求，长图自动切片并按顺序拼接。"""
    url, key = get_credentials()
    chunks = image_chunks(Path(img_path))
    texts = []
    for chunk in chunks:
        texts.append(_request_vision(url, key, chunk.mime, chunk.data, _OCR_PROMPT,
                                     timeout=timeout, retries=retries, chunk_index=chunk.index))
    return "\n".join(t.strip() for t in texts if t.strip())


def process(src: Path) -> bool:
    stem = src.name
    while Path(stem).suffix and Path(Path(stem).stem).suffix:
        stem = Path(stem).stem
    out = OUT_ROOT / f"{stem}.txt"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[OCR] {src.name} ...", flush=True)
    t0 = time.time()
    try:
        text = vision_ocr(src, retries=int(os.environ.get("VISION_RETRIES", "3")))
    except Exception as e:
        print(f"  ✗ {src.name}: {e}", flush=True)
        return False
    out.write_text(text, encoding="utf-8")
    print(f"  ✅ {out.name}  {len(text)}字 耗时{time.time()-t0:.0f}s", flush=True)
    return True


def main() -> int:
    args = sys.argv[1:]
    retries = 3
    if "--retry" in args:
        i = args.index("--retry")
        retries = int(args[i + 1]) if i + 1 < len(args) else 3
        args = args[:i] + args[i + 2:]
    targets = [Path(a) for a in args if Path(a).exists()] if args else [ROOT / l.strip() for l in TARGETS.read_text().splitlines() if l.strip()]
    ok = 0
    for target in targets:
        try:
            if vision_ocr(target, retries=retries):
                ok += 1
        except Exception as e:
            print(f"✗ {target.name}: {e}")
    print(f"\n视觉OCR 成功 {ok}/{len(targets)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
