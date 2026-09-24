#!/usr/bin/env python3
"""OCR研报文件夹图片，输出结构化分析与可浏览HTML。"""
from __future__ import annotations
import argparse, json, os, re
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
import pytesseract

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "generated" / "research_images.json"
OUT_HTML = ROOT / "generated" / "research_images.html"
IMAGE_DIRS = [
    Path.home() / "Desktop" / "研报共享",
    ROOT / "研究报告",
    ROOT / "generated",
]
IMG_SUFFIX = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
LANGS = "chi_sim+eng"


def _text_from_image(path: Path) -> str:
    img = Image.open(path)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return pytesseract.image_to_string(img, lang=LANGS)


def _analyze(text: str) -> dict:
    """从OCR文本抽取轻量结构化线索，不编造。"""
    t = text.strip()
    return {
        "length": len(t),
        "has_kpi": bool(re.search(r"(涨|跌|涨幅|跌幅|MA|MACD|PE|PB|ROE|支撑|压力|目标价|评级|买入|卖出|增持|减持)", t)),
        "mentions_etf": bool(re.search(r"ETF|基金|申赎|份额|净申购|净赎回", t, re.I)),
        "mentions_institution": bool(re.search(r"汇金|证金|国家队|社保|险资|公募|私募", t)),
    }


def scan(refresh: bool = False) -> list[dict]:
    if OUT_JSON.exists() and not refresh:
        try:
            return json.loads(OUT_JSON.read_text(encoding="utf-8"))
        except Exception:
            pass
    rows = []
    seen = set()
    for base in IMAGE_DIRS:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix.lower() not in IMG_SUFFIX:
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            try:
                text = _text_from_image(path)
                analysis = _analyze(text)
            except Exception as exc:
                text, analysis = "", {"error": f"{type(exc).__name__}: {str(exc)[:120]}"}
            rows.append({
                "path": str(path), "name": path.name,
                "dir": str(path.parent),
                "size": path.stat().st_size if path.exists() else 0,
                "text": text[:4000],
                "analysis": analysis,
                "ocr_at": datetime.now(timezone.utc).isoformat(),
            })
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def render_html(rows: list[dict]) -> str:
    cards = []
    for i, r in enumerate(rows):
        a = r.get("analysis") or {}
        if a.get("error"):
            card = f"<div class='card error'><div class='fname'>{r['name']}</div><pre>{a['error']}</pre></div>"
        else:
            tags = " ".join([
                "分析要点" if a.get("has_kpi") else "",
                "ETF/基金" if a.get("mentions_etf") else "",
                "机构线索" if a.get("mentions_institution") else "",
            ]).strip() or "无明确线索"
            card = f"""<div class='card'><div class='fname'>{r['name']}</div><div class='meta'>{tags} · {a.get('length',0)}字 · {r['dir']}</div><pre>{r['text'][:1200]}</pre></div>"""
        cards.append(card)
    return f"""<!doctype html><html lang=zh><head><meta charset=utf-8><title>研报图片OCR分析</title>
<style>*{{box-sizing:border-box;margin:0;padding:0}}body{{background:#0d1117;color:#e6edf3;font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;padding:20px}}
h1{{font-size:20px;margin-bottom:6px}}.sub{{color:#8b949e;font-size:12px;margin-bottom:16px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:12px}}
.fname{{font-weight:700;margin-bottom:6px}}.meta{{color:#8b949e;font-size:12px;margin-bottom:8px}}
pre{{white-space:pre-wrap;font-size:13px;color:#c9d1d9;max-height:360px;overflow:auto}}
.error{{border-color:#f85149}}</style></head><body><h1>📷 研报图片OCR分析</h1>
<div class=sub>共 {len(rows)} 张图片 · 生成 {datetime.now().strftime('%Y-%m-%d %H:%M')}</div><div class=grid>{''.join(cards)}</div></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="强制重新OCR")
    ap.add_argument("--html-only", action="store_true", help="仅用已有JSON重建HTML")
    args = ap.parse_args()
    rows = json.loads(OUT_JSON.read_text(encoding="utf-8")) if args.html_only and OUT_JSON.exists() else scan(refresh=args.refresh)
    OUT_HTML.write_text(render_html(rows), encoding="utf-8")
    print(f"OCR rows={len(rows)} -> {OUT_HTML}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
