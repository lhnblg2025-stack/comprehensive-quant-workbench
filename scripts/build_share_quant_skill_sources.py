#!/usr/bin/env python3
"""Inventory and extract quant/trading book sources from shared folders.

Outputs are written under generated/share_quant_skill_build/. The script avoids
copying source files and only creates derived text/metadata for skill synthesis.
"""

from __future__ import annotations

import csv
import html
import json
import re
import shutil
import subprocess
import sys
import zipfile
from html.parser import HTMLParser
from pathlib import Path


ROOTS = [
    Path("/mnt/hgfs/share"),
    Path("/mnt/hgfs/共享文件夹/books"),
    Path("/mnt/hgfs/共享文件夹/pdf"),
    Path("/mnt/hgfs/共享文件夹/量化"),
]

OUT_DIR = Path("generated/share_quant_skill_build")
TEXT_DIR = OUT_DIR / "texts"

BOOK_EXTS = {".pdf", ".epub", ".mobi", ".azw3"}
EXCLUDE_RE = re.compile(
    r"(~lock|ExpenseReceipt|eTicket|invitation|assessment brief|grading rubric|referencing guide|"
    r"SDG|test_write|gold_report|zsxq_research|联创电子|\.xlsx$|\.csv$|\.zip$|\.html$)",
    re.I,
)
QUANT_RE = re.compile(
    r"(量化|算法交易|交易|投机|操盘|股票|证券|投资|估值|财务报表|金融|市场|技术分析|"
    r"均线|量价|威科夫|斐波那契|海龟|利弗莫尔|欧奈尔|多空|起涨|成交量|资金|"
    r"portfolio|trading|algorithmic|quantitative|financial machine learning|market microstructure|"
    r"valuation|damodaran|penman|oneil|wyckoff|fibonacci|turtle|livermore|appel)",
    re.I,
)


class TextHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip = False

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag.lower() in {"script", "style"}:
            self.skip = True
        if tag.lower() in {"p", "br", "div", "section", "h1", "h2", "h3", "li"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style"}:
            self.skip = False
        if tag.lower() in {"p", "div", "section", "li"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip:
            self.parts.append(data)

    def text(self) -> str:
        raw = html.unescape(" ".join(self.parts))
        raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
        raw = re.sub(r"\n\s*\n\s*\n+", "\n\n", raw)
        return raw.strip()


def run(cmd: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)


def safe_stem(path: Path) -> str:
    stem = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", path.stem).strip("._")
    if len(stem) > 120:
        stem = stem[:120]
    return stem or "source"


def list_sources() -> list[Path]:
    files: list[Path] = []
    for root in ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in BOOK_EXTS:
                continue
            text_path = str(path)
            if EXCLUDE_RE.search(text_path):
                continue
            if QUANT_RE.search(text_path):
                files.append(path)
    return sorted(set(files), key=lambda p: str(p).lower())


def pdf_pages(path: Path) -> int | None:
    if not shutil.which("pdfinfo"):
        return None
    proc = run(["pdfinfo", str(path)], timeout=60)
    match = re.search(r"^Pages:\s+(\d+)", proc.stdout, re.M)
    return int(match.group(1)) if match else None


def extract_pdf(path: Path, out_path: Path) -> tuple[int, str]:
    if not shutil.which("pdftotext"):
        return 0, "pdftotext_missing"
    proc = run(["pdftotext", "-layout", "-enc", "UTF-8", str(path), str(out_path)], timeout=300)
    if proc.returncode != 0:
        return 0, f"pdftotext_failed:{proc.stderr[:200]}"
    text = out_path.read_text(encoding="utf-8", errors="ignore") if out_path.exists() else ""
    return len(text.strip()), "text_layer"


def extract_epub(path: Path, out_path: Path) -> tuple[int, str]:
    try:
        parts: list[str] = []
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith((".html", ".htm", ".xhtml"))]
            for name in names:
                try:
                    data = zf.read(name)
                except KeyError:
                    continue
                parser = TextHTMLParser()
                parser.feed(data.decode("utf-8", errors="ignore"))
                text = parser.text()
                if text:
                    parts.append(text)
        full_text = "\n\n".join(parts).strip()
        out_path.write_text(full_text, encoding="utf-8")
        return len(full_text), "epub_html"
    except zipfile.BadZipFile:
        return 0, "bad_epub_zip"


def classify_status(ext: str, chars: int, pages: int | None) -> str:
    if ext == ".mobi" or ext == ".azw3":
        return "needs_converter"
    if ext == ".pdf" and pages and chars < max(800, pages * 80):
        return "needs_ocr"
    if chars >= 1000:
        return "usable_text"
    return "low_text"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    sources = list_sources()
    rows: list[dict[str, object]] = []
    for idx, path in enumerate(sources, 1):
        ext = path.suffix.lower()
        out_path = TEXT_DIR / f"{idx:03d}_{safe_stem(path)}.txt"
        pages = pdf_pages(path) if ext == ".pdf" else None
        chars = 0
        method = "unsupported"
        if ext == ".pdf":
            chars, method = extract_pdf(path, out_path)
        elif ext == ".epub":
            chars, method = extract_epub(path, out_path)
        elif ext in {".mobi", ".azw3"}:
            out_path = Path("")
            method = "needs_calibre_or_kindlegen"
        rows.append(
            {
                "idx": idx,
                "path": str(path),
                "name": path.name,
                "ext": ext,
                "size_bytes": path.stat().st_size,
                "pages": pages or "",
                "chars": chars,
                "method": method,
                "status": classify_status(ext, chars, pages),
                "text_path": str(out_path) if out_path else "",
            }
        )
        print(f"[{idx}/{len(sources)}] {classify_status(ext, chars, pages):15s} {chars:8d} {path.name}")

    csv_path = OUT_DIR / "source_inventory.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["idx", "path"])
        writer.writeheader()
        writer.writerows(rows)
    (OUT_DIR / "source_inventory.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    by_status: dict[str, int] = {}
    for row in rows:
        by_status[str(row["status"])] = by_status.get(str(row["status"]), 0) + 1
    report = ["# Share Quant Source Inventory", "", f"Total candidates: {len(rows)}", ""]
    for status, count in sorted(by_status.items()):
        report.append(f"- {status}: {count}")
    report.extend(["", "## Sources", ""])
    for row in rows:
        report.append(
            f"- [{row['idx']:03}] {row['status']} | chars={row['chars']} | pages={row['pages']} | {row['path']}"
        )
    (OUT_DIR / "source_inventory.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\nWrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
