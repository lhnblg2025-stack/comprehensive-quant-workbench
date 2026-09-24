#!/usr/bin/env python3
"""Turn recovered books into auditable, searchable research skills.

The generated files are research references. They preserve source metadata and
excerpts, but never claim to be the deleted private strategy implementation.
"""
from __future__ import annotations

import html
import json
import re
import subprocess
import zipfile
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOKS = ROOT / "recovered_knowledge" / "books"
OUT = ROOT / "recovered_knowledge" / "skills" / "generated_books"
MANIFEST = OUT / "manifest.json"
CATALOG = ROOT / "data_warehouse" / "knowledge" / "skills_catalog.json"
EXTS = {".pdf", ".epub", ".mobi", ".azw3"}
EXCLUDE = re.compile(r"(~lock|ExpenseReceipt|eTicket|receipt|\.deb$)", re.I)
TOPICS = re.compile(r"量化|算法|交易|投机|操盘|股票|证券|投资|估值|财务|金融|市场|技术分析|均线|量价|威科夫|斐波那契|海龟|利弗莫尔|因子|组合|风险|portfolio|trading|quant|financial|market|factor|risk|backtest|microstructure|machine learning|time series", re.I)


class Parser(HTMLParser):
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
        value = html.unescape(" ".join(self.parts))
        value = re.sub(r"[ \t\r\f\v]+", " ", value)
        return re.sub(r"\n\s*\n\s*\n+", "\n\n", value).strip()


def slug(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", value).strip("-").lower()
    return value[:100] or "book"


def extract(path: Path) -> tuple[str, str]:
    if path.suffix.lower() == ".pdf":
        txt = path.with_suffix(".bookskill.txt")
        proc = subprocess.run(["pdftotext", "-layout", "-enc", "UTF-8", str(path), str(txt)], capture_output=True, text=True, timeout=300)
        if proc.returncode == 0 and txt.exists():
            return txt.read_text(encoding="utf-8", errors="ignore"), "pdftotext"
        return "", "pdf_text_unavailable"
    if path.suffix.lower() == ".epub":
        parts = []
        try:
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    if name.lower().endswith((".html", ".htm", ".xhtml")):
                        parser = Parser()
                        parser.feed(archive.read(name).decode("utf-8", errors="ignore"))
                        if parser.text():
                            parts.append(parser.text())
        except (OSError, zipfile.BadZipFile):
            return "", "epub_unreadable"
        return "\n\n".join(parts), "epub_html"
    return "", "needs_calibre_conversion"


def excerpts(text: str) -> list[str]:
    chunks = []
    for line in re.split(r"\n+", text):
        line = re.sub(r"\s+", " ", line).strip()
        if 30 <= len(line) <= 320 and not line.isdigit():
            chunks.append(line)
        if len(chunks) >= 8:
            break
    return chunks


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    all_books = [p for p in sorted(BOOKS.rglob("*")) if p.is_file() and p.suffix.lower() in EXTS and not EXCLUDE.search(p.name)]
    candidates = [p for p in all_books if TOPICS.search(p.name)]
    if len(candidates) < 82:
        candidates.extend(p for p in all_books if p not in candidates)
    rows = []
    for index, path in enumerate(candidates[:82], 1):
        text, method = extract(path)
        title = path.stem
        target = OUT / f"{index:03d}-{slug(title)}.md"
        source = str(path.relative_to(ROOT))
        sample = excerpts(text)
        status = "text_extracted" if len(text.strip()) >= 200 else ("metadata_only" if method == "needs_calibre_conversion" else "low_text")
        body = [
            f"# 研究技能：{title}", "", 
            "> 资产性质：从恢复书籍生成的研究参考 skill。不是原始私有策略，也不构成生产交易信号。", "",
            "## 来源", f"- 文件：`{source}`", f"- 提取方式：`{method}`", f"- 状态：`{status}`", f"- 文件大小：`{path.stat().st_size}` 字节", "",
            "## 研究用途", "- 将本书的概念、规则和风险约束作为候选研究假设。", "- 任何策略参数必须经过独立样本、成本、容量和 PIT 数据门禁。", "- 该 skill 只能提供解释和研究方向，不能替代私有策略源码。", "",
            "## 提取证据",
        ]
        body.extend([f"- {line}" for line in sample] or ["- 正文未能直接提取；需要人工或 Calibre/OCR 转换后再索引。"])
        body += ["", "## 运行边界", "- 来源缺失、正文不可读或历史数据不足时，输出 `DATA_BLOCKED`。", "- 不允许把书籍摘要当作回测结果、IC 证据或生产模型权重。", ""]
        target.write_text("\n".join(body), encoding="utf-8")
        rows.append({"skill": str(target.relative_to(ROOT)), "source": source, "method": method, "status": status, "chars": len(text), "size": path.stat().st_size})
    MANIFEST.write_text(json.dumps({"version": 1, "generated": len(rows), "requested": 82, "entries": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    catalog = [{"id": Path(item["skill"]).stem, "source": item["source"], "skill": item["skill"], "kind": "book_generated", "status": item["status"]} for item in rows]
    for path in sorted((ROOT / "recovered_knowledge" / "skills").glob("*.md")):
        catalog.append({"id": path.stem, "source": str(path.relative_to(ROOT)), "skill": str(path.relative_to(ROOT)), "kind": "recovered_skill", "status": "source_skill"})
    for path in sorted((ROOT / "recovered_external" / "fincheck_v5_full_data" / "src" / "fincheck" / "skills").glob("*.md")):
        catalog.append({"id": path.stem, "source": str(path.relative_to(ROOT)), "skill": str(path.relative_to(ROOT)), "kind": "competition_skill", "status": "source_skill"})
    CATALOG.parent.mkdir(parents=True, exist_ok=True)
    CATALOG.write_text(json.dumps({"version": 1, "count": len(catalog), "entries": catalog}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"generated": len(rows), "requested": 82, "catalog": len(catalog), "out": str(OUT.relative_to(ROOT)), "manifest": str(MANIFEST.relative_to(ROOT))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
