#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""缠论 RAG 知识库（2026-08-22 —— 用户 epub 做成 rag）

从《图解缠论》epub 解析的 50 章全文 → 分块索引(关键词/标题/内容)，
供复盘/决策检索"缠论规则"（如买点判定/走势结构/三剧本）。

用法:
  python3 scripts/chan_rag.py --build    # 建索引(幂等, 读已有jsonl)
  python3 scripts/chan_rag.py --query "中枢 买点"   # 检索
  python3 scripts/chan_rag.py --query "三买 走势" --top 5
零依赖(标准库), 关键词+标题双检索。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KB = ROOT / "data_warehouse" / "chan_theory"
INDEX_FILE = KB / "chan_index.json"


def _chapters() -> list[dict]:
    """读 epub 解析的章节 jsonl。"""
    arts = []
    p = KB / "chan_chapters.jsonl"
    if not p.exists():
        return arts
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            if d.get("text") and len(d["text"]) > 50:
                arts.append({"title": d.get("file", ""), "text": d["text"]})
    return arts


def _chunk(text: str, size: int = 800, overlap: int = 60) -> list[str]:
    """按段落分块(避免切断句子)。"""
    paras = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    chunks = []
    cur = ""
    for p in paras:
        if len(cur) + len(p) > size and cur:
            chunks.append(cur)
            cur = p[-overlap:] if overlap else ""  # 简单重叠
        cur += p + "\n"
    if cur:
        chunks.append(cur)
    return chunks


def build_index(force: bool = False) -> dict:
    """建分块索引(每块: 关键词集合)。"""
    if INDEX_FILE.exists() and not force:
        return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    arts = _chapters()
    blocks = []
    for a in arts:
        for i, c in enumerate(_chunk(a["text"])):
            # 关键词提取: 中文2-4字词(简化取动词名词常见)
            words = set(re.findall(r"[\u4e00-\u9fa5]{2,4}", c))
            # 去高频停词
            stop = {"可以", "一个", "就是", "我们", "进行", "什么", "怎么", "这样", "出现", "走势", "操作", "交易", "股票", "市场", "时候"}
            words = {w for w in words if w not in stop and len(w) >= 2}
            blocks.append({"chunk_id": f"{a['title']}#{i}", "text": c[:800],
                           "words": sorted(words)})
    idx = {"blocks": blocks, "total": len(blocks), "source": "图解缠论_epub_50章"}
    INDEX_FILE.write_text(json.dumps(idx, ensure_ascii=False), encoding="utf-8")
    print(f"RAG 索引已建: {len(blocks)} 块")
    return idx


def query(q: str, top: int = 5) -> list[dict]:
    """关键词检索: query 词集 vs 块词集 重叠打分。"""
    idx = build_index()
    qwords = set(re.findall(r"[\u4e00-\u9fa5]{2,4}", q))
    if not qwords:
        return []
    scored = []
    for b in idx.get("blocks", []):
        overlap = len(qwords & set(b.get("words", [])))
        if overlap:
            scored.append((overlap, b))
    scored.sort(key=lambda x: -x[0])
    return [{"chunk": b["text"][:400], "score": s} for s, b in scored[:top]]


def query_md(q: str, top: int = 5) -> str:
    """检索 → Markdown 段(供研报/复盘)。"""
    hits = query(q, top)
    if not hits:
        return f"- 缠论库无'{q}'命中"
    L = [f"**缠论知识检索: {q}**"]
    for h in hits:
        L.append(f"- {h['chunk'][:150].replace(chr(10),' ')} (score{h['score']})")
    return "\n".join(L)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    if "--build" in sys.argv:
        build_index(force=True)
    elif "--query" in sys.argv:
        i = sys.argv.index("--query")
        q = sys.argv[i + 1]
        top = int(sys.argv[i + 3]) if len(sys.argv) > i + 3 and sys.argv[i + 2] == "--top" else 5
        print(query_md(q, top))
    else:
        b = build_index()
        print(f"缠论库: {b['total']} 块, 用 --query 检索")