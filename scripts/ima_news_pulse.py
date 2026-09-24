#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IMA 知识库研报检索（2026-08-22 —— 用户"IMA的不能加进去吗，全部对齐")

用 IMA search_knowledge 搜当日热点主题 → 返回命中研报/新闻标题，
填充深度复盘的"研报验证/产业驱动"节（对标模板 1.7 热点三源验证）。

小函数:
 1. search_topic(q)     → 单主题搜索(带重试)
 2. hot_research(topics) → 多热点主题批量搜索 → {主题: [研报标题]}
 3. research_md(topics)  → Markdown 段(供 HTML 研报/复盘接入)
零新增依赖, 复用 download_ima_queue.call(node)。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

KB_ID = "-ppxYIh-G5nXXmPVx3yCtNl71cl_K-g1WRDRtqmcQMA="
# 默认热点主题(当日复盘核心方向, 可按需覆盖)
DEFAULT_TOPICS = ["黄金", "半导体", "存储", "农业", "机器人", "算力", "粮食", "新能源", "腾讯", "阿里", "美团", "小米", "KWEB", "恒生科技", "中概互联网"]


def search_topic(query: str, limit: int = 6) -> list[str]:
    """单主题搜索: 远程 IMA API 优先 → 本地 ima_export 研报库标题关键词兜底. 零假数据."""
    hits = _remote_search(query, limit)
    if hits:
        return hits
    return _local_search(query, limit)


def _remote_search(query: str, limit: int = 6) -> list[str]:
    """远程 IMA search_knowledge（含重试）。"""
    try:
        import download_ima_queue as q
        r = q.call("openapi/wiki/v1/search_knowledge",
                   {"query": query, "knowledge_base_id": KB_ID, "cursor": ""})
        infos = (r.get("data") or {}).get("info_list", [])
        titles = [str(i.get("title", "")) for i in infos[:limit] if i.get("title")]
        return titles
    except Exception as e:  # noqa: BLE001
        print(f"[ima_pulse] {query} 远程搜索失败: {str(e)[:50]}")
        return []


def _local_search(query: str, limit: int = 6) -> list[str]:
    """本地兜底：按时间扫描研报、会议纪要、产业概念库和新闻导出索引。"""
    base = ROOT / "data_warehouse" / "ima_export"
    kw = query.lower()
    records: list[dict] = []
    roots = [
        (base / "研报库", "研报库"),
        (base / "media" / "会议纪要", "会议纪要"),
        (base / "media" / "题材概念产业库", "题材概念产业库"),
        (base / "media" / "新闻", "新闻"),
    ]
    for root, kind in roots:
        if not root.is_dir():
            continue
        for f in root.rglob("_items.jsonl"):
            _read_ima_index(f, kind, kw, records)
        meta = root / "meta.jsonl"
        if meta.exists():
            _read_ima_index(meta, kind, kw, records)
    records.sort(key=lambda x: (x.get("date", ""), x.get("title", "")), reverse=True)
    seen: set[str] = set()
    hits: list[str] = []
    for r in records:
        key = r["title"]
        if key in seen:
            continue
        seen.add(key)
        hits.append(f"[{r['kind']} · {r.get('date','未知日期')}] {key}")
        if len(hits) >= limit:
            break
    return hits


def _read_ima_index(path: Path, kind: str, keyword: str, out: list[dict]) -> None:
    """读取一个 IMA 索引，不因单行脏数据中断全库扫描。"""
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except Exception:
                continue
            title = str(row.get("title") or row.get("name") or "").strip()
            if not title or keyword not in title.lower():
                continue
            text = " ".join(str(row.get(k, "")) for k in ("date", "created_at", "updated_at", "path", "folder"))
            match = __import__("re").search(r"(20\\d{2}[-./年]?\\d{1,2}[-./月]?\\d{1,2})", text)
            out.append({"title": title, "kind": kind, "date": match.group(1) if match else "", "path": str(path)})
    except OSError:
        return


def hot_research(topics: list[str] | None = None, per_topic: int = 4) -> dict:
    """多热点批量搜索 → {topic: [title,...]}。"""
    topics = topics or DEFAULT_TOPICS
    out = {}
    for t in topics:
        hits = search_topic(t, limit=per_topic)
        # 去重
        seen = set()
        uniq = []
        for h in hits:
            if h not in seen:
                seen.add(h)
                uniq.append(h)
        out[t] = uniq
    return out


def research_md(topics: list[str] | None = None, per_topic: int = 4) -> str:
    """研报验证段 → Markdown。"""
    L = ["## 📚 IMA知识库研报验证（热点主题）"]
    res = hot_research(topics, per_topic)
    total = sum(len(v) for v in res.values())
    if not total:
        L.append("- IMA库搜索无命中(主题词需匹配库内标题)")
        return "\n".join(L)
    for topic, hits in res.items():
        if not hits:
            continue
        L.append(f"### {topic}")
        for h in hits[:4]:
            L.append(f"- {h[:70]}")
    return "\n".join(L)


def research_data(topics: list[str] | None = None, per_topic: int = 4) -> dict:
    """结构化数据(供 HTML 研报表)。"""
    res = hot_research(topics, per_topic)
    return {"topics": [{"topic": t, "hits": v[:4]} for t, v in res.items() if v]}


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    topics = sys.argv[1:] or None
    print(research_md(topics))