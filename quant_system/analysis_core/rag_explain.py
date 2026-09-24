"""
rag_explain — RAG 解释缓存公共块（子进程检索 + 本地关键词兜底 + 7 天缓存）

behavior_system / emotion_system 重复的 7 个函数收敛于此，参数化公共模块。
只依赖 stdlib + pandas（不 import 任何 analysis_core 业务模块）。
"""

from __future__ import annotations
import logging

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

CACHE_TTL_DAYS = 7          # 静态查询结果缓存有效期（天）
SUBPROCESS_TIMEOUT = 15     # 子进程检索超时（秒）


def load_rag_cache(path: Path) -> list[dict] | None:
    """读 RAG 结果缓存（7 天有效期）；缺失/过期/异常 → None。"""
    try:
        p = Path(path)
        if not p.exists():
            return None
        age = datetime.now().timestamp() - p.stat().st_mtime
        if age > CACHE_TTL_DAYS * 86400:
            return None
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def save_rag_cache(path: Path, items: list[dict]) -> None:
    """写 RAG 结果缓存（失败静默）。"""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).error(f"[rag_explain] 操作失败: {e}", exc_info=True)


def rag_via_subprocess(query: str, k: int, root: Path,
                       env_extra: dict | None = None) -> list[dict] | None:
    """子进程调用 knowledge_rag.search；超时/失败返回 None（主进程零延迟）。"""
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    if env_extra:
        env = {**env, **env_extra}
    code = (
        "import json,sys;"
        f"sys.path.insert(0,{str(root)!r});"
        "from quant_system.analysis_core.knowledge_rag import search;"
        f"print(json.dumps(search({query!r},k={k}),ensure_ascii=False))"
    )
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, timeout=SUBPROCESS_TIMEOUT, env=env)
        if out.returncode == 0 and out.stdout.strip():
            rows = json.loads(out.stdout)
            if isinstance(rows, list) and rows:
                return [fmt_rag_item(r) for r in rows]
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).error(f"[rag_explain] 操作失败: {e}", exc_info=True)
    return None


def keyword_search(query: str, k: int, root: Path) -> list[dict]:
    """本地关键词兜底（与 knowledge_rag 关键词评分一致，读同一 index.parquet）。"""
    index_file = Path(root) / "data_cache" / "knowledge_rag" / "index.parquet"
    if not index_file.exists():
        return [{"error": f"knowledge_rag 索引缺失: {index_file}"}]
    try:
        df = pd.read_parquet(index_file)
        words = [w for w in re.split(r"\s+", query) if len(w) >= 2]
        scored = []
        for i, row in df.iterrows():
            s = sum(1 for w in words if w in row["text"])
            if s:
                scored.append((s, i))
        scored.sort(key=lambda x: -x[0])
        seen: set[str] = set()
        out = []
        for s, i in scored:
            row = df.iloc[i]
            f = str(row["file"])
            if f in seen:
                continue
            seen.add(f)
            out.append({"file": f, "cat": row["cat"],
                        "score": float(s), "text": row["text"][:500]})
            if len(out) >= k:
                break
        return out or [{"error": "关键词无命中，索引存在但内容不覆盖该查询"}]
    except Exception as e:  # noqa: BLE001
        return [{"error": f"{type(e).__name__}: {str(e)[:120]}"}]


def fmt_rag_item(item: dict) -> dict:
    """RAG 命中 → 展示格式（source/cat/score/text 或 {"source","note"}）。"""
    if "error" in item:
        return {"source": "knowledge_rag", "note": item["error"]}
    return {
        "source": item.get("file", "knowledge_rag"),
        "cat": item.get("cat", ""),
        "score": item.get("score"),
        "text": (item.get("text") or "")[:120],
    }


def rag_explain(query: str, k: int, cache_path: Path, root: Path) -> list[dict]:
    """RAG 解释：缓存 → 子进程检索 → 本地关键词兜底（结果写缓存后返回）。"""
    cached = load_rag_cache(cache_path)
    if cached is not None:
        return cached
    items = rag_via_subprocess(query, k, root)
    if not items:
        items = [fmt_rag_item(i) for i in keyword_search(query, k, root)]
    save_rag_cache(cache_path, items)
    return items
