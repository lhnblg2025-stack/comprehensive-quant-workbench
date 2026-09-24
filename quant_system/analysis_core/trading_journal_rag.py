"""
trading_journal_rag — 个人交易日志 RAG（"历史我怎么做"）

把用户每次手动交易/否决系统建议的记录变成"个人交易直觉数据库":
  1. journal_entry(): 追加日志到 data_warehouse/market/trading_journal.parquet
  2. build_index():   复用 knowledge_rag 的索引格式（index.parquet + embeddings.npy，
                      sentence-transformers paraphrase-multilingual-MiniLM，本地缓存、禁止网络），
                      存到 generated/journal_rag/
  3. recall():        把当前市场状态序列化成查询串，检索相似历史日志
  4. report():        日报汇总（总日志数/已验证笔数/胜率/近7日决策）

无日志时所有接口返回明确"等待数据"状态，不崩溃。
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import date as _date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# 禁止网络: 与 knowledge_rag 共用本地缓存的 sentence-transformers 模型
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

JOURNAL_FILE = ROOT / "data_warehouse" / "market" / "trading_journal.parquet"
INDEX_DIR = ROOT / "generated" / "journal_rag"
INDEX_FILE = INDEX_DIR / "index.parquet"
EMB_FILE = INDEX_DIR / "embeddings.npy"

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # 与 knowledge_rag 相同，复用本地缓存
COLS = ["date", "market_state_json", "decision", "result", "note", "created_at"]
CST = timezone(timedelta(hours=8))

_model_cache = None


def _now_str() -> str:
    return datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S")


def _market_state_text(market_state: dict) -> str:
    """把市场状态序列化成查询串: {"情绪": "修复", "机制": "震荡市"} -> "情绪=修复 机制=震荡市"。"""
    if not market_state:
        return ""
    return " ".join(f"{k}={v}" for k, v in market_state.items() if v not in (None, ""))


def _ensure_table() -> pd.DataFrame:
    """日志 parquet 不存在时自动建空表（等待态）。"""
    if not JOURNAL_FILE.exists():
        return pd.DataFrame(columns=COLS)
    try:
        df = pd.read_parquet(JOURNAL_FILE)
    except Exception:
        return pd.DataFrame(columns=COLS)
    return df


def _to_date(x) -> _date:
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, _date):
        return x
    return datetime.strptime(str(x)[:10], "%Y-%m-%d").date()


def _get_model():
    """懒加载向量模型（进程内复用，避免每次重建都重新加载）。"""
    global _model_cache
    if _model_cache is None:
        from sentence_transformers import SentenceTransformer
        _model_cache = SentenceTransformer(MODEL_NAME)
    return _model_cache


def _clean_result(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


def journal_entry(date, market_state: dict, decision: str, result: float | None = None,
                  note: str = "") -> dict:
    """追加一条日志。result=None 表示未验证（持仓中）。入库后自动增量重建索引。"""
    df = _ensure_table()
    row = {
        "date": date.isoformat() if isinstance(date, (_date, datetime)) else str(date)[:10],
        "market_state_json": json.dumps(market_state or {}, ensure_ascii=False),
        "decision": str(decision),
        "result": _clean_result(result),
        "note": str(note or ""),
        "created_at": _now_str(),
    }
    new = pd.DataFrame([row], columns=COLS)
    new["result"] = pd.to_numeric(new["result"], errors="coerce")  # None -> NaN, parquet 可写
    df = new.copy() if df.empty else pd.concat([df, new], ignore_index=True)
    df["result"] = pd.to_numeric(df["result"], errors="coerce")
    JOURNAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(JOURNAL_FILE, index=False)
    # 每次新日志入库后增量重建（日志量小，全量重建可接受）；失败不影响日志落盘
    try:
        build_index(verbose=False)
    except Exception as e:
        print(f"[trading_journal_rag] 索引重建失败(日志已保存): {str(e)[:120]}", file=sys.stderr)
    return {"status": "ok", "row": row, "total": int(len(df))}


def build_index(verbose: bool = True) -> tuple[pd.DataFrame, np.ndarray | None]:
    """把 parquet 日志转成与 knowledge_rag 兼容的向量索引（index.parquet + embeddings.npy）。"""
    df = _ensure_table()
    if df.empty:
        # 无日志 -> 清理索引，进入等待态
        for f in (INDEX_FILE, EMB_FILE):
            if f.exists():
                f.unlink()
        if verbose:
            print("[journal_rag] 无日志, 等待数据")
        return pd.DataFrame(), None
    rows = []
    for i, row in df.iterrows():
        try:
            ms = json.loads(row.get("market_state_json") or "{}")
        except Exception:
            ms = {}
        rows.append({
            "file": "trading_journal",
            "cat": "journal",
            "chunk_id": int(i),
            "text": _market_state_text(ms),
            "date": str(row.get("date", ""))[:10],
            "market_state_json": str(row.get("market_state_json") or ""),
            "decision": str(row.get("decision", "")),
            "result": _clean_result(row.get("result")),
            "note": str(row.get("note", "")),
            "created_at": str(row.get("created_at", "")),
        })
    idx = pd.DataFrame(rows)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    idx.to_parquet(INDEX_FILE, index=False)
    emb = None
    try:
        model = _get_model()
        vecs = model.encode(idx["text"].tolist(), batch_size=32, show_progress_bar=False)
        emb = np.asarray(vecs, dtype=np.float32)
        np.save(EMB_FILE, emb)
        if verbose:
            print(f"[journal_rag] 向量索引: {len(idx)} 条 / {emb.shape[1]} 维")
    except Exception as e:
        if verbose:
            print(f"[journal_rag] 向量不可用(降级关键词): {str(e)[:80]}")
    return idx, emb


def _load_index() -> tuple[pd.DataFrame, np.ndarray | None]:
    if not INDEX_FILE.exists():
        raise FileNotFoundError("先 build_index()（或先写入日志）")
    idx = pd.read_parquet(INDEX_FILE)
    emb = np.load(EMB_FILE) if EMB_FILE.exists() else None
    if emb is not None and emb.shape[0] != len(idx):
        emb = None  # 维度不匹配视为无向量，走关键词兜底
    return idx, emb


def _row_out(idx: pd.DataFrame, i: int, score: float) -> dict:
    r = idx.iloc[i]
    return {
        "date": str(r.get("date", "")),
        "decision": str(r.get("decision", "")),
        "result": _clean_result(r.get("result")),
        "note": str(r.get("note", "")),
        "score": round(float(score), 4),
    }


def recall(market_state: dict, k: int = 3) -> list[dict]:
    """检索相似历史日志，按 score 降序返回 {date, decision, result, note, score}。
    无日志/无索引时返回 []（等待数据状态）。"""
    try:
        idx, emb = _load_index()
    except FileNotFoundError:
        return []
    if idx.empty:
        return []
    query = _market_state_text(market_state)
    results: list[dict] = []
    if emb is not None:
        try:
            model = _get_model()
            qv = model.encode([query])[0]
            sims = emb @ qv / (np.linalg.norm(emb, axis=1) * np.linalg.norm(qv) + 1e-9)
            for i in np.argsort(-sims)[:k]:
                results.append(_row_out(idx, i, float(sims[i])))
        except Exception:
            emb = None
    if not results:
        # 关键词兜底（与 knowledge_rag 同策略）
        words = [w for w in re.split(r"\s+", query) if len(w) >= 2]
        scored = []
        for i, row in idx.iterrows():
            s = sum(1 for w in words if w in str(row.get("text", "")))
            if s:
                scored.append((s, i))
        scored.sort(key=lambda x: -x[0])
        for s, i in scored[:k]:
            results.append(_row_out(idx, i, float(s)))
    results.sort(key=lambda x: -x["score"])
    return results


def report(date=None) -> dict:
    """日报汇总: 总日志数 / 已验证笔数 / 胜率 / 近7日决策。无日志返回等待态。"""
    df = _ensure_table()
    if df.empty:
        return {"status": "waiting", "message": "等待数据: 尚无交易日志",
                "total": 0, "verified": 0, "win_rate": None, "recent7": []}
    ref = _to_date(date) if date is not None else datetime.now(CST).date()
    total = int(len(df))
    results = pd.to_numeric(df["result"], errors="coerce").dropna()
    verified = int(len(results))
    wins = int((results > 0).sum())
    win_rate = round(wins / verified, 4) if verified else None
    df = df.copy()
    df["_d"] = df["date"].astype(str).str[:10]
    cutoff = (ref - timedelta(days=6)).strftime("%Y-%m-%d")
    recent = df[df["_d"] >= cutoff].sort_values("_d", ascending=False)
    recent7 = [{"date": str(r["_d"]), "decision": str(r.get("decision", "")),
                "result": _clean_result(r.get("result"))}
               for _, r in recent.head(7).iterrows()]
    return {"status": "ok", "total": total, "verified": verified,
            "win_rate": win_rate, "recent7": recent7}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="个人交易日志 RAG")
    ap.add_argument("--build", action="store_true", help="重建索引")
    ap.add_argument("--entry", action="store_true", help="追加日志(交互参数见 --date 等)")
    ap.add_argument("--date", default="")
    ap.add_argument("--state", default="", help="市场状态 JSON，如 '{\"情绪\":\"修复\",\"机制\":\"震荡市\"}'")
    ap.add_argument("--decision", default="")
    ap.add_argument("--result", type=float, default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--recall", default="", help="市场状态 JSON，检索相似日志")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.build:
        build_index()
    if args.entry:
        st = json.loads(args.state) if args.state else {}
        r = journal_entry(args.date or datetime.now(CST).date().isoformat(),
                          st, args.decision, args.result, args.note)
        print(json.dumps(r, ensure_ascii=False))
    if args.recall:
        st = json.loads(args.recall) if args.recall else {}
        for r in recall(st, k=args.k):
            print(json.dumps(r, ensure_ascii=False))
    if args.report:
        print(json.dumps(report(), ensure_ascii=False))
    if not (args.build or args.entry or args.recall or args.report):
        ap.print_help()
