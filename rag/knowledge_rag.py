"""
knowledge_rag — 知识层（V11 底层重构 Phase B）

把克隆的 skills 心法库 + 交易规则库 + 设计文档 统一索引，
供分析 Agent 检索引用（本地优先，sentence-transformers 向量 + 关键词兜底）。

索引范围:
  skills/trading-mastery/        26位游资心法(49个md)
  skills/youzi-trading-skill/    23位游资体系
  skills/chen-xiaoqun-skill/     陈小群框架
  skills/UZI-Skill/skills/       中长线深度分析框架
  项目文档/A股市场交易规则硬编码库.md
  项目文档/AI投研分析系统V11_深度设计方案.md

用法:
  python3 -m quant_system.analysis_core.knowledge_rag --build     # 建索引(向量+关键词)
  python3 -m quant_system.analysis_core.knowledge_rag --search "情绪周期 退潮 空仓"
"""

from __future__ import annotations
import logging

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

CACHE_DIR = ROOT / "data_cache" / "knowledge_rag"
INDEX_FILE = CACHE_DIR / "index.parquet"
EMB_FILE = CACHE_DIR / "embeddings.npy"
EMB_META_FILE = CACHE_DIR / "embeddings.meta.json"
FALLBACK_EMB_DIM = 256

# 模型缓存：避免每次检索都重新加载 sentence-transformers（首载数秒，之后命中缓存）。
_MODEL = None
_MODEL_LOADING = False


def _get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _MODEL


def _hashed_embedding(text: str, dim: int = FALLBACK_EMB_DIM) -> np.ndarray:
    """Deterministic local embedding used when the transformer is unavailable.

    This is a retrieval fallback, not a semantic model. Character trigrams
    keep Chinese text usable without downloading model weights and make the
    generated ``embeddings.npy`` reproducible across runs.
    """
    vec = np.zeros(dim, dtype=np.float32)
    normalized = re.sub(r"\s+", " ", str(text).lower())
    grams = [normalized[i:i + 3] for i in range(max(0, len(normalized) - 2))]
    for gram in grams:
        digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vec[bucket] += sign
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec


def warm_model() -> bool:
    """后台预热向量模型；失败返回 False（关键词模式仍可用）。"""
    global _MODEL_LOADING
    import threading
    if _MODEL is not None or _MODEL_LOADING:
        return _MODEL is not None

    def _bg():
        global _MODEL_LOADING
        _MODEL_LOADING = True
        try:
            _get_model()
            logging.getLogger(__name__).info("[knowledge_rag] 向量模型预热完成")
        except Exception as e:  # noqa: BLE001
            logging.getLogger(__name__).warning(
                "[knowledge_rag] 向量模型预热失败(关键词模式): %s", str(e)[:120])
        finally:
            _MODEL_LOADING = False

    threading.Thread(target=_bg, daemon=True, name="rag-model-warm").start()
    return True

# 索引入口（相对 workspace）
INDEX_ROOTS = [
    ("data_warehouse/knowledge/skills", "恢复技能库(挂载资料)"),
    ("data_warehouse/knowledge/skills/generated_books", "书籍生成研究技能"),
    ("data_warehouse/ima_export/media", "恢复IMA图片资料"),
    ("研究报告", "恢复研究报告与IC报告"),
    ("skills/trading-mastery", "游资心法库(26位)"),
    ("skills/youzi-trading-skill", "游资交易体系(23位)"),
    ("skills/chen-xiaoqun-skill", "陈小群框架"),
    ("skills/UZI-Skill/skills", "中长线深度研究框架"),
    ("项目文档/量化交易系统/A股市场交易规则硬编码库.md", "交易规则"),
    ("项目文档/量化交易系统/AI投研分析系统V11_深度设计方案.md", "V11设计"),
    ("项目文档/量化交易系统/AI代码审查_DLAI课程笔记.md", "AI代码审查方法论"),
    # IC 量化书籍知识库（2026-08-14 新增，12本，源自 ~/Desktop/Books_skills/IC）
    ("skills/quant-black-box-2e", "IC书籍:打开量化投资的黑箱(第2版)"),
    ("skills/fundamental-quantamental", "IC书籍:基本面量化投资"),
    ("skills/factor-investing-method-practice", "IC书籍:因子投资方法与实践"),
    ("skills/active-equity-management", "IC书籍:Active Equity Management"),
    ("skills/causal-factor-investing", "IC书籍:Causal Factor Investing"),
    ("skills/quant-equity-investing-techniques", "IC书籍:Quantitative Equity Investing"),
    ("skills/quantitative-momentum", "IC书籍:Quantitative Momentum"),
    ("skills/quantitative-value", "IC书籍:Quantitative Value"),
    ("skills/systematic-trading-carver", "IC书籍:Systematic Trading"),
    ("skills/elements-quant-investing", "IC书籍:The Elements of Quantitative Investing"),
    ("skills/quantamental-revolution", "IC书籍:The Quantamental Revolution"),
    ("skills/factor-based-investing-guide", "IC书籍:Factor-Based Investing Guide"),
    ("skills/llm-factor-mining", "方法论:LLM因子挖掘(LLMQuant借鉴)"),
    ("skills/technical-analysis-murphy", "书籍:金融市场技术分析(墨菲)"),
    ("skills/moving-average-channel-timing", "书籍:均线通道择时"),
    ("skills/macd-moving-average-systems", "书籍:MACD均线系统"),
    ("skills/volume-price-analysis", "书籍:量价分析实操指南"),
    ("skills/turtle-donchian-breakout-system", "书籍:海龟唐奇安突破系统"),
    ("skills/turtle-system-robustness-testing", "书籍:海龟稳健性检验"),
    ("skills/turtle-n-volatility-position-sizing", "书籍:海龟N值仓位"),
    ("skills/turtle-trading-system", "书籍:海龟交易法则"),
    ("skills/wyckoff-accumulation-breakout", "书籍:威科夫吸筹突破"),
    ("skills/wyckoff-distribution-exit", "书籍:威科夫派发离场"),
    ("skills/wyckoff-trading-method", "书籍:威科夫交易法"),
    ("skills/wyckoff-relative-strength-leaders", "书籍:威科夫相对强度龙头"),
    ("skills/livermore-pivotal-point-breakout", "书籍:利弗莫尔关键点突破"),
    ("skills/livermore-leaders-sector-confirmation", "书籍:利弗莫尔领头羊确认"),
    ("skills/livermore-pyramiding-cash-discipline", "书籍:利弗莫尔金字塔加仓"),
    ("skills/appel-market-cycle-segmentation", "书籍:阿佩尔市场周期分段"),
    ("skills/appel-vix-buy-zone", "书籍:阿佩尔VIX买区"),
    ("skills/appel-t-formation-market-cycle", "书籍:阿佩尔T形态"),
    ("skills/fibo-forex-trading", "书籍:斐波那契高级交易法"),
    ("skills/bull-bear-reversal", "书籍:多空转折一手抓"),
    ("skills/factor-investing-framework", "书籍:因子投资框架(石川)"),
    ("skills/factor-investing-alpha-factors", "书籍:因子投资Alpha因子"),
    ("skills/factor-investing-ashare-practice", "书籍:因子投资A股实践"),
    ("skills/factor-investing-multifactor-combination", "书籍:多因子合成"),
    ("skills/factor-neutralization", "书籍:因子中性化"),
    ("skills/multi-factor-stock-selection", "书籍:多因子选股"),
    ("skills/active-alpha-forecast-combination", "书籍:主动Alpha预测组合"),
    ("skills/alpha-information-time-scale", "书籍:Alpha信息时间尺度"),
    ("skills/systematic-trading-framework", "书籍:系统化交易框架(卡佛)"),
    ("skills/systematic-trading-position-sizing", "书籍:系统化仓位管理"),
    ("skills/systematic-trading-portfolio-diversification", "书籍:系统化组合分散"),
    ("skills/systematic-trading-forecasting", "书籍:系统化交易预测"),
    ("skills/ml-quant-finance-intro", "书籍:ML量化金融导论"),
    ("skills/ml-quant-timeseries-deep", "书籍:ML时序深度学习"),
    ("skills/ml-quant-supervised-learning", "书籍:ML监督学习"),
    ("skills/ml-quant-reinforcement-trading", "书籍:RL强化学习交易"),
    ("skills/ai-trading-ml-pipeline", "书籍:AI交易ML流水线"),
    ("skills/portfolio-mpt-optimization", "书籍:MPT组合优化"),
    ("skills/active-portfolio-construction", "书籍:主动组合构建"),
    ("skills/quantitative-risk-management", "书籍:量化风险管理"),
    ("skills/capital-allocation-risk-aversion", "书籍:资本配置风险厌恶"),
    ("skills/backtesting-framework", "书籍:回测框架"),
    ("skills/algo-backtest-bias-audit", "书籍:回测偏差审计"),
    ("skills/risk-management-specialist", "书籍:风险管理专家"),
    ("skills/behavioral-finance-anomalies", "书籍:行为金融异象"),
    ("skills/cognitive-bias-decision-audit", "书籍:认知偏差决策审计"),
    ("skills/prospect-theory-risk-framing", "书籍:前景理论风险框架"),
    ("skills/trader-vic-psychological-discipline", "书籍:交易心理纪律"),
    ("skills/dcf-valuation-mastery", "书籍:DCF估值(Damodaran)"),
    ("skills/macro-four-driver-asset-map", "书籍:宏观四驱资产映射"),
    # 2026-08-15: 高频量化 + NLP 知识库提取(8 个 skill, 参照开源项目学习
    # awesome-high-frequency-trading 与 ai_quant_trade NLP 方向查漏补缺)
    ("skills/algorithmic-hft-cartea", "HFT:算法高频交易(Cartea)"),
    ("skills/hft-backtesting-reality", "HFT:回测现实约束"),
    ("skills/hft-blackbox-durbin", "HFT:黑盒高频交易(Durbin)"),
    ("skills/hft-guide-aldridge", "HFT:高频交易指南(Aldridge)"),
    ("skills/hft-latency-infrastructure", "HFT:低延迟基础设施"),
    ("skills/hft-market-microstructure", "HFT:市场微观结构"),
    ("skills/hft-strategy-taxonomy", "HFT:策略分类学"),
    ("skills/nlp-info-extraction-finance", "NLP:金融信息抽取"),
    # 2026-08-15: books_skills「高频交易,NLP」文件夹补齐(有书未成库的 6 本)
    ("skills/order-flow-trading", "HFT:订单流交易(Valtos)"),
    ("skills/dl-quant-trading-zhang", "书籍:DL量化交易(Zhang&Zohren)"),
    ("skills/coding-capital-algo-trading", "书籍:算法交易艺术(Strauss)"),
    ("skills/algo-trading-mastering-koru", "书籍:算法交易精通(Koru)"),
    ("skills/ml-financial-data-modeling", "书籍:金融数据ML建模(Chen)"),
    ("skills/ml-financial-data-chen", "书籍:金融数据ML建模(旧版书摘)"),
    ("skills/cpp-finance-hanson", "书籍:C++金融编程(Hanson)"),
    ("data_warehouse/ima_export/media", "IMA导出资料(研报/会议纪要)"),
]
MAX_FILE_KB = 300  # 大文件跳过（如 UZI 的 yaml 库）


def _scan_files() -> list[tuple[Path, str, str]]:
    """返回 [(path, category, text)]，md/txt 纯文本，pdf 跳过。"""
    out = []
    for root, cat in INDEX_ROOTS:
        p = ROOT / root
        if not p.exists():
            # Legacy roots are recorded in the recovery catalog; canonical
            # generated skills are the runtime source.
            continue
        if p.is_file():
            candidates = [p]
        else:
            candidates = sorted(p.rglob("*"))
        for f in candidates:
            if f.suffix.lower() not in (".md", ".txt"):
                continue
            if f.stat().st_size > MAX_FILE_KB * 1024:
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                logging.getLogger(__name__).error(f"[knowledge_rag] 操作失败: {e}", exc_info=True)
                continue
            if len(text.strip()) < 50:
                continue
            out.append((f, cat, text))
    return out


def _chunk(text: str, size: int = 400, overlap: int = 80) -> list[str]:
    """按段落切块（保持语义完整）。"""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for para in paras:
        if len(cur) + len(para) > size and cur:
            chunks.append(cur)
            cur = para
        else:
            cur = f"{cur}\n{para}" if cur else para
    if cur:
        chunks.append(cur)
    return chunks


def build_index(verbose: bool = True) -> tuple[pd.DataFrame, np.ndarray | None]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    files = _scan_files()
    rows = []
    for path, cat, text in files:
        for i, ch in enumerate(_chunk(text)):
            rows.append({"file": str(path.relative_to(ROOT)), "cat": cat,
                         "chunk_id": i, "text": ch})
    df = pd.DataFrame(rows)
    df.to_parquet(INDEX_FILE, index=False)

    # 向量索引（sentence-transformers，本地；失败则仅关键词模式）
    emb = None
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        vecs = model.encode(df["text"].tolist(), batch_size=32, show_progress_bar=False)
        emb = np.asarray(vecs, dtype=np.float32)
        np.save(EMB_FILE, emb)
        EMB_META_FILE.write_text(json.dumps({
            "model": "paraphrase-multilingual-MiniLM-L12-v2",
            "kind": "sentence_transformer", "dimension": int(emb.shape[1])
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        if verbose:
            print(f"[rag] 向量索引: {len(df)} 块 / {emb.shape[1]} 维")
    except Exception as e:
        emb = np.vstack([_hashed_embedding(text) for text in df["text"].tolist()]) if not df.empty else np.empty((0, FALLBACK_EMB_DIM), dtype=np.float32)
        np.save(EMB_FILE, emb)
        EMB_META_FILE.write_text(json.dumps({
            "model": "hashed_char_trigram_v1",
            "kind": "deterministic_retrieval_fallback",
            "dimension": FALLBACK_EMB_DIM,
            "reason": str(e)[:200]
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        if verbose:
            print(f"[rag] 语义模型不可用，已生成确定性降级向量: {str(e)[:80]}")
    if verbose:
        print(f"[rag] 文本索引: {len(df)} 块 / {len(files)} 文件")
        print(f"[rag] 分类分布: {df['cat'].value_counts().to_dict()}")
    return df, emb


def _load() -> tuple[pd.DataFrame, np.ndarray | None]:
    if not INDEX_FILE.exists():
        raise FileNotFoundError("先运行 --build")
    df = pd.read_parquet(INDEX_FILE)
    emb = np.load(EMB_FILE) if EMB_FILE.exists() else None
    return df, emb


def search(query: str, k: int = 5, verbose: bool = False, use_vector: bool | None = None) -> list[dict]:
    """检索: 向量相似优先，关键词兜底。

    use_vector=None（默认）: 模型已加载才走向量，否则直接关键词——
    避免 Web 请求首次触发模型加载（数十秒）导致接口超时。
    use_vector=True/False: 强制指定模式。
    """
    df, emb = _load()
    if use_vector is None:
        # Hashed fallback vectors are safe to use without loading a model;
        # transformer vectors are used only after the model is available.
        fallback_ready = EMB_META_FILE.exists() and "hashed_char_trigram" in EMB_META_FILE.read_text(encoding="utf-8")
        use_vector = emb is not None and (_MODEL is not None or fallback_ready)
    if use_vector and emb is not None:
        try:
            if _MODEL is None:
                qv = _hashed_embedding(query, emb.shape[1])
            else:
                qv = _get_model().encode([query])[0]
            sims = emb @ qv / (np.linalg.norm(emb, axis=1) * np.linalg.norm(qv) + 1e-9)
            idxs = np.argsort(-sims)[:k]
            return [{"file": df.iloc[i]["file"], "cat": df.iloc[i]["cat"],
                     "score": round(float(sims[i]), 3), "text": df.iloc[i]["text"][:500]}
                    for i in idxs]
        except Exception as e:
            if verbose:
                print(f"[rag] 向量检索失败,降级关键词: {str(e)[:60]}")
    # 关键词兜底
    words = [w for w in re.split(r"\s+", query) if len(w) >= 2]
    scored = []
    for i, row in df.iterrows():
        s = sum(1 for w in words if w in row["text"])
        if s:
            scored.append((s, i))
    scored.sort(key=lambda x: -x[0])
    return [{"file": df.iloc[i]["file"], "cat": df.iloc[i]["cat"],
             "score": s, "text": df.iloc[i]["text"][:500]}
            for s, i in scored[:k]]


def report_source_status() -> dict:
    """数据源与知识库状态总览（供前端/日报调用）。"""
    from quant_system.analysis_core.data_sources import SOURCE_STATUS
    return {"sources": SOURCE_STATUS,
            "knowledge": {c: int(n) for c, n in pd.read_parquet(INDEX_FILE)["cat"].value_counts().items()}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="知识层 RAG")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--search", type=str, default=None)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    if args.build:
        build_index()
    if args.search:
        for r in search(args.search, k=args.k):
            print(f"[{r['score']}] ({r['cat']}) {r['file']}")
            print(f"   {r['text'][:200].replace(chr(10), ' ')}")
    if not (args.build or args.search):
        ap.print_help()
