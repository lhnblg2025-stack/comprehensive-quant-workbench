# -*- coding: utf-8 -*-
"""研报工作流：研报 NLP + 新闻/图片 OCR + 产业链/短线龙头融合。

2026-08-20 新增：
- 研报接入：本地目录 + 有道云笔记 MCP（预留 ingest 入口，外部 MCP 拉取后注入）
- 研报 NLP：从文本提取 股票代码/公司/概念/行业/评级/目标价/产业链/龙头/催化
- 图片文字识别：复用 quant_system.ocr_util（tesseract），处理研报截图/新闻图
- 融合：把研报信号映射到 chain_map（产业链）与 leader_follower（短线龙头），
  产出 research_flow_{date}.json，供 pipeline/battle_map 使用。

设计原则：所有外部源（MCP/API）失败必须可见降级，不伪造研报内容。
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

logger = logging.getLogger("research_flow")

CST = timezone(timedelta(hours=8))
ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = ROOT / "data_warehouse" / "research_reports"
IMA_EXPORT_DIR = ROOT / "data_warehouse" / "ima_export" / "media"
IMA_UNIFIED_DIR = ROOT / "data_warehouse" / "ima_export" / "unified"
OUT_DIR = ROOT / "generated"
CONCEPT_MEMBER = ROOT / "data_warehouse" / "classification" / "concept_member.parquet"
CONCEPT_BOARD = ROOT / "data_warehouse" / "classification" / "concept_board.parquet"
# 研报概念 → 成分股映射缓存
_concept_stock_cache: dict[str, dict[str, Any]] | None = None
_concept_names_cache: list[str] | None = None
_stock_name_cache: dict[str, str] | None = None
_chain_prop_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}

# 评级词典（由强到弱）
RATING_STRONG = ("强烈推荐", "买入", "推荐", "增持", "优于大市", "跑赢行业", "超配")
RATING_NEUTRAL = ("中性", "持有", "同步大市", "标配", "审慎推荐")
RATING_WEAK = ("卖出", "减持", "回避", "低于大市", "低配")

# 产业链位置关键词
LAYER_KEYWORDS = {
    "上游": ("上游", "原料", "材料", "矿", "资源", "设备", "零部件", "硅料", "锂盐"),
    "中游": ("中游", "制造", "电池", "组件", "模组", "加工", "代工"),
    "下游": ("下游", "整车", "应用", "终端", "消费", "运营", "渠道"),
}

# 龙头/地位关键词
LEADER_KEYWORDS = ("龙头", "第一", "市占率", "核心受益", "领先", "主导", "稀缺", "唯一")


def _safe_list(x: Any) -> list:
    return x if isinstance(x, list) else []


def _dedup(seq: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in seq:
        s = str(s).strip()
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ────────────────────────────────────────────────────────────
# 研报接入
# ────────────────────────────────────────────────────────────

def _parse_report_date(path: Path) -> str:
    """从研报路径/文件名解析真实日期；解析失败回退今天。

    规则（按优先级）：
    a) 路径含 YYYY年/MM月/YYYYMMDD 或 YYYY/MM/DD 层级 → 取日期；
    b) 文件名/路径含 20260819、2026-08-19、260819 等格式
       （两位年份按 20xx 处理；6 位数字只认 2xxxxx 日期前缀，避开股票代码）；
    c) 均失败 → 回退今天（_resolve_date(None) 原语义，run_today 等不受影响）。
    """
    s = str(path)

    def _fmt(y: int, mo: int, d: int) -> str | None:
        if 2000 <= y <= 2099 and 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{mo:02d}-{d:02d}"
        return None

    # a) 路径层级日期：YYYY/MM/DD 或 YYYY年MM月DD日（三字段分隔齐全，避免误吞 YYYYMMDD）
    m = re.search(r"(?<!\d)(\d{4})[年/](\d{1,2})[月/](\d{1,2})(?!\d)", s)
    if m:
        dt = _fmt(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if dt:
            return dt

    # b) 8 位/分隔日期：20260819、2026-08-19、20260819_xxx
    m = re.search(r"(?<!\d)(20\d{2})[-_]?(\d{2})[-_]?(\d{2})(?!\d)", s)
    if m:
        dt = _fmt(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if dt:
            return dt

    # b2) 6 位日期：260819（两位年份按 20xx；只认 2x 开头，避开 600519 等股票代码）
    m = re.search(r"(?<!\d)(2[0-9])(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)", s)
    if m:
        dt = _fmt(2000 + int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if dt:
            return dt

    return _resolve_date(None)


def _clean_title(path: Path) -> str:
    """Strip image double extensions from OCR text filenames (x.png.txt → x)."""
    name = path.name
    for ext in (".png.txt", ".jpg.txt", ".jpeg.txt", ".webp.txt", ".bmp.txt"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return path.stem


def load_ima_unified_reports() -> list[dict[str, Any]]:
    """读取统一 IMA 全量 manifest/items，作为唯一优先入口。"""
    unified_dir = IMA_UNIFIED_DIR
    # Keep test/deployment overrides of the legacy export root isolated.
    if not str(IMA_EXPORT_DIR).endswith("/media") and not str(IMA_EXPORT_DIR).endswith("\\media"):
        unified_dir = Path(IMA_EXPORT_DIR) / "unified"
    path = unified_dir / "items.jsonl"
    if not path.exists():
        return []
    reports: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("kind") == "folder" or not row.get("title"):
            continue
        title = str(row.get("title") or "")
        content = str(row.get("content") or row.get("text") or row.get("highlight_content") or "")
        reports.append({"source": "ima_unified", "title": title, "path": row.get("path") or row.get("media_id"), "content": content or title, "date": _parse_report_date(Path(str(row.get("path") or title))), "media_id": row.get("media_id"), "evidence_origin": "ima_full_enumeration"})
    return reports


def load_ima_index_reports() -> list[dict[str, Any]]:
    """IMA 导出索引（_items.jsonl / meta.jsonl）→ 研报记录。

    研报库/会议纪要/题材概念产业库/新闻均按标题+路径日期纳入；正文优先
    链接目录内提取文本，没有正文时保留标题索引（NLP 无法解析内容，但
    标题仍进入融合统计）。
    """
    reports: list[dict[str, Any]] = []
    # Derive every IMA index location from the injectable export root. This
    # keeps tests and isolated deployments from silently reading the real home
    # warehouse when their configured export directory is empty.
    export_root = Path(IMA_EXPORT_DIR)
    bases = (
        export_root.parent / "研报库",
        export_root / "会议纪要",
        export_root / "题材概念产业库",
        export_root / "新闻",
    )
    seen: set[tuple[str, str]] = set()
    for base in bases:
        if not base.is_dir():
            continue
        index_files = sorted(base.rglob("_items.jsonl")) + sorted(base.rglob("meta.jsonl"))
        for idx in index_files:
            try:
                lines = idx.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            for line in lines:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                title = str(row.get("title") or row.get("name") or "").strip()
                if not title:
                    continue
                ref = str(row.get("path") or idx)
                key = (title, ref)
                if key in seen:
                    continue
                seen.add(key)
                content = ""
                if idx.parent.is_dir():
                    ref_path = Path(ref)
                    candidates = []
                    if ref_path.suffix.lower() == ".txt":
                        candidates.extend((idx.parent / ref_path.name, ref_path))
                    candidates.extend((
                        idx.parent / f"{ref_path.stem}.txt",
                        idx.parent / f"{title}.txt",
                    ))
                    seen_candidates: set[Path] = set()
                    for txt in candidates:
                        txt = Path(txt)
                        if txt in seen_candidates or not txt.is_file():
                            continue
                        seen_candidates.add(txt)
                        try:
                            text = txt.read_text(encoding="utf-8", errors="ignore")
                        except OSError:
                            continue
                        if len(text.strip()) >= 20:
                            content = text
                            break
                # Preserve index metadata: many IMA records have a company/code
                # field even when the exported text body is unavailable.
                indexed_text = " ".join(str(row.get(key) or "") for key in (
                    "code", "codes", "symbol", "ticker", "company", "company_name", "name", "summary", "content"
                ))
                reports.append({
                    "source": "ima_index",
                    "title": title,
                    "path": ref,
                    "content": content or indexed_text,
                    "index_metadata": row,
                    "date": _parse_report_date(Path(ref)),
                })
    return reports


def load_local_reports() -> list[dict[str, Any]]:
    """读取本地研报目录 + IMA 导出文本（md/txt），失败返回空列表并告警。"""
    reports: list[dict[str, Any]] = []
    for base in (RESEARCH_DIR, IMA_EXPORT_DIR):
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in (".md", ".txt"):
                continue
            # 跳过提取工具的 jsonl
            if p.name == "extracted.jsonl" or p.name == "meta.jsonl":
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
                if len(text.strip()) < 50:
                    continue
                src = "ima_export" if base == IMA_EXPORT_DIR else "local"
                reports.append({
                    "source": src,
                    "title": _clean_title(p),
                    "path": str(p),
                    "content": text,
                    "date": _parse_report_date(p),
                })
            except Exception as e:  # noqa: BLE001
                logger.warning("[research_flow] 读取研报失败 %s: %s", p, e)
    return reports


def ingest_text(title: str, content: str, source: str = "mcp") -> dict[str, Any]:
    """外部（如有道云 MCP）研报注入入口。

    调用方负责从 MCP 拉取笔记/研报后，调用本函数把文本纳入当次分析。
    """
    return {
        "source": source,
        "title": title,
        "path": f"{source}:{title}",
        "content": content,
        "date": _parse_report_date(Path(title)),
    }


def ingest_youdao_notes(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """批量注入有道云笔记 MCP 拉取的研报/笔记。

    notes 每项至少含 title/content（可由有道云 MCP 的 list/get 工具映射而来）。
    返回可供 run_today(reports=...) 直接使用的研报列表；失败/空内容跳过。
    """
    out: list[dict[str, Any]] = []
    for n in notes or []:
        title = str(n.get("title") or n.get("name") or "有道云研报")
        content = str(n.get("content") or n.get("text") or "")
        if len(content.strip()) < 20:
            continue
        out.append(ingest_text(title, content, source="youdao_ynote"))
    return out


# ────────────────────────────────────────────────────────────
# 图片文字识别
# ────────────────────────────────────────────────────────────

def ocr_report_image(path: str | Path) -> dict[str, Any]:
    """对研报/新闻图片做 OCR，返回文本；失败返回 error，不伪造。"""
    try:
        from quant_system.ocr_util import ocr_image_path, is_ocr_error
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"ocr_util 不可用: {e}"}
    text = ocr_image_path(str(path))
    if is_ocr_error(text):
        return {"ok": False, "error": text}
    return {"ok": True, "text": text}


# ────────────────────────────────────────────────────────────
# 研报 NLP
# ────────────────────────────────────────────────────────────

def _load_concept_names() -> list[str]:
    """从 concept_board 读取中文概念名（board_name），供研报文本匹配（带缓存）。"""
    global _concept_names_cache
    if _concept_names_cache is not None:
        return _concept_names_cache
    out: list[str] = []
    if CONCEPT_BOARD.exists():
        try:
            df = pd.read_parquet(CONCEPT_BOARD)
            names = [str(x) for x in pd.unique(df["board_name"].dropna()) if str(x).strip()]
            if names:
                out = _dedup(names)
        except Exception as e:  # noqa: BLE001
            logger.warning("[research_flow] 概念板块表读取失败: %s", e)
    if not out and CONCEPT_MEMBER.exists():
        try:
            df = pd.read_parquet(CONCEPT_MEMBER)
            names = [str(x) for x in pd.unique(df["concept"].dropna()) if str(x).strip()]
            out = _dedup([n for n in names if not n.startswith("BK")])
        except Exception as e:  # noqa: BLE001
            logger.warning("[research_flow] 概念表读取失败: %s", e)
    _concept_names_cache = out
    return out


def _load_stock_names() -> dict[str, str]:
    """返回 {6位代码: 公司名}，用于研报中公司名→代码（带缓存）。"""
    global _stock_name_cache
    if _stock_name_cache is not None:
        return _stock_name_cache
    if not CONCEPT_MEMBER.exists():
        _stock_name_cache = {}
        return {}
    try:
        df = pd.read_parquet(CONCEPT_MEMBER)
        m = {str(r.code).zfill(6): str(r.name) for r in df.itertuples(index=False) if str(r.code).strip()}
        _stock_name_cache = m
        return m
    except Exception as e:  # noqa: BLE001
        logger.warning("[research_flow] 股票名表读取失败: %s", e)
        _stock_name_cache = {}
        return {}


def _is_valid_stock_code(code: str) -> bool:
    """研报文本中的 6 位数字是否为有效 A 股代码（过滤年份/成交额/页面号误报）。

    规则：
    a) 排除 19xxxx/20xxxx 开头（年份误报）；
    b) 股票名表非空时 6 位码必须命中股票表（空表时回退规则 a，不全删）；
    c) 排除纯重复数字（000000/111111）与明显大值（表为空时排除 >699999）。
    """
    if not re.fullmatch(r"\d{6}", code):
        return False
    if code.startswith(("19", "20")):
        return False
    if len(set(code)) == 1:
        return False
    names = _load_stock_names()
    if names:
        return code in names
    return int(code) <= 699999


def extract_research_entities(text: str) -> dict[str, Any]:
    """从研报文本提取结构化信息。

    Returns:
        {codes, companies, concepts, sectors, rating, target_price,
         chain_layers, leader_flags, catalysts}
    """
    if not text:
        return {"codes": [], "companies": [], "concepts": [], "sectors": [],
                "rating": None, "target_price": None, "chain_layers": [],
                "leader_flags": [], "catalysts": []}

    codes = _dedup(c for c in re.findall(r"(?<!\d)(\d{6})(?!\d)", text)
                   if _is_valid_stock_code(c))

    # Company names are retained even without a code in the source. The later
    # evidence layer assigns B-level attribution instead of silently dropping it.
    companies: list[str] = []
    stock_name2code = _load_stock_names()
    for code, name in stock_name2code.items():
        if name and len(name) >= 2 and name in text:
            companies.append(f"{name}({code})")
    # Also keep bare names from the index metadata/title for name-only records.
    for name in re.findall(r"(?:公司|标的|个股|推荐)[:：]?([\u4e00-\u9fff]{2,12})", text):
        if name not in companies and len(name) >= 2:
            companies.append(name)

    # 概念/行业：匹配概念名（BK 代码不带）
    concepts: list[str] = []
    concept_names = _load_concept_names()
    for c in concept_names:
        # 跳过 BK 代码前缀，只用中文概念名
        if c.startswith("BK") or len(c) < 2:
            continue
        if c in text:
            concepts.append(c)

    # 评级
    rating = None
    for kw in RATING_STRONG:
        if kw in text:
            rating = "strong"
            break
    if rating is None:
        for kw in RATING_NEUTRAL:
            if kw in text:
                rating = "neutral"
                break
    if rating is None:
        for kw in RATING_WEAK:
            if kw in text:
                rating = "weak"
                break

    # 目标价
    target_price = None
    m = re.search(r"(?:目标价|看至|合理估值|对应股价)[^\d]{0,4}(\d+(?:\.\d+)?)\s*(?:元|港元|港币)", text)
    if m:
        target_price = float(m.group(1))

    # 产业链位置
    chain_layers = [layer for layer, kws in LAYER_KEYWORDS.items() if any(k in text for k in kws)]

    # 龙头/地位
    leader_flags = [kw for kw in LEADER_KEYWORDS if kw in text]

    # 催化/事件
    catalyst_kws = ("政策", "订单", "中标", "获批", "涨价", "降价", "新品", "量产",
                    "扩产", "签约", "补贴", "规划", "会议", "公告", "业绩预增")
    catalysts = [kw for kw in catalyst_kws if kw in text]

    return {
        "codes": codes,
        "companies": companies,
        "concepts": _dedup(concepts)[:20],
        "sectors": [],
        "rating": rating,
        "target_price": target_price,
        "chain_layers": chain_layers,
        "leader_flags": leader_flags,
        "catalysts": _dedup(catalysts),
    }


# ────────────────────────────────────────────────────────────
# 融合：产业链 + 短线龙头
# ────────────────────────────────────────────────────────────

def _resolve_date(date: str | None) -> str:
    if date:
        return str(date)[:10]
    return datetime.now(CST).strftime("%Y-%m-%d")


_leader_diffusion_cache: dict[str, dict[str, Any]] = {}


def _leader_diffusion(date: str) -> dict[str, Any]:
    """读取当日 leader_follower 产物；当日缺失回退最近产物（标注 actual_date）。

    审计 2026-08-20：非交易日/未运行时当日无产物 → leader_hits 恒空。
    回退最近一个存在的产物，让研报-龙头融合在非交易日照样可用（诚实标注实际日期）。
    """
    if date in _leader_diffusion_cache:
        return _leader_diffusion_cache[date]
    p = OUT_DIR / f"leader_follower_{date}.json"
    actual = date
    if not p.exists():
        # 回退：找最近的 leader_follower_*.json
        cands = sorted(OUT_DIR.glob("leader_follower_*.json"))
        if not cands:
            _leader_diffusion_cache[date] = {"date": date, "actual_date": None,
                                             "active_concepts": 0, "top_signals": [], "diffusion": []}
            return _leader_diffusion_cache[date]
        p = cands[-1]
        actual = p.stem.replace("leader_follower_", "")
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        d["requested_date"] = date
        d["actual_date"] = actual
        _leader_diffusion_cache[date] = d
        return d
    except Exception as e:  # noqa: BLE001
        logger.warning("[research_flow] leader_follower 读取失败: %s", e)
        _leader_diffusion_cache[date] = {"date": date, "actual_date": actual,
                                         "active_concepts": 0, "top_signals": [], "diffusion": []}
        return _leader_diffusion_cache[date]


def _chain_temperatures(date: str, concepts: list[str]) -> list[dict[str, Any]]:
    """根据研报概念找产业链链温度（题材名→成分股→申万一级→链/层→温度）。

    审计 2026-08-20：原实现用 chains_of()（要求概念恰好是申万行业），研报概念
    （AI应用/无人机等概念板块名）永远匹配不到 → chain_hits 恒空。
    改用 chain_map.propagate_from_concept() 桥接题材→产业链。
    """
    try:
        from quant_system.analysis_core import chain_map, industry_graph
        cm = chain_map.load()
        out: list[dict[str, Any]] = []
        # 成分股过多的概念（宽泛题材）推导出的链信号置信度低，标记降级
        stock_counts = {c: len(v) for c, v in resolve_concept_stocks(concepts, top_n=100000).items()}
        for concept in concepts:
            broad = int(stock_counts.get(concept, 0) or 0) > 400
            key = (concept, date)
            props = _chain_prop_cache.get(key)
            if props is None:
                props = cm.propagate_from_concept(concept, date) or []
                _chain_prop_cache[key] = props
            for p_ in props[:3]:
                out.append({
                    "concept": concept,
                    "chain": p_.get("chain"),
                    "layer": p_.get("layer"),
                    "node": p_.get("node"),
                    "sectors": p_.get("sectors", []),
                    "temperature": p_.get("mid_rets"),
                    "signal": p_.get("signal"),
                    "broad_concept": broad,
                    "n_stocks": int(stock_counts.get(concept, 0) or 0),
                })
        # 去重（concept+chain+layer）；宽泛概念排后
        seen = set()
        dedup = []
        for x in sorted(out, key=lambda x: (x.get("broad_concept", False), x.get("concept", ""))):
            k = (x.get("concept"), x.get("chain"), x.get("layer"))
            if k not in seen:
                seen.add(k)
                dedup.append(x)
        return dedup[:10]
    except Exception as e:  # noqa: BLE001
        logger.warning("[research_flow] 产业链融合失败: %s", e)
        return []


_board_code2name: dict[str, str] | None = None


def _load_board_code2name() -> dict[str, str]:
    """BK代码 → 中文概念名（从 concept_board）。"""
    global _board_code2name
    if _board_code2name is not None:
        return _board_code2name
    m: dict[str, str] = {}
    if CONCEPT_BOARD.exists():
        try:
            df = pd.read_parquet(CONCEPT_BOARD)
            for r in df.itertuples(index=False):
                bc = str(r.board_code)
                if bc:
                    m[bc] = str(r.board_name)
        except Exception as e:  # noqa: BLE001
            logger.warning("[research_flow] 板块代码映射读取失败: %s", e)
    _board_code2name = m
    return m


def _fuse_leader(concepts: list[str], leader: dict[str, Any]) -> list[dict[str, Any]]:
    """研报概念与短线龙头扩散信号做交集。

    审计 2026-08-20：leader_follower 的 concept 是 BK/THS 代码（如 BK0835、THS:300082），
    与研报中文概念名（AI应用/PCB）对不上 → 交集恒空。先把 leader 的 BK 代码映射为
    中文概念名，再求交集。
    """
    concepts = set(concepts)
    b2n = _load_board_code2name()
    hits: list[dict[str, Any]] = []
    for sig in _safe_list(leader.get("diffusion")):
        c = sig.get("concept")
        if not c:
            continue
        # 中文名直接匹配；BK代码 → 中文名匹配
        cand = c
        if c.startswith("BK"):
            cand = b2n.get(c, c)
        if cand in concepts:
            hits.append({
                "concept": cand,
                "raw_concept": c,
                "signal": sig.get("signal"),
                "leader": sig.get("leader"),
                "leader_boards": sig.get("leader_boards"),
                "zt_cnt": sig.get("zt_cnt"),
                "diffusion": sig.get("diffusion"),
                "stocks": sig.get("stocks"),
            })
    return hits


def fetch_youdao_reports(max_notes: int = 20, timeout: int = 8) -> list[dict[str, Any]]:
    """通过有道云 MCP 拉取研报笔记并转换为 research_flow 可用的报告列表。

    网络不可达/未配置时返回空列表（不抛异常，不伪造）。
    """
    try:
        import json as _json
        import os as _os
        cfg_path = _os.path.expanduser("${OPENCLAW_CONFIG:-$HOME/.config/openclaw/config.json}")
        if not _os.path.exists(cfg_path):
            return []
        cfg = _json.load(open(cfg_path, encoding="utf-8"))
        srv = cfg["mcp"]["servers"]["youdao-ynote"]
        from quant_system.analysis_core.youdao_mcp import YoudaoMCPClient
        with YoudaoMCPClient(srv["url"], headers=srv.get("headers", {}),
                             timeout=timeout) as client:
            notes = client.fetch_research_notes(max_notes=max_notes)
        return ingest_youdao_notes(notes)
    except Exception as e:  # noqa: BLE001
        logger.warning("[research_flow] 有道云拉取失败，降级为空: %s", e)
        return []


def ingest_news_text(title: str, content: str) -> dict[str, Any]:
    """新闻文本入口：把新闻标题/正文纳入研报工作流（与研报同管线 NLP）。"""
    return ingest_text(title, content, source="news")


def scan_ocr_dir(directory: str | Path, patterns: tuple[str, ...] = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")) -> list[str]:
    """扫描目录下的图片，返回图片路径列表（供 OCR）。"""
    d = Path(directory)
    if not d.exists():
        return []
    out: list[str] = []
    for pat in patterns:
        out.extend(str(p) for p in sorted(d.rglob(pat)) if p.is_file())
    return out


def _concept_stock_map() -> dict[str, dict[str, Any]]:
    """中文概念名 → BK代码 + 成分股(代码/公司)。带模块级缓存。"""
    global _concept_stock_cache
    if _concept_stock_cache is not None:
        return _concept_stock_cache
    cache: dict[str, dict[str, Any]] = {}
    try:
        if CONCEPT_BOARD.exists() and CONCEPT_MEMBER.exists():
            b = pd.read_parquet(CONCEPT_BOARD)
            m = pd.read_parquet(CONCEPT_MEMBER)
            code2name = {}
            for r in b.itertuples(index=False):
                code2name[str(r.board_code)] = str(r.board_name)
            # 按 BK 聚合成分股
            by_bk: dict[str, list[dict]] = {}
            for r in m.itertuples(index=False):
                c = str(r.concept)
                by_bk.setdefault(c, []).append({"code": str(r.code).zfill(6), "name": str(r.name)})
            # 中文名 → 平台
            for bk, name in code2name.items():
                if not name or bk not in by_bk:
                    continue
                # 用概念名前几字做索引（含"AI应用"等），并支持子串匹配
                cache[name] = {"bk": bk, "stocks": by_bk[bk][:500]}
    except Exception as e:  # noqa: BLE001
        logger.warning("[research_flow] 概念成分股映射失败: %s", e)
    _concept_stock_cache = cache
    return cache


def resolve_concept_stocks(concepts: list[str], top_n: int = 30) -> dict[str, list[dict[str, str]]]:
    """把研报概念解析为成分股列表 {概念: [{code,name}...]}。

    用 concept_board 的 board_name 做精确/子串匹配，再从 concept_member 取成分股。
    """
    mp = _concept_stock_map()
    out: dict[str, list[dict[str, str]]] = {}
    for con in concepts:
        if not con:
            continue
        # 精确匹配优先，其次子串匹配
        hit = mp.get(con)
        if hit is None:
            for name, info in mp.items():
                if con and (con in name or name in con):
                    hit = info
                    break
        if hit:
            out[con] = hit["stocks"][:top_n]
    return out


def run_today(date: str | None = None, reports: list[dict[str, Any]] | None = None,
              ocr_images: list[str | Path] | None = None,
              ocr_dir: str | Path | None = None,
              use_youdao: bool = False) -> dict[str, Any]:
    """执行研报工作流并落盘。

    Args:
        date: 分析日期
        reports: 外部注入研报（若有）；缺省读本地目录
        ocr_images: 需要 OCR 的研报/新闻图片路径列表
    """
    date = _resolve_date(date)
    # 审计 2026-08-20：当日产物缓存——未显式传参时复用当天已生成结果，
    # 避免 pipeline 多次调用重复跑 NLP/OCR（性能）
    _default_call = (reports is None and not ocr_images and not ocr_dir and not use_youdao)
    if _default_call:
        cached_path = OUT_DIR / f"research_flow_{date}.json"
        if cached_path.exists():
            try:
                import datetime as _dt
                mtime = _dt.datetime.fromtimestamp(cached_path.stat().st_mtime)
                today = _dt.datetime.now()
                if mtime.date() == today.date():
                    return json.loads(cached_path.read_text(encoding="utf-8"))
            except Exception:
                pass
    if reports is None:
        unified = load_ima_unified_reports()
        reports = load_local_reports() + (unified if unified else load_ima_index_reports())
    if use_youdao:
        # 有道云 MCP 拉取（短超时，失败静默降级为空，不阻塞主链路）
        yd = fetch_youdao_reports()
        if yd:
            reports = reports + yd

    # 图片 OCR 并入报告文本（支持显式图片列表 + 目录扫描）
    ocr_images = list(ocr_images or [])
    if ocr_dir:
        ocr_images += scan_ocr_dir(ocr_dir)
    ocr_results: list[dict[str, Any]] = []
    for img in ocr_images or []:
        r = ocr_report_image(img)
        r["path"] = str(img)
        ocr_results.append(r)
        if r.get("ok"):
            reports.append({
                "source": "ocr", "title": Path(str(img)).stem,
                "path": str(img), "content": r["text"], "date": date,
            })

    parsed: list[dict[str, Any]] = []
    for rep in reports:
        entities = extract_research_entities(str(rep.get("content", "")))
        parsed.append({
            "title": rep.get("title", ""),
            "source": rep.get("source", "unknown"),
            "path": rep.get("path", ""),
            "date": rep.get("date", date),
            **entities,
        })

    # 汇总概念/代码/公司
    all_concepts = _dedup(c for p in parsed for c in p.get("concepts", []))
    all_codes = _dedup(c for p in parsed for c in p.get("codes", []))
    all_companies = _dedup(c for p in parsed for c in p.get("companies", []))

    leader = _leader_diffusion(date)
    chain_hits = _chain_temperatures(date, all_concepts)
    leader_hits = _fuse_leader(all_concepts, leader)
    # 研报概念 → 成分股（供 battle_map/日报直接引用看好的标的）
    concept_stocks = resolve_concept_stocks(all_concepts)

    result = {
        "date": date,
        "n_reports": len(parsed),
        "n_ocr": len(ocr_results),
        "all_concepts": all_concepts,
        "all_codes": all_codes,
        "all_companies": all_companies,
        "concept_stocks": concept_stocks,
        "parsed": parsed,
        "chain_hits": chain_hits,
        "leader_hits": leader_hits,
        "leader_active_concepts": leader.get("active_concepts", 0),
        "_leader_actual_date": leader.get("actual_date"),
        "_leader_requested_date": leader.get("requested_date"),
        "ocr_results": ocr_results,
        "generated_at": datetime.now(CST).isoformat(),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"research_flow_{date}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="研报工作流：NLP+OCR+产业链/龙头融合")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD")
    ap.add_argument("--ocr", action="append", default=[], help="图片路径，可多次")
    ap.add_argument("--youdao", action="store_true", help="尝试从有道云MCP拉取研报")
    ap.add_argument("--ocr-dir", default=None, help="批量OCR目录（新闻/研报图片）")
    args = ap.parse_args()
    res = run_today(args.date, ocr_images=args.ocr, ocr_dir=args.ocr_dir, use_youdao=args.youdao)
    print(json.dumps(res, ensure_ascii=False, indent=2))
