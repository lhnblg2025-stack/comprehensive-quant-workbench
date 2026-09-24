"""
alternative_data — 另类数据接入框架（云爬虫真跑版 v2）

架构（spec_alternative_data.md）:
  统一 DataSource 注册表 {name, fetch(), transform(), validate()}，按优先级:
    1. 云爬虫源（真跑，优先）: 巨潮公告热度 cninfo（云 C:\\quant\\crawler_cninfo.py
       每日抓 stock_notice_report("全部", date) → pull_cloud_data.py 拉回
       data_warehouse/cninfo/cninfo_YYYYMMDD.json，本模块只读本地云数据，不本机爬取）
    2. 本地计算源（兜底，真实计算）:
       supply_chain_heat  产业链联动（industry_graph.links 边表 × theme_cycle 活跃概念）
       new_stock_activity 次新涨停占比（kline 首根K线推导上市日期，上市<2年）
       consumption_heat   零售/食品饮料/家电 概念涨停家数占比（theme_cycle）
       retail_focus       涨停股换手/成交额（股吧热榜不可用时的本地代理）
    3. 海外/反爬源: 明确标 unavailable（招聘/Boss 等），必须挂本地代理实现，不留空壳

网络源（按 spec 边界）:
  - 巨潮公告: 云爬 → 本机读 data_warehouse/cninfo/（真数据，不本机爬）
  - 东财股吧人气: 本机 akshare stock_hot_rank_em 可用则调；不可用 → 标 unavailable
    并用本地代理 retail_focus 替代
  - 招聘景气: Boss/智联反爬 → 标 unavailable，用本地代理 new_stock_activity 替代

输出: generated/alt_data_report_{YYYYMMDD}.json
用法:
  python3 -m quant_system.analysis_core.alternative_data --sources
  python3 -m quant_system.analysis_core.alternative_data --report
"""

from __future__ import annotations
import logging

import argparse
import json
import re
import socket
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.config import MARKET_DIR, ZT_HISTORY  # noqa: E402
from quant_system.analysis_core.social_sentiment import BULL_WORDS, BEAR_WORDS  # noqa: E402

try:
    from quant_system.rate_limiter import get_limiter  # noqa: E402
except ImportError:  # rate_limiter 缺失时只降级为不限流，不阻断
    get_limiter = None

CST = timezone(timedelta(hours=8))

CNINFO_DIR = ROOT / "data_warehouse" / "cninfo"
HOT_RANK_DIR = ROOT / "data_warehouse" / "hot_rank"
SOCIAL_DIR = ROOT / "data_warehouse" / "social"
KLINE_DIR = MARKET_DIR.parent / "kline"
THEME_CYCLE = MARKET_DIR / "theme_cycle.parquet"
INDUSTRY_GRAPH = ROOT / "generated" / "industry_graph.parquet"
ZT_EM_DAILY = MARKET_DIR / "zt_pool_em_daily.parquet"
GENERATED_DIR = ROOT / "generated"
LISTING_CACHE = ROOT / "data_warehouse" / "stock" / "listing_dates.parquet"
CAR_TOTAL_CPCA = ROOT / "data_warehouse" / "industry" / "car_total_cpca.parquet"

ACTIVE_TH = 3            # 概念活跃阈值: 当日涨停家数 ≥3（与 industry_graph 对齐）
NEW_STOCK_DAYS = 730     # 上市 <2 年 = 次新
LOOKBACK_DAYS = 20       # 信号基线: 前 N 个交易日中位数
MAJOR_KW = ["强赎", "强制赎回", "要约", "配股", "减持"]
DENSE_TH = 50            # 密集披露: 单股公告数 ≥50

# 消费口径: 零售/食品饮料/家电 概念（避免"消费电子"等泛电子概念混入）
CONS_KW = ["零售", "食品", "饮料", "白酒", "乳业", "啤酒", "家电", "家居",
           "免税", "预制菜", "电商", "新消费", "文娱消费"]


# ── 通用工具 ─────────────────────────────────────────────
def _today() -> str:
    return datetime.now(CST).strftime("%Y%m%d")


def _parse_day(d: str | None) -> datetime:
    """返回 naive datetime（与 theme_cycle/kline 的 naive 日期对齐比较）。"""
    if not d:
        return datetime.now(CST).replace(tzinfo=None)
    s = str(d).strip().replace("-", "")
    if len(s) == 8 and s.isdigit():
        return datetime.strptime(s, "%Y%m%d")
    return datetime.now(CST).replace(tzinfo=None)


def _to_date(x) -> pd.Timestamp | None:
    try:
        return pd.to_datetime(x, errors="coerce")
    except Exception:  # noqa: BLE001
        return None


def _signal(cur: float | None, base: float | None,
            up_th: float = 1.25, down_th: float = 0.8) -> str:
    """相对基线: 升温/降温/中性。"""
    if cur is None or base is None or base <= 0:
        return "中性"
    if cur >= base * up_th:
        return "升温"
    if cur <= base * down_th:
        return "降温"
    return "中性"


def _baseline(series: list[float | None], n: int = LOOKBACK_DAYS) -> float | None:
    vals = [v for v in series if v is not None and np.isfinite(v)]
    if not vals:
        return None
    return float(np.median(vals[-n:]))


def _trade_dates(end: pd.Timestamp, n: int = 40) -> list[pd.Timestamp]:
    """theme_cycle 中 ≤end 的最近 n 个交易日。"""
    tc = _theme_cycle()
    dates = sorted(d for d in tc["date"].unique() if d <= end)
    return dates[-n:]


# ── 数据加载（lru_cache 防重复读盘）────────────────────────
@lru_cache(maxsize=1)
def _theme_cycle() -> pd.DataFrame:
    if not THEME_CYCLE.exists():
        return pd.DataFrame(columns=["date", "concept", "zt_cnt", "board_name"])
    df = pd.read_parquet(THEME_CYCLE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["zt_cnt"] = pd.to_numeric(df["zt_cnt"], errors="coerce").fillna(0)
    return df


@lru_cache(maxsize=1)
def _industry_edges() -> pd.DataFrame:
    if not INDUSTRY_GRAPH.exists():
        return pd.DataFrame(columns=["src", "dst", "lift"])
    return pd.read_parquet(INDUSTRY_GRAPH)


@lru_cache(maxsize=1)
def _zt_history() -> pd.DataFrame:
    if not ZT_HISTORY.exists():
        return pd.DataFrame(columns=["date", "code", "is_zt"])
    df = pd.read_parquet(ZT_HISTORY)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["code"] = df["code"].astype(str).str.zfill(6)
    return df[df["is_zt"]].copy()


@lru_cache(maxsize=1)
def _zt_em_daily() -> pd.DataFrame:
    if not ZT_EM_DAILY.exists():
        return pd.DataFrame(columns=["date", "code", "turnover", "amount", "is_zt"])
    df = pd.read_parquet(ZT_EM_DAILY)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    return df[df["is_zt"]].copy()


def _load_listing_dates() -> pd.DataFrame:
    """code → 上市日期（kline 首根K线，全库唯一可用口径；缓存 parquet 复用）。"""
    if LISTING_CACHE.exists():
        df = pd.read_parquet(LISTING_CACHE)
        if len(df) > 1000:
            return df
    files = sorted(KLINE_DIR.glob("*.parquet")) if KLINE_DIR.exists() else []
    rows = []
    for p in files:
        try:
            d = pd.read_parquet(p, columns=["date"])
            rows.append((p.stem, d["date"].min()))
        except Exception as e:  # noqa: BLE001
            logging.getLogger(__name__).error(f"[alternative_data] 操作失败: {e}", exc_info=True)
            continue
    df = pd.DataFrame(rows, columns=["code", "list_date"])
    df["list_date"] = pd.to_datetime(df["list_date"], errors="coerce")
    try:
        LISTING_CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(LISTING_CACHE)
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).error(f"[alternative_data] 操作失败: {e}", exc_info=True)
    return df


def _anchor_date(end: pd.Timestamp) -> pd.Timestamp | None:
    """≤end 的最新有数据交易日（theme_cycle 对齐）。"""
    dates = _trade_dates(end, n=1)
    return dates[-1] if dates else None


# ── 云爬虫源: 巨潮公告热度 ───────────────────────────────
def _latest_cninfo_file(end: pd.Timestamp | None = None) -> Path | None:
    if not CNINFO_DIR.exists():
        return None
    files = sorted(CNINFO_DIR.glob("cninfo_*.json"))
    if not files:
        return None
    if end is None:
        return files[-1]
    end_s = end.strftime("%Y%m%d")
    cands = [f for f in files if f.stem.replace("cninfo_", "") <= end_s]
    return cands[-1] if cands else None


def _stem_date(stem: str) -> str | None:
    """从文件名 stem 提取 YYYYMMDD（云快照命名约定），无则 None。"""
    m = re.search(r"(\d{8})", stem)
    return m.group(1) if m else None


def _latest_parquet(data_dir: Path, pattern: str, end: pd.Timestamp | None = None) -> Path | None:
    """按文件名日期排序取最新 parquet（文件名须含 YYYYMMDD，字典序=日期序）。

    end 非空时只取 ≤end 的快照，避免历史查询读到未来快照（前视偏差防护）。
    """
    if not data_dir.exists():
        return None
    files = sorted(f for f in data_dir.glob(pattern) if _stem_date(f.stem) is not None)
    if not files:
        return None
    if end is None:
        return files[-1]
    end_s = end.strftime("%Y%m%d")
    cands = [f for f in files if _stem_date(f.stem) <= end_s]
    return cands[-1] if cands else None


def _latest_json(data_dir: Path, pattern: str, end: pd.Timestamp | None = None) -> Path | None:
    """按文件名日期排序取最新 json（文件名须含 YYYYMMDD，字典序=日期序）。"""
    if not data_dir.exists():
        return None
    files = sorted(f for f in data_dir.glob(pattern) if _stem_date(f.stem) is not None)
    if not files:
        return None
    if end is None:
        return files[-1]
    end_s = end.strftime("%Y%m%d")
    cands = [f for f in files if _stem_date(f.stem) <= end_s]
    return cands[-1] if cands else None


def _dict_ratio(texts: list[str]) -> tuple[int, int, float | None]:
    """用 social_sentiment 的 BULL/BEAR 词典对关键词列表打多空分。"""
    joined = " ".join(str(t) for t in texts)
    bull = sum(1 for w in BULL_WORDS if w in joined)
    bear = sum(1 for w in BEAR_WORDS if w in joined)
    ratio = round(bull / bear, 3) if bear else None
    return bull, bear, ratio


def _normalize_code(raw) -> str | None:
    """'SZ301308'/'SH600664' → '301308'（6 位纯数字），无法归一化返回 None。"""
    if raw is None:
        return None
    m = re.search(r"(\d{6})", str(raw))
    return m.group(1) if m else None


def _hot_rank_kline_metrics(df: pd.DataFrame, as_of: str) -> tuple[float | None, float | None, str]:
    """本地 kline 补齐热榜当日涨跌幅/成交额集中度。

    云接口 stockrank/getAllCurrentList 只返回 sc/rk/rc/hisRc 四字段，无涨跌幅；
    改用 data_warehouse/kline/{code}.parquet 取快照日期(≤as_of)最近交易日的
    pct_chg/amount。目录缺失/文件坏/当日无数据 → 指标 None + note 说明，不伪造。
    """
    if "code" not in df.columns:
        return None, None, "；快照无 code 列→无法用 kline 补齐涨跌幅"
    if not KLINE_DIR.exists():
        return None, None, "；data_warehouse/kline 不存在→avg_pct/top10_concentration=None"
    as_of_ts = pd.to_datetime(as_of, format="%Y%m%d", errors="coerce")
    pcts: list[float] = []
    amts: list[float] = []
    matched = missing = bad = 0
    for raw in df["code"].tolist():
        code = _normalize_code(raw)
        if code is None:
            missing += 1
            continue
        p = KLINE_DIR / f"{code}.parquet"
        if not p.exists():
            missing += 1
            continue
        try:
            k = pd.read_parquet(p, columns=["date", "pct_chg", "amount"])
            k["date"] = pd.to_datetime(k["date"], errors="coerce")
            k = k.dropna(subset=["date"])
        except Exception:  # noqa: BLE001
            bad += 1
            continue
        if pd.notna(as_of_ts):
            k = k[k["date"] <= as_of_ts]
        if k.empty:
            missing += 1
            continue
        row = k.loc[k["date"].idxmax()]
        matched += 1
        pct = pd.to_numeric(row["pct_chg"], errors="coerce")
        if pd.notna(pct):
            pcts.append(float(pct))
        amt = pd.to_numeric(row["amount"], errors="coerce")
        if pd.notna(amt) and amt > 0:
            amts.append(float(amt))
    avg_pct = round(float(np.mean(pcts)), 2) if pcts else None
    top10_concentration = None
    if amts:
        total = float(sum(amts))
        if total > 0:
            top10_concentration = round(float(sum(amts[:10]) / total), 4)
    note = f"；kline 补齐: 匹配 {matched}/{len(df)}"
    if missing:
        note += f"，未匹配 {missing}"
    if bad:
        note += f"，文件读取失败 {bad}"
    if avg_pct is None:
        note += "→avg_pct=None（当日无涨跌幅数据）"
    if top10_concentration is None:
        note += "→top10_concentration=None（当日无成交额数据）"
    return avg_pct, top10_concentration, note


def _hot_rank_summary(end: pd.Timestamp) -> dict:
    """东财热榜云快照（腾讯云 18:35）→ top100 摘要。

    云接口只返回 code/rank/rank_change/hist_rank（无涨跌幅/成交额列），
    avg_pct / top10_concentration 由本地 kline 补齐（≤快照日期最近交易日），
    匹配不到/无数据如实 None，不伪造。
    """
    path = _latest_parquet(HOT_RANK_DIR, "hot_rank_*.parquet", end)
    if path is None:
        return {"status": "unavailable", "as_of": None, "n": 0, "avg_pct": None,
                "top10_concentration": None,
                "reason": "data_warehouse/hot_rank/ 无 hot_rank_*.parquet（云爬虫未拉回）"}
    try:
        df = pd.read_parquet(path)
    except Exception as e:  # noqa: BLE001
        return {"status": "unavailable", "as_of": None, "n": 0, "avg_pct": None,
                "top10_concentration": None,
                "reason": f"{path.name} 读取失败: {type(e).__name__}: {e}"}
    if df is None or len(df) == 0:
        return {"status": "unavailable", "as_of": None, "n": 0, "avg_pct": None,
                "top10_concentration": None, "reason": f"{path.name} 为空"}
    as_of = _stem_date(path.stem) or str(path.stem).replace("hot_rank_", "")
    n = int(len(df))
    reason = "东财热榜top100云快照"
    avg_pct: float | None = None
    pct_col = next((c for c in ("pct_change", "涨跌幅") if c in df.columns), None)
    if pct_col is not None:
        vals = pd.to_numeric(df[pct_col].astype(str).str.replace("%", "", regex=False),
                             errors="coerce").dropna()
        if len(vals):
            avg_pct = round(float(vals.mean()), 2)
            reason += f"；涨跌幅来自快照列({pct_col})"
    top10_concentration: float | None = None
    amt_col = next((c for c in ("amount", "成交额") if c in df.columns), None)
    if amt_col is not None:
        amt = pd.to_numeric(df[amt_col], errors="coerce").dropna()
        if len(amt) and float(amt.sum()) > 0:
            top10_concentration = round(float(amt.head(10).sum() / amt.sum()), 4)
            reason += f"；成交额来自快照列({amt_col})"
    # 快照无涨跌幅/成交额列（云接口限制）→ 本地 kline 补齐
    if avg_pct is None or top10_concentration is None:
        k_avg, k_top10, note = _hot_rank_kline_metrics(df, as_of)
        if avg_pct is None:
            avg_pct = k_avg
        if top10_concentration is None:
            top10_concentration = k_top10
        reason += note
    return {"status": "available", "as_of": as_of, "n": n,
            "avg_pct": avg_pct, "top10_concentration": top10_concentration,
            "reason": reason}


def _social_summary(end: pd.Timestamp) -> dict:
    """社交快照（Vultr 18:40）→ weibo 舆情 + baidu 热搜摘要。"""
    weibo_path = _latest_parquet(SOCIAL_DIR, "weibo_*.parquet", end)
    weibo: dict = {"status": "unavailable", "as_of": None, "n": 0,
                   "bull_cnt": 0, "bear_cnt": 0, "bull_bear_ratio": None,
                   "reason": "data_warehouse/social/ 无 weibo_*.parquet（Vultr 未拉回）"}
    if weibo_path is not None:
        try:
            wdf = pd.read_parquet(weibo_path)
            if wdf is not None and len(wdf):
                as_of = _stem_date(weibo_path.stem) or str(weibo_path.stem).replace("weibo_", "")
                names = wdf["name"].dropna().astype(str).tolist() if "name" in wdf else []
                bull, bear, ratio = _dict_ratio(names)
                weibo = {"status": "available", "as_of": as_of, "n": int(len(wdf)),
                         "bull_cnt": bull, "bear_cnt": bear, "bull_bear_ratio": ratio,
                         "top": names[:10],
                         "reason": "微博个股舆情涨跌率快照（BULL/BEAR 词典多空比）"}
            else:
                weibo = {**weibo, "reason": f"{weibo_path.name} 为空"}
        except Exception as e:  # noqa: BLE001
            weibo = {"status": "unavailable", "as_of": None, "n": 0,
                     "bull_cnt": 0, "bear_cnt": 0, "bull_bear_ratio": None,
                     "reason": f"{weibo_path.name} 读取失败: {type(e).__name__}: {e}"}

    baidu: dict = {"status": "unavailable", "as_of": None, "n": 0, "top": [],
                   "bull_cnt": 0, "bear_cnt": 0, "bull_bear_ratio": None,
                   "reason": "data_warehouse/social/ 无 baidu_hot_*.json|parquet"}
    baidu_path = _latest_json(SOCIAL_DIR, "baidu_hot_*.json", end)
    if baidu_path is None:
        baidu_path = _latest_parquet(SOCIAL_DIR, "baidu_hot_*.parquet", end)
    if baidu_path is not None:
        try:
            if baidu_path.suffix == ".json":
                with baidu_path.open(encoding="utf-8") as f:
                    jd = json.load(f)
                items = jd.get("items") if isinstance(jd, dict) else jd
                as_of = _stem_date(baidu_path.stem) or str(baidu_path.stem).replace("baidu_hot_", "")
                if isinstance(items, list):
                    names = [str(x.get("名称/代码", "")) for x in items if isinstance(x, dict)]
                else:
                    names = []
                if not names:
                    baidu = {"status": "unavailable", "as_of": None, "n": 0, "top": [],
                             "bull_cnt": 0, "bear_cnt": 0, "bull_bear_ratio": None,
                             "reason": f"{baidu_path.name} 无有效名称项"}
                else:
                    bull, bear, ratio = _dict_ratio(names)
                    baidu = {"status": "available", "as_of": as_of, "n": len(names),
                             "top": names[:10], "bull_cnt": bull, "bear_cnt": bear,
                             "bull_bear_ratio": ratio,
                             "reason": "百度股市通热搜快照 Top10 关键词"}
            else:
                bdf = pd.read_parquet(baidu_path)
                if bdf is not None and len(bdf):
                    as_of = _stem_date(baidu_path.stem) or str(baidu_path.stem).replace("baidu_hot_", "")
                    name_col = next((c for c in ("名称/代码", "name") if c in bdf.columns), None)
                    names = bdf[name_col].dropna().astype(str).tolist() if name_col else []
                    if not names:
                        baidu = {"status": "unavailable", "as_of": None, "n": 0, "top": [],
                                 "bull_cnt": 0, "bear_cnt": 0, "bull_bear_ratio": None,
                                 "reason": f"{baidu_path.name} 无名称列或名称为空"}
                    else:
                        bull, bear, ratio = _dict_ratio(names)
                        baidu = {"status": "available", "as_of": as_of, "n": len(names),
                                 "top": names[:10], "bull_cnt": bull, "bear_cnt": bear,
                                 "bull_bear_ratio": ratio,
                                 "reason": "百度股市通热搜快照 Top10 关键词（parquet）"}
                else:
                    baidu = {**baidu, "reason": f"{baidu_path.name} 为空"}
        except Exception as e:  # noqa: BLE001
            baidu = {"status": "unavailable", "as_of": None, "n": 0, "top": [],
                     "bull_cnt": 0, "bear_cnt": 0, "bull_bear_ratio": None,
                     "reason": f"{baidu_path.name} 读取失败: {type(e).__name__}: {e}"}

    # 本机实时社交基座（股吧/B站/百度）是当前有效兜底，不再被停更微博单独卡住。
    local_items = []
    for pattern, label in (("guba_*.parquet", "股吧"), ("bilibili_*.parquet", "B站"), ("baidu_*.parquet", "百度"), ("rank_*.parquet", "热榜")):
        p = _latest_parquet(SOCIAL_DIR, pattern, end)
        if p is None:
            continue
        try:
            frame = pd.read_parquet(p)
            if len(frame):
                local_items.append({"label": label, "as_of": _stem_date(p.stem), "n": int(len(frame),), "file": p.name})
        except Exception:
            continue
    local_items = [x for x in local_items if x.get("as_of")]
    local_latest = max((x["as_of"] for x in local_items), default=None)
    local = {"status": "available" if local_items else "unavailable", "as_of": local_latest,
             "items": local_items, "reason": "本机股吧/B站/百度/热榜滚动快照" if local_items else "本机社交快照暂无"}
    candidates = [x.get("as_of") for x in (weibo, baidu, local) if x.get("status") == "available" and x.get("as_of")]
    ok = bool(candidates)
    as_of = max(candidates) if candidates else None
    return {"status": "available" if ok else "unavailable",
            "as_of": as_of, "weibo": weibo, "baidu": baidu, "local": local,
            "reason": ("社交快照: " + weibo.get("reason", "") + " | " + baidu.get("reason", "") + " | " + local.get("reason", ""))}

def cloud_sources(date: str | None = None) -> dict:
    """读取 data_warehouse/cninfo/ 最新云爬数据 → 公告扰动指标。

    2026-08-11 扩展: 返回 dict 追加 hot_rank（东财热榜云快照）与 social
    （微博+百度社交快照），现有调用方读 status/reason/signal 不受影响。

    边界: 云数据缺失 → 返回 unavailable + 原因，不崩溃。
    """
    end = _parse_day(date)
    hot_rank = _hot_rank_summary(end)
    social = _social_summary(end)
    path = _latest_cninfo_file(end)
    if path is None:
        return {
            "status": "unavailable", "name": "cninfo",
            "reason": f"data_warehouse/cninfo/ 无 ≤{end.strftime('%Y%m%d')} 的 "
                      f"cninfo_*.json（云爬虫未拉回，先跑 scripts/pull_cloud_data.py）",
            "hot_rank": hot_rank, "social": social,
        }
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001
        return {"status": "unavailable", "name": "cninfo",
                "reason": f"{path.name} 解析失败: {type(e).__name__}: {e}",
                "hot_rank": hot_rank, "social": social}
    if not isinstance(data, list) or not data:
        return {"status": "unavailable", "name": "cninfo",
                "reason": f"{path.name} 为空列表",
                "hot_rank": hot_rank, "social": social}
    if not isinstance(data[0], dict):
        return {"status": "unavailable", "name": "cninfo",
                "reason": f"{path.name} 结构异常（非 dict 列表）",
                "hot_rank": hot_rank, "social": social}

    total = len(data)
    major_cnt: Counter = Counter()
    for d in data:
        text = f"{d.get('公告标题', '')} {d.get('公告类型', '')}"
        for kw in MAJOR_KW:
            if kw in text:
                major_cnt[kw] += 1
                break
    major_total = sum(major_cnt.values())

    by_code: Counter = Counter()
    for d in data:
        by_code[str(d.get("代码", "")).zfill(6)] += 1
    dense = {c: n for c, n in by_code.items() if n >= DENSE_TH}
    dense_top = sorted(dense.items(), key=lambda x: -x[1])[:10]
    name_map = {}
    for d in data:
        c = str(d.get("代码", "")).zfill(6)
        name_map.setdefault(c, str(d.get("名称", "")))

    if major_total >= 800 or len(dense) >= 200:
        level = "高扰动"
    elif major_total >= 300 or len(dense) >= 80:
        level = "中扰动"
    else:
        level = "低扰动"

    return {
        "status": "available",
        "name": "cninfo",
        "file": path.name,
        "date": str(path.stem.replace("cninfo_", "")),
        "total_announcements": total,
        "major_event_cnt": major_total,
        "major_events": {k: major_cnt.get(k, 0) for k in MAJOR_KW},
        "dense_stock_cnt": len(dense),
        "dense_top10": [{"code": c, "name": name_map.get(c, ""), "count": n}
                        for c, n in dense_top],
        "signal": level,
        "note": "公告总数/重大事件(强赎/要约/配股/减持)/密集披露股 → 市场公告扰动",
        "hot_rank": hot_rank,
        "social": social,
    }


# ── 网络源: 东财股吧人气（本机 akshare，不可用→本地代理）────────
@lru_cache(maxsize=1)
def _guba_hot_rank_akshare() -> dict:
    """尝试 akshare 东财人气榜。失败返回原因（调用方标 unavailable）。"""
    import akshare as ak
    socket.setdefaulttimeout(8)
    try:
        if get_limiter is not None:
            get_limiter("eastmoney.com").wait()  # akshare 内部直连东财，网络入口节流
        for fn in ("stock_hot_rank_em", "stock_hot_rank_latest_em"):
            f = getattr(ak, fn, None)
            if f is None:
                continue
            t0 = time.time()
            try:
                df = f()
                if df is not None and len(df) > 0:
                    return {"ok": True, "api": fn, "rows": len(df), "elapsed_s": round(time.time() - t0, 1)}
            except Exception as e:  # noqa: BLE001
                return {"ok": False, "api": fn, "reason": f"{type(e).__name__}: {str(e)[:120]}",
                        "elapsed_s": round(time.time() - t0, 1)}
        return {"ok": False, "reason": "akshare 无可用热榜接口"}
    finally:
        socket.setdefaulttimeout(None)


def _guba_source(date: str | None = None) -> dict:
    """东财股吧人气: 可用→真实热榜；不可用→unavailable + 本地代理 retail_focus。"""
    probe = _guba_hot_rank_akshare()
    if probe.get("ok"):
        return {"status": "available", "name": "guba_hot_rank", "detail": probe,
                "note": "东财人气榜(akshare 本机可用)"}
    return {
        "status": "unavailable",
        "name": "guba_hot_rank",
        "reason": probe.get("reason", "接口不可用"),
        "fallback": "retail_focus（涨停股换手/成交额，本地代理）",
        "note": "散户关注度 → 用本地代理替代，不硬编码死",
    }


def _recruitment_source() -> dict:
    """招聘景气: Boss/智联反爬 → 明确 unavailable，本地代理 new_stock_activity。"""
    return {
        "status": "unavailable",
        "name": "recruitment",
        "reason": "Boss直聘/智联反爬且无云爬通道，招聘网络源放弃",
        "fallback": "new_stock_activity（次新涨停占比，本地代理）",
        "note": "次新活跃度代理招聘景气（按 spec 换本地实现，非空壳）",
    }


# ── 本地代理指标（真实计算）──────────────────────────────
def _supply_chain_series(end: pd.Timestamp, n: int = 40) -> list[tuple[pd.Timestamp, float, int]]:
    tc = _theme_cycle()
    edges = _industry_edges()
    if tc.empty or edges.empty:
        return []
    out = []
    for d in _trade_dates(end, n=n):
        dtc = tc[tc["date"] == d]
        active = set(dtc[dtc["zt_cnt"] >= ACTIVE_TH]["concept"])
        lift_sum = 0.0
        cnt = 0
        for _, r in edges.iterrows():
            if r["src"] in active and r["dst"] in active:
                lift_sum += float(r["lift"])
                cnt += 1
        out.append((d, lift_sum, cnt))
    return out


def _new_stock_series(end: pd.Timestamp, n: int = 40) -> list[tuple[pd.Timestamp, float | None, int, int]]:
    zt = _zt_history()
    ld = _load_listing_dates()
    if zt.empty:
        return []
    ld_map = ld.set_index("code")["list_date"].to_dict()
    out = []
    for d in _trade_dates(end, n=n):
        dzt = zt[zt["date"] == d]
        total = len(dzt)
        if total == 0:
            out.append((d, None, 0, 0))
            continue
        days = (d - dzt["code"].map(ld_map)).dt.days.fillna(99999)
        new_cnt = int((days <= NEW_STOCK_DAYS).sum())
        out.append((d, new_cnt / total if total else None, new_cnt, total))
    return out


def _consumption_series(end: pd.Timestamp, n: int = 40) -> list[tuple[pd.Timestamp, float | None, int, int]]:
    tc = _theme_cycle()
    if tc.empty:
        return []

    def _match(name) -> bool:
        s = str(name)
        if "电子" in s:
            return False
        return any(k in s for k in CONS_KW)

    out = []
    for d in _trade_dates(end, n=n):
        dtc = tc[tc["date"] == d]
        if dtc.empty:
            out.append((d, None, 0, 0))
            continue
        cons = dtc[dtc["board_name"].map(_match)]
        cons_zt = int(cons["zt_cnt"].sum())
        all_zt = int(dtc["zt_cnt"].sum())
        out.append((d, cons_zt / all_zt if all_zt else None, cons_zt, all_zt))
    return out


def _retail_focus_series(end: pd.Timestamp, n: int = 40) -> list[tuple[pd.Timestamp, float | None, float | None]]:
    """涨停股平均换手率 + 涨停成交额合计（股吧热榜本地代理）。"""
    em = _zt_em_daily()
    if em.empty:
        return []
    dates = sorted(d for d in em["date"].dropna().unique() if d <= end)[-n:]
    out = []
    for d in dates:
        dzt = em[em["date"] == d]
        to = float(dzt["turnover"].mean()) if dzt["turnover"].notna().any() else None
        amt = float(dzt["amount"].sum()) if dzt["amount"].notna().any() else None
        out.append((d, to, amt))
    return out


def _car_boom() -> dict:
    """乘联会批发总量景气（本地代理）: 最近两期环比。

    宽表 car_total_cpca.parquet（月份 × 年份）按时序展平，取最近两期批发总量算环比:
    升 >5% → 升温；降 >5% → 降温；否则中性（复用 _signal 阈值）。
    文件缺失/列不符/期数不足 → status=unavailable + reason，不抛异常。
    """
    try:
        if not CAR_TOTAL_CPCA.exists():
            return {"status": "unavailable", "signal": "中性", "detail": "",
                    "unit": "万辆", "as_of": None, "reason": "car_total_cpca.parquet 缺失"}
        df = pd.read_parquet(CAR_TOTAL_CPCA)
        if "月份" not in df.columns:
            return {"status": "unavailable", "signal": "中性", "detail": "",
                    "unit": "万辆", "as_of": None,
                    "reason": "car_total_cpca.parquet 列不符: 缺 月份"}
        # 展平宽表: (年, 月) → 批发总量（万辆）
        recs: list[tuple[int, int, float]] = []
        for col in df.columns:
            if col == "月份":
                continue
            m = re.match(r"(\d{4})年", str(col))
            if not m:
                continue
            year = int(m.group(1))
            for _, r in df.iterrows():
                mm = re.match(r"(\d+)月", str(r["月份"]))
                if not mm:
                    continue
                val = pd.to_numeric(r[col], errors="coerce")
                if pd.notna(val):
                    recs.append((year, int(mm.group(1)), float(val)))
        if len(recs) < 2:
            return {"status": "unavailable", "signal": "中性", "detail": "",
                    "unit": "万辆", "as_of": None,
                    "reason": "car_total_cpca.parquet 有效期数不足 2 期"}
        recs.sort()
        (y_prev, m_prev, v_prev), (y_cur, m_cur, v_cur) = recs[-2], recs[-1]
        pct = (v_cur / v_prev - 1) * 100
        sig = _signal(v_cur, v_prev, 1.05, 0.95)
        return {
            "status": "available", "signal": sig,
            "detail": f"乘联会批发总量 {y_cur}年{m_cur}月 {v_cur:.1f}万辆，"
                      f"环比 {pct:+.1f}%（前值 {y_prev}年{m_prev}月 {v_prev:.1f}）",
            "unit": "万辆", "as_of": f"{y_cur}-{m_cur:02d}",
            "cur": round(float(v_cur), 2), "prev": round(float(v_prev), 2),
            "pct_chg": round(pct, 2),
        }
    except Exception as e:  # noqa: BLE001
        return {"status": "unavailable", "signal": "中性", "detail": "",
                "unit": "万辆", "as_of": None, "reason": f"{type(e).__name__}: {e}"}

def local_proxies(date: str | None = None) -> dict:
    """本地代理指标 + signal(升温/降温/中性)。

    全部基于本地 parquet 真实计算（industry_graph.links / theme_cycle /
    zt_pool / kline 上市日期推导），无网络。
    """
    end = _parse_day(date)
    anchor = _anchor_date(end)
    result: dict = {"status": "available", "date": None, "proxies": {}, "signal": "中性"}

    # 0) 汽车景气（乘联会批发总量环比; 月度数据不依赖 theme_cycle 交易日锚点）
    result["proxies"]["car_boom"] = _car_boom()

    if anchor is None:
        result["status"] = "unavailable"
        result["reason"] = "theme_cycle 无 ≤ 目标日期 的交易日数据"
        return result
    result["date"] = anchor.strftime("%Y%m%d")
    votes: list[int] = []
    if result["proxies"]["car_boom"]["status"] == "available":
        votes.append(1 if result["proxies"]["car_boom"]["signal"] == "升温"
                     else -1 if result["proxies"]["car_boom"]["signal"] == "降温" else 0)

    # 1) 产业链联动强度
    sc = _supply_chain_series(anchor)
    if sc:
        vals = [v for _, v, _ in sc]
        cur, base = vals[-1], _baseline(vals)
        sig = _signal(cur, base, 1.25, 0.8)
        votes.append(1 if sig == "升温" else -1 if sig == "降温" else 0)
        result["proxies"]["supply_chain_heat"] = {
            "value": round(float(cur), 1), "base_median": round(float(base), 1) if base else None,
            "active_edge_cnt": int(sc[-1][2]), "signal": sig,
            "desc": "产业链联动强度=活跃概念间 industry_graph 传导边 lift 之和",
        }

    # 2) 次新活跃度（招聘景气代理）
    ns = _new_stock_series(anchor)
    if ns:
        vals = [v for _, v, _, _ in ns]
        cur, base = vals[-1], _baseline(vals)
        sig = _signal(cur, base, 1.3, 0.7)
        votes.append(1 if sig == "升温" else -1 if sig == "降温" else 0)
        result["proxies"]["new_stock_activity"] = {
            "value": round(float(cur) * 100, 2) if cur is not None else None,
            "base_median": round(float(base) * 100, 2) if base is not None else None,
            "new_zt_cnt": int(ns[-1][2]), "total_zt_cnt": int(ns[-1][3]),
            "signal": sig, "unit": "%",
            "desc": "次新涨停占比=上市<2年股票涨停数/总涨停数（招聘景气本地代理）",
        }

    # 3) 消费热度
    cs = _consumption_series(anchor)
    if cs:
        vals = [v for _, v, _, _ in cs]
        cur, base = vals[-1], _baseline(vals)
        sig = _signal(cur, base, 1.25, 0.8)
        votes.append(1 if sig == "升温" else -1 if sig == "降温" else 0)
        result["proxies"]["consumption_heat"] = {
            "value": round(float(cur) * 100, 2) if cur is not None else None,
            "base_median": round(float(base) * 100, 2) if base is not None else None,
            "cons_zt_cnt": int(cs[-1][2]), "all_board_zt_cnt": int(cs[-1][3]),
            "signal": sig, "unit": "%",
            "desc": "消费热度=零售/食品饮料/家电概念涨停家数占比（theme_cycle）",
        }

    # 4) 散户关注度（股吧热榜本地代理）
    rf = _retail_focus_series(anchor)
    if rf:
        to_vals = [v for _, v, _ in rf if v is not None]
        amt_vals = [v for _, _, v in rf if v is not None]
        cur_to, base_to = to_vals[-1] if to_vals else None, _baseline(to_vals)
        cur_amt, base_amt = amt_vals[-1] if amt_vals else None, _baseline(amt_vals)
        sig_to = _signal(cur_to, base_to, 1.15, 0.85)
        sig_amt = _signal(cur_amt, base_amt, 1.15, 0.85)
        sig = "升温" if (sig_to == "升温" or sig_amt == "升温") and sig_to != "降温" and sig_amt != "降温" \
            else "降温" if (sig_to == "降温" or sig_amt == "降温") else "中性"
        votes.append(1 if sig == "升温" else -1 if sig == "降温" else 0)
        result["proxies"]["retail_focus"] = {
            "value": round(float(cur_to), 2) if cur_to is not None else None,
            "base_median": round(float(base_to), 2) if base_to is not None else None,
            "zt_amount_yi": round(float(cur_amt) / 1e8, 1) if cur_amt is not None else None,
            "signal": sig, "unit": "%换手",
            "desc": "散户关注度代理=涨停股平均换手率+涨停成交额（股吧热榜不可用兜底）",
        }

    # 汇总信号: 升温 +1 / 降温 -1 / 中性 0 投票
    s = sum(votes)
    if votes:
        result["signal"] = "升温" if s >= 2 else "降温" if s <= -2 else "中性"
    result["vote_detail"] = {"up": votes.count(1), "down": votes.count(-1), "neutral": votes.count(0)}
    return result


# ── 注册表与报告 ─────────────────────────────────────────
def list_sources(date: str | None = None) -> list[dict]:
    """全部源 + 状态(available/unavailable)。"""
    end = _parse_day(date)
    cn = cloud_sources(date)
    guba = _guba_source(date)
    rec = _recruitment_source()
    lp = local_proxies(date)
    proxies = lp.get("proxies", {})
    sources = [
        {"name": "cninfo", "kind": "cloud", "desc": "巨潮公告热度(云爬虫拉回)",
         "status": cn["status"], "detail": cn.get("reason") or cn.get("signal", ""),
         "source": "data_warehouse/cninfo/"},
        {"name": "guba_hot_rank", "kind": "network", "desc": "东财股吧/人气榜(akshare)",
         "status": guba["status"], "detail": guba.get("reason", "可用"),
         "fallback": guba.get("fallback", "")},
        {"name": "recruitment", "kind": "overseas", "desc": "招聘景气(Boss/智联,反爬)",
         "status": rec["status"], "detail": rec["reason"],
         "fallback": rec["fallback"]},
        {"name": "supply_chain_heat", "kind": "local", "desc": "产业链联动(industry_graph.links)",
         "status": "available" if "supply_chain_heat" in proxies else "unavailable",
         "detail": proxies.get("supply_chain_heat", {}).get("signal", "无数据")},
        {"name": "new_stock_activity", "kind": "local", "desc": "次新涨停占比(招聘代理)",
         "status": "available" if "new_stock_activity" in proxies else "unavailable",
         "detail": proxies.get("new_stock_activity", {}).get("signal", "无数据")},
        {"name": "consumption_heat", "kind": "local", "desc": "消费概念涨停占比(theme_cycle)",
         "status": "available" if "consumption_heat" in proxies else "unavailable",
         "detail": proxies.get("consumption_heat", {}).get("signal", "无数据")},
        {"name": "retail_focus", "kind": "local", "desc": "散户关注度代理(股吧兜底)",
         "status": "available" if "retail_focus" in proxies else "unavailable",
         "detail": proxies.get("retail_focus", {}).get("signal", "无数据")},
        {"name": "car_boom", "kind": "local", "desc": "汽车景气(乘联会批发总量环比)",
         "status": proxies.get("car_boom", {}).get("status", "unavailable"),
         "detail": proxies.get("car_boom", {}).get("signal", "无数据")},
    ]
    return sources


def run_report(date: str | None = None) -> dict:
    """汇总云源+本地代理+源状态 → generated/alt_data_report_{date}.json。"""
    report_date = _today() if date is None else str(date).replace("-", "")
    cloud = cloud_sources(date)
    proxies = local_proxies(date)
    guba = _guba_source(date)
    rec = _recruitment_source()

    out = {
        "date": report_date,
        "generated_at": datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S %z"),
        "cloud_sources": cloud,
        "local_proxies": proxies,
        "network_sources": {
            "guba_hot_rank": guba,
            "recruitment": rec,
        },
        "sources": list_sources(date),
    }
    try:
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        path = GENERATED_DIR / f"alt_data_report_{report_date}.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        out["saved_to"] = str(path)
    except Exception as e:  # noqa: BLE001
        out["save_error"] = f"{type(e).__name__}: {e}"
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="另类数据框架 v2")
    ap.add_argument("--sources", action="store_true", help="列出全部源+状态")
    ap.add_argument("--report", action="store_true", help="生成汇总报告")
    ap.add_argument("--date", default=None, help="YYYYMMDD，默认最新")
    args = ap.parse_args()
    if args.sources:
        print(json.dumps(list_sources(args.date), ensure_ascii=False, indent=2))
    elif args.report:
        r = run_report(args.date)
        print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
    else:
        print(json.dumps(run_report(args.date), ensure_ascii=False, indent=2, default=str))
