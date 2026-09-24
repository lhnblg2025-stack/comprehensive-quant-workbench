"""
behavior_system — 体系10 行为金融系统（V11）

方法论借鉴（skills/，先读 skill 再写）:
  behavioral-finance-anomalies    行为金融异象: 动量反转/处置效应/羊群效应/
                                   过度自信/损失厌恶/锚定; 异象需行为机制+套利受限
  expert-intuition-validity-check 专家直觉有效性: 低有效性环境用公式/基线替代直觉
                                   → 本系统全部用可观测数据的显式公式，不做主观拍板
  cognitive-bias-decision-audit   认知偏差审计: 锚定/羊群/损失厌恶 → 慢检查 +
                                   逆向信号（极端=反向）; 先列偏差风险再给判断

行为偏差在数据中的可检测痕迹:
  处置效应: 上涨后放量抛售(解套盘)/下跌缩量惜售 → 涨停股次日 高开回落率 + 负溢价
  羊群效应: 板块齐涨齐跌(指数收益相关性高)、涨停家数暴增/行业集中 → 相关性+集中度
  过度自信: 连板高度极高+炸板率低(情绪亢奋) → 最高板×(1-炸板率) 组合
  追涨杀跌: 涨停次日高开低走 / 跌停次日低开 → 高开幅度+高开低走率+杀跌率
  锚定:     价格反复测试前高前低(密集区行为) → 指数在关键位附近的触碰次数

输入（data_warehouse 现有 parquet，路径与列已确认）:
  data_warehouse/market/zt_daily_stats.parquet     连板天梯聚合（premium/zb_rate/max_board/jr1...）
  data_warehouse/market/zt_pool_history.parquet    涨停池历史长表（is_zt/is_dt/next_pct）
  data_warehouse/market/zt_pool_em_daily.parquet   涨停池明细（industry → 涨停行业集中度）
  data_warehouse/market/fusion.parquet             融合温度 + 行业广度（申万/中证）
  data_warehouse/market/index_daily*.parquet       指数日线（OHLC → 收益相关性与锚定关键位）
  data_warehouse/market/a_high_low.parquet         创新高/新低家数
  generated/a_share_data/*-meta.json               真实涨跌家数（上涨/下跌/平盘，剔除ST口径）
  data_warehouse/kline/*.parquet                   个股K线（近窗口 涨停股次日开盘vs收盘）

输出:
  generated/behavior_report_{date}.md

统一接口:
  BehaviorSystem().detect(date) / report(date) / view(date)
  view() 返回 {agent:'行为金融', signal, view, confidence, evidence, weight, status, detail}

防御: 任一数据源缺失 → 跳过对应指数并标注 data_status，单指标失败不中断。

用法:
  python3 -m quant_system.analysis_core.behavior_system --date 2026-08-11
  python3 -m quant_system.analysis_core.behavior_system --date 2026-08-11 --report
"""

from __future__ import annotations
import logging

import argparse
import json
import os
import sys
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent  # workspace 根
sys.path.insert(0, str(ROOT))

# 离线优先: sentence-transformers 在本环境不可下载模型 → knowledge_rag 走关键词兜底
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from quant_system.analysis_core.config import (  # noqa: E402
    MARKET_DIR,
    ZT_DAILY_STATS,
    ZT_EM_DAILY,
    ZT_HISTORY,
)

from quant_system.analysis_core.common import (  # noqa: E402
    fmt,
    load_zt_stats,
    norm_date,
    num,
    
    today,
)
from quant_system.analysis_core.fusion import read_fusion_latest  # noqa: E402
from quant_system.analysis_core.rag_explain import rag_explain  # noqa: E402

CST = timezone(timedelta(hours=8))
OUT_DIR = Path(__file__).resolve().parent.parent / "generated"

A_SHARE_META_DIR = ROOT / "generated" / "a_share_data"  # 真实涨跌家数（周度拉取）
KLINE_DIR = ROOT / "data_warehouse" / "kline"
A_HIGH_LOW = MARKET_DIR / "a_high_low.parquet"
FUSION = MARKET_DIR / "fusion.parquet"
INDEX_DAILY = MARKET_DIR / "index_daily.parquet"
INDEX_NAMED = sorted(MARKET_DIR.glob("index_daily_*.parquet"))  # 多指数相关性

RAG_QUERY = "行为金融 羊群效应 处置效应 逆向"
RAG_K = 3
RAG_CACHE = OUT_DIR / "rag_behavior_cache.json"  # 静态查询结果缓存（子进程检索一次后即读缓存）

# ── 指数构成权重（缺失时按可用项归一化）──────────────────
HERD_WEIGHTS = {"corr": 0.35, "conc": 0.25, "breadth": 0.20, "industry": 0.20}
CHASE_WEIGHTS = {"gap": 0.40, "giveback": 0.30, "selloff": 0.30}
DISP_WEIGHTS = {"premium": 0.60, "giveback": 0.40}
COMPOSITE_WEIGHTS = {"disposition": 0.20, "herding": 0.20, "overconfidence": 0.25,
                     "chase": 0.20, "anchoring": 0.15}

# ── 行为状态判定阈值（V1 经验值，随预测记录库校准）────────
STATE_RULES = {
    "extreme_optimism": {"overconfidence": 62, "herding": 55, "chase": 50},
    "extreme_pessimism": {"disposition": 62, "temp_max": 35, "zt_cnt_max": 45},
    "herding": {"herding": 58},
}

STATE_CN = {
    "rational": "理性",
    "herding": "从众",
    "extreme_optimism": "极端乐观",
    "extreme_pessimism": "极端悲观",
}


def _recent_mean(df: pd.DataFrame, col: str, n: int = 3) -> float | None:
    """列最近 n 个非 NaN 值的均值（NaN 如当日溢价天然缺失 → 取前值均值）。"""
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    if not len(vals):
        return None
    return round(float(vals.tail(n).mean()), 4)


def _isna(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x)


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


# ════════════════════════════════════════════════════════════
# 数据加载（防御: 缺失返回 None 并标注）
# ════════════════════════════════════════════════════════════

def _load_fusion(date: str) -> tuple[float | None, str | None, dict | None]:
    """fusion.parquet 最新 ≤date。返回 (温度, 实际日期, industry 元数据)。"""
    row = read_fusion_latest(date)
    if row is None:
        return None, None, None
    temp = num(row.get("temperature"))
    industry = None
    raw = row.get("industry")
    if isinstance(raw, str):
        try:
            industry = json.loads(raw)
        except Exception:  # noqa: BLE001
            industry = None
    elif isinstance(raw, dict):
        industry = raw
    return temp, row["date"], industry


def _load_em_pool(date: str) -> pd.DataFrame | None:
    """东财涨停池明细 ≤date 最新一日（industry/zb_times → 涨停行业集中度）。"""
    if not ZT_EM_DAILY.exists():
        return None
    try:
        df = pd.read_parquet(ZT_EM_DAILY)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        hist = df[df["date"] <= pd.Timestamp(date)].sort_values("date")
        if hist.empty:
            return None
        return hist[hist["date"] == hist["date"].max()]
    except Exception:  # noqa: BLE001
        return None


def _load_pool_history(date: str, days: int = 4) -> pd.DataFrame | None:
    """涨停池历史长表 ≤date 尾部 days 个交易日（is_zt/is_dt/next_pct）。"""
    if not ZT_HISTORY.exists():
        return None
    try:
        df = pd.read_parquet(ZT_HISTORY)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        hist = df[df["date"] <= pd.Timestamp(date)]
        dates = hist["date"].unique()
        if len(dates) == 0:
            return None
        tail = dates[-days:]
        return hist[hist["date"].isin(tail)]
    except Exception:  # noqa: BLE001
        return None


def _load_index_corr(date: str, lookback: int = 20) -> float | None:
    """多指数日收益平均两两相关系数（板块齐涨齐跌程度，羊群代理）。"""
    rets = None
    n_idx = 0
    files = list(INDEX_NAMED) + ([INDEX_DAILY] if INDEX_DAILY.exists() else [])
    for f in files:
        try:
            df = pd.read_parquet(f, columns=["date", "close"])
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date", "close"])
            df = df[df["date"] <= pd.Timestamp(date)].sort_values("date")
            if len(df) < lookback + 2:
                continue
            r = df["close"].pct_change().dropna().tail(lookback).reset_index(drop=True)
            r.name = f.stem
            rets = r if rets is None else pd.concat([rets, r], axis=1)
            n_idx += 1
        except Exception as e:  # noqa: BLE001
            logging.getLogger(__name__).error(f"[behavior_system] 操作失败: {e}", exc_info=True)
            continue
    if rets is None or n_idx < 3 or len(rets) < lookback * 0.5:
        return None
    corr = rets.corr()
    vals = corr.values[np.triu_indices(len(corr), k=1)]
    if not len(vals):
        return None
    return round(float(np.nanmean(vals)), 4)


def _load_index_close(date: str) -> pd.DataFrame | None:
    """指数收盘序列（锚定关键位用）。

    只用主指数 index_daily.parquet（沪深300 序列）；主指数缺失/过短时用
    index_daily_沪深300.parquet 补齐，二者为同一序列，不做多指数拼接。
    """
    candidates = [INDEX_DAILY, MARKET_DIR / "index_daily_沪深300.parquet"]
    for f in candidates:
        if not f.exists():
            continue
        try:
            df = pd.read_parquet(f, columns=["date", "close"])
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date", "close"])
            df = df[df["date"] <= pd.Timestamp(date)].sort_values("date")
            if len(df) >= 30:
                return df.reset_index(drop=True)
        except Exception as e:  # noqa: BLE001
            logging.getLogger(__name__).error(f"[behavior_system] 操作失败: {e}", exc_info=True)
            continue
    return None


def _load_a_high_low(date: str) -> dict | None:
    """创新高/新低家数（宽度代理）≤date 最新一日。"""
    if not A_HIGH_LOW.exists():
        return None
    try:
        df = pd.read_parquet(A_HIGH_LOW)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        hist = df[df["date"] <= pd.Timestamp(date)].sort_values("date")
        if hist.empty:
            return None
        r = hist.iloc[-1]
        return {
            "as_of": r["date"].strftime("%Y-%m-%d"),
            "high20": int(r.get("high20") or 0), "low20": int(r.get("low20") or 0),
            "high60": int(r.get("high60") or 0), "low60": int(r.get("low60") or 0),
        }
    except Exception:  # noqa: BLE001
        return None


def _load_ad_counts(date: str) -> dict | None:
    """真实涨跌家数: generated/a_share_data/{YYYYMMDD}-meta.json（剔除ST口径）。"""
    if not A_SHARE_META_DIR.exists():
        return None
    try:
        files = sorted(A_SHARE_META_DIR.glob("*-meta.json"))
        if not files:
            return None
        target = pd.Timestamp(date)
        best = None
        for f in files:
            try:
                d = pd.Timestamp(f.name.split("-")[0])
            except Exception as e:  # noqa: BLE001
                logging.getLogger(__name__).error(f"[behavior_system] 操作失败: {e}", exc_info=True)
                continue
            if d <= target and (best is None or d > best[0]):
                best = (d, f)
        if best is None:
            return None
        d, f = best
        meta = json.loads(f.read_text(encoding="utf-8"))
        bread = meta.get("breadth") or meta.get("raw_breadth") or {}
        up = num(bread.get("上涨"))
        down = num(bread.get("下跌"))
        flat = num(bread.get("平盘"), 0.0)
        if up is None or down is None:
            return None
        total = up + down + (flat or 0)
        return {
            "as_of": d.strftime("%Y-%m-%d"),
            "up": int(up), "down": int(down), "flat": int(flat or 0),
            "total": int(total or bread.get("股票数") or (up + down)),
            "up_ratio": round(up / max(total, 1), 4),
            "up5": int(bread.get("涨超5%") or 0), "down5": int(bread.get("跌超5%") or 0),
        }
    except Exception:  # noqa: BLE001
        return None


def _load_nextday_kline(pool: pd.DataFrame | None, date: str) -> pd.DataFrame | None:
    """近窗口涨停股次日开盘/收盘（追涨杀跌 + 处置抛压的核心证据）。

    对池内每只涨停股读 kline/{code}.parquet，取该股 T+1 交易日 open/close:
      gap      = open_{t+1} / close_t - 1   （高开幅度）
      down     = close_{t+1} / open_{t+1} - 1（高开后回落幅度）
    池内日期用 zt_pool_history 实际交易日的次日，无 kline 的股票跳过。
    """
    if pool is None or pool.empty:
        return None
    try:
        codes = sorted(pool["code"].astype(str).unique())
        rows = []
        for code in codes:
            f = KLINE_DIR / f"{code}.parquet"
            if not f.exists():
                continue
            df = pd.read_parquet(f, columns=["date", "open", "close"])
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date", "open", "close"]).sort_values("date")
            df = df[df["date"] <= pd.Timestamp(date)]  # 防前视: 分析日当天可用，剔除未来行
            if len(df) < 2:
                continue
            # 2026-08-14 修复: np.datetime64 与 pd.Timestamp hash 不相等
            # (== True 但 hash 不同) → dict 查找 miss → 次日K线恒空。
            # 统一用 Timestamp 作 key。
            idx = {pd.Timestamp(d): i for i, d in enumerate(df["date"].values)}
            for _, r in pool[pool["code"].astype(str) == code].iterrows():
                i = idx.get(r["date"])
                if i is None or i + 1 >= len(df):
                    continue
                nxt = df.iloc[i + 1]
                close_t = num(r.get("close"))
                if not close_t or close_t <= 0 or not num(nxt["open"]):
                    continue
                rows.append({
                    "date": r["date"], "code": code,
                    "gap": num(nxt["open"]) / close_t - 1.0,
                    "down": num(nxt["close"]) / num(nxt["open"]) - 1.0,
                    "next_pct": num(r.get("next_pct")),
                })
        if not rows:
            return None
        return pd.DataFrame(rows)
    except Exception:  # noqa: BLE001
        return None


# ════════════════════════════════════════════════════════════
# 六项行为指数
# ════════════════════════════════════════════════════════════

def _disposition_index(stats: pd.DataFrame | None, kday: pd.DataFrame | None,
                       premium3: float | None) -> dict:
    """1. 处置效应指数 0-100: 涨后抛压（高开回落=解套盘）+ 负溢价=分歧。

    高溢价=追涨承接强(羊群未分歧) → 抛压低分；低/负溢价=抛压释放 → 高分。
    """
    ev: list[str] = []
    parts: dict[str, float] = {}

    # 溢价代理（60%）：3日均溢价
    prem_score = None
    if premium3 is not None:
        prem_score = round(_clip01((1.5 - premium3) / 5.0) * 100, 1)
        ev.append(f"3日均溢价{premium3:+.2f}%→抛压分{prem_score:.0f}")
        parts["premium"] = prem_score

    # K线证据（40%）：涨>5%股(涨停池代理)次日 高开回落率 / 平均回落
    giveback_score = None
    if kday is not None and len(kday):
        g = kday[kday["gap"] > 0.01]
        if len(g):
            mean_down = float(g["down"].mean())
            giveback_score = round(_clip01(-mean_down / 0.02) * 100, 1)
            ev.append(f"高开股{len(g)}只 平均回落{mean_down:+.2%}→抛压分{giveback_score:.0f}")
            parts["giveback"] = giveback_score
        else:
            ev.append("近窗口无高开>1%的涨停股（承接弱）→ 抛压证据不足")

    score = _weighted(parts, DISP_WEIGHTS)
    return {"score": score, "parts": parts, "evidence": ev}


def _herding_index(corr: float | None, em_pool: pd.DataFrame | None,
                   ad: dict | None, fusion_industry: dict | None,
                   hl: dict | None, zt_cnt: int | None) -> dict:
    """2. 羊群效应指数 0-100: 指数相关性 + 涨停行业集中度 + 涨跌家数普涨/普跌 + 行业广度。"""
    ev: list[str] = []
    parts: dict[str, float] = {}

    if corr is not None:
        c = _clip01((corr - 0.3) / 0.5) * 100
        parts["corr"] = round(c, 1)
        ev.append(f"多指数20日平均相关{corr:.2f}→{parts['corr']:.0f}")

    if em_pool is not None and len(em_pool):
        zt = em_pool[em_pool.get("is_zt", pd.Series(False, index=em_pool.index)).fillna(False)]
        ind = zt["industry"].dropna()
        if len(ind):
            top1 = ind.value_counts(normalize=True).iloc[0]
            parts["conc"] = round(_clip01((top1 - 0.12) / 0.4) * 100, 1)
            top3 = ind.value_counts(normalize=True).head(3).sum()
            ev.append(f"涨停行业集中度 第1名{top1:.0%}(Top3 {top3:.0%})→{parts['conc']:.0f}")

    if ad is not None:
        ur = ad["up_ratio"]
        spread = max(ur, 1 - ur)
        parts["breadth"] = round(_clip01((spread - 0.52) / 0.3) * 100, 1)
        ev.append(f"涨跌家数 涨{ad['up']}/跌{ad['down']} 占比{ur:.0%}→齐涨齐跌分{parts['breadth']:.0f}")

    if fusion_industry is not None and num(fusion_industry.get("breadth")) is not None:
        br = float(fusion_industry["breadth"])
        spread = max(br, 1 - br)
        parts["industry"] = round(_clip01((spread - 0.60) / 0.3) * 100, 1)
        src = "申万" if fusion_industry.get("csi", {}).get("status") == "available" else "行业"
        ev.append(f"行业广度{br:.0%}({fusion_industry.get('as_of')})→{parts['industry']:.0f}")

    if zt_cnt is not None:
        ev.append(f"涨停{int(zt_cnt)}家" + ("（家数高=题材集中共振）" if zt_cnt >= 80 else ""))

    score = _weighted(parts, HERD_WEIGHTS)
    return {"score": score, "parts": parts, "evidence": ev}


def _overconfidence_index(stats: pd.DataFrame | None) -> dict:
    """3. 过度自信指数 0-100: 连板高度 × (1-炸板率)，高板低炸=自信顶峰。"""
    ev: list[str] = []
    if stats is None or stats.empty:
        return {"score": None, "parts": {}, "evidence": ev}
    row = stats.iloc[-1]
    mb3 = num(row.get("max_board_t3"), num(row.get("max_board")))
    zb3 = num(row.get("zb_rate_t3"), num(row.get("zb_rate")))
    zt3 = num(row.get("zt_cnt_t3"), num(row.get("zt_cnt")))
    jr1 = num(row.get("jr1_t3"), num(row.get("jr1")))
    if mb3 is None:
        return {"score": None, "parts": {}, "evidence": ["连板高度缺失，跳过过度自信指数"]}

    board = _clip01((mb3 - 1) / 5.0)
    quality = 1.0 - _clip01((zb3 or 0.0) / 0.5)
    breadth = _clip01((zt3 or 0.0) / 80.0)  # 涨停家数广度: 家数极少时高板≠自信
    score = round(100 * board * (0.55 + 0.45 * quality) * breadth, 1)
    ev.append(f"3日均最高板{mb3:.1f} × 炸板率{(zb3 or 0)*100:.0f}%(质量{quality*100:.0f}%) "
              f"× 涨停家数因子{breadth:.0%}({zt3:.0f}家) → 自信分{score:.0f}")
    if jr1 is not None:
        ev.append(f"1进2晋级率{jr1:.0%}" + ("（接力一致=自信强化）" if jr1 >= 0.3 else ""))
    return {"score": score, "parts": {"board": round(board * 100, 1),
                                      "quality": round(quality * 100, 1),
                                      "breadth": round(breadth * 100, 1)}, "evidence": ev}


def _chase_index(kday: pd.DataFrame | None, pool: pd.DataFrame | None,
                 premium3: float | None) -> dict:
    """4. 追涨杀跌指数 0-100: 涨停次日高开(追涨) + 高开低走(被套) + 杀跌率(恐慌)。"""
    ev: list[str] = []
    parts: dict[str, float] = {}

    if kday is not None and len(kday):
        gap = float(kday["gap"].mean())
        giveback_rate = float(((kday["gap"] > 0.01) & (kday["down"] < 0)).mean())
        selloff_rate = float((kday["next_pct"].fillna(0) <= -3.0).mean())
        parts["gap"] = round(_clip01(gap / 0.05) * 100, 1)
        parts["giveback"] = round(_clip01(giveback_rate / 0.30) * 100, 1)
        parts["selloff"] = round(_clip01(selloff_rate / 0.25) * 100, 1)
        ev.append(f"涨停次日 平均高开{gap:+.2%} 高开低走率{giveback_rate:.0%} 杀跌率(≤-3%){selloff_rate:.0%}")
        if pool is not None and len(pool):
            dt = pool[pool.get("is_dt", pd.Series(False, index=pool.index)).fillna(False)]
            dt_next = dt["next_pct"].dropna()
            if len(dt_next):
                ev.append(f"跌停股次日平均{dt_next.mean():+.2f}%" +
                          ("（低开延续=杀跌）" if dt_next.mean() < 0 else ""))
    elif premium3 is not None:
        parts["gap"] = round(_clip01(max(premium3, 0) / 4.0) * 100, 1)
        parts["selloff"] = round(_clip01(max(-premium3, 0) / 4.0) * 100, 1)
        parts["giveback"] = 50.0
        ev.append(f"K线窗口缺失 → 溢价代理: 3日均溢价{premium3:+.2f}%")
    else:
        ev.append("追涨杀跌证据缺失（无K线窗口且无溢价）")

    score = _weighted(parts, CHASE_WEIGHTS)
    return {"score": score, "parts": parts, "evidence": ev}


def _anchoring_index(close_df: pd.DataFrame | None, lookback: int = 30) -> dict:
    """5. 锚定强度 0-100: 指数在 前高/前低/整数关口 附近的反复触碰次数。

    关键位: 滚动120日前高/前低、前60日前高/前低、最近整数关口(每1000点)。
    触碰判定: 收盘价距关键位 ≤0.5%。
    """
    ev: list[str] = []
    if close_df is None or len(close_df) < lookback + 125:
        return {"score": None, "parts": {}, "evidence": ["指数历史不足(需≥155交易日)，锚定指数跳过"]}
    closes = close_df["close"].astype(float).reset_index(drop=True)
    n = len(closes)
    touches: dict[str, int] = {"前高120": 0, "前低120": 0, "前高60": 0, "前低60": 0, "整数关口": 0}
    recent = closes.iloc[n - lookback:]
    for i in range(n - lookback, n):
        c = closes.iloc[i]
        if c <= 0:
            continue
        h120 = closes.iloc[max(0, i - 120):i].max()
        l120 = closes.iloc[max(0, i - 120):i].min()
        h60 = closes.iloc[max(0, i - 60):i].max()
        l60 = closes.iloc[max(0, i - 60):i].min()
        int_lvl = round(c / 1000.0) * 1000.0
        for key, lvl in (("前高120", h120), ("前低120", l120), ("前高60", h60),
                         ("前低60", l60), ("整数关口", int_lvl)):
            if lvl and abs(c - lvl) / lvl <= 0.005:
                touches[key] += 1
    total = sum(touches.values())
    score = round(_clip01(total / 18.0) * 100, 1) if total else 0.0
    hit = {k: v for k, v in touches.items() if v}
    ev.append(f"近{lookback}日关键位触碰 {total} 次" +
              ("(" + "/".join(f"{k}{v}" for k, v in hit.items()) + ")" if hit else "无"))
    ev.append(f"最新收盘 {recent.iloc[-1]:.0f} → 锚定分{score:.0f}"
              + ("（关键位密集测试→突破/假突破风险）" if score >= 60 else ""))
    return {"score": score, "parts": {"touches": total, **touches}, "evidence": ev}


# ════════════════════════════════════════════════════════════
# 工具
# ════════════════════════════════════════════════════════════

def _weighted(parts: dict[str, float], weights: dict[str, float]) -> float | None:
    """按可用分项加权平均（缺失分项自动剔除并归一化权重）。"""
    total_w = sum(w for k, w in weights.items() if k in parts)
    if not total_w or not parts:
        return None
    return round(sum(parts[k] * weights[k] for k in parts) / total_w, 1)


def _proxy_temp(row: pd.Series) -> float:
    """代理温度 0-100（连板天梯特征，fusion 缺失时兜底，公式透明）。"""
    def _c(x, lo=0.0, hi=1.0):
        v = num(x, np.nan)
        if _isna(v):
            return 0.5
        return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))

    zt = _c(row.get("zt_cnt"), 0, 100)
    mb = _c(row.get("max_board"), 0, 7)
    zb = 1.0 - _c(row.get("zb_rate"), 0, 0.6)
    dt = 1.0 - _c(row.get("dt_cnt"), 0, 40)
    jr = _c(row.get("jr1") if not _isna(row.get("jr1")) else row.get("jr1_t3"), 0, 0.5)
    prem = num(row.get("premium"))
    if _isna(prem):
        prem = num(row.get("premium_t3"), 0.0) or 0.0
    prem_norm = (float(np.clip(prem, -5, 5)) / 5 + 1) / 2
    proxy = (50 + 22 * (2 * zt - 1) + 10 * (2 * mb - 1) + 8 * (2 * zb - 1)
             + 6 * (2 * dt - 1) + 4 * (2 * jr - 1) + 4 * (2 * prem_norm - 1))
    return round(float(np.clip(proxy, 5, 95)), 1)


# ════════════════════════════════════════════════════════════
# 综合: 行为状态 + 逆向信号
# ════════════════════════════════════════════════════════════

def _composite(indices: dict, temp: float | None, zt_cnt: int | None,
               zt_cnt_t3: float | None, premium3: float | None) -> dict:
    """标准化六指数 → 行为状态(理性/从众/极端乐观/极端悲观) + 逆向信号(极端=反向)。"""
    disp = (indices["disposition"].get("score") if indices["disposition"].get("score") is not None else 0.0)
    herd = (indices["herding"].get("score") if indices["herding"].get("score") is not None else 0.0)
    conf_ = (indices["overconfidence"].get("score") if indices["overconfidence"].get("score") is not None else 0.0)
    chase = (indices["chase"].get("score") if indices["chase"].get("score") is not None else 0.0)
    anchor = (indices["anchoring"].get("score") if indices["anchoring"].get("score") is not None else 0.0)

    avail = {"disposition": disp, "herding": herd, "overconfidence": conf_,
             "chase": chase, "anchoring": anchor}
    comp_parts = {k: v for k, v in avail.items() if v is not None}
    composite = _weighted(comp_parts, COMPOSITE_WEIGHTS) if comp_parts else None

    # ── 状态判定（规则优先，方向来自不同偏差组合）──
    state = "rational"
    state_ev: list[str] = []
    contrarian = {"trigger": None, "direction": None, "reason": None, "intensity": 0}

    r_opt = STATE_RULES["extreme_optimism"]
    # 乐观确认: 溢价为正(追涨有承接) 且 温度/家数/追涨任一确认，防止齐跌高相关误判
    optimism_sentiment = (premium3 is None or premium3 > 0) and (
        (temp is not None and temp >= 55) or (zt_cnt_t3 is not None and zt_cnt_t3 >= 60)
        or (herd >= r_opt["herding"]) or chase >= r_opt["chase"])
    if conf_ >= r_opt["overconfidence"] and optimism_sentiment:
        state = "extreme_optimism"
        intensity = _clip01((conf_ - r_opt["overconfidence"]) / 38.0)
        contrarian = {"trigger": "极端乐观",
                      "direction": "逆向看空",
                      "reason": f"过度自信{conf_:.0f} + 羊群{herd:.0f} + 追涨{chase:.0f} → 情绪亢奋顶峰，极端=反向",
                      "intensity": round(intensity, 2)}
        state_ev.append(f"过度自信{conf_:.0f}≥{r_opt['overconfidence']} 且 溢价{fmt(premium3, 2)}%>0 + "
                      f"温度{fmt(temp, 0)}/家数{fmt(zt_cnt_t3, 0)}/羊群{herd:.0f}/追涨{chase:.0f} 任一确认")

    r_pes = STATE_RULES["extreme_pessimism"]
    if (disp >= r_pes["disposition"]
            and (zt_cnt is not None and zt_cnt <= r_pes["zt_cnt_max"])
            and (premium3 is not None and premium3 <= -1.0)
            and (temp is None or temp <= 45)):
        if state != "extreme_optimism":
            state = "extreme_pessimism"
            intensity = _clip01((disp - r_pes["disposition"]) / 38.0)
            contrarian = {"trigger": "极端悲观",
                          "direction": "逆向看多",
                          "reason": f"处置抛压{disp:.0f} + 涨停仅{zt_cnt}家 + 溢价{premium3:+.2f}% → 恐慌抛售释放，极端=反向",
                          "intensity": round(intensity, 2)}
            state_ev.append(f"处置抛压{disp:.0f}≥{r_pes['disposition']} 且 涨停{zt_cnt}≤{r_pes['zt_cnt_max']} 且 溢价{premium3:+.2f}%≤-1")

    if state == "rational" and herd >= STATE_RULES["herding"]["herding"]:
        state = "herding"
        contrarian = {"trigger": "从众",
                      "direction": "不追高",
                      "reason": f"羊群效应{herd:.0f} 显著（齐涨齐跌+行业集中），跟随盘面易高位接盘",
                      "intensity": round(_clip01((herd - 58) / 30.0), 2)}
        state_ev.append(f"羊群{herd:.0f}≥{STATE_RULES['herding']['herding']} 未达极端 → 从众警示")

    if state == "rational":
        state_ev.append(f"各指数均未触发极端/羊群阈值（过度自信{conf_:.0f}/羊群{herd:.0f}/"
                        f"处置{disp:.0f}/追涨杀跌{chase:.0f}/锚定{anchor:.0f}）")

    # ── 信号/置信（极端=反向；从众=中性偏谨慎；理性=中性低置信）──
    if state == "extreme_optimism":
        signal, view, conf = "看空", "空", 0.62 + 0.28 * contrarian["intensity"]
    elif state == "extreme_pessimism":
        signal, view, conf = "看多", "多", 0.62 + 0.28 * contrarian["intensity"]
    elif state == "herding":
        signal, view, conf = "中性", "震荡", 0.52 + 0.08 * contrarian["intensity"]
    else:
        signal, view, conf = "中性", "震荡", 0.40
    if anchor >= 60:
        conf = min(0.92, conf + 0.05)
    return {
        "composite_score": composite,
        "state": state,
        "state_cn": STATE_CN[state],
        "state_evidence": state_ev,
        "signal": signal,
        "view": view,
        "confidence": round(min(conf, 0.92), 2),
        "contrarian": contrarian,
        "anchor_note": "关键位密集测试→谨防假突破/假破位" if anchor >= 60 else None,
    }


# ════════════════════════════════════════════════════════════
# RAG 解释（公共块 rag_explain：子进程检索 + 本地关键词兜底 + 7天缓存）
# ════════════════════════════════════════════════════════════

def _rag_explain() -> list[dict]:
    """RAG 依据: knowledge_rag.search（子进程）→ 结果缓存 → 本地关键词兜底。"""
    return rag_explain(RAG_QUERY, RAG_K, RAG_CACHE, ROOT)


# 统一接口
# ════════════════════════════════════════════════════════════

class BehaviorSystem:
    """体系10 行为金融系统 — detect/report/view 统一接口。"""

    def __init__(self, out_dir: Path | str | None = None):
        self.out_dir = Path(out_dir) if out_dir else OUT_DIR

    # ── detect ─────────────────────────────────────────────
    def detect(self, date: str | None = None) -> dict:
        date = norm_date(date)
        data_status: dict[str, str] = {}

        # 1. 数据加载
        stats = load_zt_stats(ZT_DAILY_STATS, date)
        if stats is None or stats.empty:
            data_status["连板天梯"] = f"缺失 {ZT_DAILY_STATS} → 跳过全部指数"
            return {
                "date": date, "agent": "行为金融", "signal": "中性",
                "view": "震荡", "confidence": 0.0, "status": "degraded",
                "evidence": [f"连板天梯数据缺失，无法计算行为指数: {ZT_DAILY_STATS}"],
                "data_status": data_status,
            }
        data_date = stats["date"].iloc[-1].strftime("%Y-%m-%d")
        stats_lag = max(0, (pd.Timestamp(date) - stats["date"].iloc[-1]).days)
        if stats_lag > 7:
            data_status["连板天梯"] = (f"落后 {stats_lag} 日(as_of={data_date}) → 按最新可用交易日判定")

        fusion_temp, fusion_as_of, fusion_industry = _load_fusion(date)
        if fusion_temp is None:
            data_status["fusion温度"] = "缺失 → 温度用代理温度兜底，不参与羊群行业广度"
        if fusion_industry is not None and fusion_industry.get("status") != "available":
            data_status["行业广度"] = "fusion industry 不可用 → 行业广度分量跳过"
            fusion_industry = None

        em_pool = _load_em_pool(date)
        if em_pool is None:
            data_status["涨停池明细"] = f"缺失 {ZT_EM_DAILY} → 涨停行业集中度跳过"

        pool = _load_pool_history(date)
        if pool is None or pool.empty:
            data_status["涨停池历史"] = f"缺失 {ZT_HISTORY} → 追涨杀跌K线窗口跳过"

        corr = _load_index_corr(date)
        if corr is None:
            data_status["指数相关性"] = "多指数日线不足(需≥3个指数×20日) → 相关性分量跳过"

        close_df = _load_index_close(date)
        if close_df is None:
            data_status["指数收盘"] = "指数日线不足 → 锚定强度跳过"

        hl = _load_a_high_low(date)
        ad = _load_ad_counts(date)
        if ad is None:
            data_status["涨跌家数"] = (f"缺失 a_share_data meta → "
                                       f"{'用新高/新低比兜底' if hl is not None else '跳过广度分量'}")

        # 2. 温度
        row = stats.iloc[-1]
        temp = fusion_temp if fusion_temp is not None else _proxy_temp(row)
        temp_src = f"fusion({fusion_as_of})" if fusion_temp is not None else "代理温度(fusion缺失)"

        # 3. 六项指数（单指标失败独立跳过）
        premium3 = _recent_mean(stats, "premium", 3) or num(row.get("premium"))
        kday = _load_nextday_kline(pool, date) if pool is not None else None
        if kday is None or kday.empty:
            data_status["次日开盘/收盘"] = "K线窗口不可用 → 处置/追涨用溢价代理"

        indices = {
            "disposition": _disposition_index(stats, kday, premium3),
            "herding": _herding_index(corr, em_pool, ad, fusion_industry, hl,
                                      num(row.get("zt_cnt"))),
            "overconfidence": _overconfidence_index(stats),
            "chase": _chase_index(kday, pool, premium3),
            "anchoring": _anchoring_index(close_df),
        }
        for name, it in indices.items():
            if it.get("score") is None:
                data_status[f"指数.{name}"] = "分项数据不足 → 跳过（综合权重自动剔除）"

        # 4. 综合
        comp = _composite(indices, temp, num(row.get("zt_cnt")),
                          num(row.get("zt_cnt_t3"), num(row.get("zt_cnt"))), premium3)

        # 5. RAG 依据
        rag = _rag_explain()

        # 6. 证据
        evidence = [
            f"行为状态: {comp['state_cn']} | 综合行为强度 {fmt(comp['composite_score'], 0)} | 温度 {fmt(temp, 0)}({temp_src})",
            *[f"[{cn}] {it['evidence'][0] if it['evidence'] else '分项缺失'}"
              for cn, it in zip(("处置效应", "羊群效应", "过度自信", "追涨杀跌", "锚定强度"),
                                indices.values())],
            *comp["state_evidence"],
        ]
        if comp["contrarian"].get("trigger"):
            evidence.append(f"逆向信号: {comp['contrarian']['direction']} "
                            f"（触发: {comp['contrarian']['trigger']}，强度 {comp['contrarian']['intensity']:.0%}）")
        if comp.get("anchor_note"):
            evidence.append(comp["anchor_note"])

        return {
            "date": date,
            "data_date": data_date,
            "agent": "行为金融",
            "temperature": round(float(temp), 1),
            "temperature_source": temp_src,
            "indices": indices,
            "composite_score": comp["composite_score"],
            "state": comp["state"],
            "state_cn": comp["state_cn"],
            "state_evidence": comp["state_evidence"],
            "contrarian": comp["contrarian"],
            "signal": comp["signal"],
            "view": comp["view"],
            "confidence": comp["confidence"],
            "evidence": evidence,
            "rag": rag,
            "data_status": data_status,
            "status": "ok" if not data_status else "degraded",
        }

    # ── report ─────────────────────────────────────────────
    def report(self, date: str | None = None, out_dir: Path | str | None = None) -> Path:
        return self._write_report(self.detect(date), out_dir)

    def _write_report(self, r: dict, out_dir: Path | str | None = None) -> Path:
        out = self.out_dir if out_dir is None else Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"behavior_report_{r.get('data_date', r['date'])}.md"
        path.write_text(_render_markdown(r), encoding="utf-8")
        return path

    # ── view ───────────────────────────────────────────────
    def view(self, date: str | None = None) -> dict:
        """多智能体视图: {agent, signal, view, confidence, evidence, weight, status, detail}。"""
        return self._view(self.detect(date))

    @staticmethod
    def _view(r: dict) -> dict:
        return {
            "agent": "行为金融",
            "signal": r["signal"],
            "view": r["view"],
            "confidence": r["confidence"],
            "evidence": r["evidence"],
            "weight": 1.0,
            "status": r["status"],
            "detail": {
                "date": r.get("data_date", r["date"]),
                "state": r.get("state_cn", "未知"),
                "composite_score": r.get("composite_score"),
                "contrarian": r.get("contrarian"),
                "indices": {k: {"score": v.get("score"), "parts": v.get("parts")}
                            for k, v in (r.get("indices") or {}).items()},
            },
        }


# ════════════════════════════════════════════════════════════
# 报告渲染
# ════════════════════════════════════════════════════════════

_INDEX_CN = {
    "disposition": "处置效应指数",
    "herding": "羊群效应指数",
    "overconfidence": "过度自信指数",
    "chase": "追涨杀跌指数",
    "anchoring": "锚定强度",
}
_INDEX_EXPLAIN = {
    "disposition": "涨后抛压（高开回落=解套盘）+ 低溢价分歧 → 高分=抛压释放",
    "herding": "指数相关性 + 涨停行业集中 + 齐涨齐跌 → 高分=从众",
    "overconfidence": "连板高度 × (1-炸板率) → 高板低炸=自信顶峰",
    "chase": "涨停次日高开/高开低走/杀跌率 → 高分=追涨杀跌行为显著",
    "anchoring": "指数反复测试前高/前低/整数关口 → 高分=锚定密集区",
}


def _render_markdown(r: dict) -> str:
    lines = [
        f"# 行为金融系统报告 — {r.get('data_date', r['date'])}",
        "",
        f"- 结论: **{r['signal']}** | 置信 {r['confidence']:.0%} | 行为状态: **{r.get('state_cn', '未知')}**",
        f"- 综合行为强度: {fmt(r.get('composite_score'), 0)} | 情绪温度: **{fmt(r.get('temperature'), 0)}**"
        f"（{r.get('temperature_source')}）",
        "",
        "## 六项行为指数（0-100）",
    ]
    for key, cn in _INDEX_CN.items():
        it = (r.get("indices") or {}).get(key) or {}
        score = it.get("score")
        score_txt = f"**{score:.0f}**" if score is not None else "NA(跳过)"
        parts_txt = " | ".join(f"{k}:{v:.0f}" for k, v in (it.get("parts") or {}).items()) or "无分项"
        lines.append(f"- {cn}: {score_txt}（{_INDEX_EXPLAIN[key]}）")
        lines.append(f"  - 分项: {parts_txt}")
        lines += [f"  - {e}" for e in (it.get("evidence") or [])]
    lines += [
        "",
        "## 行为状态判定",
    ]
    lines += [f"- {e}" for e in (r.get("state_evidence") or [])]
    lines += [
        "",
        "## 逆向信号（极端=反向）",
    ]
    ct = r.get("contrarian") or {}
    if ct.get("trigger"):
        lines.append(f"- 触发: **{ct['trigger']}** → **{ct['direction']}**（强度 {ct['intensity']:.0%}）")
        lines.append(f"- 机制: {ct.get('reason')}")
    else:
        lines.append("- 无极端信号，未触发逆向（保持中性，不逆势抢跑）")
    if r.get("anchor_note"):
        lines.append(f"- 锚定提示: {r['anchor_note']}")
    lines += [
        "",
        "## 证据链",
    ]
    lines += [f"- {e}" for e in r.get("evidence", [])]
    lines += [
        "",
        "## RAG 方法论依据（knowledge_rag）",
    ]
    for item in (r.get("rag") or []):
        if "note" in item:
            lines.append(f"- {item['note']}")
        else:
            lines.append(f"- [{item.get('cat','')}] {item.get('source','')} (score={item.get('score')}): "
                         f"{(item.get('text') or '')[:80]}")
    lines += [
        "",
        "## 数据状态",
    ]
    if r.get("data_status"):
        for k, v in r["data_status"].items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- 全部数据源可用")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="体系10 行为金融系统")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD（默认今天）")
    ap.add_argument("--report", action="store_true", help="写入 generated/behavior_report_{date}.md")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--out-dir", default=None, help="报告输出目录（默认 仓库 generated/）")
    args = ap.parse_args()

    bs = BehaviorSystem(out_dir=args.out_dir)
    r = bs.detect(args.date)
    if args.report:
        path = bs._write_report(r, args.out_dir)
        msg = f"[behavior_system] 报告已写入: {path}"
        (print if not args.json else sys.stderr.write)(msg + chr(10))
    out = r if args.json else bs._view(r)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
if __name__ == "__main__":
    main()
