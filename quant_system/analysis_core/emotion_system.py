"""
emotion_system — 体系4 情绪周期系统（V11）

方法论借鉴（skills/，先读 skill 再写）:
  internal-order-six-stage-cycle   内部订单六阶段循环 → 情绪完整周期:
                                   吸筹(accumulate)→上涨(markup)→派发(distribute)
                                   →下跌(decline)→恐慌(panic)→筑底(bottom)
  a-share-market-state-monitor     五层看盘框架: 趋势/宽度/情绪/资金/风险定价
  behavioral-finance-anomalies     行为金融: 追涨杀跌/羊群效应/过度自信 → 极端=反向

输入（data_warehouse 现有 parquet，路径与列已确认）:
  data_warehouse/market/fusion.parquet        融合温度 0-100（date/temperature/tag/emotion_stage）
  data_warehouse/market/zt_daily_stats.parquet 连板天梯聚合（date/zt_cnt/dt_cnt/max_board/
                                               zb_rate/premium/jr1/ladder_json...）
  data_warehouse/market/zt_pool_em_daily.parquet 涨停池明细（东财，含 board_count/industry/seal_fund）
  data_warehouse/market/zt_pool_history.parquet   涨停池历史长表（K线重建）
  data_warehouse/market/a_high_low.parquet   创新高/新低家数（date/high20/low20/high60/low60...）
  generated/a_share_data/*-meta.json         真实涨跌家数（breadth: 上涨/下跌/平盘，剔除ST口径）

检测:
  1. 情绪温度状态: <30冰点 / 30-45低迷 / 45-60中性 / 60-75活跃 / 75-90亢奋 / >90疯狂
  2. 六阶段识别: 温度序列20日斜率 + 涨停家数/连板高度/炸板率/跌停家数/晋级率/溢价
  3. 结构指标: 涨停/跌停比、连板高度(最高板)、炸板率、涨跌家数比
  4. 极端信号: 温度>90 或 <20 且结构指标确认 → 反转预警（行为金融: 极端=反向）
  5. 综合: 阶段×极端 → 看多/看空/中性 + confidence

输出:
  generated/emotion_report_{date}.md
统一接口:
  EmotionSystem().detect(date) / report(date) / view(date)
  view() 返回 {agent:'情绪周期', signal, confidence, evidence}

防御: 任一数据源缺失 → 跳过标注 data_status，不中断。

用法:
  python3 -m quant_system.analysis_core.emotion_system --date 2026-08-10
  python3 -m quant_system.analysis_core.emotion_system --date 2026-08-10 --report
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

# 离线优先：sentence-transformers 在本环境不可下载模型 → 让 knowledge_rag 立即走关键词兜底
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
    signal_view,
    
)
from quant_system.analysis_core.fusion import read_fusion_latest  # noqa: E402
from quant_system.analysis_core.rag_explain import rag_explain  # noqa: E402

CST = timezone(timedelta(hours=8))
OUT_DIR = Path(__file__).resolve().parent.parent / "generated"

A_SHARE_META_DIR = ROOT / "generated" / "a_share_data"  # 真实涨跌家数（周度拉取）
KLINE_DIR = ROOT / "data_warehouse" / "kline"
A_HIGH_LOW = MARKET_DIR / "a_high_low.parquet"
FUSION = MARKET_DIR / "fusion.parquet"

RAG_QUERY = "情绪周期 六阶段 涨跌停 打板"
RAG_K = 3

# ── 温度分档 ───────────────────────────────────────────────
TEMP_BANDS = [
    (float("-inf"), 30, "ice", "冰点"),
    (30, 45, "low", "低迷"),
    (45, 60, "neutral", "中性"),
    (60, 75, "active", "活跃"),
    (75, 90, "excited", "亢奋"),
    (90, float("inf"), "crazy", "疯狂"),
]

# ── 六阶段（吸筹→上涨→派发→下跌→恐慌→筑底）────────────────
STAGES = ["accumulate", "markup", "distribute", "decline", "panic", "bottom"]
STAGE_CN = {
    "accumulate": "吸筹",
    "markup": "上涨",
    "distribute": "派发",
    "decline": "下跌",
    "panic": "恐慌",
    "bottom": "筑底",
}
STAGE_SIGN = {  # 阶段默认方向（极端信号可反转）
    "accumulate": "看多",
    "markup": "看多",
    "distribute": "看空",
    "decline": "看空",
    "panic": "看空",
    "bottom": "看多",
}

# 阶段打分规则: (特征, 比较符, 阈值, 权重)
# 特征: temp/slope20/slope5/zt_cnt/zt_chg/max_board/mb_chg/zb_rate/dt_cnt/zdr/premium/jr1/adr
STAGE_RULES: dict[str, list[tuple[str, str, float, int]]] = {
    "panic":      [("temp", "lt", 25, 3), ("slope20", "lt", -0.8, 2), ("dt_cnt", "ge", 30, 3),
                   ("zdr", "le", 0.5, 2), ("zb_rate", "ge", 0.45, 2), ("zt_cnt", "le", 30, 1)],
    "decline":    [("temp", "lt", 45, 2), ("slope5", "lt", -1.0, 2), ("zt_chg", "le", -15, 2),
                   ("mb_chg", "le", -2, 1), ("dt_cnt", "ge", 10, 1), ("zb_rate", "ge", 0.35, 1),
                   ("premium", "lt", 0, 1)],
    "distribute": [("temp", "ge", 45, 2), ("temp", "lt", 85, 1), ("slope5", "lt", -1.0, 2),
                   ("zb_rate", "ge", 0.28, 2), ("premium", "lt", 0, 2), ("jr1", "lt", 0.10, 2),
                   ("zdr", "lt", 10, 1), ("zt_chg", "le", -5, 1)],
    "bottom":     [("temp", "ge", 20, 2), ("temp", "lt", 45, 2), ("slope5", "ge", -0.5, 2),
                   ("dt_cnt", "lt", 10, 1), ("zb_rate", "le", 0.30, 1), ("zt_chg", "ge", 0, 1),
                   ("zdr", "ge", 1, 1)],
    "accumulate": [("temp", "ge", 35, 2), ("temp", "lt", 65, 2), ("slope20", "gt", 0, 2),
                   ("zt_cnt", "le", 80, 1), ("max_board", "ge", 2, 1), ("zb_rate", "le", 0.30, 1),
                   ("premium", "ge", 0, 1), ("zdr", "ge", 1, 1)],
    "markup":     [("temp", "ge", 55, 2), ("slope20", "gt", 0.15, 2), ("zt_cnt", "ge", 55, 1),
                   ("max_board", "ge", 3, 1), ("zdr", "ge", 1.5, 1), ("zb_rate", "le", 0.25, 1),
                   ("premium", "ge", 0, 1)],
}
STAGE_MAX = {k: sum(w for _, _, _, w in v) for k, v in STAGE_RULES.items()}

# 阶段硬门槛（不满足直接不参与打分）
STAGE_GATES: dict[str, list[tuple[str, str, float]]] = {
    "panic":      [("temp", "lt", 40)],
    "decline":    [("zt_chg", "lt", 0)],
    "distribute": [("temp", "ge", 45)],
    "bottom":     [("temp", "lt", 45)],
    "accumulate": [("temp", "lt", 65)],
    "markup":     [("temp", "ge", 50)],
}

FEATURE_CN = {
    "temp": "温度", "slope20": "20日斜率", "slope5": "5日斜率",
    "zt_cnt": "涨停家数", "zt_chg": "涨停家数环比", "max_board": "最高板",
    "mb_chg": "最高板环比", "zb_rate": "炸板率", "dt_cnt": "跌停家数",
    "zdr": "涨停/跌停比", "premium": "昨日涨停溢价", "jr1": "1进2晋级率",
    "adr": "涨跌家数比",
}
OP_CN = {"ge": "≥", "gt": ">", "le": "≤", "lt": "<"}


def _isna(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x)


# ════════════════════════════════════════════════════════════
# 数据加载（防御: 缺失返回 None 并标注）
# ════════════════════════════════════════════════════════════

def _load_fusion(date: str) -> tuple[float | None, str | None, int | None]:
    """fusion.parquet 最新 ≤date 的温度。返回 (temp, 实际日期, 落后交易日天数)。"""
    row = read_fusion_latest(date, ["temperature"])
    if row is None:
        return None, None, None
    temp = num(row.get("temperature"))
    if temp is None:
        return None, None, None
    as_of = row["date"]
    lag = max(0, (pd.Timestamp(date) - pd.Timestamp(as_of)).days)
    return temp, as_of, int(lag)


def _load_zt_pool(date: str) -> pd.DataFrame | None:
    """涨停池明细（东财日更优先，K线重建兜底）— 用于报告展示最高板个股。"""
    for path, src in ((ZT_EM_DAILY, "em"), (ZT_HISTORY, "kl")):
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            day = df[df["date"] == pd.Timestamp(date)]
            if len(day):
                day = day.copy()
                day["_src"] = src
                return day
        except Exception as e:  # noqa: BLE001
            logging.getLogger(__name__).error(f"[emotion_system] 操作失败: {e}", exc_info=True)
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
            "ratio": round((int(r.get("high20") or 0) + 1) / (int(r.get("low20") or 0) + 1), 3),
        }
    except Exception:  # noqa: BLE001
        return None


def _load_ad_counts(date: str) -> dict | None:
    """真实涨跌家数: generated/a_share_data/{YYYYMMDD}-meta.json（周度拉取，剔除ST口径）。"""
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
                logging.getLogger(__name__).error(f"[emotion_system] 操作失败: {e}", exc_info=True)
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
        flat = num(bread.get("平盘"))
        if up is None or down is None:
            return None
        return {
            "as_of": d.strftime("%Y-%m-%d"),
            "up": int(up), "down": int(down),
            "flat": int(flat or 0),
            "total": int(bread.get("股票数") or (up + down + (flat or 0))),
            "ratio": round(up / max(down, 1), 3),
            "advance_pct": round(up / max(up + down + (flat or 0), 1), 4),
        }
    except Exception:  # noqa: BLE001
        return None


# ════════════════════════════════════════════════════════════
# 温度序列: 代理温度（连板天梯特征）→ fusion 锚定 → fusion 覆盖
# ════════════════════════════════════════════════════════════

def _proxy_temp(row: pd.Series) -> float:
    """代理温度 0-100（透明公式）: 涨停家数/最高板/炸板率/跌停家数/晋级率/溢价。"""
    def _c(x, lo=0.0, hi=1.0):
        v = num(x, np.nan)
        if _isna(v):
            return 0.5
        return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))

    zt = _c(row.get("zt_cnt"), 0, 100)
    mb = _c(row.get("max_board"), 0, 7)
    zb = 1.0 - _c(row.get("zb_rate"), 0, 0.6)          # 炸板率越高 → 温度越低
    dt = 1.0 - _c(row.get("dt_cnt"), 0, 40)            # 跌停越多 → 温度越低
    jr = _c(row.get("jr1") if not _isna(row.get("jr1")) else row.get("jr1_t3"), 0, 0.5)
    prem = num(row.get("premium"))
    if _isna(prem):
        prem = num(row.get("premium_t3"), 0.0) or 0.0
    prem_norm = (float(np.clip(prem, -5, 5)) / 5 + 1) / 2
    proxy = (50
             + 22 * (2 * zt - 1)
             + 10 * (2 * mb - 1)
             + 8 * (2 * zb - 1)
             + 6 * (2 * dt - 1)
             + 4 * (2 * jr - 1)
             + 4 * (2 * prem_norm - 1))
    return round(float(np.clip(proxy, 5, 95)), 1)


def _global_fusion_offset() -> float:
    """全局锚定偏移 = median(fusion实际温度 - 代理温度)，全量历史口径，尺度全局一致。"""
    try:
        if not FUSION.exists() or not ZT_DAILY_STATS.exists():
            return 0.0
        fu = pd.read_parquet(FUSION, columns=["date", "temperature"])
        fu["date"] = pd.to_datetime(fu["date"], errors="coerce")
        fu = fu.dropna(subset=["date", "temperature"])
        if fu.empty:
            return 0.0
        full = pd.read_parquet(ZT_DAILY_STATS)
        full["date"] = pd.to_datetime(full["date"], errors="coerce")
        full["temp_proxy"] = full.apply(_proxy_temp, axis=1)
        merged = full.merge(fu, on="date", how="inner")
        if merged.empty:
            return 0.0
        diff = (merged["temperature"] - merged["temp_proxy"]).to_numpy(dtype=float)
        return float(np.median(diff))
    except Exception:  # noqa: BLE001
        return 0.0


def _temperature_series(stats: pd.DataFrame, offset: float = 0.0) -> pd.DataFrame:
    """返回带 temp/slope20/slope5 的 df；temp = fusion 实际温度(窗口内有) + 全局锚定代理温度。"""
    df = stats.copy()
    df["temp_proxy"] = df.apply(_proxy_temp, axis=1)

    # fusion 实际温度覆盖（窗口内）
    fusion_rows: list[tuple[str, float]] = []
    if FUSION.exists():
        try:
            fu = pd.read_parquet(FUSION, columns=["date", "temperature"])
            fu["date"] = pd.to_datetime(fu["date"], errors="coerce")
            fu = fu.dropna(subset=["date", "temperature"])
            for _, r in fu.iterrows():
                fusion_rows.append((r["date"], float(r["temperature"])))
        except Exception:  # noqa: BLE001
            fusion_rows = []

    fu_map = {pd.Timestamp(d): v for d, v in fusion_rows}
    df["temp_fusion"] = df["date"].map(fu_map)

    df["zt_chg"] = df["zt_cnt"].diff()          # 涨停家数环比
    df["mb_chg"] = df["max_board"].diff()        # 最高板环比
    df["temp_adj"] = (df["temp_proxy"] + offset).clip(5, 95).round(1)
    df["temp"] = df["temp_fusion"].fillna(df["temp_adj"]).round(1)

    def _slope(s: pd.Series, n: int) -> float:
        vals = s.tail(n).to_numpy(dtype=float)
        if len(vals) < 3:
            return 0.0
        return float(np.polyfit(np.arange(len(vals)), vals, 1)[0])

    df["slope20"] = df["temp"].rolling(20).apply(lambda s: _slope(s, 20), raw=False)
    df["slope5"] = df["temp"].rolling(5).apply(lambda s: _slope(s, 5), raw=False)
    return df


# ════════════════════════════════════════════════════════════
# 六阶段识别
# ════════════════════════════════════════════════════════════

def _ok(row: pd.Series, col: str, op: str, bound: float) -> bool:
    v = row.get(col)
    if v is None or _isna(v):
        return False
    try:
        v = float(v)
    except (TypeError, ValueError):
        return False
    return {"ge": v >= bound, "gt": v > bound, "le": v <= bound, "lt": v < bound}[op]


def _stage_score(row: pd.Series, stage: str) -> tuple[int, int, list[str]]:
    hits, maxs, ev = 0, STAGE_MAX[stage], []
    for col, op, bound, w in STAGE_RULES[stage]:
        if _ok(row, col, op, bound):
            hits += w
            ev.append(f"{FEATURE_CN.get(col, col)}{OP_CN[op]}{bound:g}")
    return hits, maxs, ev


def _detect_stage(row: pd.Series) -> tuple[str, float, list[str]]:
    """六阶段判定: 强特征覆盖（派发/下跌/恐慌/筑底）→ 加权打分兜底。"""
    # ① 恐慌: 跌停潮 / 冰点+跌停扩散
    if (not _isna(row.get("dt_cnt")) and row["dt_cnt"] >= 100) or \
       (_ok(row, "temp", "lt", 20) and (row.get("dt_cnt") or 0) >= 20):
        return "panic", 0.90, [f"跌停{row.get('dt_cnt'):.0f}家(跌停潮)", f"温度{row.get('temp'):.0f}<20 冰点恐慌"]
    # ② 派发: 高位分歧（亏钱效应+晋级率崩+短期急冷） / 亢奋放量炸板 / 顶部天量跳水
    if (row.get("premium") is not None and not _isna(row.get("premium"))
            and row.get("jr1") is not None and not _isna(row.get("jr1"))
            and _ok(row, "premium", "lt", 0) and _ok(row, "jr1", "lt", 0.10)
            and _ok(row, "slope5", "lt", -1.0)) or \
       (_ok(row, "temp", "ge", 85) and _ok(row, "zb_rate", "ge", 0.25)) or \
       (_ok(row, "temp", "ge", 80) and _ok(row, "zt_chg", "le", -30)):
        ev = [f"昨日涨停溢价{row.get('premium'):+.2f}%(亏钱效应)" if not _isna(row.get("premium")) else f"温度{row.get('temp'):.0f}天量跳水",
              f"1进2晋级率{row.get('jr1'):.0%}" if not _isna(row.get("jr1")) else "",
              f"5日温度斜率{row.get('slope5'):+.1f}/日"]
        return "distribute", 0.78, [e for e in ev if e]
    # ③ 下跌: 涨停急杀（数量崩塌+短期急冷）
    if _ok(row, "zt_chg", "le", -40) and (_ok(row, "slope5", "le", -1.0) or _ok(row, "temp", "le", 60)):
        return "decline", 0.82, [f"涨停家数环比{row.get('zt_chg'):+.0f}家", f"最高板{row.get('max_board'):.0f}",
                                 f"5日温度斜率{row.get('slope5'):+.1f}/日"]
    # ④ 筑底: 低位企稳（急跌后反抽、跌停收敛）
    if _ok(row, "temp", "le", 40) and _ok(row, "slope5", "ge", -0.5) and _ok(row, "dt_cnt", "lt", 10) \
            and _ok(row, "zb_rate", "le", 0.30) and (row.get("zt_chg") or 0) >= 0:
        return "bottom", 0.70, [f"温度{row.get('temp'):.0f}低位企稳", f"跌停{row.get('dt_cnt'):.0f}家收敛",
                                f"5日斜率{row.get('slope5'):+.1f}拐头"]
    # ⑤ 打分兜底（带硬门槛）
    sc: dict[str, int] = {}
    for st in STAGES:
        gated = any(not _ok(row, col, op, b) for col, op, b in STAGE_GATES[st])
        if gated:
            sc[st] = -1
        else:
            sc[st], _, _ = _stage_score(row, st)
    top = max(STAGES, key=lambda s: sc[s])
    top_score, _, top_ev = _stage_score(row, top)
    second = sorted((sc[s] for s in STAGES if s != top), reverse=True)[0]
    if top_score <= 0:
        return "accumulate", 0.30, ["特征不足，默认吸筹观察（低置信）"]
    conf = min(0.95, 0.45 + 0.5 * top_score / STAGE_MAX[top])
    if top_score - second <= 2 and top_score < 4:  # 候选接近 → 降置信
        conf = min(conf, 0.45)
    return top, round(conf, 2), top_ev


# ════════════════════════════════════════════════════════════
# 极端信号（行为金融: 极端=反向）
# ════════════════════════════════════════════════════════════

def _detect_extreme(row: pd.Series) -> dict:
    """温度>90 或 <20 且结构指标确认 → 反转预警。"""
    temp = row.get("temp")
    zdr = row.get("zdr")
    zb = row.get("zb_rate")
    mb = row.get("max_board")
    dt = row.get("dt_cnt")
    adr = row.get("adr")
    zt = row.get("zt_cnt")
    out = {"is_extreme": False, "direction": None, "band": None,
           "reversal": False, "bias": None, "confirm": [], "note": None}

    if temp is not None and not _isna(temp) and temp > 90:
        confirms = []
        if zdr is not None and not _isna(zdr) and zdr >= 2:
            confirms.append(f"涨停/跌停比{zdr:.1f}≥2")
        if zb is not None and not _isna(zb) and zb >= 0.25:
            confirms.append(f"炸板率{zb:.0%}≥25%")
        if mb is not None and not _isna(mb) and mb >= 6:
            confirms.append(f"最高{mb:.0f}板≥6")
        if not confirms:
            return out
        out.update({
            "is_extreme": True, "direction": "high", "band": "疯狂",
            "reversal": True, "bias": "过度自信/羊群效应(追涨杀跌)",
            "confirm": confirms,
            "note": f"温度{temp:.0f}>90 进入疯狂区，结构指标确认后按行为金融'极端=反向' → 反转预警看空",
        })
        return out

    if temp is not None and not _isna(temp) and temp < 20:
        confirms = []
        if dt is not None and not _isna(dt) and dt >= 10:
            confirms.append(f"跌停{dt:.0f}家≥10")
        if adr is not None and not _isna(adr) and adr <= 0.3:
            confirms.append(f"涨跌家数比{adr:.2f}≤0.3")
        if zt is not None and not _isna(zt) and zt <= 25:
            confirms.append(f"涨停仅{zt:.0f}家≤25")
        if not confirms:
            return out
        out.update({
            "is_extreme": True, "direction": "low", "band": "冰点",
            "reversal": True, "bias": "恐慌过度/损失厌恶(杀跌)",
            "confirm": confirms,
            "note": f"温度{temp:.0f}<20 进入冰点区，结构指标确认后按行为金融'极端=反向' → 反转预警看多(超跌修复预期)",
        })
    return out


def _combine_signal(stage: str, conf: float, extreme: dict, temp) -> tuple[str, float, list[str]]:
    """阶段×极端 → 看多/看空/中性 + confidence。"""
    ev: list[str] = []
    if extreme.get("is_extreme"):
        if extreme["direction"] == "high":
            sig, c = "看空", round(min(0.95, 0.62 + conf * 0.2), 2)
            ev.append(f"极端亢奋反转预警（{extreme['bias']}）→ 看空")
        else:
            sig, c = "看多", round(min(0.85, 0.55 + conf * 0.2), 2)
            ev.append(f"极端冰点反转预警（{extreme['bias']}）→ 看多(左侧超跌修复预期)")
        return sig, c, ev

    base = STAGE_SIGN[stage]
    ev.append(f"阶段{STAGE_CN[stage]} → 默认{base}")
    # 过热折价: 亢奋区(85-90)追涨容错率低，做多置信打折并提示
    if temp is not None and not _isna(temp) and temp >= 85 and base == "看多":
        sig, c = "看多", round(min(conf, 0.55), 2)
        ev.append(f"温度{temp:.0f}亢奋区(75-90)，追涨容错率低 → 置信上限0.55")
    elif base == "看多":
        sig, c = "看多", round(min(conf, 0.85), 2)
    elif base == "看空":
        sig, c = "看空", round(min(conf, 0.85), 2)
    else:  # 理论不可达
        sig, c = "中性", round(conf, 2)
    if conf < 0.45:
        ev.append("特征置信低，方向仅作参考")
    return sig, c, ev


# ════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════
# RAG 解释（公共块 rag_explain：子进程检索 + 本地关键词兜底 + 7天缓存）
# ════════════════════════════════════════════════════════════

RAG_CACHE = OUT_DIR / "rag_emotion_cache.json"  # 静态查询结果缓存（子进程检索一次后即读缓存）


def _rag_explain() -> list[dict]:
    """RAG 依据: knowledge_rag.search（子进程）→ 结果缓存 → 本地关键词兜底。"""
    return rag_explain(RAG_QUERY, RAG_K, RAG_CACHE, ROOT)


# 统一接口
# ════════════════════════════════════════════════════════════

class EmotionSystem:
    """情绪周期系统 — detect/report/view 统一接口。"""

    def __init__(self, out_dir: Path | str | None = None):
        self.out_dir = Path(out_dir) if out_dir else OUT_DIR

    # ── detect ─────────────────────────────────────────────
    def detect(self, date: str | None = None) -> dict:
        date = norm_date(date)
        data_status: dict[str, str] = {}

        # 1. 数据加载
        fusion_temp, fusion_as_of, fusion_lag = _load_fusion(date)
        stats = load_zt_stats(ZT_DAILY_STATS, date)
        if fusion_temp is None:
            data_status["fusion温度"] = f"缺失（≤{date} 无 fusion.parquet 记录）→ 用代理温度"
        if stats is None or stats.empty:
            data_status["连板天梯"] = f"缺失 {ZT_DAILY_STATS} → 跳过六阶段判定"
            return {
                "date": date, "agent": "情绪周期", "signal": "中性",
                "confidence": 0.0, "status": "degraded",
                "evidence": [f"连板天梯数据缺失，无法判定情绪阶段: {ZT_DAILY_STATS}"],
                "data_status": data_status,
                "temperature": fusion_temp, "temperature_as_of": fusion_as_of,
                "stage": None, "stage_cn": "未知", "extreme": None,
            }
        if fusion_temp is not None and (fusion_lag or 0) > 3:
            data_status["fusion温度"] = f"落后 {fusion_lag} 日(as_of={fusion_as_of}) → 已标注不静默使用"
        stats_lag = max(0, (pd.Timestamp(date) - stats["date"].iloc[-1]).days)
        if stats_lag > 7:
            data_status["连板天梯"] = (f"落后 {stats_lag} 日(as_of={stats['date'].iloc[-1].date()}) "
                                       "→ 结构指标按最新可用交易日判定")

        # 2. 温度序列 + 斜率
        ser = _temperature_series(stats, offset=_global_fusion_offset())
        row = ser.iloc[-1].copy()
        temp = num(row["temp"])
        temp_as_of = row["date"].strftime("%Y-%m-%d")
        fusion_days = int(ser["temp_fusion"].notna().sum()) if "temp_fusion" in ser.columns else 0
        if fusion_days == 0 and fusion_temp is not None:
            # fusion 实际温度日期不在 stats 尾部 → 用最新 fusion 温度并标注
            temp = fusion_temp
            row["temp"] = fusion_temp
            data_status["fusion温度"] = f"fusion 最新记录 {fusion_as_of} 不在天梯窗口 → 温度取 fusion 最新值"
        temp_src = f"fusion({fusion_days}日锚定)" if fusion_days else "代理温度(无fusion历史)"
        if temp is None:
            temp = 50.0
            row["temp"] = 50.0

        # 3. 结构指标
        zt_cnt = num(row.get("zt_cnt")); dt_cnt = num(row.get("dt_cnt"))
        zb_rate = num(row.get("zb_rate")); max_board = num(row.get("max_board"))
        premium = num(row.get("premium")); jr1 = num(row.get("jr1"))
        zdr = round(zt_cnt / max(dt_cnt, 1), 3) if (zt_cnt is not None and dt_cnt is not None) else None
        hl = _load_a_high_low(date)
        ad = _load_ad_counts(date)

        adr: float | None = None
        if ad is not None:
            adr = ad["ratio"]
            data_status["涨跌家数"] = f"a_share_data {ad['as_of']}（真实口径: 涨{ad['up']}/跌{ad['down']}）"
        elif hl is not None:
            adr = hl["ratio"]
            data_status["涨跌家数"] = f"缺失真实涨跌家数 → 用新高/新低比代理(a_high_low {hl['as_of']})"
        else:
            data_status["涨跌家数"] = "缺失（无 a_share_data meta / a_high_low）→ 涨跌家数比跳过"

        row["zdr"] = zdr
        row["adr"] = adr
        if zdr is None:
            data_status["涨停/跌停比"] = "缺失（跌停家数无数据）→ 跳过"

        # 4. 六阶段
        stage, stage_conf, stage_ev = _detect_stage(row)

        # 5. 极端信号
        extreme = _detect_extreme(row)

        # 6. 综合信号
        signal, conf, sig_ev = _combine_signal(stage, stage_conf, extreme, temp)

        # 7. 涨停池亮点（报告用）
        pool = _load_zt_pool(temp_as_of)
        pool_top: list[dict] = []
        if pool is not None:
            zt = pool[pool.get("is_zt", pd.Series(False, index=pool.index)).fillna(False)]
            if len(zt):
                top = zt.sort_values("board_count", ascending=False).head(5)
                for _, r in top.iterrows():
                    pool_top.append({
                        "code": str(r.get("code", "")), "name": str(r.get("name", "")),
                        "board": int(r.get("board_count") or 0),
                        "industry": str(r.get("industry", "") or "")[:12],
                    })
        else:
            data_status["涨停池"] = f"缺失 {ZT_EM_DAILY}/{ZT_HISTORY} → 跳过个股亮点"

        # 8. RAG 依据
        rag = _rag_explain()

        # 9. 行为金融注解
        behavior: list[str] = []
        if extreme.get("is_extreme"):
            behavior.append(f"极端{extreme['band']} → {extreme['bias']}，按'极端=反向'处理")
        else:
            if zdr is not None and zdr >= 15:
                behavior.append(f"涨停/跌停比{zdr:.0f} 极高 → 追涨/羊群效应显著，警惕情绪反转")
            if (max_board or 0) >= 7 or (temp is not None and temp >= 80):
                behavior.append(f"最高{max_board:.0f}板/温度{temp:.0f} → 过度自信升温区，注意分歧风险")
            if stage in ("panic", "decline") and dt_cnt is not None and dt_cnt >= 10:
                behavior.append(f"跌停{dt_cnt:.0f}家 → 损失厌恶主导的恐慌抛售，追跌需谨慎")
            if stage in ("accumulate", "bottom"):
                behavior.append("低位吸筹/筑底 → 避免因'锚定近期恐慌'错过修复，逆向布局需结构确认")

        evidence = [
            f"温度{temp:.0f}({_temp_band_cn(temp)}) 20日斜率{num(row.get('slope20'), 0):+.2f}/日 5日斜率{num(row.get('slope5'), 0):+.2f}/日",
            f"六阶段={STAGE_CN[stage]} 置信{stage_conf:.2f} | 证据: {'; '.join(stage_ev) if stage_ev else '打分兜底'}",
            f"结构: 涨停{zt_cnt}家 跌停{dt_cnt}家 涨停/跌停比{zdr if zdr is not None else '—'} "
            f"最高{max_board}板 炸板率{(zb_rate or 0):.0%} 溢价{premium if premium is not None else float('nan'):+.2f}%",
            *sig_ev,
        ]
        if extreme.get("is_extreme"):
            evidence.append(f"极端信号: {'; '.join(extreme['confirm'])} → {extreme['note']}")

        return {
            "date": date,
            "data_date": temp_as_of,
            "agent": "情绪周期",
            "temperature": round(float(temp), 1),
            "temperature_source": temp_src,
            "temperature_as_of": fusion_as_of,
            "temp_band": _temp_band_cn(temp),
            "slope20": round(num(row.get("slope20"), 0), 3),
            "slope5": round(num(row.get("slope5"), 0), 3),
            "stage": stage,
            "stage_cn": STAGE_CN[stage],
            "stage_confidence": round(float(stage_conf), 2),
            "stage_evidence": stage_ev,
            "structure": {
                "zt_cnt": None if zt_cnt is None else int(zt_cnt),
                "dt_cnt": None if dt_cnt is None else int(dt_cnt),
                "zdr": zdr,
                "max_board": None if max_board is None else int(max_board),
                "zb_rate": None if zb_rate is None else round(float(zb_rate), 4),
                "premium": premium,
                "jr1": jr1,
                "adr": adr,
                "new_high20": (hl or {}).get("high20"),
                "new_low20": (hl or {}).get("low20"),
            },
            "ad_source": ad,
            "extreme": extreme,
            "behavior": behavior,
            "signal": signal,
            "confidence": round(float(conf), 2),
            "evidence": evidence,
            "pool_top": pool_top,
            "rag": rag,
            "data_status": data_status,
            "status": "ok" if not data_status else "degraded",
        }

    # ── report ─────────────────────────────────────────────
    def report(self, date: str | None = None, out_dir: Path | str | None = None) -> Path:
        date = norm_date(date)
        out = self.out_dir if out_dir is None else Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        r = self.detect(date)
        path = out / f"emotion_report_{r['data_date']}.md"
        path.write_text(_render_markdown(r), encoding="utf-8")
        return path

    # ── view ───────────────────────────────────────────────
    def view(self, date: str | None = None) -> dict:
        """多智能体视图: {agent, signal, view, confidence, evidence, weight, status, detail}。"""
        r = self.detect(date)
        return {
            "agent": "情绪周期",
            "signal": r["signal"],
            "view": signal_view(r["signal"]),
            "confidence": r["confidence"],
            "evidence": r["evidence"],
            "weight": 1.0,
            "status": r["status"],
            "detail": {
                "date": r["data_date"],
                "temperature": r["temperature"],
                "temp_band": r["temp_band"],
                "stage": r["stage_cn"],
                "stage_confidence": r["stage_confidence"],
                "extreme": r["extreme"],
            },
        }


def _temp_band_cn(temp: float) -> str:
    for lo, hi, _, cn in TEMP_BANDS:
        if lo <= temp < hi or (hi == float("inf") and temp >= lo):
            return cn
    return "中性"


def _render_markdown(r: dict) -> str:
    s = r["structure"]
    ext = r["extreme"]
    lines = [
        f"# 情绪周期系统报告 — {r['date']}",
        "",
        f"- 结论: **{r['signal']}** | 置信 {r['confidence']:.0%} | 六阶段: **{r['stage_cn']}**"
        f"（阶段置信 {r['stage_confidence']:.0%}）",
        f"- 情绪温度: **{r['temperature']:.0f}（{r['temp_band']}）** | 数据源: {r['temperature_source']}",
        f"- 温度斜率: 20日 {r['slope20']:+.2f}/日 | 5日 {r['slope5']:+.2f}/日",
        "",
        "## 结构指标",
        f"- 涨停家数: {s['zt_cnt']} | 跌停家数: {s['dt_cnt']} | 涨停/跌停比: {fmt(s['zdr'], 2)}",
        f"- 连板高度(最高板): {s['max_board']} | 炸板率: {fmt(s['zb_rate'] * 100, 1) + '%' if s['zb_rate'] is not None else '—'}",
        f"- 涨跌家数比: {fmt(s['adr'], 2)}"
        + (f"（{r['ad_source']['as_of']} 真实口径 涨{r['ad_source']['up']}/跌{r['ad_source']['down']}）"
           if r.get("ad_source") else "（缺失，跳过标注）"),
        f"- 新高20/新低20: {s['new_high20']}/{s['new_low20']} | 昨日涨停溢价: {fmt(s['premium'], 2)}%"
        f" | 1进2晋级率: {fmt(s['jr1'] * 100, 1) + '%' if s['jr1'] is not None else '—'}",
        "",
        "## 六阶段证据",
        f"- 判定: {STAGE_CN[r['stage']]}（{'; '.join(r['stage_evidence']) if r['stage_evidence'] else '打分兜底'}）",
        "",
        "## 极端信号（行为金融: 极端=反向）",
    ]
    if ext.get("is_extreme"):
        lines.append(f"- **{ext['band']}极端触发** | 方向: {'过热→反转看空' if ext['direction']=='high' else '冰点→反转看多'}")
        lines.append(f"- 确认指标: {'; '.join(ext['confirm'])}")
        lines.append(f"- 行为机制: {ext['bias']}")
        lines.append(f"- {ext['note']}")
    else:
        lines.append("- 无极端信号（温度处于非极端区间，不触发反转预警）")
    lines += [
        "",
        "## 行为金融注解",
    ]
    lines += [f"- {b}" for b in (r.get("behavior") or ["无显著行为偏置"])]
    lines += [
        "",
        "## 涨停池亮点（最高板Top5）",
    ]
    if r.get("pool_top"):
        for p in r["pool_top"]:
            lines.append(f"- {p['name']}({p['code']}) {p['board']}连板 | {p['industry']}")
    else:
        lines.append("- 涨停池数据缺失，跳过")
    lines += [
        "",
        "## RAG 方法论依据（knowledge_rag）",
    ]
    for item in (r.get("rag") or []):
        if "note" in item:
            lines.append(f"- {item['note']}")
        else:
            lines.append(f"- [{item.get('cat','')}] {item.get('source','')} (score={item.get('score')}): {item.get('text','')[:80]}")
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
    ap = argparse.ArgumentParser(description="体系4 情绪周期系统")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD（默认今天）")
    ap.add_argument("--report", action="store_true", help="写入 generated/emotion_report_{date}.md")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--out-dir", default=None, help="报告输出目录（默认 仓库 generated/）")
    args = ap.parse_args()

    es = EmotionSystem(out_dir=args.out_dir)
    if args.report:
        path = es.report(args.date)
        print(f"[emotion_system] 报告已写入: {path}")
    r = es.detect(args.date)
    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
    else:
        print(json.dumps(es.view(args.date), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
