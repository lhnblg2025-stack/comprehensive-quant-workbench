"""
macro_overseas — 宏观风向标（海外资产 → A股映射）

用户宏观论点落地:
  美债收益率高 → 信用危机担忧 → 黄金上涨
  金油同涨 = 滞胀/避险信号

借鉴 skills:
  - animal-spirits-risk-forecasting: 恐慌/避险情绪作为危机传导机制
    （美债高利率→信用/流动性担忧→避险买金→黄金走强；防御资产受追捧）
  - commodity-cyclical-valuation: 大宗商品/周期资产的驱动识别
    （金/油分别代表避险需求与经济需求；金油同涨=滞胀，金涨油跌=避险不滞胀）

数据源（akshare，本机连不上→ status=unavailable + reason，不伪造）:
  黄金  伦敦金现 XAU:  ak.futures_foreign_hist("XAU")   ←首选（sina外盘日线，含5日历史）
        降级链: COMEX黄金(GC) → 新浪外盘实时(伦敦金/COMEX黄金) → 上海金基准价(SGE)
  原油  布伦特 OIL / WTI(CL): ak.futures_foreign_hist("OIL"/"CL") → 新浪外盘实时
  美债  US 10Y:  ak.bond_zh_us_rate()（东财中美国债收益率，列"美国国债收益率10年"）
  VIX  可选:     新浪外盘实时/日线（本机无 macro_usa_vix，失败跳过）

核心输出（--report → generated/macro_overseas_{date}.md）:
  1. 各资产最新值 + 5日涨跌幅
  2. 信号: gold_up(5日>1.5%) / oil_up(5日>1.5%) / yield_high(10Y>4.5%) / 金油同涨
  3. 宏观模式标签（用户逻辑 + skill）:
     滞胀避险模式 / 利率压制 / 通胀对冲 / 避险不滞胀 / 中性 / 数据不可用
  4. A股映射建议（偏好/规避板块 + 全市场温度建议值，不自动改 fusion）

缓存: generated/macro_overseas_{date}.json，当日重复调用不重复抓（--refresh 强制重抓）

用法:
  python3 -m quant_system.analysis_core.macro_overseas                     # 打印JSON(走缓存)
  python3 -m quant_system.analysis_core.macro_overseas --date 2026-08-11 --report
  python3 -m quant_system.analysis_core.macro_overseas --refresh           # 当日强制重抓
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent  # 仓库根 quant_system（v11 报告统一输出目录）
OUT_DIR = ROOT / "generated"
sys.path.insert(0, str(ROOT))

try:
    import akshare as ak
except Exception as e:  # akshare 缺失/损坏：全部源标注不可用，不崩溃
    ak = None
    AK_IMPORT_ERROR = str(e)
else:
    AK_IMPORT_ERROR = None

CST = timezone(timedelta(hours=8))

# ── 判定阈值（用户给定）─────────────────────────────
GOLD_UP_PCT = 1.5     # 金5日涨跌幅阈值 %
OIL_UP_PCT = 1.5      # 油5日涨跌幅阈值 %
YIELD_HIGH = 4.5      # 美债10Y 阈值 %
TEMP_SUGGESTION_STAGFLATION = -10   # 滞胀避险模式 → 全市场温度建议（仅建议，不改 fusion）

# ── 新浪外盘品种代码 ────────────────────────────────
SINA_GOLD_HIST = [("XAU", "伦敦金现"), ("GC", "COMEX黄金")]
SINA_GOLD_RT = ["伦敦金", "COMEX黄金"]
SINA_OIL_HIST = [("OIL", "布伦特原油"), ("CL", "WTI原油")]
SINA_OIL_RT = ["布伦特原油", "NYMEX原油"]

# ── 宏观模式 → A股映射 ──────────────────────────────
MODE_MAP = {
    "滞胀避险模式": {
        "prefer": ["黄金/贵金属", "资源(油气/有色/煤炭)", "红利大盘价值(银行/电力/公用/高股息)"],
        "avoid": ["科技成长(半导体/AI/消费电子)", "创业板/科创板(高估值杀估值)"],
        "temp_suggestion": TEMP_SUGGESTION_STAGFLATION,
        "note": "利率高企→信用/流动性担忧(animal-spirits 避险传导)→黄金走强；"
                "金油同涨=滞胀风险，资金从长久期成长撤向实物资产与高股息。",
    },
    "利率压制": {
        "prefer": ["价值/红利(银行/公用/高股息)", "现金流稳定的资源龙头"],
        "avoid": ["长久期成长(科技/新能源/创新药，估值承压)"],
        "temp_suggestion": None,
        "note": "仅美债10Y>4.5%：高贴现率压制成长估值，价值相对占优。",
    },
    "通胀对冲": {
        "prefer": ["资源/周期(油气/有色/煤炭/化工)", "黄金(通胀+避险双重属性)"],
        "avoid": ["成本挤压的纯防守消费", "长久期成长(通胀→贴现率上升)"],
        "temp_suggestion": None,
        "note": "金油同涨(无利率压制)：需求+通胀预期升温，商品/周期占优。",
    },
    "避险不滞胀": {
        "prefer": ["黄金/贵金属", "防御(公用/医药/必选消费)", "高股息红利"],
        "avoid": ["强周期(油气/有色/煤炭)", "高Beta成长(资金撤向防御)"],
        "temp_suggestion": None,
        "note": "金涨油跌：避险情绪占主导但无通胀扩散，防御偏好(commodity-cyclical 驱动分化)。",
    },
    "中性": {
        "prefer": ["均衡配置，跟随市场主线"],
        "avoid": ["无明确规避"],
        "temp_suggestion": None,
        "note": "未触发任一宏观模式，海外资产不构成方向性约束。",
    },
    "数据不可用": {
        "prefer": None,
        "avoid": None,
        "temp_suggestion": None,
        "note": "核心海外资产全部不可用，本次不做宏观模式判定，待数据源恢复。",
    },
}


def _now() -> datetime:
    return datetime.now(CST)


def _cache_path(date: str) -> Path:
    return OUT_DIR / f"macro_overseas_{date}.json"


# ────────────────────────────────────────────────────────────
# 数据抓取（每源独立 try/except，失败只记 reason）
# ────────────────────────────────────────────────────────────
def _attempts() -> list[dict]:
    """记录每个源的尝试结果，便于报告数据可用性。"""
    return []


def _series_from_hist(df: pd.DataFrame, name: str) -> tuple[pd.Series | None, str]:
    """把外盘历史K线（sina 数组/中文列两种形态）规整为 close 序列。

    sina futures_foreign_hist 返回 array-of-arrays → 列 0..5(日期/开/高/低/收/量)。
    """
    if df is None or df.empty:
        return None, f"{name}: 空数据"
    if "日期" in df.columns and "收盘" in df.columns:
        date_col, close_col = "日期", "收盘"
    elif 0 in df.columns and 4 in df.columns:
        date_col, close_col = 0, 4
    elif "date" in df.columns and "close" in df.columns:
        date_col, close_col = "date", "close"
    elif "日期" in df.columns:
        date_col = "日期"
        num_cols = [c for c in df.columns if c != "日期"
                    and pd.to_numeric(df[c], errors="coerce").notna().any()]
        if not num_cols:
            return None, f"{name}: 列结构未知，无非数值列 {list(df.columns)[:8]}"
        close_col = next((c for c in num_cols
                          if any(k in str(c) for k in ("伦敦金", "收盘", "价格", "金", "基准价", "收益率"))),
                         num_cols[0])
    else:
        return None, f"{name}: 列结构未知 {list(df.columns)[:8]}"
    dates = pd.to_datetime(df[date_col], errors="coerce")
    close = pd.to_numeric(df[close_col], errors="coerce")
    s = pd.Series(close.values, index=dates).dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if len(s) < 2:
        return None, f"{name}: 有效收盘不足2条"
    return s, ""


def _chg5(s: pd.Series, name: str) -> tuple[float, float, str | None]:
    """5日涨跌幅%: 最新收盘 vs 5个交易日前的收盘。返回 (pct, latest, anchor_date)。"""
    latest = float(s.iloc[-1])
    anchor_idx = max(0, len(s) - 6)
    anchor = float(s.iloc[anchor_idx])
    pct = (latest / anchor - 1.0) * 100.0 if anchor else np.nan
    anchor_date = s.index[anchor_idx].date().isoformat()
    note = None
    if len(s) < 6:
        note = f"{name}: 历史仅{len(s)}条，5日涨跌按可用区间近似"
    return float(pct), latest, anchor_date


def _try(fn, **kw) -> tuple[pd.DataFrame | None, str | None]:
    """执行抓取，返回 (df, error)。"""
    if ak is None:
        return None, f"akshare不可用: {AK_IMPORT_ERROR}"
    try:
        df = fn(**kw)
    except Exception as e:
        return None, f"{fn.__name__}: {type(e).__name__}: {str(e)[:140]}"
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return None, f"{fn.__name__}: 空数据"
    return df, None


def _fetch_gold(attempts: list) -> dict:
    """黄金：伦敦金现(XAU) → COMEX黄金(GC) → 新浪实时 → 上海金基准价。"""
    # 1) 若 akshare 新版存在 macro_usa_london_gold，优先尝试
    if ak is not None and hasattr(ak, "macro_usa_london_gold"):
        df, err = _try(getattr(ak, "macro_usa_london_gold"))
        if err:
            attempts.append({"源": "macro_usa_london_gold", "错误": err})
        elif df is not None:
            s, e2 = _series_from_hist(df, "伦敦金现")
            if e2:
                attempts.append({"源": "macro_usa_london_gold", "错误": e2})
            else:
                pct, latest, anchor = _chg5(s, "伦敦金现")
                return {"name": "伦敦金现", "latest": round(latest, 2), "unit": "美元/盎司",
                        "as_of": s.index[-1].date().isoformat(), "chg5_pct": round(pct, 2),
                        "chg5_from": anchor, "source": "ak.macro_usa_london_gold"}
    # 2) 新浪外盘历史日线 XAU（伦敦金现）
    for code, name in SINA_GOLD_HIST:
        df, err = _try(ak.futures_foreign_hist, symbol=code)
        if err:
            attempts.append({"源": f"futures_foreign_hist({code})", "错误": err})
            continue
        s, e2 = _series_from_hist(df, name)
        if e2:
            attempts.append({"源": f"futures_foreign_hist({code})", "错误": e2})
            continue
        pct, latest, anchor = _chg5(s, name)
        return {"name": name, "latest": round(latest, 2), "unit": "美元/盎司",
                "as_of": s.index[-1].date().isoformat(), "chg5_pct": round(pct, 2),
                "chg5_from": anchor, "source": f"ak.futures_foreign_hist({code})"}
    # 3) 新浪外盘实时（仅最新价，无5日历史）——逐个品种传字符串，
    #    兼容 futures_foreign_commodity_realtime 的 symbol 仅接受 str 的 akshare 版本
    for name in SINA_GOLD_RT:
        df, err = _try(ak.futures_foreign_commodity_realtime, symbol=name)
        if err:
            attempts.append({"源": f"实时行情({name})", "错误": err})
            continue
        if df is None or "最新价" not in df.columns:
            attempts.append({"源": f"实时行情({name})", "错误": "返回列无最新价"})
            continue
        rt = df[df["名称"] == name].copy()
        rt = rt[pd.to_numeric(rt["最新价"], errors="coerce").notna()]
        if rt.empty:
            attempts.append({"源": f"实时行情({name})", "错误": "无有效最新价"})
            continue
        row = rt.iloc[0]
        return {"name": str(row["名称"]), "latest": float(row["最新价"]), "unit": "美元/盎司",
                "as_of": str(row.get("日期", "")), "chg5_pct": None, "chg5_from": None,
                "source": "ak.futures_foreign_commodity_realtime(实时)",
                "note": "实时源无5日历史，5日涨跌幅缺失"}
    # 4) 上海金基准价（国内现货代理，历史完整）
    df, err = _try(ak.spot_golden_benchmark_sge)
    if err:
        attempts.append({"源": "spot_golden_benchmark_sge", "错误": err})
    elif df is not None:
        s, e2 = _series_from_hist(df, "上海金基准价")
        if e2:
            attempts.append({"源": "spot_golden_benchmark_sge", "错误": e2})
        else:
            pct, latest, anchor = _chg5(s, "上海金基准价")
            return {"name": "上海金基准价(国内现货代理)", "latest": round(latest, 2), "unit": "元/克",
                    "as_of": s.index[-1].date().isoformat(), "chg5_pct": round(pct, 2),
                    "chg5_from": anchor, "source": "ak.spot_golden_benchmark_sge",
                    "note": "非伦敦金现，为上海金基准价替代口径"}
    reasons = "; ".join(a["错误"] for a in attempts if a["源"].startswith(("macro_usa_london_gold", "futures_foreign", "实时行情", "spot_golden")))
    return {"name": "伦敦金现", "status": "unavailable", "latest": None, "unit": "美元/盎司",
            "as_of": None, "chg5_pct": None, "chg5_from": None,
            "source": None, "reason": reasons or "无可用黄金数据源"}


def _fetch_oil(attempts: list) -> dict:
    """原油：布伦特(OIL) → WTI(CL) → 新浪实时。"""
    for code, name in SINA_OIL_HIST:
        df, err = _try(ak.futures_foreign_hist, symbol=code)
        if err:
            attempts.append({"源": f"futures_foreign_hist({code})", "错误": err})
            continue
        s, e2 = _series_from_hist(df, name)
        if e2:
            attempts.append({"源": f"futures_foreign_hist({code})", "错误": e2})
            continue
        pct, latest, anchor = _chg5(s, name)
        return {"name": name, "latest": round(latest, 2), "unit": "美元/桶",
                "as_of": s.index[-1].date().isoformat(), "chg5_pct": round(pct, 2),
                "chg5_from": anchor, "source": f"ak.futures_foreign_hist({code})"}
    for name in SINA_OIL_RT:
        df, err = _try(ak.futures_foreign_commodity_realtime, symbol=name)
        if err:
            attempts.append({"源": f"实时行情({name})", "错误": err})
            continue
        if df is None or "最新价" not in df.columns:
            attempts.append({"源": f"实时行情({name})", "错误": "返回列无最新价"})
            continue
        rt = df[df["名称"] == name].copy()
        rt = rt[pd.to_numeric(rt["最新价"], errors="coerce").notna()]
        if rt.empty:
            attempts.append({"源": f"实时行情({name})", "错误": "无有效最新价"})
            continue
        row = rt.iloc[0]
        return {"name": str(row["名称"]), "latest": float(row["最新价"]), "unit": "美元/桶",
                "as_of": str(row.get("日期", "")), "chg5_pct": None, "chg5_from": None,
                "source": "ak.futures_foreign_commodity_realtime(实时)",
                "note": "实时源无5日历史，5日涨跌幅缺失"}
    reasons = "; ".join(a["错误"] for a in attempts if a["源"].startswith(("futures_foreign_hist", "实时行情")))
    return {"name": "原油(布伦特/WTI)", "status": "unavailable", "latest": None, "unit": "美元/桶",
            "as_of": None, "chg5_pct": None, "chg5_from": None,
            "source": None, "reason": reasons or "无可用原油数据源"}


def _fetch_us10y(attempts: list) -> dict:
    """美债10Y：东财中美国债收益率（历史日频，含5日变化）。"""
    df, err = _try(ak.bond_zh_us_rate, start_date="20240101")
    if err:
        attempts.append({"源": "bond_zh_us_rate", "错误": err})
        return {"name": "美债10年期收益率", "status": "unavailable", "latest": None, "unit": "%",
                "as_of": None, "chg5_pct": None, "chg5_bp": None, "chg5_from": None,
                "source": None, "reason": err}
    col = next((c for c in df.columns if "美国" in c and "10年" in c and "2年" not in c), None)
    if col is None:
        return {"name": "美债10年期收益率", "status": "unavailable", "latest": None, "unit": "%",
                "as_of": None, "chg5_pct": None, "chg5_bp": None, "chg5_from": None,
                "source": None, "reason": f"bond_zh_us_rate: 找不到美国10年期列 {list(df.columns)[:8]}"}
    dates = pd.to_datetime(df["日期"], errors="coerce")
    vals = pd.to_numeric(df[col], errors="coerce")
    yc = pd.Series(vals.values, index=dates).dropna()
    yc = yc[~yc.index.duplicated(keep="last")].sort_index()
    if len(yc) < 2:
        return {"name": "美债10年期收益率", "status": "unavailable", "latest": None, "unit": "%",
                "as_of": None, "chg5_pct": None, "chg5_bp": None, "chg5_from": None,
                "source": None, "reason": "bond_zh_us_rate: 美国10Y有效值不足2条"}
    latest = float(yc.iloc[-1])
    anchor_idx = max(0, len(yc) - 6)
    anchor = float(yc.iloc[anchor_idx])
    bp = (latest - anchor) * 100.0
    pct = (latest / anchor - 1.0) * 100.0 if anchor else np.nan
    return {"name": "美债10年期收益率", "latest": round(latest, 2), "unit": "%",
            "as_of": yc.index[-1].date().isoformat(), "chg5_pct": round(float(pct), 2),
            "chg5_bp": round(float(bp), 1), "chg5_from": yc.index[anchor_idx].date().isoformat(),
            "source": "ak.bond_zh_us_rate"}


def _fetch_vix(attempts: list) -> dict:
    """VIX（可选）：失败跳过，不参与模式判定。"""
    if ak is not None and hasattr(ak, "macro_usa_vix"):
        df, err = _try(getattr(ak, "macro_usa_vix"))
        if not err and df is not None:
            s, e2 = _series_from_hist(df, "VIX")
            if not e2:
                pct, latest, anchor = _chg5(s, "VIX")
                return {"name": "VIX恐慌指数", "latest": round(latest, 2), "unit": "点",
                        "as_of": s.index[-1].date().isoformat(), "chg5_pct": round(pct, 2),
                        "chg5_from": anchor, "source": "ak.macro_usa_vix"}
            attempts.append({"源": "macro_usa_vix", "错误": e2})
        elif err:
            attempts.append({"源": "macro_usa_vix", "错误": err})
    # 新浪外盘实时/日线（多数版本无 VIX 品种，属预期失败）
    df, err = _try(ak.futures_foreign_commodity_realtime, symbol="VIX")
    if not err and df is not None and "最新价" in df.columns and not df.empty:
        row = df.iloc[0]
        return {"name": "VIX恐慌指数", "latest": float(row["最新价"]), "unit": "点",
                "as_of": str(row.get("日期", "")), "chg5_pct": None, "chg5_from": None,
                "source": "新浪外盘实时", "note": "实时源无5日历史"}
    df, err = _try(ak.futures_foreign_hist, symbol="VIX")
    if not err and df is not None:
        s, e2 = _series_from_hist(df, "VIX")
        if not e2:
            pct, latest, anchor = _chg5(s, "VIX")
            return {"name": "VIX恐慌指数", "latest": round(latest, 2), "unit": "点",
                    "as_of": s.index[-1].date().isoformat(), "chg5_pct": round(pct, 2),
                    "chg5_from": anchor, "source": "ak.futures_foreign_hist(VIX)"}
    attempts.append({"源": "VIX(可选)", "错误": "本机无 macro_usa_vix 且新浪外盘无 VIX 行情（可选源，跳过）"})
    return {"name": "VIX恐慌指数", "status": "unavailable", "latest": None, "unit": "点",
            "as_of": None, "chg5_pct": None, "chg5_from": None,
            "source": None, "reason": "可选源不可用，跳过（不影响模式判定）"}


# ────────────────────────────────────────────────────────────
# 信号 & 模式判定（只用可用资产）
# ────────────────────────────────────────────────────────────
def _available(asset: dict) -> bool:
    return asset.get("status", "available") == "available" and asset.get("latest") is not None


def _decide(assets: dict) -> tuple[str, dict, list[str], list[str]]:
    gold, oil, yc = assets["gold"], assets["oil"], assets["us10y"]

    gold_up = None
    if _available(gold) and gold.get("chg5_pct") is not None:
        gold_up = gold["chg5_pct"] > GOLD_UP_PCT
    oil_up = None
    if _available(oil) and oil.get("chg5_pct") is not None:
        oil_up = oil["chg5_pct"] > OIL_UP_PCT
    yield_high = (yc["latest"] > YIELD_HIGH) if _available(yc) else None
    gold_oil_up = (gold_up is True and oil_up is True) if (gold_up is not None and oil_up is not None) else None
    oil_down = None
    if _available(oil) and oil.get("chg5_pct") is not None:
        oil_down = oil["chg5_pct"] < 0

    flags = {
        "gold_up": gold_up,
        "oil_up": oil_up,
        "yield_high": yield_high,
        "gold_oil_up": gold_oil_up,
        "oil_down": oil_down,
    }

    any_available = any(_available(a) for a in assets.values())
    if not any_available:
        return "数据不可用", flags, ["核心海外资产全部不可用"], []

    evidence = []
    if yield_high:
        evidence.append(f"美债10Y {yc['latest']}% > {YIELD_HIGH}%（利率高企）")
    if gold_up:
        evidence.append(f"黄金5日 {gold['chg5_pct']:+.1f}% > +{GOLD_UP_PCT}%（避险买金）")
    if oil_up:
        evidence.append(f"原油5日 {oil['chg5_pct']:+.1f}% > +{OIL_UP_PCT}%（油价走强）")
    if oil_down:
        evidence.append(f"原油5日 {oil['chg5_pct']:+.1f}% < 0（油价回落）")

    if yield_high is True and gold_oil_up is True:
        mode = "滞胀避险模式"
    elif yield_high is True:
        mode = "利率压制"
    elif gold_oil_up is True:
        mode = "通胀对冲"
    elif gold_up is True and oil_down is True:
        mode = "避险不滞胀"
    else:
        mode = "中性"

    warnings = []
    core_flags = [gold_up, oil_up, yield_high]
    if any(v is None for v in core_flags) and any(v is not None for v in core_flags):
        warnings.append("部分数据不可用，判定基于可用资产")
    return mode, flags, evidence or ["未触发明显信号"], warnings


# ────────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────────
def analyze(date: str | None = None, refresh: bool = False) -> dict:
    """抓取(或读当日缓存) → 判定 → 写缓存 JSON。返回结果 dict。"""
    date = date or _now().date().isoformat()
    cache = _cache_path(date)
    if cache.exists() and not refresh:
        res = json.loads(cache.read_text(encoding="utf-8"))
        res["cached"] = True
        return res

    attempts: list[dict] = []
    assets = {
        "gold": _fetch_gold(attempts),
        "oil": _fetch_oil(attempts),
        "us10y": _fetch_us10y(attempts),
        "vix": _fetch_vix(attempts),
    }
    mode, flags, evidence, warnings = _decide(assets)
    mapping = MODE_MAP.get(mode, MODE_MAP["中性"])

    res = {
        "date": date,
        "generated_at": _now().isoformat(timespec="seconds"),
        "assets": assets,
        "flags": flags,
        "mode": mode,
        "evidence": evidence,
        "warnings": warnings,
        "partial_unavailable": bool(warnings),
        "mapping": mapping,
        "attempts": attempts,
        "cached": False,
        "note": "温度建议值仅输出参考，不自动修改 fusion",
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return res


def get_macro_overseas(date: str | None = None) -> dict:
    """供 watch_card/日报 引用的轻量只读接口（读当日缓存，不联网）。"""
    date = date or _now().date().isoformat()
    p = _cache_path(date)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    def _unavailable(name: str, unit: str) -> dict:
        return {"name": name, "status": "unavailable", "latest": None, "unit": unit,
                "as_of": None, "chg5_pct": None, "chg5_from": None, "source": None,
                "reason": "当日无缓存，未抓取"}

    return {
        "date": date,
        "generated_at": None,
        "assets": {
            "gold": _unavailable("伦敦金现", "美元/盎司"),
            "oil": _unavailable("原油(布伦特/WTI)", "美元/桶"),
            "us10y": {**_unavailable("美债10年期收益率", "%"), "chg5_bp": None},
            "vix": _unavailable("VIX恐慌指数", "点"),
        },
        "flags": {"gold_up": None, "oil_up": None, "yield_high": None,
                  "gold_oil_up": None, "oil_down": None},
        "mode": "数据不可用",
        "evidence": ["当日无缓存"],
        "warnings": [],
        "partial_unavailable": False,
        "mapping": MODE_MAP["数据不可用"],
        "attempts": [],
        "cached": False,
        "note": "当日无缓存；请先运行 python3 -m quant_system.analysis_core.macro_overseas --report",
    }


# ────────────────────────────────────────────────────────────
# Markdown 报告
# ────────────────────────────────────────────────────────────
def _asset_line(key: str, a: dict) -> str:
    if not _available(a):
        return f"| {a['name']} | — | — | ⚠️ unavailable | {a.get('reason', '')[:60]} |"
    chg = a.get("chg5_pct")
    if key == "us10y":
        chg_str = f"{a.get('chg5_bp'):+.1f}bp" if a.get("chg5_bp") is not None else "—"
        latest_str = f"{a['latest']}%"
    elif chg is not None:
        chg_str = f"{chg:+.2f}%"
        latest_str = f"{a['latest']} {a['unit']}"
    else:
        chg_str = "—(无5日历史)"
        latest_str = f"{a['latest']} {a['unit']}"
    return (f"| {a['name']} | {latest_str} | {chg_str} | ✅ {a['source']} | "
            f"as_of {a['as_of']} |")


def build_report(res: dict) -> str:
    date = res["date"]
    a = res["assets"]
    flags = res["flags"]
    m = res["mapping"]

    def flag_str(v):
        if v is None:
            return "n/a(资产不可用)"
        return "✅ True" if v else "❌ False"

    lines = [
        f"# 🌐 宏观风向标 {date}",
        "",
        "> 用户宏观论点：美债收益率高 → 信用危机担忧 → 黄金上涨；金油同涨 = 滞胀/避险信号",
        f"> 生成时间: {res.get('generated_at', '')} | 数据源: akshare | 缓存: {'命中' if res.get('cached') else '新抓取'}",
        "",
        "## 📊 资产快照（最新值 + 5日涨跌幅）",
        "",
        "| 资产 | 最新值 | 5日涨跌 | 状态/源 | 日期 |",
        "|---|---|---|---|---|",
    ]
    for key in ("gold", "oil", "us10y", "vix"):
        lines.append(_asset_line(key, a[key]))
    lines += [
        "",
        "## 🧭 信号判定",
        f"- gold_up（金5日 > +1.5%）: {flag_str(flags['gold_up'])}",
        f"- oil_up（油5日 > +1.5%）: {flag_str(flags['oil_up'])}",
        f"- yield_high（美债10Y > 4.5%）: {flag_str(flags['yield_high'])}",
        f"- 金油同涨: {flag_str(flags['gold_oil_up'])}",
        "",
        "## 🏷️ 宏观模式",
        f"**{res['mode']}**",
    ]
    if res["mode"] != "数据不可用":
        lines.append(f"- 依据: {'；'.join(res.get('evidence', []))}")
        lines.append(f"- 逻辑: {m['note']}")
    else:
        lines.append(f"- 依据: {'；'.join(res.get('evidence', []))}")
    for w in res.get("warnings", []):
        lines.append(f"- ⚠️ {w}")
    lines += [
        "",
        "## 🇨🇳 A股映射建议（供 watch_card/日报引用）",
    ]
    if m.get("prefer"):
        lines.append(f"- 偏好: {'、'.join(m['prefer'])}")
        lines.append(f"- 规避: {'、'.join(m['avoid'])}")
    else:
        lines.append("- 偏好/规避: 数据不可用，暂不给出板块映射")
    if m.get("temp_suggestion") is not None:
        lines.append(f"- 全市场温度建议: **{m['temp_suggestion']}**（仅建议，不自动改 fusion）")
    else:
        lines.append("- 全市场温度建议: 不调整")
    lines += [
        "",
        "## ⚠️ 数据可用性",
    ]
    for key in ("gold", "oil", "us10y", "vix"):
        x = a[key]
        if _available(x):
            lines.append(f"- {x['name']}: ✅ available（{x['source']}）")
        else:
            lines.append(f"- {x['name']}: ⚠️ unavailable（{x.get('reason', '')[:90]}）")
    if res.get("attempts"):
        lines.append("")
        lines.append("**抓取尝试明细:**")
        for t in res["attempts"][-8:]:
            lines.append(f"- {t.get('源')}: {t.get('错误', '')[:100]}")
    lines += [
        "",
        "---",
        "*V11 宏观风向标自动生成 | 模式为概率判断，仅供仓位与板块参考，不构成投资建议*",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="宏观风向标（海外资产 → A股映射）")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD，默认当日(CST)")
    ap.add_argument("--report", action="store_true", help="生成 generated/macro_overseas_{date}.md")
    ap.add_argument("--refresh", action="store_true", help="当日强制重抓，忽略缓存")
    args = ap.parse_args()

    result = analyze(args.date, refresh=args.refresh)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    if args.report:
        out = OUT_DIR / f"macro_overseas_{result['date']}.md"
        out.write_text(build_report(result), encoding="utf-8")
        print(f"\n✅ 报告已生成: {out}")
