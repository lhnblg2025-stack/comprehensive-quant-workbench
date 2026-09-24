"""
macro_system — 体系5 宏观周期系统（V11 宏观主引擎）

方法论（先读 skills 再实现）:
  - macro-four-driver-asset-map: 增长/通胀/流动性/风险偏好 四驱动 → 资产映射
    （Dalio: 四驱动力 → 股/债/商品/黄金/现金 的回报含义，先打分方向再映射资产）
  - animal-spirits-risk-forecasting: 市场情绪(动物精神)驱动的风险定价；
    滞胀 = 金涨 + 油涨 + 美债高（避险/信用担忧传导 → 防御资产受追捧）
  - commodity-cyclical-valuation: 大宗商品周期 → 周期股/资源股驱动
    （油涨 → PPI 输入性通胀 + 资源盈利；金油分化 → 避险 vs 需求）
  - big-cycle-empire: 长周期债务/信用周期定位（社融/M2/利率大周期）

输入:
  1. macro_overseas: generated/macro_overseas_{date}.json
     （伦敦金/布伦特油/美债10Y/VIX + 海外模式，get_macro_overseas 只读缓存不联网）
  2. 宏观数据仓: data_warehouse/macro/*.parquet
     cpi_yearly / ppi_yearly / pmi_yearly / gdp_yearly /
     industrial_production_yoy / m2_yearly / shibor_all / lpr / shrzgm
  3. fusion 温度: data_warehouse/market/fusion.parquet（风险偏好代理）

检测:
  1. 四驱动状态: 增长(PMI/工业增加值/GDP)、通胀(CPI/PPI)、流动性(M2/SHIBOR/LPR)、
     风险偏好(VIX/fusion温度) 各判 强/中性/弱（流动性: 宽/中性/紧）
  2. 宏观模式: 美林时钟简化版（复苏/过热/滞胀/衰退/中性过渡）
  1b. 消费信心(CCI): 消费者信心指数 <88 且环比下降 → 风险偏好减档 -0.1；
      CCI ≥92 → +0.1；88~92 中性（macro_learner 实证，缺失/异常返回 None 不参与）
  3. 资产映射: 模式 → A股板块偏好
  4. 海外交叉: macro_overseas 滞胀避险模式（金涨+油涨+美债高）触发 → 强化滞胀判定
  5. 综合: 模式 + 置信（数据新鲜度加权：≤45天参与，>45天标注不参与）

接口:
  class MacroSystem: detect(date) / report(date) / view(date)
  view 返回 {agent:'宏观周期', signal, confidence, evidence}

输出: generated/macro_report_{date}.md

用法:
  python3 -m quant_system.analysis_core.macro_system --date 2026-08-11 --report
  python3 -m quant_system.analysis_core.macro_system --date 2026-08-11 --report --skip-rag
"""

from __future__ import annotations
import logging

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).resolve().parent.parent.parent  # ${PROJECT_ROOT}
REPO_ROOT = Path(__file__).resolve().parent.parent          # quant_system 仓库根
sys.path.insert(0, str(WORKSPACE))
from quant_system.analysis_core.common import num  # noqa: E402
from quant_system.analysis_core.fusion import read_fusion_latest  # noqa: E402

MACRO_DIR = WORKSPACE / "data_warehouse" / "macro"
MARKET_DIR = WORKSPACE / "data_warehouse" / "market"
OUT_DIR = REPO_ROOT / "generated"

CST = timezone(timedelta(hours=8))

STALE_DAYS = 45          # 陈旧阈值：超过 45 天不参与判定（仅标注）
RAG_QUERY = "美林时钟 滞胀 复苏 资产配置"
RAG_K = 3

AGENT_NAME = "宏观周期"
VIEW_WEIGHT = 1.0         # view() 默认权重（multi_agent 仲裁权重独立维护于 multi_agent.BASE_WEIGHTS）

# ── 四驱动指标阈值（强/宽 边界，均含边界）────────────────
PMI_STRONG, PMI_WEAK = 51.0, 49.0
GDP_STRONG, GDP_WEAK = 5.5, 4.5
IND_STRONG, IND_WEAK = 7.0, 5.0
CPI_STRONG, CPI_WEAK = 3.0, 1.0
PPI_STRONG, PPI_WEAK = 3.0, 0.0
M2_WIDE, M2_TIGHT = 10.0, 8.0
SHIBOR_WIDE, SHIBOR_TIGHT = 1.8, 2.5
LPR_WIDE, LPR_TIGHT = 3.0, 4.0
TEMP_HOT, TEMP_COLD = 70, 35
VIX_HIGH, VIX_LOW = 25, 18
CCI_WEAK = 88.0           # macro_learner 实证: CCI<88 消费板块承压, p<0.05
CCI_STRONG = 92.0
CCI_RISK_ADJ = 0.1        # 消费信心对 risk_appetite 的加减档幅度

# 各驱动内指标权重（需归一，缺失时按参与指标重新归一）
DRIVER_WEIGHTS = {
    "growth":  [("PMI制造业指数", 0.5), ("GDP同比", 0.3), ("工业增加值同比", 0.2)],
    "inflation": [("CPI同比", 0.6), ("PPI同比", 0.4)],
    "liquidity": [("M2同比", 0.5), ("SHIBOR 3M", 0.3), ("LPR 1Y", 0.2)],
    "risk_appetite": [("fusion温度", 0.6), ("VIX", 0.4)],
}

# 模式 → A股板块映射（含 temp_hint 仅建议，不自动改 fusion）
MODE_MAP: dict[str, dict] = {
    "复苏": {
        "prefer": ["股票/周期(券商/地产链/中游制造)", "科技成长(风险偏好回升)"],
        "avoid": ["防御/债券(资金流出)"],
        "temp_hint": 5,
        "note": "美林时钟复苏: 增长强+通胀弱+流动性宽 → 股票/周期占优，盈利修复驱动。",
    },
    "过热": {
        "prefer": ["商品/资源(油气/有色/煤炭/化工)", "周期股(盈利弹性)"],
        "avoid": ["长久期债券(利率上行)", "高估值成长(贴现率压制)"],
        "temp_hint": 5,
        "note": "美林时钟过热: 增长强+通胀强 → 商品/资源占优；过热末期警惕政策收紧。",
    },
    "滞胀": {
        "prefer": ["黄金/贵金属", "资源(油气/有色/煤炭)", "红利防御(银行/电力/公用/高股息)", "现金"],
        "avoid": ["科技成长(半导体/AI/消费电子)", "长久期高估值(创业板/科创板)", "债券(利率高企)"],
        "temp_hint": -10,
        "note": "滞胀: 增长弱+通胀强 → 黄金/防御/现金；动物精神避险传导，资金从长久期成长撤向实物资产与高股息。",
    },
    "滞胀（避险强化）": {
        "prefer": ["黄金/贵金属", "资源(油气/有色/煤炭)", "红利防御(银行/电力/公用/高股息)", "现金"],
        "avoid": ["科技成长(半导体/AI/消费电子)", "长久期高估值(创业板/科创板)", "债券(利率高企)"],
        "temp_hint": -10,
        "note": "滞胀（避险强化）: 国内四驱动未完全共振，但海外金涨+油涨+美债高（动物精神滞胀特征）强化滞胀判定 → 黄金/防御/现金，规避成长。",
    },
    "滞胀（海外交叉确认）": {
        "prefer": ["黄金/贵金属", "资源(油气/有色/煤炭)", "红利防御(银行/电力/公用/高股息)", "现金"],
        "avoid": ["科技成长(半导体/AI/消费电子)", "长久期高估值(创业板/科创板)", "债券(利率高企)"],
        "temp_hint": -10,
        "note": "滞胀（海外交叉确认）: 国内四驱动已指向滞胀，海外金油美债同步确认 → 黄金/防御/现金信号强化。",
    },
    "衰退": {
        "prefer": ["债券/利率债", "高股息防御(公用/医药/必选消费)", "超跌成长反弹(政策托底)"],
        "avoid": ["强周期/资源(需求下行)", "高杠杆(信用风险)"],
        "temp_hint": -15,
        "note": "美林时钟衰退: 增长弱+通胀弱+流动性宽 → 债券/成长反弹；等待政策底确认。",
    },
    "中性过渡": {
        "prefer": ["均衡配置, 防御略重"],
        "avoid": ["无明确规避"],
        "temp_hint": 0,
        "note": "四驱动方向未共振或数据不足，处于周期过渡带，等待数据确认。",
    },
}

# 模式 → multi_agent 观点信号
VIEW_SIGNAL: dict[str, str] = {
    "复苏": "多",
    "过热": "多",
    "滞胀": "防守",
    "滞胀（避险强化）": "防守",
    "滞胀（海外交叉确认）": "防守",
    "衰退": "空",
    "中性过渡": "震荡",
}


# ────────────────────────────────────────────
# 数据读取（全部本地 parquet，不联网）
# ────────────────────────────────────────────
MONTH_PUBLISH_LAG_DAYS = 10  # 月度数据发布滞后：月末 +10 天才可视为已知（防月初前视）

def _month_asof(month_str) -> pd.Timestamp | None:
    """'2026年07月份' / '2026-07' → 2026-07-31 + 10天发布滞后 = 2026-08-10。

    月度指标在月初并不已知（需月末汇总 + 发布窗口），as_of 取 月末+发布滞后，
    避免月初即用当月/当月数据造成前视；STALE_DAYS 新鲜度逻辑保持不变。
    """
    m = (re.match(r"(\d{4})年(\d{1,2})月", str(month_str))
         or re.match(r"(\d{4})-(\d{1,2})", str(month_str)))
    if not m:
        return None
    month_end = pd.Timestamp(int(m.group(1)), int(m.group(2)), 1) + pd.offsets.MonthEnd(0)
    return month_end + pd.Timedelta(days=MONTH_PUBLISH_LAG_DAYS)


# 2026-08-21 审计修复: GDP 等季度指标官方滞后约 45 天发布（Q1 约 4 月中），
# 原 _quarter_asof 直接取季末导致 ref 刚过季末即误用当季 GDP（比月度口径多 ~1.5 个月前视窗口）。
QUARTER_PUBLISH_LAG_DAYS = 45


def _quarter_asof(quarter_str) -> pd.Timestamp | None:
    """'2026年第1-2季度' → 期末 2026-06-30 + 发布滞后 45 天（8-14 附近）。"""
    m = re.match(r"(\d{4})年第(\d+)(?:-(\d+))?季度", str(quarter_str))
    if not m:
        return None
    end_q = int(m.group(3) or m.group(2))
    month = {1: 3, 2: 6, 3: 9, 4: 12}.get(end_q)
    if month is None:
        return None
    end = pd.Timestamp(int(m.group(1)), month, 1) + pd.offsets.MonthEnd(0)
    return end + pd.Timedelta(days=QUARTER_PUBLISH_LAG_DAYS)


def _monthly_latest(path: Path, value_col: str, ref: pd.Timestamp) -> tuple[float | None, pd.Timestamp | None]:
    """月份列 parquet → ≤ref 最新一期 (value, as_of)。"""
    if not path.exists():
        return None, None
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None, None
    if "月份" not in df.columns or value_col not in df.columns:
        return None, None
    rows: list[tuple[pd.Timestamp, float]] = []
    for _, r in df.iterrows():
        a = _month_asof(r["月份"])
        v = pd.to_numeric(r.get(value_col), errors="coerce")
        if a is None or pd.isna(v) or a > ref:
            continue
        rows.append((float(v), a))
    if not rows:
        return None, None
    rows.sort(key=lambda x: x[1])
    return rows[-1]


def _datecol_latest(path: Path, value_col: str, date_col: str, ref: pd.Timestamp) -> tuple[float | None, pd.Timestamp | None]:
    """日期列 parquet → ≤ref 最新一期 (value, as_of)。"""
    if not path.exists():
        return None, None
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None, None
    if date_col not in df.columns or value_col not in df.columns:
        return None, None
    dates = pd.to_datetime(df[date_col], errors="coerce")
    vals = pd.to_numeric(df[value_col], errors="coerce")
    df2 = pd.DataFrame({"d": dates, "v": vals}).dropna()
    df2 = df2[df2["d"] <= ref]
    if df2.empty:
        return None, None
    row = df2.sort_values("d").iloc[-1]
    return float(row["v"]), row["d"]


def _indicator(name: str, value: float | None, as_of: pd.Timestamp | None,
               ref: pd.Timestamp, weight: float, state_fn) -> dict:
    """构造单指标记录：新鲜度标注 + 状态判定（陈旧>45天不参与）。"""
    lag = max(0, (ref.normalize() - as_of.normalize()).days) if as_of is not None else None
    fresh = lag is not None and lag <= STALE_DAYS
    state = state_fn(value) if (fresh and value is not None) else None
    if lag is None:
        note = "数据缺失"
    elif fresh:
        note = ""
    else:
        note = f"陈旧{lag}天>45，不参与"
    return {"name": name, "value": value, "as_of": as_of, "lag_days": lag,
            "fresh": bool(fresh), "state": state, "weight": weight, "note": note}


def _score(state: str | None) -> float:
    """状态 → 方向分（强/宽 +1，中性 0，弱/紧 -1；缺失按 0 但不参与）。"""
    return {"强": 1.0, "宽": 1.0, "中性": 0.0, "弱": -1.0, "紧": -1.0}.get(state, 0.0)


def _combine(driver: str, indicators: list[dict]) -> dict:
    """驱动内加权合并 → state/score/evidence。"""
    part = [i for i in indicators if i["fresh"]]
    evidence = []
    for i in indicators:
        if i["fresh"]:
            evidence.append(f"{i['name']} {i['value']}（{i['as_of'].strftime('%Y-%m')}，滞后{i['lag_days']}天）→ {i['state']}")
        elif i["lag_days"] is not None:
            evidence.append(f"{i['name']} {i['value']}（{i['as_of'].strftime('%Y-%m')}，{i['note']}）")
        else:
            evidence.append(f"{i['name']} 数据缺失")
    if not part:
        return {"name": driver, "state": "数据缺失", "score": None,
                "indicators": indicators, "evidence": evidence,
                "note": "核心指标全部缺失或陈旧，无法判定"}
    total = sum(i["weight"] for i in part)
    score = sum(i["weight"] * _score(i["state"]) for i in part) / total
    if score >= 0.35:
        state = "宽" if driver == "流动性" else "强"
    elif score <= -0.35:
        state = "紧" if driver == "流动性" else "弱"
    else:
        state = "中性"
    return {"name": driver, "state": state, "score": round(score, 3),
            "indicators": indicators, "evidence": evidence, "note": ""}


def _fusion_temperature(ref: pd.Timestamp) -> tuple[float | None, pd.Timestamp | None]:
    """fusion 温度（风险偏好代理）：fusion.parquet ≤ref 最新一期。"""
    row = read_fusion_latest(ref, ["temperature"])
    if row is None:
        return None, None
    temp = num(row.get("temperature"))
    if temp is None:
        return None, None
    return temp, pd.Timestamp(row["date"])


def _m2_trend(ref: pd.Timestamp, n: int = 4) -> list[dict]:
    """M2 同比近 n 期（含 as_of，用于 big-cycle 观察，不参与新鲜度过滤）。"""
    vals = []
    if (MACRO_DIR / "m2_yearly.parquet").exists():
        try:
            df = pd.read_parquet(MACRO_DIR / "m2_yearly.parquet")
            rows = []
            for _, r in df.iterrows():
                a = _month_asof(r["月份"])
                v = pd.to_numeric(r.get("货币和准货币(M2)-同比增长"), errors="coerce")
                if a is not None and pd.notna(v) and a <= ref:
                    rows.append((float(v), a))
            rows.sort(key=lambda x: x[1])
            vals = [{"as_of": a.strftime("%Y-%m"), "value": v} for v, a in rows[-n:]]
        except Exception:
            vals = []
    return vals


# ────────────────────────────────────────────
# 四驱动判定
# ────────────────────────────────────────────
def _driver_growth(ref: pd.Timestamp) -> dict:
    pmis, pmi_as = _monthly_latest(MACRO_DIR / "pmi_yearly.parquet", "制造业-指数", ref)
    gdp, gdp_as = None, None
    if (MACRO_DIR / "gdp_yearly.parquet").exists():
        try:
            df = pd.read_parquet(MACRO_DIR / "gdp_yearly.parquet")
            rows = []
            for _, r in df.iterrows():
                a = _quarter_asof(r["季度"])
                v = pd.to_numeric(r.get("国内生产总值-同比增长"), errors="coerce")
                if a is not None and pd.notna(v) and a <= ref:
                    rows.append((float(v), a))
            if rows:
                rows.sort(key=lambda x: x[1])
                gdp, gdp_as = rows[-1]
        except Exception:
            gdp, gdp_as = None, None
    ind, ind_as = _datecol_latest(MACRO_DIR / "industrial_production_yoy.parquet",
                                  "今值", "日期", ref)
    inds = [
        _indicator("PMI制造业指数", pmis, pmi_as, ref, 0.5, _state_pmi),
        _indicator("GDP同比", gdp, gdp_as, ref, 0.3, _state_gdp),
        _indicator("工业增加值同比", ind, ind_as, ref, 0.2, _state_ind),
    ]
    return _combine("增长", inds)


def _driver_inflation(ref: pd.Timestamp) -> dict:
    cpi, cpi_as = _monthly_latest(MACRO_DIR / "cpi_yearly.parquet", "全国-同比增长", ref)
    ppi, ppi_as = _monthly_latest(MACRO_DIR / "ppi_yearly.parquet", "当月同比增长", ref)
    inds = [
        _indicator("CPI同比", cpi, cpi_as, ref, 0.6, _state_cpi),
        _indicator("PPI同比", ppi, ppi_as, ref, 0.4, _state_ppi),
    ]
    d = _combine("通胀", inds)
    if cpi is not None and ppi is not None and cpi < 1 and ppi > 3:
        d["note"] = "上下游通胀分化（CPI弱 + PPI强）：下游通缩、上游成本压力，滞胀风险主要由 PPI/大宗传导"
    elif cpi is not None and cpi < 0:
        d["note"] = "CPI 同比为负，通缩压力（区别于滞胀）"
    return d


def _driver_liquidity(ref: pd.Timestamp) -> dict:
    m2, m2_as = _monthly_latest(MACRO_DIR / "m2_yearly.parquet",
                                "货币和准货币(M2)-同比增长", ref)
    shibor, shibor_as = None, None
    if (MACRO_DIR / "shibor_all.parquet").exists():
        try:
            df = pd.read_parquet(MACRO_DIR / "shibor_all.parquet", columns=["日期", "3M-定价"])
            dates = pd.to_datetime(df["日期"], errors="coerce")
            vals = pd.to_numeric(df["3M-定价"], errors="coerce")
            df2 = pd.DataFrame({"d": dates, "v": vals}).dropna()
            df2 = df2[df2["d"] <= ref]
            if not df2.empty:
                row = df2.sort_values("d").iloc[-1]
                shibor, shibor_as = float(row["v"]), row["d"]
        except Exception:
            shibor, shibor_as = None, None
    lpr, lpr_as = _datecol_latest(MACRO_DIR / "lpr.parquet", "LPR1Y", "TRADE_DATE", ref)
    inds = [
        _indicator("M2同比", m2, m2_as, ref, 0.5, _state_m2),
        _indicator("SHIBOR 3M", shibor, shibor_as, ref, 0.3, _state_shibor),
        _indicator("LPR 1Y", lpr, lpr_as, ref, 0.2, _state_lpr),
    ]
    return _combine("流动性", inds)


def _driver_risk(ref: pd.Timestamp, overseas: dict, fusion_temp: tuple) -> dict:
    temp, temp_as = fusion_temp
    vix = None
    vix_as = None
    assets = overseas.get("assets") or {}
    vix_a = assets.get("vix") or {}
    if vix_a.get("latest") is not None:
        vix = float(vix_a["latest"])
        try:
            vix_as = pd.Timestamp(vix_a.get("as_of") or ref.strftime("%Y-%m-%d"))
        except Exception:
            vix_as = ref
    inds = [
        _indicator("fusion温度", temp, temp_as, ref, 0.6, _state_temp),
        _indicator("VIX", vix, vix_as, ref, 0.4, _state_vix),
    ]
    d = _combine("风险偏好", inds)
    if temp is not None and temp_as is not None:
        d["note"] = f"fusion 温度 {temp:.0f}（≤35 弱 / ≥70 强），作为风险偏好代理"
    return d


def _cci_asof(month_str) -> pd.Timestamp | None:
    """消费信心月份解析：兼容 'YYYY-MM'（consumer_confidence 实际格式）与
    'YYYY年MM月份'，统一按 月末+发布滞后10天 计算 as_of（与 _month_asof 同口径）。"""
    s = str(month_str).strip()
    m = re.match(r"^(\d{4})-(\d{1,2})$", s)
    if m:
        s = f"{int(m.group(1))}年{int(m.group(2))}月"
    return _month_asof(s)


def _driver_consumer_confidence(ref: pd.Timestamp) -> dict | None:
    """消费信心驱动（macro_learner 实证: CCI<88 消费板块承压, p<0.05）。

    CCI < 88 且环比下降 → risk_appetite 减档 -0.1；CCI ≥ 92 → +0.1；88~92 中性。
    数据缺失/异常/陈旧(>STALE_DAYS) → 返回 None，不进入 drivers，不影响其他驱动（降级闭环）。
    防前视: 只取 ≤ref 且含发布滞后（月末+10 天）的最新一期 CCI。
    """
    p = MACRO_DIR / "consumer_confidence.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        if "月份" not in df.columns or "消费者信心指数-指数值" not in df.columns:
            return None
        rows: list[tuple[pd.Timestamp, float]] = []
        for _, r in df.iterrows():
            a = _cci_asof(r["月份"])
            v = pd.to_numeric(r.get("消费者信心指数-指数值"), errors="coerce")
            if a is None or pd.isna(v) or a > ref:
                continue
            rows.append((float(v), a))
        if not rows:
            return None
        rows.sort(key=lambda x: x[1])
        latest_v, latest_a = rows[-1]
        lag = max(0, (ref.normalize() - latest_a.normalize()).days)
        if lag > STALE_DAYS:  # 陈旧不参与（与 _indicator 口径一致）
            return None
        mom = latest_v - rows[-2][0] if len(rows) >= 2 else None
        mom_txt = f"环比{mom:+.1f}" if mom is not None else "环比数据不足"
        if latest_v < CCI_WEAK and mom is not None and mom < 0:
            state, adj = "弱", -CCI_RISK_ADJ
        elif latest_v >= CCI_STRONG:
            state, adj = "强", CCI_RISK_ADJ
        else:
            state, adj = "中性", 0.0
        adj_txt = "减档" if adj < 0 else ("加档" if adj > 0 else "不改")
        return {
            "name": "消费信心",
            "state": state,
            "score": adj,
            "cci": round(latest_v, 1),
            "mom": None if mom is None else round(mom, 2),
            "as_of": latest_a,
            "adjustment": adj,
            "indicators": [{
                "name": "消费者信心指数(CCI)",
                "value": round(latest_v, 1),
                "as_of": latest_a,
                "lag_days": lag,
                "fresh": True,
                "state": state,
                "weight": 1.0,
                "note": "",
            }],
            "evidence": [f"消费者信心指数(CCI) {latest_v}（{latest_a.strftime('%Y-%m')}，"
                         f"{mom_txt}）→ 风险偏好{adj_txt}"],
            "note": "阈值来自 macro_learner 实证（CCI<88 消费板块承压, p<0.05）；"
                    "CCI≥92 信心偏强 → +0.1；88~92 中性",
        }
    except Exception:  # noqa: BLE001  异常 → None，降级闭环
        return None


# 状态判定函数（值 → 强/中性/弱 或 宽/中性/紧）
def _state_pmi(v): return "强" if v >= PMI_STRONG else ("弱" if v < 50 else "中性")
def _state_gdp(v): return "强" if v >= GDP_STRONG else ("弱" if v <= GDP_WEAK else "中性")
def _state_ind(v): return "强" if v >= IND_STRONG else ("弱" if v <= IND_WEAK else "中性")
def _state_cpi(v): return "强" if v > CPI_STRONG else ("弱" if v < CPI_WEAK else "中性")
def _state_ppi(v): return "强" if v > PPI_STRONG else ("弱" if v < PPI_WEAK else "中性")
def _state_m2(v): return "宽" if v > M2_WIDE else ("紧" if v < M2_TIGHT else "中性")
def _state_shibor(v): return "宽" if v < SHIBOR_WIDE else ("紧" if v > SHIBOR_TIGHT else "中性")
def _state_lpr(v): return "宽" if v <= LPR_WIDE else ("紧" if v > LPR_TIGHT else "中性")
def _state_temp(v): return "强" if v >= TEMP_HOT else ("弱" if v <= TEMP_COLD else "中性")
def _state_vix(v): return "弱" if v >= VIX_HIGH else ("强" if v <= VIX_LOW else "中性")


# ────────────────────────────────────────────
# 模式判定（美林时钟简化 + 海外交叉）
# ────────────────────────────────────────────
def _state_or_missing(state: str) -> str:
    return "中性" if state in ("数据缺失", None) else state


def _base_mode(drivers: dict) -> tuple[str, list[str]]:
    """四驱动 → 美林时钟简化模式。缺失驱动按中性处理并提示。"""
    growth = _state_or_missing(drivers["growth"]["state"])
    inflation = _state_or_missing(drivers["inflation"]["state"])
    liquidity = _state_or_missing(drivers["liquidity"]["state"])
    ev: list[str] = []

    if growth == "强":
        if inflation == "强":
            mode = "过热"
            ev.append("增长强 + 通胀强 → 过热（商品/资源占优）")
        else:
            mode = "复苏"
            ev.append(f"增长强 + 通胀{inflation} → 复苏（股票/周期）")
            if liquidity != "宽":
                ev.append(f"⚠️ 复苏但流动性未确认宽松（{liquidity}），反弹持续性存疑")
    elif growth == "弱":
        if inflation == "强":
            mode = "滞胀"
            ev.append("增长弱 + 通胀强 → 滞胀（黄金/防御/现金）")
        elif inflation == "弱" and liquidity == "宽":
            mode = "衰退"
            ev.append("增长弱 + 通胀弱 + 流动性宽 → 衰退（债券/成长反弹）")
        elif inflation == "弱":
            mode = "中性过渡"
            ev.append(f"增长弱 + 通胀弱但流动性未确认宽松（{liquidity}）→ 衰退边缘，待确认")
        else:
            mode = "中性过渡"
            ev.append(f"增长弱 + 通胀{inflation} → 滞胀/衰退过渡带")
    else:
        mode = "中性过渡"
        ev.append(f"增长{growth}（方向未明）→ 中性过渡")
    return mode, ev


def _cross_check(base_mode: str, drivers: dict, overseas: dict) -> tuple[str, list[str], bool]:
    """macro_overseas 交叉：滞胀避险模式（金涨+油涨+美债高）→ 强化滞胀判定。"""
    overseas_mode = overseas.get("mode")
    flags = overseas.get("flags") or {}
    stag = overseas_mode == "滞胀避险模式"
    boost = False
    notes: list[str] = []
    growth = _state_or_missing(drivers["growth"]["state"])
    inflation = _state_or_missing(drivers["inflation"]["state"])

    if stag:
        notes.append("海外交叉: 滞胀避险模式触发（美债高企 + 金涨 + 油涨 = 动物精神滞胀特征）")
        if base_mode == "滞胀":
            mode, boost = "滞胀（海外交叉确认）", True
            notes.append("国内四驱动已判滞胀，海外交叉确认 → 滞胀信号强化")
        elif base_mode in ("中性过渡", "衰退") and growth != "强" and inflation in ("中性", "强"):
            mode, boost = "滞胀（避险强化）", True
            notes.append(f"基础判定为{base_mode}（增长{growth}+通胀{inflation}），"
                         "海外金油美债同向滞胀 → 强化为滞胀（避险）")
        elif base_mode == "复苏":
            mode = "复苏"
            notes.append("⚠️ 海外滞胀避险信号与国内复苏判定背离，复苏需谨慎（防御板块对冲）")
        else:
            mode = base_mode
    else:
        mode = base_mode
        if overseas_mode == "数据不可用":
            notes.append("海外数据不可用，无法交叉验证（仅国内四驱动判定）")
        elif overseas_mode in ("利率压制", "通胀对冲", "避险不滞胀"):
            notes.append(f"海外模式为 {overseas_mode}，未触发滞胀避险交叉（不强化滞胀）")
    return mode, notes, boost


# ────────────────────────────────────────────
# 置信度（数据新鲜度加权）
# ────────────────────────────────────────────
def _freshness_factor(lag_days: int | None) -> float:
    if lag_days is None:
        return 0.0
    if lag_days <= 15:
        return 1.0
    if lag_days <= 30:
        return 0.9
    if lag_days <= STALE_DAYS:
        return 0.8
    return 0.0


def _confidence(drivers: dict, boost: bool) -> float:
    freshes, covers = [], []
    for dv in drivers.values():
        inds = dv["indicators"]
        part = [i for i in inds if i["fresh"]]
        freshes.append(float(np.mean([_freshness_factor(i["lag_days"]) for i in part])) if part else 0.0)
        covers.append(len(part) / len(inds) if inds else 0.0)
    conf = 0.35 + 0.25 * float(np.mean(freshes)) + 0.20 * float(np.mean(covers))
    if boost:
        conf += 0.10
    return round(min(0.95, max(0.15, conf)), 2)


# ────────────────────────────────────────────
# RAG 方法论依据
# ────────────────────────────────────────────
def _rag_basis(date: str) -> list[dict]:
    cache = OUT_DIR / f"macro_rag_{date}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception as e:
            logging.getLogger(__name__).error(f"[macro_system] 操作失败: {e}", exc_info=True)
    try:
        from quant_system.analysis_core import knowledge_rag
        hits = knowledge_rag.search(RAG_QUERY, k=RAG_K)
        out = [{"file": h.get("file"), "cat": h.get("cat"), "score": h.get("score"),
                "text": str(h.get("text", ""))[:220]} for h in hits]
    except Exception as e:
        out = [{"error": f"{type(e).__name__}: {str(e)[:140]}"}]
    try:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logging.getLogger(__name__).error(f"[macro_system] 操作失败: {e}", exc_info=True)
    return out


# ────────────────────────────────────────────
# big-cycle 长周期债务/信用观察（big-cycle-empire 简化落地）
# ────────────────────────────────────────────
def _big_cycle(ref: pd.Timestamp) -> dict:
    m2_trend = _m2_trend(ref)
    shr: list[dict] = []
    p = MACRO_DIR / "shrzgm.parquet"
    if p.exists():
        try:
            df = pd.read_parquet(p)
            rows = []
            for _, r in df.iterrows():
                # 2026-08-21 审计修复: 月份列是 datetime64，str() 会带 "-" 导致解析 month=0
                # 直接取日期时间戳；兼容字符串 "2026-08" 与 datetime 类型。
                ms = r["月份"]
                try:
                    if isinstance(ms, (pd.Timestamp,)):
                        a = ms.normalize()
                    else:
                        a = pd.Timestamp(str(ms)[:7] + "-01")
                    a = pd.Timestamp(a.year, a.month, 1)
                except Exception as e:
                    logging.getLogger(__name__).error(f"[macro_system] 操作失败: {e}", exc_info=True)
                    continue
                v = pd.to_numeric(r.get("社会融资规模增量"), errors="coerce")
                if pd.notna(v) and a <= ref:
                    rows.append((a, float(v)))
            rows.sort()
            shr = [{"as_of": a.strftime("%Y-%m"), "value": v} for a, v in rows[-3:]]
        except Exception:
            shr = []
    m2_vals = [x["value"] for x in m2_trend]
    label = "信用平稳"
    if len(m2_vals) >= 3 and m2_vals[-1] < m2_vals[-2] <= m2_vals[-3]:
        label = "信用边际收缩"
    elif len(m2_vals) >= 2 and m2_vals[-1] > m2_vals[-2]:
        label = "信用扩张"
    shr_weak = bool(shr) and shr[-1]["value"] <= 0 or (len(shr) >= 2 and shr[-1]["value"] < shr[-2]["value"])
    if shr_weak:
        label = "信用边际收缩"
    parts = [f"M2同比 {'→'.join(f'{v}%' for v in m2_vals) or '缺失'}"]
    if shr:
        shr_txt = "/".join(f"{x['value'] / 1e4:.1f}万亿" for x in shr)
        parts.append(f"社融增量 {shr_txt}")
    note = (f"长周期债务/信用周期观察（big-cycle-empire）: {label}。"
            "紧信用/去杠杆阶段历史上支撑黄金/红利/防御，压制周期需求与高杠杆资产；"
            "宽松预期未兑现前，成长反弹持续性受限。")
    return {"label": label, "m2_trend": m2_trend, "shrzgm": shr,
            "summary": "；".join(parts), "note": note}


# ────────────────────────────────────────────
# 主类
# ────────────────────────────────────────────
class MacroSystem:
    """体系5 宏观周期系统统一接口: detect / report / view。"""

    def __init__(self, date: str | None = None):
        self.default_date = date

    def _date(self, date: str | None) -> str:
        return date or self.default_date or datetime.now(CST).date().isoformat()

    def _overseas(self, date: str) -> dict:
        try:
            from quant_system.analysis_core.macro_overseas import get_macro_overseas
            return get_macro_overseas(date)
        except Exception as e:  # noqa: BLE001
            return {"date": date, "mode": "数据不可用", "assets": {}, "flags": {},
                    "evidence": [f"macro_overseas 异常: {str(e)[:120]}"]}

    def detect(self, date: str | None = None, with_rag: bool = True) -> dict:
        date = self._date(date)
        ref = pd.Timestamp(date[:10])
        overseas = self._overseas(date)

        drivers = {
            "growth": _driver_growth(ref),
            "inflation": _driver_inflation(ref),
            "liquidity": _driver_liquidity(ref),
            "risk_appetite": _driver_risk(ref, overseas, _fusion_temperature(ref)),
        }
        # 消费信心驱动（macro_learner 实证 CCI<88 消费板块承压, p<0.05）。
        # 缺失/异常 → None，不进入 drivers，不影响其他驱动（降级闭环）。
        cci_driver = _driver_consumer_confidence(ref)
        if cci_driver is not None:
            cci_adj = cci_driver.get("adjustment") or 0.0
            if cci_adj:
                ra = drivers["risk_appetite"]
                if ra.get("score") is not None:
                    ra["score"] = round(ra["score"] + cci_adj, 3)
                    if ra["score"] >= 0.35:
                        ra["state"] = "强"
                    elif ra["score"] <= -0.35:
                        ra["state"] = "弱"
                    else:
                        ra["state"] = "中性"
                mom_txt = (f"环比{cci_driver['mom']:+.1f}"
                           if cci_driver.get("mom") is not None else "环比数据不足")
                ra["note"] = (ra.get("note") or "") + (
                    f"；消费信心{cci_driver['state']}(CCI {cci_driver['cci']}, {mom_txt}) → "
                    f"风险偏好{'+' if cci_adj > 0 else ''}{cci_adj:g}")
            drivers["consumer_confidence"] = cci_driver
        base_mode, base_ev = _base_mode(drivers)
        mode, cross_notes, boost = _cross_check(base_mode, drivers, overseas)
        mapping = MODE_MAP.get(mode) or MODE_MAP.get(base_mode) or MODE_MAP["中性过渡"]
        confidence = _confidence(drivers, boost)
        rag = _rag_basis(date) if with_rag else []

        evidence: list[str] = []
        for key, cn in (("growth", "增长"), ("inflation", "通胀"),
                        ("liquidity", "流动性"), ("risk_appetite", "风险偏好")):
            dv = drivers[key]
            if dv["state"] != "数据缺失":
                evidence.append(f"{cn}: {dv['state']}（{'；'.join(e for e in dv['evidence'] if '→' in e) or '见明细'}）")
        if cci_driver is not None:
            mom_txt = (f"环比{cci_driver['mom']:+.1f}"
                       if cci_driver.get("mom") is not None else "环比数据不足")
            evidence.append(f"消费信心: {cci_driver['state']}({cci_driver['cci']}, {mom_txt})")
        evidence.extend(base_ev)
        evidence.extend(cross_notes)

        # commodity-cyclical 交叉: 油价 → PPI 传导
        assets = overseas.get("assets") or {}
        oil = assets.get("oil") or {}
        if oil.get("chg5_pct") is not None and oil["chg5_pct"] > 1.5:
            evidence.append(f"油价5日 +{oil['chg5_pct']:.1f}% → 输入性通胀/资源盈利传导（commodity-cyclical）")

        warnings: list[str] = []
        stale = []
        for dv in drivers.values():
            for i in dv["indicators"]:
                if i["lag_days"] is not None and i["lag_days"] > STALE_DAYS:
                    stale.append(f"{i['name']}（as_of {i['as_of'].strftime('%Y-%m-%d')}，滞后{i['lag_days']}天）")
        if stale:
            warnings.append("陈旧数据不参与: " + "；".join(stale))
        missing = []
        for dv in drivers.values():
            for i in dv["indicators"]:
                if i["lag_days"] is None:
                    missing.append(i["name"])
        if missing:
            warnings.append("数据缺失: " + "、".join(missing) + "（不参与判定）")
        for dv in drivers.values():
            if dv["state"] == "数据缺失":
                warnings.append(f"{dv['name']} 驱动数据不可用，按中性处理")

        big_cycle = _big_cycle(ref)
        status = "ok" if all(d["state"] != "数据缺失" for d in drivers.values()) else "degraded"

        return {
            "date": date,
            "generated_at": datetime.now(CST).isoformat(timespec="seconds"),
            "status": status,
            "system": "体系5 宏观周期系统",
            "drivers": drivers,
            "base_mode": base_mode,
            "mode": mode,
            "confidence": confidence,
            "asset_mapping": mapping,
            "big_cycle": big_cycle,
            "macro_overseas": {
                "mode": overseas.get("mode"),
                "flags": overseas.get("flags"),
                "assets": {k: {kk: v.get(kk) for kk in ("name", "latest", "as_of", "chg5_pct")}
                           for k, v in (overseas.get("assets") or {}).items()},
            },
            "rag": rag,
            "evidence": evidence,
            "warnings": warnings,
            "missing": missing,
            "stale": stale,
        }

    def view(self, date: str | None = None) -> dict:
        """供 multi_agent/日报 引用的专家观点。"""
        d = self.detect(date)
        return {
            "agent": AGENT_NAME,
            "signal": VIEW_SIGNAL.get(d["mode"], "震荡"),
            "view": VIEW_SIGNAL.get(d["mode"], "震荡"),
            "confidence": d["confidence"],
            "evidence": d["evidence"],
            "weight": VIEW_WEIGHT,
            "status": "ok" if d["status"] == "ok" else "degraded",
            "detail": {
                "date": d["date"],
                "mode": d["mode"],
                "base_mode": d["base_mode"],
                "drivers": {k: v["state"] for k, v in d["drivers"].items()},
                "asset_mapping": d["asset_mapping"],
            },
        }

    def report(self, date: str | None = None, with_rag: bool = True) -> Path:
        """生成 generated/macro_report_{date}.md，返回文件路径。"""
        d = self.detect(date, with_rag=with_rag)
        md = _build_report(d)
        # 宏观规律（AI 自学习）章节：读 ≤date 最新 macro_learner_*.json（只读）
        try:
            from quant_system.analysis_core.macro_learner import load_latest_learner_result
            learner = load_latest_learner_result(d["date"])
        except Exception:  # noqa: BLE001  规律引擎挂 → 原报告逻辑不受影响
            learner = None
        md += "\n" + _macro_learner_section(learner)
        out = OUT_DIR / f"macro_report_{d['date']}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        return out


# ────────────────────────────────────────────
# Markdown 报告
# ────────────────────────────────────────────
def _driver_table(drivers: dict) -> str:
    lines = ["| 驱动 | 状态 | 指标明细 |", "|---|---|---|"]
    for key, cn in (("growth", "增长"), ("inflation", "通胀"),
                    ("liquidity", "流动性"), ("risk_appetite", "风险偏好"),
                    ("consumer_confidence", "消费信心")):
        if key not in drivers:
            continue  # 消费信心缺失/异常 → 该驱动不展示（降级闭环）
        dv = drivers[key]
        detail = "；".join(dv["evidence"]) or "无可用数据"
        lines.append(f"| {cn} | **{dv['state']}** | {detail} |")
        if dv["note"]:
            lines.append(f"| — | — | 📝 {dv['note']} |")
    return "\n".join(lines)


def _macro_learner_section(learner: dict | None) -> str:
    """宏观规律（AI 自学习）章节：只读 macro_learner_*.json，缺失显示未运行。"""
    lines = [
        "",
        "## 七、宏观规律（AI 自学习）",
        "",
    ]
    if not learner:
        lines.append("- 规律引擎未运行（generated/macro_learner_*.json 缺失，先跑 "
                     "`python3 -m quant_system.analysis_core.macro_learner --date <date>`）")
        return "\n".join(lines)
    c = learner.get("cci") or {}
    if c.get("latest_cci") is not None:
        lines.append(
            f"- 当前 CCI: **{c['latest_cci']}**（{c.get('latest_month')}，环比 {c.get('mom_chg')}，"
            f"历史分位 {c.get('pct_rank')}，相对均值 {c.get('vs_hist_mean')}）")
    th = learner.get("thresholds") or []
    if th:
        lines.append("- 触发阈值规律:")
        for t in th[:8]:
            direction = t.get("direction", "")
            sign = "≥" if direction == "高信心利好" else "<"
            lines.append(
                f"  - {t['sector']}: CCI{sign}{t.get('threshold')} → 下月超额"
                f"{max(t.get('pos_excess', 0), t.get('neg_excess', 0)) * 100:+.1f}% "
                f"(n={t.get('n_neg', 0)}/{t.get('n_pos', 0)}, p={t.get('p_value', 1):.2f}, {direction})")
    else:
        lines.append("- 阈值规律: 无（样本不足或未达显著性门槛）")
    adj = learner.get("sector_adjustments") or []
    if adj:
        lines.append("- 板块权重建议:")
        for a in adj[:8]:
            lines.append(f"  - **{a['sector']}**: factor {a.get('factor')} — {a.get('reason')}")
    else:
        lines.append("- 板块权重建议: 无（当前落点未触发或超额 <1.5%）")
    deg = learner.get("degraded") or []
    for x in deg[:5]:
        lines.append(f"- ⚠️ {x}")
    return "\n".join(lines)


def _build_report(d: dict) -> str:
    ov = d["macro_overseas"]
    m = d["asset_mapping"]
    lines = [
        f"# 🌏 宏观周期系统 {d['date']}",
        "",
        "> 体系5 宏观周期 | 方法论: macro-four-driver-asset-map（四驱动资产映射） + "
        "animal-spirits-risk-forecasting（动物精神风险定价） + "
        "commodity-cyclical-valuation（大宗周期→周期股） + big-cycle-empire（债务/信用大周期）",
        f"> 生成时间: {d.get('generated_at', '')} | 数据: 本地宏观数据仓 + macro_overseas + fusion 温度",
        "",
        "## 一、驱动状态（四驱动 + 消费信心）",
        "",
        _driver_table(d["drivers"]),
        "",
        "## 二、宏观模式判定（美林时钟简化 + 海外交叉）",
        "",
        f"**{d['mode']}**",
        f"- 基础判定（四驱动）: {d['base_mode']}",
        f"- 置信度: **{d['confidence']}**（数据新鲜度加权）",
    ]
    for e in d["evidence"]:
        lines.append(f"- {e}")
    lines += [
        "",
        "## 三、资产映射（A股板块偏好）",
        f"- 偏好: {'、'.join(m['prefer'])}",
        f"- 规避: {'、'.join(m['avoid'])}",
        f"- 逻辑: {m['note']}",
        f"- 全市场温度建议: **{m['temp_hint']}**（仅建议，不自动改 fusion）",
        "",
        "## 四、长周期债务/信用周期（big-cycle-empire）",
        f"- 定位: **{d['big_cycle']['label']}**",
        f"- 数据: {d['big_cycle']['summary']}",
        f"- 含义: {d['big_cycle']['note']}",
        "",
        "## 五、海外资产交叉（macro_overseas）",
        f"- 海外模式: {ov['mode']}",
    ]
    assets = ov["assets"] or {}
    for key, cn in (("gold", "黄金"), ("oil", "原油"), ("us10y", "美债10Y"), ("vix", "VIX")):
        a = assets.get(key) or {}
        if a.get("latest") is not None:
            chg = a.get("chg5_pct")
            chg_txt = f"{chg:+.2f}%" if chg is not None else "—"
            lines.append(f"- {cn}: {a['latest']}（5日 {chg_txt}）")
        else:
            lines.append(f"- {cn}: 不可用")
    lines += [
        "",
        "## 六、RAG 方法论依据",
        "",
    ]
    if d["rag"]:
        for h in d["rag"]:
            if "error" in h:
                lines.append(f"- ⚠️ RAG 检索异常: {h['error']}")
            else:
                lines.append(f"- [{h.get('score')}] ({h.get('cat', '')}) {h.get('file', '')}")
                lines.append(f"  {str(h.get('text', ''))[:120].replace(chr(10), ' ')}")
    else:
        lines.append("- RAG 未启用（--skip-rag）")
    lines += [
        "",
        "## ⚠️ 数据可用性与新鲜度",
        f"- 陈旧(>45天)不参与: {('；'.join(d['stale']) if d['stale'] else '无')}",
        f"- 缺失: {('、'.join(d['missing']) if d['missing'] else '无')}",
    ]
    for w in d["warnings"]:
        lines.append(f"- ⚠️ {w}")
    lines += [
        "",
        "---",
        "*V11 宏观周期系统自动生成 | 模式为概率判断（置信度加权），仅供仓位与板块参考，不构成投资建议*",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="体系5 宏观周期系统")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD，默认当日(CST)")
    ap.add_argument("--report", action="store_true", help="生成 generated/macro_report_{date}.md")
    ap.add_argument("--skip-rag", action="store_true", help="跳过 RAG 检索（更快）")
    args = ap.parse_args()

    ms = MacroSystem()
    result = ms.detect(args.date, with_rag=not args.skip_rag)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    if args.report:
        md = ms.report(args.date, with_rag=not args.skip_rag)
        fname = OUT_DIR / f"macro_report_{result['date']}.md"
        print(f"\n✅ 报告已生成: {fname}")
