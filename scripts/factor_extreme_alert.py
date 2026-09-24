#!/usr/bin/env python3
"""factor_extreme_alert.py — 多因子极端提醒引擎 (V1)

设计目标（用户硬性要求）:
- 结合众多因子考虑：哪怕影响因子权重很小，但是出现了极端，也要提醒
- 盘中预警不能只靠一两根均线/指标，要用多因子体系作为盘中和复盘的依据

因子体系（5 大类 50+ 因子）:
  1. 技术面: MA乖离(20/60/144/300)、MACD、RSI(6/14)、KDJ、CCI、BOLL带宽/位置、
     ATR波动、量比、换手、突破新高新低、RSRS、ADX趋势强度、OBV背离
  2. 资金面: 主力净流入(近5/10日)、北向持股变动、两融余额变动、大宗折溢价、龙虎榜
  3. 估值面: PE/PB/PS 分位(3年)、PE/PB 绝对极值、股息率、PEG、总市值
  4. 财务面: ROE、毛利率、营收/净利增速、负债率、经营现金流、存货周转
  5. 情绪/事件面: 涨停/跌停状态、炸板、换手爆量、连板高度、大宗减持、解禁临近

提醒逻辑:
- 每个因子定义: 方向(direction) + 极端条件(quantile 或阈值)
- 任一项达极端(如历史分位 ≥95% 或 ≤5%，或 z-score ≥2.5) → 生成提醒
- 不因权重小而忽略：所有因子独立判断，极端即提醒，但提醒标注权重级别

输出: dict(score, level, extremes[], signals[], reasons[], advice)
"""

from __future__ import annotations
import json
import logging

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WH = ROOT / "data_warehouse"


# ─────────────────────────────────────────────────────────────
# 因子定义
# ─────────────────────────────────────────────────────────────

@dataclass
class FactorDef:
    """单个因子定义。direction: 1=越高越危险/超买, -1=越低越危险/超卖
    check_hi: 是否检测高分位/上限（超买类因子开，超卖类因子关）
    check_lo: 是否检测低分位/下限（超卖类因子开，超买类因子关）"""
    name: str                 # 因子名（中文）
    category: str             # 大类
    weight: float             # 权重（用于评分，但极端提醒不依赖权重）
    direction: int            # 1 / -1
    extreme_hi: float | None = None   # 绝对阈值上限（如 RSI>85）
    extreme_lo: float | None = None   # 绝对阈值下限（如 RSI<15）
    pct_hi: float = 0.95      # 历史分位上限（95%）
    pct_lo: float = 0.05      # 历史分位下限（5%）
    z_hi: float | None = 2.5  # z-score 上限
    z_lo: float | None = -2.5 # z-score 下限
    check_hi: bool = True     # 是否检测高位极端
    check_lo: bool = True     # 是否检测低位极端
    desc: str = ""            # 说明
    hi_label: str = "高位极端"  # 高位语义（振荡指标使用“超买”）
    lo_label: str = "低位极端"  # 低位语义（振荡指标使用“超卖”）
    percentile_hi_floor: float | None = None  # 分位报警还必须达到的实际强度
    percentile_lo_ceiling: float | None = None # 低分位报警还必须低于的实际强度


# 日线技术因子（从 add_technical_indicators 输出取）
TECH_FACTORS: list[FactorDef] = [
    FactorDef("RSI14", "技术面", 3.0, 1, extreme_hi=80, extreme_lo=20, desc="相对强弱指标（>80超买 <20超卖）", hi_label="超买", lo_label="超卖"),
    FactorDef("RSI6", "技术面", 2.5, 1, extreme_hi=85, extreme_lo=15, desc="短期相对强弱（>85超买 <15超卖）", hi_label="超买", lo_label="超卖"),
    FactorDef("KDJ_J", "技术面", 2.0, 1, extreme_hi=100, extreme_lo=-10, desc="随机指标J值（>100超买 <-10超卖）", hi_label="超买", lo_label="超卖"),
    FactorDef("CCI20", "技术面", 1.5, 1, extreme_hi=200, extreme_lo=-200, desc="顺势指标（>200超买 <-200超卖）", hi_label="超买", lo_label="超卖"),
    FactorDef("MACD柱 极端", "技术面", 2.0, 1, z_hi=3.0, z_lo=-3.0, desc="MACD柱状图"),
    FactorDef("BOLL带宽 极端", "技术面", 1.5, 1, pct_hi=0.97, check_lo=False, desc="布林带宽度(波动率)"),
    FactorDef("BOLL位置", "技术面", 1.0, 1, extreme_hi=1.05, extreme_lo=-0.05, desc="价格在布林带内位置"),
    FactorDef("MA20乖离", "技术面", 2.0, 1, pct_hi=0.98, pct_lo=0.02, desc="收盘价偏离20日均线（正=超买，负=超卖）"),
    FactorDef("MA60乖离", "技术面", 1.5, 1, pct_hi=0.98, pct_lo=0.02, desc="收盘价偏离60日均线"),
    FactorDef("MA144乖离", "技术面", 1.0, 1, pct_hi=0.98, pct_lo=0.02, desc="收盘价偏离144日均线"),
    FactorDef("MA300乖离", "技术面", 1.0, 1, pct_hi=0.98, pct_lo=0.02, desc="收盘价偏离300日均线"),
    FactorDef("量比 极端", "技术面", 2.5, 1, extreme_hi=5.0, check_lo=False,
              pct_hi=0.99, z_hi=3.0, percentile_hi_floor=3.0,
              desc="当日成交量/20日均量；分位报警需同时达到量比3倍"),
    FactorDef("换手率 爆量", "技术面", 2.0, 1, pct_hi=0.97, check_lo=False, desc="换手率历史分位"),
    FactorDef("ATR波动 极端", "技术面", 1.5, 1, pct_hi=0.97, check_lo=False, desc="真实波幅"),
    FactorDef("RSRS_z 极端", "技术面", 1.0, 1, z_hi=2.0, z_lo=-2.0, desc="RSRS斜率z分位"),
    FactorDef("60日新高突破", "技术面", 2.0, 1, extreme_hi=0.5, check_lo=False, desc="突破60日新高(1=突破)"),
    FactorDef("60日新低", "技术面", 2.0, -1, extreme_hi=0.5, check_lo=False, desc="创60日新低(1=破位)"),
    FactorDef("5日跌幅 极端", "技术面", 2.0, -1, extreme_lo=-15.0, check_hi=False, desc="5日累计跌幅超15%"),
    FactorDef("20日跌幅 极端", "技术面", 2.0, -1, extreme_lo=-25.0, check_hi=False, desc="20日累计跌幅超25%"),
    FactorDef("OBV动量背离", "技术面", 1.0, -1, extreme_hi=0.5, check_lo=False, desc="OBV与价格背离(价升量缩)" ),
]

# 资金面（需要 events/ 数据）
FUND_FACTORS: list[FactorDef] = [
    FactorDef("主力净流入5日", "资金面", 3.0, 1, pct_hi=0.95, pct_lo=0.05, desc="近5日主力资金净流入"),
    FactorDef("主力净流入10日", "资金面", 3.0, 1, pct_hi=0.95, pct_lo=0.05, desc="近10日主力资金净流入"),
    FactorDef("北向持股变动", "资金面", 2.5, 1, pct_hi=0.95, pct_lo=0.05, desc="北向资金持股比例变动"),
    FactorDef("两融余额变动", "资金面", 2.0, 1, pct_hi=0.95, pct_lo=0.05, desc="融资余额环比变动"),
    FactorDef("大宗交易折溢价", "资金面", 1.5, -1, extreme_hi=0.5, check_lo=False, desc="大宗交易折价率极端(折价>8%)"),
    FactorDef("龙虎榜净买入", "资金面", 2.0, 1, extreme_hi=0.5, check_lo=False, desc="龙虎榜机构/游资净买入"),
    FactorDef("机构龙虎榜净买", "资金面", 2.5, 1, extreme_hi=0.5, check_lo=False, desc="机构席位龙虎榜净买入(>0触发)"),
    FactorDef("股东户数变化", "资金面", 1.5, -1, extreme_lo=-20.0, check_hi=False, desc="股东户数减少>20%筹码集中(负值警惕)"),
]

# 估值面（valuation/）
VAL_FACTORS: list[FactorDef] = [
    FactorDef("PE_TTM 分位", "估值面", 3.0, 1, pct_hi=0.95, pct_lo=0.05, desc="PE_TTM 三年分位"),
    FactorDef("PB 分位", "估值面", 2.5, 1, pct_hi=0.95, pct_lo=0.05, desc="PB 三年分位"),
    FactorDef("PS 分位", "估值面", 1.5, 1, pct_hi=0.95, pct_lo=0.05, desc="PS 三年分位"),
    FactorDef("PE_TTM 绝对", "估值面", 1.0, 1, extreme_hi=150.0, desc="PE_TTM 绝对极值"),
    FactorDef("PB 绝对", "估值面", 1.0, 1, extreme_hi=20.0, desc="PB 绝对极值"),
    FactorDef("股息率 极端", "估值面", 1.0, -1, pct_hi=0.95, desc="股息率极端高(可能陷阱)"),
]

# 财务面（financial/）
FIN_FACTORS: list[FactorDef] = [
    FactorDef("ROE 极端", "财务面", 2.5, -1, extreme_hi=40.0, desc="净资产收益率(>40%需警惕)"),
    FactorDef("毛利率 极端", "财务面", 1.5, -1, extreme_hi=90.0, desc="毛利率"),
    FactorDef("营收增速 极端", "财务面", 2.0, -1, extreme_hi=100.0, extreme_lo=-50.0, desc="主营业务收入增长率"),
    FactorDef("净利增速 极端", "财务面", 2.0, -1, extreme_hi=200.0, extreme_lo=-80.0, desc="净利润增长率"),
    FactorDef("资产负债率 极端", "财务面", 1.5, -1, extreme_hi=85.0, desc="资产负债率>85%"),
    FactorDef("经营现金流/净利", "财务面", 1.0, -1, extreme_lo=-0.5, desc="现金流质量差"),
]

# 情绪/事件面（market/zt_pool 等）
# C4-fix (2026-08-07): 二值因子（涨停/跌停/炸板/OBV背离/大宗折溢价）无阈值无hist时
#   _check_factor 恒 False 永不触发 → 全部补 extreme_hi=0.5（value=1 即 >0.5 触发），
#   同时 _check_factor 增加 value==1.0 的布尔触发分支（兼容极端情况）。
SENTI_FACTORS: list[FactorDef] = [
    FactorDef("涨停状态", "情绪面", 3.0, 1, extreme_hi=0.5, check_lo=False, desc="当日涨停(高风险提示)"),
    FactorDef("跌停状态", "情绪面", 3.0, -1, extreme_hi=0.5, check_lo=False, desc="当日跌停"),
    FactorDef("连板高度", "情绪面", 2.0, 1, extreme_hi=2.5, desc="连板≥3(妖股风险)"),
    FactorDef("炸板", "情绪面", 1.5, -1, extreme_hi=0.5, check_lo=False, desc="曾涨停后炸板"),
]

ALL_FACTORS = TECH_FACTORS + FUND_FACTORS + VAL_FACTORS + FIN_FACTORS + SENTI_FACTORS


# ─────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────

def load_kline(symbol: str, days: int = 800) -> pd.DataFrame:
    """加载日K线（优先 data_warehouse，其次 quant_v6 缓存）"""
    p = WH / "kline" / f"{symbol}.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").tail(days).reset_index(drop=True)
    return pd.DataFrame()


def load_valuation(symbol: str, days: int = 800) -> pd.DataFrame:
    p = WH / "valuation" / f"{symbol}.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"])
        # 同名新旧 schema 并存时逐行合并。仓库中旧列可能全 NaN，不能删除
        # 有值的新列，否则会把真实估值数据误报为缺失。
        for canonical, alternate in (("pe_ttm", "peTTM"), ("pb", "pbMRQ"), ("ps", "psTTM"), ("pcf", "pcfNcfTTM")):
            if canonical in df.columns and alternate in df.columns:
                df[canonical] = pd.to_numeric(df[canonical], errors="coerce").combine_first(
                    pd.to_numeric(df[alternate], errors="coerce")
                )
                df = df.drop(columns=[alternate])
            elif alternate in df.columns:
                df = df.rename(columns={alternate: canonical})
            if canonical in df.columns:
                df[canonical] = pd.to_numeric(df[canonical], errors="coerce")
        return df.sort_values("date").tail(days).reset_index(drop=True)
    return pd.DataFrame()


def load_financial(symbol: str) -> pd.DataFrame:
    p = WH / "financial" / f"{symbol}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    # 正常 schema 是"选项=指标名, 报告期=列"。转置/损坏文件里"日期"列
    # 也会被硬编码成 NaN，必须拒绝而不是让所有财务因子静默缺失。
    if "日期" not in df.columns:
        return pd.DataFrame()
    if df["日期"].isna().all():
        return pd.DataFrame()
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    df = df.dropna(subset=["日期"])
    # 转置后"指标作为列、日期作为行"的变体里，部分指标名字符串
    # 出现在列名中的情况也要能正确回填，避免把宽表当普通时间序列错读。
    return df.sort_values("日期").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# 技术指标计算（精简版，避免依赖 quant_system 环境）
# ─────────────────────────────────────────────────────────────

def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    return dif, dea, (dif - dea) * 2


def _kdj(df: pd.DataFrame, window: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    low_n = df["low"].rolling(window).min()
    high_n = df["high"].rolling(window).max()
    rsv = (df["close"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """给日K加技术指标（与 quant_system.indicators 对齐的精简版）"""
    out = df.copy().sort_values("date").reset_index(drop=True)
    close = out["close"]
    for w in (5, 10, 20, 60, 144, 300):
        out[f"ma{w}"] = close.rolling(w).mean()
    out["volume_ma20"] = out["volume"].rolling(20).mean()
    out["volume_ratio"] = out["volume"] / out["volume_ma20"].replace(0, np.nan)
    out["rsi_6"] = _rsi(close, 6)
    out["rsi_14"] = _rsi(close, 14)
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    out["boll_mid"] = mid
    out["boll_upper"] = mid + 2 * std
    out["boll_lower"] = mid - 2 * std
    out["boll_width_pct"] = (out["boll_upper"] - out["boll_lower"]) / mid.replace(0, np.nan) * 100
    out["boll_pos"] = (close - out["boll_lower"]) / (out["boll_upper"] - out["boll_lower"]).replace(0, np.nan)
    out["atr_14"] = _atr(out["high"], out["low"], close)
    out["atr_pct"] = out["atr_14"] / close * 100
    out["macd_dif"], out["macd_dea"], out["macd_hist"] = _macd(close)
    out["kdj_k"], out["kdj_d"], out["kdj_j"] = _kdj(out)
    tp = (out["high"] + out["low"] + close) / 3
    tp_ma = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    out["cci_20"] = (tp - tp_ma) / (tp_mad.replace(0, np.nan) * 0.015)
    out["pct_5"] = close.pct_change(5) * 100
    out["pct_20"] = close.pct_change(20) * 100
    out["drawdown_60"] = (close / close.rolling(60).max() - 1) * 100
    # 不含当日的新高新低（避免当日价格恒>自身低点的自引用）
    out["high_60"] = close.shift(1).rolling(60).max()
    out["low_60"] = close.shift(1).rolling(60).min()
    out["turnover"] = df.get("turnover", out["volume"] / 1e8)
    out["obv"] = ((close.diff().fillna(0).gt(0).astype(int) - close.diff().fillna(0).lt(0).astype(int)) * out["volume"].fillna(0)).cumsum()
    return out


# ─────────────────────────────────────────────────────────────
# 极端检测核心
# ─────────────────────────────────────────────────────────────

def _pct_rank(series: pd.Series, value: float) -> float:
    """value 在 series 中的分位（0-1）"""
    s = series.dropna()
    if len(s) < 20 or math.isnan(value):
        return 0.5
    return float((s < value).mean())


def _zscore(series: pd.Series, value: float) -> float:
    s = series.dropna()
    if len(s) < 20:
        return 0.0
    std = s.std()
    if std is None or math.isnan(std) or std < 1e-12:
        return 0.0
    return float((value - s.mean()) / std)


def _check_factor(f: FactorDef, value: float | None, hist: pd.Series | None) -> tuple[bool, str | None]:
    """检查单因子是否极端。返回 (是否极端, 提醒文本) 只检测该因子方向对应的极端"""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False, None
    if hist is not None and len(hist.dropna()) >= 20:
        pct = _pct_rank(hist, value)
        z = _zscore(hist, value)
        hi_strength_ok = f.percentile_hi_floor is None or value >= f.percentile_hi_floor
        lo_strength_ok = f.percentile_lo_ceiling is None or value <= f.percentile_lo_ceiling
        if f.check_hi and f.pct_hi and pct >= f.pct_hi and hi_strength_ok:
            return True, f"{f.name}: {value:.2f}（{f.hi_label}，历史分位 {pct*100:.0f}%，超 {f.pct_hi*100:.0f}% 极端）"
        if f.check_lo and f.pct_lo and pct <= f.pct_lo and lo_strength_ok:
            return True, f"{f.name}: {value:.2f}（{f.lo_label}，历史分位 {pct*100:.0f}%，低于 {f.pct_lo*100:.0f}% 极端）"
        if f.check_hi and f.z_hi is not None and z >= f.z_hi and hi_strength_ok:
            return True, f"{f.name}: {value:.2f}（{f.hi_label}，z-score {z:.1f}，超 {f.z_hi:.1f}σ）"
        if f.check_lo and f.z_lo is not None and z <= f.z_lo and lo_strength_ok:
            return True, f"{f.name}: {value:.2f}（{f.lo_label}，z-score {z:.1f}，低于 {f.z_lo:.1f}σ）"
    if f.check_hi and f.extreme_hi is not None and value > f.extreme_hi:
        return True, f"{f.name}: {value:.2f}（{f.hi_label}，超上限 {f.extreme_hi:.2f}）"
    if f.check_lo and f.extreme_lo is not None and value < f.extreme_lo:
        return True, f"{f.name}: {value:.2f}（{f.lo_label}，破下限 {f.extreme_lo:.2f}）"
    # C4-fix: 二值因子布尔触发（涨停/跌停/炸板/OBV背离等 value∈{0,1}）
    #   即使 hist 不足 20 个样本，value==1.0 也视为触发
    if f.extreme_hi == 0.5 and value >= 1.0:
        return True, f"{f.name}: 触发（{value:.0f}）"
    if f.extreme_lo == 0.5 and value >= 1.0:
        return True, f"{f.name}: 触发（{value:.0f}）"
    return False, None


def compute_factors(symbol: str) -> dict[str, Any]:
    """计算一只股票的全部因子 + 极端检测"""
    kline = load_kline(symbol)
    if kline.empty:
        return {"symbol": symbol, "ok": False, "error": "无K线数据"}

    ind = add_indicators(kline)
    last = ind.iloc[-1]
    hist = ind.iloc[:-1]  # 极端检测用历史数据（不含当日，避免自引用）

    vals: dict[str, float] = {}
    hist_series: dict[str, pd.Series] = {}

    # 技术面
    for ma in (20, 60, 144, 300):
        key = f"MA{ma}乖离"
        vals[key] = float((last["close"] / last[f"ma{ma}"] - 1) * 100) if not math.isnan(last[f"ma{ma}"]) else np.nan
        hist_series[key] = (hist["close"] / hist[f"ma{ma}"] - 1) * 100
    vals["RSI14"] = float(last["rsi_14"]) if not math.isnan(last["rsi_14"]) else np.nan
    hist_series["RSI14"] = hist["rsi_14"]
    vals["RSI6"] = float(last["rsi_6"]) if not math.isnan(last["rsi_6"]) else np.nan
    hist_series["RSI6"] = hist["rsi_6"]
    vals["KDJ_J"] = float(last["kdj_j"]) if not math.isnan(last["kdj_j"]) else np.nan
    hist_series["KDJ_J"] = hist["kdj_j"]
    vals["CCI20"] = float(last["cci_20"]) if not math.isnan(last["cci_20"]) else np.nan
    hist_series["CCI20"] = hist["cci_20"]
    vals["MACD柱 极端"] = float(last["macd_hist"]) if not math.isnan(last["macd_hist"]) else np.nan
    hist_series["MACD柱 极端"] = hist["macd_hist"]
    vals["BOLL带宽 极端"] = float(last["boll_width_pct"]) if not math.isnan(last["boll_width_pct"]) else np.nan
    hist_series["BOLL带宽 极端"] = hist["boll_width_pct"]
    vals["BOLL位置"] = float(last["boll_pos"]) if not math.isnan(last["boll_pos"]) else np.nan
    hist_series["BOLL位置"] = hist["boll_pos"]
    vals["量比 极端"] = float(last["volume_ratio"]) if not math.isnan(last["volume_ratio"]) else np.nan
    hist_series["量比 极端"] = hist["volume_ratio"]
    vals["换手率 爆量"] = float(last["turnover"]) if not math.isnan(last["turnover"]) else np.nan
    hist_series["换手率 爆量"] = hist["turnover"]
    vals["ATR波动 极端"] = float(last["atr_pct"]) if not math.isnan(last["atr_pct"]) else np.nan
    hist_series["ATR波动 极端"] = hist["atr_pct"]
    vals["5日跌幅 极端"] = float(last["pct_5"]) if not math.isnan(last["pct_5"]) else np.nan
    hist_series["5日跌幅 极端"] = hist["pct_5"]
    vals["20日跌幅 极端"] = float(last["pct_20"]) if not math.isnan(last["pct_20"]) else np.nan
    hist_series["20日跌幅 极端"] = hist["pct_20"]
    vals["60日新高突破"] = 1.0 if (not math.isnan(last["high_60"]) and last["close"] >= last["high_60"]) else 0.0
    vals["60日新低"] = 1.0 if (not math.isnan(last["low_60"]) and last["close"] <= last["low_60"]) else 0.0
    # RSRS 简化（20日线性回归斜率）
    # C1-fix (2026-08-07): V5.4 当前=全序列 polyfit、历史=20日滚动 → 量纲不一致，
    #   趋势股永远报警。改为当前值也取 20 日滚动斜率的最后一个，与历史同口径。
    if len(ind) >= 40:
        hh = ind["high"].rolling(20).max().dropna()
        ll = ind["low"].rolling(20).min().dropna()
        rs = (hh / ll.replace(0, np.nan)).dropna()
        if len(rs) >= 30:
            slope_hist = rs.rolling(20).apply(lambda s: np.polyfit(np.arange(len(s)), s, 1)[0] if len(s) >= 10 else np.nan, raw=True).dropna()
            vals["RSRS_z 极端"] = float(slope_hist.iloc[-1]) if len(slope_hist) else np.nan
            hist_series["RSRS_z 极端"] = slope_hist.iloc[:-1] if len(slope_hist) > 1 else slope_hist
    # OBV 动量（近20日 OBV 斜率与价格斜率方向）
    try:
        if len(ind) >= 40:
            obv20 = ind["obv"].tail(20)
            close20 = ind["close"].tail(20)
            obv_slope = np.polyfit(np.arange(len(obv20)), obv20, 1)[0]
            price_slope = np.polyfit(np.arange(len(close20)), close20, 1)[0]
            if price_slope > 0 and obv_slope <= 0:
                vals["OBV动量背离"] = 1.0  # 价升量缩背离
            elif price_slope < 0 and obv_slope >= 0:
                vals["OBV动量背离"] = -1.0  # 价跌量增（放量下跌）
            else:
                vals["OBV动量背离"] = 0.0
    except Exception as e:
        logging.getLogger(__name__).error(f"[factor_extreme_alert] 操作失败: {e}", exc_info=True)

    # 估值面
    val = load_valuation(symbol)
    valuation_schema_available = False
    if not val.empty:
        vlast = val.iloc[-1]
        for col, fname in (("pe_ttm", "PE_TTM 分位"), ("pb", "PB 分位"), ("ps", "PS 分位")):
            if col not in val.columns:
                continue
            valid_values = pd.to_numeric(val[col], errors="coerce").dropna()
            if valid_values.empty:
                continue
            valuation_schema_available = True
            current = float(valid_values.iloc[-1])
            vals[fname] = current
            history_values = valid_values.iloc[:-1]
            if len(history_values) >= 60:
                hist_series[fname] = history_values
        if "PE_TTM 分位" in vals:
            vals["PE_TTM 绝对"] = vals["PE_TTM 分位"]
        if "PB 分位" in vals:
            vals["PB 绝对"] = vals["PB 分位"]
        if "dv_ttm" in val.columns:
            dividends = pd.to_numeric(val["dv_ttm"], errors="coerce").dropna()
            if not dividends.empty:
                vals["股息率 极端"] = float(dividends.iloc[-1])
                hist_series["股息率 极端"] = dividends.iloc[:-1]

    # 财务面。宽表按"选项=指标"保存时，当前取每列最近非空值；
    # 转置/异常 schema 已被 load_financial 拒绝，不会在此误报缺失。
    fin = load_financial(symbol)
    if not fin.empty:
        fin_map = {
            "净资产收益率(%)": "ROE 极端",
            "销售毛利率(%)": "毛利率 极端",
            "主营业务收入增长率(%)": "营收增速 极端",
            "净利润增长率(%)": "净利增速 极端",
            "资产负债率(%)": "资产负债率 极端",
            "经营现金净流量与净利润的比率(%)": "经营现金流/净利",
        }
        fin = fin.apply(lambda s: pd.to_numeric(s, errors="coerce"))
        for col, fname in fin_map.items():
            if col not in fin.columns:
                continue
            recent = fin[col].dropna()
            if recent.empty:
                continue
            value = float(recent.iloc[-1])
            if math.isfinite(value):
                vals[fname] = value

    # 情绪面：涨停池
    # C2a-fix (2026-08-07): zt_pool 文件名=日期、无日期列，V5.4 取 zt_files[-1]
    #   （文件系统排序最后）不与 K 线最后日期对齐 → 虚假“当日涨停”或漏报。
    #   改为：按 K 线最后日期定位文件，取 ≤ 该日期的最后一个文件；
    #   找不到则回退到文件列表最后一个（数据缺口时保持可用）。
    try:
        zt_dir = WH / "market" / "zt_pool"
        if zt_dir.is_dir():
            zt_files = sorted(zt_dir.glob("*.parquet"))
            if zt_files:
                target_file = zt_files[-1]
                if len(kline) > 0:
                    k_last = pd.to_datetime(kline["date"]).iloc[-1]
                    k_last_s = k_last.strftime("%Y%m%d")
                    le = [f for f in zt_files if f.stem <= k_last_s]
                    if le:
                        target_file = le[-1]
                latest_zt = pd.read_parquet(target_file)
                code_col = "代码" if "代码" in latest_zt.columns else ("证券代码" if "证券代码" in latest_zt.columns else None)
                if code_col:
                    # C2b-fix: 代码列可能带 .0（"600519.0"）→ 去后缀再补零
                    codes = latest_zt[code_col].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
                    if symbol in codes.values:
                        vals["涨停状态"] = 1.0
                        # 连板高度
                        for col in ("连板数", "连板", "几天几板"):
                            if col in latest_zt.columns:
                                row = latest_zt[codes == symbol]
                                vals["连板高度"] = float(row[col].iloc[0]) if len(row) else 0.0
                                break
                        # C4-fix: 炸板 — 涨停池内炸板次数>0 视为“曾涨停后炸板”
                        #   （zt_pool 实测有“炸板次数”列；V5.4 无此逻辑恒不触发）
                        if "炸板次数" in latest_zt.columns:
                            row = latest_zt[codes == symbol]
                            if len(row) and float(row["炸板次数"].iloc[0]) > 0:
                                vals["炸板"] = 1.0
    except Exception as e:
        logging.getLogger(__name__).error(f"[factor_extreme_alert] 操作失败: {e}", exc_info=True)

    # 情绪面：跌停池（zt_pool_dtgc）
    # C4-fix (2026-08-07): V5.4 只有 FactorDef 无计算逻辑 → 跌停状态恒不触发。
    #   按与涨停池相同的日期对齐规则读取 zt_pool_dtgc。
    try:
        dt_dir = WH / "market" / "zt_pool_dtgc"
        if dt_dir.is_dir():
            dt_files = sorted(dt_dir.glob("*.parquet"))
            if dt_files:
                target_file = dt_files[-1]
                if len(kline) > 0:
                    k_last = pd.to_datetime(kline["date"]).iloc[-1]
                    k_last_s = k_last.strftime("%Y%m%d")
                    le = [f for f in dt_files if f.stem <= k_last_s]
                    if le:
                        target_file = le[-1]
                latest_dt = pd.read_parquet(target_file)
                code_col = "代码" if "代码" in latest_dt.columns else ("证券代码" if "证券代码" in latest_dt.columns else None)
                if code_col:
                    # C2b-fix: 代码列可能带 .0 → 去后缀再补零
                    codes = latest_dt[code_col].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
                    if symbol in codes.values:
                        vals["跌停状态"] = 1.0
    except Exception as e:
        logging.getLogger(__name__).error(f"[factor_extreme_alert] 操作失败: {e}", exc_info=True)

    # 资金面：大宗交易折溢价
    # M3-fix (2026-08-07): block.parquet 实测无折溢价率列 → 用成交价 vs 当日收盘价自算
    try:
        blk = WH / "events" / "block.parquet"
        if blk.exists():
            b = pd.read_parquet(blk)
            code_col = "证券代码" if "证券代码" in b.columns else ("代码" if "代码" in b.columns else None)
            if code_col:
                b["_c"] = b[code_col].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
                bsym = b[b["_c"] == symbol]
                if not bsym.empty:
                    got = False
                    for col in ("折溢价率(%)", "折溢价率"):
                        if col in bsym.columns:
                            vals["大宗交易折溢价"] = float(bsym[col].iloc[-1])
                            got = True
                            break
                    if not got and len(bsym) and "成交价" in bsym.columns and len(kline) > 0:
                        # M3-fix: 自算折溢价 = 成交价/当日收盘价 - 1（取最近一笔）
                        try:
                            kc = kline.set_index(pd.to_datetime(kline["date"]))["close"]
                            last_row = bsym.iloc[-1]
                            t = pd.to_datetime(last_row.get("交易日期"))
                            if t in kc.index:
                                px = float(last_row["成交价"])
                                close_px = float(kc.loc[t])
                                if close_px > 0:
                                    vals["大宗交易折溢价"] = (px / close_px - 1) * 100.0
                        except Exception as e:
                            logging.getLogger(__name__).error(f"[factor_extreme_alert] 操作失败: {e}", exc_info=True)
    except Exception as e:
        logging.getLogger(__name__).error(f"[factor_extreme_alert] 操作失败: {e}", exc_info=True)

    # C5-fix (2026-08-07): 资金面 5/6 因子从未计算（主力净流入/北向/两融/龙虎榜）——
    #   events/ 及 market/margin_detail_*、market/lhb_* 数据在但代码不读。
    #   以下接入：龙虎榜（近30日净买额）、两融（近5日融资余额变动）。
    #   主力净流入/北向：需个股级资金流数据，当前 data_warehouse 无个股资金流文件，
    #   留空（hist 不足时 _check_factor 自动不触发，不会误报）。

    # 龙虎榜：近 30 日该股上榜净买额（lhb 季度文件）
    try:
        if len(kline) > 0:
            k_last_dt = pd.to_datetime(kline["date"]).iloc[-1]
            cutoff = k_last_dt - pd.Timedelta(days=30)
            lhb_files = sorted((WH / "market").glob("lhb_*.parquet"))
            lhb_rows = []
            for lf in lhb_files:
                try:
                    ldf = pd.read_parquet(lf)
                    if "上榜日" in ldf.columns and "代码" in ldf.columns:
                        ld = pd.to_datetime(ldf["上榜日"], errors="coerce")
                        ldf = ldf[(ld >= cutoff) & (ld <= k_last_dt)]
                        if len(ldf):
                            ldf = ldf.copy()
                            ldf["_c"] = ldf["代码"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
                            lhb_rows.append(ldf[ldf["_c"] == symbol])
                except Exception as e:
                    logging.getLogger(__name__).error(f"[factor_extreme_alert] 操作失败: {e}", exc_info=True)
                    continue
            if lhb_rows:
                lhb_all = pd.concat(lhb_rows, ignore_index=True)
                if len(lhb_all) and "龙虎榜净买额" in lhb_all.columns:
                    vals["龙虎榜净买入"] = 1.0 if float(lhb_all["龙虎榜净买额"].sum()) > 0 else 0.0
    except Exception as e:
        logging.getLogger(__name__).error(f"[factor_extreme_alert] 操作失败: {e}", exc_info=True)

    # 两融：近 5 个交易日融资余额变动（margin_detail_sz/sh 逐日文件）
    try:
        if len(kline) > 0:
            k_last_dt = pd.to_datetime(kline["date"]).iloc[-1]
            # 沪市 6xx → SH 文件；深市 0/3xx → SZ 文件
            mkt = "sh" if symbol.startswith("6") else "sz"
            mdir = WH / "market" / f"margin_detail_{mkt}"
            if mdir.is_dir():
                mfiles = sorted(mdir.glob("*.parquet"))
                # 取 ≤ K线最后日期的最近 6 个文件
                mfiles_le = [f for f in mfiles if f.stem <= k_last_dt.strftime("%Y%m%d")]
                recent = mfiles_le[-6:]
                balances = []
                for mf in recent:
                    try:
                        mdf = pd.read_parquet(mf)
                        if "证券代码" not in mdf.columns:
                            continue
                        mc = mdf["证券代码"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
                        row = mdf[mc == symbol]
                        if len(row) and "融资余额" in mdf.columns:
                            balances.append(float(row["融资余额"].iloc[0]))
                    except Exception as e:
                        logging.getLogger(__name__).error(f"[factor_extreme_alert] 操作失败: {e}", exc_info=True)
                        continue
                if len(balances) >= 3:
                    chg = (balances[-1] - balances[0]) / balances[0] if balances[0] > 0 else 0.0
                    vals["两融余额变动"] = chg * 100.0  # 百分比
    except Exception as e:
        logging.getLogger(__name__).error(f"[factor_extreme_alert] 操作失败: {e}", exc_info=True)

    # 机构龙虎榜净买（lhb_jgmmtj_em.parquet，近30日）
    try:
        if len(kline) > 0:
            k_last_dt = pd.to_datetime(kline["date"]).iloc[-1]
            cutoff = k_last_dt - pd.Timedelta(days=30)
            jgf = WH / "market" / "lhb_jgmmtj_em.parquet"
            if jgf.exists():
                jdf = pd.read_parquet(jgf)
                if "上榜日期" in jdf.columns and "代码" in jdf.columns and "机构买入净额" in jdf.columns:
                    jd = pd.to_datetime(jdf["上榜日期"], errors="coerce")
                    jdf = jdf[(jd >= cutoff) & (jd <= k_last_dt)]
                    if len(jdf):
                        jdf = jdf.copy()
                        jdf["_c"] = jdf["代码"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
                        row = jdf[jdf["_c"] == symbol]
                        if len(row):
                            vals["机构龙虎榜净买"] = 1.0 if float(row["机构买入净额"].sum()) > 0 else 0.0
    except Exception as e:
        logging.getLogger(__name__).error(f"[factor_extreme_alert] 操作失败: {e}", exc_info=True)

    # Prefer the refreshed focused-watchlist snapshot; fall back to the
    # all-market slow snapshot only when this symbol is not covered there.
    try:
        focused = ROOT / "generated" / "watchlist_holders.json"
        focused_hit = False
        if focused.exists():
            payload = json.loads(focused.read_text(encoding="utf-8"))
            for item in payload.get("items", []):
                if str(item.get("code", "")).zfill(6) == symbol and item.get("status") == "available":
                    value = item.get("change_pct")
                    if value is not None:
                        vals["股东户数变化"] = float(value)
                        focused_hit = True
                    break
        if not focused_hit:
            gdf = WH / "market" / "gdhs_all.parquet"
            if gdf.exists():
                gd = pd.read_parquet(gdf)
                if "代码" in gd.columns and "股东户数-增减比例" in gd.columns:
                    gc = gd["代码"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
                    row = gd[gc == symbol]
                    if len(row) and pd.notna(row["股东户数-增减比例"].iloc[0]):
                        vals["股东户数变化"] = float(row["股东户数-增减比例"].iloc[0])
    except Exception as e:
        logging.getLogger(__name__).error(f"[factor_extreme_alert] 操作失败: {e}", exc_info=True)

    # 极端检测。所有定义因子都返回；没有数据的因子明确标为 missing，避免
    # 页面把“本次有值数量”误报成“系统因子总数”。
    extremes: list[dict] = []
    signals: list[dict] = []
    for f in ALL_FACTORS:
        raw_value = vals.get(f.name)
        available = raw_value is not None and not (
            isinstance(raw_value, (float, np.floating)) and math.isnan(float(raw_value))
        )
        hit, msg = _check_factor(f, float(raw_value) if available else None, hist_series.get(f.name))
        value = round(float(raw_value), 3) if available else None
        if hit:
            extremes.append({
                "factor": f.name, "category": f.category, "weight": f.weight,
                "value": value, "message": msg, "direction": f.direction,
                "level": "high" if f.weight >= 2.5 else "medium",
            })
        signal = {
            "factor": f.name, "category": f.category, "weight": f.weight,
            "value": value, "available": available, "status": "available" if available else "missing",
            "extreme": hit, "desc": f.desc,
        }
        if available and f.hi_label == "超买" and f.lo_label == "超卖":
            if f.extreme_hi is not None and raw_value > f.extreme_hi:
                signal["state"] = "超买"
            elif f.extreme_lo is not None and raw_value < f.extreme_lo:
                signal["state"] = "超卖"
            else:
                signal["state"] = "中性"
        if not available:
            if f.category == "估值面" and val.empty:
                reason = "估值文件缺失"
            elif f.category == "估值面" and not valuation_schema_available:
                reason = "估值文件存在但有效字段为空或 schema 不兼容"
            elif f.category == "财务面" and fin.empty:
                reason = "财务文件缺失"
            elif f.category == "财务面" and "日期" in fin.columns and fin["日期"].isna().all():
                reason = "财务文件存在但为转置/异常 schema，禁止冒充有效因子"
            else:
                reason = "当前数据仓未提供所需字段、事件未触发或历史样本不足"
            signal["missing_reason"] = reason
        signals.append(signal)

    # 综合评分（0-100，越高越危险/超买）
    score = 50.0
    for e in extremes:
        score += e["weight"] * (3 if e["direction"] == 1 else 2)
    score = max(0.0, min(100.0, score))

    level = "危险" if score >= 75 else "警惕" if score >= 60 else "正常"
    if any(e["direction"] == -1 and e["weight"] >= 2.5 for e in extremes):
        level = "危险"

    return {
        "symbol": symbol,
        "ok": True,
        "date": str(last["date"].date()) if hasattr(last["date"], "date") else str(last["date"])[:10],
        "close": round(float(last["close"]), 2),
        "score": round(score, 1),
        "level": level,
        "extremes": extremes,
        "n_extremes": len(extremes),
        "factors": signals,
        "n_factors": len(signals),
        "n_factor_definitions": len(ALL_FACTORS),
        "n_available_factors": sum(1 for item in signals if item["available"]),
        "n_missing_factors": sum(1 for item in signals if not item["available"]),
        "factor_count_scope": "definitions",
    }


def analyze_batch(symbols: list[str], max_workers: int = 4) -> list[dict]:
    """批量分析（简单串行，避免依赖 concurrent 环境问题）"""
    results = []
    for s in symbols:
        try:
            r = compute_factors(s)
            if r.get("ok"):
                results.append(r)
        except Exception as e:  # noqa: BLE001
            results.append({"symbol": s, "ok": False, "error": str(e)})
    return results


def format_alert_text(r: dict) -> str:
    """把单只股票的极端提醒格式化为文本"""
    if not r.get("ok"):
        return f"{r.get('symbol')}: {r.get('error', '未知错误')}"
    lines = [f"📊 {r['symbol']} 多因子扫描 {r.get('date','')} — 综合{r.get('score')}分 [{r.get('level')}]"]
    if not r.get("extremes"):
        lines.append("  无极端因子")
    else:
        for e in r["extremes"]:
            icon = "🔴" if e["level"] == "high" else "🟡"
            lines.append(f"  {icon} [{e['category']}] {e['message']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="多因子极端提醒引擎")
    ap.add_argument("syms", nargs="*", default=None, help="股票代码列表（缺省读 watchlist.json）")
    ap.add_argument("--quiet", action="store_true",
                    help="仅输出有极端因子的股票（无极端则无输出，用于定时任务防刷屏）")
    args = ap.parse_args()
    syms = args.syms
    if not syms:
        # 2026-08-14: 默认自选清单（watchlist.json 列表格式），cron 不必硬编码股票
        try:
            import json as _json
            wl_path = Path(__file__).resolve().parents[1] / "quant_web" / "watchlist.json"
            data = _json.loads(wl_path.read_text(encoding="utf-8"))
            syms = [str(x) for x in data] if isinstance(data, list) else list(data.get("watch", []))
        except Exception:  # noqa: BLE001
            syms = ["002714"]
    for s in syms:
        res = compute_factors(s)
        if args.quiet and not (res.get("ok") and res.get("extremes")):
            continue
        print(format_alert_text(res))
        if res.get("ok"):
            print(f"  因子总数: {res['n_factors']}, 极端数: {res['n_extremes']}")
