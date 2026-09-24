#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""risk_system — 体系6 风险纪律系统（Elder 2%/6%规则 + 生存纪律 + 沉没成本 + Appel 盈亏比）

方法论依据（skills）:
  - elder-risk-management-journaling: 风险铁三角（止损/持仓规模/账户资金）、
    单笔风险 1%-2% 规则、月度回撤 6% 停手、止损=入场-ATR 倍数、单票仓位上限与分散化。
  - trade-survival-discipline: 生存第一、截断亏损让利润奔跑、胜率不重要盈亏比重要、
    单笔止损金额不超过账户 1%-2%、波动越大仓位越小。
  - sunk-cost-exit-discipline: 沉没成本不参与决策（买入价不是持有理由），
    按未来收益/风险决定去留，亏损深+无反转信号 → 纪律性离场。
  - appel-profit-loss-ratio-risk: 以利润损失比(PLR)评估信号质量，盈亏比不足拒绝交易/降仓。

输入:
  data_warehouse/kline/{6位代码}.parquet  日K（date/open/high/low/close/volume）
  data_warehouse/market/stock_names.parquet / quant_system/stock_name_map.json  代码-名称
  config/watchlist.json 或 --watch          自选池（缺省读 watchlist）
  --positions / config/positions.json       持仓（code: 建仓价），可选；缺省按"新开仓"口径

检测项:
  1. 个股止损纪律   止损位 = close - 1.5×ATR14（复用 watch_card 通道口径）；
                     当前价距止损距离%；close ≤ 止损 或 盘中 low ≤ 止损 → '违规持仓' 警告。
  2. 仓位建议       2%规则: 建议仓位金额 = 2%×资金/(入场-止损)×入场；
                     单票上限 20%；市场温度 < 40 → 总仓位半仓。
  3. 回撤检查       自选池近 20 日等权净值最大回撤；> 6% → '月度纪律触发,停手'（elder 6%规则）。
  4. 盈亏比检查     目标位 = max(60日密集区上沿, 60日前高)，盈亏比 = (目标-当前)/(当前-止损)；
                     < 1.5 → '盈亏比不足' 警告（Appel PLR）。
  5. 沉没成本检查   持仓亏损 > 15% 且无反转信号(close>MA5 且 close>MA10) → '沉没成本陷阱' 警告。
  6. 综合聚合       风险等级(低/中/高) + 操作建议(持有/减仓/离场/观望)；
                    违规持仓/回撤触发/沉没成本 → 高+离场；盈亏比不足 → 减仓；低温 → 观望。

RAG 依据: knowledge_rag.search('风险管理 止损 2%规则 生存', k=3) 作为方法论依据（失败降级标注）。

输出: generated/risk_report_{date}.md

用法:
  python3 -m quant_system.analysis_core.risk_system --watch 601899,603993,600362,002714
  python3 -m quant_system.analysis_core.risk_system --watch 601899,603993,600362,002714 \
      --date 2026-08-11 --capital 1000000 --positions '{"601899": 21.0, "002714": 60.0}'
"""

from __future__ import annotations
import logging

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent  # workspace（与 analysis_core.config 同约定）
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.watch_card import ATR_MULT, atr14  # noqa: E402  （复用 ATR/通道口径）
from quant_system.analysis_core.common import (  # noqa: E402
    fmt,
    load_names,
    read_kline_window,
)
from quant_system.analysis_core.volume_profile import calc_volume_profile  # noqa: E402
from quant_system.analysis_core import knowledge_rag  # noqa: E402

KLINE_DIR = ROOT / "data_warehouse" / "kline"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "generated"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "watchlist.json"
POSITIONS_PATH = Path(__file__).resolve().parent.parent / "config" / "positions.json"

CODE_RE = re.compile(r"^\d{6}$")
LOOKBACK_DAYS = 400            # 日历回看窗口（覆盖 60日前高 + ATR14 + 20日回撤窗口）

# ── 纪律参数（可配置）───────────────────────────────────────
DEFAULT_CAPITAL = 1_000_000    # 初始资金，默认 100 万
RISK_PER_TRADE = 0.02          # elder 2% 规则：单笔最大亏损 ≤ 资金 2%
MAX_POSITION_PCT = 0.20        # 单票仓位上限（默认 20%）
TEMP_HALF_THRESHOLD = 40       # 市场温度 < 40 → 总仓位半仓
DD_WINDOW = 20                 # 回撤检查窗口（交易日）
DD_TRIGGER = 0.06              # elder 6% 规则：近 20 日回撤 > 6% → 停手
PLR_MIN = 1.5                  # Appel 盈亏比下限
SUNK_LOSS_PCT = 0.15           # 持仓亏损 > 15% 触发沉没成本检查
PLR_DAYS = 60                  # 目标位：60 日密集区/前高窗口
RAG_QUERY = "风险管理 止损 2%规则 生存"
RAG_K = 3


# 综合聚合权重
CRIT_WEIGHT = 2                # 违规持仓 / 回撤触发 / 沉没成本
WARN_WEIGHT = 1                # 盈亏比不足 / 低温 / 深亏但出现反转信号
HIGH_SCORE = 4                 # 得分 ≥4 → 高风险


# ────────────────────────────────────────────────────────────
# 数据读取
# ────────────────────────────────────────────────────────────
def _load_kline(code: str, target: pd.Timestamp,
                window_start: pd.Timestamp | None = None) -> tuple[pd.DataFrame | None, str]:
    """日K（date/open/high/low/close/volume），截断至 target；缺失/异常标注错误。"""
    path = KLINE_DIR / f"{code}.parquet"
    if not path.exists():
        return None, "kline文件缺失"
    ws = (window_start if window_start is not None
          else pd.Timestamp(target) - pd.Timedelta(days=LOOKBACK_DAYS))
    try:
        df = read_kline_window(path, ["date", "open", "high", "low", "close", "volume"], ws)
    except Exception as e:
        return None, f"读取失败: {str(e)[:60]}"
    if df is None or df.empty:
        return None, "无数据"
    if not {"date", "close"}.issubset(df.columns):
        return None, "缺关键列"
    for c in ("open", "high", "low", "volume"):
        if c not in df.columns:
            df[c] = np.nan
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).drop_duplicates("date").sort_values("date")
    df = df[df["date"] <= pd.Timestamp(target).normalize()]
    if df.empty:
        return None, "目标日无交易"
    return df.reset_index(drop=True), ""


def _latest_common_day(codes: list[str]) -> pd.Timestamp:
    """自选池 kline 的最新共同交易日。"""
    maxima: list[pd.Timestamp] = []
    for code in codes:
        p = KLINE_DIR / f"{code}.parquet"
        if not p.exists():
            continue
        try:
            d = pd.read_parquet(p, columns=["date"])
            m = pd.to_datetime(d["date"], errors="coerce").max()
            if m is not None and pd.notna(m):
                maxima.append(m)
        except Exception as e:
            logging.getLogger(__name__).error(f"[risk_system] 操作失败: {e}", exc_info=True)
            continue
    return (min(maxima).normalize() if maxima else pd.Timestamp.now().normalize())


# ────────────────────────────────────────────────────────────
# 市场温度
# ────────────────────────────────────────────────────────────
def market_temperature(date: str | None = None) -> dict:
    """市场温度 0-100：取 emotion_system.detect(date) 的温度（无前视，失败标注降级）。"""
    try:
        from quant_system.analysis_core.emotion_system import EmotionSystem
        emo = EmotionSystem().detect(date)
        temp = emo.get("temperature")
        temp = float(temp) if temp is not None else 50.0
        band = emo.get("temp_band") or ""
        stage_cn = emo.get("stage_cn") or emo.get("stage") or ""
        # ok 判定以温度值可用且锚定 fusion 为准（其他辅助源降级不影响温度可靠性）
        ok = emo.get("temperature") is not None and emo.get("temperature_as_of") is not None
        return {"temperature": int(temp),
                "source": f"情绪周期温度({band}/{stage_cn})",
                "stage": emo.get("stage"), "ok": bool(ok)}
    except Exception as e:
        return {"temperature": 50, "source": f"温度不可用({str(e)[:60]})",
                "stage": None, "ok": False}


# ────────────────────────────────────────────────────────────
# 检测 1：个股止损纪律
# ────────────────────────────────────────────────────────────
def _check_stop(code: str, df: pd.DataFrame, close: float, atr: float) -> dict:
    """止损位 = close - 1.5×ATR14（watch_card 通道口径）；破位 → '违规持仓'。"""
    if not np.isfinite(atr) or atr <= 0:
        return {"ok": False, "note": "ATR不足, 无法计算止损",
                "stop": None, "dist_pct": None, "broken": False, "warning": None}
    stop = close - ATR_MULT * atr
    dist_pct = (close - stop) / close * 100.0 if close > 0 else np.nan
    low = float(df["low"].astype(float).iloc[-1]) if len(df) else np.nan
    close_broken = bool(close <= stop)
    # 盘中破位口径（与 watch_card 盘中触支撑一致）：low ≤ 止损 即视为已破
    low_broken = bool(np.isfinite(low) and low <= stop)
    broken = close_broken or low_broken
    warning = None
    if broken:
        kind = "收盘" if close_broken else "盘中最低"
        warning = {"type": "违规持仓", "level": "crit", "code": code,
                   "msg": f"{kind} {low if not close_broken else close:.2f} 已跌破止损位 "
                          f"{stop:.2f}（距止损 {dist_pct:.1f}%），止损纪律失效，禁止扛单（生存第一）"}
    return {"ok": True, "stop": stop, "dist_pct": dist_pct, "broken": broken,
            "close_broken": close_broken, "low_broken": low_broken, "warning": warning}


# ────────────────────────────────────────────────────────────
# 检测 2：仓位建议（2% 规则 + 单票上限 + 温度半仓）
# ────────────────────────────────────────────────────────────
def _position_advice(code: str, entry: float, close: float, stop: float | None,
                     capital: float, temp: int) -> dict:
    """建议仓位金额/股数：2%×资金/(入场-止损)×入场；上限 20%；温度<40 总仓位半仓。"""
    if stop is None or not np.isfinite(stop) or not np.isfinite(entry) or entry <= 0:
        return {"ok": False, "note": "止损/入场不可用, 无法按 2% 规则计算",
                "notional": 0.0, "pct": 0.0, "shares": 0}
    risk_per_share = entry - stop
    if risk_per_share <= 0:
        return {"ok": False, "note": "入场价≤止损价, 仓位规则失效",
                "notional": 0.0, "pct": 0.0, "shares": 0}
    max_loss = capital * RISK_PER_TRADE
    notional = max_loss / risk_per_share * entry          # 2% 规则反推
    pct = notional / capital
    capped = False
    if pct > MAX_POSITION_PCT:                            # 单票上限
        notional = capital * MAX_POSITION_PCT
        pct = MAX_POSITION_PCT
        capped = True
    shares = int(notional / entry // 100 * 100) if entry > 0 else 0  # A股 100 股整数倍
    return {"ok": True, "entry": entry, "risk_per_share": risk_per_share,
            "notional": notional, "pct": pct, "shares": shares,
            "capped": capped, "max_loss_yuan": max_loss}


def _total_position_advice(advices: list[dict], capital: float, temp: int) -> dict:
    """总仓位建议：建议金额合计；市场温度 < 40 → 半仓。"""
    valid = [a for a in advices if a.get("ok")]
    total = float(sum(a.get("notional", 0.0) for a in valid))
    half = bool(temp < TEMP_HALF_THRESHOLD)
    if half:
        total = min(total * 0.5, capital * 0.5)
    return {"ok": True, "total_notional": total,
            "total_pct": total / capital if capital > 0 else 0.0,
            "half_position": half,
            "note": f"市场温度 {temp} < {TEMP_HALF_THRESHOLD} → 总仓位半仓" if half
                    else f"市场温度 {temp} ≥ {TEMP_HALF_THRESHOLD} → 正常仓位"}


# ────────────────────────────────────────────────────────────
# 检测 3：自选池回撤（elder 6% 规则）
# ────────────────────────────────────────────────────────────
def _pool_drawdown(codes: list[str], target: pd.Timestamp,
                   window: int = DD_WINDOW) -> dict:
    """自选池近 window 个交易日等权净值最大回撤 + 各股最大回撤。"""
    closes: dict[str, pd.Series] = {}
    missing: list[str] = []
    for code in codes:
        df, err = _load_kline(code, target)
        if df is None:
            missing.append(f"{code}({err})")
            continue
        closes[code] = df.set_index("date")["close"].astype(float)
    if not closes:
        return {"ok": False, "note": f"无可用K线: {', '.join(missing)}",
                "pool_dd": None, "per_stock": {}, "triggered": False}
    panel = pd.DataFrame(closes).sort_index().tail(window)
    per_stock: dict[str, float] = {}
    for c in panel.columns:
        s = panel[c].dropna()
        if len(s) < 2:
            continue
        norm = s / s.iloc[0]
        per_stock[c] = float((norm / norm.cummax() - 1.0).min())
    idx_norm = panel / panel.iloc[0]
    idx = idx_norm.mean(axis=1)                      # 等权池净值（skipna 默认）
    idx = idx.dropna()
    if len(idx) < 2:
        return {"ok": False, "note": "对齐后样本不足", "pool_dd": None,
                "per_stock": per_stock, "triggered": False}
    pool_dd = float((idx / idx.cummax() - 1.0).min())
    return {"ok": True, "pool_dd": pool_dd, "per_stock": per_stock,
            "window": window, "days": len(idx), "missing": missing,
            "triggered": bool(pool_dd < -DD_TRIGGER),
            "warning": {"type": "月度纪律触发", "level": "crit", "code": "池",
                        "msg": f"自选池近{window}日最大回撤 {abs(pool_dd):.1%} > {DD_TRIGGER:.0%}，"
                               f"elder 6% 规则触发，停手"}
            if pool_dd < -DD_TRIGGER else None}


# ────────────────────────────────────────────────────────────
# 检测 4：盈亏比（Appel PLR）
# ────────────────────────────────────────────────────────────
def _target_price(df: pd.DataFrame, close: float) -> float | None:
    """目标位 = max(60日密集区上沿, 60日前高)；不可用返回 None。"""
    prior = df["close"].astype(float).iloc[:-1]
    prior_high = float(prior.tail(PLR_DAYS).max()) if len(prior) else np.nan
    vp_upper = np.nan
    try:
        vp = calc_volume_profile(df.tail(PLR_DAYS).reset_index(drop=True))
        if vp.get("ok") and vp.get("upper") is not None:
            vp_upper = float(vp["upper"])
    except Exception:
        vp_upper = np.nan
    cands = [x for x in (prior_high, vp_upper) if x is not None and np.isfinite(x)]
    return max(cands) if cands else None


def _check_plr(code: str, df: pd.DataFrame, close: float, stop: float | None) -> dict:
    """盈亏比 = (目标位-当前)/(当前-止损)；< 1.5 → '盈亏比不足'（Appel PLR）。"""
    if stop is None or not np.isfinite(stop):
        return {"ok": False, "note": "止损不可用, 盈亏比无法计算",
                "plr": None, "target": None, "warning": None}
    risk = close - stop
    target = _target_price(df, close)
    if target is None:
        return {"ok": False, "note": "目标位不可用(前高/密集区不足)",
                "plr": None, "target": None, "warning": None}
    reward = target - close
    if risk <= 0:
        plr = 0.0
    else:
        plr = reward / risk
    warning = None
    if plr < PLR_MIN:
        warning = {"type": "盈亏比不足", "level": "warn", "code": code,
                   "msg": f"目标位 {target:.2f} 盈亏比 {plr:.2f} < {PLR_MIN:.1f}，"
                          f"赔率不达标（Appel PLR），拒绝追入/降仓"}
    return {"ok": True, "plr": float(plr), "target": target, "reward": reward,
            "risk": risk, "warning": warning}


# ────────────────────────────────────────────────────────────
# 检测 5：沉没成本（Thaler）
# ────────────────────────────────────────────────────────────
def _check_sunk_cost(code: str, entry: float | None, close: float, df: pd.DataFrame) -> dict:
    """持仓亏损 >15% 且无反转信号(close>MA5 且 close>MA10) → '沉没成本陷阱'。"""
    if entry is None or not np.isfinite(entry):
        return {"ok": False, "note": "无持仓建仓价, 沉没成本检查跳过",
                "loss_pct": None, "reversal": None, "warning": None}
    loss_pct = (close - entry) / entry
    closes = df["close"].astype(float)
    ma5 = float(closes.rolling(5).mean().iloc[-1]) if len(closes) >= 5 else np.nan
    ma10 = float(closes.rolling(10).mean().iloc[-1]) if len(closes) >= 10 else np.nan
    reversal = bool(np.isfinite(ma5) and np.isfinite(ma10) and close > ma5 and close > ma10)
    warning = None
    if loss_pct < -SUNK_LOSS_PCT and not reversal:
        warning = {"type": "沉没成本陷阱", "level": "crit", "code": code,
                   "msg": f"持仓亏损 {loss_pct:.1%} > {SUNK_LOSS_PCT:.0%} 且无反转信号"
                          f"(close{'>' if close > ma5 else '≤'}MA5/{'>' if close > ma10 else '≤'}MA10)，"
                          f"沉没成本不参与决策（Thaler），建议纪律性离场"}
    return {"ok": True, "entry": entry, "loss_pct": loss_pct, "ma5": ma5, "ma10": ma10,
            "reversal": reversal, "warning": warning}


# ────────────────────────────────────────────────────────────
# RAG 依据
# ────────────────────────────────────────────────────────────
def _rag_basis() -> list[dict]:
    """knowledge_rag 检索方法论依据；失败/索引缺失降级标注。"""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")      # 离线优先：模型已缓存则直接加载，避免联网重试
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        rows = knowledge_rag.search(RAG_QUERY, k=RAG_K)
        return [{"file": r.get("file"), "cat": r.get("cat"), "score": r.get("score"),
                 "text": (r.get("text") or "")[:200]} for r in rows]
    except Exception as e:
        return [{"file": None, "cat": "degraded", "score": None,
                 "text": f"RAG 不可用: {str(e)[:100]}"}]


# ────────────────────────────────────────────────────────────
# 综合聚合
# ────────────────────────────────────────────────────────────
def _aggregate(warnings: list[dict], temperature: dict, n_stocks_ok: int) -> dict:
    crit = [w for w in warnings if w.get("level") == "crit"]
    warn = [w for w in warnings if w.get("level") == "warn"]
    score = CRIT_WEIGHT * len(crit) + WARN_WEIGHT * len(warn)
    has_crit = len(crit) > 0
    if has_crit or score >= HIGH_SCORE:
        level = "高"
    elif warn:
        level = "中"
    else:
        level = "低"

    stop_broken = any(w["type"] == "违规持仓" for w in crit)
    dd_trigger = any(w["type"] == "月度纪律触发" for w in crit)
    sunk = any(w["type"] == "沉没成本陷阱" for w in crit)
    plr_low = any(w["type"] == "盈亏比不足" for w in warn + crit)
    temp = temperature.get("temperature", 50)

    if stop_broken or dd_trigger or sunk:
        suggestion = "离场"
    elif plr_low:
        suggestion = "减仓"
    elif temp < TEMP_HALF_THRESHOLD:
        suggestion = "观望"
    else:
        suggestion = "持有"

    conf = 0.45 + 0.06 * len(warn) + 0.10 * len(crit)
    if n_stocks_ok == 0:
        conf = 0.30
    conf = round(min(conf, 0.95), 2)
    return {"risk_level": level, "suggestion": suggestion, "score": score,
            "crit_count": len(crit), "warn_count": len(warn),
            "confidence": conf, "signal": f"{level}-{suggestion}"}


# ────────────────────────────────────────────────────────────
# RiskSystem 统一接口
# ────────────────────────────────────────────────────────────
class RiskSystem:
    """体系6 风险纪律系统：detect(date)/report(date)/view(date)。"""

    def __init__(self, capital: float = DEFAULT_CAPITAL, watch: list[str] | None = None,
                 positions: dict[str, float] | None = None,
                 out_dir: Path | str | None = None):
        self.capital = float(capital)
        self.positions = dict(positions or {})
        self.watch = [c.strip().zfill(6) for c in (watch or [])]
        self.out_dir = Path(out_dir) if out_dir else DEFAULT_OUT_DIR
        self._rag_cache: list[dict] | None = None  # 模型加载昂贵，同实例只检索一次

    # ── 检测 ──────────────────────────────────────────────
    def detect(self, date: str | None = None) -> dict:
        codes = self.watch or load_watch_codes()
        if not codes:
            return {"date": str(date or ""), "status": "degraded",
                    "note": "自选池为空", "risk_level": "低", "suggestion": "观望",
                    "signal": "低-观望", "confidence": 0.3, "evidence": ["无自选股, 无风险输入"]}
        target = pd.Timestamp(date).normalize() if date else _latest_common_day(codes)
        date_str = str(target.date())
        names, _ = load_names()

        temperature = market_temperature(date_str or date)
        temp = temperature["temperature"]

        stocks: dict[str, dict] = {}
        warnings: list[dict] = []
        missing: list[dict] = []
        advices: list[dict] = []
        n_ok = 0
        for code in codes:
            df, err = _load_kline(code, target)
            if df is None:
                missing.append({"code": code, "name": names.get(code, code), "error": err})
                continue
            close = float(df["close"].astype(float).iloc[-1])
            atr = float(atr14(df)) if len(df) >= 15 else np.nan
            stop_res = _check_stop(code, df, close, atr)
            entry = self.positions.get(code)
            pos_adv = _position_advice(code, entry if entry is not None else close,
                                       close, stop_res.get("stop"), self.capital, temp)
            plr_res = _check_plr(code, df, close, stop_res.get("stop"))
            sunk_res = _check_sunk_cost(code, entry, close, df)
            for r in (stop_res, plr_res, sunk_res):
                w = r.get("warning")
                if w:
                    warnings.append(w)
            advices.append(pos_adv)
            n_ok += 1
            stocks[code] = {
                "code": code, "name": names.get(code, code), "date": date_str,
                "close": close, "atr": atr,
                "stop": stop_res, "position": pos_adv, "plr": plr_res,
                "sunk_cost": sunk_res,
            }

        dd = _pool_drawdown(codes, target)
        if dd.get("warning"):
            warnings.append(dd["warning"])
        if dd.get("missing"):
            missing.extend({"code": m.split("(")[0], "name": m.split("(")[0],
                            "error": m.split("(", 1)[1].rstrip(")")} for m in dd["missing"])

        agg = _aggregate(warnings, temperature, n_ok)
        if self._rag_cache is None:
            self._rag_cache = _rag_basis()
        rag = self._rag_cache
        data_status: dict[str, str] = {}
        missing_codes = sorted({m["code"] for m in missing})
        if missing:
            data_status["个股K线"] = f"{len(missing_codes)}/{len(codes)} 缺失"
        return {
            "date": date_str,
            "status": "degraded" if (missing_codes or not n_ok) else "ok",
            "missing_count": len(missing_codes),
            "data_status": data_status,
            "account": {"capital": self.capital, "risk_per_trade": RISK_PER_TRADE,
                        "max_position_pct": MAX_POSITION_PCT,
                        "temp_half_threshold": TEMP_HALF_THRESHOLD},
            "temperature": temperature,
            "total_position": _total_position_advice(advices, self.capital, temp),
            "drawdown": dd,
            "stocks": stocks,
            "warnings": warnings,
            "missing": missing,
            "rag": rag,
            "risk_level": agg["risk_level"],
            "suggestion": agg["suggestion"],
            "confidence": agg["confidence"],
            "signal": agg["signal"],
            "score": agg["score"],
        }

    # ── 报告 ──────────────────────────────────────────────
    def report(self, date: str | None = None) -> Path:
        """生成 generated/risk_report_{date}.md，返回文件路径。"""
        res = self.detect(date)
        md = render_report(res)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        out = self.out_dir / f"risk_report_{res['date']}.md"
        out.write_text(md, encoding="utf-8")
        print(f"[risk] 报告已保存: {out}")
        return out

    # ── 观点（multi_agent 同构）───────────────────────────
    def view(self, date: str | None = None) -> dict:
        """风险纪律专家观点：{agent, signal, confidence, evidence}。"""
        res = self.detect(date)
        ev = [f"{res['signal']}（风险等级 {res['risk_level']}，操作建议 {res['suggestion']}）",
              f"温度 {res['temperature']['temperature']}（{res['temperature']['source']}）",
              f"自选池近{DD_WINDOW}日回撤 "
              f"{abs(res['drawdown']['pool_dd']):.1%}" if res["drawdown"].get("pool_dd") is not None
              else "自选池回撤不可用"]
        ev += [w["msg"] for w in res["warnings"][:5]]
        if res["rag"]:
            src = next((r for r in res["rag"] if r.get("file")), None)
            if src:
                ev.append(f"方法论依据: {src['file']}")
        if res["missing"]:
            ev.append(f"数据缺失 {len(res['missing'])} 项: "
                      + ", ".join(f"{m['code']}({m['error']})" for m in res["missing"][:3]))
        risk_view = "防守" if res["risk_level"] in ("高", "中") else (
            "防守" if res["suggestion"] in ("离场", "减仓") else "震荡")
        return {"agent": "风险纪律", "signal": res["signal"], "view": risk_view,
                "confidence": res["confidence"], "evidence": ev,
                "weight": 1.0, "status": res["status"], "detail": res}


def load_watch_codes() -> list[str]:
    """自选池：优先 --watch 显式传入，其次 config/watchlist.json。"""
    if CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            return [str(c).zfill(6) for c in cfg.get("watch", []) if str(c).strip()]
        except Exception as e:
            logging.getLogger(__name__).error(f"[risk_system] 操作失败: {e}", exc_info=True)
    return []


def load_positions_from_file() -> dict[str, float]:
    """config/positions.json（可选）：{code: 建仓价}。"""
    if POSITIONS_PATH.exists():
        try:
            raw = json.loads(POSITIONS_PATH.read_text(encoding="utf-8"))
            return {str(k).zfill(6): float(v) for k, v in raw.items()}
        except Exception as e:
            logging.getLogger(__name__).error(f"[risk_system] 操作失败: {e}", exc_info=True)
    return {}


# ────────────────────────────────────────────────────────────
# 渲染
# ────────────────────────────────────────────────────────────
def render_report(res: dict) -> str:
    lines = [
        f"# 🛡️ 风险纪律报告 {res['date']}",
        "",
        f"## 综合结论",
        f"- 风险等级: **{res['risk_level']}** | 操作建议: **{res['suggestion']}** "
        f"| 置信度 {res['confidence']}",
        f"- 市场温度: {res['temperature']['temperature']}/100（{res['temperature']['source']}）",
        f"- 账户: 初始资金 {res['account']['capital']:,.0f} 元 | 单笔风险 "
        f"{res['account']['risk_per_trade']:.0%} | 单票上限 "
        f"{res['account']['max_position_pct']:.0%} | 温度<{res['account']['temp_half_threshold']} 半仓",
        "",
        "## 仓位建议（2% 规则）",
        f"- 建议总仓位: {res['total_position']['total_pct']:.1%} "
        f"（{res['total_position']['total_notional']:,.0f} 元）| {res['total_position']['note']}",
        "",
        "## 回撤检查（elder 6% 规则）",
    ]
    dd = res["drawdown"]
    if dd.get("pool_dd") is not None:
        lines.append(f"- 自选池近{dd['window']}日等权最大回撤: **{abs(dd['pool_dd']):.1%}**"
                     f"{' ⛔ 月度纪律触发,停手' if dd['triggered'] else ''}")
        for c, d in dd.get("per_stock", {}).items():
            lines.append(f"  - {c}: {abs(d):.1%}")
    else:
        lines.append(f"- {dd.get('note', '回撤不可用')}")
    if dd.get("missing"):
        lines.append(f"- 数据缺失: {', '.join(dd['missing'])}")

    lines += ["", "## 个股纪律明细", ""]
    for code, s in res["stocks"].items():
        st = s["stop"]
        pa = s["position"]
        pl = s["plr"]
        sk = s["sunk_cost"]
        stop_txt = f"{fmt(st.get('stop'))}（距 {fmt(st.get('dist_pct'), 1)}%）" if st.get("ok") else st.get("note", "—")
        pos_txt = (f"{pa['pct']:.1%}（{pa['notional']:,.0f}元 / {pa['shares']}股）"
                   + (" ⚠️触及20%上限" if pa.get("capped") else "")) if pa.get("ok") else pa.get("note", "—")
        plr_txt = f"{fmt(pl.get('plr'))}（目标 {fmt(pl.get('target'))}）" if pl.get("ok") else pl.get("note", "—")
        sk_txt = (f"亏损 {sk['loss_pct']:.1%} 反转信号 {'有' if sk['reversal'] else '无'}"
                  if sk.get("ok") else sk.get("note", "—"))
        lines += [
            f"### {code} {s['name']}（收盘 {fmt(s['close'])} | ATR14 {fmt(s['atr'])}）",
            f"- 止损位(close-1.5×ATR): {stop_txt}",
            f"- 仓位建议: {pos_txt}",
            f"- 盈亏比: {plr_txt}",
            f"- 沉没成本: {sk_txt}",
        ]

    lines += ["", "## ⚠️ 风险警告"]
    if res["warnings"]:
        for w in res["warnings"]:
            icon = "⛔" if w["level"] == "crit" else "⚠️"
            lines.append(f"- {icon} [{w['type']}] {w['code']}: {w['msg']}")
    else:
        lines.append("- 无触发")

    lines += ["", "## 📚 RAG 方法论依据"]
    for r in res["rag"]:
        lines.append(f"- ({r.get('cat', '?')}) {r.get('file') or '—'} [score {r.get('score')}]: "
                     f"{r.get('text', '').strip()[:120]}")
    if res["missing"]:
        lines += ["", "## ⚠️ 数据缺失"]
        for m in res["missing"]:
            lines.append(f"- {m['code']} {m['name']}: {m['error']}")
    lines += ["", "---", "*体系6 风险纪律 | Elder 2%/6%规则 + 生存纪律 + Thaler 沉没成本 + Appel PLR | "
                  "方法: knowledge_rag 检索 + 本地K线计算，仅供纪律参考*"]
    return "\n".join(lines) + "\n"


def _render_text(res: dict) -> str:
    lines = [
        f"[风险纪律] {res['date']} | 风险等级 {res['risk_level']} | 操作建议 {res['suggestion']} "
        f"| 置信 {res['confidence']}",
        f"  市场温度 {res['temperature']['temperature']}/100（{res['temperature']['source']}）",
    ]
    dd = res["drawdown"]
    if dd.get("pool_dd") is not None:
        lines.append(f"  自选池近{dd['window']}日回撤 {abs(dd['pool_dd']):.1%}"
                     + (" ⛔月度纪律触发" if dd["triggered"] else ""))
    lines.append(f"  建议总仓位 {res['total_position']['total_pct']:.1%}"
                 f"（{res['total_position']['note']}）")
    for code, s in res["stocks"].items():
        st, pa, pl = s["stop"], s["position"], s["plr"]
        stop_txt = f"{fmt(st.get('stop'))}(距{fmt(st.get('dist_pct'), 1)}%)" if st.get("ok") else st.get("note")
        pos_txt = f"{pa['pct']:.1%}/{pa['shares']}股" if pa.get("ok") else pa.get("note")
        plr_txt = f"{fmt(pl.get('plr'))}" if pl.get("ok") else pl.get("note")
        flag = "⛔" if st.get("broken") else ("⚠️" if st.get("intraday_break") else " ")
        lines.append(f"  {flag} {code} {s['name']}: 止损 {stop_txt} | 仓位 {pos_txt} | 盈亏比 {plr_txt}")
    for w in res["warnings"]:
        lines.append(f"  ⚠️ [{w['type']}] {w['code']}: {w['msg']}")
    for m in res["missing"]:
        lines.append(f"  ⚠️ 数据缺失: {m['code']} {m['name']}: {m['error']}")
    rag = next((r for r in res["rag"] if r.get("file")), None)
    if rag:
        lines.append(f"  RAG: {rag['file']}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="体系6 风险纪律系统（止损/仓位/回撤/盈亏比/沉没成本）")
    ap.add_argument("--watch", default=None, help="自选股代码，逗号分隔（缺省读 config/watchlist.json）")
    ap.add_argument("--date", default=None, help="分析日期 YYYY-MM-DD（默认最新共同交易日）")
    ap.add_argument("--capital", type=float, default=DEFAULT_CAPITAL, help="初始资金（默认 100万）")
    ap.add_argument("--positions", default=None,
                    help="持仓 JSON，如 '{\"601899\": 21.0}' 或 @文件路径（缺省读 config/positions.json）")
    ap.add_argument("--no-report", action="store_true", help="不写 generated/risk_report_{date}.md")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="报告输出目录（默认仓库 generated/）")
    args = ap.parse_args()

    codes = ([c.strip().zfill(6) for c in args.watch.split(",") if c.strip()]
             if args.watch else load_watch_codes())
    bad = [c for c in codes if not CODE_RE.match(c)]
    if bad:
        print(f"[错误] 非法代码: {','.join(bad)}（需6位数字）")
        sys.exit(1)
    if not codes:
        print("[错误] --watch 不能为空")
        sys.exit(1)

    positions = load_positions_from_file()
    if args.positions:
        raw = args.positions.strip()
        if raw.startswith("@"):
            raw = Path(raw[1:]).read_text(encoding="utf-8")
        positions.update({str(k).zfill(6): float(v) for k, v in json.loads(raw).items()})

    rs = RiskSystem(capital=args.capital, watch=codes, positions=positions,
                    out_dir=args.out_dir)
    res = rs.detect(args.date)
    print(_render_text(res))
    if not args.no_report:
        rs.report(args.date)


if __name__ == "__main__":
    main()
