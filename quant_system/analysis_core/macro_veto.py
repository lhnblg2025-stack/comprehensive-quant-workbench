"""
macro_veto — 宏观 + 交易级否决层（V11 多空仲裁）

短线情绪可能过热，宏观估值/流动性可能已透支 → 裁判员模块。
当宏观综合评分低于阈值时，强制压制短线激进仓位。

本地数据源（全部已有）:
  1. 沪深300/中证全指 PE 历史分位（index_pe.parquet）
  2. 巴菲特指标 = 总市值/GDP（buffett_index.parquet）
  3. 市场机制（regime_classifier 输出，高波 = 风险加成）
  4. zt_daily_stats.parquet：涨停/炸板/跌停/连板/亏钱效应交易级情绪统计
  5. index_daily_沪深300.parquet：指数日线，供交易级规则计算涨跌幅与三连跌

宏观裁决规则:
  hard veto (仓位系数 0.3, 禁打板):
    - macro_risk >= 3
    - trade_risk >= 2（单条 weight2 交易规则命中即 hard）
    - macro_risk + trade_risk >= 3
  soft veto (仓位系数 0.6):
    - 除 hard 外，macro_risk >= 1 或 trade_risk >= 1
  none: 不干预（系数 1.0）

交易级/情绪级规则（默认启用 TRADING_RULES_ENABLED，可通过 veto(trading_rules=False) 按粒度关闭）:
  R1 blast_retreat   : 炸板率>40% 且 涨停<30家            → +2
  R2 idx3d_down_weak : 沪深300三连跌 且 炸板率>30%         → +2
  R3 profit_drought  : 涨停<20家 且 沪深300近5日跌超3%     → +1
  R4 limit_down_surge: 跌停数>涨停数 且 涨停<30家          → +2
  R5 jr1_loss_2d     : 当日及前一日 jr1 均<0               → +1
  R6 big_loss_wave   : 大面>50家 且 炸板率>30%             → +1
  R7 high_board_break: 前一日最高连板>=6 且 当日<4         → +1
  R8 panic_day       : 沪深300 当日涨跌幅<-4%              → +2

输出: generated/macro_veto_{date}.json，供 battle_map 乘入仓位建议

用法:
  python3 -m quant_system.analysis_core.macro_veto [--date 2026-08-07]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.config import MARKET_DIR, ZT_DAILY_STATS  # noqa: E402
from quant_system.analysis_core.regime_classifier import get_regime  # noqa: E402
from quant_system.policy import get_policy  # noqa: E402

CST = timezone(timedelta(hours=8))
INDEX_PE = MARKET_DIR / "index_pe.parquet"
BUFFETT = ROOT / "data_warehouse" / "oneoff" / "buffett_index.parquet"
INDEX_HS300 = MARKET_DIR / "index_daily_沪深300.parquet"

HARD_COEF, SOFT_COEF = 0.3, 0.6
PE_HARD, PE_SOFT = 0.95, 0.80
BUFFETT_HARD, BUFFETT_SOFT = 1.10, 0.95
TRADING_RULES_ENABLED = True


def _num(row: dict | pd.Series | None, key: str) -> float | None:
    """从 dict/Series 安全读取数值；缺失或 NaN 返回 None。"""
    if row is None:
        return None
    val = row.get(key) if hasattr(row, "get") else None
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _load_zt_row(date: str) -> pd.Series | None:
    """读取 zt_daily_stats 中 date<=目标日的最后一行，并补充前一日 jr1/max_board。"""
    if not ZT_DAILY_STATS.exists():
        return None
    try:
        df = pd.read_parquet(ZT_DAILY_STATS)
    except Exception:
        return None
    if df.empty or "date" not in df.columns:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= pd.Timestamp(date)].sort_values("date")
    if df.empty:
        return None
    row = df.iloc[-1].copy()
    if len(df) > 1:
        prev = df.iloc[-2]
        row["jr1_prev"] = prev["jr1"] if "jr1" in prev.index else None
        row["max_board_prev"] = prev["max_board"] if "max_board" in prev.index else None
    else:
        row["jr1_prev"] = None
        row["max_board_prev"] = None
    return row


def _load_hs300_ctx(date: str) -> dict | None:
    """读取沪深300截至目标日的数据，构造交易级规则所需上下文。"""
    if not INDEX_HS300.exists():
        return None
    try:
        df = pd.read_parquet(INDEX_HS300)
    except Exception:
        return None
    if df.empty or "date" not in df.columns or "close" not in df.columns:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= pd.Timestamp(date)].sort_values("date").reset_index(drop=True)
    if df.empty:
        return None
    close = df["close"].astype(float)
    ctx: dict = {"ret_1d": None, "ret_5d": None, "down3": None}
    if len(df) >= 2:
        ctx["ret_1d"] = float(close.iloc[-1] / close.iloc[-2] - 1)
    if len(df) >= 6:
        ctx["ret_5d"] = float(close.iloc[-1] / close.iloc[-6] - 1)
    if len(df) >= 3:
        ctx["down3"] = bool(close.iloc[-1] < close.iloc[-2] < close.iloc[-3])
    return ctx


def _hit(rule_id: str, name: str, reason: str, evidence: str, weight: int) -> dict:
    return {"rule_id": rule_id, "name": name, "reason": reason,
            "evidence": evidence, "weight": weight}


def _check_trade_rules(zt_row: dict | pd.Series | None,
                       hs_ctx: dict | None = None) -> list[dict]:
    """逐条判定 8 条交易级/情绪级规则；任一数据缺失则该条跳过。"""
    zt = zt_row if zt_row is not None else {}
    hs = hs_ctx if hs_ctx is not None else {}
    hits: list[dict] = []

    r1_zb_rate = get_policy("veto.r1_zb_rate", 0.40)
    r1_zt_cnt = get_policy("veto.r1_zt_cnt", 30)
    r2_zb_rate = get_policy("veto.r2_zb_rate", 0.30)
    r3_zt_cnt = get_policy("veto.r3_zt_cnt", 20)
    r4_zt_cnt = get_policy("veto.r4_zt_cnt", 30)
    r5_weight = get_policy("veto.r5_weight", 1)
    r6_big_loss = get_policy("veto.r6_big_loss", 50)
    r6_zb_rate = get_policy("veto.r6_zb_rate", 0.30)
    r8_panic = get_policy("veto.r8_panic", -0.04)

    zt_cnt = _num(zt, "zt_cnt")
    zb_rate = _num(zt, "zb_rate")
    dt_cnt = _num(zt, "dt_cnt")
    big_loss_cnt = _num(zt, "big_loss_cnt")
    jr1 = _num(zt, "jr1")
    jr1_prev = _num(zt, "jr1_prev")
    max_board = _num(zt, "max_board")
    max_board_prev = _num(zt, "max_board_prev")
    ret_1d = _num(hs, "ret_1d")
    ret_5d = _num(hs, "ret_5d")
    down3 = hs.get("down3")

    if zb_rate is not None and zt_cnt is not None and zb_rate > r1_zb_rate and zt_cnt < r1_zt_cnt:
        hits.append(_hit("R1", "blast_retreat", "炸板率>40% 且涨停<30家，情绪退潮",
                         "error_book特征: 炸板率高+涨停少=情绪退潮", 2))
    if down3 is True and zb_rate is not None and zb_rate > r2_zb_rate:
        hits.append(_hit("R2", "idx3d_down_weak", "沪深300三连跌且炸板率>30%，系统风险窗口",
                         "指数连跌+炸板率高=系统风险窗口", 2))
    if zt_cnt is not None and ret_5d is not None and zt_cnt < r3_zt_cnt and ret_5d < -0.03:
        hits.append(_hit("R3", "profit_drought", "涨停<20家且沪深300近5日跌超3%，赚钱效应冰点",
                         "涨停枯竭+指数走弱=赚钱效应冰点", 1))
    if dt_cnt is not None and zt_cnt is not None and dt_cnt > zt_cnt and zt_cnt < r4_zt_cnt:
        hits.append(_hit("R4", "limit_down_surge", "跌停数>涨停数且涨停<30家，极端弱势",
                         "跌停数>涨停数且涨停<30家=极端弱势", 2))
    if jr1 is not None and jr1_prev is not None and jr1 < 0 and jr1_prev < 0:
        hits.append(_hit("R5", "jr1_loss_2d", "当日及前一日打板均为亏钱效应",
                         "error_book特征: 打板连续两日亏钱效应", r5_weight))
    if big_loss_cnt is not None and zb_rate is not None and big_loss_cnt > r6_big_loss and zb_rate > r6_zb_rate:
        hits.append(_hit("R6", "big_loss_wave", "大面>50家且炸板率>30%，退潮期",
                         "大面潮+高炸板=退潮期", 1))
    if max_board_prev is not None and max_board is not None and max_board_prev >= 6 and max_board < 4:
        hits.append(_hit("R7", "high_board_break", "前一日最高连板>=6且当日<4，情绪退潮拐点",
                         "高位连板断板=情绪退潮拐点", 1))
    if ret_1d is not None and ret_1d < r8_panic:
        hits.append(_hit("R8", "panic_day", "沪深300单日跌超4%，恐慌抛售日",
                         "恐慌抛售日(与温度计恐慌档呼应)", 2))
    return hits


def _pe_percentile(date: str) -> float | None:
    """沪深300 滚动市盈率近5年分位（0-1）。"""
    if not INDEX_PE.exists():
        return None
    df = pd.read_parquet(INDEX_PE)
    df["日期"] = pd.to_datetime(df["日期"])
    hs300 = df[df["symbol"] == "沪深300"].copy()
    if hs300.empty:
        return None
    hs300 = hs300[hs300["日期"] <= pd.Timestamp(date)].sort_values("日期")
    if hs300.empty:
        return None
    col = "滚动市盈率"
    val = hs300[col].iloc[-1]
    hist = hs300[hs300["日期"] >= pd.Timestamp(date) - pd.Timedelta(days=365 * 5)][col].dropna()
    if len(hist) < 60 or pd.isna(val):
        return None
    return float((hist <= val).mean())


def _buffett_ratio(date: str) -> float | None:
    """总市值/GDP。"""
    if not BUFFETT.exists():
        return None
    df = pd.read_parquet(BUFFETT)
    df["日期"] = pd.to_datetime(df["日期"])
    df = df[df["日期"] <= pd.Timestamp(date)].sort_values("日期")
    if df.empty or df["GDP"].iloc[-1] <= 0:
        return None
    return float(df["总市值"].iloc[-1] / df["GDP"].iloc[-1])


def veto(date: str | None = None, trading_rules: bool = TRADING_RULES_ENABLED) -> dict:
    date = date or datetime.now(CST).date().isoformat()
    regime = get_regime(date)
    pe_pct = _pe_percentile(date)
    buffett = _buffett_ratio(date)
    high_vol = "高波" in regime.get("regime", "")

    reasons: list[str] = []
    trade_hits: list[dict] = []
    level = "none"
    coef = 1.0

    # 宏观分与交易分分开累计；单条 weight2 交易规则即可触发 hard
    pe_hard = get_policy("veto.pe_hard", PE_HARD)
    pe_soft = get_policy("veto.pe_soft", PE_SOFT)
    buffett_hard = get_policy("veto.buffett_hard", BUFFETT_HARD)
    buffett_soft = get_policy("veto.buffett_soft", BUFFETT_SOFT)
    macro_risk = 0
    if pe_pct is not None:
        if pe_pct > pe_hard:
            macro_risk += 2
            reasons.append(f"沪深300 PE分位 {pe_pct:.0%} > 95%")
        elif pe_pct > pe_soft:
            macro_risk += 1
            reasons.append(f"沪深300 PE分位 {pe_pct:.0%} > 80%")
    if buffett is not None:
        if buffett > buffett_hard:
            macro_risk += 2
            reasons.append(f"巴菲特指标 {buffett:.0%} > 110%")
        elif buffett > buffett_soft:
            macro_risk += 1
            reasons.append(f"巴菲特指标 {buffett:.0%} > 95%")
    if high_vol:
        macro_risk += 1
        reasons.append(f"高波市({regime.get('regime','')})")

    trade_risk = 0
    if trading_rules:
        zt_row = _load_zt_row(date)
        hs_ctx = _load_hs300_ctx(date)
        for hit in _check_trade_rules(zt_row, hs_ctx):
            trade_risk += int(hit.get("weight", 0))
            reasons.append(f"{hit['rule_id']} {hit['name']}: {hit['reason']}")
            trade_hits.append({
                "rule_id": hit["rule_id"],
                "name": hit["name"],
                "reason": hit["reason"],
                "evidence": hit["evidence"],
                "weight": hit["weight"],
            })

    hard_coef = get_policy("veto.hard_coef", HARD_COEF)
    soft_coef = get_policy("veto.soft_coef", SOFT_COEF)
    if macro_risk >= 3 or trade_risk >= 2 or (macro_risk + trade_risk) >= 3:
        level, coef = "hard", hard_coef
    elif macro_risk >= 1 or trade_risk >= 1:
        level, coef = "soft", soft_coef
    if level == "hard":
        reasons.append(f"→ 🔴 硬否决: 短线仓位系数 {hard_coef}, 禁打板")
    elif level == "soft":
        reasons.append(f"→ 🟠 软否决: 短线仓位系数 {soft_coef}")

    res = {
        "date": date,
        "level": level,
        "position_coef": coef,
        "pe_percentile": round(pe_pct, 3) if pe_pct is not None else None,
        "buffett_ratio": round(buffett, 3) if buffett is not None else None,
        "regime": regime.get("regime", ""),
        "reasons": reasons,
        "trade_hits": trade_hits,
    }
    out = ROOT / "generated" / f"macro_veto_{date}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


def get_veto(date: str | None = None) -> dict:
    """供 battle_map 调用的轻量只读接口。"""
    date = date or datetime.now(CST).date().isoformat()
    p = ROOT / "generated" / f"macro_veto_{date}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    # 审计 2026-08-16：缺失不得当作“无否决”；返回 unavailable 供上层降级可见
    return {"date": date, "level": "unavailable", "position_coef": None,
            "reasons": ["macro_veto 文件缺失，无法判断否决"], "unavailable": True}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="宏观否决层")
    ap.add_argument("--date", default=None)
    args = ap.parse_args()
    r = veto(args.date)
    print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
