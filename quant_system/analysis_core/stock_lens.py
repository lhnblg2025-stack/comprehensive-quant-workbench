#!/usr/bin/env python3
"""
stock_lens.py — 个股全景透视（短线基因 + 中线筹码 + 决策合成）
================================================================
用户硬诉求: 数据基座要用好用活 —— 短线看龙虎榜/概念股/资金流向,
中线看融资盘/股东户数/基金持仓/估值, 并给出可解释的决策依据。

设计依据（skills 知识库方法论）:
  短线层（游资心法, skills/trading-mastery 26位）:
    - 龙虎榜席位: "看龙虎榜游资是否进场" —— 游资营业部 vs 机构专用,
      用 lhb「上榜后1/2/5/10日」字段统计历史胜率/期望, 作为跟随的后验依据
    - 涨停/连板基因: 首板→连板→分歧→低吸/打板 (zt_pool_history 连板数)
    - 概念共振: 个股所属概念当日涨停家数 = 板块热度 (concept_member + zt_pool)
    - 情绪周期: 位置决定仓位 (emotion_cycle)
  中线层（IC 书籍 + 价值投资）:
    - 融资盘: 个股融资余额趋势 = 杠杆资金共识 (margin_detail)
    - 股东户数: 户数降+户均市值升 = 筹码集中/主力吸筹 (gdhs)
    - 基金持仓: 家数/增减 = 机构认可 (fund_portfolio_hold)
    - 估值: PE/PB 分位 + ROE (valuation/financial)

数据基座（data_warehouse, 双机一致）:
  market/lhb_20*.parquet          龙虎榜日汇总(净买额/上榜后N日/机构解读)
  market/lhb_hyyyb_em.parquet     游资营业部(买入股票名称串)
  market/zt_pool_history.parquet  涨停池历史(连板数/次日表现)
  classification/concept_member.parquet  概念→成分股
  market/stock_market_fund_flow.parquet  全市场主力资金(大盘环境)
  market/margin_detail_{sh,sz}/*.parquet 个股融资余额(按日分文件)
  market/gdhs_all.parquet         股东户数增减
  market/fund_portfolio_hold.parquet    基金持仓
  valuation/*.parquet / financial/*.parquet  估值与财务

输出: dict(短线分/中线分/四象限结论/每项数据源与缺失标注) + JSON 落盘
用法:
  python3 -m quant_system.analysis_core.stock_lens --code 600519
  python3 -m quant_system.analysis_core.stock_lens --watch      # 自选+持仓
  python3 -m quant_system.analysis_core.stock_lens --code 600519 --date 2026-08-14
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

CST = timezone(timedelta(hours=8))
MARKET_DIR = ROOT / "data_warehouse" / "market"
CLASS_DIR = ROOT / "data_warehouse" / "classification"
VALUATION_DIR = ROOT / "data_warehouse" / "valuation"
FINANCIAL_DIR = ROOT / "data_warehouse" / "financial"
GENERATED = ROOT / "generated"

_log = logging.getLogger(__name__)

# ── 评分权重（集中声明, 可解释）──
W_ZT_GENE = 25      # 涨停基因
W_LHB = 25          # 龙虎榜
W_CONCEPT = 20      # 概念共振
W_FLOW = 20         # 个股资金流
W_MARGIN = 30       # 融资盘
W_GDHS = 25         # 股东户数(筹码)
W_FUND_HOLD = 20    # 基金持仓
W_VALUE = 25        # 估值
STRONG = 60.0
WEAK = 40.0
LOOKBACK_LHB = 60    # 龙虎榜统计回看天数
LOOKBACK_ZT = 60     # 涨停基因回看天数


def _today() -> str:
    return datetime.now(CST).strftime("%Y-%m-%d")


def _safe_float(v) -> float:
    try:
        if v is None:
            return float("nan")
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _norm_code(code: str) -> str:
    return str(code).zfill(6)


def _disclosure_safe_end(end: pd.Timestamp) -> pd.Timestamp:
    """披露时间保护(V12.3 审计 P1-4): 若 end 为"今天"且当前未到 15:00 收盘,
    涨停池/龙虎榜当日数据尚未公开, 上界收窄到前一日, 防盘中前视。
    历史回测(end 为过去日期)不受影响。"""
    now = datetime.now(CST)
    if end.date() == now.date() and now.hour * 60 + now.minute < 15 * 60:
        return end - pd.Timedelta(days=1)
    return end


def _load_lhb_window(code: str, end: pd.Timestamp, days: int = LOOKBACK_LHB) -> pd.DataFrame:
    """回看窗口内该股全部龙虎榜记录（lhb_20*.parquet 季度段文件）。"""
    end = _disclosure_safe_end(end)
    rows = []
    start = end - pd.Timedelta(days=days * 1.7)  # 日历日宽放
    for f in sorted(MARKET_DIR.glob("lhb_20*.parquet")):
        if "hyyyb" in f.name or "jgmmtj" in f.name or "ggtj" in f.name:
            continue
        try:
            d = pd.read_parquet(f)
        except Exception:  # noqa: BLE001
            continue
        if "代码" not in d.columns or "上榜日" not in d.columns:
            continue
        d = d[d["代码"].astype(str).str.zfill(6) == code]
        if d.empty:
            continue
        d["上榜日"] = pd.to_datetime(d["上榜日"], errors="coerce")
        d = d[(d["上榜日"] >= start) & (d["上榜日"] <= end)]
        if not d.empty:
            rows.append(d)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).drop_duplicates("上榜日")


def _load_zt_history(code: str, end: pd.Timestamp, days: int = LOOKBACK_ZT) -> pd.DataFrame:
    try:
        d = pd.read_parquet(MARKET_DIR / "zt_pool_history.parquet")
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    d = d[d["code"].astype(str).str.zfill(6) == code].copy()
    if d.empty:
        return d
    end = _disclosure_safe_end(end)
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    return d[(d["date"] >= end - pd.Timedelta(days=days * 1.7)) & (d["date"] <= end)]


def _concept_of(code: str) -> list[str]:
    try:
        m = pd.read_parquet(CLASS_DIR / "concept_member.parquet")
    except Exception:  # noqa: BLE001
        return []
    sub = m[m["code"].astype(str).str.zfill(6) == code]
    return sorted(sub["concept"].astype(str).tolist())


def _concept_heat(concepts: list[str], end: pd.Timestamp) -> dict:
    """概念热度: 概念内近5日涨停家数(最新日) + 概念成员数。"""
    if not concepts:
        return {"concepts": [], "zt_counts": {}, "hot": False}
    try:
        zt = pd.read_parquet(MARKET_DIR / "zt_pool_history.parquet")
        mem = pd.read_parquet(CLASS_DIR / "concept_member.parquet")
    except Exception:  # noqa: BLE001
        return {"concepts": concepts, "zt_counts": {}, "hot": False}
    mem = mem[mem["concept"].astype(str).isin(concepts)]
    codes = set(mem["code"].astype(str).str.zfill(6))
    zt = zt[zt["code"].astype(str).str.zfill(6).isin(codes)].copy()
    if zt.empty:
        return {"concepts": concepts, "zt_counts": {}, "hot": False}
    zt["date"] = pd.to_datetime(zt["date"], errors="coerce")
    zt = zt[zt["date"] >= end - pd.Timedelta(days=7)]
    if zt.empty:
        return {"concepts": concepts, "zt_counts": {}, "hot": False}
    latest = zt["date"].max()
    zt_latest = zt[zt["date"] == latest]
    counts = zt_latest.groupby("code").size().to_dict()
    member_n = int(zt_latest["code"].nunique())
    return {"concepts": concepts, "zt_counts": {"zt_stocks": len(counts), "concept_members_zt": member_n},
            "hot": member_n >= 3, "latest_zt_date": latest.strftime("%Y-%m-%d")}


def _market_flow_env(end: pd.Timestamp) -> dict:
    """全市场主力资金环境（stock_market_fund_flow 末行）。"""
    try:
        d = pd.read_parquet(MARKET_DIR / "stock_market_fund_flow.parquet")
    except Exception:  # noqa: BLE001
        return {"ok": False}
    d["日期"] = pd.to_datetime(d["日期"], errors="coerce")
    d = d[d["日期"] <= end]
    if d.empty:
        return {"ok": False}
    last = d.iloc[-1]
    net = _safe_float(last.get("主力净流入-净额"))
    pct = _safe_float(last.get("主力净流入-净占比"))
    return {"ok": True, "date": last["日期"].strftime("%Y-%m-%d"),
            "main_net": net, "main_pct": pct,
            "direction": "流入" if net and net > 0 else "流出"}


def _margin_series(code: str) -> pd.DataFrame:
    """个股融资余额序列（margin_detail_sh/sz 按日文件, 取该股历史）。"""
    rows = []
    for market in ("margin_detail_sh", "margin_detail_sz"):
        base = MARKET_DIR / market
        if not base.exists():
            continue
        for f in sorted(base.glob("*.parquet"))[-260:]:  # 约一年
            try:
                d = pd.read_parquet(f)
            except Exception:  # noqa: BLE001
                continue
            if "标的证券代码" not in d.columns:
                continue
            sub = d[d["标的证券代码"].astype(str).str.zfill(6) == code]
            if not sub.empty:
                rows.append(sub[["信用交易日期", "融资余额", "融资买入额", "融资偿还额"]])
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df["信用交易日期"] = pd.to_datetime(df["信用交易日期"], errors="coerce")
    return df.drop_duplicates("信用交易日期").sort_values("信用交易日期")


def _gdhs_info(code: str) -> dict:
    try:
        d = pd.read_parquet(MARKET_DIR / "gdhs_all.parquet")
    except Exception:  # noqa: BLE001
        return {}
    sub = d[d["代码"].astype(str).str.zfill(6) == code]
    if sub.empty:
        return {}
    r = sub.iloc[0]
    return {"change_pct": _safe_float(r.get("股东户数-增减比例")),
            "last": _safe_float(r.get("股东户数-本次")),
            "date": str(r.get("股东户数统计截止日-本次", ""))[:10]}


def _fund_hold(code: str) -> dict:
    try:
        d = pd.read_parquet(MARKET_DIR / "fund_portfolio_hold.parquet")
    except Exception:  # noqa: BLE001
        return {}
    sub = d[d["股票代码"].astype(str).str.zfill(6) == code]
    if sub.empty:
        return {}
    r = sub.iloc[0]
    return {"fund_count": int(_safe_float(r.get("持有基金家数", 0))),
            "change": str(r.get("持股变化", "")),
            "change_pct": _safe_float(r.get("持股变动比例"))}


def _valuation(code: str) -> dict:
    """PE/PB 最新 + ROE(近4期加权净资产收益率求和, 与 after_close_extra 同口径)。"""
    out = {"pe": None, "pb": None, "roe": None}
    try:
        f = VALUATION_DIR / f"{code}.parquet"
        if f.exists():
            v = pd.read_parquet(f)
            if "peTTM" in v.columns:
                pe = v["peTTM"].dropna()
                if not pe.empty:
                    out["pe"] = round(float(pe.iloc[-1]), 1)
            if "pbMRQ" in v.columns:
                pb = v["pbMRQ"].dropna()
                if not pb.empty:
                    out["pb"] = round(float(pb.iloc[-1]), 2)
    except Exception:  # noqa: BLE001
        pass
    try:
        f = FINANCIAL_DIR / f"{code}.parquet"
        if f.exists():
            fd = pd.read_parquet(f)
            roe_col = ("加权净资产收益率(%)" if "加权净资产收益率(%)" in fd.columns
                       else ("ROE" if "ROE" in fd.columns else None))
            if roe_col:
                vals = fd[roe_col].dropna().tail(4)
                if not vals.empty:
                    out["roe"] = round(float(vals.sum()), 1)
    except Exception:  # noqa: BLE001
        pass
    return out


# ───────────────────────── 短线层 ─────────────────────────

def _short_angle(code: str, end: pd.Timestamp, name_hint: str = "") -> dict:
    """短线四要素: 涨停基因 / 龙虎榜 / 概念共振 / 资金流 → 分数与依据。"""
    parts: list[dict] = []
    score = 0.0
    reasons: list[str] = []

    # 1) 涨停基因 (zt_pool_history)
    zt = _load_zt_history(code, end)
    zt_recent = zt[zt["date"] >= end - pd.Timedelta(days=30)] if not zt.empty else zt
    zt_days = int(len(zt_recent)) if not zt_recent.empty else 0
    max_board = int(zt_recent["board_count"].max()) if (not zt_recent.empty and "board_count" in zt_recent) else 0
    score_zt = min(W_ZT_GENE, zt_days * 10 + (5 if max_board >= 2 else 0))
    score += score_zt
    parts.append({"key": "涨停基因", "score": score_zt, "max": W_ZT_GENE,
                  "data": {"zt_30d": zt_days, "max_board": max_board}})
    if zt_days:
        reasons.append(f"近30日涨停{zt_days}次" + (f"(最高{max_board}连板)" if max_board >= 2 else ""))

    # 2) 龙虎榜 (lhb + hyyyb 席位)
    lhb = _load_lhb_window(code, end)
    lhb_n = len(lhb)
    score_lhb = 0.0
    lhb_detail: dict = {"times_60d": lhb_n}
    if lhb_n:
        net = lhb["龙虎榜净买额"].astype(float) if "龙虎榜净买额" in lhb else pd.Series(dtype=float)
        net_pos = int((net > 0).sum()) if len(net) else 0
        score_lhb += min(10, net_pos * 4)
        lhb_detail["net_buy_days"] = net_pos
        # 上榜后1日胜率(后验依据)
        if "上榜后1日" in lhb:
            fwd = lhb["上榜后1日"].astype(float).dropna()
            if len(fwd):
                win = float((fwd > 0).mean())
                avg = float(fwd.mean())
                score_lhb += 10 if win >= 0.5 else 0
                lhb_detail["fwd1_winrate"] = round(win, 2)
                lhb_detail["fwd1_avg"] = round(avg, 2)
                reasons.append(f"上榜后1日胜率{win:.0%}")
        # 机构席位（"解读"列含"机构"）
        if "解读" in lhb:
            inst = int(lhb["解读"].astype(str).str.contains("机构", na=False).sum())
            lhb_detail["institution_days"] = inst
            if inst:
                reasons.append(f"{inst}次机构席位")
    # 游资席位: hyyyb 买入股票名称串包含股票名
    youzi_hit = False
    if name_hint:
        try:
            hy = pd.read_parquet(MARKET_DIR / "lhb_hyyyb_em.parquet")
            hy["上榜日"] = pd.to_datetime(hy["上榜日"], errors="coerce")
            hy = hy[hy["上榜日"] >= end - pd.Timedelta(days=LOOKBACK_LHB * 1.7)]
            hit = hy[hy["买入股票"].astype(str).str.contains(name_hint, na=False)]
            if not hit.empty:
                youzi_hit = True
                lhb_detail["youzi_brokers"] = int(len(hit))
                lhb_detail["youzi_net"] = round(float(hit["总买卖净额"].sum()) / 1e8, 2)
                score_lhb += 10
                reasons.append(f"游资席位{len(hit)}家")
        except Exception:  # noqa: BLE001
            pass
    # 席位深度: broker_gaming 个股买5/卖5 结构判定(需 lhb_stock_detail_daily 累积)
    try:
        from quant_system.analysis_core import broker_gaming  # noqa: PLC0415
        seat = broker_gaming.stock_gaming(end.strftime("%Y%m%d"), code)
        if seat.get("note") != "无席位数据":
            structure = str(seat.get("structure", ""))
            lhb_detail["seat_structure"] = structure
            score_lhb += 5
            reasons.append(f"席位: {structure[:32]}")
    except Exception:  # noqa: BLE001 - 席位明细未累积时降级
        pass
    score += min(score_lhb, W_LHB)
    parts.append({"key": "龙虎榜", "score": min(score_lhb, W_LHB), "max": W_LHB, "data": lhb_detail})

    # 3) 概念共振 + 生命周期阶段
    concepts = _concept_of(code)
    heat = _concept_heat(concepts, end)
    stage = _concept_stage(code, concepts)
    stage_bonus = {"启动": 6, "发酵": 5, "高潮": 3, "退潮": -8}.get(stage.get("stage", ""), 0)
    score_con = max(0, min(W_CONCEPT, (14 if heat.get("hot") else 0) + (6 if concepts else 0) + stage_bonus))
    score += score_con
    stage_data = dict(heat)
    stage_data["stage"] = stage.get("stage", "")
    stage_data["stage_note"] = stage.get("note", "")
    parts.append({"key": "概念共振", "score": score_con, "max": W_CONCEPT, "data": stage_data})
    if heat.get("hot"):
        reasons.append("所属概念涨停家数≥3(热点)")
    elif concepts:
        reasons.append("有概念归属但未热")
    if stage.get("stage"):
        reasons.append(f"概念阶段: {stage['stage']}")

    # 3.5) 热度人气 (hot_rank)  — V12.3 P2-9: 传 end, 消除历史 date 前视
    hot = _hot_rank(code, end)
    score_hot = 0.0
    if hot.get("ok"):
        rank = hot["rank"]
        if rank <= 30:
            score_hot = 10
            reasons.append(f"人气榜第{rank}名")
        elif rank <= 100:
            score_hot = 6
            reasons.append(f"人气榜第{rank}名")
        elif rank <= 300:
            score_hot = 3
    score += score_hot
    parts.append({"key": "热度人气", "score": score_hot, "max": 10, "data": hot})

    # 4) 个股资金流 (复用 fund_flow_divergence)
    flow = {"ok": False}
    try:
        from quant_system.analysis_core.fund_flow_divergence import check_divergence  # noqa: PLC0415
        flow = check_divergence(code, end)
    except Exception as e:  # noqa: BLE001
        flow = {"note": f"资金流不可用: {str(e)[:60]}"}
    main_net = flow.get("main_net")
    flow_ok = isinstance(main_net, (int, float)) and main_net == main_net
    score_flow = 0.0
    if flow_ok:
        score_flow = 14 if main_net > 0 else (6 if abs(main_net) < 1e6 else 0)
        reasons.append(f"主力净流入{main_net / 1e8:+.2f}亿" if abs(main_net) >= 1e6 else "主力资金接近平衡")
    elif flow.get("signal") == "unavailable":
        pass  # 数据缺失不计分不扣分
    score += score_flow
    parts.append({"key": "资金流", "score": score_flow, "max": W_FLOW,
                  "data": {"main_net": main_net, "signal": flow.get("signal"),
                           "note": flow.get("note", ""), "ok": flow_ok}})

    env = _market_flow_env(end)
    return {"score": round(min(score, 100), 1), "parts": parts, "reasons": reasons,
            "market_env": env, "data_sources": {
                "涨停基因": "market/zt_pool_history.parquet",
                "龙虎榜": "market/lhb_20*.parquet + lhb_hyyyb_em.parquet",
                "概念": "classification/concept_member.parquet",
                "资金流": "fund_flow_divergence(东财)"}}


# ───────────────────────── 中线层 ─────────────────────────

def _mid_angle(code: str, end: pd.Timestamp) -> dict:
    """中线四要素: 融资盘 / 股东户数 / 基金持仓 / 估值 → 分数与依据。"""
    parts: list[dict] = []
    score = 0.0
    reasons: list[str] = []

    # 1) 融资盘 (margin_detail)
    mg = _margin_series(code)
    score_mg = 0.0
    mg_detail: dict = {"has_data": not mg.empty}
    if not mg.empty:
        mg = mg[mg["信用交易日期"] <= end].tail(40)
        if len(mg):
            first_bal = float(mg["融资余额"].iloc[0]) if "融资余额" in mg else 0.0
            last_bal = float(mg["融资余额"].iloc[-1]) if "融资余额" in mg else 0.0
            chg = (last_bal - first_bal) / first_bal if first_bal else 0.0
            mg_detail["balance"] = last_bal
            mg_detail["chg_40d"] = round(chg * 100, 1)
            mg_detail["last_date"] = mg["信用交易日期"].iloc[-1].strftime("%Y-%m-%d")
            if chg > 0.05:
                score_mg = W_MARGIN * 0.9
                reasons.append(f"融资余额40日+{chg * 100:.0f}%(杠杆加仓)")
            elif chg > 0:
                score_mg = W_MARGIN * 0.5
                reasons.append(f"融资余额40日+{chg * 100:.0f}%(温和)")
            else:
                score_mg = W_MARGIN * 0.15
                reasons.append(f"融资余额40日{chg * 100:.0f}%(去杠杆)")
    score += score_mg
    parts.append({"key": "融资盘", "score": score_mg, "max": W_MARGIN, "data": mg_detail})

    # 2) 股东户数 (gdhs)
    g = _gdhs_info(code)
    score_g = 0.0
    if g.get("change_pct") is not None:
        chg = g["change_pct"]
        if chg < -5:
            score_g = W_GDHS * 0.9
            reasons.append(f"股东户数-{abs(chg):.0f}%(筹码集中)")
        elif chg < 0:
            score_g = W_GDHS * 0.5
            reasons.append(f"股东户数{chg:.0f}%(微降)")
        else:
            score_g = W_GDHS * 0.1
            reasons.append(f"股东户数+{chg:.0f}%(散户化)")
    score += score_g
    parts.append({"key": "筹码(股东户数)", "score": score_g, "max": W_GDHS, "data": g})

    # 3) 基金持仓
    fh = _fund_hold(code)
    score_fh = 0.0
    if fh.get("fund_count") is not None:
        if fh["fund_count"] >= 10:
            score_fh = W_FUND_HOLD * 0.8
            reasons.append(f"基金{fh['fund_count']}家持有")
        elif fh["fund_count"] >= 1:
            score_fh = W_FUND_HOLD * 0.4
            reasons.append(f"基金{fh['fund_count']}家持有(少)")
        if fh.get("change") == "增仓":
            score_fh = min(score_fh + W_FUND_HOLD * 0.2, W_FUND_HOLD)
            reasons.append("基金增仓")
    score += score_fh
    parts.append({"key": "基金持仓", "score": score_fh, "max": W_FUND_HOLD, "data": fh})

    # 4) 估值 (PE/PB/ROE)
    val = _valuation(code)
    score_v = 0.0
    if val.get("pe") is not None and val.get("pb") is not None:
        if val["pe"] < 20 and val["pb"] < 1.6:
            score_v = W_VALUE * 0.9
            reasons.append(f"低估(PE{val['pe']}/PB{val['pb']})")
        elif val["pe"] < 30:
            score_v = W_VALUE * 0.5
            reasons.append(f"估值合理(PE{val['pe']})")
    if val.get("roe") is not None and val["roe"] >= 8:
        score_v = min(score_v + W_VALUE * 0.1, W_VALUE)
        reasons.append(f"ROE{val['roe']}%")
    score += score_v
    parts.append({"key": "估值", "score": score_v, "max": W_VALUE, "data": val})

    # 5) 产业链联动 (industry_graph 个股关联网络 + chain_map)
    chain = _chain_view(code)
    score_ch = 0.0
    n_links = len(chain.get("links", []))
    if n_links:
        score_ch = min(10, 4 + n_links)
        reasons.append(f"产业链关联{n_links}个板块")
    score += score_ch
    parts.append({"key": "产业链联动", "score": score_ch, "max": 10, "data": chain})

    # 6) 事件信号: 公告利好/利空 + 大宗机构接盘
    sig = _announcement_signals(code, end)
    bt = _block_trades(code, end)
    ev = max(-12.0, (sig.get("score") or 0.0) + (3 if bt.get("institution_buyer") else 0.0))
    score += ev
    parts.append({"key": "事件信号", "score": max(0.0, ev), "max": 10,
                  "data": {"announce": sig, "block": bt}})
    if sig.get("bullish"):
        reasons.append(f"公告利好×{len(sig['bullish'])}")
    if sig.get("bearish"):
        reasons.append(f"公告利空×{len(sig['bearish'])}")
    if bt.get("institution_buyer"):
        reasons.append(f"大宗机构接盘{bt['institution_buyer']}笔")

    return {"score": round(min(max(score, 0), 100), 1), "parts": parts, "reasons": reasons,
            "data_sources": {
                "融资盘": "market/margin_detail_{sh,sz}/*.parquet",
                "筹码": "market/gdhs_all.parquet",
                "基金": "market/fund_portfolio_hold.parquet",
                "估值": "valuation/*.parquet + financial/*.parquet"}}


# ───────────────────────── 决策合成 ─────────────────────────

def _synthesize(short: dict, mid: dict) -> dict:
    s, m = short["score"], mid["score"]
    if s >= STRONG and m >= STRONG:
        verdict, tone = "短中线共振(双强)", "进攻"
        advice = "短线资金+中线筹码形成合力, 可重点跟踪; 注意情绪周期位置控制仓位"
    elif s >= STRONG:
        verdict, tone = "短线驱动(情绪)", "进攻偏谨慎"
        advice = "游资/涨停/概念共振主导, 快进快出, 破5日线或缩量即走"
    elif m >= STRONG:
        verdict, tone = "中线驱动(价值)", "防守反击"
        advice = "低估+筹码集中+机构认可, 分批低吸, 止损设估值支撑位下方"
    elif s <= WEAK and m <= WEAK:
        verdict, tone = "双弱(观望)", "回避"
        advice = "短线无基因且中线无支撑, 回避或等待信号"
    else:
        verdict, tone = "分歧(短中背离)", "观察"
        advice = "短中方向不一致, 等一方确认; 优先尊重中期趋势"
    return {"verdict": verdict, "tone": tone, "advice": advice,
            "short_score": s, "mid_score": m}


def analyze(code: str, date: str | None = None, name_hint: str = "") -> dict:
    """个股全景透视主入口(三视角: 短线/中线/长线)。date=YYYY-MM-DD(默认最近交易日)。"""
    code = _norm_code(code)
    end = pd.Timestamp(date or _today())
    if end.tzinfo is not None:
        end = end.tz_localize(None)
    short = _short_angle(code, end, name_hint)
    mid = _mid_angle(code, end)
    long_ = _long_angle(code, end)
    synth = _synthesize3(short, mid, long_)
    return {
        "code": code, "name": name_hint, "as_of": end.strftime("%Y-%m-%d"),
        "generated_at": datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S"),
        "synth": synth,
        "short": {"score": short["score"], "parts": short["parts"],
                  "reasons": short["reasons"], "market_env": short["market_env"],
                  "data_sources": short["data_sources"]},
        "mid": {"score": mid["score"], "parts": mid["parts"],
                "reasons": mid["reasons"], "data_sources": mid["data_sources"]},
        "long": {"score": long_["score"], "parts": long_["parts"],
                 "reasons": long_["reasons"], "data_sources": long_["data_sources"]},
    }


def analyze_watchlist() -> list[dict]:
    """真实自选(quant_web/watchlist.json) + 持仓 全量透视（去重）。

    注意: watchlist.get_watchlist() 返回全市场 409 只股票池(非自选),
    不能用作扫描范围(2026-08-14 intraday_guard 同款修复)。
    """
    codes: list[str] = []
    try:
        from quant_system.trade_db import get_positions  # noqa: PLC0415
        for p in get_positions():
            if p.get("symbol"):
                codes.append(str(p["symbol"]).zfill(6))
    except Exception:  # noqa: BLE001
        pass
    try:
        wl = ROOT / "quant_web" / "watchlist.json"
        if wl.exists():
            data = json.loads(wl.read_text(encoding="utf-8"))
            items = data if isinstance(data, list) else data.get("stocks", [])
            for it in items:
                if isinstance(it, str):
                    code = it
                else:
                    code = str(it.get("code", it.get("symbol", "")))
                if code:
                    codes.append(_norm_code(code))
    except Exception:  # noqa: BLE001
        pass
    names: dict[str, str] = {}
    try:
        # 名字映射: 优先 stock_names.parquet(数据基座), 兜底自选文件内嵌名
        nm = pd.read_parquet(MARKET_DIR / "stock_names.parquet")
        if "code" in nm.columns and "name" in nm.columns:
            names = dict(zip(nm["code"].astype(str).str.zfill(6), nm["name"].astype(str)))
    except Exception:  # noqa: BLE001
        pass
    try:
        wl = ROOT / "quant_web" / "watchlist.json"
        if wl.exists():
            data = json.loads(wl.read_text(encoding="utf-8"))
            items = data if isinstance(data, list) else data.get("stocks", [])
            for it in items:
                if isinstance(it, dict) and it.get("code") and it.get("name"):
                    names.setdefault(_norm_code(it["code"]), str(it["name"]))
    except Exception:  # noqa: BLE001
        pass
    out = []
    for c in dict.fromkeys(codes):
        try:
            out.append(analyze(c, name_hint=names.get(c, "")))
        except Exception as e:  # noqa: BLE001
            _log.error(f"[stock_lens] {c} 分析失败: {e}")
    out.sort(key=lambda x: -max(x["synth"]["short_score"], x["synth"]["mid_score"],
                                x["synth"]["long_score"]))
    return out


def save_report(items: list[dict], date: str | None = None) -> Path:
    """落盘 generated/stock_lens_{date}.json（供决策链/前端读取）。"""
    GENERATED.mkdir(exist_ok=True)
    f = GENERATED / f"stock_lens_{date or _today()}.json"
    f.write_text(json.dumps({"date": date or _today(), "items": items},
                            ensure_ascii=False, indent=1), encoding="utf-8")
    return f


def main() -> None:
    ap = argparse.ArgumentParser(description="个股全景透视 (stock_lens)")
    ap.add_argument("--code", help="单股代码, 如 600519")
    ap.add_argument("--watch", action="store_true", help="自选+持仓全量")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD")
    ap.add_argument("--save", action="store_true", help="落盘 generated/stock_lens_*.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if args.watch:
        items = analyze_watchlist()
        for it in items:
            s, m, l = it["synth"]["short_score"], it["synth"]["mid_score"], it["synth"]["long_score"]
            print(f"{it['code']} {it['name']} | 短线{s} 中线{m} 长线{l} | {it['synth']['verdict']}")
            for r in it["short"]["reasons"] + it["mid"]["reasons"] + it["long"]["reasons"]:
                print(f"    · {r}")
        if args.save:
            print("落盘:", save_report(items, args.date))
    elif args.code:
        it = analyze(args.code, args.date)
        s, m, l = it["synth"]["short_score"], it["synth"]["mid_score"], it["synth"]["long_score"]
        print(f"{it['code']} {it['name']} | 短线{s} 中线{m} 长线{l} | {it['synth']['verdict']}")
        print(f"建议: {it['synth']['advice']}")
        for sec in ("short", "mid", "long"):
            print(f"── {sec} ──")
            for p in it[sec]["parts"]:
                print(f"  {p['key']}: {p['score']}/{p['max']}")
            for r in it[sec]["reasons"]:
                print(f"    · {r}")
        if args.save:
            print("落盘:", save_report([it], args.date))
    else:
        ap.print_help()



# ═══════════════════════════════════════════════════════════════
# V12.3 二期: 三视角全景增强 — 短线(热度/概念生命周期) + 中线
# (产业链/大宗/公告) + 长线(宏观/深度价值/质量成长/市场水位)
# ═══════════════════════════════════════════════════════════════

MACRO_DIR = ROOT / "data_warehouse" / "macro"
EVENTS_DIR = ROOT / "data_warehouse" / "events"
CNINFO_DIR = ROOT / "data_warehouse" / "cninfo"
HOT_DIR = ROOT / "data_warehouse" / "hot_rank"
ONEOFF_DIR = ROOT / "data_warehouse" / "oneoff"

# 公告类型 → 利好/利空/中性（事件驱动信号）
_BULLISH_KW = ("业绩预增", "业绩快报", "回购", "增持", "中标", "签订", "重组", "收购",
               "分红", "送转", "股权激励", "净利润增长", "扭亏", "预盈")
_BEARISH_KW = ("减持", "质押", "诉讼", "仲裁", "处罚", "立案", "亏损", "预亏", "预减",
               "退市", "风险警示", "冻结", "违约", "商誉减值", "计提")
_BLOCK_BULLISH = ("机构专用", "瑞银", "摩根", "高盛", "花旗", "中金", "中信证券", "沪股通", "深股通")


def _load_cninfo(code: str, end: pd.Timestamp, days: int = 30) -> list[dict]:
    """cninfo 巨潮公告流: 该股近 N 天公告列表（公告日期字段为披露日）。"""
    out = []
    start = end - pd.Timedelta(days=days)
    for f in sorted(CNINFO_DIR.glob("*.json")):
        try:
            recs = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for r in recs:
            if str(r.get("代码", "")).zfill(6) != code:
                continue
            d = str(r.get("公告日期", ""))[:10]
            if d and start <= pd.Timestamp(d) <= end:
                out.append({"date": d, "title": str(r.get("公告标题", "")),
                            "type": str(r.get("公告类型", ""))})
    out.sort(key=lambda x: x["date"])
    return out[-15:]  # 最多 15 条


def _announcement_signals(code: str, end: pd.Timestamp) -> dict:
    """公告事件: 利好/利空/中性分类 → 事件信号分(±)。"""
    recs = _load_cninfo(code, end)
    if not recs:
        return {"ok": False, "records": 0, "score": 0.0, "bullish": [], "bearish": []}
    bullish, bearish = [], []
    for r in recs:
        title = r["title"]
        if any(k in title for k in _BULLISH_KW):
            bullish.append(r["title"][:50])
        elif any(k in title for k in _BEARISH_KW):
            bearish.append(r["title"][:50])
    score = min(10, len(bullish) * 4) - min(15, len(bearish) * 8)
    return {"ok": True, "records": len(recs), "score": score,
            "bullish": bullish[:4], "bearish": bearish[:4],
            "latest": recs[-1]["title"][:60] if recs else ""}


def _block_trades(code: str, end: pd.Timestamp, days: int = 30) -> dict:
    """大宗交易: 近30日笔数/折溢价率/机构接盘（block.parquet 是东财大宗交易）。"""
    try:
        d = pd.read_parquet(EVENTS_DIR / "block.parquet")
    except Exception:  # noqa: BLE001
        return {"ok": False}
    d = d[d["证券代码"].astype(str).str.zfill(6) == code].copy()
    if d.empty:
        return {"ok": False}
    d["交易日期"] = pd.to_datetime(d["交易日期"], errors="coerce")
    d = d[(d["交易日期"] >= end - pd.Timedelta(days=days)) & (d["交易日期"] <= end)]
    if d.empty:
        return {"ok": False, "records": 0, "note": "近30日无大宗"}
    inst = int(d["买方营业部"].astype(str).str.contains("|".join(_BLOCK_BULLISH), na=False).sum())
    return {"ok": True, "records": len(d), "institution_buyer": inst,
            "amount_30d": round(float(d["成交额"].sum()) / 1e8, 2),
            "last_date": d["交易日期"].iloc[-1].strftime("%Y-%m-%d")}


def _hot_rank(code: str, end: pd.Timestamp | None = None) -> dict:
    """人气榜: 最近热度文件中的排名与变化（短线人气指标）。

    V12.3 审计 P2-9 修复: 原实现恒取 files[-1](当下最新), 历史 --date 时用当下人气
    前视。现按 end 选最近一日 ≤ end 的热度文件; end 缺省视为今天。"""
    end = end or _today()
    end_str = str(end)[:10].replace("-", "")
    try:
        files = sorted(HOT_DIR.glob("hot_rank_*.parquet"))
        if not files:
            return {"ok": False}
        # 选文件名日期 ≤ end 的最近一份（免前视）
        pick = None
        for f in files:
            day = f.stem.replace("hot_rank_", "")
            if day <= end_str:
                pick = f
            else:
                break  # files 已排序, 超界即停
        if pick is None:
            return {"ok": False}
        d = pd.read_parquet(pick)
        sub = d[d["code"].astype(str).str.replace("SH", "").str.replace("SZ", "").str.zfill(6) == code]
        if sub.empty:
            return {"ok": False}
        r = sub.iloc[0]
        return {"ok": True, "rank": int(r.get("rank", 0)), "rank_change": int(r.get("rank_change", 0)),
                "pct_chg": _safe_float(r.get("pct_chg"))}
    except Exception:  # noqa: BLE001
        return {"ok": False}


def _concept_stage(code: str, concepts: list[str]) -> dict:
    """概念生命周期: 首概念生命周期阶段(启动/发酵/高潮/退潮)。影响短线"概念共振"加分。

    V12.3 审计 P1-1 修复: 原实现只读 concept_archive 的 note/summary(两键恒空),
    stage_bonus 四键全不可达(死代码)。改为复用 theme_cycle.parquet 的 stage_cn
    (启动/分歧/爆发/回流/退潮) 映射到 启动/发酵/高潮/退潮。"""
    if not concepts:
        return {"ok": False, "note": "无概念归属"}
    STAGE_MAP = {"启动": "启动", "分歧": "发酵", "爆发": "高潮",
                 "回流": "发酵", "退潮": "退潮"}
    try:
        import pandas as _pd
        tc = _pd.read_parquet(MARKET_DIR / "theme_cycle.parquet")
        if tc is None or tc.empty or "stage_cn" not in tc.columns:
            return {"ok": False, "note": "theme_cycle 无数据"}
        bare = [str(c).split(":")[-1] for c in concepts]
        tc2 = tc.copy()
        tc2["_bare"] = tc2["concept"].astype(str).str.replace(r"^(BK\d+|THS:\d+)", "", regex=True)
        sub = tc2[tc2["_bare"].isin(bare)]
        if sub.empty:
            # 段码体系不一致(concept_member 东财 BK vs theme_cycle 同花顺)匹配失败
            # → 中性降级"有历史档案"(不虚高不误判), 不返回空导致 stage 空白
            return {"ok": True, "concept": concepts[0], "name": "",
                    "stage": "有历史档案", "stage_cn": "", "note": "概念不在 theme_cycle"}
        recent = sub.sort_values("date")
        latest = recent.iloc[-1]
        stage_cn = str(latest.get("stage_cn", ""))
        stage = STAGE_MAP.get(stage_cn, "有历史档案")
        return {"ok": True, "concept": concepts[0],
                "name": str(latest.get("board_name", latest.get("concept", ""))),
                "stage": stage, "stage_cn": stage_cn,
                "zt_cnt": int(latest.get("zt_cnt", 0) or 0),
                "date": str(latest.get("date", ""))[:10]}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "note": f"生命周期不可用: {str(e)[:50]}"}


def _chain_view(code: str) -> dict:
    """产业链/联动网络: industry_graph 个股关联 + chain_map 产业链温度。"""
    out: dict = {"ok": False}
    try:
        from quant_system.analysis_core.industry_graph import stock_links  # noqa: PLC0415
        links = stock_links(code, top_k=5)
        out["links"] = links.get("links", [])[:5]
        out["concepts"] = links.get("concepts", [])[:5]
        out["ok"] = bool(out["links"] or out["concepts"])
    except Exception as e:  # noqa: BLE001
        out["note"] = f"联动网络不可用: {str(e)[:60]}"
    try:
        from quant_system.analysis_core.chain_map import load as chain_load  # noqa: PLC0415
        cm = chain_load()
        out["chain_count"] = len(cm.chains) if hasattr(cm, "chains") else 0
    except Exception:  # noqa: BLE001
        pass
    return out


def _macro_env() -> dict:
    """宏观驱动: CPI 同比趋势 + 财新PMI景气 + 北向资金流向(市场级)。"""
    env: dict = {"ok": False, "scores": {}}
    try:
        cpi = pd.read_parquet(MACRO_DIR / "cpi_yearly.parquet")
        cpi_yoy = cpi["全国-同比增长"].dropna()
        if len(cpi_yoy) >= 3:
            recent = float(cpi_yoy.iloc[-1])
            trend = float(cpi_yoy.iloc[-1] - cpi_yoy.iloc[-4]) if len(cpi_yoy) >= 4 else 0.0
            env["cpi_yoy"] = round(recent, 1)
            env["cpi_trend"] = round(trend, 1)
            env["scores"]["通胀"] = 8 if (0 <= recent <= 3) else (4 if recent < 0 else 2)
    except Exception:  # noqa: BLE001
        pass
    try:
        pmi = pd.read_parquet(MACRO_DIR / "cx_pmi_yearly.parquet")
        sub = pmi[pmi["商品"].astype(str).str.contains("PMI", na=False)]
        val = sub["今值"].dropna()
        if len(val):
            v = float(val.iloc[-1])
            env["pmi"] = round(v, 1)
            env["scores"]["景气"] = 8 if v >= 50 else 2
    except Exception:  # noqa: BLE001
        pass
    try:
        n = pd.read_parquet(EVENTS_DIR / "north.parquet")
        n["日期"] = pd.to_datetime(n["日期"], errors="coerce")
        n = n.dropna(subset=["当日成交净买额"]).tail(20)
        if len(n):
            pos_days = int((n["当日成交净买额"].astype(float) > 0).sum())
            env["north_pos_20d"] = pos_days
            env["scores"]["外资"] = 8 if pos_days >= 12 else (4 if pos_days >= 7 else 2)
    except Exception:  # noqa: BLE001
        pass
    env["ok"] = bool(env.get("scores"))
    return env


def _market_water_level() -> dict:
    """市场水位: qvix恐慌分位 + 破净比例 + 全市场PE + 巴菲特指标。"""
    out: dict = {"ok": False}
    try:
        q = pd.read_parquet(MARKET_DIR / "qvix.parquet")
        q["date"] = pd.to_datetime(q["date"], errors="coerce")
        q = q.dropna(subset=["close"]).tail(250)
        if len(q) > 20:
            cur = float(q["close"].iloc[-1])
            pct = float((q["close"] < cur).mean())
            out["qvix"] = round(cur, 1)
            out["qvix_pct"] = round(pct, 2)
            out["scores"] = {"恐慌分位": 8 if pct >= 0.8 else (5 if pct >= 0.5 else 3)}
    except Exception:  # noqa: BLE001
        pass
    try:
        bn = pd.read_parquet(ONEOFF_DIR / "a_below_net.parquet")
        bn["date"] = pd.to_datetime(bn["date"], errors="coerce")
        bn = bn.dropna(subset=["below_net_asset_ratio"]).tail(5)
        if len(bn):
            ratio = float(bn["below_net_asset_ratio"].iloc[-1])
            out["below_net_ratio"] = round(ratio, 3)
            out["scores"] = out.get("scores", {})
            out["scores"]["破净水位"] = 7 if ratio >= 0.08 else (4 if ratio >= 0.05 else 2)
    except Exception:  # noqa: BLE001
        pass
    out["ok"] = bool(out.get("scores"))
    return out


def _quality_growth(code: str) -> dict:
    """质量成长: ROE近4期 + 销售净利率 + 每股经营性现金流 + 每股未分配利润。"""
    out: dict = {"ok": False, "scores": {}}
    try:
        f = FINANCIAL_DIR / f"{code}.parquet"
        if not f.exists():
            return out
        fd = pd.read_parquet(f)
        roe_col = ("加权净资产收益率(%)" if "加权净资产收益率(%)" in fd.columns
                   else ("ROE" if "ROE" in fd.columns else None))
        if roe_col:
            vals = fd[roe_col].dropna().tail(4)
            if len(vals) >= 2:
                roe = float(vals.sum())
                stable = float(vals.std()) if len(vals) >= 3 else 0.0
                out["roe_4q"] = round(roe, 1)
                out["roe_vol"] = round(stable, 1)
                out["scores"]["盈利质量"] = 8 if roe >= 15 else (5 if roe >= 8 else 2)
        if "销售净利率(%)" in fd.columns:
            npm = fd["销售净利率(%)"].dropna()
            if len(npm):
                out["net_margin"] = round(float(npm.iloc[-1]), 1)
        if "每股经营性现金流(元)" in fd.columns:
            cf = fd["每股经营性现金流(元)"].dropna()
            if len(cf) and float(cf.iloc[-1]) > 0:
                out["ocf_ps"] = round(float(cf.iloc[-1]), 2)
                out["scores"]["现金流"] = 5
        if "每股未分配利润(元)" in fd.columns:
            up = fd["每股未分配利润(元)"].dropna()
            if len(up) and float(up.iloc[-1]) > 0:
                out["undist_profit_ps"] = round(float(up.iloc[-1]), 2)
                out["scores"]["分红能力"] = 5
    except Exception:  # noqa: BLE001
        pass
    out["ok"] = bool(out.get("scores"))
    return out


def _deep_value(code: str) -> dict:
    """深度价值: PE/PB 历史分位（近250交易日序列）。"""
    out: dict = {"ok": False, "scores": {}}
    try:
        f = VALUATION_DIR / f"{code}.parquet"
        if not f.exists():
            return out
        v = pd.read_parquet(f)
        pe = v["peTTM"].dropna().tail(250)
        pb = v["pbMRQ"].dropna().tail(250)
        if len(pe) >= 30:
            cur_pe = float(pe.iloc[-1])
            pe_pct = float((pe < cur_pe).mean())
            out["pe"] = round(cur_pe, 1)
            out["pe_pct"] = round(pe_pct, 2)
            out["scores"]["估值分位"] = 8 if pe_pct <= 0.3 else (5 if pe_pct <= 0.5 else 2)
        if len(pb) >= 30:
            cur_pb = float(pb.iloc[-1])
            pb_pct = float((pb < cur_pb).mean())
            out["pb"] = round(cur_pb, 2)
            out["pb_pct"] = round(pb_pct, 2)
            out["scores"]["市净分位"] = 7 if pb_pct <= 0.3 else (4 if pb_pct <= 0.5 else 2)
    except Exception:  # noqa: BLE001
        pass
    out["ok"] = bool(out.get("scores"))
    return out


def _long_angle(code: str, end: pd.Timestamp | None = None) -> dict:
    """长线五要素: 宏观驱动 / 深度价值 / 质量成长 / 市场水位 / 事件底仓。"""
    parts: list[dict] = []
    score = 0.0
    reasons: list[str] = []

    env = _macro_env()
    s_env = sum(env.get("scores", {}).values())
    score += s_env
    parts.append({"key": "宏观驱动", "score": s_env, "max": 24, "data": env})
    if env.get("pmi") is not None:
        reasons.append(f"财新PMI{env['pmi']}" + ("(荣枯线上)" if env["pmi"] >= 50 else "(荣枯线下)"))
    if env.get("cpi_yoy") is not None:
        reasons.append(f"CPI同比{env['cpi_yoy']}%")

    dv = _deep_value(code)
    s_dv = sum(dv.get("scores", {}).values())
    score += s_dv
    parts.append({"key": "深度价值", "score": s_dv, "max": 15, "data": dv})
    if dv.get("pe_pct") is not None:
        reasons.append(f"PE{int(dv['pe'])}位于历史{dv['pe_pct']:.0%}分位")

    qg = _quality_growth(code)
    s_qg = sum(qg.get("scores", {}).values())
    score += s_qg
    parts.append({"key": "质量成长", "score": s_qg, "max": 18, "data": qg})
    if qg.get("roe_4q") is not None:
        reasons.append(f"近4期ROE合计{qg['roe_4q']}%")

    wl = _market_water_level()
    s_wl = sum(wl.get("scores", {}).values())
    score += s_wl
    parts.append({"key": "市场水位", "score": s_wl, "max": 15, "data": wl})
    if wl.get("qvix_pct") is not None:
        reasons.append(f"恐慌指数位于{wl['qvix_pct']:.0%}分位")
    if wl.get("below_net_ratio") is not None:
        reasons.append(f"全市场破净{wl['below_net_ratio']:.1%}")

    # 事件底仓: 公告利好 + 大宗机构接盘
    sig = _announcement_signals(code, end or pd.Timestamp(_today()))
    ev_score = max(0.0, min(14, (sig.get("score") or 0) + 4))
    score += ev_score
    parts.append({"key": "事件底仓", "score": ev_score, "max": 14, "data": sig})
    if sig.get("bullish"):
        reasons.append(f"公告利好×{len(sig['bullish'])}")
    if sig.get("bearish"):
        reasons.append(f"公告利空×{len(sig['bearish'])}")

    return {"score": round(min(score, 86), 1), "parts": parts, "reasons": reasons,
            "data_sources": {
                "宏观": "data_warehouse/macro/*.parquet + events/north.parquet",
                "估值分位": "valuation/*.parquet",
                "质量": "financial/*.parquet",
                "水位": "market/qvix.parquet + oneoff/a_below_net.parquet",
                "事件": "cninfo/*.json + events/block.parquet"}}


def _synthesize3(short: dict, mid: dict, long_: dict) -> dict:
    """三维合成: 长线为底仓/中线为波段/短线为博弈。

    V12.3 审计 P1-6: 短/中 max=100, 长 max=86(宏观24+深度15+质量18+水位15+事件14),
    统一归一化到 100 制再判强(避免长线因分母小更苛刻/判强不等价)。"""
    s_raw, m_raw, l_raw = short["score"], mid["score"], long_["score"]
    s, m, l = s_raw, m_raw, min(100.0, l_raw / 86.0 * 100.0)  # 长线 86→100 归一
    strong = [k for k, v in (("短线", s), ("中线", m), ("长线", l)) if v >= STRONG]
    n = len(strong)
    if n == 3:
        verdict, tone, advice = "三线共振(主升)", "进攻", \
            "短线题材+中线筹码+长线价值全部共振, 强势主升特征; 分批跟进, 回踩不破关键均线持有"
    elif n == 2:
        if "短线" in strong and "中线" in strong:
            verdict, tone, advice = "情绪+波段共振", "进攻偏谨慎", \
                "资金面与筹码面形成合力, 可波段持有; 短线破位即减, 中线逻辑未破可留底仓"
        elif "中线" in strong and "长线" in strong:
            verdict, tone, advice = "价值波段", "防守反击", \
                "低估+筹码+宏观支持, 中线价值品种; 逢回调分批建仓, 止损设估值支撑下方"
        else:
            verdict, tone, advice = "题材价值", "谨慎进攻", \
                "题材热度+长线价值, 但缺中线筹码确认; 短线参与为主, 仓位控制"
    elif n == 1:
        if "短线" in strong:
            verdict, tone, advice = "短线驱动(纯情绪)", "进攻偏谨慎", \
                "游资博弈主导, 快进快出, 严格止损; 中线/长线未确认不重仓"
        elif "中线" in strong:
            verdict, tone, advice = "中线驱动(筹码)", "防守反击", \
                "筹码与资金面支持, 可波段持有; 关注长线估值是否配合"
        else:
            verdict, tone, advice = "长线价值(左侧)", "逢低布局", \
                "宏观+估值+质量支持, 当前可能处于左侧; 分批低吸, 耐心持有"
    else:
        verdict, tone, advice = "三线皆弱(回避)", "回避", \
            "短中长均无支撑, 回避或等待右侧信号确认"
    return {"verdict": verdict, "tone": tone, "advice": advice,
            "short_score": s_raw, "mid_score": m_raw, "long_score": l_raw,
            "strong_axes": strong}


if __name__ == "__main__":
    main()


