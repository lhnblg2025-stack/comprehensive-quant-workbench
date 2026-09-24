"""
resonance_scorer — 五维共振评分器（V11 信号系统核心）

把"多源交叉验证"从手工变成引擎：对每个当日活跃概念，
从 涨停池/量能/龙虎榜席位/研报热度 四个本地可用的维度加权评分
（新闻维度预留，本地无新闻缓存时权重重分配）：

  涨停维度 40%: 概念当日涨停家数（theme_cycle）
  量能维度 25%: 概念涨停股成交额合计（EM三池 × concept_member）
  席位维度 20%: 当日龙虎榜机构净买个股命中概念成分（lhb + 解读含机构）
  研报维度 15%: 概念名关键词命中当日研报文本条数（可选）

输出分级（供 battle_map 攻击分组）:
  score ≥ 2.0 → 🔴 主线（核心攻击）
  1.2 ≤ score < 2.0 → 🟡 支线（轻仓试错）
  score < 1.2 → ⚪ 非主线（不参与/回避）

输出: generated/resonance_{date}.parquet + resonance_{date}.json

用法:
  python3 -m quant_system.analysis_core.resonance_scorer --today
"""

from __future__ import annotations
import logging

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.config import MARKET_DIR, ZT_EM_DAILY  # noqa: E402

CST = timezone(timedelta(hours=8))
CONCEPT_MEMBER = MARKET_DIR.parent / "classification" / "concept_member.parquet"
THEME_CYCLE = MARKET_DIR / "theme_cycle.parquet"

# 维度权重（新闻无本地缓存时自动重分配）
W_LIMIT, W_VOL, W_FLOW, W_BROKER, W_REPORT = 0.35, 0.15, 0.20, 0.15, 0.15
SCORE_MAIN, SCORE_SUB = 2.0, 1.2

# 泛化/交易属性板块（非题材，过滤掉；与 theme_cycle.BROAD_KEYWORDS 对齐+补充）
BROAD_KEYWORDS = ["融资融券", "股通", "标准普尔", "MSCI", "富时", "证金", "汇金", "转融",
                  "昨日", "高振幅", "打板", "龙虎榜", "涨停", "跌停", "预盈", "预亏",
                  "次新", "破净", "高股息", "低价股", "高送转", "举牌", "股权转让",
                  "最近多板", "东方财富热股", "小盘股", "中盘股", "大盘股", "活跃股", "高换手",
                  "中报预增", "年报预增", "一季报预增", "三季报预增", "预盈预增", "预亏预减",
                  "回购", "增持", "减持", "股权激励", "员工持股", "国企改革", "央企",
                  "重组", "借壳", "ST", "摘帽", "题材股", "业绩预增", "高送转预期",
                  "新高", "新低", "活跃", "创新高", "创历史",
                  "QFII", "社保", "养老金", "机构重仓", "基金重仓", "百元股", "破发股",
                  "破增发价", "股权分散", "股权集中", "高贝塔", "低贝塔", "微盘股", "超跌", "绩优", "绩差",
                  "中证500", "中证1000", "上证50", "沪深300", "深成500", "创业板综", "上证380",                  "高市净率", "低市净率", "高市盈率", "低市盈率", "高贝塔", "低贝塔",
                  "新型城镇化", "乡村振兴", "一带一路", "西部大开发", "京津冀", "长三角", "粤港澳",
                  "深股通", "沪股通", "富时罗素", "标普道琼斯", "央视50", "茅指数", "宁组合",
                  "科技风格", "大盘成长", "中盘成长", "小盘成长", "中报首亏", "中报预减",
                  "业绩预减", "业绩首亏", "亏损股", "微利股"]


def is_broad_board(name: str) -> bool:
    return any(k in name for k in BROAD_KEYWORDS)


def _latest_trade_date() -> str:
    """取 theme_cycle 最新日期作为分析日。"""
    df = pd.read_parquet(THEME_CYCLE)
    return str(df["date"].max())[:10]


def _load_concept_map() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """code → 概念代码集合 + concept → 成分代码集合（东财 BK + THS 双源）。

    复用 theme_cycle.load_concept_map（lru_cache 缓存整表读取，THS 概念带
    'THS:' 前缀），概念集比原单东财源更大，属预期增强。
    theme_cycle 在模块级 import 本模块 → 延迟 import 避免循环导入。
    """
    from quant_system.analysis_core.theme_cycle import load_concept_map as _tc_load
    code2con: dict[str, set[str]] = {}
    con2codes: dict[str, set[str]] = {}
    for code, cons in _tc_load().items():
        codes = set(cons)
        code2con[code] = codes
        for c in codes:
            con2codes.setdefault(c, set()).add(code)
    return code2con, con2codes


def _load_lhb_day(date: str) -> tuple[set[str], set[str]]:
    """当日龙虎榜: (净买>0的代码集, 机构买入的代码集)。"""
    inst_hit: set[str] = set()
    net_buy_hit: set[str] = set()
    files = sorted(MARKET_DIR.glob("lhb_20*.parquet"))
    for f in files:
        try:
            df = pd.read_parquet(f, columns=["代码", "上榜日", "龙虎榜净买额", "解读"])
        except Exception as e:
            logging.getLogger(__name__).error(f"[resonance_scorer] 操作失败: {e}", exc_info=True)
            continue
        df["代码"] = df["代码"].astype(str).str.zfill(6)
        day = df[pd.to_datetime(df["上榜日"]).dt.strftime("%Y-%m-%d") == date]
        if day.empty:
            continue
        net_buy_hit |= set(day[day["龙虎榜净买额"] > 0]["代码"])
        inst = day[day["解读"].astype(str).str.contains("机构", na=False)]
        inst_hit |= set(inst["代码"])
    return net_buy_hit, inst_hit


def _load_report_text(date: str) -> str:
    """当日研报文本（zsxq 大佛投研缓存，缺则空串）。"""
    candidates = [
        ROOT / "file_read_pools" / date / f"zsxq_reports_{date}.md",
        ROOT / "generated" / f"zsxq_reports_{date}.md",
        ROOT / "data_warehouse" / "reports" / f"zsxq_reports_{date}.md",
    ]
    for p in candidates:
        if p.exists():
            return p.read_text(encoding="utf-8", errors="ignore")
    return ""


def _load_sector_flow(date: str) -> dict[str, float]:
    """概念板块主力净额(亿) → {概念名: 净额}。

    前视防护: 仅当 date == 今天 才用即时抓取（今天评今天）;
    历史日期尝试按日历史接口，不可用返回 {} → 调用方权重重分配。
    缓存落盘 sector_fund_flow.parquet（即时抓取时）。
    """
    cache = MARKET_DIR / "sector_fund_flow.parquet"
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    got = None
    if date == today:
        try:
            import akshare as ak
            df = ak.stock_fund_flow_concept(symbol="即时")
            if df is not None and len(df):
                df = df.rename(columns={"行业": "concept_name", "净额": "net_yi"})
                df["date"] = today
                got = df[["date", "concept_name", "net_yi"]].copy()
                # 无锁 read-modify-write 并发防护（2026-08-10 审计: cron+手动并发会互相覆盖）
                try:
                    import fcntl
                    with open(cache, "a+") as _lk:
                        fcntl.flock(_lk, fcntl.LOCK_EX)
                        if cache.exists():
                            old = pd.read_parquet(cache)
                            old = old[old["date"] != today]  # 同日覆盖
                            got = pd.concat([old, got], ignore_index=True)
                        got = got.drop_duplicates(subset=["date", "concept_name"]).tail(5000)
                        got.to_parquet(cache, index=False)
                        fcntl.flock(_lk, fcntl.LOCK_UN)
                except ImportError:
                    if cache.exists():
                        old = pd.read_parquet(cache)
                        old = old[old["date"] != today]
                        got = pd.concat([old, got], ignore_index=True)
                    got = got.drop_duplicates(subset=["date", "concept_name"]).tail(5000)
                    got.to_parquet(cache, index=False)
        except Exception as e:
            print(f"  [resonance] 板块资金流抓取失败({str(e)[:60]})")
        # 抓取失败 → 回退最近缓存（但仅限今天）
        if got is None and cache.exists():
            got = pd.read_parquet(cache)
    else:
        # 历史日期: 尝试按日历史接口（不可用则无此维度）
        try:
            import akshare as ak
            hist_rows = []
            for probe in ("PCB概念", "CRO概念"):
                try:
                    hd = ak.stock_concept_fund_flow_hist(symbol=probe)
                    if hd is not None and len(hd) and "日期" in hd.columns:
                        hd = hd[hd["日期"].astype(str).str.contains(date[:7])]
                        hist_rows.append(hd)
                except Exception as e:
                    logging.getLogger(__name__).error(f"[resonance_scorer] 操作失败: {e}", exc_info=True)
                    continue
            if hist_rows:
                allh = pd.concat(hist_rows, ignore_index=True)
                if "净额" in allh.columns:
                    day = allh[allh["日期"].astype(str).str.contains(date)]
                    if len(day):
                        got = day.rename(columns={"行业": "concept_name", "净额": "net_yi"})[["concept_name", "net_yi"]]
        except Exception:
            got = None
    if got is None or got.empty:
        return {}
    if "date" in got.columns:
        latest = got.sort_values("date").groupby("concept_name").tail(1)
        return dict(zip(latest["concept_name"], latest["net_yi"].astype(float)))
    return dict(zip(got["concept_name"], got["net_yi"].astype(float)))


def score_day(date: str | None = None) -> pd.DataFrame:
    date = date or _latest_trade_date()
    tc = pd.read_parquet(THEME_CYCLE)
    # 2026-08-10 审计：题材周期数据落后 >3 日 → 输出 lag_days 标注（消费方按需降级）
    _lag = (datetime.now(CST).date() - pd.Timestamp(date).date()).days
    tc["date"] = pd.to_datetime(tc["date"]).dt.strftime("%Y-%m-%d")
    day = tc[tc["date"] == date].copy()
    if day.empty:
        return pd.DataFrame()

    em = pd.read_parquet(ZT_EM_DAILY)
    em["date"] = pd.to_datetime(em["date"]).dt.strftime("%Y-%m-%d")
    em_day = em[(em["date"] == date) & (em["is_zt"])].copy()
    em_day["code"] = em_day["code"].astype(str).str.zfill(6)
    em_amt = dict(zip(em_day["code"], em_day["amount"]))

    code2con, con2codes = _load_concept_map()
    net_buy_hit, inst_hit = _load_lhb_day(date)
    report_text = _load_report_text(date)
    sector_flow = _load_sector_flow(date)
    has_flow = bool(sector_flow)

    # 维度权重（资金流缺失时重分配给涨停/量能）
    w_l, w_v, w_f = W_LIMIT, W_VOL, W_FLOW
    if not has_flow:
        w_f = 0.0
        w_l += W_FLOW * 0.6
        w_v += W_FLOW * 0.4
    # 席位维度权重（LHB T+1 缺失时重分配给涨停/量能/资金流）
    seat_avail = bool(net_buy_hit or inst_hit)
    w_b = W_BROKER if seat_avail else 0.0
    if not seat_avail:
        extra = W_BROKER
        w_l += extra * 0.4
        w_v += extra * 0.3
        w_f += extra * 0.3

    rows = []
    for _, r in day.iterrows():
        con = r["concept"]
        zt_cnt = int(r["zt_cnt"]) if pd.notna(r["zt_cnt"]) else 0
        board_name = r.get("board_name", con)

        # ── 1. 涨停维度 ──
        s_limit = 0.0
        if zt_cnt >= 8:
            s_limit = 1.2
        elif zt_cnt >= 5:
            s_limit = 0.8
        elif zt_cnt >= 3:
            s_limit = 0.4

        # ── 2. 量能维度（概念涨停股成交额合计）──
        codes = con2codes.get(con, set())
        amt = sum(v for c, v in em_amt.items() if c in codes)
        amt_yi = amt / 1e8
        s_vol = 0.0
        if amt_yi >= 200:
            s_vol = 0.75
        elif amt_yi >= 100:
            s_vol = 0.5
        elif amt_yi >= 50:
            s_vol = 0.25

        # ── 2.5 资金流维度（板块主力净额, 亿）──
        flow_yi = 0.0
        for nm, fv in sector_flow.items():
            if nm and (nm in board_name or board_name in nm):
                flow_yi = fv
                break
        s_flow = 0.0
        if flow_yi >= 50:
            s_flow = 1.0
        elif flow_yi >= 20:
            s_flow = 0.6
        elif flow_yi >= 5:
            s_flow = 0.3
        elif flow_yi <= -30:
            s_flow = -0.6
        elif flow_yi <= -10:
            s_flow = -0.3

        # ── 3. 席位维度（龙虎榜命中）──
        hit_net = bool(codes & net_buy_hit)
        hit_inst = bool(codes & inst_hit)
        s_broker = (0.4 if hit_inst else 0.0) + (0.2 if hit_net else 0.0)

        # ── 4. 研报维度（概念名关键词命中）──
        s_report = 0.0
        if report_text and board_name:
            cnt = report_text.count(board_name)
            if cnt >= 15:
                s_report = 0.45
            elif cnt >= 10:
                s_report = 0.3
            elif cnt >= 5:
                s_report = 0.15

        # ── 加权总分（资金流可为负）──
        score = s_limit * w_l * 3 + s_vol * w_v * 3 + s_flow * w_f * 3 + s_broker * w_b * 3 + s_report * W_REPORT * 3
        score = round(min(3.0, max(-1.0, score)), 2)
        level = "主线" if score >= SCORE_MAIN else ("支线" if score >= SCORE_SUB else "非主线")

        if is_broad_board(board_name):
            continue

        rows.append({
            "date": date, "concept": con, "board_name": board_name,
            "zt_cnt": zt_cnt, "amount_yi": round(amt_yi, 1),
            "flow_yi": round(flow_yi, 1),
            "broker_net_hit": hit_net, "broker_inst_hit": hit_inst,
            "report_hits": report_text.count(board_name) if report_text else 0,
            "s_limit": s_limit, "s_vol": s_vol, "s_broker": s_broker, "s_report": s_report,
            "score": score, "level": level, "role": r.get("role", ""),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("score", ascending=False).reset_index(drop=True)
        out["seat_note"] = "" if seat_avail else "T+1缺失"
        out["lag_days"] = max(0, _lag)
        # 动态分位分级（避免大量板块挤在阈值上）: 前20%且≥1.5=主线, 前50%且≥1.0=支线
        if len(out) >= 10:
            q80 = out["score"].quantile(0.80)
            q50 = out["score"].quantile(0.50)
            out["level"] = np.where(
                (out["score"] >= max(1.5, q80)), "主线",
                np.where((out["score"] >= max(1.0, q50)), "支线", "非主线"))
        else:
            out["level"] = np.where(out["score"] >= SCORE_MAIN, "主线",
                                     np.where(out["score"] >= SCORE_SUB, "支线", "非主线"))
        OUT_PQ = ROOT / "generated" / f"resonance_{date}.parquet"
        OUT_PQ.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(OUT_PQ, index=False)
        (ROOT / "generated" / f"resonance_{date}.json").write_text(
            json.dumps(json.loads(out.to_json(orient="records")), ensure_ascii=False, indent=2),
            encoding="utf-8")
    return out


def print_top(df: pd.DataFrame, n: int = 15) -> str:
    if df.empty:
        return "无当日活跃概念"
    lines = ["🔥 五维共振评分（多源交叉验证）", "-" * 60]
    if len(df) and int(df.iloc[0].get("lag_days", 0)) > 3:
        lines.append(f"⚠️ 数据降级: 题材周期落后 {int(df.iloc[0]['lag_days'])} 日，评分仅供参考")
    lines.append(f"{'板块':<14}{'涨停':<5}{'量能(亿)':<9}{'席位':<8}{'研报':<5}{'评分':<6}级别")
    for _, r in df.head(n).iterrows():
        seat = ("机构+" if r["broker_inst_hit"] else "") + ("净买" if r["broker_net_hit"] else "-")
        lines.append(f"{str(r['board_name'])[:13]:<14}{int(r['zt_cnt']):<5}"
                     f"{r['amount_yi']:<9.1f}{seat:<8}{int(r['report_hits']):<5}"
                     f"{r['score']:<6.2f}{r['level']}")
    mains = df[df["level"] == "主线"]
    if len(mains):
        lines.append(f"\n🔴 主线: {', '.join(mains['board_name'].astype(str).head(6))}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="五维共振评分")
    ap.add_argument("--today", action="store_true")
    ap.add_argument("--date", default=None)
    args = ap.parse_args()
    df = score_day(args.date)
    print(print_top(df))
