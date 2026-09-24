#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline, auditable whole-market weekly report built only from data_warehouse.

Public API: build_market_weekly(as_of=None) -> JSON-safe dict.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DW = ROOT / "data_warehouse"
MARKET = DW / "market"
INDUSTRY = DW / "industry"

INDEXES = {
    "上证": "上证指数", "深成": "深证成指", "沪深300": "沪深300", "中证500": "中证500",
    "中证1000": "中证1000", "上证50": "上证50", "创业板": "创业板指", "科创50": "科创50", "红利": "红利指数",
}
PE_INDEXES = ["沪深300", "中证500", "中证1000", "上证50"]


def _json(v: Any) -> Any:
    if v is None or isinstance(v, (str, bool, int)):
        return v
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, float):
        return None if not math.isfinite(v) else round(v, 8)
    if hasattr(v, "item"):
        return _json(v.item())
    if isinstance(v, dict):
        return {str(k): _json(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_json(x) for x in v]
    return str(v)


def _read(rel: str):
    try:
        import pandas as pd
        p = DW / rel
        return pd.read_parquet(p) if p.exists() else None
    except Exception:
        return None


def _dt(v):
    import pandas as pd
    try:
        return pd.to_datetime(v).dt.normalize() if hasattr(v, "dt") else pd.to_datetime(v).normalize()
    except Exception:
        return None


def _num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def _status(value: Any = None, *, source: str, observed: Any = None, note: str = "", status: str | None = None) -> dict:
    return {"status": status or ("available" if value is not None else "missing"), "value": _json(value),
            "observed_date": _json(observed), "source": source, "note": note}


def _chapter(conclusion: str, evidence: Any, implication: str, next_check: str) -> dict:
    return {"conclusion": conclusion, "evidence": _json(evidence), "implication": implication, "next_check": next_check}


def _week_dates(as_of: date):
    """Return the current Monday-to-cutoff trading week without future dates."""
    start = as_of - timedelta(days=as_of.weekday())
    end = min(as_of, start + timedelta(days=4))
    return start, end


def _index_report(start, end, gaps):
    import pandas as pd
    out = {}
    for label, file_label in INDEXES.items():
        source = f"data_warehouse/market/index_daily_{file_label}.parquet"
        d = _read(f"market/index_daily_{file_label}.parquet")
        if d is None or d.empty:
            out[label] = _status(source=source, note="文件缺失或不可读"); gaps.append(source); continue
        d = d.copy(); d["_date"] = pd.to_datetime(d["date"]).dt.date
        w = d[(d["_date"] >= start) & (d["_date"] <= end)].sort_values("_date")
        if w.empty or w["close"].dropna().empty:
            out[label] = _status(source=source, note="本周无可用交易日"); gaps.append(f"{source}: {start}~{end}"); continue
        close = w["close"].dropna(); first, last = _num(close.iloc[0]), _num(close.iloc[-1])
        change = (last / first - 1) if first not in (None, 0) and last is not None else None
        volume_sum = _num(pd.to_numeric(w.get("volume"), errors="coerce").sum()) if "volume" in w else None
        prior = d[(d["_date"] >= start - timedelta(days=7)) & (d["_date"] <= end - timedelta(days=7))]
        prior_volume = _num(pd.to_numeric(prior.get("volume"), errors="coerce").sum()) if "volume" in prior else None
        volume_change = (volume_sum / prior_volume - 1) if volume_sum is not None and prior_volume else None
        vals = {"first_close": first, "last_close": last, "weekly_return": change,
                "weekly_return_pct": change * 100 if change is not None else None,
                "week_high": _num(pd.to_numeric(w["high"], errors="coerce").max()) if "high" in w else None,
                "week_low": _num(pd.to_numeric(w["low"], errors="coerce").min()) if "low" in w else None,
                "volume_sum": volume_sum, "prior_week_volume_sum": prior_volume,
                "volume_vs_prior_week": volume_change, "trading_days": int(len(w)),
                "first_date": w["_date"].iloc[0], "last_date": w["_date"].iloc[-1]}
        out[label] = _status(vals, source=source, observed=vals["last_date"])
    return out


def _margin(start, end, gaps):
    import pandas as pd
    frames = []
    for market in ("sh", "sz"):
        source = f"data_warehouse/market/market_margin_{market}.parquet"; d = _read(f"market/market_margin_{market}.parquet")
        if d is not None and not d.empty:
            d = d.copy(); d["_date"] = pd.to_datetime(d["日期"]).dt.date; frames.append(d)
        else: gaps.append(source)
    if not frames:
        gaps.append("market_margin: file_missing")
        return _status(source="data_warehouse/market/market_margin_sh.parquet + market_margin_sz.parquet", note="沪深两融文件均缺失", status="missing")
    d = pd.concat(frames, ignore_index=True); w = d[(d._date >= start) & (d._date <= end)]
    if w.empty:
        gaps.append("market_margin: no_observation_in_period")
        return _status(source="data_warehouse/market/market_margin_[sh|sz].parquet", note="本周无披露日", status="missing")
    sums = w.groupby("_date")[["融资买入额", "融资余额", "融券余额", "融资融券余额"]].sum().sort_index()
    first, last = sums.iloc[0], sums.iloc[-1]
    value = {"week_start": {"date": sums.index[0], **{k: _num(first[k]) for k in sums.columns}},
             "week_end": {"date": sums.index[-1], **{k: _num(last[k]) for k in sums.columns}},
             "融资买入额_sum": _num(sums["融资买入额"].sum()),
             "融券余额_change": _num(last["融券余额"] - first["融券余额"]),
             "融资融券余额_change": _num(last["融资融券余额"] - first["融资融券余额"]), "disclosure_dates": list(sums.index)}
    return _status(value, source="data_warehouse/market/market_margin_sh.parquet + market_margin_sz.parquet", observed=sums.index[-1])


def _flow(start, end, gaps):
    import pandas as pd
    source = "data_warehouse/market/stock_market_fund_flow.parquet"; d = _read("market/stock_market_fund_flow.parquet")
    if d is None or d.empty:
        gaps.append("market_flow: file_missing")
        return _status(source=source, note="文件缺失", status="missing")
    d = d.copy(); d["_date"] = pd.to_datetime(d["日期"]).dt.date; w = d[(d._date >= start) & (d._date <= end)].sort_values("_date")
    if w.empty:
        gaps.append("market_flow: no_observation_in_period")
        return _status(source=source, note="本周无数据", status="missing")
    col = "主力净流入-净额"
    daily = [{"date": row["_date"], "主力净流入": _num(row.get(col))} for _, row in w.iterrows()]
    value = {"主力净流入_sum_yi": _num(w[col].sum() / 1e8) if col in w else None,
        "主力净流入_sum_raw": _num(w[col].sum()) if col in w else None,
        "trading_days": len(w), "coverage_start": w["_date"].iloc[0], "coverage_end": w["_date"].iloc[-1],
        "coverage_complete": len(w) == 5, "daily": daily}
    if len(w) < 5:
        value["coverage_note"] = "数据覆盖不足，不能代表完整周累计"
        gaps.append("market_flow: partial_coverage")
        return _status(value, source=source, observed=w._date.iloc[-1], note=value["coverage_note"], status="partial")
    return _status(value, source=source, observed=w._date.iloc[-1], note=value.get("coverage_note", ""))


def _basis(start, end, gaps):
    import pandas as pd
    source = "data_warehouse/market/futures_basis.parquet"; d = _read("market/futures_basis.parquet")
    if d is None or d.empty:
        gaps.append("futures_basis: file_missing")
        return _status(source=source, note="文件缺失", status="missing")
    d = d.copy(); d["_date"] = pd.to_datetime(d.date).dt.date; w = d[(d._date >= start) & (d._date <= end)]
    if w.empty:
        gaps.append("futures_basis: no_observation_in_period")
        return _status(source=source, note="本周无数据", status="missing")
    result = {}
    for c, g in w.groupby("contract"):
        vals = g.basis_rate.dropna()
        result[str(c)] = {"weekly_mean": _num(vals.mean()), "last": _num(vals.iloc[-1]) if len(vals) else None,
                          "high": _num(vals.max()) if len(vals) else None, "low": _num(vals.min()) if len(vals) else None,
                          "observed_date": g._date.max(), "observations": len(vals)}
    return _status(result, source=source, observed=w._date.max())


def _valuation(as_of, gaps):
    import pandas as pd
    source = "data_warehouse/market/index_pe.parquet"; d = _read("market/index_pe.parquet")
    if d is None or d.empty: gaps.append(source); return {}
    d = d.copy(); d["_date"] = pd.to_datetime(d["日期"]).dt.date; out = {}
    for name in PE_INDEXES:
        g = d[(d.symbol == name) & (d._date <= as_of)].sort_values("_date")
        col = "滚动市盈率" if "滚动市盈率" in g else "静态市盈率"
        vals = pd.to_numeric(g[col], errors="coerce").dropna()
        if vals.empty: out[name] = _status(source=source, note="无历史估值"); continue
        cur = _num(vals.iloc[-1]); pct = float((vals <= cur).mean()) if cur is not None else None
        out[name] = _status({"pe": cur, "historical_percentile": pct, "history_count": len(vals)}, source=source, observed=g._date.iloc[-1])
    return out


def _macro(as_of, gaps):
    import pandas as pd
    specs = {"pmi": ("macro_monthly__pmi.parquet", "月份", "制造业-指数"), "cpi": ("macro_monthly__cpi.parquet", "月份", "全国-同比增长"), "ppi": ("macro_monthly__ppi.parquet", "月份", "当月同比增长")}
    out = {}
    for key, (fn, dc, vc) in specs.items():
        source = f"data_warehouse/market/{fn}"; d = _read(f"market/{fn}")
        if d is None or d.empty: gaps.append(source); out[key] = _status(source=source, note="文件缺失"); continue
        d = d.copy(); d["_date"] = pd.to_datetime(d[dc].astype(str).str.extract(r"(\d{4})年?(\d{1,2})?", expand=True).apply(lambda r: f"{r.iloc[0]}-{int(r.iloc[1] or 1):02d}-01", axis=1), errors="coerce").dt.date
        g = d[d._date <= as_of].sort_values("_date").dropna(subset=[vc])
        if g.empty: out[key] = _status(source=source, note="无真实观察"); continue
        cur = _num(g[vc].iloc[-1]); prev = _num(g[vc].iloc[-2]) if len(g) > 1 else None
        out[key] = _status({"value": cur, "change": cur - prev if cur is not None and prev is not None else None, "metric": vc}, source=source, observed=g._date.iloc[-1])
    for key, fn, dc, vc in [("shibor", "shibor.parquet", "date", "rate"), ("repo", "repo_rate.parquet", "date", "FR007"), ("us_rate", "rates__us_rate.parquet", "日期", "美国国债收益率10年")]:
        source=f"data_warehouse/market/{fn}"; d=_read(f"market/{fn}")
        if d is None or d.empty: gaps.append(source); out[key]=_status(source=source, note="文件缺失"); continue
        d=d.copy(); d["_date"]=pd.to_datetime(d[dc]).dt.date; g=d[d._date<=as_of].sort_values("_date").dropna(subset=[vc])
        if g.empty: out[key]=_status(source=source, note="无真实观察"); continue
        cur=_num(g[vc].iloc[-1]); prev=_num(g[vc].iloc[-2]) if len(g)>1 else None
        out[key]=_status({"value":cur,"change":cur-prev if cur is not None and prev is not None else None,"metric":vc},source=source,observed=g._date.iloc[-1])
    return out


def _industries(start, end, gaps):
    import pandas as pd
    source="data_warehouse/industry/sw_first_hist.parquet"; d=_read("industry/sw_first_hist.parquet"); m=_read("industry/sw_first.parquet")
    if d is None or d.empty:
        gaps.append("industry: file_missing")
        return _status(source=source, note="申万一级历史文件缺失", status="missing")
    d=d.copy(); d["_date"]=pd.to_datetime(d["日期"]).dt.date; w=d[(d._date>=start)&(d._date<=end)].sort_values("_date")
    if w.empty:
        gaps.append("industry: no_observation_in_period")
        return _status(source=source, note="本周无行业观察", status="missing")
    rows=[]
    for code,g in w.groupby("代码"):
        g=g.dropna(subset=["收盘"]); first,last=(_num(g.收盘.iloc[0]),_num(g.收盘.iloc[-1])) if len(g) else (None,None)
        rows.append({"code":str(code),"name":str(code),"weekly_return":last/first-1 if first and last else None,"amount_sum":_num(g["成交额"].sum()),"observations":len(g)})
    if m is not None and not m.empty:
        mp=dict(zip(m["行业代码"].astype(str).str.replace('.SI','',regex=False),m["行业名称"].astype(str)))
        for r in rows: r["name"]=mp.get(r["code"],r["name"])
    rows=[r for r in rows if r["weekly_return"] is not None]; rows.sort(key=lambda x:x["weekly_return"], reverse=True)
    value={"classification":"申万一级指数","top10":rows[:10],"bottom10":rows[-10:][::-1],"count":len(rows),"note":"这是申万一级指数行情，不是主力资金流"}
    return _status(value,source=source,observed=w._date.max())


def build_market_weekly(as_of=None) -> dict:
    import pandas as pd
    if as_of is None:
        candidates=[]
        for p in MARKET.glob("index_daily_*.parquet"):
            d=_read(str(p.relative_to(DW)))
            if d is not None and not d.empty: candidates.append(pd.to_datetime(d.date).max().date())
        as_of=max(candidates) if candidates else date.today()
    elif isinstance(as_of, str): as_of=pd.to_datetime(as_of).date()
    elif isinstance(as_of, datetime): as_of=as_of.date()
    start,end=_week_dates(as_of); gaps=[]
    indices=_index_report(start,end,gaps)
    trading_days=max([x["value"]["trading_days"] for x in indices.values() if x.get("status")=="available"] or [0])
    index_returns = {k: ((v or {}).get("value") or {}).get("weekly_return") for k, v in indices.items()}
    valid_returns = [value for value in index_returns.values() if value is not None]
    summary = {"index_returns": index_returns, "positive_indexes": sum(1 for value in valid_returns if value > 0), "negative_indexes": sum(1 for value in valid_returns if value < 0), "trading_days": trading_days}
    margin=_margin(start,end,gaps); flow=_flow(start,end,gaps); basis=_basis(start,end,gaps); valuation=_valuation(as_of,gaps); macro=_macro(as_of,gaps); industries=_industries(start,end,gaps)
    rs={}
    base=((indices.get("沪深300") or {}).get("value") or {}).get("weekly_return")
    for k,v in indices.items():
        v = v or {}; r=((v.get("value") or {}).get("weekly_return")); rs[k]={"weekly_return":r,"excess_vs_沪深300":r-base if r is not None and base is not None else None,"rolling_26w_excess":None,"note":"滚动26周需跨周历史计算，当前未计算"}
    sources=sorted(set([f"data_warehouse/market/index_daily_{x}.parquet" for x in INDEXES.values()]+[
        "data_warehouse/market/market_margin_sh.parquet","data_warehouse/market/market_margin_sz.parquet",
        "data_warehouse/market/stock_market_fund_flow.parquet","data_warehouse/market/futures_basis.parquet",
        "data_warehouse/market/index_pe.parquet","data_warehouse/market/macro_monthly__pmi.parquet",
        "data_warehouse/market/macro_monthly__cpi.parquet","data_warehouse/market/macro_monthly__ppi.parquet",
        "data_warehouse/market/shibor.parquet","data_warehouse/market/repo_rate.parquet",
        "data_warehouse/market/rates__us_rate.parquet","data_warehouse/industry/sw_first_hist.parquet",
        "data_warehouse/industry/sw_first.parquet"]))
    flow_value = flow.get("value") or {}
    risk=["市场资金流覆盖不足" if flow_value.get("coverage_complete") is False else None,"相对强弱滚动26周未计算","宽度指标缺少全市场涨跌家数历史"]
    risk=[x for x in risk if x]
    # 计算缺失和分析覆盖不足同样属于报告缺口，不能只写在章节
    # 文案里却把契约状态误报为完整可用。
    computed_gaps = []
    if gaps:
        gaps.append("decision: 本周不形成方向性结论")
    if any(item.get("rolling_26w_excess") is None for item in rs.values()):
        computed_gaps.append("relative_strength.rolling_26w_excess")
    computed_gaps.append("market_breadth.advance_decline")
    gaps.extend(computed_gaps)
    def module_conclusion(label, module, available_text, missing_text):
        status = module.get("status") if isinstance(module, dict) else None
        observed = module.get("observed_date") if isinstance(module, dict) else None
        if status == "available":
            return f"{available_text}（实际观察日：{observed or as_of}）"
        note = module.get("note") if isinstance(module, dict) else None
        return f"{missing_text} 当前状态：{status or '未确认'}。{note or ''}"

    chapters={
      "核心结论":_chapter(f"本周 {summary['positive_indexes']} 个指数上涨、{summary['negative_indexes']} 个指数下跌；指数行情可用，其余模块按真实状态降级。",summary,"指数用于描述市场表面方向；资金、宽度和行业缺口不外推。","先完成数据质量检查后再形成仓位结论。"),
      "宏观观察":_chapter("宏观观察按每项真实发布日期呈现，不能把滞后背景当作本周同步信号。",macro,"结合实际观察日判断宏观背景。","宏观模块没有可用观察，不形成宏观方向结论。"),
      "指数涨跌":_chapter("指数周度涨跌和高低点按日线计算。",indices,"比较大盘、成长和红利风格。","指数行情缺失，不形成市场方向结论。"),
      "成交量宽度":_chapter("成交量可用；全市场涨跌家数宽度历史未在指定数据中发现。",{"trading_days":trading_days,"breadth":None},"不以单一指数成交量替代市场宽度。","补充全市场涨跌家数后计算宽度。"),
      "两融":_chapter(module_conclusion("两融", margin, "沪深两融本周按披露日完成合并。", "本周没有沪深两融披露观测，不能形成杠杆资金方向结论。"),margin,"余额变化与融资买入额用于验证风险偏好。","下一披露日补齐后再判断。"),
      "市场资金":_chapter(module_conclusion("市场资金", flow, "市场资金完成周度汇总。", "本周没有市场资金流观测，不能形成主力资金方向结论。"),flow,"覆盖不足时不外推完整周。","补齐完整五日资金流。"),
      "股指期货基差":_chapter(module_conclusion("股指期货基差", basis, "期指基差提供 IF/IC/IM/IH 的周均、末值和区间。", "本周没有 IF/IC/IM/IH 基差观测，不能形成基差方向结论。"),basis,"观察股指期货风险定价。","补齐下一周期指基差。"),
      "估值":_chapter("四指数滚动市盈率及历史分位来自 index_pe。",valuation,"分位仅表示样本历史位置，不构成投资建议。","估值缺失，不形成估值结论。"),
      "相对强弱":_chapter("相对沪深300的周超额已计算，滚动26周因本版未形成周序列暂缺。",rs,"避免把单周超额当作趋势确认。","累计满26个周点后计算滚动超额。"),
      "行业轮动":_chapter(module_conclusion("行业轮动", industries, "行业排序使用申万一级指数历史行情。", "本周没有申万行业观测，不能形成行业轮动结论。"),industries,"行业成交额与收益用于轮动观察，不称为主力资金。","补齐本周行业日线。"),
      "风险":_chapter("风险标记集中在覆盖不足和未计算指标。",risk,"报告不对缺失数据作方向性替代。","优先补齐资金、宽度和滚动序列。"),
      "下周验证":_chapter("验证项已按数据缺口和关键观察列出。",["资金流完整覆盖","两融新披露","股指期货基差","26周相对强弱"],"将验证结果与本周基线对照。","下周生成时自动复核。"),
    }
    gap_details = []
    for gap in sorted(set(gaps)):
        field, _, reason = gap.partition(": ")
        reason = reason or "not_computed"
        impact = {
            "no_observation_in_period": "本周不形成方向性结论，仅保留历史背景",
            "partial_coverage": "不能代表完整周累计，禁止外推",
        }.get(reason, "对应模块降级，不替代为其他数据")
        gap_details.append({"field": field, "reason": reason, "impact": impact,
                            "fallback": "补齐报告区间且日期不晚于截点后自动恢复"})
    report = {"schema_version": "market_weekly.v2", "scope": "analysis_decision_assist_only", "scope_detail": {"analysis": "全市场指数、流动性、资金、期指、估值、宏观与申万一级行业", "execution": "仅作分析与决策辅助，不连接券商执行"}, "period_start": start, "period_end": end, "trading_days": trading_days, "as_of": as_of, "summary": summary, "indices": indices, "margin": margin, "market_flow": flow, "basis": basis, "valuation": valuation, "relative_strength": rs, "macro": macro, "industries": industries, "risk_flags": risk, "watchlist": ["两融余额变化", "主力净流入完整覆盖", "股指期货基差周均与末值", "沪深300相对强弱"], "sources": sources, "data_gaps": gap_details, "chapters": chapters}
    # Keep the weekly data contract as the source of truth while exposing the
    # richer report contract for all report consumers.
    try:
        try:
            from scripts.report_contract import make_report_contract, validate_report_contract
        except ImportError:
            from report_contract import make_report_contract
        evidence = []
        for label, item in indices.items():
            if item.get("status") == "available":
                evidence.append({"id": f"ev-{len(evidence) + 1}", "claim": f"{label}本周涨跌", "value": item.get("value", {}).get("weekly_return"), "unit": "比例", "observed_at": item.get("observed_date") or as_of, "source_ref": f"src-{sources.index(item.get('source', '')) + 1}" if item.get("source") in sources else "src-1", "quality": "primary"})
        gap_fields = {item.get("field") for item in report["data_gaps"] if isinstance(item, dict)}
        module_status = {
            "market_margin": margin, "market_flow": flow, "futures_basis": basis,
            "industry": industries,
        }
        sources_contract = []
        for i, source in enumerate(sources):
            module = next((key for key in module_status if key in source), None)
            module_value = module_status.get(module, {}) if module else {}
            sources_contract.append({"id": f"src-{i + 1}", "name": source, "kind": "warehouse",
                "status": module_value.get("status", "available"),
                "observed_at": module_value.get("observed_date") or as_of,
                "locator": source, "note": module_value.get("note") or "本地真实数据仓库；以实际观察日为准"})
        chapters_contract = [{"title": title, **chapter} for title, chapter in chapters.items()]
        report["research_contract"] = make_report_contract("weekly", "全市场周度报告", as_of, subject={"id": "market", "name": "全市场", "kind": "market"}, report_id=f"weekly:market:{as_of}", period_start=start, period_end=end, trading_days=trading_days, summary={"stance": "observe", "summary": chapters["核心结论"]["conclusion"], "horizon": "weeks", "confidence": None}, evidence=evidence, transmission=[{"from": "指数/流动性/资金", "to": "风格与风险预算", "mechanism": "周度趋势、资金与两融共同决定风险偏好", "direction": "mixed", "evidence_refs": [item["id"] for item in evidence], "confidence": None}], risks=[{"description": value, "trigger": value, "impact": "报告降级或不作方向性推断", "severity": "medium", "evidence_refs": [item["id"] for item in evidence[:3]]} for value in risk], actions=[{"action": "validate", "target": item, "condition": "下周取得完整且同日数据", "invalidated_by": "数据日期错位或来源失败", "horizon": "next_week", "evidence_refs": [item["id"] for item in evidence[:3]]} for item in report["watchlist"]], data_gaps=report["data_gaps"], sources=sources_contract, chapters=chapters_contract, metadata={"data_contract": "market_weekly.v2"})
        contract_problems = validate_report_contract(report["research_contract"])
        report["research_contract"]["quality"]["validated"] = not contract_problems
        if contract_problems:
            report["research_contract_error"] = "; ".join(contract_problems[:12])
    except Exception as exc:
        report["research_contract_error"] = str(exc)[:160]
    return _json(report)


def render_markdown(r: dict) -> str:
    def pct(x): return "缺失" if x is None else f"{x*100:.2f}%"
    lines=[f"# 全市场周度报告（{r['period_start']} 至 {r['period_end']}）",f"数据截点：{r['as_of']}；交易日：{r['trading_days']}；Schema：{r['schema_version']}","","## 核心结论",r['chapters']['核心结论']['conclusion'],"", "## 宏观观察"]
    for k,v in r['macro'].items(): lines.append(f"- **{k}**：{v.get('value',{}).get('value','缺失') if v.get('status')=='available' else '缺失'}；观察日 {v.get('observed_date','缺失')}；{v.get('note','')}")
    lines += ["", "## 指数涨跌", "|指数|周涨跌|周高|周低|成交量|状态|", "|---|---:|---:|---:|---:|---|"]
    for k,v in r['indices'].items():
        x=v.get('value') or {}; lines.append(f"|{k}|{pct(x.get('weekly_return'))}|{x.get('week_high','缺失')}|{x.get('week_low','缺失')}|{x.get('volume_sum','缺失')}|{v.get('status')}|")
    for title in ["成交量宽度","两融","市场资金","股指期货基差","估值","相对强弱","行业轮动","风险","下周验证"]:
        c=r['chapters'][title]; lines += ["",f"## {title}",c['conclusion'],f"证据：`{json.dumps(c['evidence'],ensure_ascii=False,default=str)[:4000]}`",f"含义：{c['implication']}",f"下次检查：{c['next_check']}"]
    lines += ["", "## 数据来源与缺口", *[f"- {x}" for x in r['sources']], ""]
    if r.get("data_gaps"):
        lines.append("### 缺口影响")
        for gap in r["data_gaps"]:
            if isinstance(gap, dict):
                lines.append(f"- {gap.get('field')}: {gap.get('reason')}；影响：{gap.get('impact')}；恢复：{gap.get('fallback')}")
            else:
                lines.append(f"- {gap}")
    else:
        lines.append("缺口：无")
    return "\n".join(lines)+"\n"


def main():
    ap=argparse.ArgumentParser(description="离线全市场周度报告")
    ap.add_argument("--as-of", help="数据截点 YYYY-MM-DD")
    ap.add_argument("--out", help="输出 markdown 路径或 .json 路径")
    args=ap.parse_args(); report=build_market_weekly(args.as_of); md=render_markdown(report)
    if args.out:
        p=Path(args.out); p.parent.mkdir(parents=True,exist_ok=True)
        if p.suffix.lower()=='.json':
            p.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding='utf-8')
            p.with_suffix('.md').write_text(md,encoding='utf-8')
        else:
            p.write_text(md,encoding='utf-8')
            p.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding='utf-8')
        from report_contract import validate_and_write_contract
        validate_and_write_contract(p, report['research_contract'])
    else: print(md)

if __name__ == '__main__': main()
