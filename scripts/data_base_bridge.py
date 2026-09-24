#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据基座融合桥（2026-08-21 —— 用户"榨干所有模块，数据基座到分析决策链"）

把尚未接入决策链的数据基座榨成轻量信号（全部小函数、秒级、缺失降级）：
 1. macro_pulse()     → 宏观指标脉动（CPI/PPI/失业率/M1/M2 最新值+趋势）
 2. hot_pulse()       → 东财热榜热度（top 股/涨家数 → 市场关注方向）
 3. institutional()   → 季度机构动向（新进/增持 top 股，读 quarterly）
 4. industry_roll()   → 行业轮动（industry 数据 → 强势行业）

并入决策链: 作为 market_temperature 的补充分量 + 决策卡"基座覆盖"展示。
原则: 只读本地 parquet、单项缺失返回空不炸链。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
DW = ROOT / "data_warehouse"


def _read(rel: str, tail: int | None = None):
    """读 data_warehouse/{rel}，尾部 tail 行；缺失返回 None。"""
    import pandas as pd
    p = DW / rel
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if tail and len(df) > tail:
        df = df.tail(tail)
    return df


# ═══════════════════════════════════════════════════════════
# 1. 宏观指标脉动
# ═══════════════════════════════════════════════════════════
_MACRO_KEY = {"cpi", "ppi", "m1", "m2", "urban_unemployment", "pmi", "social_financing"}


def macro_pulse() -> dict:
    """宏观指标最新值（30 文件：cpi/ppi/m1/m2/失业率/pmi 等）。"""
    out = {"items": [], "count": 0, "note": ""}
    try:
        import glob
        for f in sorted(glob.glob(str(DW / "macro" / "*.parquet"))):
            name = f.split("/")[-1].replace(".parquet", "").lower()
            if name not in _MACRO_KEY:
                continue
            try:
                df = _read(f"macro/{name}.parquet")
                if df is None or not len(df):
                    continue
                # 按 date 排序取最新值
                if "date" in df.columns:
                    df = df.sort_values("date")
                    v = df.iloc[-1]
                    val = v.get("value")
                    d = str(v.get("date", ""))[:10]
                    if val is not None and val == val:
                        out["items"].append({"name": name.replace("urban_unemployment", "失业率"),
                                             "value": float(val),
                                             "date": d,
                                             "pct_chg": ""})
            except Exception:  # noqa: BLE001
                continue
        out["count"] = len(out["items"])
        out["note"] = "宏观" if out["count"] else "宏观数据缺失"
    except Exception as e:  # noqa: BLE001
        out["note"] = f"宏观异常: {str(e)[:40]}"
    return out


# ═══════════════════════════════════════════════════════════
# 2. 东财热榜热度
# ═══════════════════════════════════════════════════════════
def hot_pulse(top_n: int = 8) -> dict:
    """东财热榜: top 股 + 涨跌家数 → 市场关注方向。"""
    import glob
    out = {"top": [], "up_n": 0, "down_n": 0, "note": ""}
    try:
        fs = sorted(glob.glob(str(DW / "hot_rank" / "hot_rank_*.parquet")))
        if not fs:
            out["note"] = "热榜缺失"
            return out
        df = _read(f"hot_rank/{fs[-1].split('/')[-1]}")
        if df is None or not len(df):
            out["note"] = "热榜空"
            return out
        # top 股
        df = df.sort_values("rank") if "rank" in df.columns else df
        for _, r in df.head(top_n).iterrows():
            out["top"].append({"code": str(r.get("code", "")), "rank": int(r.get("rank", 0) or 0),
                               "pct_chg": float(r.get("pct_chg", 0) or 0)})
        if "pct_chg" in df.columns:
            out["up_n"] = int((df["pct_chg"] > 0).sum())
            out["down_n"] = int((df["pct_chg"] <= 0).sum())
        out["note"] = f"热榜{len(out['top'])}股 涨{out['up_n']}跌{out['down_n']}"
    except Exception as e:  # noqa: BLE001
        out["note"] = f"热榜异常: {str(e)[:40]}"
    return out


# ═══════════════════════════════════════════════════════════
# 3. 季度机构动向
# ═══════════════════════════════════════════════════════════
def institutional(top_n: int = 6) -> dict:
    """季度持仓: 新进/增持 top（读 quarterly/gdfx_*_新进.parquet 等）。"""
    import glob
    out = {"new_entries": [], "increased": [], "note": ""}
    try:
        fs = sorted(glob.glob(str(DW / "quarterly" / "*_新进.parquet")))
        if fs:
            df = _read(f"quarterly/{fs[-1].split('/')[-1]}")
            if df is not None and len(df):
                for _, r in df.head(top_n).iterrows():
                    out["new_entries"].append({
                        "name": r.get("股票简称") or r.get("股东名称", ""),
                        "code": str(r.get("股票代码", "") or ""),
                        "type": r.get("股东类型", ""), "report": str(r.get("报告期", ""))[:10]})
        up = sorted(glob.glob(str(DW / "quarterly" / "*增持.parquet")))
        if up:
            df2 = _read(f"quarterly/{up[-1].split('/')[-1]}")
            if df2 is not None and len(df2):
                for _, r in df2.head(top_n).iterrows():
                    out["increased"].append({"name": r.get("股票简称") or r.get("股东名称", ""),
                                             "code": str(r.get("股票代码", "") or "")})
        out["note"] = f"机构新进{len(out['new_entries'])} 增持{len(out['increased'])}"
        if not out["new_entries"] and not out["increased"]:
            out["note"] = "季度持仓数据缺失"
    except Exception as e:  # noqa: BLE001
        out["note"] = f"机构数据异常: {str(e)[:40]}"
    return out


# ═══════════════════════════════════════════════════════════
# 4. 行业轮动
# ═══════════════════════════════════════════════════════════
def industry_roll(top_n: int = 6) -> dict:
    """行业轮动: industry 目录最新行业表现 → 强势行业。"""
    import glob
    out = {"strong": [], "note": ""}
    try:
        fs = sorted(glob.glob(str(DW / "industry" / "*.parquet")))
        if not fs:
            out["note"] = "行业数据缺失"
            return out
        # 优先 sw_second_spot(申万二级现价: 最新价/昨收盘算涨跌)
        spot = DW / "industry" / "sw_second_spot.parquet"
        if spot.exists():
            df = _read("industry/sw_second_spot.parquet")
            if df is not None and len(df) and "最新价" in df.columns:
                df = df.copy()
                df["chg"] = (df["最新价"].astype(float) / df["昨收盘"].astype(float) - 1) * 100
                nm_col = "指数名称" if "指数名称" in df.columns else df.columns[1]
                for _, r in df.sort_values("chg", ascending=False).head(top_n).iterrows():
                    out["strong"].append({"name": str(r.get(nm_col, "?")),
                                          "chg": round(float(r.get("chg", 0) or 0), 2)})
                out["note"] = f"行业强: {'、'.join(str(s['name'])[:8] for s in out['strong'][:3])}"
        if not out["strong"]:
            # fallback: 其他行业表
            for f in fs[-5:]:
                try:
                    df = _read(f"industry/{f.split('/')[-1]}")
                    if df is None or not len(df): continue
                    cols = [c for c in df.columns if "chg" in c.lower() or "涨跌" in c]
                    if not cols: continue
                    nm_col = "name" if "name" in df.columns else (df.columns[1] if len(df.columns) > 1 else df.columns[0])
                    for _, r in df.sort_values(cols[0], ascending=False).head(top_n).iterrows():
                        out["strong"].append({"name": str(r.get(nm_col, "?")), "chg": float(r.get(cols[0], 0) or 0)})
                    if out["strong"]:
                        out["note"] = f"行业强: {'、'.join(s['name'] for s in out['strong'][:3])}"
                        break
                except Exception:  # noqa: BLE001
                    continue
        if not out["strong"]:
            out["note"] = "行业数据未解析"
    except Exception as e:  # noqa: BLE001
        out["note"] = f"行业异常: {str(e)[:40]}"
    return out


# ═══════════════════════════════════════════════════════════
# 5. 北向 + 两融（资金面补充）
# ═══════════════════════════════════════════════════════════
def north_margin() -> dict:
    """资金基座: 仅两融余额(北向接口已停更 2024+, 2026-08-21 移除北向)。"""
    out = {"margin_total_yi": None, "note": ""}
    try:
        m = _read("events/margin.parquet")
        if m is not None and len(m) and "融资融券余额" in m.columns:
            out["margin_total_yi"] = round(float(m["融资融券余额"].sum() or 0) / 1e8, 0)
            out["note"] = f"两融{out['margin_total_yi']:.0f}亿"
        else:
            out["note"] = "两融数据缺失"
    except Exception as e:  # noqa: BLE001
        out["note"] = f"资金域异常: {str(e)[:40]}"
    return out


# ═══════════════════════════════════════════════════════════
# 6. 风格剪刀差（大盘 vs 小盘，用指数 stats 简化）
# ═══════════════════════════════════════════════════════════
def style_pulse() -> dict:
    """大小盘风格: 上证50/沪深300(大盘) vs 中证1000/中证500(小盘) 近5日涨跌。"""
    out = {"large_pct5": None, "small_pct5": None, "style": "", "note": ""}
    try:
        import pandas as pd
        def _idx5(nm):
            df = _read(f"market/index_daily_{nm}.parquet")
            if df is None or len(df) < 6:
                return None
            c = df.sort_values("date")["close"].astype(float)
            return float((c.iloc[-1] / c.iloc[-6] - 1) * 100)
        large = [_idx5("上证50"), _idx5("沪深300")]
        small = [_idx5("中证1000"), _idx5("中证500")]
        lv = [x for x in large if x is not None]
        sv = [x for x in small if x is not None]
        if lv and sv:
            out["large_pct5"] = round(sum(lv) / len(lv), 2)
            out["small_pct5"] = round(sum(sv) / len(sv), 2)
            out["style"] = "小盘强" if out["small_pct5"] > out["large_pct5"] + 0.5 else (
                "大盘强" if out["large_pct5"] > out["small_pct5"] + 0.5 else "均衡")
            out["note"] = f"风格: {out['style']} (大盘{out['large_pct5']:+.1f}% 小盘{out['small_pct5']:+.1f}%)"
        else:
            out["note"] = "风格数据缺失"
    except Exception as e:  # noqa: BLE001
        out["note"] = f"风格异常: {str(e)[:40]}"
    return out


# ═══════════════════════════════════════════════════════════
# 7. 事件域: 大宗交易 + ETF 资金流
# ═══════════════════════════════════════════════════════════
def event_pulse() -> dict:
    """大宗交易(最新日成交额合计/笔数) + ETF 净流入(近5日估)。"""
    out = {"block_amount_yi": None, "block_date": "", "block_cnt": 0,
           "etf_flow": None, "note": ""}
    try:
        import pandas as pd
        b = _read("events/block.parquet")
        if b is not None and len(b) and "交易日期" in b.columns and "成交额" in b.columns:
            b = b.sort_values("交易日期")
            last_day = b["交易日期"].iloc[-1]
            day = b[b["交易日期"] == last_day]
            out["block_amount_yi"] = round(float(day["成交额"].sum() or 0) / 1e8, 2)
            out["block_cnt"] = int(len(day))
            out["block_date"] = str(last_day)[:10]
            # 折溢率(负=折价, 机构出货信号)
            if "折溢率" in day.columns:
                dis = day["折溢率"].dropna()
                if len(dis):
                    out["block_discount"] = round(float(dis.mean()), 2)
        e = _read("events/etf_flow.parquet")
        if e is not None and len(e) > 6 and "amount" in e.columns and "close" in e.columns:
            # ETF 资金流估算: 近5日 amount 方向(拐头)
            e = e.sort_values("date") if "date" in e.columns else e
            c = e["close"].astype(float); a = e["amount"].astype(float)
            chg5 = float((c.iloc[-1] / c.iloc[-6] - 1) * 100)
            amt_avg = float(a.tail(5).mean())
            out["etf_flow"] = {"chg_pct5": round(chg5, 2), "amount_avg_yi": round(amt_avg / 1e8, 1)}
        parts = []
        if out["block_amount_yi"] is not None:
            _d = out.get("block_discount")
            parts.append(f"大宗{out['block_amount_yi']:.1f}亿({out['block_cnt']}笔)"
                         + (f" 折{_d:+.2f}%" if _d is not None else ""))
        if out["etf_flow"]:
            parts.append(f"ETF5日{out['etf_flow']['chg_pct5']:+.1f}% 均{out['etf_flow']['amount_avg_yi']}亿")
        out["note"] = " ".join(parts) or "事件域缺失"
    except Exception as e:  # noqa: BLE001
        out["note"] = f"事件域异常: {str(e)[:40]}"
    return out


# ═══════════════════════════════════════════════════════════
# 8. 概念生命周期（活跃概念/再次炒作）
# ═══════════════════════════════════════════════════════════
def concept_pulse() -> dict:
    """概念生命周期: 活跃概念数 + 再次炒作计数（concept_lifecycle.concept_report 3.9s）。"""
    out = {"active": 0, "again": 0, "again_names": [], "note": ""}
    try:
        import sys as _s2
        _s2.path.insert(0, str(ROOT / "quant_system"))
        from quant_system.analysis_core import concept_lifecycle  # noqa: PLC0415
        r = concept_lifecycle.concept_report()
        if isinstance(r, dict):
            out["active"] = int(r.get("active_concept_count", 0) or 0)
            out["again"] = int(r.get("again_count", 0) or 0)
            out["again_names"] = (r.get("again_concepts") or [])[:5]
            out["note"] = f"概念{out['active']}活跃 再炒作{out['again']}" +                           ("(" + ", ".join(str(x)[:8] for x in out["again_names"][:3]) + ")" if out["again_names"] else "")
    except Exception as e:  # noqa: BLE001
        out["note"] = f"概念周期异常: {str(e)[:40]}"
    return out


# ═══════════════════════════════════════════════════════════
# 综合: 全基座脉动（供决策链/前端）
# ═══════════════════════════════════════════════════════════
def data_base_pulse() -> dict:
    """四合一基座脉动（榨干剩余基座到决策链）。"""
    return {
        "macro": macro_pulse(),
        "hot": hot_pulse(),
        "institutional": institutional(),
        "industry": industry_roll(),
        "north_margin": north_margin(),
        "style": style_pulse(),
        "event": event_pulse(),
        "concept": concept_pulse(),
    }


if __name__ == "__main__":
    import sys as _s
    try:
        _s.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    p = data_base_pulse()
    print(json.dumps(p, ensure_ascii=False, indent=1)[:800])