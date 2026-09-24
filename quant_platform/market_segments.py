"""
quant_platform.market_segments — 市场口径分层（V3 复盘体系 P5）

用户 #11：日常分析复盘要针对主板、非ST、北交双创、科创创业。
用户 #12：分析指标要多样化，明确复盘针对的是周频还是日频。

本模块把全市场快照按板块口径分层统计：
- 全A（原始，附录用）
- 主板（60/00，剔除 ST/*ST）
- 非ST主板（60/00 剔除 ST）
- 双创（创业板 30/301 + 科创板 688）
- 北交所（8/4/920，单独口径）
- 风格分层（大盘/中盘/小盘/微盘 → 由市值分位）

每层输出：涨跌家数/等权涨跌幅/涨跌停/量能/成交额。
供日报（日频）与周报（周频）复用。
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def classify_board(code: str) -> str:
    """按代码前缀分类板块。返回: main / gem(创业板) / star(科创板) / bj(北交所) / other。

    兼容格式：600519 / sz000001 / SH600519 / 920982 / bj920982。
    """
    c = str(code).strip().lower()
    # 去交易所前缀（sz/sh/bj 或带点后缀）
    for suf in (".sh", ".sz", ".bj"):
        if c.endswith(suf):
            c = c[: -len(suf)]
            break
    for pre in ("sh", "sz", "bj"):
        if c.startswith(pre):
            c = c[len(pre):]
            break
    if not c.isdigit():
        return "other"
    if c.startswith(("8", "4", "92")):
        return "bj"
    if c.startswith(("30", "301", "302")):
        return "gem"          # 创业板
    if c.startswith(("688", "689")):
        return "star"         # 科创板
    if c.startswith(("60", "00", "001", "002", "003")):
        return "main"         # 主板（沪60/深00）
    return "other"            # 指数/基金/其他


def is_st(name: str) -> bool:
    """ST/*ST 判定（名称含 ST）。"""
    return "ST" in str(name).upper()


def segment_market(spot: pd.DataFrame) -> dict[str, Any]:
    """全市场快照分层统计。

    Args:
        spot: 快照 DataFrame，需含列: 代码/名称/涨跌幅/成交额（可缺成交额）。

    Returns:
        {segment: {指标...}, "meta": {...}}
    """
    if spot is None or spot.empty:
        return {"meta": {"error": "empty spot"}}
    df = spot.copy()
    # 兼容英文列（realtime 快照: code/name/pct_chg/amount_wan）与中文列（日报 spot）
    if "代码" not in df.columns and "code" in df.columns:
        df["代码"] = df["code"]
    if "名称" not in df.columns and "name" in df.columns:
        df["名称"] = df["name"]
    if "涨跌幅" not in df.columns and "pct_chg" in df.columns:
        df["涨跌幅"] = pd.to_numeric(df["pct_chg"], errors="coerce")
    if "成交额" not in df.columns and "amount_wan" in df.columns:
        df["成交额"] = pd.to_numeric(df["amount_wan"], errors="coerce") * 1e4  # 万元→元
    if "代码" not in df.columns:
        return {"meta": {"error": "missing 代码/code column"}}
    df["代码"] = df["代码"].astype(str)
    df["_board"] = df["代码"].map(classify_board)
    df["_st"] = df["名称"].astype(str).map(is_st) if "名称" in df.columns else False

    segs: dict[str, pd.DataFrame] = {
        "全A": df,
        "主板": df[(df["_board"] == "main") & (~df["_st"])],
        "创业板": df[(df["_board"] == "gem") & (~df["_st"])],
        "科创板": df[(df["_board"] == "star") & (~df["_st"])],
        "双创": df[(df["_board"].isin(["gem", "star"])) & (~df["_st"])],
        "北交所": df[df["_board"] == "bj"],
        "剔除ST北交": df[~df["_st"] & (df["_board"] != "bj")],
    }

    out: dict[str, Any] = {}
    # 空板块也返回完整键集（None/0），避免下游渲染 KeyError（用户 #6 输出一致性）
    _EMPTY = {"股票数": 0, "等权涨跌幅%": None, "涨跌幅中位数%": None,
              "上涨": 0, "下跌": 0, "上涨占比%": None, "涨超5%": 0, "跌超5%": 0,
              "近似涨停": 0, "近似跌停": 0}
    for name, sub in segs.items():
        if sub.empty or "涨跌幅" not in sub.columns:
            out[name] = dict(_EMPTY)
            continue
        ret = pd.to_numeric(sub["涨跌幅"], errors="coerce").dropna()
        if ret.empty:
            out[name] = {**_EMPTY, "股票数": int(len(sub))}
            continue
        limit_up = int((ret >= 9.8).sum())   # 简化：主板阈值；双创/北交实际更高，见下方修正
        limit_down = int((ret <= -9.8).sum())
        # 涨跌停阈值按板块修正
        if name in ("创业板", "科创板", "双创"):
            limit_up = int((ret >= 19.8).sum())
            limit_down = int((ret <= -19.8).sum())
        elif name == "北交所":
            limit_up = int((ret >= 29.8).sum())
            limit_down = int((ret <= -29.8).sum())
        entry = {
            "股票数": int(len(sub)),
            "等权涨跌幅%": round(float(ret.mean()), 2),
            "涨跌幅中位数%": round(float(ret.median()), 2),
            "上涨": int((ret > 0).sum()),
            "下跌": int((ret < 0).sum()),
            "上涨占比%": round(float((ret > 0).mean() * 100), 1),
            "涨超5%": int((ret >= 5).sum()),
            "跌超5%": int((ret <= -5).sum()),
            "近似涨停": limit_up,
            "近似跌停": limit_down,
        }
        if "成交额" in sub.columns:
            amt = pd.to_numeric(sub["成交额"], errors="coerce").dropna()
            if len(amt):
                entry["成交额(亿)"] = round(float(amt.sum()) / 1e8, 1)
        out[name] = entry
    out["meta"] = {"segments": list(segs.keys()), "note": "口径: 主板=60/00剔除ST; 双创=创业+科创剔除ST; 北交单独; 涨跌停阈值按板块(主板9.8/双创19.8/北交29.8)"}
    return out


def style_segments(spot: pd.DataFrame) -> dict[str, Any]:
    """风格分层（大盘/小盘/微盘）：按成交额分位近似（无市值时）。"""
    if spot is None or spot.empty:
        return {"meta": {"error": "empty spot"}}
    df = spot.copy()
    # 兼容英文列（realtime 快照: amount_wan 万元）
    if "成交额" not in df.columns and "amount_wan" in df.columns:
        df["成交额"] = pd.to_numeric(df["amount_wan"], errors="coerce") * 1e4
    if "涨跌幅" not in df.columns and "pct_chg" in df.columns:
        df["涨跌幅"] = pd.to_numeric(df["pct_chg"], errors="coerce")
    if "成交额" not in df.columns:
        return {"meta": {"error": "need 成交额"}}
    df = df.dropna(subset=["成交额"]).copy()
    if df.empty:
        return {"meta": {"error": "empty after dropna"}}
    df["成交额"] = pd.to_numeric(df["成交额"], errors="coerce")
    amt = df["成交额"].dropna()
    if amt.empty:
        return {"meta": {"error": "no valid 成交额"}}
    q33, q67 = amt.quantile(0.33), amt.quantile(0.67)
    out = {}
    for name, mask in (("大票(成交额top33%)", df["成交额"] >= q67),
                       ("中票", (df["成交额"] >= q33) & (df["成交额"] < q67)),
                       ("小票(成交额bottom33%)", df["成交额"] < q33)):
        sub = df[mask]
        if sub.empty or "涨跌幅" not in sub.columns:
            out[name] = {"股票数": int(len(sub)), "等权涨跌幅%": None}
            continue
        ret = pd.to_numeric(sub["涨跌幅"], errors="coerce").dropna()
        out[name] = {
            "股票数": int(len(sub)),
            "等权涨跌幅%": round(float(ret.mean()), 2) if len(ret) else None,
            "上涨占比%": round(float((ret > 0).mean() * 100), 1) if len(ret) else None,
        }
    out["meta"] = {"note": "风格近似：成交额分位（无市值快照时）；正式口径用市值需 realtime 快照含流通市值"}
    return out


def short_medium_long(spot: pd.DataFrame, lhb: pd.DataFrame | None = None,
                      ind_map: pd.DataFrame | None = None,
                      valuation_files: list | None = None) -> dict[str, Any]:
    """短/中/长线视角指标（用户 #13）。

    - 短线：龙虎榜活跃度（近N日上榜家数/机构参与）、涨停梯队（连板数分布）
    - 中线：行业强弱（按申万行业等权涨幅 Top/Bottom）
    - 长线：估值分位（全市场 peTTM 中位数）、宏观温度（预留）
    """
    out: dict[str, Any] = {}
    # 短线：龙虎榜活跃度
    if lhb is not None and not lhb.empty:
        out["短线_龙虎榜"] = {
            "近N日上榜股票数": int(lhb["代码"].nunique()) if "代码" in lhb else None,
            "总上榜次数": int(len(lhb)),
        }
    # 短线：涨停梯队（从 spot 近似，兼容英文列 pct_chg）
    if spot is not None and not spot.empty:
        _col = "涨跌幅" if "涨跌幅" in spot.columns else ("pct_chg" if "pct_chg" in spot.columns else None)
        _code_col = "代码" if "代码" in spot.columns else ("code" if "code" in spot.columns else None)
        if _col:
            ret = pd.to_numeric(spot[_col], errors="coerce").dropna()
            out["短线_涨停梯队"] = {
                "涨停数(≥9.8%)": int((ret >= 9.8).sum()),
                "20cm涨停数(≥19.8%)": int((ret >= 19.8).sum()),
            }
    # 短线：涨停股所属概念聚合（热点题材识别，用户 #10/#11/#13）
    # 涨停股代码 → concept_member 反查 → 概念计数，Top 概念即当日热点题材
    try:
        from quant_system.data_store import DataStore as _DS
        _ds = _DS()
        cm = _ds.get_dataset("concept_member")
        _col = "涨跌幅" if spot is not None and "涨跌幅" in spot.columns else ("pct_chg" if spot is not None and "pct_chg" in spot.columns else None)
        _code_col = "代码" if spot is not None and "代码" in spot.columns else ("code" if spot is not None and "code" in spot.columns else None)
        if cm is not None and not cm.empty and spot is not None and not spot.empty \
                and _code_col and _col:
            ret = pd.to_numeric(spot[_col], errors="coerce")
            limit_codes = set(spot.loc[ret >= 9.8, _code_col].astype(str).str.replace(r"^(sh|sz|bj)", "", regex=True).str.zfill(6))
            if limit_codes:
                sub = cm[cm["code"].astype(str).str.zfill(6).isin(limit_codes)]
                if not sub.empty:
                    # BK 代码 → 中文名（concept_board 映射）
                    cname = {}
                    try:
                        cb = _ds.get_dataset("concept_board")
                        if cb is not None and not cb.empty and "board_code" in cb.columns and "board_name" in cb.columns:
                            cname = dict(zip(cb["board_code"].astype(str), cb["board_name"].astype(str)))
                    except Exception:
                        pass
                    cnt = sub["concept"].astype(str).value_counts().head(15)
                    # 排除通用风格标签（融资融券/沪深股通/指数成分等），只留真题材
                    _NOISE = {"融资融券", "沪股通", "深股通", "机构重仓", "基金重仓", "QFII重仓",
                              "标准普尔", "富时罗素", "MSCI中国", "创业板综", "上证50_", "沪深300_",
                              "中证500", "中证1000", "深成500", "昨日涨停", "昨日连板",
                              "昨日高振幅", "昨日较强", "昨日较弱", "昨日上榜", "昨日跌停",
                              "转融券标的", "注册制次新股", "小盘股", "中盘股", "大盘股", "微盘股"}
                    rows = []
                    for k, v in cnt.items():
                        nm = cname.get(k, k)
                        if nm in _NOISE or ("成份" in nm) or ("样本" in nm):
                            continue
                        rows.append({"concept": nm, "涨停家数": int(v)})
                    out["短线_热点题材(涨停概念)"] = {
                        "涨停股数": len(limit_codes),
                        "Top概念": rows[:8],
                    }
    except Exception:
        pass
    # 中线：行业强弱（真实接线 segment_industry）
    if ind_map is not None and spot is not None:
        try:
            ind = segment_industry(spot, ind_map)
            out["中线_行业轮动"] = ind if "error" not in ind else ind["error"]
        except Exception as e:
            out["中线_行业轮动"] = f"计算失败: {e}"
    else:
        out["中线_行业轮动"] = "需 sw_industry_map 关联（传入 ind_map）"
    # 长线：估值温度（真实接线 valuation_temperature）
    if valuation_files:
        try:
            out["长线_估值温度"] = valuation_temperature(valuation_files)
        except Exception as e:
            out["长线_估值温度"] = f"计算失败: {e}"
    else:
        out["长线_估值温度"] = "需 valuation 全市场 peTTM 中位数（传入 valuation_files）"
    return out


def segment_industry(spot: pd.DataFrame, ind_map: pd.DataFrame) -> dict[str, Any]:
    """行业分层：申万行业等权涨跌幅/上涨家数占比（中线视角）。"""
    if spot is None or spot.empty or ind_map is None or ind_map.empty:
        return {"meta": {"error": "need spot + industry map"}}
    if "代码" not in spot.columns or "代码" not in ind_map.columns:
        return {"meta": {"error": "missing 代码"}}
    df = spot.merge(ind_map[["代码", "industry"]], on="代码", how="left")
    df = df.dropna(subset=["industry"])
    if df.empty or "涨跌幅" not in df.columns:
        return {"meta": {"error": "no industry matched"}}
    df["涨跌幅"] = pd.to_numeric(df["涨跌幅"], errors="coerce")
    g = df.groupby("industry").agg(
        股票数=("代码", "count"),
        等权涨跌幅=("涨跌幅", "mean"),
        上涨占比=("涨跌幅", lambda s: (s > 0).mean() * 100),
    ).sort_values("等权涨跌幅", ascending=False)
    return {
        "行业数": int(len(g)),
        "Top5": g.head(5).round(2).to_dict("index"),
        "Bottom5": g.tail(5).round(2).to_dict("index"),
        "meta": {"note": "申万行业等权涨跌幅，未剔除ST"},
    }


def valuation_temperature(valuation_files: list, sample: int = 300) -> dict[str, Any]:
    """长线估值温度：全市场 peTTM/pbMRQ 中位数（抽样）。"""
    import random
    import os
    from pathlib import Path

    if not valuation_files:
        return {"error": "no valuation files"}
    random.seed(7)
    sample_files = random.sample(valuation_files, min(sample, len(valuation_files)))
    pes, pbs = [], []
    for p in sample_files:
        try:
            df = pd.read_parquet(p)
            if df.empty:
                continue
            last = df.iloc[-1]
            if "peTTM" in df.columns and pd.notna(last.get("peTTM")) and last["peTTM"] > 0:
                pes.append(float(last["peTTM"]))
            if "pbMRQ" in df.columns and pd.notna(last.get("pbMRQ")) and last["pbMRQ"] > 0:
                pbs.append(float(last["pbMRQ"]))
        except Exception:
            continue
    return {
        "样本数": len(pes),
        "peTTM中位数": round(float(pd.Series(pes).median()), 2) if pes else None,
        "pbMRQ中位数": round(float(pd.Series(pbs).median()), 2) if pbs else None,
        "meta": {"note": f"抽样 {len(pes)} 只（全市场约5200），长线估值温度参考"},
    }
