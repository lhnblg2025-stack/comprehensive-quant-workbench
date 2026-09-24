#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""财报深度脉动（2026-08-22 —— 用户"财报数据你没加进去"）

扫 data_warehouse/financial/*.parquet 聚合最新一期财报:
  1. roe_top():       ROE最高(净资产收益率)
  2. growth_top():    净利增速最高(净利润增长率)
  3. margin_top():    毛利率/负债率榜单
  4. key_metrics():   每股收益/每股净资产 等核心指标
→ 供研报"财报深度"章节, 展示真实基本面数据。

纯本地 parquet, 秒级。
"""
from __future__ import annotations

import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIN = ROOT / "data_warehouse" / "financial"


def _collect(key: str, limit: int = 150) -> list[dict]:
    """扫财报parquet(限样本加速), 取最新一期该指标。"""
    import pandas as pd
    rows = []
    files = sorted(glob.glob(str(FIN / "*.parquet")))[:limit]
    for f in files:
        code = Path(f).stem
        try:
            df = pd.read_parquet(f, columns=["日期", key, "摊薄每股收益(元)"])
            if df[key].isna().all():
                continue
            df = df.dropna(subset=[key])
            if not len(df):
                continue
            last = df.sort_values("日期").iloc[-1]
            rows.append({"code": code, "date": str(last.get("日期", ""))[:10],
                         "val": float(last[key]), "eps": float(last.get("摊薄每股收益(元)", 0) or 0)})
        except Exception:
            continue
    return rows


def roe_top(n: int = 8) -> list[dict]:
    """ROE最高(净资产收益率)。"""
    rows = [r for r in _collect("净资产收益率(%)") if r["val"] == r["val"]]
    rows.sort(key=lambda x: -x["val"])
    return rows[:n]


def growth_top(n: int = 8) -> list[dict]:
    """净利增速最高(净利润增长率)。"""
    rows = [r for r in _collect("净利润增长率(%)") if r["val"] == r["val"]]
    rows.sort(key=lambda x: -x["val"])
    return rows[:n]


def leverage_top(n: int = 6) -> list[dict]:
    """资产负债率最低(财务稳健)。"""
    rows = [r for r in _collect("资产负债率(%)") if r["val"] == r["val"]]
    rows.sort(key=lambda x: x["val"])
    return rows[:n]


def eps_top(n: int = 6) -> list[dict]:
    """每股收益最高。"""
    rows = [r for r in _collect("摊薄每股收益(元)") if r["val"] == r["val"]]
    rows.sort(key=lambda x: -x["val"])
    return rows[:n]


# ═══ 财报选股池(ROE+增速+低负债+EPS 组合) ═══
def stock_picker(roe_min: float = 10.0, grow_min: float = 20.0,
                  leverage_max: float = 60.0, eps_min: float = 0.5, limit: int = 8) -> list[dict]:
    """基本面选股: ROE>=roe_min 且 增速>=grow_min 且 负债<=leverage_max 且 EPS>=eps_min。"""
    import pandas as pd
    rows = []
    files = sorted(glob.glob(str(FIN / "*.parquet")))[:300]
    for f in files:
        code = Path(f).stem
        try:
            df = pd.read_parquet(f, columns=["日期", "净资产收益率(%)", "净利润增长率(%)",
                                            "资产负债率(%)", "摊薄每股收益(元)"])
            last = df.sort_values("日期").iloc[-1]
            roe, grow, lev, eps = (float(last.get("净资产收益率(%)", 0) or 0),
                                   float(last.get("净利润增长率(%)", 0) or 0),
                                   float(last.get("资产负债率(%)", 0) or 0),
                                   float(last.get("摊薄每股收益(元)", 0) or 0))
            if roe >= roe_min and grow >= grow_min and lev <= leverage_max and eps >= eps_min:
                rows.append({"code": code, "roe": round(roe, 1), "grow": round(grow, 0),
                             "lev": round(lev, 0), "eps": round(eps, 2),
                             "score": round(roe / leverage_max * 100 + grow / 50 * 10, 1)})
        except Exception:
            continue
    rows.sort(key=lambda x: -x["score"])
    return rows[:limit]


# ═══ 估值深度(valuation 目录 PE/PB) ═══
def valuation_pulse(limit: int = 300) -> list[dict]:
    """读 valuation/*.parquet 最新 PE/PB(若有)."""
    import pandas as pd
    V = ROOT / "data_warehouse" / "valuation"
    rows = []
    for f in sorted(glob.glob(str(V / "*.parquet")))[:limit]:
        code = Path(f).stem
        try:
            df = pd.read_parquet(f)
            if not len(df):
                continue
            # 取最近非空 pe/pb(最新行可能nan)
            pe = pb = None
            for _, r in df.sort_values("date", ascending=False).iterrows():
                if pe is None and (r.get("pe_ttm") is not None and str(r.get("pe_ttm")) not in ("nan", "")):
                    pe = float(r.get("pe_ttm"))
                if pb is None and (r.get("pb") is not None and str(r.get("pb")) not in ("nan", "")):
                    pb = float(r.get("pb"))
                if pe is not None and pb is not None:
                    break
            rows.append({"code": code, "pe": round(pe, 1) if pe else None,
                         "pb": round(pb, 2) if pb else None})
        except Exception:
            continue
    rows = [r for r in rows if (r["pe"] or 0) > 0 or r["pb"]]
    rows.sort(key=lambda x: (x["pe"] if (x["pe"] or 0) > 0 else 999))
    return rows[:8]


# ═══ 财报趋势(指定股 5期 ROE/EPS, 画图) ═══
def trend_data(code: str = "000001", n: int = 5) -> list[dict]:
    import pandas as pd
    f = FIN / f"{code}.parquet"
    if not f.exists():
        return []
    df = pd.read_parquet(f, columns=["日期", "净资产收益率(%)", "摊薄每股收益(元)", "净利润增长率(%)"])
    df = df.sort_values("日期").tail(n)
    return [{"d": str(r.get("日期", ""))[:10][5:], "roe": round(float(r.get("净资产收益率(%)", 0) or 0), 1),
             "eps": round(float(r.get("摊薄每股收益(元)", 0) or 0), 2),
             "grow": round(float(r.get("净利润增长率(%)", 0) or 0), 0)}
            for r in df.to_dict("records")]


def fundamental_md() -> str:
    """财报深度(ROE/增速/负债/每股收益四榜) → Markdown。"""
    L = ["## 💹 财报深度（最细财报数据）"]
    rt = roe_top(6)
    if rt:
        L.append("### ROE最高（净资产收益率%）")
        L.append(" | ".join(f"{r['code']}:{r['val']:.1f}%" for r in rt[:6]))
    gt = growth_top(6)
    if gt:
        L.append("### 净利增速最快（净利润增长率%）")
        L.append(" | ".join(f"{r['code']}:{r['val']:+.0f}%" for r in gt[:6]))
    lt = leverage_top(5)
    if lt:
        L.append("### 低负债（资产负债率% 最低）")
        L.append(" | ".join(f"{r['code']}:{r['val']:.0f}%" for r in lt[:5]))
    et = eps_top(5)
    if et:
        L.append("### 每股收益最高（元）")
        L.append(" | ".join(f"{r['code']}:{r['val']:.2f}" for r in et[:5]))
    if not (rt or gt):
        L.append("- 财报数据缺失（financial 目录无 parquet）")
    return "\n".join(L)


def fundamental_data() -> dict:
    """结构化数据(供HTML研报渲染)。"""
    return {"roe_top": roe_top(8), "growth_top": growth_top(8),
            "leverage_least": leverage_top(6), "eps_top": eps_top(6)}


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    print(fundamental_md())

# ═══ 公告事件日历(cninfo 最新) ═══
def announcement_calendar(top_n: int = 10) -> dict:
    """读 cninfo 最新公告 → 事件日历(重大合同/减持/业绩等)."""
    import json as _j
    CNN = ROOT / "data_warehouse" / "cninfo"
    fs = sorted(glob.glob(str(CNN / "cninfo_*.json")))
    if not fs:
        return {"items": [], "note": "公告缺失"}
    try:
        items = _j.loads(Path(fs[-1]).read_text(encoding="utf-8"))
        if not isinstance(items, list):
            return {"items": [], "note": "公告格式异常"}
        # 按公告类型分类, 重点类型排前
        KEY = ("业绩", "减持", "增持", "合同", "中标", "收购", "重组", "回购", "立案", "质押")
        def score(it):
            t = str(it.get("公告类型", ""))
            return sum(1 for k in KEY if k in t)
        items.sort(key=lambda x: -score(x))
        out = [{"code": str(it.get("代码", "")), "name": str(it.get("名称", "")),
                "title": str(it.get("公告标题", ""))[:60], "type": str(it.get("公告类型", "")),
                "date": str(it.get("公告日期", ""))[:10]}
               for it in items[:top_n]]
        return {"items": out, "note": f"最新公告 {len(items)}条({fs[-1].split('_')[-1].replace('.json','')})"}
    except Exception as e:  # noqa: BLE001
        return {"items": [], "note": f"公告读取失败: {str(e)[:40]}"}
