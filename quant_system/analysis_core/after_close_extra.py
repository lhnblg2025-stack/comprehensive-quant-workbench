#!/usr/bin/env python3
"""
after_close_extra.py — 盘后增强：龙虎榜短线 + 中长线低估池 (2026-08-14)
=====================================================================
用户硬诉求: 盘后必须包含 产业链/龙虎榜等短线内容 + 中长线低估机会提醒。
产业链已由 battle_map「❹ 产业链联动」覆盖（chain_map 先验 242 条），
本模块补齐另外两块，供 battle_map 渲染两章:

  ❺ 龙虎榜资金（短线）: 最近交易日游资营业部净买 Top + 个股龙虎榜净买 Top
  ❻ 中长线低估池（价值）: 大中盘低 PE/PB + ROE/增长 + 位置安全的低估名单

数据:
  - data_warehouse/market/lhb_hyyyb_em.parquet  活跃营业部(游资席位)
  - data_warehouse/market/lhb_2026*.parquet     龙虎榜日汇总(净买额/上榜后表现)
  - data_warehouse/valuation/*.parquet          peTTM/pbMRQ
  - data_warehouse/financial/*.parquet          ROE/增长
  - watchlist.fetch_quotes                      实时报价(含52周高低/市值)

用法:
  python3 -m quant_system.analysis_core.after_close_extra [--date 2026-08-13]
  (pipeline --daily 已集成; 产物 generated/after_close_extra_{date}.json/.md)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

CST = timezone(timedelta(hours=8))
MARKET_DIR = ROOT / "data_warehouse" / "market"
VALUATION_DIR = ROOT / "data_warehouse" / "valuation"
FINANCIAL_DIR = ROOT / "data_warehouse" / "financial"
OUT_JSON = ROOT / "generated"
# 中长线低估过滤阈值（集中声明）
VALUE_PE_MAX = 20.0
VALUE_PB_MAX = 1.6
VALUE_ROE_MIN = 8.0
VALUE_52W_DIST_MAX = 0.45   # 距52周低 <45%（位置不追高）
VALUE_TOP_N = 10


def _today() -> str:
    return datetime.now(CST).strftime("%Y-%m-%d")


def _safe_float(v) -> float:
    try:
        if v is None:
            return float("nan")
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def lhb_short_term(date: str | None = None) -> dict:
    """龙虎榜短线: 游资营业部净买 Top + 个股龙虎榜净买 Top（最近上榜日）。"""
    date = date or _today()
    out = {"date": date, "broker_top": [], "stock_top": [], "error": None}
    try:
        # ── 1. 游资营业部（hyyyb）: 选最近上榜日 ──
        f = MARKET_DIR / "lhb_hyyyb_em.parquet"
        if f.exists():
            df = pd.read_parquet(f)
            if "上榜日" in df.columns:
                df["上榜日"] = pd.to_datetime(df["上榜日"])
                # 审计 2026-08-16：历史 date 必须取 <=date 的最近上榜日，禁止读未来
                hist_df = df[df["上榜日"] <= pd.to_datetime(date)]
                if hist_df.empty:
                    hist_df = df
                day = hist_df["上榜日"].max()
                recent = hist_df[hist_df["上榜日"] == day].copy()
                if "买入总金额" in recent.columns:
                    # 2026-08-14: 营业部按买入金额排序=游资活跃席位(当日可能整体净卖,
                    # 净买额排序无意义), net 保留总买卖净额供标注方向
                    recent = recent.sort_values("买入总金额", ascending=False).head(5)
                    for _, r in recent.iterrows():
                        out["broker_top"].append({
                            "name": str(r.get("营业部名称", "")),
                            "buy": round(_safe_float(r.get("买入总金额")) / 1e8, 2),
                            "net": round(_safe_float(r.get("总买卖净额")) / 1e8, 2),
                            "stocks": str(r.get("买入股票", ""))[:40],
                            "day": day.strftime("%Y-%m-%d"),
                        })
        # ── 2. 个股龙虎榜净买（日汇总段文件, 最近上榜日）──
        files = sorted(MARKET_DIR.glob("lhb_20*.parquet"))
        day_df = None
        for fp in files:
            if "hyyyb" in fp.name or "jgmmtj" in fp.name or "ggtj" in fp.name:
                continue
            try:
                d = pd.read_parquet(fp)
            except Exception:  # noqa: BLE001
                continue
            if "上榜日" not in d.columns:
                continue
            d["上榜日"] = pd.to_datetime(d["上榜日"])
            if day_df is None or d["上榜日"].max() > day_df["上榜日"].max():
                day_df = d
        if day_df is not None:
            day_df = day_df[day_df["上榜日"] <= pd.to_datetime(date)]  # 审计 2026-08-16：历史不读未来龙虎榜
            if day_df.empty:
                day_df = None
        if day_df is not None:
            day = day_df["上榜日"].max()
            recent = day_df[day_df["上榜日"] == day].copy()
            if "龙虎榜净买额" in recent.columns:
                recent = recent.sort_values("龙虎榜净买额", ascending=False).head(8)
                for _, r in recent.iterrows():
                    out["stock_top"].append({
                        "code": str(r.get("代码", "")).zfill(6),
                        "name": str(r.get("名称", "")),
                        "net": round(_safe_float(r.get("龙虎榜净买额")) / 1e8, 2),
                        "pct": round(_safe_float(r.get("涨跌幅")), 2),
                        "reason": str(r.get("上榜原因", ""))[:30],
                        "fwd1": round(_safe_float(r.get("上榜后1日")), 2),
                        "day": day.strftime("%Y-%m-%d"),
                    })
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e)[:120]
    return out


def _hist_value_at(sym: str, date: str) -> dict | None:
    """历史时点估值快照（V12.3 审计 P1-3: 消除 value_picks 实时价前视）。

    从当日 valuation parquet（peTTM/pbMRQ）+ 当日 K 线（close/52周低/流通市值）
    取 time-consistent 数据; 任一块在 date 无当日记录 → None（降级跳过, 不退回实时价）。
    """
    vp = VALUATION_DIR / f"{sym}.parquet"
    if not vp.exists():
        return None
    try:
        v = pd.read_parquet(vp)
        v["date"] = pd.to_datetime(v["date"])
        row = v[v["date"].astype(str) <= date]
        if row.empty:
            return None
        r = row.iloc[-1]
        pe = float(r["peTTM"]) if not pd.isna(r.get("peTTM")) else 0.0
        pb = float(r["pbMRQ"]) if not pd.isna(r.get("pbMRQ")) else 0.0
        kp = ROOT / "data_warehouse" / "kline" / f"{sym}.parquet"
        out = {"pe": pe, "pb": pb, "price": 0.0, "low52": 0.0, "cap": 0.0}
        if kp.exists():
            k = pd.read_parquet(kp)
            k["date"] = pd.to_datetime(k["date"])
            kh = k[k["date"].astype(str) <= date]
            if kh.empty:
                return None
            out["price"] = float(kh.iloc[-1]["close"])
            out["low52"] = float(kh["low"].tail(250).min())
            # 流通市值(元) ≈ close × 流通股本(股)
            sh = kh.iloc[-1].get("outstanding_share")
            if pd.notna(sh) and sh and sh > 0 and out["price"] > 0:
                out["cap"] = out["price"] * float(sh) / 1e8  # 亿元
        return out
    except Exception:  # noqa: BLE001
        return None


def value_picks(date: str | None = None, top_n: int = VALUE_TOP_N) -> list[dict]:
    """中长线低估池: 大中盘低 PE/PB + ROE/增长 + 位置安全, 打分排序。

    性能: 409 只大中盘池(不扫全市场), 财务/估值逐文件读(单文件~10ms, 总计~10s)。
    V12.3 审计 P1-3: 历史 --date 用当日估值/K线快照（不引入实时价前视）;
    仅 date == 今天 且盘后时用实时 fetch_quotes。
    """
    date = date or _today()
    today = _today()
    historical = date < today
    rows: list[dict] = []
    try:
        from quant_system.watchlist import get_watchlist, fetch_quotes
        stocks = get_watchlist()
        syms = [s["symbol"] for s in stocks]
        quotes = {} if historical else {q["symbol"]: q for q in fetch_quotes(syms)}
        for st in stocks:
            sym = st["symbol"]
            if historical:
                h = _hist_value_at(sym, date)
                if h is None:
                    continue  # 历史快照不可得 → 降级跳过, 绝不用实时价冒充
                pe, pb = h["pe"], h["pb"]
                price, low52, cap = h["price"], h["low52"], h["cap"]
                name = st.get("name", "")
            else:
                q = quotes.get(sym) or {}
                pe = _safe_float(q.get("pe_ttm", 0))
                pb = _safe_float(q.get("pb", 0))
                price = _safe_float(q.get("price", 0))
                low52 = _safe_float(q.get("low_52w", 0))
                cap = _safe_float(q.get("market_cap_yi", 0))
                name = q.get("name", st.get("name", ""))
            if pe <= 0 or pb <= 0 or price <= 0:
                continue
            if pe > VALUE_PE_MAX or pb > VALUE_PB_MAX:
                continue
            if cap and cap < 100:  # 大中盘口径: ≥100亿
                continue
            dist52 = (price / low52 - 1) if low52 and low52 > 0 else 9.9
            if dist52 > VALUE_52W_DIST_MAX:
                continue
            # 财务加分: ROE / 净利增长（读最新一期）
            # 2026-08-14: ①优先加权ROE(年化)②单季口径下取近4期求和近似年化,
            # 避免银行Q1单季2-3%被误读为全年低ROE
            roe = growth = float("nan")
            try:
                fp = FINANCIAL_DIR / f"{sym}.parquet"
                if fp.exists():
                    fd = pd.read_parquet(fp)
                    roe_col = ("加权净资产收益率(%)" if "加权净资产收益率(%)" in fd.columns
                               else "净资产收益率(%)")
                    if roe_col in fd.columns:
                        vals = fd[roe_col].dropna().tail(4)
                        if len(vals):
                            roe = float(vals.sum())
                    if "净利润增长率(%)" in fd.columns:
                        g = fd.dropna(subset=["净利润增长率(%)"])
                        if len(g):
                            growth = _safe_float(g.iloc[-1]["净利润增长率(%)"])
            except Exception:  # noqa: BLE001
                pass
            # 打分: 低PE 40% + 低PB 30% + ROE 20% + 增长 10%
            score = 0.0
            score += (1 - pe / VALUE_PE_MAX) * 40
            score += (1 - pb / VALUE_PB_MAX) * 30
            if roe == roe:  # not NaN
                score += min(20, max(0, (roe - VALUE_ROE_MIN) / 10 * 20))
            if growth == growth:
                score += min(10, max(0, growth / 20 * 10))
            rows.append({
                "code": sym, "name": name,
                "price": round(price, 2), "pe": round(pe, 1), "pb": round(pb, 2),
                "roe": round(roe, 1) if roe == roe else None,
                "growth": round(growth, 1) if growth == growth else None,
                "dist52w": round(dist52 * 100, 1),
                "cap": round(cap) if cap == cap else None,
                "score": round(score, 1),
            })
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).error(f"[after_close_extra] 低估扫描失败: {e}", exc_info=True)
    rows.sort(key=lambda r: -r["score"])
    return rows[:top_n]


def build_extra(date: str | None = None) -> dict:
    """组装盘后增强产物 → generated/after_close_extra_{date}.json + .md。"""
    date = date or _today()
    extra = {
        "date": date,
        "lhb": lhb_short_term(date),
        "value_picks": value_picks(date),
        "built_at": datetime.now(CST).isoformat(timespec="seconds"),
    }
    OUT_JSON.mkdir(parents=True, exist_ok=True)
    (OUT_JSON / f"after_close_extra_{date}.json").write_text(
        json.dumps(extra, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT_JSON / f"after_close_extra_{date}.md").write_text(
        render_extra_md(extra), encoding="utf-8")
    return extra


def render_extra_md(extra: dict) -> str:
    """渲染两章 Markdown（battle_map 引用）。"""
    date = extra.get("date", "")
    lines = [f"# 📈 盘后增强 {date}", ""]
    # ❺ 龙虎榜
    lhb = extra.get("lhb", {}) or {}
    lines.append("## ❺ 龙虎榜资金（短线）")
    if lhb.get("error"):
        lines.append(f"- ⚠️ 龙虎榜数据不可用: {lhb['error']}")
    bt = lhb.get("broker_top", [])
    if bt:
        lines.append(f"**游资营业部净买 Top**（{bt[0].get('day', '')}）:")
        for b in bt:
            net_tag = f"净{b['net']:+.2f}亿" if b.get("net") is not None else ""
            lines.append(f"- 🏦 {b['name']} 买入 {b.get('buy', 0):.2f}亿 {net_tag} | 涉及: {b['stocks']}")
    st = lhb.get("stock_top", [])
    if st:
        lines.append(f"**个股龙虎榜净买 Top**（{st[0].get('day', '')}）:")
        for s in st:
            fwd = f" | 后1日 {s['fwd1']:+.1f}%" if s.get("fwd1") is not None and s["fwd1"] == s["fwd1"] else ""
            lines.append(f"- 💰 {s['name']}({s['code']}) 净买 {s['net']:.2f}亿 "
                         f"({s['pct']:+.1f}%) {s['reason']}{fwd}")
    if not bt and not st:
        lines.append("- 无龙虎榜数据（非交易日或数据缺失）")
    # ❻ 中长线低估
    lines += ["", "## ❻ 中长线低估池（价值提醒）"]
    vp = extra.get("value_picks", [])
    if vp:
        lines.append(f"**大中盘低估值名单**（低PE/PB + 位置安全, 打分排序）:")
        for v in vp:
            roe = f"ROE {v['roe']:.0f}%" if v.get("roe") is not None else "ROE -"
            gr = f"增 {v['growth']:.0f}%" if v.get("growth") is not None else ""
            lines.append(f"- 💎 {v['name']}({v['code']}) PE {v['pe']:.1f} PB {v['pb']:.2f} "
                         f"{roe} {gr} | 距52周低+{v['dist52w']:.0f}% | 打分 {v['score']}")
    else:
        lines.append("- 当前无符合低估条件的标的（数据缺失或估值未达阈值）")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="盘后增强: 龙虎榜短线+中长线低估池")
    ap.add_argument("--date", default=None, help="日期 YYYY-MM-DD（默认今天）")
    args = ap.parse_args()
    extra = build_extra(args.date)
    print(render_extra_md(extra))
    print(f"\n[after_close_extra] 已生成 "
          f"generated/after_close_extra_{extra['date']}.json/.md")
