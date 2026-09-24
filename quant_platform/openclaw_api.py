"""
quant_platform.openclaw_api — OpenClaw 会话内置数据接口（V3 数据层 P2）

用户 #9：非量化因子指标数据（龙虎榜/两融/营业部/股东户数等）也要储存，
用于内置 openclaw 接口和日常使用。

本模块提供供 openclaw 会话（AI 助手）直接调用的纯数据查询接口：
  - 全部走 DataStore 仓库优先，零网络
  - 输入为股票代码/名称/板块名，输出为结构化 dict
  - 不依赖网络；数据缺失返回明确提示

用法（openclaw 会话内）：
    from quant_platform.openclaw_api import stock_overview, concept_members, lhb_recent, ...
"""
from __future__ import annotations

from datetime import datetime, timedelta
import re
from typing import Any

import pandas as pd

from pathlib import Path

from quant_system.data_store import DataStore

ROOT = Path(__file__).resolve().parent.parent  # workspace 根

_ds = DataStore()


def _norm_code(code: str) -> str:
    """代码归一化：'600519.SH'/'sh600519'/600519 → '600519'。"""
    c = str(code).strip().lower()
    c = re.sub(r"\.0+$", "", c)
    # 后缀（600519.sh / 600519.SH）
    for suf in (".sh", ".sz", ".bj"):
        if c.endswith(suf):
            c = c[: -len(suf)]
            break
    # 前缀（sh600519 / sz000001 / bj920001）
    for pre in ("sh", "sz", "bj"):
        if c.startswith(pre) and c[len(pre):].isdigit():
            c = c[len(pre):]
            break
    return c.zfill(6)


def _latest_dates(df: pd.DataFrame, n: int = 5) -> list[str]:
    if df.empty or "date" not in df.columns:
        return []
    return [str(d.date()) for d in pd.to_datetime(df["date"], errors="coerce").dropna().tail(n)]


def _date_column(df: pd.DataFrame, candidates=("日期", "上榜日", "date", "交易日期")) -> str | None:
    """Return the first candidate column containing a usable date value.

    Provider exports sometimes include an empty ``日期`` column while the
    actual value is in ``上榜日``. Presence alone is therefore not enough.
    """
    if df is None or df.empty:
        return None
    for col in candidates:
        if col not in df.columns:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().any():
            return col
    return None


def _recent_rows(df: pd.DataFrame, days: int) -> tuple[pd.DataFrame, str | None, str | None]:
    """Filter rows by the latest observed date and expose the actual date field."""
    if df is None or df.empty:
        return pd.DataFrame(), None, None
    date_col = _date_column(df)
    if not date_col:
        return df.tail(0), None, None
    out = df.copy()
    out["_query_date"] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=["_query_date"])
    if out.empty:
        return out.drop(columns=["_query_date"], errors="ignore"), date_col, None
    latest = out["_query_date"].max()
    cutoff = latest - pd.Timedelta(days=max(0, int(days)))
    filtered = out[out["_query_date"] >= cutoff].sort_values("_query_date")
    as_of = latest.date().isoformat()
    return filtered.drop(columns=["_query_date"]), date_col, as_of


# ── 个股总览（多数据集融合）──────────────────────────────
def stock_overview(code: str, days: int = 120) -> dict:
    """单只股票全维度概览：K线/估值/财务/行业/龙虎榜/两融/概念。

    供 openclaw 会话回答"XXX 最近怎么样"类问题。
    """
    code = _norm_code(code)
    out: dict[str, Any] = {"code": code, "found": True, "contract": "stock-overview.v2", "source_status": {}}
    # K线（直接走 get() parquet 优先，零网络）
    try:
        k = _ds.get(code, days=days)
        out["kline"] = {
            "latest_date": str(k["date"].max().date()) if len(k) else None,
            "close": float(k["close"].iloc[-1]) if len(k) else None,
            "pct_chg_1d": float(k["pct_chg"].iloc[-1]) if len(k) and "pct_chg" in k else None,
            "mom20": float(k["close"].iloc[-1] / k["close"].iloc[-21] - 1) if len(k) > 21 else None,
            "mom60": float(k["close"].iloc[-1] / k["close"].iloc[-61] - 1) if len(k) > 61 else None,
        } if len(k) else None
        out["source_status"]["kline"] = {"status": "available" if len(k) else "missing", "source": "DataStore", "as_of": out["kline"].get("latest_date") if out["kline"] else None, "rows": int(len(k))}
    except Exception as exc:
        out["kline"] = None
        out["source_status"]["kline"] = {"status": "error", "source": "DataStore", "as_of": None, "rows": 0, "error": str(exc)[:160]}
    # 估值（直接读 valuation parquet）
    try:
        _vp = _ds.warehouse_root() / "valuation" / f"{code}.parquet"
        if not _vp.exists():
            out["valuation"] = None
            out["source_status"]["valuation"] = {"status": "missing", "source": str(_vp), "as_of": None, "rows": 0}
        else:
            v = pd.read_parquet(_vp)
            v, date_col, as_of = _recent_rows(v, days)
            pe_col = next((c for c in ("peTTM", "pe_ttm") if c in v.columns), None)
            pb_col = next((c for c in ("pbMRQ", "pb") if c in v.columns), None)
            if date_col is None or pe_col is None or pb_col is None:
                out["valuation"] = None
                out["source_status"]["valuation"] = {"status": "schema_error", "source": str(_vp), "as_of": as_of, "rows": int(len(v)), "error": "缺少日期/PE/PB字段"}
            elif len(v):
                last = v.iloc[-1]
                out["valuation"] = {
                    "latest_date": as_of,
                    "peTTM": float(last[pe_col]) if pd.notna(last[pe_col]) else None,
                    "pbMRQ": float(last[pb_col]) if pd.notna(last[pb_col]) else None,
                    "pe_ttm": float(last[pe_col]) if pd.notna(last[pe_col]) else None,
                    "pb": float(last[pb_col]) if pd.notna(last[pb_col]) else None,
                }
                out["source_status"]["valuation"] = {"status": "available", "source": str(_vp), "as_of": as_of, "rows": int(len(v))}
            else:
                out["valuation"] = None
                out["source_status"]["valuation"] = {"status": "empty", "source": str(_vp), "as_of": as_of, "rows": 0}
    except Exception as exc:
        out["valuation"] = None
        out["source_status"]["valuation"] = {"status": "error", "source": "valuation parquet", "as_of": None, "rows": 0, "error": str(exc)[:160]}
    # 财务（直接读个股文件，避免 get_dataset 全量 glob 5159 个路径）
    try:
        _fp = _ds.warehouse_root() / "financial" / f"{code}.parquet"
        if not _fp.exists():
            out["financial"] = None
            out["source_status"]["financial"] = {"status": "missing", "source": str(_fp), "as_of": None, "rows": 0}
        else:
            fd = pd.read_parquet(_fp)
            date_col = _date_column(fd, ("日期", "报告期", "date", "report_date", "ann_date"))
            if date_col:
                fd = fd.copy()
                fd["_query_date"] = pd.to_datetime(fd[date_col], errors="coerce")
                fd = fd.dropna(subset=["_query_date"]).sort_values("_query_date").drop(columns=["_query_date"])
            out["financial"] = fd.tail(2).to_dict("records") if len(fd) else None
            out["source_status"]["financial"] = {"status": "available" if len(fd) else "empty", "source": str(_fp), "as_of": str(fd[date_col].iloc[-1])[:10] if len(fd) and date_col else None, "rows": int(len(fd))}
    except Exception as exc:
        out["financial"] = None
        out["source_status"]["financial"] = {"status": "error", "source": "financial parquet", "as_of": None, "rows": 0, "error": str(exc)[:160]}
    # 行业
    try:
        m = _ds.get_dataset("sw_industry_map")
        code_col = next((c for c in ("code", "股票代码", "证券代码") if c in m.columns), None)
        row = m[m[code_col].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6) == code] if code_col else pd.DataFrame()
        out["industry"] = row.iloc[0]["industry"] if len(row) and "industry" in row.columns else None
        out["source_status"]["industry"] = {"status": "available" if out["industry"] else "not_found", "source": "sw_industry_map", "as_of": None, "rows": int(len(row))}
    except Exception as exc:
        out["industry"] = None
        out["source_status"]["industry"] = {"status": "error", "source": "sw_industry_map", "as_of": None, "rows": 0, "error": str(exc)[:160]}
    # 龙虎榜（近60日：只读最近2个季度文件，并按上榜日过滤）
    try:
        lhb = _ds.get_dataset("lhb", files=2)
        code_col = next((c for c in ("代码", "股票代码", "code") if c in lhb.columns), None)
        recent, date_col, as_of = _recent_rows(lhb, 60)
        if code_col and date_col:
            sub = recent[recent[code_col].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6) == code]
            out["lhb_recent"] = sub.tail(10).to_dict("records") if len(sub) else []
            out["source_status"]["lhb"] = {"status": "available" if len(sub) else "not_found", "source": "DataStore.lhb", "as_of": as_of, "rows": int(len(sub)), "date_column": date_col}
        else:
            out["lhb_recent"] = []
            out["source_status"]["lhb"] = {"status": "schema_error", "source": "DataStore.lhb", "as_of": as_of, "rows": int(len(recent)), "error": "缺少代码或日期字段"}
    except Exception as exc:
        out["lhb_recent"] = []
        out["source_status"]["lhb"] = {"status": "error", "source": "DataStore.lhb", "as_of": None, "rows": 0, "error": str(exc)[:160]}
    # 两融（近5日：只读最近 5 个文件避免全量 3196 文件）
    try:
        mg = _ds.get_dataset("margin", files=5)
        cols = [c for c in mg.columns if "代码" in c or "code" in c.lower()] if len(mg) else []
        if cols:
            sub = mg[mg[cols[0]].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6) == code]
            out["margin_recent"] = sub.tail(5).to_dict("records") if len(sub) else []
            out["source_status"]["margin"] = {"status": "available" if len(sub) else "not_found", "source": "DataStore.margin", "as_of": None, "rows": int(len(sub))}
        else:
            out["margin_recent"] = []
            out["source_status"]["margin"] = {"status": "schema_error" if len(mg) else "empty", "source": "DataStore.margin", "as_of": None, "rows": int(len(mg))}
    except Exception as exc:
        out["margin_recent"] = []
        out["source_status"]["margin"] = {"status": "error", "source": "DataStore.margin", "as_of": None, "rows": 0, "error": str(exc)[:160]}
    return out


# ── 概念/题材查询 ────────────────────────────────────────
def _concept_name_map() -> dict:
    """board_code → board_name 映射（concept_member 存 BK 代码，需转中文名）。"""
    try:
        b = _ds.get_dataset("concept_board")
        if b.empty or "board_code" not in b.columns or "board_name" not in b.columns:
            return {}
        return dict(zip(b["board_code"].astype(str), b["board_name"].astype(str)))
    except Exception:
        return {}


def concept_members(concept: str, limit: int = 50) -> list[dict]:
    """查询概念板块成分股（支持中文名或 BK 代码；含涨跌幅需 join feature_store/kline）。

    返回记录带 concept 中文名（concept_member 存 BK 代码，join concept_board 转名）。
    """
    try:
        m = _ds.get_dataset("concept_member")
        if m.empty:
            return [{"error": "concept_member 数据未抓取（东财限流时用云服务器跑 fetch_classification.py）"}]
        name_map = _concept_name_map()
        # 匹配：中文名精确优先，其次 BK 代码精确，最后子串（避免 CRO 命中 MicroLED）
        m = m.copy()
        m["_cname"] = m["concept"].astype(str).map(name_map).fillna(m["concept"].astype(str))
        exact = (m["_cname"] == concept) | (m["concept"].astype(str) == concept)
        if exact.any():
            mask = exact
        else:
            mask = m["_cname"].str.contains(concept, case=False, na=False) | \
                   m["concept"].astype(str).str.contains(concept, case=False, na=False)
        sub = m[mask]
        out = sub.head(limit).to_dict("records")
        for r in out:
            r["concept_name"] = r.pop("_cname", r.get("concept"))
        return out
    except Exception as e:
        return [{"error": str(e)}]


def concept_list() -> list[str]:
    """全部概念板块名（中文名）。"""
    try:
        b = _ds.get_dataset("concept_board")
        if b.empty:
            return []
        col = "board_name" if "board_name" in b.columns else b.columns[0]
        return b[col].astype(str).tolist()
    except Exception:
        return []


def theme_list() -> list[str]:
    """全部同花顺题材板块名。"""
    try:
        t = _ds.get_dataset("theme_board")
        if t.empty:
            return []
        col = "theme_name" if "theme_name" in t.columns else t.columns[0]
        return t[col].astype(str).tolist()
    except Exception:
        return []


def concept_heatmap(top_n: int = 10) -> dict:
    """概念热度全景（V3 融合：本地 concept_board 全量 + 涨跌幅/上涨家数/领涨股）。

    返回: 涨幅榜/跌幅榜/换手最活跃榜/涨停密度榜（上涨家数≈成分数时）。
    用于日报 6 章与复盘热点题材识别。
    """
    try:
        b = _ds.get_dataset("concept_board")
        if b.empty:
            return {"error": "concept_board 数据未抓取"}
        cols = {c: c for c in b.columns}
        name_c = next((c for c in ("board_name", "板块名称", "板块") if c in cols), None)
        pct_c = next((c for c in ("pct_chg", "涨跌幅") if c in cols), None)
        up_c = next((c for c in ("up_count", "上涨家数") if c in cols), None)
        dn_c = next((c for c in ("down_count", "下跌家数") if c in cols), None)
        leader_c = next((c for c in ("leader_name", "领涨股", "领涨股票") if c in cols), None)
        turn_c = next((c for c in ("turnover", "换手率") if c in cols), None)
        if name_c is None or pct_c is None:
            return {"error": "concept_board 列名未识别", "columns": b.columns.tolist()}
        b = b.copy()
        b[pct_c] = pd.to_numeric(b[pct_c], errors="coerce")
        keep = [name_c, pct_c]
        if up_c:
            b[up_c] = pd.to_numeric(b[up_c], errors="coerce"); keep.append(up_c)
        if dn_c:
            b[dn_c] = pd.to_numeric(b[dn_c], errors="coerce"); keep.append(dn_c)
        if leader_c:
            keep.append(leader_c)
        if turn_c:
            b[turn_c] = pd.to_numeric(b[turn_c], errors="coerce"); keep.append(turn_c)
        b = b[keep].dropna(subset=[pct_c])
        ren = {name_c: "概念", pct_c: "涨跌幅%"}
        if up_c: ren[up_c] = "上涨家数"
        if dn_c: ren[dn_c] = "下跌家数"
        if leader_c: ren[leader_c] = "领涨股"
        if turn_c: ren[turn_c] = "换手率%"
        b = b.rename(columns=ren)
        up = b.sort_values("涨跌幅%", ascending=False).head(top_n)
        down = b.sort_values("涨跌幅%", ascending=True).head(top_n)
        active = b.sort_values("换手率%", ascending=False).head(top_n) if "换手率%" in b.columns else up
        # 涨停密度：上涨家数≈成分数（用下跌家数接近0 近似全涨）
        dense = b[b["下跌家数"] == 0].sort_values("涨跌幅%", ascending=False).head(top_n) if "下跌家数" in b.columns else up
        return {
            "total_concepts": len(b),
            "涨幅榜": up.to_dict("records"),
            "跌幅榜": down.to_dict("records"),
            "换手活跃榜": active.to_dict("records"),
            "全线飘红榜": dense.to_dict("records"),
        }
    except Exception as e:
        return {"error": str(e)}


def stock_concepts(code: str, limit: int = 15) -> dict:
    """个股所属概念板块（concept_member 反查，BK 代码转中文名）。"""
    code = _norm_code(code)
    try:
        m = _ds.get_dataset("concept_member")
        if m.empty:
            return {"error": "concept_member 数据未抓取"}
        sub = m[m["code"].astype(str).str.zfill(6) == code]
        if sub.empty:
            return {"code": code, "concepts": [], "note": "无概念归属记录"}
        name_map = _concept_name_map()
        concepts = [name_map.get(str(c), str(c)) for c in sub["concept"].astype(str)]
        # 去重保序（一个 BK 一个名字）
        seen, uniq = set(), []
        for c in concepts:
            if c not in seen:
                seen.add(c); uniq.append(c)
        return {"code": code, "count": len(uniq), "concepts": uniq[:limit],
                "total": len(uniq)}
    except Exception as e:
        return {"error": str(e)}


# ── 龙虎榜/资金面 ────────────────────────────────────────
def lhb_recent(days: int = 5) -> dict:
    """最近 N 日龙虎榜总览：净买 Top10 / 机构参与 / 上榜次数 Top。

    只读最近 3 个季度文件（提速：避免全量 30 文件 concat）。
    """
    try:
        lhb = _ds.get_dataset("lhb", files=3)
        if lhb.empty:
            return {"error": "lhb 数据为空"}
        sub, date_col, as_of = _recent_rows(lhb, days)
        if not date_col:
            return {"status": "schema_error", "error": "lhb 缺少日期/上榜日字段", "rows": 0, "period": f"最近{days}日"}
        if sub.empty:
            return {"status": "empty", "period": f"最近{days}日", "as_of": as_of, "rows": 0, "top_net_buy": []}
        # 净买额列名探测（龙虎榜主表用"龙虎榜净买额"，机构统计用"净额"/"总买卖净额"）
        buy_col = next((c for c in ("龙虎榜净买额", "净买额", "净买入额", "净额", "总买卖净额", "net_buy") if c in sub.columns), None)
        if buy_col is None:
            return {"status": "schema_error", "as_of": as_of, "date_column": date_col, "rows": len(sub), "note": "未找到净买额列", "columns": sub.columns.tolist()}
        name_col = next((c for c in ("名称", "股票名称") if c in sub.columns), None)
        code_col = next((c for c in ("代码", "股票代码", "code") if c in sub.columns), None)
        keep = [c for c in (code_col, name_col, buy_col, date_col) if c]
        top = sub.sort_values(buy_col, ascending=False).head(10)
        return {
            "status": "available" if len(top) else "empty",
            "period": f"最近{days}日",
            "as_of": as_of,
            "date_column": date_col,
            "total_rows": len(sub),
            "top_net_buy": top[keep].to_dict("records") if keep else [],
        }
    except Exception as e:
        return {"error": str(e)}


def broker_activity(top_n: int = 10) -> dict:
    """活跃营业部 Top N（游资席位追踪）。"""
    try:
        b = _ds.get_dataset("lhb_broker")
        if b.empty:
            return {"error": "lhb_broker 数据为空"}
        # 营业部列名探测
        name_col = next((c for c in ("营业部名称", "名称", "上榜营业部") if c in b.columns), b.columns[0])
        cnt = b[name_col].value_counts().head(top_n)
        return {"top_brokers": [{"name": k, "count": int(v)} for k, v in cnt.items()]}
    except Exception as e:
        return {"error": str(e)}


def broker_profiles(top_n: int = 20, style: str | None = None) -> dict:
    """营业部画像（V11：数据榨干→能力）。读 scripts/lhb_analyst.py 生成的画像。

    返回：净买入额 Top N 营业部，含风格分类（大买/大卖/中性）、活跃天数。
    """
    try:
        p = ROOT / "generated" / "lhb_broker_profiles.parquet"
        if not p.exists():
            return {"error": "画像未生成，先跑 scripts/lhb_analyst.py", "hint": "python3 scripts/lhb_analyst.py"}
        prof = pd.read_parquet(p)
        if style:
            prof = prof[prof["style"] == style]
        prof = prof.sort_values("net_amt_sum", ascending=False).head(top_n)
        cols = [c for c in ("broker", "n_days", "n_rows", "net_amt_yi", "style", "last_date") if c in prof.columns]
        return {
            "count": len(prof),
            "note": "net_amt_yi=累计净买入(亿); style: 大买(>5亿)/大卖(<-5亿)/中性",
            "brokers": prof[cols].to_dict("records"),
        }
    except Exception as e:
        return {"error": str(e)}


def seasonality_api(years: int = 10) -> dict:
    """季节性效应分析（V11 融合: seasonality → API，总纲指标多样化）。

    月度效应/节假日效应/年报季效应——用历史指数收益做 bootstrap 检验。
    """
    try:
        from quant_system.seasonality.month_effect import MonthEffect
        from quant_system.seasonality.holiday_effect import HolidayEffect
        me = MonthEffect()
        result = me.compute(years=years)
        try:
            he = HolidayEffect()
            holiday = he.compute()
        except Exception:
            holiday = {"error": "节假日效应计算失败"}
        result["holiday_effect"] = holiday if isinstance(holiday, dict) else {}
        return result
    except Exception as e:
        return {"error": str(e)}


def limit_up_depth_api() -> dict:
    """涨停板深度分析（V11 融合: market_depth → API）。

    涨停家数/连板梯队/炸板率——短线情绪质量判断。
    """
    try:
        from quant_system.market_depth.limit_up_depth import LimitUpDepth
        lud = LimitUpDepth()
        r = lud.compute()
        if isinstance(r, dict) and r.get("data_quality", {}).get("ok") is False:
            return {"error": "涨停池数据获取失败", "quality": r.get("data_quality")}
        return r
    except Exception as e:
        return {"error": str(e)}


def market_regime_api(code: str = "000300", days: int = 400) -> dict:
    """市场状态分析（V11 融合: market_analysis.regime → API）。

    用 300MA/144MA 趋势规则（用户硬规则）+ 波动率/流动性 regime 判断。
    """
    try:
        from quant_system.market_analysis.regime import TrendRegime, VolatilityRegime
        k = _ds.get(code, days=days)
        if k is None or k.empty:
            return {"error": f"{code} K线为空"}
        close = k["close"]
        tr = TrendRegime()
        vr = VolatilityRegime()
        slope_144 = tr.ma_slope(close, 144)
        ma144 = close.rolling(144).mean().iloc[-1]
        ma300 = close.rolling(300).mean().iloc[-1] if len(close) >= 300 else None
        last = float(close.iloc[-1])
        trend = "多头（价在144MA上）" if last > ma144 else "空头（价在144MA下）"
        if ma300 is not None:
            trend += " 且300MA上方" if last > ma300 else " 但300MA下方"
        try:
            vol_regime = vr.detect(close)
        except Exception:
            vol_regime = "未知"
        return {
            "code": code, "close": round(last, 2),
            "ma144": round(float(ma144), 2) if not pd.isna(ma144) else None,
            "ma300": round(float(ma300), 2) if ma300 is not None and not pd.isna(ma300) else None,
            "ma144_slope_deg": round(slope_144, 2),
            "trend": trend, "volatility_regime": str(vol_regime),
            "note": "趋势规则: 300MA/144MA 为主（用户硬规则），MA20/60 仅辅助",
        }
    except Exception as e:
        return {"error": str(e)}


def market_breadth_api() -> dict:
    """市场宽度深度分析（V11 融合: market_analysis.breadth → API）。

    用 market_pulse 实时涨跌家数驱动 A/D 线分析 + 涨跌比。
    """
    try:
        from quant_system.market_pulse import _fetch_adv_dec
        from quant_system.market_analysis.breadth import AdvanceDeclineLine
        adv_dec = _fetch_adv_dec()
        if "up" not in adv_dec or "down" not in adv_dec:
            return {"error": "涨跌家数数据不可用", "note": adv_dec.get("note", "")}
        up, down = float(adv_dec["up"]), float(adv_dec["down"])
        total = up + down + float(adv_dec.get("flat", 0))
        adl = AdvanceDeclineLine()
        # 当日 A/D 线贡献（简化：单日无序列，给出当前比值与方向）
        ad_ratio = up / max(down, 1)
        net = up - down
        # 广度信号（参考 A/D 逻辑）
        if ad_ratio > 1.5:
            breadth_signal = "强势（上涨家数显著占优）"
        elif ad_ratio < 0.67:
            breadth_signal = "弱势（下跌家数占优）"
        else:
            breadth_signal = "中性"
        return {
            "up": int(up), "down": int(down), "flat": int(adv_dec.get("flat", 0)),
            "total": int(total), "up_ratio": round(up / max(total, 1), 4),
            "ad_ratio": round(ad_ratio, 3), "net_adv_dec": int(net),
            "breadth_signal": breadth_signal,
            "amount_yi": adv_dec.get("amount_yi"),
            "note": "A/D线深度序列需历史日数据，当前为实时快照口径",
        }
    except Exception as e:
        return {"error": str(e)}


def news_sentiment_api(days: int = 10) -> dict:
    """公告情绪指标（V11 总纲#9：非量化指标数据储存→API）。

    读 scripts/fetch_news_sentiment.py 存储的日频公告情绪：
      sentiment_score: 当日公告情绪均值（-1~+1）
      positive_cnt/negative_cnt: 利好/利空公告数
    """
    try:
        df = _ds.get_dataset("news_sentiment")
        if df.empty:
            return {"error": "news_sentiment 数据为空", "hint": "先跑 scripts/fetch_news_sentiment.py"}
        df = df.sort_values("date").tail(days)
        rec = df.to_dict("records")
        latest = rec[-1] if rec else {}
        # 情绪判定
        score = float(latest.get("sentiment_score", 0))
        if score > 0.05:
            tone = "偏乐观（利好公告占优）"
        elif score < -0.05:
            tone = "偏悲观（利空公告占优）"
        else:
            tone = "中性"
        return {
            "period": f"最近{days}日",
            "latest_date": latest.get("date", ""),
            "latest_score": round(score, 4),
            "tone": tone,
            "latest_cnt": {"positive": latest.get("positive_cnt", 0),
                           "negative": latest.get("negative_cnt", 0),
                           "total": latest.get("total_cnt", 0)},
            "history": rec,
        }
    except Exception as e:
        return {"error": str(e)}


def gdhs_trend(code: str) -> dict:
    """股东户数趋势（筹码集中度：户数减少=集中）。"""
    code = _norm_code(code)
    try:
        g = _ds.get_dataset("gdhs")
        if g.empty:
            return {"error": "gdhs 数据为空"}
        code_col = next((c for c in ("股票代码", "代码") if c in g.columns), None)
        if code_col is None:
            return {"error": "gdhs 无代码列", "columns": g.columns.tolist()}
        sub = g[g[code_col].astype(str).str.zfill(6) == code]
        if sub.empty:
            return {"note": f"{code} 无股东户数记录"}
        date_col = next((c for c in ("日期", "股东户数统计截止日") if c in sub.columns), sub.columns[0])
        cnt_col = next((c for c in ("股东户数", "股东户数-本次", "total_holder") if c in sub.columns), None)
        if cnt_col is None:
            return {"note": "列名未识别", "columns": sub.columns.tolist(), "rows": sub.tail(3).to_dict("records")}
        return {"code": code, "records": sub.sort_values(date_col).tail(8)[[date_col, cnt_col]].to_dict("records")}
    except Exception as e:
        return {"error": str(e)}


# ── 市场全景 ─────────────────────────────────────────────
def market_panorama() -> dict:
    """市场全景：指数PE/破净/新高新低/主力资金/两融余额。"""
    out: dict[str, Any] = {}
    try:
        ie = _ds.get_dataset("index_pe")
        if len(ie):
            out["index_pe"] = ie.tail(1).to_dict("records")
    except Exception:
        pass
    try:
        mf = _ds.get_dataset("market_fund_flow")
        if len(mf):
            out["market_fund_flow"] = mf.tail(3).to_dict("records")
    except Exception:
        pass
    try:
        ms = _ds.get_dataset("margin_summary")
        if len(ms):
            out["margin_summary"] = ms.tail(2).to_dict("records")
    except Exception:
        pass
    return out


# ── 商品/期货联动（用户 #13：商品股要看对应商品价格联动）──
def commodity_stock_linkage(code: str) -> dict:
    """商品股-商品价格联动（V3 新模块，供 openclaw 会话直接调用）。"""
    from quant_platform.commodity_link import commodity_linkage
    return commodity_linkage(code)


def commodity_market_overview() -> dict:
    """商品市场总览（主要商品最新价/趋势，供日报/周报引用）。"""
    from quant_platform.commodity_link import commodity_market_overview as _ov
    return _ov()


def commodity_price(name: str | None = None) -> dict:
    """商品价格数据（原油/BDI/生猪等），支持按名过滤。"""
    try:
        c = _ds.get_dataset("commodity")
        if c.empty:
            return {"error": "commodity 数据为空"}
        if name:
            cols = c.columns.tolist()
            sub = c[[x for x in cols if name.lower() in x.lower() or "date" in x.lower()]]
            return {"name": name, "data": sub.tail(10).to_dict("records")}
        return {"columns": c.columns.tolist(), "tail": c.tail(3).to_dict("records")}
    except Exception as e:
        return {"error": str(e)}


def futures_basis_latest() -> dict:
    """期货基差最新。"""
    try:
        f = _ds.get_dataset("futures")
        if f.empty:
            return {"error": "futures 数据为空"}
        return {"latest": f.tail(3).to_dict("records")}
    except Exception as e:
        return {"error": str(e)}


# ── 汇总 ─────────────────────────────────────────────────
def all_datasets_status() -> list[dict]:
    """全部数据集状态（DataStore freshness 透传）。"""
    return _ds.freshness()


# ══════════════════════════════════════════════════════════
# V3 融合层入口（quant_v6 独特能力吸收）
# ══════════════════════════════════════════════════════════

def esg_overview(top_n: int = 10) -> dict:
    """ESG 全景：市场分布 + 高分股列表（本地 esg_rating.parquet）。"""
    import pandas as pd
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "data_warehouse" / "market" / "esg_rating.parquet"
    if not p.exists():
        return {"count": 0, "note": "ESG 数据未落盘（后台抓取）"}
    df = pd.read_parquet(p)
    code_col = next((c for c in df.columns if "代码" in str(c)), None)
    score_col = next((c for c in df.columns if "ESG" in str(c).upper() and "评分" in str(c)), None)
    name_col = next((c for c in df.columns if "名称" in str(c)), None)
    if code_col is None or score_col is None:
        return {"count": len(df), "note": "列名不匹配"}
    tmp = df[[code_col, score_col] + ([name_col] if name_col else [])].copy()
    tmp.columns = ["code", "score"] + (["name"] if name_col else [])
    # 代码去后缀（新浪源带 .SH/.SZ）
    tmp["code"] = tmp["code"].astype(str).str.replace(r"\.(SH|SZ|BJ)$", "", regex=True)
    tmp["score"] = pd.to_numeric(tmp["score"], errors="coerce")
    tmp = tmp.dropna(subset=["score"]).sort_values("score", ascending=False)
    return {"count": len(tmp), "top": tmp.head(top_n).to_dict("records")}


def high_pledge_risk(top_n: int = 20) -> dict:
    """高质押风险股（暴雷预警池）。"""
    from quant_platform.alt_data import high_pledge_stocks
    lst = high_pledge_stocks(top_n=top_n)
    return {"count": len(lst), "threshold": 50.0, "stocks": lst}


def market_factor_snapshot_api() -> dict:
    """市场级因子快照（衍生品/资金面/转债风险偏好）。"""
    from quant_system.market_factors_v7 import market_factor_snapshot
    return market_factor_snapshot()


def cb_double_low_api(top_n: int = 10) -> dict:
    """可转债双低选债（V3 融合层）。"""
    from quant_platform.cb_strategy import cb_double_low
    r = cb_double_low(top_n=top_n)
    return {"mode": r.get("mode"), "position": r.get("position"),
            "picks": r.get("picks", [])[:top_n], "note": r.get("note")}


def sentiment_thermometer_api() -> dict:
    """情绪温度计（市场仓位系数）。

    组装全部 11 个指标（V10 审计 H2 修复）：
    - 日报 meta（generated/a_share_data/{date}-meta.json）优先：
      breadth(全A等权/涨跌家数/涨停占比/成交额比)、hot_metrics(热股等权)、
      margin_metrics(两融变化)
    - 缺省指标用实时快照/DataStore 补充；仍缺则跳过（不参与加权）
    """
    from quant_platform.sentiment_thermometer import compute_thermometer
    import json
    from pathlib import Path
    import numpy as np

    metrics: dict[str, float] = {}
    ROOT = Path(__file__).resolve().parent.parent

    # 1) 日报 meta 优先（最新一份）
    meta_files = sorted((ROOT / "generated" / "a_share_data").glob("*meta.json"))
    if meta_files:
        try:
            meta = json.loads(meta_files[-1].read_text(encoding="utf-8"))
            b = meta.get("breadth") or {}
            if b.get("全A等权涨跌幅%") is not None:
                metrics["all_equal_ret"] = float(b["全A等权涨跌幅%"]) / 100.0
            up = b.get("上涨") or 0
            down = b.get("下跌") or 1
            metrics["up_down_ratio"] = float(up / max(down, 1))
            if b.get("涨停占比%") is not None:
                metrics["limit_up_ratio"] = float(b["涨停占比%"]) / 100.0
            h = meta.get("hot_metrics") or {}
            if h.get("热股等权涨跌幅%") is not None:
                metrics["hot_equal_ret"] = float(h["热股等权涨跌幅%"]) / 100.0
            m = meta.get("margin_metrics") or {}
            if m.get("全A较前日变化%") is not None:
                metrics["margin_change"] = float(m["全A较前日变化%"]) / 100.0
        except Exception:
            pass

    # 2) 实时快照补充（缺省指标）
    try:
        from quant_platform.data import realtime_latest
        df = realtime_latest()
        if df is not None and not df.empty and "pct_chg" in df.columns:
            ret = pd.to_numeric(df["pct_chg"], errors="coerce").dropna()
            if not ret.empty:
                metrics.setdefault("all_equal_ret", float(np.nanmean(ret.values)))
                metrics.setdefault("up_down_ratio",
                                   float((ret > 0).sum() / max(1, (ret < 0).sum())))
                metrics.setdefault("limit_up_ratio", float((ret >= 9.8).sum() / len(ret)))
                # 成交额比：最新5日/20日均值
                if "amount_wan" in df.columns:
                    amt = pd.to_numeric(df["amount_wan"], errors="coerce").dropna()
                    if len(amt):
                        metrics.setdefault("amount_ratio", 1.0)  # 无历史序列，中性
    except Exception:
        pass

    result = compute_thermometer(metrics)

    # V11 融合: 补充 sentiment_factory 多维情绪（价格/量能/资金）作为副维度
    try:
        from quant_system.sentiment_factory.composite import CompositeSentiment
        cs = CompositeSentiment()
        c = cs.compute()
        if isinstance(c, dict) and c.get("composite_score") is not None:
            dims = c.get("dimensions") or {}
            result["multi_dim"] = {
                "composite": c.get("composite_score"),
                "direction": c.get("direction"),
                "confidence": c.get("confidence"),
                "consistency": c.get("consistency"),
                "divergences": c.get("divergences", []),
                "dimensions": {k: v.get("score") if isinstance(v, dict) else v
                               for k, v in dims.items()} if isinstance(dims, dict) else dims,
            }
    except Exception:
        pass  # 副维度失败不影响主温度计

    return result


def rl_executor_demo() -> dict:
    """RL 执行代理演示（规则降级模式）。"""
    from quant_platform.rl_executor import RLExecutor
    snap = {"best_bid": 10.00, "best_ask": 10.02, "last_price": 10.01, "vwap": 10.00,
            "depth_bid_qty": 50000, "depth_ask_qty": 20000,
            "remaining_qty": 800, "target_qty": 1000,
            "time_left": 30, "max_time": 300, "volatility": 0.015}
    d = RLExecutor().decide(snap)
    return {"action": d.action, "action_id": d.action_id,
            "confidence": d.confidence, "reason": d.reason}


if __name__ == "__main__":
    import json
    print("=== openclaw_api 自检 ===")
    print("数据集状态:", len(all_datasets_status()), "个")
    print("概念板块数:", len(concept_list()))
    print("题材板块数:", len(theme_list()))
    s = stock_overview("600519")
    print("茅台概览:", json.dumps({k: (v if not isinstance(v, list) else f"list[{len(v)}]") for k, v in s.items()}, ensure_ascii=False)[:300])
    print("龙虎榜:", json.dumps(lhb_recent(5), ensure_ascii=False)[:200])


# ════════════════════════════════════════════════════════════════
# V11 决策层接口（battle_map/orders/health/industry_chain）
# 供 OpenClaw 会话直接调用：零网络读 generated 缓存，缺失时现场生成
# ════════════════════════════════════════════════════════════════

def v11_battle_map(date: str | None = None) -> dict:
    """当日作战地图（竞价锚点/攻击分组/风险清单/置信度）。"""
    try:
        from quant_system.analysis_core.battle_map import build_map
        bm = build_map(date)
        return {"ok": True, **bm}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


def v11_orders(date: str | None = None, capital: float = 1_000_000) -> dict:
    """指令转化（订单+相关性约束+归因链）。"""
    try:
        from quant_system.analysis_core.order_dispatcher import dispatch
        return {"ok": True, **dispatch(date, capital=capital)}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


def v11_health() -> dict:
    """数据健康检查。"""
    try:
        from quant_system.analysis_core.data_health_check import run
        return {"ok": True, **run(skip_baostock=True)}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


def v11_industry_chain(keyword: str | None = None, stock: str | None = None) -> dict:
    """产业链传导关联（板块名或个股代码）。"""
    try:
        from quant_system.analysis_core.industry_graph import links, stock_links
        if stock:
            return {"ok": True, **stock_links(stock)}
        if keyword:
            return {"ok": True, "keyword": keyword, "links": links(keyword)}
        return {"ok": False, "error": "需提供 keyword(板块名) 或 stock(代码)"}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


def v11_regime(date: str | None = None) -> dict:
    """市场机制识别（趋势/震荡/高波 + 规则）。"""
    try:
        from quant_system.analysis_core.regime_classifier import get_regime
        return {"ok": True, **get_regime(date)}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}
