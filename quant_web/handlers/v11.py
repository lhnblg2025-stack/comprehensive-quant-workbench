"""
handlers/v11 — V11 深度投研分析 API（融合温度/情绪/天梯/决策/数据源/RAG）

路由:
  /api/v11/daily    短线全景（融合+情绪+天梯+三情景+决策卡）
  /api/v11/sources  数据源与知识库状态
  /api/v11/search   RAG 知识检索 ?q=关键词
"""

from __future__ import annotations
import logging
import re

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MARKET = ROOT / "data_warehouse" / "market"
GENERATED = ROOT / "generated"

_HEALTH_CACHE: dict = {"ts": 0.0, "data": None}
_HEALTH_TTL = 60.0

_BATTLE_CACHE: dict = {"ts": 0.0, "data": None, "computing": False}
_BATTLE_TTL = 600.0

_DAILY_CACHE: dict = {"ts": 0.0, "data": None, "computing": False}
_DAILY_TTL = 300.0
CST = timezone(timedelta(hours=8))


def _cutoff_date(value: str | None = None) -> str:
    if value:
        return value[:10]
    try:
        from quant_system.market_clock import latest_completed_trading_day
        return latest_completed_trading_day().isoformat()
    except Exception:
        return datetime.now(CST).strftime("%Y-%m-%d")


def _fusion_latest(cutoff: str | None = None) -> dict | None:
    p = MARKET / "fusion.parquet"
    if not p.exists():
        return None
    import pandas as pd
    df = pd.read_parquet(p)
    if df.empty:
        return None
    df["_day"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if cutoff:
        df = df[df["_day"] <= cutoff]
    if df.empty:
        return None
    r = df.sort_values("_day").iloc[-1]
    return {"date": str(r["_day"]), "temperature": int(r["temperature"]),
            "tag": r["tag"], "emotion_stage": r["emotion_stage"]}


def _ladder_latest(cutoff: str | None = None) -> dict | None:
    p = MARKET / "zt_daily_stats.parquet"
    if not p.exists():
        return None
    import pandas as pd
    df = pd.read_parquet(p)
    df["_day"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if cutoff:
        df = df[df["_day"] <= cutoff]
    df = df.dropna(subset=["_day"]).sort_values("_day").tail(3)
    rows = []
    for _, r in df.iloc[::-1].iterrows():
        rows.append({"date": str(r["_day"]), "zt_cnt": int(r["zt_cnt"]),
                     "zb_cnt": int(r["zb_cnt"]), "dt_cnt": int(r["dt_cnt"]),
                     "max_board": int(r["max_board"])})
    return {"days": rows, "as_of": rows[0]["date"] if rows else None}


def handler_v11_daily(query: dict, send_json) -> None:
    import threading
    import time
    requested_raw = (query.get("date") or [""])[0] or None
    if requested_raw and not re.fullmatch(r"20\d{2}-\d{2}-\d{2}", requested_raw):
        send_json({"ok": False, "error": "date 必须为 YYYY-MM-DD"})
        return
    requested_date = _cutoff_date(requested_raw)
    now = time.time()
    force_refresh = (query.get("refresh") or ["false"])[0].lower() == "true"
    cache_key = f"daily:{requested_date}"
    if force_refresh or _DAILY_CACHE.get("key") != cache_key:
        _DAILY_CACHE["data"] = None
    if _DAILY_CACHE["data"] is not None and now - _DAILY_CACHE["ts"] < _DAILY_TTL:
        send_json(_DAILY_CACHE["data"])
        return
    if _DAILY_CACHE["computing"]:
        send_json({"ok": True, "status": "computing", "hint": "短线全景计算中，请稍后刷新"})
        return
    _DAILY_CACHE["computing"] = True

    def _bg():
        try:
            from quant_system.analysis_core import emotion_cycle, decision_card
            emo = emotion_cycle.run_today(requested_date)
            card = decision_card.build_card(requested_date)
        except Exception as e:
            emo, card = {}, {"error": str(e)[:200]}
        out = {
            "ok": True,
            "requested_date": requested_date,
            "ts": datetime.now(timezone.utc).isoformat(),
            "fusion": _fusion_latest(requested_date),
            "ladder": _ladder_latest(requested_date),
            "emotion": emo,
            "decision_card": card,
        }
        # 历史请求只读取同日命名产物；固定 short_term_daily.md 可能是当前日，
        # 不能与历史 fusion/ladder 拼成跨日报告。
        md = GENERATED / f"short_term_daily_{requested_date}.md"
        if not md.exists() and not requested_raw:
            md = GENERATED / "short_term_daily.md"
        if md.exists():
            text = md.read_text(encoding="utf-8")
            import re
            match = re.search(r"20\d{2}-\d{2}-\d{2}", text)
            report_date = match.group(0) if match else None
            if report_date == requested_date:
                out["report_md"] = text
                out["report_file"] = md.name
                out["report_date"] = report_date
                out["report_generated_at"] = datetime.fromtimestamp(md.stat().st_mtime, timezone.utc).isoformat()
            else:
                out["report_date"] = report_date
                out["report_date_mismatch"] = True
        source_dates = {"fusion": (out.get("fusion") or {}).get("date"), "ladder": (out.get("ladder") or {}).get("as_of"), "emotion": (out.get("emotion") or {}).get("data_date"), "decision_card": (out.get("decision_card") or {}).get("date"), "report": out.get("report_date")}
        mismatches = {key: value for key, value in source_dates.items() if value and value != requested_date}
        out["source_dates"] = source_dates
        out["date_mismatches"] = mismatches
        out["degraded"] = bool(mismatches) or not out.get("fusion") or not out.get("ladder")
        out["status"] = "degraded" if out["degraded"] else "available"
        _DAILY_CACHE["key"] = cache_key
        _DAILY_CACHE["data"] = out
        _DAILY_CACHE["ts"] = time.time()
        _DAILY_CACHE["computing"] = False

    threading.Thread(target=_bg, daemon=True).start()
    send_json({"ok": True, "status": "computing", "hint": "短线全景计算已启动，请稍后刷新"})


def handler_v11_sources(query: dict, send_json) -> None:
    try:
        from quant_system.analysis_core.knowledge_rag import report_source_status
        out = report_source_status()
        out["ok"] = True
    except Exception as e:
        out = {"ok": False, "error": str(e)[:200]}
    send_json(out)


def handler_v11_search(query: dict, send_json) -> None:
    q = (query.get("q") or [""])[0]
    k = int((query.get("k") or ["5"])[0])
    if not q:
        send_json({"ok": False, "error": "缺少 q 参数"})
        return
    try:
        from quant_system.analysis_core.knowledge_rag import search
        res = search(q, k=k)
        send_json({"ok": True, "query": q, "results": res})
    except Exception as e:
        send_json({"ok": False, "error": str(e)[:200]})


def handler_v11_battle(query: dict, send_json) -> None:
    """作战地图（竞价锚点/攻击分组/风险清单/产业链联动），异步计算避免前端卡住。"""
    import threading
    import time
    date = (query.get("date") or [""])[0]
    key = f"battle:{date or 'latest'}"
    now = time.time()
    c = _BATTLE_CACHE
    if c["data"] is not None and now - c["ts"] < _BATTLE_TTL:
        send_json(c["data"])
        return
    if c["computing"]:
        send_json({"ok": True, "status": "computing", "hint": "作战地图计算中，请稍后刷新"})
        return
    c["computing"] = True

    def _bg():
        try:
            from quant_system.analysis_core.battle_map import build_map, render_md
            bm = build_map(date or None)
            c["data"] = {"ok": True, "battle_map": bm, "md": render_md(bm), "key": key}
        except Exception as e:
            c["data"] = {"ok": False, "error": str(e)[:300], "key": key}
        c["ts"] = time.time()
        c["computing"] = False

    threading.Thread(target=_bg, daemon=True).start()
    send_json({"ok": True, "status": "computing", "hint": "作战地图计算已启动，请稍后刷新"})


def handler_v11_orders(query: dict, send_json) -> None:
    """指令转化 + 相关性约束。"""
    date = (query.get("date") or [""])[0]
    cap = float((query.get("capital") or ["1000000"])[0])
    try:
        from quant_system.analysis_core.order_dispatcher import dispatch
        r = dispatch(date or None, capital=cap)
        send_json({"ok": True, **r})
    except Exception as e:
        send_json({"ok": False, "error": str(e)[:300]})


def handler_v11_health(query: dict, send_json) -> None:
    """数据健康检查（60s 缓存，避免重复全量扫盘）。"""
    import time
    now = time.time()
    if _HEALTH_CACHE["data"] is not None and now - _HEALTH_CACHE["ts"] < _HEALTH_TTL:
        send_json(_HEALTH_CACHE["data"])
        return
    try:
        from quant_system.analysis_core.data_health_check import run, print_report
        r = run(skip_baostock=True)
        out = {"ok": True, **r, "md": print_report(r)}
        _HEALTH_CACHE["ts"] = now
        _HEALTH_CACHE["data"] = out
        send_json(out)
    except Exception as e:
        send_json({"ok": False, "error": str(e)[:300]})


# ────────────────────────────────────────────────────────────
# V12 行情升级 — K线 / 热力图 / 资金流瀑布 / 交互式复盘卡
# 数据只读 data_warehouse（不写）；全部失败降级 {ok: false, reason}。
# ────────────────────────────────────────────────────────────
_V12_INDEX_FILES = {
    "sh000001": ("上证指数", "index_daily_上证指数.parquet"),
    "上证指数": ("上证指数", "index_daily_上证指数.parquet"),
    "000001.sh": ("上证指数", "index_daily_上证指数.parquet"),
    "sz399001": ("深证成指", "index_daily_深证成指.parquet"),
    "深证成指": ("深证成指", "index_daily_深证成指.parquet"),
    "深成指": ("深证成指", "index_daily_深证成指.parquet"),
    "399001.sz": ("深证成指", "index_daily_深证成指.parquet"),
    "sz399006": ("创业板指", "index_daily_创业板指.parquet"),
    "创业板指": ("创业板指", "index_daily_创业板指.parquet"),
    "399006": ("创业板指", "index_daily_创业板指.parquet"),
    "399006.sz": ("创业板指", "index_daily_创业板指.parquet"),
    "sh000300": ("沪深300", "index_daily_沪深300.parquet"),
    "hs300": ("沪深300", "index_daily_沪深300.parquet"),
    "沪深300": ("沪深300", "index_daily_沪深300.parquet"),
    "000300": ("沪深300", "index_daily_沪深300.parquet"),
    "000300.sh": ("沪深300", "index_daily_沪深300.parquet"),
}
_V12_INDEX_FALLBACK = "index_daily.parquet"  # 沪深300 兜底（watch_card 同口径）
_V12_KLINE_COLS = ["date", "open", "high", "low", "close", "volume"]


def _v12_round(v, nd: int = 3):
    """NaN/inf → None，否则保留 nd 位。"""
    import numpy as np
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, nd) if np.isfinite(f) else None


def _v12_load_kline(symbol: str):
    """读取日K：6 位代码走 data_warehouse/kline；指数别名走 market/index_daily_*。

    Returns: (df|None, kind, code)。kind ∈ {stock, index}；文件缺失 → (None, kind, code)。
    """
    import re
    from quant_system.analysis_core.common import KLINE_DIR, MARKET_DIR, read_kline_window
    key = symbol.strip().lower()
    idx = _V12_INDEX_FILES.get(key) or _V12_INDEX_FILES.get(symbol.strip())
    if idx is not None:
        name, fn = idx
        path = MARKET_DIR / fn
        if not path.exists() and fn == "index_daily_沪深300.parquet":
            path = MARKET_DIR / _V12_INDEX_FALLBACK
        if not path.exists():
            return None, "index", name
        df = read_kline_window(path, _V12_KLINE_COLS, pd.Timestamp("1990-01-01"))
        return df, "index", name
    code = symbol.strip().zfill(6)
    if not re.fullmatch(r"\d{6}", code):
        return None, "stock", symbol.strip()
    path = KLINE_DIR / f"{code}.parquet"
    if not path.exists():
        return None, "stock", code
    df = read_kline_window(path, _V12_KLINE_COLS, pd.Timestamp("1990-01-01"))
    return df, "stock", code


def handler_v12_kline(query: dict, send_json) -> None:
    """GET /api/v12/kline?symbol=600519&days=120&adjust=qfq

    返回 {symbol, name, dates, ohlc[[o,c,l,h]...], volumes, ma20, ma60, ma144,
    atr14, chip{poc,lower,upper}}。防前视：仅取 ≤days 根、截止最新已完成交易日，
    指标均向后滚动计算。
    """
    symbol = (query.get("symbol") or ["600519"])[0].strip()
    try:
        days = max(1, min(int((query.get("days") or ["120"])[0]), 500))
    except Exception:  # noqa: BLE001
        days = 120
    adjust = (query.get("adjust") or ["qfq"])[0]
    # 2026-08-22 稳定化: 仓库日K本身为前复权(qfq); adjust 仅回显且在非 qfq 时诚实标注
    if adjust not in ("qfq", "hfq"):
        adjust_note = f"请求 adjust={adjust}，但仓库仅前复权数据 → 按 qfq 返回"
        adjust = "qfq"
    else:
        adjust_note = "仓库日K为前复权(qfq)数据，adjust 参数即数据真实属性"
    try:
        import numpy as np
        from quant_system.analysis_core.common import load_names
        from quant_system.analysis_core.volume_profile import calc_volume_profile
        from quant_system.analysis_core.watch_card import atr14

        df, kind, code = _v12_load_kline(symbol)
        if df is None or df.empty:
            send_json({"ok": False, "reason": f"数据未覆盖：{symbol} 无本地日K"})
            return
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for c in ("open", "high", "low", "volume"):
            if c not in df.columns:
                df[c] = np.nan
        df = (df.dropna(subset=["date", "close"]).drop_duplicates("date")
                .sort_values("date"))
        if df.empty:
            send_json({"ok": False, "reason": "数据未覆盖：K线无有效行"})
            return
        close = df["close"].astype(float)
        ma20 = close.rolling(20, min_periods=20).mean()
        ma60 = close.rolling(60, min_periods=60).mean()
        ma144 = close.rolling(144, min_periods=144).mean()
        tail = df.iloc[-days:]

        def _take(s, n: int) -> list:
            return [_v12_round(v) for v in s.iloc[-n:].to_numpy()]

        vp = calc_volume_profile(df)
        chip = {"poc": _v12_round(vp.get("poc")), "lower": _v12_round(vp.get("lower")),
                "upper": _v12_round(vp.get("upper")), "note": vp.get("note"),
                "days": vp.get("n")}
        atr = atr14(df)
        names, _ = load_names()
        name = names.get(code, code) if kind == "stock" else code
        send_json({
            "ok": True,
            "symbol": symbol,
            "name": name,
            "kind": kind,
            "adjust": adjust,
            "adjust_note": adjust_note,
            "dates": [d.strftime("%Y-%m-%d") for d in tail["date"]],
            "ohlc": [[_v12_round(o), _v12_round(c), _v12_round(l), _v12_round(h)]
                     for o, h, l, c in zip(tail["open"].astype(float),
                                           tail["high"].astype(float),
                                           tail["low"].astype(float),
                                           tail["close"].astype(float))],
            "volumes": [_v12_round(v, 0) for v in tail["volume"].astype(float)],
            "ma20": _take(ma20, days),
            "ma60": _take(ma60, days),
            "ma144": _take(ma144, days),
            "atr14": _v12_round(atr),
            "chip": chip,
            "as_of": str(tail["date"].iloc[-1].date()),
        })
    except Exception as e:  # noqa: BLE001
        send_json({"ok": False, "reason": f"K线数据失败: {str(e)[:150]}"})


def _v12_heatmap_sw1() -> tuple[list[dict], str]:
    """申万一级热力：sw_first_hist 最新完成交易日的行业涨跌幅（收盘近似口径）。"""
    from quant_system.analysis_core.common import ROOT
    base = ROOT / "data_warehouse" / "industry"
    hist = pd.read_parquet(base / "sw_first_hist.parquet")
    meta = pd.read_parquet(base / "sw_first.parquet")
    name_map = {}
    if {"行业代码", "行业名称"}.issubset(meta.columns):
        for c, n in zip(meta["行业代码"].astype(str), meta["行业名称"].astype(str)):
            name_map[c.split(".")[0].zfill(6)] = str(n)
    df = hist[["代码", "日期", "收盘"]].copy()
    df["代码"] = df["代码"].astype(str).str.zfill(6)
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    df = df.dropna(subset=["日期", "收盘"]).sort_values(["代码", "日期"])
    if df.empty:
        return [], ""
    last = df["日期"].max()
    items = []
    for code, g in df[df["日期"] <= last].groupby("代码").tail(2).groupby("代码"):
        if len(g) < 2:
            continue
        prev, cur = float(g["收盘"].iloc[0]), float(g["收盘"].iloc[1])
        if prev <= 0:
            continue
        pct = (cur / prev - 1.0) * 100.0
        items.append({"name": name_map.get(code, code), "code": code,
                      "value": round(pct, 2), "up": pct >= 0, "down": pct < 0})
    # 附带各行业前 5 大权重成分（点击联动 K线/复盘卡用）
    try:
        cons = pd.read_parquet(base / "sw_first_cons.parquet")
        if {"行业代码", "证券代码", "证券名称", "最新权重"}.issubset(cons.columns):
            cons["行业代码"] = cons["行业代码"].astype(str).str.zfill(6)
            cons_map: dict[str, list[dict]] = {}
            for code, g in cons.groupby("行业代码"):
                top = g.sort_values("最新权重", ascending=False).head(5)
                cons_map[code] = [{"code": str(r["证券代码"]).zfill(6), "name": str(r["证券名称"])}
                                  for _, r in top.iterrows()]
            for it in items:
                it["constituents"] = cons_map.get(it["code"], [])[:5]
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).error(f"[v11] 操作失败: {e}", exc_info=True)
    items.sort(key=lambda x: x["value"], reverse=True)
    return items, str(last.date())


def _v12_heatmap_concept() -> tuple[list[dict], str]:
    """概念热力：按交易日取 concept_board 的最新快照。"""
    from quant_system.analysis_core.common import ROOT
    p = ROOT / "data_warehouse" / "classification" / "concept_board.parquet"
    df = pd.read_parquet(p)
    if "ts" not in df.columns:
        return [], ""
    df["_day"] = pd.to_datetime(df["ts"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["pct_chg"] = pd.to_numeric(df.get("pct_chg"), errors="coerce")
    df = df.dropna(subset=["_day", "pct_chg"])
    if df.empty:
        return [], ""
    last_day = df["_day"].max()
    items = []
    for _, r in df[df["_day"] == last_day].iterrows():
        pct = float(r["pct_chg"])
        items.append({
            "name": str(r["board_name"]),
            "code": str(r["board_code"]),
            "value": round(pct, 2),
            "up": pct >= 0,
            "down": pct < 0,
            "leader_code": str(r["leader_code"]).zfill(6) if pd.notna(r.get("leader_code")) else "",
            "leader_name": str(r["leader_name"]) if pd.notna(r.get("leader_name")) else "",
        })
    items.sort(key=lambda x: x["value"], reverse=True)
    return items, last_day


_MARKET_TREEMAP_CACHE: dict = {"ts": 0.0, "payload": None}
_MARKET_TREEMAP_TTL = 900


def _v12_market_treemap() -> tuple[list[dict], str, dict]:
    """Two-level market treemap: industry parents and stock children."""
    from quant_system.analysis_core.common import ROOT
    base = ROOT / "data_warehouse"
    import pandas as pd
    industry = pd.read_parquet(base / "industry" / "sw_first_hist.parquet")
    meta = pd.read_parquet(base / "industry" / "sw_first.parquet")
    cons = pd.read_parquet(base / "industry" / "sw_first_cons.parquet")
    kline_dir = base / "kline"
    industry["date"] = pd.to_datetime(industry["日期"], errors="coerce")
    industry["close"] = pd.to_numeric(industry["收盘"], errors="coerce")
    industry = industry.dropna(subset=["date", "close"]).sort_values(["代码", "date"])
    last_day = industry["date"].max()
    names = dict(zip(meta["行业代码"].astype(str).str.split(".").str[0].str.zfill(6), meta["行业名称"].astype(str)))
    parents = []
    for code, group in industry[industry["date"] <= last_day].groupby("代码"):
        tail = group.tail(2)
        if len(tail) < 2 or float(tail.iloc[0]["close"]) <= 0:
            continue
        pct = (float(tail.iloc[-1]["close"]) / float(tail.iloc[0]["close"]) - 1) * 100
        parents.append({"id": str(code).zfill(6), "name": names.get(str(code).zfill(6), str(code)),
                        "parent": "全市场", "value": 1, "pct": round(pct, 2), "kind": "industry"})
    parents.sort(key=lambda item: item["pct"], reverse=True)
    parent_by_code = {item["id"]: item for item in parents}
    children = []
    cons["行业代码"] = cons["行业代码"].astype(str).str.split(".").str[0].str.zfill(6)
    cons["证券代码"] = cons["证券代码"].astype(str).str.extract(r"(\d{6})")[0]
    from concurrent.futures import ThreadPoolExecutor

    rows = [(row, parent_by_code.get(row["行业代码"])) for _, row in cons.iterrows()]
    rows = [(row, parent) for row, parent in rows if parent and isinstance(row.get("证券代码"), str)
            and (kline_dir / f"{row['证券代码']}.parquet").exists()]

    def read_child(pair):
        row, parent = pair
        code = row["证券代码"]
        try:
            kd = pd.read_parquet(kline_dir / f"{code}.parquet", columns=["date", "close", "amount"])
            kd["date"] = pd.to_datetime(kd["date"], errors="coerce")
            kd["close"] = pd.to_numeric(kd["close"], errors="coerce")
            kd = kd.dropna(subset=["date", "close"]).sort_values("date").tail(2)
            if len(kd) < 2 or float(kd.iloc[0]["close"]) <= 0:
                return None
            pct = (float(kd.iloc[-1]["close"]) / float(kd.iloc[0]["close"]) - 1) * 100
            amount_value = pd.to_numeric(kd.iloc[-1].get("amount", 0), errors="coerce")
            amount = float(amount_value) if pd.notna(amount_value) else 0.0
            return {"id": code, "name": str(row.get("证券名称") or code), "parent": parent["id"],
                    "value": max(amount, 1), "pct": round(pct, 2), "kind": "stock"}
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=12) as pool:
        children = [item for item in pool.map(read_child, rows) if item]
    # Limit only children; retain every industry so heatmap does not imply full coverage.
    by_parent = {}
    for item in sorted(children, key=lambda x: x["value"], reverse=True):
        by_parent.setdefault(item["parent"], []).append(item)
    # Plotly treemap 需要显式根节点，行业为第一层、个股为第二层。
    # 父节点面积必须等于其子节点面积之和，不能用固定 value=1，
    # 否则 Plotly 会把行业比例压扁，个股成交额也无法比较。
    selected_children = [child for values in by_parent.values() for child in values[:20]]
    child_totals = {}
    for child in selected_children:
        child_totals[child["parent"]] = child_totals.get(child["parent"], 0) + child["value"]
    parents = [parent for parent in parents if child_totals.get(parent["id"], 0) > 0]
    for parent in parents:
        parent["parent"] = "market"
        parent["value"] = child_totals[parent["id"]]
    root = {"id": "market", "name": "全市场", "parent": "", "value": sum(child_totals.values()) or 1, "pct": 0, "kind": "root"}
    items = [root] + parents + selected_children
    coverage = {"industry_count": len(parents), "stock_count": sum(min(len(v), 20) for v in by_parent.values()),
                "industry_total": len(meta), "stock_coverage": "行业成分前20只且有本地K线",
                "status": "历史结构参考" if str(last_day.date()) < pd.Timestamp.now().strftime("%Y-%m-%d") else "最新交易日"}
    return items, str(last_day.date()), coverage


def _v12_fund_treemap() -> tuple[list[dict], str, dict]:
    """Two-level treemap sized by real stock main_net inflow/outflow."""
    from quant_system.analysis_core.common import ROOT
    import pandas as pd
    base = ROOT / "data_warehouse"
    flow_files = sorted((base / "market" / "fund_flow").glob("fund_flow_*.parquet"))
    if not flow_files:
        return [], "", {"status": "缺少逐股主力资金文件"}
    flow = pd.read_parquet(flow_files[-1])
    required = {"code", "date", "main_net"}
    if not required.issubset(flow.columns):
        return [], "", {"status": "逐股主力资金字段不完整"}
    flow["code"] = flow["code"].astype(str).str.extract(r"(\d{6})")[0]
    flow["main_net"] = pd.to_numeric(flow["main_net"], errors="coerce")
    flow = flow.dropna(subset=["code", "main_net"])
    if flow.empty:
        return [], "", {"status": "没有有效逐股主力资金"}
    as_of = str(flow["date"].astype(str).max())[:10]
    flow = flow[flow["date"].astype(str).str[:10] == as_of]
    cons = pd.read_parquet(base / "industry" / "sw_first_cons.parquet")
    meta = pd.read_parquet(base / "industry" / "sw_first.parquet")
    cons["industry_code"] = cons["行业代码"].astype(str).str.split(".").str[0].str.zfill(6)
    cons["code"] = cons["证券代码"].astype(str).str.extract(r"(\d{6})")[0]
    names = dict(zip(meta["行业代码"].astype(str).str.split(".").str[0].str.zfill(6), meta["行业名称"].astype(str)))
    joined = flow.merge(cons[["code", "industry_code", "证券名称"]], on="code", how="inner")
    if joined.empty:
        return [], as_of, {"status": "资金文件与行业成分无法匹配"}
    joined = joined.drop_duplicates(["code", "industry_code"], keep="last")
    items = []
    for industry_code, group in joined.groupby("industry_code"):
        parent_name = names.get(industry_code, industry_code)
        for _, row in group.sort_values("main_net", key=lambda s: s.abs(), ascending=False).head(30).iterrows():
            net_yi = float(row["main_net"]) / 1e8
            items.append({"id": f"fund-stock-{row['code']}", "name": str(row.get("证券名称") or row["code"]),
                          "parent": f"fund-industry-{industry_code}", "value": max(abs(net_yi), 0.001),
                          "net_yi": round(net_yi, 3), "kind": "stock", "code": row["code"]})
    by_parent = {}
    for item in items:
        by_parent.setdefault(item["parent"], []).append(item)
    parents = []
    for parent_id, children in by_parent.items():
        code = parent_id.replace("fund-industry-", "")
        total = sum(child["value"] for child in children)
        net = sum(child["net_yi"] for child in children)
        parents.append({"id": parent_id, "name": names.get(code, code), "parent": "fund-market",
                        "value": total, "net_yi": round(net, 3), "kind": "industry"})
    root = {"id": "fund-market", "name": "资金流向", "parent": "", "value": sum(item["value"] for item in parents), "net_yi": round(sum(item["net_yi"] for item in parents), 3), "kind": "root"}
    return [root] + sorted(parents, key=lambda item: abs(item["net_yi"]), reverse=True) + items, as_of, {
        "status": "真实逐股主力资金", "industry_count": len(parents), "stock_count": len(items),
        "source": flow_files[-1].name, "unit": "亿元"
    }


def handler_v12_fund_treemap(query: dict, send_json) -> None:
    try:
        items, as_of, coverage = _v12_fund_treemap()
        if not items:
            send_json({"ok": False, "reason": coverage.get("status", "资金热力图未覆盖"), "coverage": coverage})
            return
        send_json({"ok": True, "items": items, "as_of": as_of, "coverage": coverage})
    except Exception as e:
        send_json({"ok": False, "reason": f"资金热力图失败: {str(e)[:180]}"})


def handler_v12_market_treemap(query: dict, send_json) -> None:
    import time
    try:
        if _MARKET_TREEMAP_CACHE["payload"] is not None and time.time() - _MARKET_TREEMAP_CACHE["ts"] < 900:
            items, as_of, coverage = _MARKET_TREEMAP_CACHE["payload"]
        else:
            items, as_of, coverage = _v12_market_treemap()
            _MARKET_TREEMAP_CACHE.update({"ts": time.time(), "payload": (items, as_of, coverage)})
        if not items:
            send_json({"ok": False, "reason": "本地行业/个股行情未覆盖"})
            return
        send_json({"ok": True, "items": items, "as_of": as_of, "coverage": coverage})
    except Exception as e:
        send_json({"ok": False, "reason": f"全市场热力图失败: {str(e)[:180]}"})


def handler_v12_heatmap(query: dict, send_json) -> None:
    """GET /api/v12/heatmap?type=sw1|concept → {items:[{name,value,up,down}], as_of}。"""
    typ = (query.get("type") or ["sw1"])[0].strip().lower()
    if typ not in ("sw1", "concept"):
        send_json({"ok": False, "reason": "type 仅支持 sw1|concept"})
        return
    try:
        items, as_of = _v12_heatmap_sw1() if typ == "sw1" else _v12_heatmap_concept()
        if not items:
            send_json({"ok": False, "reason": "数据未覆盖：无最新交易日热力数据"})
            return
        send_json({"ok": True, "type": typ, "items": items, "as_of": as_of})
    except Exception as e:  # noqa: BLE001
        send_json({"ok": False, "reason": f"热力图数据失败: {str(e)[:150]}"})


def _v12_fund_flow(typ: str) -> tuple[list[dict], str, str]:
    """主力净流入（亿元），严格按概念/行业专表归因。"""
    from quant_system.analysis_core.common import ROOT
    try:
        from quant_system.market_clock import latest_completed_trading_day
        completed_day = latest_completed_trading_day().isoformat()
    except Exception:
        completed_day = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d")
    if typ == "concept":
        p = ROOT / "data_warehouse" / "market" / "concept_fund_flow_intraday.parquet"
        if not p.exists():
            return [], "", p.as_posix()
        df = pd.read_parquet(p)
        if "type" in df.columns:
            df = df[df["type"].astype(str).isin({"概念", "concept"})]
        name_col, value_col = "name", "main_net_yi"
        day_col = "date" if "date" in df.columns else "ts"
        if name_col not in df.columns or value_col not in df.columns or day_col not in df.columns:
            return [], "", f"{p.as_posix()}字段不完整"
        df = df.copy()
        df["_day"] = pd.to_datetime(df[day_col], errors="coerce").dt.strftime("%Y-%m-%d")
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=["_day", value_col])
        if df.empty:
            return [], "", p.as_posix()
        df = df[df["_day"] <= completed_day]
        if df.empty:
            return [], "", p.as_posix()
        last_day = df["_day"].max()
        day = df[df["_day"] == last_day]
        if "ts" in day.columns:
            ts = pd.to_datetime(day["ts"], errors="coerce")
            if ts.notna().any():
                day = day[ts == ts.max()]
        items = [{"name": str(r[name_col]), "main_net_inflow": round(float(r[value_col]), 2)}
                 for _, r in day.sort_values(value_col, ascending=False).iterrows()]
        return items, last_day, p.as_posix()
    try:
        import akshare as ak  # 行业资金流：东财实时
        df = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流")
        source = "akshare.stock_sector_fund_flow_rank(行业资金流)"
        as_of = completed_day
        as_of_kind = "completed_trading_day_cutoff"
    except Exception:
        # Network failure may fall back only to an explicitly labelled industry
        # artifact. Concept rows must never be returned as "sector".
        intraday = ROOT / "data_warehouse" / "market" / "industry_fund_flow.parquet"
        if not intraday.exists():
            raise
        local = pd.read_parquet(intraday)
        if "type" in local.columns:
            local = local[local["type"].astype(str).isin({"行业", "industry"})]
        local["_day"] = pd.to_datetime(local.get("date", local.get("ts")), errors="coerce").dt.strftime("%Y-%m-%d")
        local = local[local["_day"].notna()]
        if local.empty:
            return [], "", "本地行业资金缓存为空（实时行业源不可用）"
        as_of = local["_day"].max()
        day = local[local["_day"] == as_of]
        if "ts" in day.columns:
            ts = pd.to_datetime(day["ts"], errors="coerce")
            if ts.notna().any():
                day = day[ts == ts.max()]
        value_col = "main_net_yi" if "main_net_yi" in day.columns else "main_net"
        items = [{"name": str(r["name"]), "main_net_inflow": round(float(r[value_col]) / (1e8 if value_col == "main_net" else 1), 2)} for _, r in day.iterrows()]
        items.sort(key=lambda x: x["main_net_inflow"], reverse=True)
        return items, as_of, "data_warehouse/market/industry_fund_flow.parquet（实时源不可用）"

    if df is None or df.empty or "名称" not in df.columns:
        return [], "", source
    col = "今日主力净流入-净额"
    if col not in df.columns:
        return [], "", source
    rows = []
    for _, r in df.iterrows():
        try:
            v = float(r[col])
        except (TypeError, ValueError):
            continue
        rows.append({"name": str(r["名称"]), "main_net_inflow": round(v / 1e8, 2)})
    rows.sort(key=lambda x: x["main_net_inflow"], reverse=True)
    return rows, as_of, source


def handler_v12_fund_flow(query: dict, send_json) -> None:
    """GET /api/v12/fund_flow?type=sector|concept → {items:[{name,main_net_inflow(亿)}], as_of}。"""
    typ = (query.get("type") or ["concept"])[0].strip().lower()
    if typ not in ("sector", "concept"):
        send_json({"ok": False, "reason": "type 仅支持 sector|concept"})
        return
    try:
        items, as_of, src = _v12_fund_flow(typ)
        if not items:
            send_json({"ok": False, "reason": "数据未覆盖：无资金流数据"})
            return
        send_json({"ok": True, "type": typ, "items": items, "as_of": as_of, "source": src})
    except Exception as e:  # noqa: BLE001
        send_json({"ok": False, "reason": f"资金流数据失败: {str(e)[:150]}"})


def handler_v12_etf_flow(query: dict, send_json) -> None:
    """ETF行情、份额快照与申赎状态；份额差分只作为方向代理。"""
    import re

    try:
        from quant_system.analysis_core.common import ROOT
        import pandas as pd

        raw_code = str((query.get("code") or [""])[0]).strip()
        try:
            days = max(5, min(120, int((query.get("days") or ["30"])[0])))
        except (TypeError, ValueError):
            send_json({"ok": False, "error": "days 必须是 5-120 的整数", "data_status": "invalid"})
            return

        state_path = ROOT / "data_warehouse" / "events" / "etf_state.parquet"
        hist_path = ROOT / "data_warehouse" / "events" / "etf_history.parquet"
        fallback_path = ROOT / "data_warehouse" / "events" / "etf_flow.parquet"
        selected = raw_code.zfill(6) if raw_code.isdigit() else ""
        history: list[dict] = []
        rows: list[dict] = []
        state_selected = pd.DataFrame()

        def _number(value, divisor=1.0, digits=2):
            if value is None or pd.isna(value):
                return None
            return round(float(value) / divisor, digits)

        def _row(r, code: str | None = None) -> dict:
            price = _number(r.get("price", r.get("close")), digits=4)
            shares = _number(r.get("shares"), digits=2)
            scale = _number(r.get("scale_yi"), digits=2)
            if scale is None and shares is not None and price is not None:
                scale = round(shares * price / 1e8, 2)
            shares_delta_m = _number(r.get("shares_delta"), divisor=1e6, digits=2)
            return {
                "date": str(pd.Timestamp(r.get("date")).date()),
                "code": code or str(r.get("code") or "").zfill(6),
                "name": None if pd.isna(r.get("name")) else str(r.get("name") or ""),
                "price": price,
                "shares_m": None if shares is None else round(shares / 1e6, 2),
                "shares_delta_m": shares_delta_m,
                "share_change_notional_yi": None if shares_delta_m is None or price is None else round(shares_delta_m * 1e6 * price / 1e8, 2),
                "scale_yi": scale,
                "discount_pct": _number(r.get("discount_pct"), digits=3),
                "amount_yi": _number(r.get("amount"), divisor=1e8, digits=2),
                "main_net_yi": _number(r.get("main_net"), divisor=1e8, digits=2),
            }

        if state_path.exists():
            state = pd.read_parquet(state_path)
            required = {"code", "date"}
            if not required.issubset(state.columns):
                raise ValueError(f"ETF状态表缺少字段: {sorted(required - set(state.columns))}")
            state["date"] = pd.to_datetime(state["date"], errors="coerce")
            state["code"] = state["code"].astype(str).str.zfill(6)
            state = state.dropna(subset=["date"]).sort_values(["code", "date"])
            if raw_code and not selected:
                if "name" not in state.columns:
                    send_json({"ok": False, "error": "ETF名称检索不可用，请输入6位代码", "data_status": "invalid"})
                    return
                latest_names = state.sort_values("date").drop_duplicates("code", keep="last")
                matches = latest_names[latest_names["name"].astype(str).str.contains(raw_code, case=False, na=False)]
                if len(matches) != 1:
                    options = matches[["code", "name"]].head(10).to_dict("records")
                    send_json({"ok": False, "error": "ETF名称未唯一匹配，请使用6位代码", "matches": options, "data_status": "ambiguous"})
                    return
                selected = str(matches.iloc[0]["code"])
            if raw_code and not re.fullmatch(r"\d{6}", selected):
                send_json({"ok": False, "error": "ETF代码必须是6位数字", "data_status": "invalid"})
                return

            last_day = state["date"].max()
            if selected:
                state_selected = state[state["code"] == selected].sort_values("date").tail(days)
                if state_selected.empty:
                    send_json({"ok": False, "error": f"ETF {selected} 无可用数据", "data_status": "missing"})
                    return
                rows_frame = state_selected.tail(1)
            else:
                amount = pd.to_numeric(state.get("amount"), errors="coerce")
                rows_frame = state[state["date"] == last_day].assign(_amount=amount).sort_values("_amount", ascending=False).head(30)
            rows = [_row(r) for _, r in rows_frame.iterrows()]
        else:
            if not fallback_path.exists():
                send_json({"ok": False, "error": "ETF资金数据不存在", "data_status": "missing"})
                return
            fallback = pd.read_parquet(fallback_path)
            # 无代码的旧表不能归因给用户指定 ETF，避免拿其他产品的数据顶替。
            if raw_code and "code" not in fallback.columns:
                send_json({"ok": False, "error": f"旧ETF资金表无法证明数据属于 {raw_code}", "data_status": "unattributed"})
                return
            fallback["date"] = pd.to_datetime(fallback["date"], errors="coerce")
            fallback = fallback.dropna(subset=["date"]).sort_values("date")
            if raw_code and not selected:
                send_json({"ok": False, "error": f"旧ETF资金表无法按名称证明数据属于 {raw_code}", "data_status": "unattributed"})
                return
            if selected and "code" in fallback.columns:
                fallback["code"] = fallback["code"].astype(str).str.zfill(6)
                fallback = fallback[fallback["code"] == selected]
            rows = [_row(r, selected or None) for _, r in fallback.tail(days).iterrows()]
            if selected and not rows:
                send_json({"ok": False, "error": f"ETF {selected} 无可用旧表数据", "data_status": "missing"})
                return

        if selected and hist_path.exists():
            hist = pd.read_parquet(hist_path)
            if {"code", "date"}.issubset(hist.columns):
                hist["code"] = hist["code"].astype(str).str.zfill(6)
                hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
                selected_hist = hist[(hist["code"] == selected) & hist["date"].notna()].sort_values("date").tail(days)
                history = [_row(r, selected) for _, r in selected_hist.iterrows()]
        if selected and not history and not state_selected.empty:
            history = [_row(r, selected) for _, r in state_selected.iterrows()]

        selected_series = history if history else rows
        latest = selected_series[-1] if selected_series else {}
        share_observations = [x for x in selected_series if x.get("shares_m") is not None]
        share_changes = [x for x in selected_series if x.get("shares_delta_m") is not None]
        delta_total = round(sum(float(x["shares_delta_m"]) for x in share_changes), 2) if share_changes else None
        latest_delta = share_changes[-1]["shares_delta_m"] if share_changes else None
        amount_recent = sum(float(x.get("amount_yi") or 0) for x in selected_series[-5:])
        amount_prior = sum(float(x.get("amount_yi") or 0) for x in selected_series[-10:-5])
        amount_change = round((amount_recent / amount_prior - 1) * 100, 2) if len(selected_series) >= 10 and amount_prior else None

        official_events = []
        official_path = ROOT / "data_warehouse" / "events" / "official_capital_events.json"
        if official_path.exists():
            raw_events = json.loads(official_path.read_text(encoding="utf-8"))
            for event in raw_events if isinstance(raw_events, list) else []:
                if event.get("error"):
                    continue
                text = str(event.get("text") or "")
                date_match = re.search(r"日期[：:]\s*(20\d{2}-\d{2}-\d{2})", text)
                official_events.append({
                    "title": event.get("title"), "url": event.get("url"),
                    "source": event.get("source"),
                    "date": event.get("date") or (date_match.group(1) if date_match else None),
                    "scope": "market_wide_etf",
                })

        try:
            from quant_system.market_clock import latest_completed_trading_day
            expected = latest_completed_trading_day().isoformat()
            freshness = "current" if latest.get("date") == expected else "stale"
        except Exception:
            expected, freshness = None, "unknown"
        data_status = (
            f"ETF行情缓存至{latest.get('date') or '未知'}；已观测{len(share_observations)}个份额快照，"
            "快照间份额变化仅作申赎方向代理；基金公司逐日申购/赎回流水未接入"
        )
        signals = {
            "amount_change_5d": amount_change,
            "shares_delta_total_m": delta_total,
            "latest_shares_delta_m": latest_delta,
            "share_observation_count": len(share_observations),
            "share_observation_start": share_observations[0]["date"] if share_observations else None,
            "share_observation_end": share_observations[-1]["date"] if share_observations else None,
            "share_delta_kind": "snapshot_to_snapshot_proxy",
            "flow_direction": "net_subscription_proxy" if latest_delta is not None and latest_delta > 0 else "net_redemption_proxy" if latest_delta is not None and latest_delta < 0 else "unknown",
            "official_flow_available": False,
            "subscription_redemption": "基金公司逐日申购/赎回流水未接入；展示值为相邻已观测份额快照差分，不连续时不得视为单日净申购/赎回",
            "huijin": f"证监会公开汇金/稳市事件{len(official_events)}条；属于市场级事件，不是该ETF每日交易流水",
        }
        send_json({
            "ok": True, "data_contract": "etf-flow.v2", "code": selected or None,
            "requested_days": days, "actual_days": len(history) if history else len(rows),
            "as_of": latest.get("date"), "expected_as_of": expected, "freshness": freshness,
            "rows": rows, "history": history, "signals": signals,
            "official_events": official_events[:10], "institutional_proxy": [],
            "data_status": data_status,
        })
    except Exception as exc:
        send_json({"ok": False, "error": f"ETF资金读取失败: {str(exc)[:180]}", "data_status": "error"})


def handler_industry_chain_history(query: dict, send_json) -> None:
    """GET /api/industry_chain_history?chain=&days=30 → 产业链温度历史。"""
    from quant_system.analysis_core.common import ROOT
    import pandas as pd
    p = ROOT / "data_warehouse" / "industry_chain" / "history.parquet"
    if not p.exists():
        send_json({"ok": False, "error": "产业链历史不存在（先运行 collect_industry_chain_history.py）"})
        return
    try:
        chain = (query.get("chain") or [""])[0].strip()
        days = max(5, min(400, int((query.get("days") or ["60"])[0])))
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        data = df[df["chain"] == chain].sort_values("date").tail(days) if chain else df.sort_values(["date", "chain"])
        cols = ["date", "chain", "chain_temperature", "signal", "node_count",
                "upstream_nodes", "midstream_nodes", "downstream_nodes",
                "upstream_temp", "midstream_temp", "downstream_temp",
                "upstream_avg_ret", "midstream_avg_ret", "downstream_avg_ret"]
        send_json({"ok": True, "chain": chain or None, "rows": data[[c for c in cols if c in data.columns]].to_dict("records"),
                   "chains": sorted(df["chain"].dropna().unique().tolist()), "as_of": str(df["date"].max().date())})
    except Exception as exc:  # noqa: BLE001
        send_json({"ok": False, "error": f"产业链历史读取失败: {str(exc)[:150]}"})


def handler_v12_stock_card(query: dict, send_json) -> None:
    """GET /api/v12/stock_card?symbol=600519 → 复用 watch_card.build_card 结构化 dict。"""
    import re
    symbol = (query.get("symbol") or [""])[0].strip()
    if not symbol:
        send_json({"ok": False, "reason": "缺少 symbol 参数"})
        return
    code = symbol.zfill(6) if symbol.isdigit() else symbol
    if not re.fullmatch(r"\d{6}", code):
        send_json({"ok": False, "reason": "symbol 需为 6 位股票代码"})
        return
    try:
        from quant_system.analysis_core import watch_card
        from quant_system.analysis_core.common import load_names
        from quant_system.analysis_core.rs_strength import load_benchmark
        bench, _ = load_benchmark()
        if bench is None or bench.empty:
            send_json({"ok": False, "reason": "RS 基准（沪深300）不可用"})
            return
        target = watch_card._latest_common_day([code], bench)
        card = watch_card.build_card(code, target, bench)
        names, _ = load_names()
        card["name"] = names.get(code, code)
        card_date = card.get("date", target.date())
        if hasattr(card_date, "strftime"):
            card_date = card_date.strftime("%Y-%m-%d")
        send_json({"ok": True, "symbol": code, "name": card["name"],
                   "date": card_date, "card": card})
    except Exception as e:  # noqa: BLE001
        send_json({"ok": False, "reason": f"复盘卡数据失败: {str(e)[:150]}"})
