#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""市场基座全量榨干（2026-08-22 —— 审计发现 market/ 95文件未全展示）

聚合 data_warehouse/market/ 全部高价值数据 → 研报"市场全景"章节:
  1. lhb_hist_near():   龙虎榜历史(近5日, 从lhb_*季度+日文件)
  2. commodity_snap():  商品期货(commodity__*: gold/copper/crude/coal/bdi)
  3. rates_snap():      利率(rates__*: fx/lpr/us_rate + shibor)
  4. macro_month():     月度宏观(macro_monthly__*: cpi/ppi/pmi/m2)
  5. zt_daily():        涨停日统计(zt_daily_stats)
  6. regime_hist():     状态机历史(regime_history)
  7. margin_snap():     两融汇总(margin + market_margin_sh/sz)
  8. fund_flow_snap():  市场资金流(stock_market_fund_flow + sector_fund_flow)
纯本地读, 秒级, 缓存1h。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MK = ROOT / "data_warehouse" / "market"
CACHE = ROOT / "generated" / "market_pano_cache.json"
_READ_STATUS: dict[str, dict] = {}


def _rd(name, **kw):
    import pandas as pd
    p = MK / f"{name}.parquet"
    if not p.exists():
        _READ_STATUS[name] = {"status": "missing", "error": "文件不存在"}
        return None
    try:
        frame = pd.read_parquet(p, **kw)
        _READ_STATUS[name] = {"status": "ok" if len(frame) else "empty", "rows": len(frame)}
        return frame
    except Exception as exc:
        _READ_STATUS[name] = {"status": "error", "error": f"{type(exc).__name__}: {str(exc)[:100]}"}
        return None


def _latest_row(df):
    if df is None or not len(df):
        return None
    try:
        return df.iloc[-1]
    except Exception:
        return None


def _record(df) -> dict | None:
    """Return a JSON-safe latest row without truncating its business fields."""
    import math

    row = _latest_row(df)
    if row is None:
        return None
    out = {}
    for key, value in row.to_dict().items():
        if value is None or (isinstance(value, float) and math.isnan(value)):
            out[str(key)] = None
        elif hasattr(value, "isoformat"):
            out[str(key)] = value.isoformat()
        elif hasattr(value, "item"):
            out[str(key)] = value.item()
        else:
            out[str(key)] = value
    return out


def _row_date(record: dict | None) -> str | None:
    for key in ("date", "日期", "trade_date", "报告期", "ts"):
        value = (record or {}).get(key)
        if value:
            return str(value)[:10]
    return None


def _json_clean(value):
    """Recursively replace numpy scalars and non-finite floats before caching."""
    import math

    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def market_pano() -> dict:
    if CACHE.exists() and time.time() - CACHE.stat().st_mtime < 3600:
        try:
            cached = json.loads(CACHE.read_text(encoding="utf-8"))
            if cached.get("schema") == "market_pano/v4":
                return cached
        except Exception:
            pass
    _READ_STATUS.clear()
    out = {"schema": "market_pano/v4", "status": "ok", "errors": []}
    # 1. 商品期货
    combos = []
    for f in sorted(MK.glob("commodity__*.parquet")):
        df = _rd(f.name.rsplit(".", 1)[0])
        r = _latest_row(df)
        if r is not None:
            nm = f.name.replace("commodity__", "").replace(".parquet", "")
            try:
                combos.append({"name": nm, "last": float(r.iloc[-1])})
            except Exception:
                pass
    out["commodity"] = combos[:6]
    # 2. 利率
    rates = []
    for f in sorted(MK.glob("rates__*.parquet")):
        df = _rd(f.name.rsplit(".", 1)[0])
        r = _latest_row(df)
        if r is not None:
            nm = f.name.replace("rates__", "").replace(".parquet", "")
            try:
                rates.append({"name": nm, "last": float(r.iloc[-1])})
            except Exception:
                pass
    out["rates"] = rates[:4]
    # 3. 月度宏观
    macro = {}
    for f in sorted(MK.glob("macro_monthly__*.parquet")):
        df = _rd(f.name.rsplit(".", 1)[0])
        if df is not None and len(df):
            nm = f.name.replace("macro_monthly__", "").replace(".parquet", "")
            try:
                last = df.iloc[-1]
                macro[nm] = float(last.iloc[-1]) if last.dtype.kind in "fi" else str(last.iloc[-1])[:15]
            except Exception:
                pass
    out["macro"] = macro
    # 4. 涨停日统计
    out["zt_stats"] = _record(_rd("zt_daily_stats"))
    # 5. 状态机历史
    out["regime"] = _record(_rd("regime_history"))
    # 6. 两融
    out["margin"] = _record(_rd("margin"))
    # 7. 市场资金流
    out["fund_flow"] = _record(_rd("stock_market_fund_flow"))
    # 8. 市场估值(平均PE)
    pe = _rd("market_heat__market_pe")
    if pe is not None and len(pe):
        try:
            r = pe.tail(1).to_dict("records")[0]
            out["market_pe"] = {str(k): str(v)[:10] for k, v in r.items()}
        except Exception:
            pass
    # 9. 恐慌指数(QVIX)
    qx = _rd("qvix")
    if qx is not None and len(qx):
        try:
            out["qvix"] = str(qx.tail(1).to_dict("records")[0])[:60]
        except Exception:
            pass
    # 10. 可转债(数量+首只)
    cb = _rd("cb_spot")
    if cb is not None and len(cb):
        try:
            out["cb"] = {"count": len(cb), "top": str(cb.head(1).to_dict("records")[0])[:40]}
        except Exception:
            pass
    # 11. 期指基差
    fb = _rd("futures_basis")
    if fb is not None and len(fb):
        try:
            out["basis"] = str(fb.tail(1).to_dict("records")[0])[:60]
        except Exception:
            pass
    dates = [_row_date(out.get(key)) for key in ("zt_stats", "regime", "margin", "fund_flow")]
    out["as_of"] = max((date for date in dates if date), default=None)
    useful = [key for key in ("commodity", "rates", "macro", "zt_stats", "regime", "margin", "fund_flow") if out.get(key)]
    out["sources"] = dict(_READ_STATUS)
    out["errors"] = [{"source": name, **meta} for name, meta in _READ_STATUS.items()
                     if meta.get("status") in ("error", "missing")]
    critical = ("zt_daily_stats", "regime_history", "margin", "stock_market_fund_flow")
    critical_ok = sum(_READ_STATUS.get(name, {}).get("status") == "ok" for name in critical)
    has_read_errors = any(meta.get("status") in ("error", "missing") for meta in _READ_STATUS.values())
    out["status"] = "ok" if len(useful) >= 5 and critical_ok >= 3 and not has_read_errors else ("partial" if useful else "unavailable")
    out = _json_clean(out)
    CACHE.write_text(json.dumps(out, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return out


def market_md() -> str:
    p = market_pano()
    L = ["## 🌐 市场基座全景（商品/利率/宏观/资金/两融/涨停统计）"]
    if p.get("commodity"):
        L.append("- 商品: " + "、".join(f"{c['name']}:{c['last']}" for c in p["commodity"]))
    if p.get("rates"):
        L.append("- 利率: " + "、".join(f"{r['name']}:{r['last']}" for r in p["rates"]))
    if p.get("macro"):
        L.append("- 月度宏观: " + "、".join(f"{k}={v}" for k, v in list(p["macro"].items())[:8]))
    for k, nm in [("zt_stats", "涨停统计"), ("regime", "状态机"), ("margin", "两融"), ("fund_flow", "市场资金流")]:
        if p.get(k):
            L.append(f"- {nm}: {p[k]}")
    if p.get("market_pe"):
        L.append(f"- 市场估值: {p['market_pe']}")
    if p.get("qvix"):
        L.append(f"- 恐慌(QVIX): {p['qvix']}")
    if p.get("cb"):
        L.append(f"- 可转债: {p['cb']}")
    if p.get("basis"):
        L.append(f"- 期指基差: {p['basis']}")
    if len(p) <= 1:
        L.append("- 市场基座数据缺失")
    return "\n".join(L)


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print(market_md())