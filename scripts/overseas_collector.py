#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""海外市场数据采集器（2026-08-21 新增 —— 用户点名"伦敦金这种海外免费数据"）

免费源: Yahoo Finance chart API（无需 key）
覆盖: 伦敦金(XAU→GC=F) / 离岸美元(USDCNH=X) / 美股指(^IXIC ^DJI ^GSPC) /
      原油(CL=F) / 美债(^TNX)
输出: SignalBlock(overseas) — 各标的最新价/涨跌幅/走势，融入每日全面复盘。
"""
from __future__ import annotations

import json
import logging
import sys
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

logger = logging.getLogger("overseas_collector")

from daily_review_collectors import SignalBlock, _safe_collect  # noqa: E402
from quant_system.product_contract import PRODUCT_VERSION, release_metadata  # noqa: E402

# Yahoo 符号: (显示名, 代码, 说明)
SYMBOLS = [
    # 指数与宏观锚点
    ("伦敦金", "GC=F", "COMEX黄金期货(美元/盎司)"),
    ("离岸人民币", "USDCNH=X", "美元/离岸人民币(升=人民币贬)"),
    ("美元指数", "DX-Y.NYB", "美元指数DXY"),
    ("纳指", "^IXIC", "纳斯达克综合指数"),
    ("标普500", "^GSPC", "标普500指数"),
    ("道指", "^DJI", "道琼斯工业指数"),
    ("费城半导体", "^SOX", "费城半导体指数"),
    ("恒生指数", "^HSI", "香港恒生指数"),
    ("WTI原油", "CL=F", "WTI原油期货(美元/桶)"),
    ("美10年债殖", "^TNX", "美国10年期国债收益率"),
    ("VIX恐慌", "^VIX", "标普500波动率指数"),
    # 贵金属与能源：用于通胀、避险和周期状态识别
    ("白银", "SI=F", "COMEX白银期货(美元/盎司)"),
    ("布伦特原油", "BZ=F", "布伦特原油期货(美元/桶)"),
    # 美股核心与半导体链
    ("英伟达", "NVDA", "美股半导体龙头"),
    ("微软", "MSFT", "美股大型科技"),
    ("苹果", "AAPL", "美股大型科技"),
    ("亚马逊", "AMZN", "美股大型科技与消费"),
    # 港股与中概互联网
    ("恒生科技", "^HSTECH", "恒生科技指数"),
    ("腾讯控股", "0700.HK", "港股中概互联网"),
    ("阿里巴巴", "9988.HK", "港股中概互联网"),
    ("美团", "3690.HK", "港股中概互联网"),
    ("京东", "9618.HK", "港股中概互联网"),
]

_D = 5  # 取近5日
MAX_STALE_WEEKDAYS = 3
GROUP_MINIMUMS = {
    "us_equity": 2,
    "commodities": 2,
    "semiconductor": 1,
    "hk_china_internet": 2,
    "macro": 2,
}


_GTIMG_SYMBOLS = [
    ("纳指100", "usNDX"), ("道指", "usDJI"), ("标普500", "usINX"),
    ("恒生指数", "hkHSI"),
]

_SINA_SYMBOLS = [
    ("伦敦金", "hf_GC", "USD/盎司", "future"),
    ("WTI原油", "hf_CL", "USD/桶", "future"),
    ("美元指数", "DINIW", "指数点", "forex"),
    ("离岸人民币", "fx_susdcnh", "USD/CNH", "forex"),
    ("费城半导体", "gb_sox", "指数点", "us_index"),
    ("纳指", "gb_ixic", "指数点", "us_index"),
    ("道指", "gb_dji", "指数点", "us_index"),
    ("标普500", "gb_inx", "指数点", "us_index"),
]


def _gtimg_quotes() -> list[dict]:
    """腾讯 gtimg 全球指数/汇率/大宗（云端可达, Yahoo被墙时兜底）。"""
    import re, urllib.request
    try:
        codes = ",".join(s for _, s in _GTIMG_SYMBOLS)
        req = urllib.request.Request(f"https://qt.gtimg.cn/q={codes}",
                                     headers={"User-Agent": "Mozilla/5.0"})
        txt = urllib.request.urlopen(req, timeout=15).read().decode("gbk", "ignore")
        out = []
        for line in txt.split(";"):
            m = re.match(r'v_(\w+)="(.*?)"', line)
            if not m:
                continue
            code, vals = m.group(1), m.group(2).split("~")
            if len(vals) < 32:
                continue
            price = float(vals[3]) if vals[3] and vals[3] != "0" else None
            # 腾讯全球指数固定字段: 30=时间, 31=涨跌额, 32=涨跌幅。
            try:
                chg_pct = float(vals[32])
            except (ValueError, IndexError):
                chg_pct = None
            label = next((l for l, c in _GTIMG_SYMBOLS if c == code), code)
            out.append({"symbol": code, "label": label,
                        "price": round(price, 2) if price else None,
                        "chg_pct": round(chg_pct, 2) if chg_pct is not None else None,
                        "currency": "USD" if code.startswith("us") else "HKD",
                        "date": vals[30][:10].replace("/", "-") if len(vals) > 30 else None,
                        "desc": label, "source": "腾讯行情"})
        return out
    except Exception as _e:  # noqa: BLE001
        print(f"[overseas] gtimg 失败: {str(_e)[:60]}")
        return []


def _sina_quotes() -> list[dict]:
    """新浪跨资产行情兜底：商品、汇率和费城半导体等美股指数。"""
    import re
    from datetime import datetime

    try:
        codes = ",".join(code for _, code, _, _ in _SINA_SYMBOLS)
        req = urllib.request.Request(
            f"https://hq.sinajs.cn/list={codes}",
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"},
        )
        text = urllib.request.urlopen(req, timeout=15).read().decode("gbk", "ignore")
        definitions = {code: (label, unit, kind) for label, code, unit, kind in _SINA_SYMBOLS}
        out = []
        for line in text.splitlines():
            match = re.search(r"hq_str_([^=]+)=\"(.*)\"", line)
            if not match or match.group(1) not in definitions:
                continue
            code, values = match.group(1), match.group(2).split(",")
            label, unit, kind = definitions[code]
            try:
                if kind == "future":
                    price, previous, date = float(values[0]), float(values[7]), values[12]
                    chg_pct = (price / previous - 1) * 100 if previous else None
                elif kind == "forex":
                    price = float(values[1])
                    previous = float(values[5])
                    chg_pct = (price / previous - 1) * 100 if previous else None
                    date = values[-1]
                else:
                    price, chg_pct = float(values[1]), float(values[2])
                    date = values[3][:10]
                if not date or not str(date)[:4].isdigit():
                    date = datetime.now().strftime("%Y-%m-%d")
                out.append({
                    "symbol": code,
                    "asset_id": code,
                    "label": label,
                    "price": round(price, 4),
                    "chg_pct": round(chg_pct, 2) if chg_pct is not None else None,
                    "currency": unit,
                    "date": str(date)[:10],
                    "desc": label,
                    "source": "新浪行情",
                })
            except (ValueError, IndexError, ZeroDivisionError):
                continue
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("[overseas] 新浪行情失败: %s", str(exc)[:80])
        return []


def _yahoo_quote(code: str) -> Optional[dict]:
    """Yahoo chart API 拉取最新价/涨跌。"""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{code}?interval=1d&range={_D}d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=12) as r:
        d = json.loads(r.read().decode("utf-8"))
    res = (d.get("chart") or {}).get("result") or []
    if not res:
        return None
    meta = res[0].get("meta", {})
    ts = res[0].get("timestamp") or []
    closes = [value for value in ((res[0].get("indicators", {}).get("quote") or [{}])[0].get("close") or [])
              if value is not None]
    if not ts or not closes:
        return None
    last = closes[-1]
    prev = closes[-2] if len(closes) > 1 else last
    chg = (last / prev - 1) * 100 if prev else 0.0
    return {
        "symbol": code, "name": meta.get("symbol", code),
        "price": round(last, 2), "chg_pct": round(chg, 2),
        "currency": meta.get("currency", ""),
        "date": meta.get("regularMarketTime"),
    }


def _observed_date(value) -> str | None:
    """Normalize a provider timestamp/date to YYYY-MM-DD."""
    from datetime import datetime, timezone

    if value is None:
        return None
    if isinstance(value, (int, float)) and value > 0:
        try:
            return datetime.fromtimestamp(value, timezone.utc).date().isoformat()
        except (OSError, OverflowError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) >= 8 and text[:8].isdigit() and "-" not in text[:8]:
        try:
            return datetime.strptime(text[:8], "%Y%m%d").date().isoformat()
        except ValueError:
            return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return text[:10] if len(text) >= 10 and text[:4].isdigit() else None


def _weekday_lag(observed: str | None, as_of: str | None = None) -> int | None:
    from datetime import date, datetime, timedelta

    if not observed:
        return None
    try:
        start = datetime.fromisoformat(observed[:10]).date()
        end = datetime.fromisoformat(as_of[:10]).date() if as_of else date.today()
    except ValueError:
        return None
    if start > end:
        return -1
    lag = 0
    cursor = start
    while cursor < end:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5:
            lag += 1
    return lag


def assess_quotes(quotes: list[dict], *, as_of: str | None = None) -> dict:
    """Validate per-symbol dates and minimum group coverage before scoring."""
    valid: list[dict] = []
    invalid: list[dict] = []
    for raw in quotes:
        quote = dict(raw)
        observed = _observed_date(quote.get("observed_at") or quote.get("date"))
        lag = _weekday_lag(observed, as_of)
        price = quote.get("price")
        chg = quote.get("chg_pct")
        reason = None
        if observed is None:
            reason = "missing_observed_at"
        elif lag is None or lag < 0:
            reason = "invalid_observed_at"
        elif lag > MAX_STALE_WEEKDAYS:
            reason = f"stale:{lag}"
        elif price is None:
            reason = "missing_price"
        elif chg is None:
            reason = "missing_change"
        else:
            try:
                float(price); float(chg)
            except (TypeError, ValueError):
                reason = "non_numeric_quote"
        quote["observed_at"] = observed
        quote["lag_weekdays"] = lag
        quote["valid"] = reason is None
        if reason:
            quote["invalid_reason"] = reason
            invalid.append(quote)
        else:
            valid.append(quote)

    labels = {str(q.get("label")) for q in valid}
    groups = {
        "us_equity": sorted(labels & {"纳指", "标普500", "道指", "纳指100", "英伟达", "微软", "苹果", "亚马逊"}),
        "commodities": sorted(labels & {"伦敦金", "白银", "WTI原油", "布伦特原油"}),
        "semiconductor": sorted(labels & {"费城半导体", "英伟达"}),
        "hk_china_internet": sorted(labels & {"恒生指数", "恒生科技", "腾讯控股", "阿里巴巴", "美团", "京东"}),
        "macro": sorted(labels & {"离岸人民币", "美元指数", "美10年债殖", "VIX恐慌"}),
    }
    failed_groups = [name for name, minimum in GROUP_MINIMUMS.items() if len(groups[name]) < minimum]
    return {
        "quotes": valid,
        "invalid_quotes": invalid,
        "factor_groups": groups,
        "coverage": {
            "available": len(valid),
            "expected": len(SYMBOLS),
            "invalid": len(invalid),
            "group_minimums": dict(GROUP_MINIMUMS),
            "failed_groups": failed_groups,
            "status": "ok" if not failed_groups else "degraded",
        },
        "score_eligible": not failed_groups,
    }


def _global_market_fallback(existing_labels: set[str]) -> list[dict]:
    """Fill missing HK/US stocks through the existing Tencent batch adapter."""
    try:
        from quant_system.global_market import fetch_global_quotes
    except Exception:
        return []
    targets = [
        ("英伟达", "NVDA"), ("微软", "MSFT"), ("苹果", "AAPL"), ("亚马逊", "AMZN"),
        ("腾讯控股", "00700"), ("阿里巴巴", "09988"), ("美团", "03690"), ("京东", "09618"),
    ]
    requested = [symbol for label, symbol in targets if label not in existing_labels]
    if not requested:
        return []
    fetched = fetch_global_quotes(requested, timeout=12)
    out = []
    for label, symbol in targets:
        if label in existing_labels:
            continue
        quote = next((value for value in fetched.values() if value.name == label or value.symbol.endswith(symbol.lstrip("0"))), None)
        if quote is None:
            continue
        out.append({
            "symbol": quote.symbol, "asset_id": quote.symbol, "label": label,
            "price": quote.price, "chg_pct": quote.change_pct,
            "currency": "HKD" if quote.symbol.startswith("hk") else "USD",
            "date": quote.time[:8] if quote.time else None, "observed_at": quote.time,
            "desc": label, "source": "腾讯行情(global_market)",
        })
    return out


def _load_artifact(max_age_hours: float = 26) -> dict | None:
    """读本机生成的海外产物; 过期(>26h)视为不可用→走实时源(gtimg), 云端自给。"""
    import glob, time
    try:
        cands = sorted(glob.glob(str(ROOT / "generated" / "overseas_*.json")))
        if cands:
            age = (time.time() - Path(cands[-1]).stat().st_mtime) / 3600
            if age > max_age_hours:
                return None  # 过期, 云端用 gtimg 实时(本机关机后自给)
            d = json.loads(Path(cands[-1]).read_text(encoding="utf-8"))
            assessed = assess_quotes(d.get("quotes") or [])
            if len(assessed["quotes"]) >= 6:
                return {**assessed, "count": len(assessed["quotes"]),
                        "confidence": 0.7 if assessed["score_eligible"] else 0.3,
                        "_source": f"artifact:{Path(cands[-1]).name}"}
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[overseas] 产物解析失败: {type(e).__name__}: {e}")
    return None


def collect_overseas(gateway=None) -> SignalBlock:
    """采集海外市场（伦敦金/美元/美指/原油/美债）。
    优先读本机产物(overseas_*.json)；无产物才实时抓 Yahoo（本机可用）。"""
    def _run() -> dict:
        art = _load_artifact()
        if art:
            return art
        import json as _j
        from datetime import datetime as _dt, timezone as _tz, timedelta as _td
        quotes = []
        for name, code, desc in SYMBOLS:
            try:
                q = _yahoo_quote(code)
                if q:
                    q["label"] = name
                    q["desc"] = desc
                    quotes.append(q)
            except Exception as e:  # noqa: BLE001
                print(f"[overseas] {name} 失败: {str(e)[:50]}")
        for quote in quotes:
            quote.setdefault("asset_id", quote.get("symbol"))
            quote.setdefault("source", "Yahoo Finance")
        if len(quotes) < 6:
            # Yahoo 全失败或部分失败时都补国内免费源，避免一个偶然成功报价
            # 阻止完整兜底。按标签去重，Yahoo 成功项优先保留。
            seen = {quote.get("label") for quote in quotes}
            for quote in _sina_quotes() + _gtimg_quotes():
                key = quote.get("label")
                if not key or key in seen or quote.get("price") is None:
                    continue
                seen.add(key)
                quotes.append(quote)
        seen = {str(quote.get("label")) for quote in quotes}
        for quote in _global_market_fallback(seen):
            if quote.get("label") not in seen:
                seen.add(str(quote.get("label")))
                quotes.append(quote)
        sources = sorted({quote.get("source") for quote in quotes if quote.get("source")})
        assessed = assess_quotes(quotes)
        _r = {**assessed, "count": len(assessed["quotes"]),
              "confidence": 0.7 if assessed["score_eligible"] else 0.3,
              "_source": "+".join(sources) if sources else "unavailable"}
        # 成功采集后自动存产物(供下次读取, 云端自给)
        if _r.get("quotes"):
            try:
                _d = _dt.now(_tz(_td(hours=8))).strftime("%Y-%m-%d")
                (ROOT / "generated" / f"overseas_{_d}.json").write_text(
                    _j.dumps({"date": _d, "quotes": _r["quotes"],
                             "factor_groups": _r.get("factor_groups", {}),
                             "coverage": _r.get("coverage", {}),
                             "score_eligible": _r.get("score_eligible", False),
                             "invalid_quotes": _r.get("invalid_quotes", []),
                             "product_version": PRODUCT_VERSION, "collected": True},
                             ensure_ascii=False), encoding="utf-8")
                _r["_saved"] = True
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[overseas] 产物写盘失败 generated/overseas_{_d}.json: {type(e).__name__}: {e}")
        return _r
    return _safe_collect(_run, "overseas", "Yahoo Finance", {}, timeout_seconds=40)


def render_overseas_md(block: SignalBlock) -> str:
    """海外市场 → Markdown 段。"""
    L = ["## 🌍 海外市场（免费源: Yahoo）"]
    if block.error or not block.value.get("quotes"):
        L.append(f"- ⚠️ 海外数据采集失败: {block.error or block.value.get('error_hint', '无数据')}")
        return "\n".join(L)
    for q in block.value["quotes"]:
        chg = q.get("chg_pct", 0)
        arrow = "🟢" if chg > 0 else ("🔴" if chg < 0 else "⚪")
        L.append(f"- {q['label']}: **{q['price']}** ({q.get('currency','')}) {arrow} {chg:+.2f}%")
    return "\n".join(L)


def save_overseas_artifact() -> Path:
    """采集海外数据并落盘 generated/overseas_{date}.json（本机执行，云端读产物）。"""
    import json as _json
    from datetime import datetime, timezone, timedelta
    b = collect_overseas()
    out = {"date": datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d"),
           "quotes": b.value.get("quotes", []),
           "factor_groups": b.value.get("factor_groups", {}),
           "coverage": b.value.get("coverage", {}),
           "score_eligible": b.value.get("score_eligible", False),
           "invalid_quotes": b.value.get("invalid_quotes", []),
           "error": b.error, "collected": True,
           "product_version": PRODUCT_VERSION}
    p = ROOT / "generated" / f"overseas_{out['date']}.json"
    p.write_text(_json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[overseas] 海外数据已存: {p} ({len(out['quotes'])}项)")
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if "--save" in sys.argv:
        save_overseas_artifact()
    else:
        b = collect_overseas()
        print(render_overseas_md(b))