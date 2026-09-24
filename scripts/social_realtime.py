#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""舆情实时采集（2026-08-22 —— 云端国内IP自给, 替代停更08-11）

国内源(东财/百度, 云端可直达):
  1. 东财全市场股吧热评 stock_comment_em() → 评论数Top个股(人气)
  2. 百度热搜A股 stock_hot_search_baidu() → 热搜榜
  3. 东财人气榜 stock_hot_rank_em() 带重试(有时断连)
→ 写 data_warehouse/social/{source}_{date}.parquet(当日), social_pulse 自动读最新。

用法:
  python3 scripts/social_realtime.py [--save]   # 采集+落盘
  python3 scripts/social_realtime.py            # 只测试打印
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOCIAL = ROOT / "data_warehouse" / "social"
CST = timezone(timedelta(hours=8))


def _pick_ak():
    import akshare as ak
    return ak


def _ensure_dir():
    SOCIAL.mkdir(parents=True, exist_ok=True)


_FIN_KW = ("股票", "股市", "A股", "基金", "财经", "行情", "投资", "牛市", "熊市", "ETF", "板块", "概念", "资金", "股")


def bilibili_hot(top_n: int = 15) -> dict:
    """B站热门榜(免wbi API): 过滤财经相关内容。"""
    import json as _j
    import urllib.request
    try:
        req = urllib.request.Request(
            "https://api.bilibili.com/x/web-interface/popular?ps=" + str(top_n),
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.bilibili.com/"})
        d = _j.loads(urllib.request.urlopen(req, timeout=15).read().decode("utf-8", "ignore"))
        vids = (d.get("data") or {}).get("list") or []
        items = [{"title": str(v.get("title") or "")[:50], "view": int(v.get("stat", {}).get("view", 0) or 0)}
                 for v in vids]
        # 财经相关筛选(标题含关键词)
        fin = [i for i in items if any(k in i["title"] for k in _FIN_KW)]
        return {"items": items[:8], "fin_items": fin[:5], "total": len(items),
                "note": f"B站热门 {len(items)}条 ({len(fin)}条财经)"}
    except Exception as e:  # noqa: BLE001
        return {"items": [], "note": f"B站采集失败: {str(e)[:40]}"}


def collect_all() -> dict:
    """采集三源 → 结构化(供落盘/复用)。"""
    from datetime import date as _d
    _ensure_dir()
    ak = _pick_ak()
    today = _d.today().strftime("%Y%m%d")
    out = {"date": today}
    # 1. 东财热评(全市场股吧)
    try:
        df = ak.stock_comment_em()
        if df is not None and len(df):
            # 列: 股票代码/股票简称/最新价/涨跌幅/讨论量/点击量/...
            col_map = {}
            for c in df.columns:
                if "代码" in str(c): col_map["code"] = c
                elif "简称" in str(c): col_map["name"] = c
                elif "讨论" in str(c): col_map["talk"] = c
            if "talk" in col_map:
                df = df.sort_values(col_map["talk"], ascending=False)
                out["pulse"] = [{"name": str(r.get(col_map.get("name", df.columns[1]), "")),
                                 "code": str(r.get(col_map.get("code", df.columns[0]), "")),
                                 "talk": int(r.get(col_map["talk"], 0) or 0)}
                                for r in df.head(20).to_dict("records")]
            df.to_parquet(SOCIAL / f"guba_{today}.parquet", index=False)
            out["guba_count"] = len(df)
    except Exception as e:  # noqa: BLE001
        out["guba_err"] = str(e)[:50]
    # 2. 百度热搜A股
    try:
        df2 = ak.stock_hot_search_baidu(symbol="A股")
        if df2 is not None and len(df2):
            out["baidu_top"] = [{"name": str(r.iloc[0]), "chg": str(r.iloc[1])} for _, r in df2.head(10).iterrows()]
            df2.to_parquet(SOCIAL / f"baidu_{today}.parquet", index=False)
    except Exception as e:  # noqa: BLE001
        out["baidu_err"] = str(e)[:50]
    # 3. B站热门榜(免wbi)
    try:
        bd2 = bilibili_hot(15)
        out["bili"] = bd2
        # B站单独落盘，保证可追溯、可复用，不只存在本次进程内。
        import pandas as pd
        bili_rows = []
        for item in (bd2.get("items") or []):
            bili_rows.append({"date": today, "source": "bilibili", "title": item.get("title", ""), "view": item.get("view", 0), "is_finance": False})
        for item in (bd2.get("fin_items") or []):
            bili_rows.append({"date": today, "source": "bilibili", "title": item.get("title", ""), "view": item.get("view", 0), "is_finance": True})
        if bili_rows:
            pd.DataFrame(bili_rows).drop_duplicates(["date", "title"]).to_parquet(SOCIAL / f"bilibili_{today}.parquet", index=False)
    except Exception as e:  # noqa: BLE001
        out["bili_err"] = str(e)[:40]
    # 4. 东财人气榜(带重试)
    for i in range(3):
        try:
            df3 = ak.stock_hot_rank_em()
            if df3 is not None and len(df3):
                df3.to_parquet(SOCIAL / f"rank_{today}.parquet", index=False)
                out["rank_count"] = len(df3)
            break
        except Exception:  # noqa: BLE001
            time.sleep(5)
    return out


def save_and_md() -> str:
    """采集+落盘+生成舆情段(供复盘)."""
    r = collect_all()
    L = ["## 💬 互联舆情（实盘采集 东财股吧+百度热搜）"]
    p = r.get("pulse") or []
    _names, _bn = "", ""
    if p:
        _names = "、".join("%s(%s)" % (x.get("name", ""), x.get("talk", "")) for x in p[:8])
        L.append("- 东财股吧热评Top: " + _names)
    bd = r.get("baidu_top") or []
    if bd:
        _bn = "、".join("%s%s" % (x.get("name", ""), x.get("chg", "")) for x in bd[:6])
        L.append("- 百度热搜A股: " + _bn)
    bl = (r.get("bili") or {}).get("fin_items") or []
    if bl:
        L.append("- B站财经热门: " + "、".join(x["title"] for x in bl[:4]))
    L.append(f"- 采集: 股吧{r.get('guba_count',0)}只 · 日期{r.get('date')}")
    if r.get("guba_err") or r.get("baidu_err"):
        L.append(f"- ⚠️ 部分失败: {r.get('guba_err','')} {r.get('baidu_err','')}")
    return "\n".join(L)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    if "--save" in sys.argv:
        r = collect_all()
        print(json_dump := __import__("json").dumps(r, ensure_ascii=False, default=str)[:400])
    else:
        print(save_and_md())