#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""社交舆情战法（2026-08-22 —— 用户"股吧微博等互联网战法用了没有")

读 data_warehouse/social/ 下股吧/微博/百度热度数据 → 舆情信号:
 1. guba_heat()      → 股吧/微博热门个股热度(读数)
 2. baidu_hot()      → 百度热搜(综合热度+涨跌)
 3. sentiment_pulse()→ 合一舆情信号(热度方向/拥挤示意)
供复盘"互联舆情"节 + 决策参考。数据停更则标注(外部采集停更)。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOCIAL = ROOT / "data_warehouse" / "social"


def _latest_files(prefix: str):
    import glob
    fs = sorted(glob.glob(str(SOCIAL / f"{prefix}*.parquet")))
    fs += sorted(glob.glob(str(SOCIAL / f"{prefix}*.json")))
    return fs


def _file_as_of(path: str, frame=None, payload: dict | None = None) -> str | None:
    """Use an explicit data date first; accept a filename date only after validation."""
    import re
    import pandas as pd

    if frame is not None:
        for column in ("date", "日期", "as_of", "updated_at"):
            if column in frame.columns:
                values = pd.to_datetime(frame[column], errors="coerce").dropna()
                if not values.empty:
                    return values.max().strftime("%Y-%m-%d")
    for key in ("as_of", "date", "updated_at"):
        value = (payload or {}).get(key)
        parsed = pd.to_datetime(value, errors="coerce") if value else None
        if parsed is not None and not pd.isna(parsed):
            return parsed.strftime("%Y-%m-%d")
    match = re.search(r"(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)", Path(path).stem)
    if match:
        try:
            return pd.Timestamp("-".join(match.groups())).strftime("%Y-%m-%d")
        except ValueError:
            return None
    return None


def _source(status: str, items: list, as_of: str | None = None, error: str | None = None, note: str = "") -> dict:
    import pandas as pd

    stale = False
    if as_of:
        stale = (pd.Timestamp.now().normalize() - pd.Timestamp(as_of).normalize()).days > 3
    final_status = "stale" if stale and status == "ok" else status
    return {"status": final_status, "as_of": as_of, "items": items, "error": error, "note": note}


def guba_heat(top_n: int = 8) -> dict:
    """股吧热度(实时采集版): 读 guba_*.parquet(东财评论, code/name/talk列)。"""
    import pandas as pd
    for prefix in ("guba", "weibo"):
        fs = _latest_files(prefix)
        if not fs:
            continue
        try:
            df = pd.read_parquet(fs[-1])
            # 东财股吧: 名称/代码/关注指数/涨跌幅; 排序用关注指数 或 讨论量
            name_col = next((c for c in df.columns if "简称" in str(c) or ("名称" in str(c) and "代码" not in str(c))), "名称")
            sort_col = next((c for c in df.columns if "关注指数" in str(c) or "讨论" in str(c) or "talk" in str(c).lower()), None)
            chg_col = next((c for c in df.columns if "涨跌幅" in str(c) or "chg" in str(c).lower()), None)
            if sort_col:
                df = df.sort_values(sort_col, ascending=False)
            items = [{"name": str(r.get(name_col, "")), "rate": int(r.get(sort_col, 0) or 0) if sort_col else 0,
                      "chg": float(r.get(chg_col, 0) or 0) if chg_col else None}
                     for r in df.head(top_n).to_dict("records")]
            return _source("ok" if items else "empty", items, _file_as_of(fs[-1], frame=df),
                           note=f"东财股吧热评 {len(items)}只")
        except Exception as exc:  # noqa: BLE001
            return _source("error", [], error=f"{type(exc).__name__}: {str(exc)[:80]}")
    return _source("missing", [], error="股吧/微博采集文件不存在")


def baidu_hot(top_n: int = 8) -> dict:
    """百度热搜(名称/涨跌幅/综合热度)。"""
    import pandas as pd
    fs = _latest_files("baidu")
    if not fs:
        return _source("missing", [], error="百度热搜采集文件不存在")
    try:
        df = pd.read_parquet(fs[-1])
        cols = list(df.columns)
        nm_col = next((c for c in cols if "名称" in c or "name" in c.lower()), cols[0])
        hot_col = next((c for c in cols if "热度" in c or "hot" in c.lower()), None)
        chg_col = next((c for c in cols if "涨跌" in c or "chg" in c.lower()), None)
        df = df.sort_values(hot_col, ascending=False) if hot_col else df
        records = df.head(top_n).to_dict("records")
        items = []
        for r in records:
            chg_raw = r.get(chg_col, 0) if chg_col else 0
            chg_v = None
            if isinstance(chg_raw, str) and "%" in chg_raw:
                try: chg_v = float(chg_raw.replace("%", "").replace("+", ""))
                except Exception: chg_v = None
            items.append({"name": str(r.get(nm_col, "")), "hot": r.get(hot_col, 0) if hot_col else 0,
                          "chg": chg_v})
        return _source("ok" if items else "empty", items, _file_as_of(fs[-1], frame=df),
                       note=f"百度热搜 {len(items)}只")
    except Exception as e:  # noqa: BLE001
        return _source("error", [], error=f"{type(e).__name__}: {str(e)[:80]}")


def xueqiu_pulse(top_n: int = 6) -> dict:
    """雪球登录态数据(playwright采集落盘 xueqiu_{date}.json)."""
    import json as _j, glob as _g
    fs = sorted(_g.glob(str(SOCIAL / "xueqiu_*.json")))
    if not fs:
        return {"status": "missing", "posts": [], "stocks": [], "as_of": None,
                "error": "雪球采集文件不存在或登录采集未运行"}
    try:
        d = _j.loads(Path(fs[-1]).read_text(encoding="utf-8"))
        posts = d.get("posts", []) or []
        stocks = d.get("stocks", []) or []
        as_of = _file_as_of(fs[-1], payload=d)
        source = _source("ok" if posts or stocks else "empty", posts[:top_n], as_of,
                         note=f"雪球热帖{len(posts)} 热门股{len(stocks)}")
        source.update({"posts": posts[:top_n], "stocks": stocks[:top_n]})
        return source
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "posts": [], "stocks": [], "as_of": None,
                "error": f"{type(e).__name__}: {str(e)[:80]}"}


def sentiment_pulse() -> dict:
    """合一舆情: 热度方向(涨跌家数比)+拥挤示意+数据新鲜度。"""
    g = guba_heat()
    b = baidu_hot()
    items = (g.get("items", []) or []) + (b.get("items", []) or [])
    # B站快照作为独立热度来源展示，不强行转化为涨跌方向。
    bili = _source("missing", [], error="B站采集文件不存在")
    try:
        fs = _latest_files("bilibili")
        if fs:
            import pandas as pd
            bd = pd.read_parquet(fs[-1])
            items_bili = bd.head(8).to_dict("records")
            bili = _source("ok" if items_bili else "empty", items_bili,
                           _file_as_of(fs[-1], frame=bd))
    except Exception as exc:
        bili = _source("error", [], error=f"{type(exc).__name__}: {str(exc)[:80]}")
    # 热度股涨跌方向
    up_n = sum(1 for i in items if (i.get("chg") or 0) > 0)
    dn_n = sum(1 for i in items if (i.get("chg") or 0) < 0)
    crowded = up_n > dn_n * 2 and (up_n + dn_n) >= 6  # 热股普涨=偏热
    return {
        "guba": g, "baidu": b, "bilibili": bili, "total": len(items),
        "hot_up": up_n, "hot_down": dn_n, "crowded": crowded,
        "note": f"舆情: 热度{len(items)}股 涨{up_n}跌{dn_n} {'🟠偏热' if crowded else '🟢正常'}",
    }


def sentiment_md() -> str:
    """舆情段 → Markdown。"""
    p = sentiment_pulse()
    L = ["## 💬 市场舆情（股吧/微博/百度/B站/雪球）"]
    L.append(f"- {p.get('note')}")
    g = p.get("guba", {})
    if g.get("items"):
        L.append(f"- 股吧/微博热股: {'、'.join(str(i['name']) for i in g['items'][:6])}")
    b = p.get("baidu", {})
    if b.get("items"):
        L.append(f"- 百度热搜: {'、'.join(str(i['name']) for i in b['items'][:6])}")
    bi = p.get("bilibili", {})
    if bi.get("items"):
        L.append(f"- B站财经/热门快照: {len(bi['items'])}条（数据日{bi.get('date','-')}）")
    xq = xueqiu_pulse()
    if xq.get("posts"):
        L.append(f"- ❄️ {xq['note']}")
        for x in xq["posts"][:3]:
            L.append(f"  - {x.get('title','')[:40]}")
    return "\n".join(L)


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    print(sentiment_md())