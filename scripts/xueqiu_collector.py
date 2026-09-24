#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""雪球采集器（2026-08-22 —— 用户选①手动登录提供cookie后接入）

用户提供浏览器登录雪球后的 cookie (xq_a_token / u) → 存放 C:/quant/config/xueqiu_cookie.json
本模块读取该 cookie → 拉取雪球高权限数据:
  1. hot_posts():  热帖 listV2(登录态)
  2. hot_stocks(): 热门股票 v5/stock/hot_stock/list.json
  3. search():     关键词搜索(用户动态)
→ 写 data_warehouse/social/xueqiu_{date}.json, social_pulse 自动读最新。

用法:
  python3 scripts/xueqiu_collector.py --test     # 测cookie是否有效
  python3 scripts/xueqiu_collector.py --collect  # 采集+落盘
"""
from __future__ import annotations

import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOCIAL = ROOT / "data_warehouse" / "social"
COOKIE_FILE = ROOT / "config" / "xueqiu_cookie.json"

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36")


def load_cookie() -> dict:
    """读登录cookie(xq_a_token/u)。

    V12.3+ 密钥治理：优先从 config/.env.secrets（env:XQ_* 占位）读取，
    明文 config/xueqiu_cookie.json 仅作旧版兼容回退；若两者皆无返回 {}。
    """
    # 1) 优先 secret_loader（env / .env.secrets）
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from secret_loader import get_secret  # noqa: PLC0415
        tok = get_secret("XQ_XQ_A_TOKEN") or get_secret("XQ_TOKEN")
        if tok:
            return {
                "xq_a_token": tok,
                "xq_r_token": get_secret("XQ_XQ_R_TOKEN") or "",
                "u": get_secret("XQ_U") or "",
                "acw_tc": get_secret("XQ_ACW_TC") or "",
            }
    except Exception:  # noqa: BLE001
        pass
    # 2) 旧版明文 JSON 兼容回退（后续迁移完成即删除）
    p = COOKIE_FILE
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _req(url: str, cookie: dict) -> str:
    tok = cookie.get("xq_a_token") or cookie.get("token", "")
    u = cookie.get("u", "")
    hdrs = {
        "User-Agent": _UA,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Referer": "https://xueqiu.com/",
        "Cookie": f"xq_a_token={tok}; u={u}; acw_tc={cookie.get('acw_tc','')}",
    }
    req = urllib.request.Request(url, headers=hdrs)
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read().decode("utf-8", "ignore")


def test_cookie() -> str:
    """试调用验证cookie有效. 400016=无效/未授权. 200=OK. """
    try:
        d = json.loads(_req("https://xueqiu.com/statuses/hot/listV2.json?since_id=-1&max_id=-1&size=3",
                            load_cookie()))
        if d.get("code") == 0 or d.get("items"):
            return f"✅ cookie有效: 热帖{len(d.get('items', []))}条"
        return f"⚠️ cookie可能无效: {json.dumps(d, ensure_ascii=False)[:80]}"
    except Exception as e:  # noqa: BLE001
        return f"❌ 访问失败: {str(e)[:60]}"


def hot_posts(count: int = 10) -> list[dict]:
    """雪球热帖(登录态)."""
    d = json.loads(_req(
        f"https://xueqiu.com/statuses/hot/listV2.json?since_id=-1&max_id=-1&size={count}",
        load_cookie()))
    items = d.get("items") or []
    out = []
    for it in items[:count]:
        t = it.get("original_status", it)
        out.append({"title": str(t.get("title") or t.get("text") or "")[:80],
                    "target": str(t.get("target", ""))[:10],
                    "like": int((t.get("like_count") or 0)), "reply": int((t.get("reply_count") or 0))})
    return out


def hot_stocks(count: int = 10) -> list[dict]:
    """雪球热门股票(v5 登录态)."""
    d = json.loads(_req(
        f"https://stock.xueqiu.com/v5/stock/hot_stock/list.json?size={count}&_type=10&type=10",
        load_cookie()))
    hs = (d.get("data") or {}).get("items") or []
    out = []
    for h in hs[:count]:
        o = h.get("quote", h)
        out.append({"name": str(o.get("name", "")), "symbol": str(o.get("symbol", "")),
                    "pct": o.get("percent", 0), "price": o.get("current", 0)})
    return out


def collect() -> dict:
    """采集+落盘 → data_warehouse/social/xueqiu_{date}.json"""
    SOCIAL.mkdir(parents=True, exist_ok=True)
    r = {"date": date.today().strftime("%Y%m%d")}
    try:
        r["posts"] = hot_posts(8)
        r["posts_n"] = len(r["posts"])
    except Exception as e:  # noqa: BLE001
        r["posts_err"] = str(e)[:60]
    try:
        r["stocks"] = hot_stocks(10)
        r["stocks_n"] = len(r["stocks"])
    except Exception as e:  # noqa: BLE001
        r["stocks_err"] = str(e)[:60]
    (SOCIAL / f"xueqiu_{r['date']}.json").write_text(
        json.dumps(r, ensure_ascii=False), encoding="utf-8")
    return r


def collect_md() -> str:
    """采集 → Markdown 段(供复盘)."""
    r = collect()
    L = ["## 雪球（登录态高权限数据）"]
    for p in r.get("posts", [])[:5]:
        L.append(f"- 🔥 {p['title']} (赞{p['like']}/评{p['reply']})")
    for s in r.get("stocks", [])[:6]:
        L.append(f"- 📈 {s['name']} {s['symbol']} {s['pct']:+.2f}% {s['price']}")
    if not r.get("posts") and not r.get("stocks"):
        L.append("- ⚠️ 无数据: " + (r.get("posts_err") or r.get("stocks_err") or "cookie未配置"))
    return "\n".join(L)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    if "--test" in sys.argv:
        print(test_cookie())
    elif "--collect" in sys.argv:
        print(collect_md())
    else:
        print(test_cookie())