#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""雪球 playwright 采集（2026-08-22 —— 过阿里云WAF, 云端/本机通用）

雪球 API 有阿里云 WAF JS 挑战(非浏览器请求返回 renderData 页)。
用 playwright chromium 加载真实浏览器环境 + 用户cookie → 访问热帖/热门股票, WAF 自动过。

用法:
  python3 scripts/xueqiu_browser.py --test     # 过WAF测cookie
  python3 scripts/xueqiu_browser.py --collect  # 采集热帖+热门股票+热议 → JSON
  python3 scripts/xueqiu_browser.py --save     # 采集+落盘+写入social目录
无需 Xvfb(云端Windows有桌面会话)。16min超时兜底。
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOCIAL = ROOT / "data_warehouse" / "social"
COOKIE_FILE = ROOT / "config" / "xueqiu_cookie.json"


def _cookie_str() -> str:
    # 优先 secret_loader（env / .env.secrets）；明文 JSON 仅旧版回退
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from secret_loader import get_secret  # noqa: PLC0415
        tok = get_secret("XQ_XQ_A_TOKEN")
        if tok:
            parts = [
                f"xq_a_token={tok}",
                f"xq_r_token={get_secret('XQ_XQ_R_TOKEN') or ''}",
                f"u={get_secret('XQ_U') or ''}",
                f"acw_tc={get_secret('XQ_ACW_TC') or ''}",
            ]
            return "; ".join(p for p in parts if not p.endswith("="))
    except Exception:  # noqa: BLE001
        pass
    try:
        ck = json.loads(COOKIE_FILE.read_text(encoding="utf-8"))
        parts = []
        for k in ("xq_a_token", "xq_r_token", "u", "device_id", "acw_tc"):
            if ck.get(k):
                parts.append(f"{k}={ck[k]}")
        return "; ".join(parts)
    except Exception:
        return ""


def _collect(headless: bool = True, timeout_ms: int = 60000) -> dict:
    """playwright 加载雪球首页 → 捕获页面自动发起的 API 响应(真实JSON, 2026-08-22 突破)

    原理: 雪球首页导航时自动请求 hot_event/hot_stock 等 API(带完整浏览器指纹+WAF通过态),
    在 response 事件捕获即可, 无需逆向签名。"""
    from playwright.sync_api import sync_playwright
    out = {"date": date.today().strftime("%Y%m%d")}
    ck_str = _cookie_str()
    captured = {}
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=headless, channel="msedge")
        except Exception:
            browser = p.chromium.launch(headless=headless)
        ctx = browser.new_context(user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "Chrome/125.0.0.0 Safari/537.36"), locale="zh-CN")
        for kv in ck_str.split(";"):
            if "=" in kv:
                k, v = kv.strip().split("=", 1)
                try:
                    ctx.add_cookies([{"name": k, "value": v, "domain": ".xueqiu.com", "path": "/"}])
                except Exception:
                    pass
        page = ctx.new_page()

        def _cap(r):
            u = r.url
            try:
                if "hot_event/list.json" in u:
                    captured["hot_event"] = r.json()
                elif "hot_stock/list.json" in u:
                    captured["hot_stock"] = r.json()
                elif "listV2" in u:
                    captured["listV2"] = r.json()
            except Exception:
                pass

        page.on("response", _cap)
        page.goto("https://xueqiu.com/", timeout=30000)
        page.wait_for_timeout(8000)
        browser.close()
    # 解析捕获
    he = (captured.get("hot_event") or {}).get("list") or []
    out["posts"] = [
        {"title": str((it.get("tag") or it.get("title") or "")[:80]),
         "reply": int(it.get("reply_count", 0) or 0),
         "like": int(it.get("like_count", 0) or 0)}
        for it in he[:10]]
    out["posts_n"] = len(out["posts"])
    hs = (captured.get("hot_stock") or {}).get("data") or {}
    out["stocks"] = [
        {"name": str((q := it.get("quote", it)).get("name", "")),
         "symbol": str(q.get("symbol", "")), "pct": float(it.get("percent", 0) or 0)}
        for it in (hs.get("items") or [])[:10]]
    out["stocks_n"] = len(out["stocks"])
    if not out["posts"] and not out["stocks"]:
        out["err"] = "无捕获(可能cookie失效或页面未自动请求)"
    return out



def collect_md() -> str:
    """采集 → Markdown(复盘/研报用)."""
    r = _collect()
    L = ["## 雪球(登录态高权限)"]
    for p in r.get("posts", [])[:5]:
        L.append(f"- 🔥 {p['title']} (赞{p['like']}/评{p['reply']})")
    for s in r.get("stocks", [])[:6]:
        L.append(f"- 📈 {s['name']} {s['symbol']} {s['pct']:+.2f}%")
    if not r.get("posts") and not r.get("stocks"):
        L.append("- ⚠️ 无数据: " + (r.get("posts_err") or r.get("stocks_err") or "cookie无效"))
    return "\n".join(L)


def save() -> dict:
    """采集+落盘 social 目录(供 social_pulse/研报 读取)."""
    r = _collect()
    SOCIAL.mkdir(parents=True, exist_ok=True)
    (SOCIAL / f"xueqiu_{r['date']}.json").write_text(
        json.dumps(r, ensure_ascii=False), encoding="utf-8")
    return r


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    if "--test" in sys.argv:
        print(_cookie_str()[:20] + "...", file=sys.stderr)
        r = _collect()
        print("hot_posts:", r.get("posts_n", 0), "| stocks:", r.get("stocks_n", 0),
              "| els:", r.get("posts_err", "") or r.get("stocks_err", ""))
        if r.get("posts"):
            print("首帖:", r["posts"][0]["title"][:60])
    elif "--save" in sys.argv:
        r = save()
        print(json.dumps(r, ensure_ascii=False)[:300])
    else:
        print(collect_md())