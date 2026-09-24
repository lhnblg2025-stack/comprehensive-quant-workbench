# -*- coding: utf-8 -*-
"""云端 quant_web 决策链+RAG 接口冒烟测试 (V12.3)"""
import json
import urllib.request

BASE = "http://127.0.0.1:8600"


def get(path, timeout=20):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    ok = True
    # 1. intraday 决策链
    d = get("/api/decision_chain?mode=intraday")
    c = (d.get("data") or {}).get("content") or {}
    print("INTRADAY ok=%s tone=%s rows=%s" % (
        d.get("ok"), c.get("bias", {}).get("tone"), len(c.get("rows") or [])))
    ok = ok and d.get("ok") is True
    # 2. after_close 盘后
    d = get("/api/decision_chain?mode=after_close")
    c = (d.get("data") or {}).get("content") or {}
    lhb = c.get("lhb") or {}
    print("AFTER_CLOSE ok=%s lhb_stock=%s value_picks=%s" % (
        d.get("ok"), len(lhb.get("stock_top") or []), len(c.get("value_picks") or [])))
    ok = ok and d.get("ok") is True
    # 3. rag_search（中文关键词，云端无 sentence_transformers 走关键词兜底）
    d = get("/api/rag_search?q=%E4%BD%8E%E5%90%B8")
    print("RAG ok=%s hits=%s first_file=%s" % (
        d.get("ok"), len(d.get("hits") or []),
        ((d.get("hits") or [{}])[0].get("file") or "")))
    ok = ok and d.get("ok") is True and len(d.get("hits") or []) > 0
    # 4. 首页
    with urllib.request.urlopen(BASE + "/", timeout=15) as r:
        body = r.read()
    print("INDEX ok=%s bytes=%s" % (r.status == 200, len(body)))
    ok = ok and r.status == 200
    print("RESULT=%s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
