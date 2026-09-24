#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build an inquiry/regulatory-letter risk index from cninfo announcement dumps.

The dedicated akshare cninfo_letters endpoint returns empty, but the daily
cninfo announcement files already carry 公告类型/公告标题. This scan turns
them into generated/inquiry_letters.json so the daily report can show real
regulatory-risk events instead of a degraded blank.
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CNINFO = ROOT / "data_warehouse" / "cninfo"
OUT = ROOT / "generated" / "inquiry_letters.json"
KEYWORDS = ("问询", "关注函", "监管函", "警示函", "监管措施", "立案", "处罚")


def tier_of(typ: str, title: str) -> int:
    """1=硬风险(立案/处罚/警示/监管措施/风险提示) 2=交易所问询 3=融资审核问询回复。"""
    text = typ + title
    if any(k in text for k in ("立案", "处罚", "警示函", "监管措施", "监管函", "关注函", "终止上市", "风险提示")):
        return 1
    if "问询函" in text:
        return 3 if "审核问询" in text or "问询函的回复" in text or "问询函回复" in text else 2
    if "问询" in text:
        return 2
    return 3



def main() -> int:
    rows = []
    for f in sorted(CNINFO.glob("cninfo_*.json")):
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        items = payload if isinstance(payload, list) else (payload.get("items") or payload.get("data") or [])
        for x in items:
            if not isinstance(x, dict):
                continue
            typ = str(x.get("公告类型") or "")
            title = str(x.get("公告标题") or "")
            if not any(k in typ + title for k in KEYWORDS):
                continue
            rows.append({
                "date": str(x.get("公告日期") or x.get("date") or "")[:10],
                "code": str(x.get("代码") or "").zfill(6),
                "name": str(x.get("名称") or ""),
                "type": typ,
                "title": title,
                "url": str(x.get("网址") or ""),
                "tier": tier_of(typ, title),
            })
    tier_names = {1: "硬风险", 2: "交易所问询", 3: "融资审核问询回复"}
    by_code = defaultdict(list)
    for r in rows:
        by_code[r["code"]].append(r)
    tier_counts = {tier_names[t]: sum(1 for r in rows if r["tier"] == t) for t in (1, 2, 3)}
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "source": "data_warehouse/cninfo/cninfo_*.json (公告类型+标题关键词分级)",
        "total_hits": len(rows),
        "tier_counts": tier_counts,
        "codes": len(by_code),
        "top_codes": sorted(
            ({"code": k, "name": v[0]["name"], "count": len(v), "tier1": sum(1 for x in v if x["tier"] == 1),
              "latest": max(x["date"] for x in v), "recent": v[:3]} for k, v in by_code.items()),
            key=lambda r: (-r["tier1"], -r["count"]), )[:30],
        "recent": sorted(rows, key=lambda r: r["date"], reverse=True)[:50],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("total_hits", "codes")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
