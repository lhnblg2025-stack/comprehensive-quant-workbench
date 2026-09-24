#!/usr/bin/env python3
"""检查三条行情链路，不写入生产行情目录。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_system.market_data_providers import (
    AkshareProvider,
    EastmoneyProvider,
    ProviderConfig,
    YahooChartProvider,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="000001")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-09-20")
    args = ap.parse_args()
    checks = []
    for name, provider, symbol in (
        ("cloud_eastmoney", EastmoneyProvider(), args.code),
        ("local_akshare", AkshareProvider(), args.code),
        ("yahoo_chart", YahooChartProvider(), f"{args.code}.SZ"),
    ):
        try:
            frame = provider.fetch(symbol, args.start, args.end, "raw")
            checks.append({"provider": name, "ok": not frame.empty, "rows": len(frame), "start": str(frame["date"].min()) if len(frame) else None, "end": str(frame["date"].max()) if len(frame) else None})
        except Exception as exc:
            checks.append({"provider": name, "ok": False, "error": str(exc)})
    print(json.dumps({"code": args.code, "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if any(item["ok"] for item in checks) else 2


if __name__ == "__main__":
    raise SystemExit(main())
