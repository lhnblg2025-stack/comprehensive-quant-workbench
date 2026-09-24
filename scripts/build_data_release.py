#!/usr/bin/env python3
"""Build the immutable release for the latest completed market session."""
from __future__ import annotations

import json
from pathlib import Path

from quant_system.data_release import build_release


def _latest_completed_day():
    import quant_system.market_clock as clock
    fn = getattr(clock, "latest_completed_trading_day", None)
    if callable(fn):
        return fn()
    fn = getattr(clock, "latest_trading_day", None)
    if callable(fn):
        from datetime import datetime, timedelta
        now = datetime.now()
        base = fn(now)
        if now.hour < 15 or (now.hour == 15 and now.minute < 30):
            prev = getattr(clock, "prev_trading_day", None)
            if callable(prev):
                return prev(now) or base
        return base
    raise RuntimeError("market_clock has no completed trading day API")


if __name__ == "__main__":
    day = _latest_completed_day().isoformat()
    root = Path(__file__).resolve().parent.parent
    release = build_release(root, day)
    repair = None
    if release["status"] != "PASS":
        from release_repair import repair as repair_release
        repair = repair_release(day, release.get("errors", []))
        release = build_release(root, day)
    print(json.dumps({"release_id": release["release_id"], "status": release["status"], "errors": release["errors"], "warnings": release.get("warnings", []), "repair": repair}, ensure_ascii=False))
    raise SystemExit(0 if release["status"] == "PASS" else 2)
