# -*- coding: utf-8 -*-
"""数据基座全景 API handler（quant_web）

GET /api/basepanorama → 全部数据基座目录: 文件数/最新日期/滞后/评级/样例表
复用 freshness_gate.scan_all + 目录文件统计, 全本地轻量秒级。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

_CACHE: dict = {"ts": 0.0, "data": None, "computing": False}
_TTL = 60.0


def _build():
    import os
    sys.path.insert(0, str(ROOT / "scripts"))
    from freshness_gate import scan_all

    dw = ROOT / "data_warehouse"
    scan = scan_all()
    detail = {}
    for d, info in scan.items():
        dp = dw / d
        files = sorted(dp.glob("*.parquet")) if dp.exists() else []
        sample = ""
        if files:
            sample = ", ".join(f.name.replace(".parquet", "") for f in files[:3])
        detail[d] = {**info, "files": len(files), "sample": sample}
    order = sorted(detail.items(), key=lambda x: ((x[1].get("lag") if x[1].get("lag") is not None else 10**9), x[1].get("level") != "fresh"))
    # 同时输出数据契约结果：目录存在不等于短线链实际可用。
    contracts = []
    try:
        from data_contract_registry import run_check
        contract_summary = run_check(verbose=False)
        contracts = contract_summary.get("results", [])
    except Exception as exc:  # noqa: BLE001
        contract_summary = {"total": 0, "ok": 0, "degraded": 0, "error": str(exc)[:160]}
    return {"ok": True, "data": {
        "total": len(detail),
        "dirs": [{"name": k, **v} for k, v in order],
        "contracts": contracts,
        "contract_summary": {k: contract_summary.get(k, 0) for k in ("total", "ok", "degraded", "missing", "stale")},
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
    }}


def handler_basepanorama(query: dict, send_json):
    import threading
    import time
    now = time.time()
    if _CACHE["data"] is not None and now - _CACHE["ts"] < _TTL:
        send_json(_CACHE["data"])
        return
    if _CACHE["computing"]:
        send_json({"ok": True, "status": "computing", "hint": "数据基座全景扫描中，请稍后刷新"})
        return
    _CACHE["computing"] = True

    def _bg():
        try:
            _CACHE["data"] = _build()
        except Exception as e:  # noqa: BLE001
            _CACHE["data"] = {"ok": False, "error": str(e)[:150]}
        _CACHE["ts"] = time.time()
        _CACHE["computing"] = False

    threading.Thread(target=_bg, daemon=True).start()
    send_json({"ok": True, "status": "computing", "hint": "数据基座全景扫描已启动，请稍后刷新"})