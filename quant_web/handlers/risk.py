"""
Fama-MacBeth & 风险模型 handlers — 3.0
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from datetime import datetime
from typing import Any, Callable

from quant_system.cache import cached, cache_stats, cache_clear
from quant_system.task_queue import submit_task, get_task_result, register_handler

logger = logging.getLogger("handlers.fama_macbeth")

# ── FM 计算 handler（注册到任务队列） ──

def _compute_fama_macbeth(params: dict) -> dict:
    """Run Fama-MacBeth computation (used by task queue)."""
    from quant_system.fama_macbeth import get_fama_macbeth
    from quant_system.data_store import get_store
    symbols = params.get("symbols", [])
    step = params.get("step_days", 10)
    start_date = params.get("start_date", "2026-05-01")

    # Pre-warm data — each symbol needs 180+ days
    store = get_store()
    for sym in symbols:
        df = store.get(sym, days=250)
        if df is None or len(df) < 20:
            # Force fetch from SQLite
            import pandas as pd
            from quant_system.data import fetch_daily
            try:
                raw = fetch_daily(sym, start="2026-01-01")
                if raw is not None and not raw.empty:
                    store.save(sym, raw)
            except Exception as e:
                logger.error(f"[risk] 操作失败: {e}", exc_info=True)

    fm = get_fama_macbeth()
    result = fm.run(symbols=symbols, start_date=start_date, step_days=step)
    return result


register_handler("fama_macbeth", _compute_fama_macbeth)


def _compute_asset_allocation(params: dict) -> dict:
    from quant_system.asset_allocation import get_allocator
    symbols = params.get("symbols", [])
    objective = params.get("objective", "max_sharpe")
    alloc = get_allocator()
    return alloc.allocation_report(symbols, objective=objective)


register_handler("asset_allocation", _compute_asset_allocation)


def _compute_risk_model(params: dict) -> dict:
    from quant_system.factor_model import get_model
    from quant_system.data_store import get_store
    symbols = params.get("symbols", [])
    date = params.get("date", datetime.now().strftime("%Y-%m-%d"))
    store = get_store()
    for sym in symbols:
        store.get(sym, days=250)
    model = get_model()
    return model.risk_model(date, symbols)


register_handler("risk_model", _compute_risk_model)


# ── HTTP handlers ──

def handle_fama_macbeth(query: dict[str, list[str]], send_json: Callable) -> None:
    """GET /api/fama_macbeth"""
    symbols_str = query.get("symbols", ["600519,000858,002714,601899,002594,300750,600036,601318,000333,600276,000568,002415"])[0]
    symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
    step = int(query.get("step_days", ["10"])[0])
    use_async = query.get("async", ["true"])[0].lower() != "false"

    if use_async:
        # Submit to task queue, return task_id
        task_id = submit_task("fama_macbeth", {
            "symbols": symbols,
            "step_days": step,
            "start_date": "2026-05-01",
        })
        send_json({"ok": True, "data": {"task_id": task_id, "status": "pending"}})
        return

    # Synchronous (original path, with cache)
    try:
        from quant_system.fama_macbeth import get_fama_macbeth
        from quant_system.data_store import get_store

        # Pre-warm
        store = get_store()
        for sym in symbols:
            store.get(sym, days=120)

        fm = get_fama_macbeth()
        result = fm.run(symbols=symbols, start_date="2026-05-01", step_days=step)
        send_json({"ok": True, "data": _json_safe(result)})
    except Exception as e:
        send_json({"ok": False, "error": str(e)[:300] + " | " + traceback.format_exc()[-200:]}, status=500)


def handle_risk_model(query: dict[str, list[str]], send_json: Callable) -> None:
    """GET /api/v3/risk_model（默认异步任务队列，返回 task_id）"""
    symbols_str = query.get("symbols", ["600519,000858,002714,601899,002594,300750,600036,601318,000333,600276,000568,002415,000001,601166,600900,600887"])[0]
    symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
    date = query.get("date", [datetime.now().strftime("%Y-%m-%d")])[0]
    use_async = query.get("async", ["true"])[0].lower() != "false"

    if use_async:
        task_id = submit_task("risk_model", {"symbols": symbols, "date": date})
        send_json({"ok": True, "data": {"task_id": task_id, "status": "pending"}})
        return

    try:
        from quant_system.factor_model import get_model
        from quant_system.data_store import get_store

        store = get_store()
        for sym in symbols:
            store.get(sym, days=250)

        model = get_model()
        result = model.risk_model(date, symbols)
        send_json({"ok": True, "data": _json_safe(result)})
    except Exception as e:
        send_json({"ok": False, "error": str(e)[:300]}, status=500)


def handle_asset_allocation(query: dict[str, list[str]], send_json: Callable) -> None:
    """GET /api/asset_allocation（默认异步任务队列，返回 task_id）"""
    symbols_str = query.get("symbols", ["600519,000858,002714,601899,002594,300750,600036,601318,000333,600276"])[0]
    symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
    objective = query.get("objective", ["max_sharpe"])[0]
    use_async = query.get("async", ["true"])[0].lower() != "false"

    if use_async:
        task_id = submit_task("asset_allocation", {"symbols": symbols, "objective": objective})
        send_json({"ok": True, "data": {"task_id": task_id, "status": "pending"}})
        return

    try:
        from quant_system.asset_allocation import get_allocator
        from quant_system.data_store import get_store

        # Pre-warm
        store = get_store()
        for sym in symbols:
            store.get(sym, days=250)

        alloc = get_allocator()
        result = alloc.allocation_report(symbols, objective=objective)
        send_json({"ok": True, "data": _json_safe(result)})
    except Exception as e:
        send_json({"ok": False, "error": str(e)[:300]}, status=500)


def handle_cache_status(send_json: Callable) -> None:
    """GET /api/cache/status — cache + task queue status"""
    try:
        from quant_system.data_store import get_store

        ds = get_store()
        cache_dir = Path(__file__).resolve().parents[2] / "generated" / "a_share_data"
        files = list(cache_dir.glob("*.json")) if cache_dir.exists() else []
        latest = max(files, key=lambda p: p.stat().st_mtime) if files else None
        total_size = sum(p.stat().st_size for p in files)

        cs = cache_stats()
        tasks = []
        try:
            from quant_system.task_queue import list_tasks as lt
            tasks = lt(limit=10)
        except Exception as e:
            logger.error(f"[risk] 操作失败: {e}", exc_info=True)

        send_json({"ok": True, "data": {
            "data_store": ds.status(),
            "cache": cs,
            "generated_files": {
                "count": len(files),
                "latest": latest.name if latest else None,
                "latest_date": datetime.fromtimestamp(latest.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S") if latest else None,
                "total_mb": round(total_size / 1024 / 1024, 2),
            },
            "tasks": tasks,
        }})
    except Exception as e:
        send_json({"ok": True, "data": {"error": str(e)}})


def handle_task_result(query: dict[str, list[str]], send_json: Callable) -> None:
    """GET /api/v3/task/<id>"""
    task_id = query.get("id", [""])[0] or (query.get("task_id", [""])[0] if "task_id" in query else "")
    if not task_id:
        send_json({"ok": False, "error": "task_id required"}, status=400)
        return
    result = get_task_result(task_id)
    if result is None:
        send_json({"ok": True, "data": {"status": "pending"}})
    else:
        send_json({"ok": True, "data": result})


def handle_v3_status(send_json: Callable) -> None:
    """GET /api/v3/status — 3.0 系统健康状态"""
    now = datetime.now()
    cs = cache_stats()
    send_json({"ok": True, "data": {
        "version": "3.0",
        "time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "uptime_hours": None,  # set by server
        "cache": cs,
        "data_freshness": "normal",
    }})


from pathlib import Path


def _json_safe(obj: Any) -> Any:
    """Recursively convert non-JSON-safe types."""
    import numpy as _np
    import pandas as _pd
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    elif isinstance(obj, (_np.integer,)):
        return int(obj)
    elif isinstance(obj, (_np.floating,)):
        return float(obj)
    elif isinstance(obj, (_np.ndarray,)):
        return _json_safe(obj.tolist())
    elif isinstance(obj, (_pd.Timestamp,)):
        return str(obj)[:19]
    elif isinstance(obj, (_pd.Series,)):
        return _json_safe(obj.to_dict())
    elif isinstance(obj, (_pd.DataFrame,)):
        return _json_safe(obj.to_dict(orient="records"))
    elif isinstance(obj, (float, int, str, bool, type(None))):
        return obj
    else:
        try:
            return str(obj)
        except Exception:
            return None
