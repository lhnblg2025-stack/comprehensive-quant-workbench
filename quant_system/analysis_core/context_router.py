"""context_router — 场景路由 ContextRouter（V12.1 阶段3 域N）。

输入市场风格象限 + 日期，输出今日应加载的数据源集合：
  global 源全量加载；命中场景的行业源追加加载。

用法:
  python3 -m quant_system.analysis_core.context_router --context 小盘 --date 2026-08-13
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Mapping

CST = timezone(__import__("datetime").timedelta(hours=8))

# 市场风格象限 → 激活场景 scope（global 恒加载）
QUADRANT_SCOPES: dict[str, list[str]] = {
    "大盘": ["global", "chain_realestate"],
    "小盘": ["global", "chain_auto"],
    "周期": ["global", "chain_cycle"],
    "消费": ["global", "chain_auto", "chain_consumer"],
}


def _today() -> str:
    return datetime.now(CST).strftime("%Y-%m-%d")


def _normalize_scopes(scopes: Any) -> list[str]:
    if isinstance(scopes, str):
        out = [scopes]
    elif isinstance(scopes, (list, tuple, set)):
        out = [str(x) for x in scopes]
    else:
        out = ["global"]
    out = [x for x in out if x]
    return out or ["global"]


def _default_registry() -> dict[str, dict]:
    from quant_system.analysis_core.data_sources import SOURCE_STATUS
    return SOURCE_STATUS


def route_sources(
    context: str | None = None,
    date: str | None = None,
    registry: Mapping[str, Mapping[str, Any]] | None = None,
) -> set[str]:
    """返回应加载的数据源名称集合。

    无象限/空象限 → 只加载 global 源。api_enabled=False 的源不参与加载。
    """
    _ = date
    registry = registry if registry is not None else _default_registry()
    context_scopes = QUADRANT_SCOPES.get(context or "", ["global"])
    active_scopes = {s for s in context_scopes if s != "global"}
    selected: set[str] = set()

    for name, info in registry.items():
        if info.get("api_enabled") is False:
            continue
        scopes = set(_normalize_scopes(info.get("scope")))
        if "global" in scopes or (active_scopes and active_scopes & scopes):
            selected.add(name)
    return sorted(selected)


def load_for_context(
    context: str | None = None,
    date: str | None = None,
    registry: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict:
    """场景路由主接口：返回可序列化加载清单。"""
    date = date or _today()
    registry = registry if registry is not None else _default_registry()
    context_scopes = QUADRANT_SCOPES.get(context or "", ["global"])
    active_scopes = sorted({s for s in context_scopes if s != "global"})
    names = sorted(route_sources(context, date, registry))
    rows: list[dict] = []
    matched: list[str] = []
    global_names: list[str] = []

    for name in names:
        info = registry.get(name, {})
        scopes = _normalize_scopes(info.get("scope"))
        is_global = "global" in scopes
        has_context = bool(set(active_scopes) & set(scopes))
        reason = "context" if (has_context and not is_global) else "global"
        rows.append({
            "source": name,
            "desc": info.get("desc", ""),
            "status": info.get("status", "active"),
            "scope": scopes,
            "reason": reason,
        })
        if reason == "context":
            matched.append(name)
        if is_global:
            global_names.append(name)

    return {
        "context": context if context else "default_global",
        "date": date,
        "scopes": ["global"] + active_scopes,
        "source_names": names,
        "count": len(names),
        "matched": matched,
        "global": global_names,
        "sources": rows,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="场景路由 ContextRouter")
    ap.add_argument("--context", default=None, help="市场风格象限：大盘/小盘/周期/消费")
    ap.add_argument("--date", default=None, help="目标日期 YYYY-MM-DD")
    args = ap.parse_args()
    print(json.dumps(load_for_context(args.context, args.date), ensure_ascii=False,
                     indent=2, default=str))
