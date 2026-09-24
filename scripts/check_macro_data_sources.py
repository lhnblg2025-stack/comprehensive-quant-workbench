#!/usr/bin/env python3
"""Health checks for shared macro data-source fallbacks.

Cron report prompts should use these shared source functions instead of
hard-coding URLs or one-off fallback paths in payload text.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable


CHECKS: tuple[tuple[str, str], ...] = (
    ("cpi_ppi", "get_cpi_ppi"),
    ("pmi", "get_pmi"),
    ("customs", "get_customs"),
    ("social_finance_m2", "get_social_finance_m2"),
    ("international_oil_archive", "get_international_oil"),
    ("official_source_registry", "get_official_source_registry"),
    ("opec_momr", "get_opec_momr"),
    ("eia_petroleum_weekly", "get_eia_petroleum_weekly"),
    ("fed_data_sources", "get_fed_data_sources"),
    ("international_oil_sources", "get_international_oil_sources"),
)


def _result_to_dict(result: Any) -> dict[str, Any]:
    if is_dataclass(result):
        return asdict(result)
    if isinstance(result, dict):
        return result
    return {"ok": False, "source": "", "data": None, "note": f"unexpected result type: {type(result)!r}"}


def _run_check(name: str, func: Callable[[], Any]) -> dict[str, Any]:
    try:
        result = _result_to_dict(func())
    except Exception as exc:  # noqa: BLE001 - health check should report all failures.
        return {"name": name, "ok": False, "source": "", "note": repr(exc)}
    return {
        "name": name,
        "ok": bool(result.get("ok")),
        "source": str(result.get("source") or ""),
        "note": str(result.get("note") or ""),
    }


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    data_sources = importlib.import_module("data_sources")
    rows: list[dict[str, Any]] = []
    for name, func_name in CHECKS:
        func = getattr(data_sources, func_name, None)
        if func is None:
            rows.append({"name": name, "ok": False, "source": "", "note": f"missing function: {func_name}"})
            continue
        rows.append(_run_check(name, func))

    for row in rows:
        print(f"{row['name']}|ok={row['ok']}|source={row['source']}|note={row['note']}")

    required = [row for row in rows if row["name"] != "international_oil_archive"]
    return 0 if all(row["ok"] for row in required) else 1


if __name__ == "__main__":
    sys.exit(main())
