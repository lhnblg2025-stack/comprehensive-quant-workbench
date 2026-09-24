"""Strategy Lab API helpers: versioned user strategies plus open-source execution."""
from __future__ import annotations

import ast
import hashlib
import json
import re

import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
STRATEGY_ROOT = ROOT / "generated" / "strategy_lab" / "strategies"
RUN_ROOT = ROOT / "generated" / "strategy_lab" / "runs"
NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{1,48}$")
ALLOWED_IMPORTS = {"pandas", "numpy"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "breakpoint"}
_SHA256_CACHE: dict[str, tuple[int, int, str]] = {}


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=True, default=str))


def _sha256(path: Path) -> str:
    """Stable dataset identity for every reproducible research artifact."""
    stat = path.stat()
    cached = _SHA256_CACHE.get(str(path))
    if cached and cached[:2] == (stat.st_mtime_ns, stat.st_size):
        return cached[2]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest()
    _SHA256_CACHE[str(path)] = (stat.st_mtime_ns, stat.st_size, value)
    return value


def _research_years(mode: str) -> int:
    modes = {"research_5y": 5, "research_10y": 10}
    if mode not in modes:
        raise ValueError("research_mode_must_be_research_5y_or_research_10y")
    return modes[mode]


def _load_quality_report(panel_path: Path) -> dict[str, Any]:
    report_path = panel_path.parent / "pit_data_quality_report.json"
    try:
        return json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    except (OSError, ValueError):
        return {}


def _research_data_gate(panel: pd.DataFrame, state: pd.DataFrame | None, panel_path: Path, *, years: int) -> tuple[pd.DataFrame, pd.Timestamp, dict[str, Any]]:
    """Select a fixed 5/10-year slice and make every data limitation explicit.

    This is deliberately a read-only gate: strategies receive no network client and
    all accepted runs reference the exact Parquet file checksum used here.
    """
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["code"] = data["code"].astype(str).str.zfill(6)
    data = data.dropna(subset=["date", "code", "raw_open", "raw_high", "raw_low", "raw_close"]).sort_values(["code", "date"])
    if data.empty:
        raise ValueError("research_panel_has_no_valid_ohlcv")
    end = pd.Timestamp(data["date"].max()).normalize()
    cutoff = end - pd.DateOffset(years=years)
    warmup_start = cutoff - pd.DateOffset(days=400)
    with_warmup = data[data["date"] >= warmup_start].copy()
    # IPO cooling: a code cannot be ranked until 252 actual daily observations
    # were known. The flag is calculated before test-window slicing to preserve
    # indicator warm-up without allowing a strategy to target pre-window dates.
    with_warmup["_history_sessions"] = with_warmup.groupby("code").cumcount() + 1
    research = with_warmup[(with_warmup["date"] >= cutoff) & (with_warmup["_history_sessions"] >= 252)].copy()
    if research.empty:
        raise ValueError("research_window_empty_after_ipo_cooling")
    counts = research.groupby("date")["code"].nunique()
    source_counts = data[data["date"] >= cutoff].groupby("date")["code"].nunique()
    coverage = (counts / source_counts.reindex(counts.index).replace(0, pd.NA)).dropna()
    min_coverage = float(coverage.min()) if len(coverage) else 0.0
    median_coverage = float(coverage.median()) if len(coverage) else 0.0
    quality = _load_quality_report(panel_path)
    report_status = str(quality.get("research_status") or quality.get("status") or "UNVERIFIED")
    trade_state_fields = {"suspended", "limit_up_locked", "limit_down_locked"}
    state_ready = state is not None and trade_state_fields.issubset(set(state.columns))
    production_blockers = []
    proxy_limitations = []
    if min_coverage < 0.80:
        production_blockers.append("cross_section_coverage_below_80pct")
    if not state_ready:
        production_blockers.append("missing_historical_trade_state")
        proxy_limitations.append("missing_authoritative_trade_state")
    if report_status == "DATA_BLOCKED":
        production_blockers.append("non_authoritative_historical_trade_state")
        proxy_limitations.extend([
            "proxy_limit_rule",
            "unknown_st_state",
            "partial_lifecycle",
            "missing_authoritative_trade_state",
        ])
    proxy_limitations = list(dict.fromkeys(proxy_limitations))
    # Research remains executable with explicit proxies. Production admission
    # remains fail-closed until authoritative PIT data is available.
    admission_eligible = not production_blockers
    gate = {
        "execution_mode": "research_proxy",
        "production_mode": "production_pit",
        "research_executable": True,
        "production_status": "DATA_BLOCKED" if production_blockers else "READY",
        "dataset_path": str(panel_path.relative_to(ROOT)),
        "dataset_sha256": _sha256(panel_path),
        "data_source": "versioned_local_parquet",
        "window": {"years": years, "start": cutoff.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d")},
        "rows": int(len(research)),
        "symbols": int(research["code"].nunique()),
        "ipo_cooling_sessions": 252,
        "coverage": {"minimum": min_coverage, "median": median_coverage, "threshold": 0.80},
        "pit_report_status": report_status,
        "trade_state_available": bool(state_ready),
        "classification": "admission_eligible" if admission_eligible else "research_only",
        "admission_eligible": admission_eligible,
        "production_blockers": production_blockers,
        # Kept as an alias for clients that already consume blockers; these are
        # production blockers and never research execution failures.
        "blockers": production_blockers,
        "proxy_limitations": proxy_limitations,
        "known_limitations": [
            "fixed_500_stock_research_sample_not_point_in_time_full_mainboard",
            "proxy_limit_rule" if "proxy_limit_rule" in proxy_limitations else "verify_trade_state_authority",
            "unknown_st_state" if "unknown_st_state" in proxy_limitations else "st_state_available",
            "partial_lifecycle" if "partial_lifecycle" in proxy_limitations else "lifecycle_available",
            "prices_may_be_adjusted_or_revised; use raw execution fields only",
        ],
    }
    return with_warmup, cutoff, gate


def _validate_source(source: str) -> None:
    if len(source) > 24000:
        raise ValueError("strategy_source_too_large")
    tree = ast.parse(source, mode="exec")
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not any(node.name == "build_targets" for node in functions):
        raise ValueError("strategy_requires_build_targets")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [alias.name.split(".")[0] for alias in node.names]
            if isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module.split(".")[0]]
            if any(module not in ALLOWED_IMPORTS for module in modules):
                raise ValueError("strategy_import_not_allowed")
        if isinstance(node, (ast.Call, ast.Attribute)) and isinstance(getattr(node, "func", None), ast.Name) and node.func.id in FORBIDDEN_NAMES:
            raise ValueError("strategy_dynamic_execution_not_allowed")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise ValueError("strategy_forbidden_name")
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise ValueError("strategy_construct_not_allowed")
    fn = next(node for node in functions if node.name == "build_targets")
    if len(fn.args.args) < 1 or len(fn.args.args) > 2:
        raise ValueError("build_targets_signature_must_be_panel_params")


def _strategy_path(name: str) -> Path:
    if not NAME_RE.fullmatch(name):
        raise ValueError("invalid_strategy_name")
    return STRATEGY_ROOT / f"{name}.py"


def _execute_strategy_source(source: str, origin: str) -> Any:
    """Compile a validated user/template strategy in the same restricted runtime."""
    _validate_source(source)

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name.split(".")[0] not in ALLOWED_IMPORTS:
            raise ImportError("strategy_import_not_allowed")
        return __import__(name, globals, locals, fromlist, level)

    namespace: dict[str, Any] = {
        "__builtins__": {
            "len": len, "range": range, "min": min, "max": max, "float": float,
            "int": int, "abs": abs, "round": round, "sorted": sorted,
            "enumerate": enumerate, "zip": zip, "__import__": safe_import,
        }
    }
    exec(compile(source, origin, "exec"), namespace, namespace)
    return namespace["build_targets"]


def _template_lifecycle() -> dict[str, dict[str, Any]]:
    """Merge full-data target audit and bounded real-execution smoke results."""
    full_path = ROOT / "generated" / "strategy_lab" / "template_full_dataset_audit.json"
    smoke_path = ROOT / "generated" / "strategy_lab" / "template_execution_smoke.json"
    execution_path = ROOT / "generated" / "strategy_lab" / "template_full_execution.json"
    try:
        full = json.loads(full_path.read_text(encoding="utf-8")).get("templates", {})
    except (OSError, ValueError, TypeError):
        full = {}
    try:
        smoke = json.loads(smoke_path.read_text(encoding="utf-8")).get("templates", {})
    except (OSError, ValueError, TypeError):
        smoke = {}
    try:
        execution = json.loads(execution_path.read_text(encoding="utf-8")).get("templates", {})
    except (OSError, ValueError, TypeError):
        execution = {}
    merged: dict[str, dict[str, Any]] = {}
    for name in set(full) | set(smoke) | set(execution):
        f, s, e = full.get(name, {}), smoke.get(name, {}), execution.get(name, {})
        merged[name] = {
            "compilable": bool(f.get("compilable", s.get("compilable", False))),
            "full_dataset_target_generation": bool(f.get("full_dataset_target_generation", False)),
            "full_dataset_comparable": bool(f.get("full_dataset_comparable", False)),
            "execution_smoke": bool(s.get("executable", False)),
            "full_execution": bool(e.get("full_execution", False)),
            "execution_status": e.get("status"),
            "execution_smoke_scope": s.get("classification"),
            "validated": False,
            "admitted": False,
            "error": e.get("error") or f.get("error") or s.get("error"),
            "target_coverage": f.get("target_coverage"),
            "artifact": "generated/strategy_lab/template_full_execution.json" if e else "generated/strategy_lab/template_full_dataset_audit.json",
        }
    return merged


def list_strategies() -> list[dict[str, Any]]:
    STRATEGY_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(STRATEGY_ROOT.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        rows.append({"name": path.stem, "bytes": len(source.encode()), "updated_at": datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds"), "valid": _validate_or_false(source)})
    return rows


def _validate_or_false(source: str) -> bool:
    try:
        _validate_source(source)
        return True
    except Exception:
        return False


def save_strategy(body: dict[str, Any]) -> dict[str, Any]:
    name = str(body.get("name") or "").strip()
    source = str(body.get("source") or "")
    path = _strategy_path(name)
    _validate_source(source)
    STRATEGY_ROOT.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return {"name": name, "path": str(path.relative_to(ROOT)), "valid": True, "updated_at": datetime.now().astimezone().isoformat(timespec="seconds")}


def read_strategy(name: str) -> dict[str, Any]:
    path = _strategy_path(name)
    if not path.is_file():
        raise FileNotFoundError("strategy_not_found")
    source = path.read_text(encoding="utf-8")
    _validate_source(source)
    return {"name": name, "source": source, "path": str(path.relative_to(ROOT))}


def run_strategy_job(params: dict[str, Any]) -> dict[str, Any]:
    name = str(params.get("name") or "").strip()
    spec = read_strategy(name)
    asset_type = str(params.get("asset_type") or "stock")
    years = _research_years(str(params.get("mode") or "research_10y"))
    panel, state, panel_path = _load_asset_panel(asset_type)
    signal_panel, cutoff, data_gate = _research_data_gate(panel, state, panel_path, years=years)

    build_targets = _execute_strategy_source(spec["source"], str(_strategy_path(name)))
    all_targets = build_targets(signal_panel, dict(params.get("strategy_params") or {}))
    if not isinstance(all_targets, dict):
        raise ValueError("build_targets_must_return_date_to_targets_dict")
    # Indicator warm-up is visible to the factor calculation, but no order may be
    # generated before the sealed research window. This prevents warm-up history
    # from being counted as in-sample performance.
    targets = {pd.Timestamp(day): weights for day, weights in all_targets.items() if pd.Timestamp(day) >= cutoff}
    if not targets:
        raise ValueError("no_targets_after_research_window_and_ipo_cooling")
    execution_panel = signal_panel[signal_panel["date"] >= cutoff].copy()
    execution_state = None
    if state is not None:
        execution_state = state.copy()
        execution_state["date"] = pd.to_datetime(execution_state["date"], errors="coerce")
        execution_state = execution_state[execution_state["date"] >= cutoff]
    from quant_system.backtest_protocol import BacktestRequest
    from quant_system.backtest_service import execute
    request = BacktestRequest.from_dict({
        "strategy": {"kind": "target_weights", "family": "user_strategy", "parameters": dict(params.get("strategy_params") or {})},
        "costs": dict(params.get("costs") or {}),
        "risk": dict(params.get("risk") or {}),
        "initial_capital": params.get("initial_capital", 1_000_000.0),
        "mode": "research",
        "dataset_id": data_gate["dataset_sha256"],
        "request_id": params.get("request_id"),
    })
    run = execute(request, execution_panel, trade_state=execution_state, targets=targets,
                  quality_path=panel_path.parent / "pit_data_quality_report.json")
    if not run.get("ok"):
        raise ValueError(run.get("error") or "canonical_backtest_failed")
    result, equity, trades = run["result"], run["equity"], run["trades"]
    curve, benchmark = _attach_benchmark(equity, asset_type)
    metrics = result.to_dict()
    target_payload = {pd.Timestamp(day).strftime("%Y-%m-%d"): {str(code): float(weight) for code, weight in sorted(weights.items())} for day, weights in sorted(targets.items(), key=lambda item: pd.Timestamp(item[0]))}
    target_sha256 = hashlib.sha256(json.dumps(target_payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    strategy_sha256 = hashlib.sha256(spec["source"].encode()).hexdigest()
    run_scope = {"start": str(execution_panel["date"].min().date()), "end": str(execution_panel["date"].max().date()), "sessions": int(execution_panel["date"].nunique()), "symbols": int(execution_panel["code"].nunique()), "target_dates": len(targets)}
    turnover = float(trades.loc[~trades.get("blocked", pd.Series(False, index=trades.index)).astype(bool), "notional"].fillna(0.0).sum() / max(1.0, len(equity) * result.initial_capital)) if len(trades) else 0.0
    metrics["daily_turnover"] = turnover
    metrics["admission_eligible"] = data_gate["admission_eligible"]
    metrics["classification"] = data_gate["classification"]
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    artifact = RUN_ROOT / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    payload = {"schema": run["schema"], "canonical": True, "spec_hash": run["spec_hash"], "spec": run["spec"], "engine": run["engine"], "strategy": name, "strategy_sha256": strategy_sha256, "asset_type": asset_type, "run_scope": run_scope, "universe_symbols": int(execution_panel["code"].nunique()), "data_gate": data_gate, "execution_gate": run["gate"], "strategy_params": dict(params.get("strategy_params") or {}), "target_sha256": target_sha256, "benchmark": benchmark, "metrics": metrics, "equity_curve": curve.to_dict("records"), "order_events": trades.to_dict("records"), "target_dates": len(targets), "generated_at": datetime.now().astimezone().isoformat(timespec="seconds")}
    artifact.write_text(json.dumps(payload, ensure_ascii=True, default=str), encoding="utf-8")
    return {"artifact": str(artifact.relative_to(ROOT)), "schema": run["schema"], "canonical": True, "spec_hash": run["spec_hash"], "spec": run["spec"], "strategy_sha256": strategy_sha256, "target_sha256": target_sha256, "run_scope": run_scope, "metrics": metrics, "benchmark": benchmark, "data_gate": data_gate, "execution_gate": run["gate"], "equity_curve": curve.to_dict("records"), "rows": len(equity), "order_events": len(trades), "order_event_rows": trades.to_dict("records"), "target_dates": len(targets), "universe_symbols": int(execution_panel["code"].nunique()), "engine": run["engine"]}


def _load_asset_panel(asset_type: str):
    if asset_type == "etf":
        source = ROOT / "data_warehouse" / "events" / "etf_history.parquet"
        raw = pd.read_parquet(source)
        panel = raw.rename(columns={"price": "raw_close", "open": "raw_open", "high": "raw_high", "low": "raw_low"}).copy()
        panel["raw_close"] = pd.to_numeric(panel["raw_close"], errors="coerce")
        panel["amount"] = pd.to_numeric(panel["amount"], errors="coerce").fillna(0.0)
        panel["volume"] = pd.to_numeric(panel["volume"], errors="coerce").fillna(0.0)
        panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
        panel["code"] = panel["code"].astype(str).str.zfill(6)
        panel = panel.dropna(subset=["date", "raw_open", "raw_high", "raw_low", "raw_close"]).sort_values(["date", "code"])
        return panel, None, source
    from quant_web.handlers.backtest_console import _load_panel
    panel, state, source = _load_panel("500")
    # Do NOT filter with today's ST/name universe: that would remove historical
    # names using information unavailable on each historical rebalance date.
    # The fixed 500-symbol panel is retained intact and the absence of
    # authoritative historical ST state is carried as a visible admission block.
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    # Board membership is encoded in the instrument code and is valid historical
    # classification here; unlike a present-day ST flag it does not leak future
    # survival information into prior rebalance dates.
    panel = panel[panel["code"].str.startswith(("00", "60"))].copy()
    if state is not None:
        state["code"] = state["code"].astype(str).str.zfill(6)
        state = state[state["code"].isin(set(panel["code"]))].copy()
    return panel, state, source


def _attach_benchmark(equity: pd.DataFrame, asset_type: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    curve = equity.copy()
    curve["date"] = pd.to_datetime(curve["date"], errors="coerce")
    curve["strategy_nav"] = curve["total_value"] / curve["total_value"].iloc[0]
    if asset_type == "etf":
        names = ["沪深300"]
    else:
        names = ["上证指数", "深证成指"]
    series = []
    for name in names:
        frame = pd.read_parquet(ROOT / "data_warehouse" / "market" / f"index_daily_{name}.parquet", columns=["date", "close"])
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        series.append(frame.drop_duplicates("date").set_index("date")["close"].astype(float).rename(name))
    prices = pd.concat(series, axis=1).sort_index().ffill()
    daily = prices.pct_change().mean(axis=1).fillna(0.0)
    benchmark_nav = (1.0 + daily).cumprod()
    curve["benchmark_nav"] = curve["date"].map(benchmark_nav).ffill().bfill()
    curve["benchmark_nav"] = curve["benchmark_nav"] / curve["benchmark_nav"].iloc[0]
    curve["excess_nav"] = curve["strategy_nav"] / curve["benchmark_nav"]
    curve["drawdown"] = curve["strategy_nav"] / curve["strategy_nav"].cummax() - 1.0
    curve["date"] = curve["date"].dt.strftime("%Y-%m-%d")
    return curve, {"name": "沪深300" if asset_type == "etf" else "上证指数+深证成指等权", "components": names, "weighting": "equal_weight_daily_returns", "total_return": float(curve["benchmark_nav"].iloc[-1] - 1.0)}


def default_source() -> str:
    return '''import pandas as pd\n\ndef build_targets(panel, params):\n    """Return {signal_date: {code: target_weight}}. Signal at close, fill next open."""\n    top_n = int(params.get("top_n", 20))\n    rebalance = int(params.get("rebalance_sessions", 5))\n    data = panel.copy()\n    data["score"] = data.groupby("code")["raw_close"].pct_change(20)\n    dates = sorted(data["date"].dropna().unique())\n    targets = {}\n    for day in dates[::max(1, rebalance)]:\n        frame = data[data["date"] == day].dropna(subset=["score"])\n        picks = frame.sort_values("score", ascending=False).head(top_n)["code"].tolist()\n        if picks:\n            targets[pd.Timestamp(day)] = {code: 1.0 / len(picks) for code in picks}\n    return targets\n'''


def handler_admission(query: dict, send_json) -> None:
    """Return the immutable, fail-closed admission report for review."""
    path = ROOT / "generated" / "strategy_lab" / "admission" / "admission_report.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        send_json({"ok": False, "error": "admission_report_missing"}, status=404)
        return
    catalog_path = ROOT / "data_warehouse" / "research_panels" / "ashare_tushare_500_10y_v2" / "pit_source_catalog.json"
    try:
        source_catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        source_catalog = {"status": "missing"}
    send_json({"ok": True, "report": report, "source_catalog": source_catalog})


def handler_data_status(query: dict, send_json) -> None:
    """Expose actual 5/10-year data availability before a researcher runs code."""
    asset_type = str((query.get("asset_type") or ["stock"])[0])
    panel, state, panel_path = _load_asset_panel(asset_type)
    reports = {}
    for years in (5, 10):
        _, _, gate = _research_data_gate(panel, state, panel_path, years=years)
        reports[f"research_{years}y"] = gate
    full_execution = {"available": False}
    execution_path = ROOT / "generated" / "strategy_lab" / "template_full_execution.json"
    try:
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        full_execution = {
            "available": True,
            "artifact": str(execution_path.relative_to(ROOT)),
            "summary": execution.get("summary", {}),
            "scope": execution.get("scope", {}),
        }
    except (OSError, ValueError, TypeError):
        pass
    # Historical vectorized/OOS artifacts remain on disk for audit only. They are
    # intentionally excluded here because they did not execute frontend source
    # through the canonical order engine and must not rate visible templates.
    send_json({"ok": True, "asset_type": asset_type, "reports": reports, "full_execution": full_execution, "ranking": {"enabled": False, "reason": "no_template_has_passed_pit_oos_multiple_testing_and_capacity_gates"}})


def handler_list(query: dict, send_json) -> None:
    from quant_web.handlers.strategy_templates import template_catalog
    lifecycle = _template_lifecycle()
    templates = []
    for item in template_catalog():
        state = lifecycle.get(item["id"], {})
        templates.append({
            **item,
            "lifecycle": {
                "compilable": bool(state.get("compilable", False)),
                "full_dataset_target_generation": bool(state.get("full_dataset_target_generation", False)),
                "full_dataset_comparable": bool(state.get("full_dataset_comparable", False)),
                "execution_smoke": bool(state.get("execution_smoke", False)),
                "full_execution": bool(state.get("full_execution", False)),
                "execution_status": state.get("execution_status"),
                "validated": False,
                "admitted": False,
                "error": state.get("error"),
                "target_coverage": state.get("target_coverage"),
                "artifact": state.get("artifact"),
            },
        })
    send_json({"ok": True, "strategies": list_strategies(), "templates": templates, "default_source": default_source(), "template_smoke_artifact": "generated/strategy_lab/template_execution_smoke.json"})


def handler_template(query: dict, send_json) -> None:
    from quant_web.handlers.strategy_templates import TEMPLATES
    template_id = (query.get("id") or [""])[0]
    if template_id not in TEMPLATES:
        send_json({"ok": False, "error": "template_not_found"}, status=404)
        return
    send_json({"ok": True, "template": {"id": template_id, **TEMPLATES[template_id]}})


def handler_get(query: dict, send_json) -> None:
    name = (query.get("name") or [""])[0]
    send_json({"ok": True, "strategy": read_strategy(name)})


def handler_save(body: dict, send_json) -> None:
    send_json({"ok": True, "strategy": save_strategy(body)})


def handler_run(body: dict, send_json) -> None:
    from quant_system.task_queue import submit_task
    name = str(body.get("name") or "").strip()
    read_strategy(name)
    task_id = submit_task("strategy_lab_run", body, max_retries=0)
    send_json({"ok": True, "accepted": True, "task_id": task_id, "strategy": name})
