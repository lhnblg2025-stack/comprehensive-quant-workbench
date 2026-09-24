"""Integrated backtest console handlers for quant_web.

Endpoints:
  GET /api/btc/overview   — data gate, engines, research-pool snapshot.
  GET /api/btc/strategies — merged predeclared strategy/challenger registry.
  GET /api/btc/run        — run one signal through the portfolio holding engine,
                            optionally cross-check a bounded Backtrader window.

These handlers are read-only except /api/btc/run, which writes results under
generated/runs/web_backtest_console/.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = ROOT / "data_warehouse" / "research_panels" / "ashare_tushare_500_10y_v2"
PANEL_PATH = DATA_ROOT / "price_discovery_panel.parquet"
TRADE_STATE_PATH = DATA_ROOT / "trade_state.parquet"
OUTPUT_ROOT = ROOT / "generated" / "runs" / "web_backtest_console"

DEFAULT_SIGNALS = (
    "short_reversal_5", "short_reversal_20", "low_vol_60",
    "downside_vol_60", "volatility_120", "range_compression",
    "mom_12_1", "mom_6_1", "mom_3_1", "rsi_reversal_14",
    "gap_reversal", "price_efficiency_20", "amihud_inverse",
    "breakout_252", "volume_ratio_20", "trend_60", "trend_120",
)


def _load_json(path: Path) -> dict | list | None:
    try:
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _gate() -> dict:
    report = _load_json(DATA_ROOT / "pit_data_quality_report.json") or {}
    return {
        "status": report.get("research_status", report.get("status", "unknown")),
        "production_status": report.get("production_status", report.get("status", "unknown")),
        "errors": report.get("production_blockers", report.get("errors", [])),
        "research_errors": report.get("errors", []),
        "coverage": report.get("coverage", {}),
        "next_action": report.get("next_action", ""),
    }


def _result_summary(prefix: str) -> dict:
    root = OUTPUT_ROOT.parent
    summary = {"available": False, "artifacts": []}
    for name in (f"{prefix}.json", f"{prefix}/result.json"):
        candidate = root / name
        if candidate.is_file():
            summary["available"] = True
            summary["artifacts"].append(str(candidate.relative_to(ROOT)))
    return summary


def _matrix_rows() -> list[dict]:
    matrix = _load_json(ROOT / "generated" / "runs" / "broad_ohlcv_factor_matrix_10y" / "broad_factor_matrix.json")
    if not isinstance(matrix, dict):
        return []
    rows = []
    for item in matrix.get("results", {}).get("results", []):
        if "long_only" not in item:
            continue
        oos = item["long_only"]["oos"]
        diagnostics = item.get("diagnostics", {})
        rows.append({
            "factor": item["factor"],
            "oos_annual_return": oos.get("annual_return"),
            "oos_sharpe": oos.get("sharpe"),
            "oos_excess_annual": item.get("vs_benchmark", {}).get("oos", {}).get("annual_return"),
            "ic_mean": diagnostics.get("ic_mean"),
            "turnover": diagnostics.get("mean_turnover"),
            "cost_x2_total_return": item.get("cost_x2_total_return"),
        })
    rows.sort(key=lambda row: row.get("oos_sharpe") or -999, reverse=True)
    return rows


def _ml_summary() -> dict:
    data = _load_json(ROOT / "generated" / "runs" / "ml_ohlcv_walk_forward_10y_corrected" / "result.json") or {}
    return {
        "status": data.get("status"),
        "windows": data.get("windows"),
        "observations": data.get("observations"),
        "annual_return_net": data.get("annual_return_net"),
        "sharpe_net": data.get("sharpe_net"),
        "max_drawdown_net": data.get("max_drawdown_net"),
        "mean_auc": data.get("mean_auc"),
        "mean_turnover": data.get("mean_turnover"),
    }


def _registry_rows() -> list[dict]:
    rows: list[dict] = []
    try:
        import yaml
    except ImportError:
        yaml = None
    if yaml is not None:
        for name, path in (
            ("multi_horizon", ROOT / "quant_system" / "configs" / "research" / "multi_horizon_challenger_registry.yaml"),
            ("proxy_intake", ROOT / "quant_system" / "configs" / "research" / "proxy_matrix_high_sharpe_intake.yaml"),
        ):
            doc = None
            try:
                doc = yaml.safe_load(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not doc:
                continue
            for horizon, spec in (doc.get("horizons") or {}).items():
                for item in spec.get("strategies", []):
                    rows.append({"source": name, "horizon": horizon, "id": item.get("id"), "family": item.get("family"), "signal": item.get("signal"), "direction": item.get("direction")})
            for item in doc.get("challengers", []) or []:
                rows.append({"source": name, "horizon": "proxy", "id": item.get("id"), "family": item.get("family"), "signal": item.get("case"), "direction": 1, "proxy_sharpe": item.get("oos_sharpe")})
    # frozen PIT candidates
    frozen = _load_json(ROOT / "generated" / "runs" / "annual_pit_frozen_diagnostic" / "annual_pit_frozen_diagnostic.json") or {}
    for item in (frozen.get("results") or {}).get("results", []):
        if "long_only" in item:
            rows.append({"source": "annual_pit_frozen", "horizon": "long", "id": item["factor"], "family": "pit", "signal": item["factor"], "direction": 1, "oos_sharpe": item["long_only"]["oos"].get("sharpe")})
    return rows


def handler_overview(query: dict, send_json) -> None:
    from quant_system.qlib_adapter import status as qlib_status
    import importlib.util
    qlib = qlib_status()
    send_json({
        "ok": True,
        "framework": qlib,
        "open_source_engines": {
            "qlib": {"role": "factor_model_research", "installed": qlib["installed"], "repository": "https://github.com/microsoft/qlib"},
            "vectorbt": {"role": "fast_parameter_sweep", "installed": importlib.util.find_spec("vectorbt") is not None, "repository": "https://github.com/polakowo/vectorbt"},
            "backtrader": {"role": "order_execution_reconciliation", "installed": importlib.util.find_spec("backtrader") is not None, "repository": "https://github.com/backtrader2/backtrader"},
        },
        "framework_legacy": qlib_status(),
        "engines": {
            "research_authority": "microsoft_qlib",
            "execution_reconciliation": "quant_system.backtrader_engine",
            "legacy_fast_reference": "quant_system.portfolio_holding_backtest",
            "protocol": "quant_system.backtest_protocol",
        },
        "data_gate": _gate(),
        "panel": {"path": str(PANEL_PATH.relative_to(ROOT)), "exists": PANEL_PATH.is_file()},
        "research_pool": {
            "predeclared_strategies": len(_registry_rows()),
            "ohlcv_factors": len(_matrix_rows()),
            "ml_challenger": _ml_summary(),
            "frozen_rolling_oos": _load_json(ROOT / "generated" / "runs" / "research_validation_suite.json") or {},
        },
        "default_signals": list(DEFAULT_SIGNALS),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    })


def handler_strategies(query: dict, send_json) -> None:
    from quant_system.strategy_registry import catalog, signal_catalog
    send_json({"ok": True, "catalog": catalog(), "signals": signal_catalog(), "strategies": _registry_rows(), "matrix": _matrix_rows(), "ml": _ml_summary()})


def _query_value(query: dict, key: str, default: str = "") -> str:
    item = query.get(key)
    return item[0] if item else default


def _add_ohlcv_factors(panel):
    """Compute the small, deterministic factor set needed by the console.

    The warehouse panel normally already contains these columns.  This fallback
    keeps alternate price panels (including the broad main-board panel) usable
    without requiring a separate factor-generation job.
    """
    import numpy as np
    import pandas as pd

    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["code"] = data["code"].astype(str).str.zfill(6)
    for dst, candidates in {"raw_open": ("raw_open", "open"), "raw_high": ("raw_high", "high"),
                            "raw_low": ("raw_low", "low"), "raw_close": ("raw_close", "close")}.items():
        if dst not in data:
            for candidate in candidates:
                if candidate in data:
                    data[dst] = pd.to_numeric(data[candidate], errors="coerce")
                    break
    data = data.sort_values(["code", "date"])
    g = data.groupby("code", group_keys=False)
    close = data["raw_close"]
    ret = g["raw_close"].pct_change()
    for name, window in (("short_reversal_5", 5), ("short_reversal_20", 20)):
        if name not in data:
            data[name] = -g["raw_close"].pct_change(window)
    for name, window in (("mom_3_1", 63), ("mom_6_1", 126), ("mom_12_1", 252),
                         ("trend_60", 60), ("trend_120", 120)):
        if name not in data:
            data[name] = g["raw_close"].pct_change(window)
    if "low_vol_60" not in data:
        data["low_vol_60"] = -ret.groupby(data["code"]).transform(lambda s: s.rolling(60, min_periods=20).std())
    if "downside_vol_60" not in data:
        data["downside_vol_60"] = -ret.where(ret < 0).groupby(data["code"]).transform(lambda s: s.rolling(60, min_periods=10).std())
    if "volatility_120" not in data:
        data["volatility_120"] = -ret.groupby(data["code"]).transform(lambda s: s.rolling(120, min_periods=30).std())
    if "volume_ratio_20" not in data and "volume" in data:
        data["volume_ratio_20"] = data["volume"] / g["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    if "range_compression" not in data:
        data["range_compression"] = -(data["raw_high"] - data["raw_low"]) / close.replace(0, np.nan)
    if "price_efficiency_20" not in data:
        data["price_efficiency_20"] = g["raw_close"].transform(lambda s: s.diff(20).abs()) / g["raw_close"].transform(lambda s: s.diff().abs().rolling(20, min_periods=10).sum())
    return data


def _load_panel(universe: str = "500"):
    import pandas as pd
    path = PANEL_PATH
    if universe == "mainboard":
        # Prefer a dedicated release, otherwise use the existing broad release
        # and apply the actual exchange-board eligibility rule: 00/60.
        dedicated = ROOT / "data_warehouse" / "research_panels" / "mainboard_10y" / "mainboard_price_factor_panel.parquet"
        path = dedicated if dedicated.is_file() else ROOT / "data_warehouse" / "research_panels" / "ashare_price_500_2016_2026" / "price_panel.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"panel_not_found:{path}")
    panel = _add_ohlcv_factors(pd.read_parquet(path))
    if universe == "mainboard":
        panel = panel[panel["code"].astype(str).str.match(r"^(00|60)")].copy()
    state_path = path.parent / "trade_state.parquet"
    state = pd.read_parquet(state_path) if state_path.is_file() else None
    return panel, state, path


def run_backtest_job(params: dict) -> dict:
    """Task-queue handler; Qlib is the research authority."""
    signal = str(params.get("signal") or "short_reversal_5")
    universe = str(params.get("universe") or "500")
    quantile = max(.05, min(.5, float(params.get("quantile", .2))))
    rebalance = max(2, min(120, int(params.get("rebalance_sessions", 5))))
    production_raw = params.get("production", False)
    production = production_raw is True or str(production_raw).lower() in {"1", "true", "yes"}
    costs = dict(params.get("costs") or {})
    risk = dict(params.get("risk") or {})
    panel, state, panel_path = _load_panel(universe)
    import pandas as pd
    from quant_system.backtest_service import run_canonical, serializable_run
    frame = panel.copy()
    frame["signal"] = pd.to_numeric(frame[signal], errors="coerce")
    if "amount" not in frame:
        frame["amount"] = pd.to_numeric(frame.get("volume", 0), errors="coerce") * frame["raw_close"]
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce").fillna(0.0)
    if "adv_amount_20" not in frame:
        frame["adv_amount_20"] = frame.groupby("code")["amount"].transform(lambda x: x.rolling(20, min_periods=1).mean())
    if state is not None and len(state):
        available = [c for c in ("date", "code", "suspended", "limit_up_locked", "limit_down_locked") if c in state.columns]
        frame = frame.merge(state[available], on=["date", "code"], how="left", suffixes=("", "_state"))
        for col in ("suspended", "limit_up_locked", "limit_down_locked"):
            if col + "_state" in frame:
                frame[col] = frame[col].fillna(frame[col + "_state"]).fillna(False)
    from quant_system.strategy_registry import find_signal
    family = (find_signal(signal) or {}).get("family", "cross_section")
    run = run_canonical(frame, signal, quantile=quantile, rebalance_sessions=rebalance,
                        production=production, trade_state=state, family=family, **costs, **risk)
    if not run.get("ok"):
        return run
    payload = serializable_run(run)
    payload.update({"universe": universe, "panel": str(panel_path.relative_to(ROOT)), "signal": signal, "generated_at": datetime.now().astimezone().isoformat(timespec="seconds")})
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    artifact = OUTPUT_ROOT / f"{universe}__{signal}__r{rebalance}__q{int(quantile * 100)}.json"
    artifact.write_text(json.dumps(payload, ensure_ascii=True, default=str), encoding="utf-8")
    return {"artifact": str(artifact.relative_to(ROOT)), "schema": payload["schema"], "spec_hash": payload["spec_hash"], "spec": payload["spec"], "metrics": payload["metrics"], "gate": payload["gate"], "engine": payload["engine"], "canonical": True, "rows": len(payload["equity_curve"]), "trades": len(payload["trades"]), "trade_rows": payload["trades"], "equity_curve": payload["equity_curve"]}


def register_task_handlers() -> None:
    from quant_system.task_queue import register_handler
    register_handler("quant_backtest", run_backtest_job)


def handler_task(query: dict, send_json) -> None:
    from quant_system.task_queue import get_task_result
    task_id = _query_value(query, "task_id")
    if not task_id:
        send_json({"ok": False, "error": "task_id_required"}, status=400)
        return
    send_json({"ok": True, "task": get_task_result(task_id)})


def handler_run(query: dict, send_json) -> None:
    def value(key: str, default: str = "") -> str:
        return _query_value(query, key, default)

    signal = value("signal") or "short_reversal_5"
    if signal not in DEFAULT_SIGNALS:
        send_json({"ok": False, "error": f"signal must be one of {list(DEFAULT_SIGNALS)}"})
        return
    universe = value("universe") or "500"
    if universe not in {"500", "mainboard"}:
        send_json({"ok": False, "error": "universe must be 500 or mainboard"})
        return
    rebalance = max(2, min(120, int(value("rebalance_sessions") or "5")))
    quantile = max(.05, min(.5, float(value("quantile") or ".2")))
    cross_check = value("cross_check") in {"1", "true", "yes"}
    full = value("full") in {"1", "true", "yes"}
    params = {"signal": signal, "universe": universe, "rebalance_sessions": rebalance, "quantile": quantile,
              "production": value("production") in {"1", "true", "yes"},
              "costs": {"slippage_bps": float(value("slippage_bps") or 10),
                        "max_adv_participation": float(value("max_adv_participation") or .1)},
              "risk": {"max_position_weight": float(value("max_position_weight") or .1)}}
    try:
        from quant_system.qlib_adapter import status as qlib_status
        framework = qlib_status()
        # Qlib is an optional research enhancer. The local execution/research
        # path remains valid and is the default when the dependency is absent.
        if full or universe == "mainboard":
            from quant_system.task_queue import submit_task
            task_id = submit_task("quant_backtest", params, max_retries=0)
            send_json({"ok": True, "accepted": True, "task_id": task_id, "engine": "quant_system.backtrader_engine", "canonical": True, "research_framework": "microsoft_qlib" if framework.get("installed") else None, "message": "已提交订单级 canonical 回测，可用 task_id 查询"})
            return
        data = run_backtest_job(params)
        if cross_check:
            from quant_system.backtest_crosscheck import bounded_crosscheck
            panel, _, _ = _load_panel(universe)
            data["cross_check"] = bounded_crosscheck(panel, signal, days=90, symbols=20, holding_sessions=rebalance, quantile=quantile)
        send_json({"ok": True, "data": data, "engine": "quant_system.backtrader_engine", "canonical": True})
    except Exception as exc:  # noqa: BLE001
        send_json({"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:240]}"})
