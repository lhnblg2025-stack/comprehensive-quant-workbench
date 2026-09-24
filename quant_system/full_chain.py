"""One-command local quant chain: PIT panel -> low-frequency research -> execution audit.

This is deliberately paper/research only. It never submits a broker order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .closed_loop import (
    LoopConfig,
    _as_dates,
    _file_hash,
    _config_hash,
    append_candidates,
    batch_replay_exits,
    candidate_library,
    candidate_performance,
    factor_diagnostics,
    init_candidate_store,
)
from .execution_adapter import run_order_level_backtest


@dataclass(frozen=True)
class FullChainConfig:
    data: str
    output_dir: str
    factors: tuple[str, ...]
    directions: tuple[tuple[str, int], ...] = ()
    rebalance: str = "weekly"
    top_n: int = 10
    quantiles: int = 5
    cost_bps: float = 15.0
    initial_capital: float = 1_000_000.0
    min_execution_dates: int = 2
    release_id: str = "local"


def _targets(library: pd.DataFrame, cfg: FullChainConfig) -> tuple[dict, dict]:
    """Convert selected candidates to equal-weight target portfolios on rebalance dates."""
    dates = pd.DatetimeIndex(sorted(library.date.unique()))
    if cfg.rebalance == "weekly":
        rebalance = set(pd.Series(dates, index=dates).groupby(dates.to_period("W")).max())
    elif cfg.rebalance == "monthly":
        rebalance = set(pd.Series(dates, index=dates).groupby(dates.to_period("M")).max())
    else:
        rebalance = set(dates)
    targets: dict = {}; candidate_ids: dict = {}
    for date in sorted(rebalance):
        rows = library[(library.date == date) & library.selected].sort_values(["composite_rank", "code"]).head(cfg.top_n)
        if rows.empty:
            continue
        weight = 1.0 / len(rows)
        targets[date] = {str(row.code): weight for row in rows.itertuples()}
        candidate_ids[date] = {str(row.code): hashlib.sha256(f"{row.factor_version}|{date.date()}|{row.code}".encode()).hexdigest()[:20] for row in rows.itertuples()}
    return targets, candidate_ids


def _execution_panel(panel: pd.DataFrame) -> pd.DataFrame:
    data = _as_dates(panel)
    aliases = {"raw_open": "open", "raw_high": "high", "raw_low": "low", "raw_close": "close"}
    if set(aliases).issubset(data.columns):
        data = data.copy()
        data[["open", "high", "low", "close"]] = data[list(aliases)].to_numpy()
    required = {"date", "code", "open", "high", "low", "close"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"execution panel missing columns: {sorted(missing)}")
    return data


def run_full_chain(cfg: FullChainConfig) -> dict[str, Any]:
    source = Path(cfg.data); output = Path(cfg.output_dir); output.mkdir(parents=True, exist_ok=True)
    panel = _as_dates(__import__("quant_system.closed_loop", fromlist=["load_panel"]).load_panel(source))
    loop_cfg = LoopConfig(factors=cfg.factors, directions=cfg.directions, top_n=cfg.top_n, quantiles=cfg.quantiles, cost_bps=cfg.cost_bps)
    library = candidate_library(panel, loop_cfg)
    diagnostics = factor_diagnostics(library, loop_cfg)
    performance = candidate_performance(library, loop_cfg)
    targets, candidate_ids = _targets(library, cfg)
    execution: dict[str, Any] = {"status": "BLOCKED", "reason": "oos_quality_gate_hold"}
    if diagnostics["quality_gate"] == "PASS" and len(targets) >= cfg.min_execution_dates:
        result = run_order_level_backtest(_execution_panel(panel), targets, capital=cfg.initial_capital,
                                          commission_rate=cfg.cost_bps / 10000.0, release_id=cfg.release_id,
                                          candidate_ids=candidate_ids)
        curve = result.equity_curve
        execution = {"status": "AVAILABLE", "rebalance": cfg.rebalance, "targets": len(targets),
                      "orders": len(result.orders), "trades": len(result.trades),
                      "total_return": result.total_return, "max_drawdown": result.max_drawdown,
                      "equity_observations": len(curve), "candidate_attribution": getattr(result, "candidate_attribution", [])}
        curve.to_csv(output / "execution_equity.csv")
        pd.DataFrame([vars(item) for item in result.trades]).to_json(output / "execution_trades.json", orient="records", force_ascii=False, default_handler=str)
    serial_targets = {str(pd.Timestamp(date).date()): values for date, values in targets.items()}
    serial_candidate_ids = {str(pd.Timestamp(date).date()): values for date, values in candidate_ids.items()}
    plan = {"schema": "full-chain-paper-plan/v1", "status": "BLOCKED_BY_QUALITY_GATE" if execution["status"] == "BLOCKED" else "PAPER_READY", "rebalance": cfg.rebalance, "targets": serial_targets, "candidate_ids": serial_candidate_ids}
    performance.to_parquet(output / "candidate_performance.parquet", index=False)
    library.to_parquet(output / "candidate_library.parquet", index=False)
    (output / "factor_diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (output / "paper_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    store = init_candidate_store(output / "candidate_store.db")
    store_rows = append_candidates(store, performance, {"plans": [{"candidate_id": cid, "symbol": symbol, "as_of": date, "status": plan["status"]} for date, values in serial_candidate_ids.items() for symbol, cid in values.items()]}, release_id=cfg.release_id, data_sha256=_file_hash(source), quality_gate=diagnostics["quality_gate"])
    manifest = {"schema": "full-chain/v1", "config": asdict(cfg), "strategy_version": _config_hash(loop_cfg), "data_sha256": _file_hash(source), "release_id": cfg.release_id, "quality_gate": diagnostics["quality_gate"], "execution": execution, "store_rows": store_rows, "artifacts": ["candidate_library.parquet", "candidate_performance.parquet", "factor_diagnostics.json", "paper_plan.json", "candidate_store.db"]}
    manifest["sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True, default=str).encode()).hexdigest()
    (output / "full_chain_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True); parser.add_argument("--output-dir", required=True); parser.add_argument("--factors", nargs="+", required=True)
    parser.add_argument("--rebalance", choices=("daily", "weekly", "monthly"), default="weekly"); parser.add_argument("--top-n", type=int, default=10); parser.add_argument("--release-id", default="local")
    args = parser.parse_args(argv)
    result = run_full_chain(FullChainConfig(args.data, args.output_dir, tuple(args.factors), rebalance=args.rebalance, top_n=args.top_n, release_id=args.release_id))
    print(json.dumps({"status": result["execution"]["status"], "quality_gate": result["quality_gate"], "output_dir": args.output_dir, "sha256": result["sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
