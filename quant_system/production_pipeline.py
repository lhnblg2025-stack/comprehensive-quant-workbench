"""Canonical production chain: release -> factor quality -> targets -> order-level audit -> decision evidence."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .data_release import load_release
from .execution_adapter import run_order_level_backtest
from .product_contract import release_metadata

CST = timezone(timedelta(hours=8))
ROOT = Path(__file__).resolve().parent.parent


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _load_research_gate(day: str, release_id: str) -> dict[str, Any]:
    """Fail closed unless a same-release closed-loop OOS gate is available."""
    candidates = sorted((ROOT / "generated").glob("closed_loop*/manifest.json"))
    candidates += sorted((ROOT / "generated").glob("full_chain*/full_chain_manifest.json"))
    if not candidates:
        return {"status": "BLOCK", "reason": "closed_loop_manifest_missing"}
    manifest = _json(candidates[-1])
    if manifest.get("release_id", "local") != release_id:
        return {"status": "BLOCK", "reason": "closed_loop_release_mismatch", "manifest_release_id": manifest.get("release_id")}
    if manifest.get("quality_gate") != "PASS":
        return {"status": "BLOCK", "reason": "closed_loop_quality_gate_hold", "quality_gate": manifest.get("quality_gate")}
    return {"status": "PASS", "manifest": candidates[-1].name, "data_sha256": manifest.get("data_sha256"), "strategy_version": manifest.get("strategy_version")}


def _load_factor_targets(day: str, release_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Use the only promoted OOS factor registry and latest target artifact."""
    registry = _json(ROOT / "generated" / "factor_quality_registry.json")
    eligible = [item for item in registry.get("factors", []) if item.get("tier") == "core"]
    candidates = sorted((ROOT / "generated").glob(f"factor_strategy_targets_{day}.parquet"))
    if not candidates:
        return pd.DataFrame(), {"status": "WARN", "reason": "target_artifact_missing", "core_factors": len(eligible)}
    target = pd.read_parquet(candidates[-1]).copy()
    required = {"date", "code", "target_weight"}
    if not required.issubset(target.columns):
        return pd.DataFrame(), {"status": "WARN", "reason": "target_schema_invalid", "core_factors": len(eligible)}
    target["date"] = pd.to_datetime(target["date"], errors="coerce")
    if "release_id" not in target.columns or not target["release_id"].eq(release_id).all():
        return pd.DataFrame(), {"status": "WARN", "reason": "target_release_mismatch", "core_factors": len(eligible)}
    target = target[target["date"].dt.date.astype(str) == day]
    return target, {"status": "PASS", "core_factors": len(eligible), "artifact": candidates[-1].name}


def run(day: str, *, release_id: str | None = None) -> dict[str, Any]:
    release = load_release(ROOT, release_id=release_id, expected_day=day)
    research_gate = _load_research_gate(day, release["release_id"])
    targets, target_meta = _load_factor_targets(day, release["release_id"])
    payload: dict[str, Any] = {
        **release_metadata(), "schema": "quant-production-pipeline/v1", "as_of": day,
        "release_id": release["release_id"], "generated_at": datetime.now(CST).isoformat(timespec="seconds"),
        "factor_targets": target_meta, "research_gate": research_gate,
    }
    if research_gate["status"] != "PASS":
        payload["status"] = "BLOCK"
        payload["order_level_audit"] = {"status": "NOT_RUN", "reason": research_gate["reason"]}
    elif targets.empty:
        payload["status"] = "WARN"
        payload["order_level_audit"] = {"status": "NOT_RUN", "reason": target_meta["reason"]}
    else:
        panel_path = ROOT / "data_warehouse" / "backtest_bundles" / "amp20_100_2020_2026_r2" / "prices_hfq_raw.parquet"
        actions = None
        execution_source = "frozen_bundle_r2"
        if panel_path.exists():
            source = pd.read_parquet(panel_path)
            # Signal factors remain HFQ inside target construction; all order
            # execution below deliberately uses only RAW OHLCV fields.
            panel = source[["date", "code", "raw_open", "raw_high", "raw_low", "raw_close", "volume", "amount"]].rename(columns={
                "raw_open": "open", "raw_high": "high", "raw_low": "low", "raw_close": "close"
            }).copy()
            actions_path = panel_path.parent / "corporate_actions.parquet"
            actions = pd.read_parquet(actions_path) if actions_path.exists() else None
        else:
            # Resident cloud nodes need not store frozen research packages.
            # Build a raw execution panel from their canonical kline warehouse.
            frames = []
            for code in sorted(targets["code"].astype(str).unique()):
                path = ROOT / "data_warehouse" / "kline" / f"{code.zfill(6)}.parquet"
                if path.exists():
                    frame = pd.read_parquet(path)
                    frame["code"] = code.zfill(6)
                    frames.append(frame)
            if not frames:
                payload["status"] = "WARN"; payload["order_level_audit"] = {"status": "NOT_RUN", "reason": "execution_panel_missing"}
                out = ROOT / "generated" / f"production_pipeline_{day}.json"
                out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
                return payload
            source = pd.concat(frames, ignore_index=True)
            columns = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount"}
            panel = source.rename(columns=columns)
            execution_source = "resident_kline_raw"
            required = ["date", "code", "open", "high", "low", "close"]
            panel = panel[[c for c in [*required, "volume", "amount"] if c in panel.columns]].copy()
        target_map = {target_date: frame.set_index("code")["target_weight"].to_dict() for target_date, frame in targets.groupby("date")}
        result = run_order_level_backtest(panel, target_map, corporate_actions=actions, release_id=release["release_id"])
        eq = result.equity_curve
        accounting_error = float((eq["total_value"] - eq["cash"] - eq["position_value"]).abs().max()) if not eq.empty else None
        has_execution = bool(len(result.orders) or len(result.trades))
        payload["status"] = "PASS" if has_execution and accounting_error is not None and accounting_error <= .01 else "WARN"
        payload["order_level_audit"] = {"status": payload["status"], "source": execution_source,
                                         "trades": len(result.trades), "orders": len(result.orders),
                                         "accounting_error": accounting_error,
                                         "reason": None if has_execution else "no_orders_or_trades_for_target_date",
                                         "release_id": getattr(result, "release_id", None)}
    canonical = json.dumps(payload, sort_keys=True, default=str).encode()
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    out = ROOT / "generated" / f"production_pipeline_{day}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return payload


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--release-id")
    args = parser.parse_args()
    result = run(args.date, release_id=args.release_id)
    print(json.dumps({"status": result["status"], "release_id": result["release_id"], "order_level_audit": result["order_level_audit"]}, ensure_ascii=False))
    raise SystemExit(0 if result["status"] in ("PASS", "WARN") else 2)
