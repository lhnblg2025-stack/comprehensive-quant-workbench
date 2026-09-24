"""Low-cost, reproducible research-to-paper-trading loop.

The module deliberately works on local parquet/CSV snapshots and has no broker,
QMT, ML or network requirement. A row in the candidate library is the contract
between research, paper execution and later attribution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LoopConfig:
    factors: tuple[str, ...]
    directions: tuple[tuple[str, int], ...] = ()
    top_n: int = 10
    horizons: tuple[int, ...] = (1, 3, 5, 10)
    quantiles: int = 5
    cost_bps: float = 15.0
    stop_atr: float = 2.0
    target_atr: float = 3.0
    time_stop: int = 5
    min_factor_coverage: float = 1.0
    min_monotonicity: float = 0.6
    min_diagnostic_dates: int = 5


def init_candidate_store(path: str | Path) -> Path:
    """Create the append-only candidate ledger used for paper feedback."""
    db_path = Path(path); db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS candidates (
                candidate_id TEXT PRIMARY KEY, as_of TEXT NOT NULL, symbol TEXT NOT NULL,
                strategy_version TEXT NOT NULL, release_id TEXT NOT NULL, data_sha256 TEXT NOT NULL,
                rank INTEGER, score REAL, factor_coverage REAL, signal_close REAL,
                plan_json TEXT NOT NULL, status TEXT NOT NULL, order_id INTEGER,
                filled_shares INTEGER DEFAULT 0, filled_price REAL, exit_price REAL,
                exit_reason TEXT, attribution_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP, updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS candidate_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, candidate_id TEXT NOT NULL,
                event TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id)
            );
            CREATE INDEX IF NOT EXISTS idx_candidate_events_candidate ON candidate_events(candidate_id);
        """)
        existing = {row[1] for row in db.execute("PRAGMA table_info(candidates)")}
        migrations = {
            "order_id": "ALTER TABLE candidates ADD COLUMN order_id INTEGER",
            "filled_shares": "ALTER TABLE candidates ADD COLUMN filled_shares INTEGER DEFAULT 0",
            "filled_price": "ALTER TABLE candidates ADD COLUMN filled_price REAL",
            "exit_price": "ALTER TABLE candidates ADD COLUMN exit_price REAL",
            "exit_reason": "ALTER TABLE candidates ADD COLUMN exit_reason TEXT",
            "attribution_json": "ALTER TABLE candidates ADD COLUMN attribution_json TEXT",
        }
        for column, sql in migrations.items():
            if column not in existing:
                db.execute(sql)
    return db_path


def append_candidates(path: str | Path, performance: pd.DataFrame, plans: dict, *,
                      release_id: str, data_sha256: str, quality_gate: str) -> int:
    """Idempotently append candidates and record the gate event."""
    db_path = init_candidate_store(path); plan_map = {p["candidate_id"]: p for p in plans.get("plans", [])}
    count = 0
    with sqlite3.connect(db_path) as db:
        for row in performance.to_dict("records"):
            candidate_id = row["candidate_id"]
            status = "PAPER_BLOCKED" if quality_gate != "PASS" else "PAPER_PENDING"
            payload = plan_map.get(candidate_id, {})
            db.execute("""INSERT INTO candidates
                (candidate_id,as_of,symbol,strategy_version,release_id,data_sha256,rank,score,factor_coverage,signal_close,plan_json,status)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(candidate_id) DO UPDATE SET updated_at=CURRENT_TIMESTAMP""",
                (candidate_id, row["as_of"], row["symbol"], row["strategy_version"], release_id, data_sha256,
                 row["rank"], row["score"], row["factor_coverage"], row["signal_close"], json.dumps(payload, ensure_ascii=False), status))
            db.execute("INSERT INTO candidate_events(candidate_id,event,payload_json) VALUES (?,?,?)",
                       (candidate_id, "quality_gate", json.dumps({"quality_gate": quality_gate}, ensure_ascii=False)))
            count += 1
    return count


_STATUS_TRANSITIONS = {
    "PAPER_BLOCKED": {"PAPER_PENDING"},
    "PAPER_PENDING": {"CONFIRMED"},
    "CONFIRMED": {"FILLED"},
    "FILLED": {"EXITED"},
    "EXITED": {"ATTRIBUTED"},
    "ATTRIBUTED": set(),
}


def get_candidate(path: str | Path, candidate_id: str) -> dict | None:
    init_candidate_store(path)
    with sqlite3.connect(path) as db:
        db.row_factory = sqlite3.Row
        row = db.execute("SELECT * FROM candidates WHERE candidate_id = ?", (candidate_id,)).fetchone()
    return dict(row) if row else None


def transition_candidate(path: str | Path, candidate_id: str, new_status: str, *, payload: dict | None = None) -> dict:
    """Move one candidate through the paper lifecycle; reject skips and regressions."""
    init_candidate_store(path); payload = payload or {}
    with sqlite3.connect(path) as db:
        db.row_factory = sqlite3.Row
        row = db.execute("SELECT status FROM candidates WHERE candidate_id = ?", (candidate_id,)).fetchone()
        if not row:
            raise KeyError(f"candidate not found: {candidate_id}")
        old_status = row["status"]
        if new_status not in _STATUS_TRANSITIONS.get(old_status, set()):
            raise ValueError(f"invalid candidate transition: {old_status} -> {new_status}")
        db.execute("UPDATE candidates SET status=?, updated_at=CURRENT_TIMESTAMP WHERE candidate_id=?", (new_status, candidate_id))
        db.execute("INSERT INTO candidate_events(candidate_id,event,payload_json) VALUES (?,?,?)",
                   (candidate_id, new_status.lower(), json.dumps(payload, ensure_ascii=False, default=str)))
    return get_candidate(path, candidate_id) or {}


def link_paper_order(path: str | Path, candidate_id: str, order_id: int) -> dict:
    """Attach an existing trade_db paper order without duplicating order storage."""
    row = get_candidate(path, candidate_id)
    if not row:
        raise KeyError(f"candidate not found: {candidate_id}")
    if row.get("order_id") == int(order_id):
        return row
    if row.get("order_id") is not None:
        raise ValueError("candidate already has a different paper order")
    with sqlite3.connect(path) as db:
        db.execute("UPDATE candidates SET order_id=?, updated_at=CURRENT_TIMESTAMP WHERE candidate_id=?", (int(order_id), candidate_id))
        db.execute("INSERT INTO candidate_events(candidate_id,event,payload_json) VALUES (?,?,?)",
                   (candidate_id, "order_linked", json.dumps({"order_id": int(order_id)})))
    return get_candidate(path, candidate_id) or {}


def create_linked_paper_order(path: str | Path, candidate_id: str, *, direction: str, shares: int,
                              suggested_price: float, order_type: str = "limit", notes: str = "") -> dict:
    """Create a manual paper order in trade_db and atomically link its id here."""
    row = get_candidate(path, candidate_id)
    if not row:
        raise KeyError(f"candidate not found: {candidate_id}")
    if row["status"] != "CONFIRMED":
        raise ValueError(f"candidate must be CONFIRMED before order creation: {row['status']}")
    from .trade_db import create_paper_order
    order = create_paper_order(symbol=row["symbol"], direction=direction, shares=shares,
                               suggested_price=suggested_price, release_id=row["release_id"],
                               order_type=order_type, notes=notes, candidate_id=candidate_id)
    if order.get("error"):
        raise ValueError(order["error"])
    return link_paper_order(path, candidate_id, int(order["id"]))


def confirm_candidate(path: str | Path, candidate_id: str, *, payload: dict | None = None) -> dict:
    return transition_candidate(path, candidate_id, "CONFIRMED", payload=payload)


def sync_trade_db_fill(path: str | Path, candidate_id: str, order_id: int) -> dict:
    """Pull a filled trade_db order into the candidate ledger."""
    from .trade_db import get_paper_order
    order = get_paper_order(order_id)
    if not order or order.get("candidate_id") != candidate_id:
        raise ValueError("trade_db order is missing or candidate_id does not match")
    shares = int(order.get("filled_shares") or 0)
    price = order.get("filled_price")
    if shares <= 0 or price is None:
        raise ValueError("trade_db order has no fill")
    return record_fill(path, candidate_id, order_id=order_id, shares=shares, price=float(price))


def record_fill(path: str | Path, candidate_id: str, *, order_id: int, shares: int, price: float) -> dict:
    row = get_candidate(path, candidate_id)
    if not row or row.get("order_id") != int(order_id):
        raise ValueError("order does not match candidate")
    if row["status"] == "FILLED" and row.get("filled_shares") == int(shares) and float(row.get("filled_price")) == float(price):
        return row
    with sqlite3.connect(path) as db:
        db.execute("UPDATE candidates SET filled_shares=?, filled_price=?, updated_at=CURRENT_TIMESTAMP WHERE candidate_id=?", (int(shares), float(price), candidate_id))
    return transition_candidate(path, candidate_id, "FILLED", payload={"order_id": order_id, "shares": shares, "price": price})


def batch_replay_exits(path: str | Path, panel: pd.DataFrame, *, max_days: int = 5) -> dict:
    """Replay exits for filled candidates and persist exit plus attribution events."""
    init_candidate_store(path)
    data = _as_dates(panel)
    with sqlite3.connect(path) as db:
        db.row_factory = sqlite3.Row
        candidates = [dict(row) for row in db.execute("SELECT * FROM candidates WHERE status = 'FILLED'").fetchall()]
    updated = 0; pending = 0
    for candidate in candidates:
        plan = json.loads(candidate.get("plan_json") or "{}")
        bars = data[(data["code"] == candidate["symbol"]) & (data["date"] > pd.Timestamp(candidate["as_of"]))].copy()
        result = replay_exit(bars, float(candidate["filled_price"]), stop_price=float(plan.get("stop_loss", candidate["filled_price"] * .95)), target_price=float(plan.get("take_profit", candidate["filled_price"] * 1.05)), max_days=max_days)
        if result["reason"] == "no_bars":
            pending += 1
            continue
        exited = record_exit(path, candidate["candidate_id"], price=float(result["exit_price"]), reason=str(result["reason"]))
        net_return = (float(result["exit_price"]) / float(candidate["filled_price"]) - 1.0) - float(plan.get("cost_bps", 0.0)) / 10000.0
        attribute_candidate(path, candidate["candidate_id"], payload={"net_return": net_return, "exit_days": result["days"], "exit_reason": result["reason"]})
        updated += 1
    return {"filled_candidates": len(candidates), "attributed": updated, "pending_no_bars": pending}


def record_exit(path: str | Path, candidate_id: str, *, price: float, reason: str) -> dict:
    with sqlite3.connect(path) as db:
        db.execute("UPDATE candidates SET exit_price=?, exit_reason=?, updated_at=CURRENT_TIMESTAMP WHERE candidate_id=?", (float(price), reason, candidate_id))
    return transition_candidate(path, candidate_id, "EXITED", payload={"price": price, "reason": reason})


def attribute_candidate(path: str | Path, candidate_id: str, *, payload: dict) -> dict:
    with sqlite3.connect(path) as db:
        db.execute("UPDATE candidates SET attribution_json=?, updated_at=CURRENT_TIMESTAMP WHERE candidate_id=?", (json.dumps(payload, ensure_ascii=False, default=str), candidate_id))
    return transition_candidate(path, candidate_id, "ATTRIBUTED", payload=payload)


def _as_dates(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["code"] = data["code"].astype(str).str.zfill(6)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date", "code"]).sort_values(["code", "date"])
    return data.drop_duplicates(["code", "date"], keep="last")


def load_panel(path: str | Path) -> pd.DataFrame:
    """Load one panel or a directory of snapshots with a strict minimum schema."""
    source = Path(path)
    files = sorted(source.glob("*.parquet")) if source.is_dir() else [source]
    if source.is_dir():
        files += sorted(source.glob("*.csv"))
    frames: list[pd.DataFrame] = []
    for file in files:
        frame = pd.read_parquet(file) if file.suffix == ".parquet" else pd.read_csv(file)
        if {"code", "date", "close"}.issubset(frame.columns):
            frames.append(frame)
    if not frames:
        raise ValueError("no usable panel; require code, date and close columns")
    data = _as_dates(pd.concat(frames, ignore_index=True))
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    if "raw_open" in data.columns:
        data["raw_open"] = pd.to_numeric(data["raw_open"], errors="coerce")
    return data[data["close"] > 0].copy()


def _forward_labels(data: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    out = data.copy()
    for horizon in sorted(set(int(x) for x in horizons)):
        if horizon < 1:
            raise ValueError("horizons must be positive")
        out[f"forward_{horizon}d"] = out.groupby("code")["close"].shift(-horizon) / out["close"] - 1.0
    # Equal-weight market return is calculated only from stocks with a valid label.
    for horizon in sorted(set(int(x) for x in horizons)):
        col = f"forward_{horizon}d"
        benchmark = out.groupby("date")[col].transform("mean")
        out[f"benchmark_{horizon}d"] = benchmark
        out[f"excess_{horizon}d"] = out[col] - benchmark
    return out


def _rank_score(data: pd.DataFrame, factors: tuple[str, ...], directions: dict[str, int], min_coverage: float = 1.0) -> pd.DataFrame:
    out = data.copy()
    ranks = []
    for factor in factors:
        if factor not in out.columns:
            continue
        direction = 1 if int(directions.get(factor, 1)) >= 0 else -1
        value = pd.to_numeric(out[factor], errors="coerce") * direction
        ranks.append(value.groupby(out["date"]).rank(pct=True))
    if not ranks:
        raise ValueError("none of the requested factors exists in panel")
    rank_frame = pd.concat(ranks, axis=1)
    out["factor_coverage"] = rank_frame.notna().mean(axis=1)
    out["composite_score"] = rank_frame.mean(axis=1, skipna=True)
    out.loc[out["factor_coverage"] < min_coverage, "composite_score"] = np.nan
    out["composite_rank"] = out.groupby("date")["composite_score"].rank(ascending=False, method="first")
    out["selected"] = out["composite_rank"] <= int(max(1, factors and 1))
    return out


def candidate_library(panel: pd.DataFrame, config: LoopConfig) -> pd.DataFrame:
    """Create one auditable row per date/code, including future labels."""
    data = _forward_labels(_as_dates(panel), config.horizons)
    directions = dict(config.directions)
    out = _rank_score(data, config.factors, directions, config.min_factor_coverage)
    out["selected"] = out["composite_rank"] <= config.top_n
    out["factor_version"] = _config_hash(config)
    return out.sort_values(["date", "composite_rank", "code"])


def candidate_performance(library: pd.DataFrame, config: LoopConfig) -> pd.DataFrame:
    """Return one selected candidate per row with stable attribution fields."""
    columns = ["candidate_id", "as_of", "symbol", "strategy_version", "rank", "score", "factor_coverage",
               "signal_close", "status", "execution_status", "exit_status", "cost_bps"]
    selected = library[library["selected"]].copy()
    if selected.empty:
        return pd.DataFrame(columns=columns)
    selected["as_of"] = selected["date"].dt.strftime("%Y-%m-%d")
    selected["symbol"] = selected["code"]
    selected["strategy_version"] = selected["factor_version"]
    selected["rank"] = selected["composite_rank"].astype(int)
    selected["score"] = selected["composite_score"]
    selected["signal_close"] = selected["close"]
    selected["candidate_id"] = selected.apply(lambda row: hashlib.sha256(
        f"{row.strategy_version}|{row.as_of}|{row.symbol}".encode()).hexdigest()[:20], axis=1)
    selected["status"] = np.where(selected[f"forward_{min(config.horizons)}d"].notna(), "EVALUATED", "PENDING")
    selected["execution_status"] = "NOT_EXECUTED"
    selected["exit_status"] = "PENDING"
    selected["cost_bps"] = config.cost_bps
    for horizon in config.horizons:
        selected[f"gross_return_{horizon}d"] = selected[f"forward_{horizon}d"]
        selected[f"benchmark_return_{horizon}d"] = selected[f"benchmark_{horizon}d"]
        selected[f"excess_return_{horizon}d"] = selected[f"excess_{horizon}d"]
        selected[f"net_return_{horizon}d"] = selected[f"forward_{horizon}d"] - config.cost_bps / 10000.0
    ordered = columns + [c for c in selected.columns if c.startswith(("gross_return_", "benchmark_return_", "excess_return_", "net_return_"))]
    return selected[ordered].sort_values(["as_of", "rank", "symbol"])


def factor_diagnostics(library: pd.DataFrame, config: LoopConfig) -> dict:
    """Return IS/OOS IC, quantile spread and monotonicity without fitting a model."""
    horizon = f"forward_{min(config.horizons)}d"
    dates = sorted(library["date"].dropna().unique())
    split = max(1, int(len(dates) * 0.7))
    date_sets = {"is": set(dates[:split]), "oos": set(dates[split:])}
    results: list[dict] = []
    directions = dict(config.directions)
    for factor in config.factors:
        if factor not in library.columns:
            results.append({"factor": factor, "status": "missing"})
            continue
        split_metrics = {}
        for split_name, split_dates in date_sets.items():
            rows = []
            for date, group in library[library["date"].isin(split_dates)].groupby("date", sort=True):
                sample = group[[factor, horizon]].apply(pd.to_numeric, errors="coerce").dropna()
                if len(sample) < config.quantiles:
                    continue
                ic = sample[factor].corr(sample[horizon], method="spearman")
                sample = sample.assign(bucket=pd.qcut(sample[factor], config.quantiles, labels=False, duplicates="drop"))
                means = sample.groupby("bucket", observed=True)[horizon].mean()
                rows.append({"ic": ic, "bucket_means": means.to_dict()})
            ic = pd.Series([r["ic"] for r in rows], dtype=float).dropna()
            bucket_values = [r["bucket_means"] for r in rows]
            means = pd.DataFrame(bucket_values).mean().sort_index() if bucket_values else pd.Series(dtype=float)
            monotonic = None
            if len(means) >= 2:
                monotonic = float(means.corr(pd.Series(range(len(means)), index=means.index), method="spearman"))
                monotonic *= 1 if int(directions.get(factor, 1)) >= 0 else -1
            split_metrics[split_name] = {"observations": len(ic), "ic_mean": float(ic.mean()) if len(ic) else None,
                "ic_positive_rate": float((ic > 0).mean()) if len(ic) else None,
                "bucket_means": {str(k): float(v) for k, v in means.items()}, "monotonicity": monotonic}
        oos = split_metrics["oos"]
        passed = bool(oos["observations"] >= config.min_diagnostic_dates and oos["monotonicity"] is not None and oos["monotonicity"] >= config.min_monotonicity)
        results.append({"factor": factor, "status": "ok", "is": split_metrics["is"], "oos": oos,
                        "monotonicity": oos["monotonicity"], "ic_mean": oos["ic_mean"],
                        "quality_gate": "PASS" if passed else "HOLD",
                        "quality_gate_reason": None if passed else "oos_insufficient_dates_or_monotonicity"})
    valid = [item for item in results if item.get("status") == "ok"]
    overall = "PASS" if valid and len(valid) == len(results) and all(item.get("quality_gate") == "PASS" for item in valid) else "HOLD"
    return {"horizon": horizon, "quantiles": config.quantiles, "quality_gate": overall, "factors": results}


def paper_plan(library: pd.DataFrame, config: LoopConfig, *, as_of: str | None = None) -> dict:
    """Build a next-session paper plan; prices are references, never orders."""
    data = library.copy()
    date = pd.Timestamp(as_of) if as_of else data["date"].max()
    day = data[data["date"] == date].sort_values("composite_rank").head(config.top_n).copy()
    if day.empty:
        raise ValueError(f"no candidates for {date.date()}")
    next_rows = data[data["date"] > date].groupby("code", sort=False).head(1).set_index("code")
    plans = []
    for row in day.itertuples():
        next_row = next_rows.loc[row.code] if row.code in next_rows.index else None
        entry = getattr(next_row, "raw_open", np.nan) if next_row is not None else np.nan
        if pd.isna(entry):
            entry = getattr(next_row, "open", np.nan) if next_row is not None else np.nan
        atr = float(getattr(row, "atr14", np.nan)) if hasattr(row, "atr14") else np.nan
        if not np.isfinite(atr):
            atr = float(row.close) * 0.03
        entry = float(entry) if pd.notna(entry) else None
        reference = entry or float(row.close)
        plans.append({"candidate_id": hashlib.sha256(f"{row.factor_version}|{date.date()}|{row.code}".encode()).hexdigest()[:20],
                      "as_of": str(date.date()), "symbol": row.code, "rank": int(row.composite_rank),
                      "score": round(float(row.composite_score), 6), "signal_close": float(row.close),
                      "next_open_reference": entry, "entry_max_price": round(reference * 1.02, 4),
                      "stop_loss": round(reference - config.stop_atr * atr, 4),
                      "take_profit": round(reference + config.target_atr * atr, 4),
                      "time_exit_days": config.time_stop, "cost_bps": config.cost_bps,
                      "invalidity": ["stale_data", "market_halt", "factor_coverage_below_threshold"],
                      "confirmation": "manual", "confirmation_result": "PENDING", "status": "PAPER_ONLY"})
    return {"schema": "quant-paper-plan/v1", "as_of": str(date.date()), "strategy_version": _config_hash(config),
            "plans": plans, "disclaimer": "Research and paper execution only; no broker order is emitted."}


def replay_exit(bars: pd.DataFrame, entry_price: float, *, stop_price: float, target_price: float,
                max_days: int = 5) -> dict:
    """Deterministic daily-bar exit replay; conservative when stop and target hit together."""
    data = bars.sort_values("date").head(max_days)
    for day_number, (_, row) in enumerate(data.iterrows(), start=1):
        low, high = float(row["low"]), float(row["high"])
        if low <= stop_price:
            return {"exit_date": str(row["date"]), "exit_price": stop_price, "reason": "stop", "days": day_number}
        if high >= target_price:
            return {"exit_date": str(row["date"]), "exit_price": target_price, "reason": "target", "days": day_number}
    if len(data):
        row = data.iloc[-1]
        return {"exit_date": str(row["date"]), "exit_price": float(row["close"]), "reason": "time_stop", "days": len(data)}
    return {"exit_date": None, "exit_price": None, "reason": "no_bars", "days": 0}


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(path.glob("**/*")) if path.is_dir() else [path]:
        if file.is_file():
            digest.update(str(file.relative_to(path) if path.is_dir() else file.name).encode())
            digest.update(file.read_bytes())
    return digest.hexdigest()


def _config_hash(config: LoopConfig) -> str:
    raw = json.dumps(asdict(config), sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def run_loop(data_path: str | Path, output_dir: str | Path, config: LoopConfig, *, release_id: str = "local") -> dict:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    source = Path(data_path)
    data_sha256 = _file_hash(source)
    library = candidate_library(load_panel(source), config)
    diagnostics = factor_diagnostics(library, config)
    performance = candidate_performance(library, config)
    plan = paper_plan(library, config)
    if diagnostics["quality_gate"] != "PASS":
        plan["status"] = "BLOCKED_BY_QUALITY_GATE"
        plan["disclaimer"] = "Paper plan is blocked until OOS factor quality gate passes; manual research only."
    store_count = append_candidates(output / "candidate_store.db", performance, plan, release_id=release_id, data_sha256=data_sha256, quality_gate=diagnostics["quality_gate"])
    library.to_parquet(output / "candidate_library.parquet", index=False)
    performance.to_parquet(output / "candidate_performance.parquet", index=False)
    diagnostics_path = output / "factor_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (output / "paper_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    manifest = {"schema": "quant-closed-loop/v1", "strategy_version": _config_hash(config),
                "config": asdict(config), "release_id": release_id, "data_sha256": data_sha256,
                "rows": len(library), "dates": int(library.date.nunique()),
                "selected_rows": int(library.selected.sum()), "performance_rows": len(performance), "store_rows": store_count,
                "quality_gate": diagnostics["quality_gate"],
                "artifacts": ["candidate_library.parquet", "candidate_performance.parquet", "factor_diagnostics.json", "paper_plan.json"]}
    manifest["sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True, default=str).encode()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"manifest": manifest, "diagnostics": diagnostics, "paper_plan": plan}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--factors", nargs="+", required=True); parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 5, 10]); parser.add_argument("--release-id", default="local")
    args = parser.parse_args(argv)
    result = run_loop(args.data, args.output_dir, LoopConfig(tuple(args.factors), top_n=args.top_n, horizons=tuple(args.horizons)), release_id=args.release_id)
    print(json.dumps({"output_dir": args.output_dir, **result["manifest"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
