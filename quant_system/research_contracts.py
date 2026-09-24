"""Stable contracts for reproducible quantitative research runs."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


@dataclass(frozen=True)
class MarketPanel:
    frame: pd.DataFrame
    calendar: pd.DatetimeIndex
    price_field: str = "close"
    source_version: str = "unknown"

    def __post_init__(self):
        required = {"date", "code", self.price_field}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"MarketPanel missing columns: {sorted(missing)}")
        if self.frame.duplicated(["date", "code"]).any():
            raise ValueError("MarketPanel has duplicate (date, code) keys")

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(sorted(pd.to_datetime(self.frame["date"]).unique()))


@dataclass(frozen=True)
class FactorSpec:
    name: str
    source: str = "factor_zoo"
    version: str = "1.0.0"
    direction: int = 1
    availability_lag_days: int = 1
    active: bool = True


@dataclass(frozen=True)
class ForwardReturn:
    frame: pd.DataFrame
    horizon: int
    entry: str = "next_trading_close"
    exit: str = "horizon_close"

    def __post_init__(self):
        required = {"signal_date", "code", "forward_return"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"ForwardReturn missing columns: {sorted(missing)}")


@dataclass(frozen=True)
class QualityResult:
    status: str
    checks: dict[str, Any]
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def make_forward_return(frame: pd.DataFrame, horizon: int = 1, calendar: Iterable[Any] | None = None) -> ForwardReturn:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    required = {"date", "code", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"forward return input missing columns: {sorted(missing)}")
    data = frame[["date", "code", "close"]].copy()
    data["date"] = pd.to_datetime(data["date"])
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(list(calendar)).unique())) if calendar is not None else pd.DatetimeIndex(sorted(data["date"].dropna().unique()))
    if len(dates) <= horizon:
        out = pd.DataFrame(columns=["signal_date", "code", "entry_price", "exit_price", "forward_return", "data_status"])
        return ForwardReturn(out, horizon)
    next_map = {dates[i]: dates[i + horizon] for i in range(len(dates) - horizon)}
    entry = data.rename(columns={"date": "signal_date", "close": "entry_price"})
    lookup = data.rename(columns={"date": "exit_date", "close": "exit_price"})
    entry["exit_date"] = entry["signal_date"].map(next_map)
    out = entry.merge(lookup, on=["exit_date", "code"], how="left")
    out["forward_return"] = out["exit_price"] / out["entry_price"] - 1.0
    out["data_status"] = out["exit_price"].notna().map({True: "ok", False: "missing_exit"})
    return ForwardReturn(out[["signal_date", "code", "entry_price", "exit_price", "forward_return", "data_status"]], horizon)


def validate_panel(frame: pd.DataFrame, calendar: Iterable[Any] | None = None) -> QualityResult:
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {"rows": int(len(frame)), "dates": int(frame["date"].nunique()) if "date" in frame else 0}
    required = {"date", "code", "close"}
    missing = sorted(required - set(frame.columns))
    if missing: errors.append(f"missing_columns:{','.join(missing)}")
    if not missing:
        dates = pd.to_datetime(frame["date"], errors="coerce")
        close = pd.to_numeric(frame["close"], errors="coerce")
        checks["invalid_dates"] = int(dates.isna().sum()); checks["non_positive_close"] = int((close <= 0).sum()); checks["duplicate_keys"] = int(frame.duplicated(["date", "code"]).sum())
        if checks["invalid_dates"]: errors.append("invalid_dates")
        if checks["non_positive_close"]: errors.append("non_positive_close")
        if checks["duplicate_keys"]: errors.append("duplicate_keys")
        if calendar is not None:
            cal = set(pd.to_datetime(list(calendar)).normalize())
            non_trading = int((~dates.dt.normalize().isin(cal)).sum())
            checks["non_trading_dates"] = non_trading
            if non_trading: warnings.append("non_trading_dates")
        if frame["code"].nunique() < 2: warnings.append("small_cross_section")
    status = "BLOCK" if errors else "WARN" if warnings else "PASS"
    return QualityResult(status, checks, tuple(errors), tuple(warnings))


def resolve_factor_specs(config: dict[str, Any]) -> tuple[FactorSpec, ...]:
    section = config.get("factors", {})
    names = section.get("names", [])
    directions = section.get("directions", {})
    versions = section.get("versions", {})
    source = str(section.get("registry", "factor_zoo"))
    specs = tuple(FactorSpec(name=str(name), source=source, version=str(versions.get(name, "1.0.0")), direction=int(directions.get(name, 1)), availability_lag_days=int(section.get("availability_lag_days", 1))) for name in names)
    if any(spec.direction not in (-1, 1) for spec in specs):
        raise ValueError("factor directions must be -1 or 1")
    return specs


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def write_manifest(output: str | Path, *, config: dict[str, Any], inputs: Iterable[str | Path], quality: QualityResult, seed: int = 42, root: Path | None = None, artifacts: Iterable[str | Path] = ()) -> dict[str, Any]:
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    files = [{"path": str(Path(p)), "sha256": sha256_file(p)} for p in inputs if Path(p).is_file()]
    artifact_files = [{"path": str(Path(p)), "bytes": Path(p).stat().st_size, "sha256": sha256_file(p)} for p in artifacts if Path(p).is_file()]
    payload: dict[str, Any] = {"schema_version": "1.0", "experiment_id": config.get("experiment", {}).get("id", out.name), "git_commit": git_commit(root), "python_version": sys.version, "platform": platform.platform(), "random_seed": seed, "config": config, "input_files": files, "artifacts": artifact_files, "quality_gate": asdict(quality), "result_status": quality.status}
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str).encode()
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    (out / "experiment_manifest.json").write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=str), encoding="utf-8")
    return payload
