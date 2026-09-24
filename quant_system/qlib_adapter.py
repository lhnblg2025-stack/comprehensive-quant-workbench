"""Qlib-first integration boundary for the A-share research system.

Qlib is the research, factor, dataset, model, portfolio and evaluation主体.
This module only adapts our PIT/price panel to Qlib's interfaces. Backtrader
remains an optional order-level execution/reconciliation adapter and is never
used as the research-framework authority.

The adapter is importable without Qlib so the web UI can report an actionable
missing-dependency state instead of silently falling back to a private engine.
"""
from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = ROOT / "data_warehouse" / "research_panels" / "ashare_tushare_500_10y_v2" / "price_discovery_panel.parquet"
DEFAULT_OUTPUT = ROOT / "generated" / "runs" / "qlib_research"

QLIB_PACKAGE = "qlib"


@dataclass(frozen=True)
class QlibStatus:
    installed: bool
    version: str | None
    panel: str
    panel_exists: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": "microsoft_qlib",
            "installed": self.installed,
            "version": self.version,
            "panel": self.panel,
            "panel_exists": self.panel_exists,
            "authority": "qlib",
            "execution_adapter": "backtrader_optional",
        }


def status(panel: str | Path = DEFAULT_PANEL) -> dict[str, Any]:
    spec = importlib.util.find_spec(QLIB_PACKAGE)
    version = None
    if spec is not None:
        try:
            import qlib  # type: ignore
            version = getattr(qlib, "__version__", None)
        except Exception:
            version = None
    return QlibStatus(spec is not None, version, str(panel), Path(panel).is_file()).to_dict()


def _require_qlib() -> None:
    if importlib.util.find_spec(QLIB_PACKAGE) is None:
        raise RuntimeError("qlib_not_installed; install microsoft qlib before running the qlib research workflow")


def normalize_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Normalize our long panel to Qlib's (datetime, instrument) convention."""
    required = {"date", "code"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"qlib_panel_missing:{','.join(missing)}")
    data = panel.copy()
    data["datetime"] = pd.to_datetime(data.pop("date"), errors="coerce")
    data["instrument"] = data.pop("code").astype(str).str.zfill(6)
    data = data.dropna(subset=["datetime", "instrument"]).sort_values(["datetime", "instrument"])
    # Qlib uses $field expressions; retain ordinary names in the interchange
    # parquet and expose an explicit mapping for custom DataHandler code.
    if "raw_close" in data and "close" not in data:
        data["close"] = pd.to_numeric(data["raw_close"], errors="coerce")
    if "raw_open" in data and "open" not in data:
        data["open"] = pd.to_numeric(data["raw_open"], errors="coerce")
    return data


def materialize_qlib_parquet(panel_path: str | Path = DEFAULT_PANEL, output_dir: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Materialize a Qlib-friendly long parquet and manifest.

    This conversion is independent of Qlib's binary dump format. It allows a
    Qlib DataHandler/Provider to be configured against a stable, auditable
    source while preserving the original data release.
    """
    source = Path(panel_path)
    if not source.is_file():
        raise FileNotFoundError(f"panel_not_found:{source}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = normalize_panel(pd.read_parquet(source))
    target = output / "qlib_panel.parquet"
    data.to_parquet(target, index=False)
    manifest = {
        "schema": "qlib_adapter_manifest/v1",
        "framework": "microsoft_qlib",
        "source_panel": str(source.relative_to(ROOT) if source.is_relative_to(ROOT) else source),
        "output": str(target.relative_to(ROOT) if target.is_relative_to(ROOT) else target),
        "rows": int(len(data)),
        "symbols": int(data["instrument"].nunique()),
        "date_min": str(data["datetime"].min().date()) if len(data) else None,
        "date_max": str(data["datetime"].max().date()) if len(data) else None,
        "fields": [c for c in data.columns if c not in {"datetime", "instrument"}],
        "qlib_required": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    return manifest


def run_qlib_workflow(*, panel_path: str | Path = DEFAULT_PANEL, config_path: str | Path | None = None) -> dict[str, Any]:
    """Run a Qlib workflow after dependency/config validation.

    The config is intentionally external: Qlib's official workflow YAML is the
    source of truth for handler, dataset, model, strategy, backtest and
    recorder composition. We do not duplicate Qlib internals here.
    """
    _require_qlib()
    source = Path(panel_path)
    if config_path is None:
        raise ValueError("qlib_config_required")
    config = Path(config_path)
    if not config.is_file():
        raise FileNotFoundError(f"qlib_config_not_found:{config}")
    # Import only after the explicit dependency check.
    import qlib  # type: ignore
    from qlib.workflow import R  # type: ignore
    qlib.init(provider_uri=str(source.parent), region="cn")
    # Official Qlib workflows are normally executed by qrun. Keeping this
    # callable small makes it suitable for the web task queue and tests.
    from qlib.utils import init_instance_by_config  # type: ignore
    spec = json.loads(config.read_text(encoding="utf-8")) if config.suffix == ".json" else None
    if spec is None:
        raise ValueError("qlib_adapter_requires_json_workflow_config_for_api")
    with R.start(experiment_name=spec.get("experiment_name", "ashare_qlib_research")):
        recorder = init_instance_by_config(spec["recorder"])
        return {"status": "configured", "framework": "microsoft_qlib", "recorder": str(recorder)}
