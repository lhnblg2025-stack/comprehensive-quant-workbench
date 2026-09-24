"""Single source of truth for the personal research platform release contract."""
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Any

PRODUCT_NAME = "综合量化研究平台"
PRODUCT_VERSION = "v0.1"
CONTRACT_SCHEMA = "quant-production-contract/v1"

PRODUCTION_COMPONENTS: dict[str, dict[str, str]] = {
    "factor_registry": {"module": "quant_system.ic_factors.registry"},
    "factor_quality_registry": {"path": "generated/factor_quality_registry.json"},
    "portfolio_builder": {"module": "quant_system.strategy_engine", "attribute": "StrategyEngine"},
    "research_backtester": {"module": "quant_system.factor_backtest_runner", "attribute": "run"},
    "execution_backtester": {"module": "quant_system.backtest_engine", "attribute": "BacktestEngine"},
    "production_pipeline": {"module": "quant_system.production_pipeline", "attribute": "run"},
    "daily_fusion": {"module": "quant_system.analysis_core.fusion", "attribute": "fuse_today"},
    "daily_review": {"path": "scripts/daily_review_chain.py"},
    "html_renderer": {"path": "scripts/html_report_generator.py"},
}

RESEARCH_ONLY_COMPONENTS: tuple[str, ...] = (
    "quant_system.factor_system",
    "quant_system.factor_model",
    "quant_system.factor_zoo",
    "quant_system.ic_factors.zoo",
)


def release_metadata() -> dict[str, Any]:
    return {
        "product": PRODUCT_NAME,
        "product_version": PRODUCT_VERSION,
        "production_contract": CONTRACT_SCHEMA,
    }


def validate_production_contract(root: str | Path | None = None) -> dict[str, Any]:
    workspace = Path(root) if root is not None else Path(__file__).resolve().parent.parent
    errors: list[str] = []
    for name, spec in PRODUCTION_COMPONENTS.items():
        path = spec.get("path")
        if path:
            if not (workspace / path).exists():
                errors.append(f"{name}:missing:{path}")
            continue
        module_name = spec.get("module", "")
        if not module_name or importlib.util.find_spec(module_name) is None:
            errors.append(f"{name}:missing_module:{module_name}")
            continue
        attribute = spec.get("attribute")
        if attribute:
            try:
                module = importlib.import_module(module_name)
                if not hasattr(module, attribute):
                    errors.append(f"{name}:missing_attribute:{module_name}.{attribute}")
            except Exception as exc:
                errors.append(f"{name}:import_error:{type(exc).__name__}:{str(exc)[:80]}")
    return {
        **release_metadata(),
        "status": "PASS" if not errors else "BLOCK",
        "components": dict(PRODUCTION_COMPONENTS),
        "research_only": list(RESEARCH_ONLY_COMPONENTS),
        "errors": errors,
    }
