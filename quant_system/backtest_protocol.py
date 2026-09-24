"""Versioned, engine-neutral canonical backtest protocol."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

SCHEMA_VERSION = "canonical-backtest/v1"
SUPPORTED_STRATEGY_KINDS = {"cross_sectional_signal", "target_weights"}


def _finite(value: Any, name: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    import math

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}_must_be_numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name}_must_be_finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name}_below_minimum:{minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name}_above_maximum:{maximum}")
    return number


@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 0.85
    stamp_duty_bps: float = 5.0
    transfer_fee_bps: float = 0.1
    slippage_bps: float = 10.0
    min_commission: float = 5.0
    max_adv_participation: float = 0.10

    def validate(self) -> "CostModel":
        for name in ("commission_bps", "stamp_duty_bps", "transfer_fee_bps", "slippage_bps", "min_commission"):
            _finite(getattr(self, name), name, minimum=0.0)
        _finite(self.max_adv_participation, "max_adv_participation", minimum=0.0, maximum=1.0)
        return self


@dataclass(frozen=True)
class RiskPolicy:
    max_position_weight: float = 0.10
    max_gross_exposure: float = 1.0
    max_names: int = 50
    lot_size: int = 100
    long_only: bool = True

    def validate(self) -> "RiskPolicy":
        _finite(self.max_position_weight, "max_position_weight", minimum=0.000001, maximum=1.0)
        _finite(self.max_gross_exposure, "max_gross_exposure", minimum=0.0, maximum=1.0)
        if int(self.max_names) < 1:
            raise ValueError("max_names_below_minimum:1")
        if int(self.lot_size) < 1:
            raise ValueError("lot_size_below_minimum:1")
        if not self.long_only:
            raise ValueError("canonical_engine_long_only")
        return self


@dataclass(frozen=True)
class StrategySpec:
    kind: str = "cross_sectional_signal"
    family: str = "cross_section"
    signal: str | None = None
    direction: int = 1
    quantile: float = 0.20
    rebalance_sessions: int = 5
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> "StrategySpec":
        if self.kind not in SUPPORTED_STRATEGY_KINDS:
            raise ValueError(f"unsupported_strategy_kind:{self.kind}")
        if self.kind == "cross_sectional_signal" and not self.signal:
            raise ValueError("signal_required")
        if int(self.direction) not in {-1, 1}:
            raise ValueError("direction_must_be_minus_one_or_one")
        _finite(self.quantile, "quantile", minimum=0.01, maximum=1.0)
        if int(self.rebalance_sessions) < 1:
            raise ValueError("rebalance_sessions_below_minimum:1")
        return self


@dataclass(frozen=True)
class BacktestRequest:
    strategy: StrategySpec
    costs: CostModel = field(default_factory=CostModel)
    risk: RiskPolicy = field(default_factory=RiskPolicy)
    initial_capital: float = 1_000_000.0
    mode: str = "research"
    dataset_id: str | None = None
    pit_release_id: str | None = None
    corporate_action_release_id: str | None = None
    request_id: str | None = None
    schema: str = SCHEMA_VERSION

    def validate(self) -> "BacktestRequest":
        if self.schema != SCHEMA_VERSION:
            raise ValueError(f"unsupported_backtest_schema:{self.schema}")
        if self.mode not in {"research", "paper", "production"}:
            raise ValueError(f"unsupported_backtest_mode:{self.mode}")
        _finite(self.initial_capital, "initial_capital", minimum=1.0)
        self.strategy.validate()
        self.costs.validate()
        self.risk.validate()
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BacktestRequest":
        strategy_raw = dict(raw.get("strategy") or {})
        costs_raw = dict(raw.get("costs") or {})
        risk_raw = dict(raw.get("risk") or {})
        request = cls(
            strategy=StrategySpec(**strategy_raw),
            costs=CostModel(**costs_raw),
            risk=RiskPolicy(**risk_raw),
            initial_capital=raw.get("initial_capital", 1_000_000.0),
            mode=str(raw.get("mode") or "research"),
            dataset_id=raw.get("dataset_id"),
            pit_release_id=raw.get("pit_release_id"),
            corporate_action_release_id=raw.get("corporate_action_release_id"),
            request_id=raw.get("request_id"),
            schema=str(raw.get("schema") or SCHEMA_VERSION),
        )
        return request.validate()


@dataclass
class UnifiedBacktestResult:
    engine: str
    status: str
    initial_capital: float
    final_value: float
    total_return: float | None
    annual_return: float | None
    sharpe: float | None
    max_drawdown: float | None
    observations: int
    orders: int
    trades: int
    rejected_orders: int = 0
    blocked_orders: int = 0
    total_fees: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    schema: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_results(primary: UnifiedBacktestResult, secondary: UnifiedBacktestResult, *, return_tolerance: float = .02, final_value_tolerance: float = .02) -> dict[str, Any]:
    return_gap = None if primary.total_return is None or secondary.total_return is None else abs(primary.total_return - secondary.total_return)
    denominator = max(abs(primary.final_value), 1.0)
    value_gap = abs(primary.final_value - secondary.final_value) / denominator
    errors = []
    if return_gap is not None and return_gap > return_tolerance:
        errors.append("total_return_gap")
    if value_gap > final_value_tolerance:
        errors.append("final_value_gap")
    return {"status": "PASS" if not errors else "BLOCK", "primary_engine": primary.engine, "secondary_engine": secondary.engine, "return_gap": return_gap, "relative_final_value_gap": value_gap, "errors": errors}
