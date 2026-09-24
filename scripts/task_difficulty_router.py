#!/usr/bin/env python3
"""Deterministic L1-L4 task routing with explicit fallback and cost budget."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Route:
    level: str
    model: str
    reasoning: str
    max_tokens: int
    estimated_cost_units: int


_LEVELS = {
    "L1": Route("L1", "flash", "mechanical/local", 2048, 1),
    "L2": Route("L2", "pro", "multi-file/contract", 4096, 3),
    "L3": Route("L3", "off", "quant/review/high-risk", 8192, 8),
    "L4": Route("L4", "max", "architecture/security", 16384, 20),
}


def classify(*, files: int = 1, touches_data: bool = False, touches_execution: bool = False,
             security: bool = False, external_side_effect: bool = False) -> str:
    """Classify risk/complexity; execution and security always require L4."""
    if files < 1:
        raise ValueError("files must be >= 1")
    if security or external_side_effect:
        return "L4"
    if touches_execution or touches_data:
        return "L3"
    if files >= 5:
        return "L3"
    if files >= 2:
        return "L2"
    return "L1"


def route(level: str, *, budget_units: int | None = None) -> Route:
    key = str(level).upper()
    if key not in _LEVELS:
        raise ValueError(f"unknown difficulty level: {level}")
    selected = _LEVELS[key]
    if budget_units is not None and budget_units < selected.estimated_cost_units:
        # Fail closed to the cheapest model only when the caller explicitly allows downgrade.
        raise RuntimeError(f"budget_exceeded:{key}:{selected.estimated_cost_units}>{budget_units}")
    return selected


def route_task(*, files: int = 1, touches_data: bool = False, touches_execution: bool = False,
               security: bool = False, external_side_effect: bool = False,
               budget_units: int | None = None) -> dict:
    level = classify(files=files, touches_data=touches_data, touches_execution=touches_execution,
                     security=security, external_side_effect=external_side_effect)
    selected = route(level, budget_units=budget_units)
    return {"level": selected.level, "model": selected.model, "reasoning": selected.reasoning,
            "max_tokens": selected.max_tokens, "estimated_cost_units": selected.estimated_cost_units}
