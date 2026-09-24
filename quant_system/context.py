"""Lightweight execution contexts shared by research and execution paths.

The contexts are intentionally small adapters: existing callers may continue to
pass strings, dates, datetimes, or ``None``.  They provide one canonical ISO date
and an optional run id without imposing a broad dependency-injection refactor.
"""
from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Mapping

_DATE_RE = re.compile(r"^20\d{2}-\d{2}-\d{2}$")
_RUN_RE = re.compile(r"^[A-Za-z0-9._-]{1,160}$")


def _iso_date(value: str | date | datetime | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        value = value.date()
    elif isinstance(value, str):
        value = value.strip()
        if not _DATE_RE.fullmatch(value):
            raise ValueError(f"as_of must be YYYY-MM-DD: {value!r}")
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError as exc:
            raise ValueError(f"as_of is not a real date: {value!r}") from exc
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"unsupported as_of type: {type(value).__name__}")


@dataclass(frozen=True)
class AsOfContext:
    """Canonical analysis cutoff date, preserving old string-style APIs."""

    value: str | date | datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _iso_date(self.value))

    @property
    def iso_date(self) -> str | None:
        return self.value

    def __str__(self) -> str:
        return self.value or ""

    def __bool__(self) -> bool:
        return self.value is not None

    def as_dict(self) -> dict[str, str | None]:
        return {"as_of": self.value}


@dataclass(frozen=True)
class RunContext:
    """Small run envelope for propagating ``run_id`` and ``as_of``."""

    run_id: str | None = None
    as_of: AsOfContext | str | date | datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        run_id = self.run_id
        if run_id is not None:
            run_id = str(run_id).strip()
            if not _RUN_RE.fullmatch(run_id):
                raise ValueError("run_id must contain only letters, digits, '.', '_' or '-'")
        as_of = self.as_of if isinstance(self.as_of, AsOfContext) else AsOfContext(self.as_of)
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "as_of", as_of)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "RunContext":
        source = os.environ if env is None else env
        return cls(run_id=source.get("QUANT_RUN_ID") or None, as_of=source.get("QUANT_AS_OF") or None)

    @classmethod
    def create(cls, as_of: str | date | datetime | None = None, run_id: str | None = None, **metadata: Any) -> "RunContext":
        return cls(run_id=run_id or f"run-{uuid.uuid4().hex[:12]}", as_of=as_of, metadata=metadata)

    @property
    def iso_date(self) -> str | None:
        return self.as_of.iso_date  # type: ignore[union-attr]

    def child_env(self, base: Mapping[str, str] | None = None) -> dict[str, str]:
        env = dict(os.environ if base is None else base)
        if self.run_id:
            env["QUANT_RUN_ID"] = self.run_id
        if self.iso_date:
            env["QUANT_AS_OF"] = self.iso_date
        return env

    def as_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "as_of": self.iso_date, "metadata": dict(self.metadata)}


def coerce_as_of(value: AsOfContext | str | date | datetime | None) -> AsOfContext:
    return value if isinstance(value, AsOfContext) else AsOfContext(value)


def coerce_run_context(value: RunContext | None = None, *, as_of: AsOfContext | str | date | datetime | None = None, run_id: str | None = None) -> RunContext:
    if value is None:
        return RunContext(run_id=run_id, as_of=as_of)
    if as_of is None and run_id is None:
        return value
    return RunContext(run_id=run_id or value.run_id, as_of=as_of or value.as_of, metadata=value.metadata)
