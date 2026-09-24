#!/usr/bin/env python3
"""Shared time-boundary assertions for IC/OOS/ML/backtest pipelines."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class TimeWindow:
    train_start: date
    train_end: date
    oos_start: date
    oos_end: date
    embargo_days: int = 0

    def validate(self) -> None:
        if self.embargo_days < 0:
            raise ValueError("embargo_days must be >= 0")
        if not self.train_start <= self.train_end:
            raise ValueError("train window is inverted")
        if not self.oos_start <= self.oos_end:
            raise ValueError("oos window is inverted")
        if self.train_end >= self.oos_start:
            raise ValueError("train labels overlap OOS start")


def assert_feature_asof(*, feature_asof: date, signal_date: date) -> None:
    if feature_asof > signal_date:
        raise ValueError(f"future_feature:{feature_asof}>{signal_date}")


def assert_label_end(*, label_end: date, oos_start: date) -> None:
    if label_end >= oos_start:
        raise ValueError(f"future_label:{label_end}>={oos_start}")


def split_metadata(window: TimeWindow, *, feature_asof: date, label_end: date,
                   code_sha: str = "") -> dict:
    window.validate()
    assert_feature_asof(feature_asof=feature_asof, signal_date=window.oos_start)
    assert_label_end(label_end=label_end, oos_start=window.oos_start)
    return {"train_start": window.train_start.isoformat(), "train_end": window.train_end.isoformat(),
            "oos_start": window.oos_start.isoformat(), "oos_end": window.oos_end.isoformat(),
            "embargo_days": window.embargo_days, "feature_asof": feature_asof.isoformat(),
            "label_end": label_end.isoformat(), "code_sha": code_sha}
