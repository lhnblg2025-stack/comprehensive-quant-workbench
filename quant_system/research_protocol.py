"""Shared research protocol for the long-horizon strategy matrix."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchProtocol:
    """Explicit sample and chronological split requirements.

    ``test_ratio`` is expressed as a fraction of the complete window, so a
    5:3 train/test split is 3/8 (37.5%) out of sample.
    """

    universe_size: int = 800
    years: int = 8
    train_years: int = 5
    test_years: int = 3

    @property
    def test_ratio(self) -> float:
        return self.test_years / (self.train_years + self.test_years)

    @property
    def train_ratio(self) -> float:
        return self.train_years / (self.train_years + self.test_years)

    def validate(self) -> None:
        if self.universe_size < 800:
            raise ValueError("universe_size_must_be_at_least_800")
        if self.years < 8 or self.train_years < 5 or self.test_years < 3:
            raise ValueError("research_window_must_be_at_least_8y_with_5_3_split")
        if self.train_years + self.test_years != self.years:
            raise ValueError("train_test_years_must_equal_total_years")


DEFAULT_PROTOCOL = ResearchProtocol()
DEFAULT_PROTOCOL.validate()
