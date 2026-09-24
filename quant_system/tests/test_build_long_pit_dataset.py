from __future__ import annotations

from quant_system.build_long_pit_dataset import _balanced_codes


def test_balanced_codes_returns_requested_count_without_star_overallocation():
    codes = [f"{value:06d}" for value in range(1, 800)] + [f"300{value:03d}" for value in range(200)] + [f"688{value:03d}" for value in range(200)]
    selected = _balanced_codes(codes, 500)
    assert len(selected) == 500
    assert len(set(selected)) == 500
    assert sum(code.startswith("68") for code in selected) == 75
