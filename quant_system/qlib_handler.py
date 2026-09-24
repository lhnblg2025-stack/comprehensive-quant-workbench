"""Qlib DataHandler contract for the staged A-share PIT provider.

The module is importable without pyqlib so configuration validation can report a
missing runtime. Instantiation requires a real Qlib installation and never falls
back to a private handler silently.
"""
from __future__ import annotations

try:
    from qlib.data.dataset.handler import DataHandlerLP  # type: ignore
except ImportError:
    DataHandlerLP = None  # type: ignore


if DataHandlerLP is not None:
    class ASharePITHandler(DataHandlerLP):
        """Daily OHLCV plus annual PIT value/quality factors."""

        def __init__(self, instruments="all", start_time=None, end_time=None, infer_processors=None, learn_processors=None, **kwargs):
            fields = [
                "$open", "$high", "$low", "$close", "$volume", "$amount",
                "$value_book_to_price_industry_neutral",
                "$quality_low_leverage_industry_neutral",
                "$value_quality_composite_industry_neutral",
            ]
            names = ["open", "high", "low", "close", "volume", "amount", "value", "quality", "value_quality"]
            data_loader = {
                "class": "QlibDataLoader",
                "kwargs": {"config": {"feature": (fields, names)}},
            }
            super().__init__(instruments=instruments, start_time=start_time, end_time=end_time, data_loader=data_loader, infer_processors=infer_processors or [], learn_processors=learn_processors or [], **kwargs)
else:
    class ASharePITHandler:  # pragma: no cover - behavior exercised without qlib
        def __init__(self, *args, **kwargs):
            raise RuntimeError("pyqlib_runtime_missing; install the vendored pyqlib environment before constructing ASharePITHandler")
