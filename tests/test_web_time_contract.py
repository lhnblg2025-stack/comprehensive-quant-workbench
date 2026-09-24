import pandas as pd

from quant_web.server import _records


def test_records_preserve_intraday_timestamp_precision():
    rows = _records(pd.DataFrame({"date": [pd.Timestamp("2026-08-27 09:35:00")], "close": [10.5]}))
    assert rows[0]["date"] == "2026-08-27 09:35:00"


def test_records_keep_daily_timestamp_as_date():
    rows = _records(pd.DataFrame({"date": [pd.Timestamp("2026-08-27 00:00:00")]}))
    assert rows[0]["date"] == "2026-08-27"
