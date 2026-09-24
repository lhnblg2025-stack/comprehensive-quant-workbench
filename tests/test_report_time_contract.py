import pandas as pd

from scripts.html_report_generator import _as_of_frame


def test_report_history_excludes_future_observations():
    frame = pd.DataFrame({
        "date": ["2026-08-20", "2026-08-21", "2026-08-22"],
        "value": [1, 2, 3],
    })
    result = _as_of_frame(frame, "2026-08-21")
    assert result["value"].tolist() == [1, 2]
    assert result["date"].max() == pd.Timestamp("2026-08-21")


def test_report_history_requires_date_column():
    frame = pd.DataFrame({"value": [1, 2]})
    assert _as_of_frame(frame, "2026-08-21").empty
