"""df_guard 空 DataFrame 形状守卫单元测试。"""
from __future__ import annotations

import pytest
import pandas as pd

import quant_system.df_guard as guard


@pytest.fixture(autouse=True)
def strict_on(monkeypatch):
    monkeypatch.setenv("DF_GUARD_STRICT", "1")


def test_none_without_allow_empty_returns_none():
    assert guard.guard_df(None, name="prices") is None


def test_none_with_allow_empty_raises():
    with pytest.raises(guard.EmptyDataError, match="prices"):
        guard.guard_df(None, name="prices", allow_empty=True)


def test_non_dataframe_raises_type_error():
    with pytest.raises(TypeError, match="pandas.DataFrame"):
        guard.guard_df({"close": [1]}, name="prices")


def test_empty_dataframe_raises():
    df = pd.DataFrame({"close": []})
    with pytest.raises(guard.EmptyDataError, match="prices 为空"):
        guard.guard_df(df, name="prices")


def test_empty_dataframe_allowed_returns_original():
    df = pd.DataFrame({"close": []})
    assert guard.guard_df(df, name="prices", allow_empty=True) is df


def test_missing_required_columns_are_listed():
    df = pd.DataFrame({"open": [1.0]})
    with pytest.raises(guard.EmptyDataError) as exc:
        guard.guard_df(df, name="bars", require_cols=("close", "volume"))

    message = str(exc.value)
    assert "close" in message
    assert "volume" in message


def test_normal_dataframe_returns_original_without_copy():
    df = pd.DataFrame({"close": [1.0], "volume": [100]})
    assert guard.guard_df(df, name="bars", require_cols=("close", "volume")) is df


def test_strict_disabled_returns_input_unchanged(monkeypatch):
    monkeypatch.setenv("DF_GUARD_STRICT", "0")
    empty = pd.DataFrame()

    assert guard.guard_df(empty, name="bars", require_cols=("close",)) is empty
    assert guard.guard_df(None, name="bars") is None
