import pandas as pd

from scripts.backfill_mainboard_free_history import BackfillConfig, is_mainboard_code, validate_frame


def test_mainboard_scope_includes_main_board_and_excludes_growth_boards():
    assert is_mainboard_code('000001')
    assert is_mainboard_code('002594')
    assert is_mainboard_code('600000')
    assert is_mainboard_code('605001')
    assert not is_mainboard_code('300750')
    assert not is_mainboard_code('688001')
    assert not is_mainboard_code('830001')


def test_backfill_validation_requires_two_year_history():
    dates = pd.date_range('2016-01-04', periods=400, freq='B')
    frame = pd.DataFrame({
        'date': dates,
        'code': '000001',
        'open': 10.0,
        'high': 10.2,
        'low': 9.8,
        'close': 10.1,
        'volume': 1000,
    })
    assert validate_frame(frame, '000001', BackfillConfig()) == []


def test_backfill_validation_blocks_short_history():
    dates = pd.date_range('2016-01-04', periods=20, freq='B')
    frame = pd.DataFrame({
        'date': dates,
        'code': '000001',
        'open': 10.0,
        'high': 10.2,
        'low': 9.8,
        'close': 10.1,
        'volume': 1000,
    })
    assert any(item.startswith('too_few_rows') for item in validate_frame(frame, '000001', BackfillConfig()))
