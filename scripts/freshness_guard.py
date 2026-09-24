#!/usr/bin/env python3
"""盘后新鲜度守卫：关键日频数据必须滚动到最新交易日。"""
import sys
from pathlib import Path
import pandas as pd
try:
    day = sys.argv[1]
except IndexError:
    print('usage: freshness_guard.py YYYY-MM-DD'); raise SystemExit(2)
root = Path('.')
try:
    expected = pd.Timestamp(day).date()
except Exception as e:
    print(f'日期非法: {day} ({e})'); raise SystemExit(2)
checks = {
    '上证指数': root/'data_warehouse/market/index_daily_上证指数.parquet',
    '沪深300': root/'data_warehouse/market/index_daily_沪深300.parquet',
    '涨停统计': root/'data_warehouse/market/zt_daily_stats.parquet',
    '融合温度': root/'data_warehouse/market/fusion.parquet',
}
errors = []
for name, path in checks.items():
    if not path.exists():
        errors.append(f'{name}:missing'); print(f'{name}: missing'); continue
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        errors.append(f'{name}:read-fail'); print(f'{name}: read-fail {e}'); continue
    if 'date' not in df.columns or df.empty:
        errors.append(f'{name}:no-date'); print(f'{name}: no-date'); continue
    actual = pd.to_datetime(df['date']).max().date()
    print(f'{name}: {actual}')
    if actual > expected:
        errors.append(f'{name}:{actual}>expected:{expected}'); print(f'{name}: FUTURE DATE {actual} > {expected}')
    elif actual < expected:
        errors.append(f'{name}:{actual}<expected:{expected}')
if errors:
    raise SystemExit('关键数据未滚动: ' + '; '.join(errors))
print('freshness ok:', expected)
