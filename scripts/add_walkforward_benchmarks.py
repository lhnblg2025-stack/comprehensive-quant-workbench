from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'generated/weekly_walkforward_20260922'
segments = pd.read_csv(OUT / 'walkforward_segments.csv')
segments['test_year'] = segments['test_year'].astype(str)
paths = {
    'mainboard': ROOT / 'data_warehouse/market/index_daily_上证指数.parquet',
    'all_board': ROOT / 'data_warehouse/market/index_daily_沪深300.parquet',
}
bench = {}
for universe, path in paths.items():
    d = pd.read_parquet(path)
    d['date'] = pd.to_datetime(d['date'])
    d = d.sort_values('date').drop_duplicates('date')
    bench[universe] = d.set_index('date')['close']

rows = []
for _, row in segments.iterrows():
    year = row['test_year']
    start = pd.Timestamp(f'{year}-01-01')
    end = pd.Timestamp('2026-09-18') if year == '2026' else pd.Timestamp(f'{year}-12-31')
    s = bench[row['universe']].loc[start:end].dropna()
    if len(s) < 2:
        btotal = np.nan
        bann = np.nan
    else:
        btotal = float(s.iloc[-1] / s.iloc[0] - 1)
        bann = float((1 + btotal) ** (252 / max(len(s) - 1, 1)) - 1)
    x = row.to_dict()
    x['benchmark_name'] = '上证指数' if row['universe'] == 'mainboard' else '沪深300'
    x['benchmark_total_return'] = btotal
    x['benchmark_annual_return'] = bann
    x['excess_total_return'] = float(row['total_return'] - btotal) if np.isfinite(btotal) else np.nan
    x['excess_annual_return'] = float(row['annual_return'] - bann) if np.isfinite(bann) else np.nan
    x['excess_positive'] = bool(x['excess_total_return'] > 0) if np.isfinite(x['excess_total_return']) else False
    rows.append(x)

out = pd.DataFrame(rows)
out.to_csv(OUT / 'walkforward_segments_with_benchmark.csv', index=False)
agg = out.groupby(['strategy', 'universe', 'fee_profile']).agg(
    test_years=('test_year', 'count'),
    positive_years=('total_return', lambda x: int((x > 0).sum())),
    positive_excess_years=('excess_total_return', lambda x: int((x > 0).sum())),
    median_annual_return=('annual_return', 'median'),
    median_excess_annual_return=('excess_annual_return', 'median'),
    mean_excess_annual_return=('excess_annual_return', 'mean'),
    worst_drawdown=('max_drawdown', 'min'),
    median_sharpe=('sharpe', 'median'),
    total_fees=('total_fees', 'sum'),
).reset_index()
agg['year_win_rate'] = agg.positive_years / agg.test_years
agg['excess_win_rate'] = agg.positive_excess_years / agg.test_years
agg.to_csv(OUT / 'walkforward_benchmark_ranking.csv', index=False)
print(agg.sort_values(['universe', 'excess_win_rate', 'median_excess_annual_return'], ascending=[True, False, False]).to_string(index=False))
