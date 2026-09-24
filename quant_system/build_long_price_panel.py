"""Build a strict long-history price panel from genuine RAW + HFQ shards.

Protocol
--------
The formal artifact must contain at least 800 symbols whose history starts no
later than ``--start`` (2018-01-01 plus a small first-trading-day tolerance)
and reaches the tail of the available market calendar.  Missing history is
never padded with synthetic rows: a symbol that does not pass every gate is
simply excluded, and the build fails if fewer than 800 survive.

Data conventions
----------------
``data_warehouse/kline_raw`` holds genuine unadjusted bars.
``data_warehouse/kline_hfq`` holds backward-adjusted bars built as
``raw * cumulative hfq factor`` (factor base 1.0 at listing).
``data_warehouse/kline`` is forward-adjusted (qfq) and is deliberately NOT
accepted as a raw source.  The panel keeps both pairs: ``close`` and the
``hfq_*`` columns are adjusted (returns, factors), while ``raw_*`` columns are
the real trade prices used for order replay, limits and tradability.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INDEX_CODES = {'000300', '000905', '000852', '000001', '399001', '399006'}
PRICE_COLUMNS = ('open', 'high', 'low', 'close')


def _dates(path: Path) -> pd.Series:
    frame = pd.read_parquet(path, columns=['date'])
    return pd.to_datetime(frame['date'], errors='coerce').dropna()


def _bar_gates(frame: pd.DataFrame, prefix: str) -> bool:
    columns = [f'{prefix}_{name}' for name in PRICE_COLUMNS]
    prices = frame[columns]
    if (prices <= 0).any().any() or prices.isna().any().any():
        return False
    high = frame[f'{prefix}_high'] + 1e-6
    low = frame[f'{prefix}_low'] - 1e-6
    if (high < frame[[f'{prefix}_open', f'{prefix}_close']].max(axis=1)).any():
        return False
    if (low > frame[[f'{prefix}_open', f'{prefix}_close']].min(axis=1)).any():
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', required=True)
    ap.add_argument('--count', type=int, default=800)
    ap.add_argument('--start', default='2018-01-01')
    ap.add_argument('--end', default='2026-09-20')
    ap.add_argument('--min-rows', type=int, default=1900)
    ap.add_argument('--start-tolerance-days', type=int, default=5)
    ap.add_argument('--max-end-gap-days', type=int, default=30)
    args = ap.parse_args()
    if args.count < 800:
        raise ValueError('count must be at least 800')
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    hfq_dir = ROOT / 'data_warehouse' / 'kline_hfq'
    raw_dir = ROOT / 'data_warehouse' / 'kline_raw'
    if not hfq_dir.is_dir() or not any(hfq_dir.glob('*.parquet')):
        raise RuntimeError('missing_hfq_source:data_warehouse/kline_hfq')
    if not raw_dir.is_dir() or not any(raw_dir.glob('*.parquet')):
        raise RuntimeError(
            'missing_raw_source:data_warehouse/kline_raw; '
            'data_warehouse/kline is qfq and refusing to label it as raw'
        )

    manifest_path = ROOT / 'data_warehouse' / 'raw_hfq_universe_manifest.json'
    manifest_codes = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        declared = [str(code).zfill(6) for code in manifest.get('codes', [])]
        if int(manifest.get('qualified', 0)) < args.count:
            raise ValueError(
                'manifest_qualified_below_requirement:'
                f'{manifest.get("qualified", 0)}<{args.count}'
            )
        if len(declared) != int(manifest.get('qualified', 0)):
            raise ValueError(
                'manifest_code_count_mismatch:'
                f'{len(declared)}!={manifest.get("qualified", 0)}'
            )
        if len(set(declared)) != len(declared):
            raise ValueError('manifest_duplicate_codes')
        missing = [
            code
            for code in declared
            if not (hfq_dir / f'{code}.parquet').is_file()
            or not (raw_dir / f'{code}.parquet').is_file()
        ]
        if missing:
            raise ValueError(f'manifest_shards_missing:{missing[:10]}')
        manifest_codes = declared

    ranges: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for path in sorted(hfq_dir.glob('*.parquet')):
        code = path.stem.zfill(6)
        if manifest_codes is not None and code not in manifest_codes:
            continue
        if manifest_codes is None and code in INDEX_CODES:
            continue
        if not (raw_dir / f'{code}.parquet').is_file():
            continue
        try:
            series = _dates(path)
        except Exception:
            continue
        if series.empty:
            continue
        ranges[code] = (series.min(), series.max())
    if not ranges:
        raise ValueError('no_raw_hfq_pairs')
    reference_end = max(value[1] for value in ranges.values())
    end_floor = reference_end - pd.Timedelta(days=args.max_end_gap_days)
    start_ceiling = pd.Timestamp(args.start) + pd.Timedelta(days=args.start_tolerance_days)
    eligible = [
        code
        for code, (first, last) in sorted(ranges.items())
        if first <= start_ceiling and last >= end_floor
    ]
    if len(eligible) < args.count:
        raise ValueError(f'insufficient_eight_year_symbols:{len(eligible)}<{args.count}')

    frames: list[pd.DataFrame] = []
    selected: list[str] = []
    rejected: dict[str, str] = {}
    for code in eligible:
        try:
            hfq = pd.read_parquet(hfq_dir / f'{code}.parquet')
            raw = pd.read_parquet(raw_dir / f'{code}.parquet')
        except Exception as error:
            rejected[code] = f'read_error:{type(error).__name__}'
            continue
        hfq['date'] = pd.to_datetime(hfq['date'], errors='coerce')
        raw['date'] = pd.to_datetime(raw['date'], errors='coerce')
        if set(hfq['date']) != set(raw['date']):
            rejected[code] = 'raw_hfq_date_mismatch'
            continue
        h = hfq.rename(columns={name: f'hfq_{name}' for name in PRICE_COLUMNS})[
            ['date', 'hfq_open', 'hfq_high', 'hfq_low', 'hfq_close', 'volume', 'amount']
        ]
        r = raw.rename(columns={name: f'raw_{name}' for name in PRICE_COLUMNS})[
            ['date', 'raw_open', 'raw_high', 'raw_low', 'raw_close']
        ]
        if 'adjust_factor' in hfq.columns:
            h = h.merge(hfq[['date', 'adjust_factor']], on='date', how='left')
        else:
            h['adjust_factor'] = h['hfq_close'] / r['raw_close']
        if 'turnover' in raw.columns:
            r = r.merge(raw[['date', 'turnover']].rename(columns={'turnover': 'turnover_rate'}), on='date', how='left')
        frame = h.merge(r, on='date', how='inner')
        frame = frame[(frame['date'] >= pd.Timestamp(args.start)) & (frame['date'] <= pd.Timestamp(args.end))].copy()
        if len(frame) < args.min_rows:
            rejected[code] = f'rows_below_min:{len(frame)}'
            continue
        if not _bar_gates(frame, 'hfq') or not _bar_gates(frame, 'raw'):
            rejected[code] = 'ohlc_gate_failed'
            continue
        if frame['adjust_factor'].isna().any() or (frame['adjust_factor'] <= 0).any():
            rejected[code] = 'adjust_factor_invalid'
            continue
        frame['code'] = code
        frame['open'] = frame['hfq_open']
        frame['high'] = frame['hfq_high']
        frame['low'] = frame['hfq_low']
        frame['close'] = frame['hfq_close']
        frame['price_adjustment_source'] = 'sina_hfq_factor'
        frame['is_tradable'] = frame['raw_open'].gt(0) & frame['raw_close'].gt(0) & frame['volume'].gt(0)
        frame['is_suspended'] = False
        frames.append(frame)
        selected.append(code)
        if len(frames) >= args.count:
            break
    if len(frames) < args.count:
        raise ValueError(f'panel_symbols_below_requirement:{len(frames)}<{args.count}')

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(['date', 'code']).reset_index(drop=True)
    columns = [
        'date', 'code', 'open', 'high', 'low', 'close',
        'hfq_open', 'hfq_high', 'hfq_low', 'hfq_close',
        'raw_open', 'raw_high', 'raw_low', 'raw_close',
        'volume', 'amount', 'adjust_factor', 'turnover_rate',
        'price_adjustment_source', 'is_tradable', 'is_suspended',
    ]
    panel = panel[[column for column in columns if column in panel.columns]]
    panel.to_parquet(out / 'price_panel.parquet', index=False)

    coverage = panel.groupby('date')['code'].nunique()
    benchmark_path = ROOT / 'data_warehouse' / 'market' / 'index_daily_沪深300.parquet'
    if not benchmark_path.is_file():
        raise RuntimeError(f'missing_benchmark:{benchmark_path}')
    benchmark = pd.read_parquet(benchmark_path)
    benchmark['date'] = pd.to_datetime(benchmark['date'], errors='coerce')
    benchmark = benchmark[
        (benchmark['date'] >= panel['date'].min()) & (benchmark['date'] <= panel['date'].max())
    ].sort_values('date')
    benchmark['return'] = pd.to_numeric(benchmark['close'], errors='coerce').pct_change()
    benchmark[['date', 'return']].dropna().to_csv(out / 'benchmark_csi300.csv', index=False)

    if panel['code'].nunique() < args.count:
        raise ValueError(f'panel_symbols_below_requirement:{panel["code"].nunique()}<{args.count}')
    if panel['date'].min() > start_ceiling:
        raise ValueError(f'panel_starts_after_requirement:{panel["date"].min().date()}>{start_ceiling.date()}')
    membership = panel.groupby('code', as_index=False)['date'].agg(start_date='min', end_date='max')
    membership['universe_id'] = 'eight_year_raw_hfq_800'
    membership.to_csv(out / 'membership.csv', index=False)
    report = {
        'schema': 'long_price_only/v2',
        'data_quality': 'historical_price_only_no_pit_fundamentals',
        'raw_source': 'data_warehouse/kline_raw (genuine unadjusted)',
        'hfq_source': 'data_warehouse/kline_hfq (raw * cumulative hfq factor)',
        'universe_manifest': (
            'data_warehouse/raw_hfq_universe_manifest.json'
            if manifest_codes is not None
            else None
        ),
        'qfq_note': 'data_warehouse/kline is qfq and was not used as raw',
        'symbols': int(panel['code'].nunique()),
        'rows': int(len(panel)),
        'dates': int(panel['date'].nunique()),
        'date_min': str(panel['date'].min().date()),
        'date_max': str(panel['date'].max().date()),
        'min_cross_section': int(coverage.min()),
        'median_cross_section': float(coverage.median()),
        'benchmark_rows': int(len(benchmark)),
        'reference_end': str(reference_end.date()),
        'start_tolerance_days': args.start_tolerance_days,
        'max_end_gap_days': args.max_end_gap_days,
        'eligible_symbols': len(eligible),
        'rejected_by_quality_gate': rejected,
    }
    (out / 'build_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: report[key] for key in (
        'symbols', 'rows', 'dates', 'date_min', 'date_max',
        'min_cross_section', 'median_cross_section', 'benchmark_rows',
    )}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
