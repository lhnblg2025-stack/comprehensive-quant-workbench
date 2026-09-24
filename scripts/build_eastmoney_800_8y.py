#!/usr/bin/env python3
"""Build an auditable >=800 symbol x 8-year raw + HFQ daily archive.

Why the old ``data_warehouse/kline`` directory cannot be used as raw
---------------------------------------------------------------------
Those files are Sina *forward-adjusted* (qfq) bars: for 600519 and 000001
their 2020-2026 values match ``adjust="qfq"`` exactly, while genuine
unadjusted closes are 13%-22% higher.  Treating them as trade prices would
silently corrupt any formal order replay, so this builder never writes there.

Outputs
-------
``data_warehouse/kline_raw/{code}.parquet``
    Genuine unadjusted OHLCV (Sina ``adjust=""``).
``data_warehouse/kline_hfq/{code}.parquet``
    Backward-adjusted bars built as ``raw * cumulative hfq factor``.
``data_warehouse/adjust_factors/{code}.parquet``
    The cumulative factor chain, kept for audit.
``data_warehouse/raw_hfq_universe_manifest.json``
    Per-symbol coverage, validation statistics and every failure reason.

Nothing is padded or synthesised.  A symbol only counts as qualified after
raw, HFQ and factor series all pass the validation gates in ``validate_pair``.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REQUIRED_RAW_COLUMNS = ('date', 'open', 'high', 'low', 'close', 'volume', 'amount')
PRICE_COLUMNS = ('open', 'high', 'low', 'close')
RANK_PREFIXES = (
    ('600', 0),
    ('000', 1),
    ('002', 2),
    ('601', 3),
    ('603', 4),
    ('300', 5),
    ('001', 6),
    ('003', 7),
    ('605', 7),
    ('688', 8),
    ('301', 8),
    ('302', 8),
)
LOCK = threading.Lock()


def log(event: dict) -> None:
    event = {'ts': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), **event}
    with LOCK:
        print(json.dumps(event, ensure_ascii=False), flush=True)


def candidate_rank(code: str) -> tuple[int, str]:
    for prefix, rank in RANK_PREFIXES:
        if code.startswith(prefix):
            return rank, code
    return 9, code


def market_symbol(code: str) -> str:
    if code.startswith(('6', '9')):
        return f'sh{code}'
    if code.startswith(('0', '2', '3')):
        return f'sz{code}'
    raise ValueError(f'unsupported_a_share_code:{code}')


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out['date'] = pd.to_datetime(out['date'], errors='coerce')
    for column in out.columns:
        if column != 'date':
            out[column] = pd.to_numeric(out[column], errors='coerce')
    out = out[out['date'].notna()]
    out = out.drop_duplicates(subset=['date'], keep='last')
    return out.sort_values('date').reset_index(drop=True)


def build_hfq(raw: pd.DataFrame, factor: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    factor = normalize_frame(factor[['date', 'hfq_factor']])
    factor = factor[factor['hfq_factor'].notna() & (factor['hfq_factor'] > 0)]
    merged = pd.merge_asof(raw, factor, on='date', direction='backward')
    if merged['hfq_factor'].isna().any():
        raise ValueError('hfq_factor_missing_for_part_of_raw_history')
    hfq = merged.copy()
    for column in PRICE_COLUMNS:
        hfq[column] = hfq[column] * hfq['hfq_factor']
    hfq = hfq[['date', *PRICE_COLUMNS, 'volume', 'amount', 'hfq_factor']].rename(
        columns={'hfq_factor': 'adjust_factor'}
    )
    return hfq, factor


def fetch_sina(code: str, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (raw, hfq, factor) from Sina via akshare."""
    import akshare as ak

    symbol = market_symbol(code)
    raw = ak.stock_zh_a_daily(symbol=symbol, start_date=start, end_date=end, adjust='')
    factor = ak.stock_zh_a_daily(symbol=symbol, start_date=start, end_date=end, adjust='hfq-factor')
    if raw is None or raw.empty:
        raise ValueError('sina_empty_raw')
    raw = normalize_frame(raw)
    raw['pct_chg'] = raw['close'].pct_change() * 100.0
    hfq, factor = build_hfq(raw, factor)
    hfq['pct_chg'] = hfq['close'].pct_change() * 100.0
    missing = [column for column in REQUIRED_RAW_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f'sina_missing_columns:{missing}')
    columns = [*REQUIRED_RAW_COLUMNS, 'outstanding_share', 'turnover', 'pct_chg']
    raw = raw[[column for column in columns if column in raw.columns]]
    return raw, hfq, factor


_EASTMONEY_HOSTS = ('push2his.eastmoney.com', '82.push2his.eastmoney.com')
_EASTMONEY_FIELDS1 = 'f1,f2,f3,f4,f5,f6'
_EASTMONEY_FIELDS2 = 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116'


def fetch_eastmoney(code: str, start: str, end: str, fqt: int, retries: int) -> pd.DataFrame:
    import requests

    market = 1 if code.startswith(('6', '9')) else 0
    params = {
        'fields1': _EASTMONEY_FIELDS1,
        'fields2': _EASTMONEY_FIELDS2,
        'ut': '7eea3edcaed734bea9cbfc24409ed989',
        'klt': '101',
        'fqt': str(fqt),
        'secid': f'{market}.{code}',
        'beg': start.replace('-', ''),
        'end': end.replace('-', ''),
    }
    last_error: Exception | None = None
    for attempt in range(retries):
        host = _EASTMONEY_HOSTS[attempt % len(_EASTMONEY_HOSTS)]
        try:
            response = requests.get(
                f'https://{host}/api/qt/stock/kline/get',
                params={**params, '_': str(time.time_ns())},
                headers={'User-Agent': 'curl/8.0', 'Referer': 'https://quote.eastmoney.com/'},
                timeout=25,
            )
            response.raise_for_status()
            rows = (response.json().get('data') or {}).get('klines') or []
            if not rows:
                raise ValueError('eastmoney_empty')
            records = []
            for row in rows:
                parts = row.split(',')
                if len(parts) >= 7:
                    records.append(
                        {
                            'date': parts[0],
                            'open': float(parts[1]),
                            'close': float(parts[2]),
                            'high': float(parts[3]),
                            'low': float(parts[4]),
                            'volume': float(parts[5]),
                            'amount': float(parts[6]),
                        }
                    )
            return normalize_frame(pd.DataFrame(records))
        except Exception as error:  # noqa: BLE001 - retried and reported upstream
            last_error = error
            time.sleep(min(8.0, 0.6 * (2**attempt)) + random.random())
    raise RuntimeError(f'eastmoney_{code}_fqt={fqt}:{last_error}')


def fetch_eastmoney_pair(code: str, start: str, end: str, retries: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = fetch_eastmoney(code, start, end, 0, retries)
    hfq = fetch_eastmoney(code, start, end, 2, retries)
    raw['pct_chg'] = raw['close'].pct_change() * 100.0
    pair = raw[['date', 'close']].merge(
        hfq[['date', 'close']], on='date', how='inner', suffixes=('_raw', '_hfq')
    )
    factor = pair[['date']].copy()
    factor['hfq_factor'] = pair['close_hfq'] / pair['close_raw']
    hfq = hfq[['date', *PRICE_COLUMNS, 'volume', 'amount']].copy()
    hfq['adjust_factor'] = factor.set_index('date')['hfq_factor'].reindex(hfq['date']).to_numpy()
    hfq['pct_chg'] = hfq['close'].pct_change() * 100.0
    return raw, hfq, factor


def validate_pair(
    code: str,
    raw: pd.DataFrame,
    hfq: pd.DataFrame,
    factor: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[bool, str, dict]:
    def fail(reason: str) -> tuple[bool, str, dict]:
        return False, reason, {}

    if raw is None or hfq is None or factor is None or raw.empty or hfq.empty or factor.empty:
        return fail('empty_series')
    for name, frame, columns in (
        ('raw', raw, REQUIRED_RAW_COLUMNS),
        ('hfq', hfq, ('date', *PRICE_COLUMNS, 'volume', 'amount', 'adjust_factor')),
        ('factor', factor, ('date', 'hfq_factor')),
    ):
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            return fail(f'{name}_missing_columns:{missing}')
        if not pd.api.types.is_datetime64_any_dtype(frame['date']):
            return fail(f'{name}_date_not_datetime')
        if frame['date'].duplicated().any():
            return fail(f'{name}_duplicate_dates')
        if not frame['date'].is_monotonic_increasing:
            return fail(f'{name}_dates_not_sorted')
    raw_dates = set(raw['date'])
    hfq_dates = set(hfq['date'])
    if raw_dates != hfq_dates:
        return fail(f'raw_hfq_date_mismatch:raw_only={len(raw_dates - hfq_dates)},hfq_only={len(hfq_dates - raw_dates)}')
    if len(raw) < args.min_rows:
        return fail(f'rows_below_min:{len(raw)}<{args.min_rows}')
    required_values = raw[list(REQUIRED_RAW_COLUMNS)].drop(columns=['date'])
    if required_values.isna().any().any():
        return fail('raw_has_nan_ohlcv')
    if (raw[list(PRICE_COLUMNS)] <= 0).any().any():
        return fail('raw_non_positive_price')
    if (hfq[list(PRICE_COLUMNS)] <= 0).any().any():
        return fail('hfq_non_positive_price')
    if (raw['volume'] < 0).any() or (raw['amount'] < 0).any():
        return fail('raw_negative_volume_or_amount')
    high_ok = (raw['high'] + 1e-6 >= raw[['open', 'close']].max(axis=1)).all()
    low_ok = (raw['low'] - 1e-6 <= raw[['open', 'close']].min(axis=1)).all()
    if not high_ok or not low_ok:
        return fail('raw_ohlc_inconsistent')
    high_ok = (hfq['high'] + 1e-6 >= hfq[['open', 'close']].max(axis=1)).all()
    low_ok = (hfq['low'] - 1e-6 <= hfq[['open', 'close']].min(axis=1)).all()
    if not high_ok or not low_ok:
        return fail('hfq_ohlc_inconsistent')
    if (factor['hfq_factor'] <= 0).any():
        return fail('non_positive_factor')
    factor_sorted = factor.sort_values('date')
    if factor_sorted['hfq_factor'].diff().min() < -1e-9:
        return fail('factor_not_monotonic_after_genuine_raw')
    first_expected = pd.Timestamp(args.start) + pd.Timedelta(days=args.start_tolerance_days)
    if raw['date'].min() > first_expected:
        return fail(f'history_starts_late:{raw["date"].min().date()}>{first_expected.date()}')
    end_floor = pd.Timestamp(args.end) - pd.Timedelta(days=args.max_end_gap_days)
    if raw['date'].max() < end_floor:
        return fail(f'history_ends_early:{raw["date"].max().date()}<{end_floor.date()}')
    ratio = hfq.set_index('date')[list(PRICE_COLUMNS)] / raw.set_index('date')[list(PRICE_COLUMNS)]
    spread = ratio.max(axis=1) / ratio.min(axis=1) - 1.0
    if float(spread.max()) > args.factor_spread_tolerance:
        return fail(f'intraday_factor_spread:{float(spread.max()):.6f}')
    stats = {
        'rows': int(len(raw)),
        'start': str(raw['date'].min().date()),
        'end': str(raw['date'].max().date()),
        'factor_first': float(factor_sorted['hfq_factor'].iloc[0]),
        'factor_last': float(factor_sorted['hfq_factor'].iloc[-1]),
        'factor_jumps': int((factor_sorted['hfq_factor'].diff().fillna(0) != 0).sum()),
        'raw_last_close': float(raw['close'].iloc[-1]),
        'hfq_last_close': float(hfq['close'].iloc[-1]),
        'intraday_factor_spread_max': float(spread.max()),
    }
    return True, 'ok', stats


def atomic_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.{threading.get_ident()}.tmp')
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def output_paths(root: Path, code: str) -> tuple[Path, Path, Path]:
    return (
        root / 'data_warehouse' / 'kline_raw' / f'{code}.parquet',
        root / 'data_warehouse' / 'kline_hfq' / f'{code}.parquet',
        root / 'data_warehouse' / 'adjust_factors' / f'{code}.parquet',
    )


def load_existing(root: Path, code: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    paths = output_paths(root, code)
    if not all(path.exists() for path in paths):
        return None
    try:
        return tuple(pd.read_parquet(path) for path in paths)  # type: ignore[return-value]
    except Exception:  # noqa: BLE001 - a corrupt shard must be refetched
        return None


def process_one(code: str, args: argparse.Namespace, root: Path) -> tuple[str, str, dict, str]:
    last_error = 'unknown'
    sources = ('sina', 'eastmoney') if args.source == 'auto' else (args.source,)
    for source in sources:
        for attempt in range(1, args.request_retries + 1):
            try:
                if source == 'sina':
                    raw, hfq, factor = fetch_sina(code, args.start, args.end)
                else:
                    raw, hfq, factor = fetch_eastmoney_pair(code, args.start, args.end, args.request_retries)
                ok, reason, stats = validate_pair(code, raw, hfq, factor, args)
                if not ok:
                    return code, 'rejected', {}, reason
                raw_path, hfq_path, factor_path = output_paths(root, code)
                atomic_write(raw, raw_path)
                atomic_write(hfq, hfq_path)
                atomic_write(factor, factor_path)
                stats['source'] = source
                return code, 'ok', stats, reason
            except Exception as error:  # noqa: BLE001 - retried, then reported
                last_error = f'{source}:{type(error).__name__}:{str(error)[:200]}'
                if attempt < args.request_retries:
                    time.sleep(min(8.0, 0.5 * (2 ** (attempt - 1))) + random.random() * 0.5)
        if args.source != 'auto':
            break
    return code, 'error', {}, last_error


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--count', type=int, default=800)
    parser.add_argument('--allow-partial', action='store_true', help='permit count<800 for smoke tests only')
    parser.add_argument('--start', default='2018-01-01')
    parser.add_argument('--end', default='2026-09-20')
    parser.add_argument('--workers', type=int, default=1, help='bounded concurrent fetches (max 4); Sina forces 1')
    parser.add_argument('--source', choices=('sina', 'eastmoney', 'auto'), default='sina')
    parser.add_argument('--resume', action='store_true', help='reuse shards that still pass every quality gate')
    parser.add_argument('--request-retries', type=int, default=3)
    parser.add_argument('--min-rows', type=int, default=1900)
    parser.add_argument('--max-end-gap-days', type=int, default=30)
    parser.add_argument('--start-tolerance-days', type=int, default=5)
    parser.add_argument('--factor-spread-tolerance', type=float, default=1e-4)
    parser.add_argument('--failure-circuit-breaker', type=int, default=15)
    parser.add_argument('--progress-every', type=int, default=20)
    parser.add_argument('--sleep-min', type=float, default=0.10)
    parser.add_argument('--sleep-max', type=float, default=0.35)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.count < 800 and not args.allow_partial:
        raise ValueError('count must be at least 800 for the formal eight-year protocol')
    root = Path(args.root).resolve()
    inventory_dir = root / 'data_warehouse' / 'kline'
    codes = sorted(
        {
            path.stem.zfill(6)
            for path in inventory_dir.glob('*.parquet')
            if path.stem.isdigit() and len(path.stem) <= 6
        },
        key=candidate_rank,
    )
    if not codes:
        raise RuntimeError(f'inventory_empty:{inventory_dir}')
    results: dict[str, dict] = {}
    failures: dict[str, dict] = {}
    if args.resume:
        for index, code in enumerate(codes, 1):
            existing = load_existing(root, code)
            if existing is None:
                continue
            ok, reason, stats = validate_pair(code, *existing, args)
            if ok:
                results[code] = {**stats, 'source': 'resume'}
            if index % 500 == 0:
                log({'phase': 'resume_scan', 'scanned': index, 'qualified': len(results)})
        log({'phase': 'resume_done', 'qualified': len(results), 'requested': args.count})
    pending = [code for code in codes if code not in results]
    manifest_path = root / 'data_warehouse' / 'raw_hfq_universe_manifest.json'
    started = time.time()
    processed = 0
    consecutive_failures = 0
    cursor = 0
    stop_reason = 'target_reached' if len(results) >= args.count else 'in_progress'
    workers = max(1, min(args.workers, 4))
    if args.source in ('sina', 'auto') and workers > 1:
        # akshare decodes Sina history with py_mini_racer (V8); concurrent
        # MiniRacer instances segfault the interpreter, so Sina stays serial.
        log({'phase': 'safety', 'warning': 'sina_source_forces_workers_1_because_py_mini_racer_segfaults', 'requested_workers': workers})
        workers = 1
    if len(results) < args.count and pending:
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='rawhfq')
        in_flight: dict = {}
        window = workers * 2
        try:
            while len(results) < args.count:
                while len(in_flight) < window and cursor < len(pending):
                    code = pending[cursor]
                    cursor += 1
                    if args.sleep_max > 0:
                        time.sleep(random.uniform(max(0.0, args.sleep_min), max(0.0, args.sleep_max)))
                    in_flight[executor.submit(process_one, code, args, root)] = code
                if not in_flight:
                    stop_reason = 'candidates_exhausted'
                    break
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    code = in_flight.pop(future)
                    processed += 1
                    try:
                        code, status, stats, reason = future.result()
                    except Exception as error:  # noqa: BLE001 - defensive
                        status, stats, reason = 'error', {}, f'{type(error).__name__}:{str(error)[:200]}'
                    if status == 'ok':
                        results[code] = stats
                        consecutive_failures = 0
                    elif status == 'error':
                        failures[code] = {'status': status, 'reason': reason}
                        consecutive_failures += 1
                    else:
                        # A rejected symbol (for example a post-2018 IPO) proves
                        # the source is healthy, so it must not trip the breaker.
                        failures[code] = {'status': status, 'reason': reason}
                        consecutive_failures = 0
                    if processed % max(1, args.progress_every) == 0 or consecutive_failures >= args.failure_circuit_breaker:
                        elapsed = max(1e-6, time.time() - started)
                        rate = processed / elapsed
                        log(
                            {
                                'phase': 'fetch',
                                'processed': processed,
                                'qualified': len(results),
                                'failures': len(failures),
                                'candidates_left': len(pending) - cursor,
                                'rows_per_min': round(rate * 60, 1),
                                'eta_minutes': round((args.count - len(results)) / max(rate, 1e-6) / 60, 1),
                            }
                        )
                    if consecutive_failures >= args.failure_circuit_breaker:
                        stop_reason = f'circuit_breaker_after_{consecutive_failures}_consecutive_failures'
                        break
                if stop_reason.startswith('circuit_breaker'):
                    break
                if len(results) >= args.count:
                    stop_reason = 'target_reached'
                    break
            if stop_reason == 'in_progress' and len(results) < args.count:
                stop_reason = 'candidates_exhausted'
        finally:
            for future in in_flight:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
    qualified_codes = sorted(results, key=candidate_rank)[: args.count]
    manifest = {
        'schema': 'raw_hfq_universe/v3',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'source': args.source,
        'hfq_convention': (
            'genuine backward adjustment: hfq = raw * cumulative hfq factor whose base is 1.0 at listing; '
            'latest hfq therefore differs from raw by a per-symbol constant, daily returns are unaffected, '
            'and raw is kept separately for price-level, limit and tradability decisions'
        ),
        'protocol': {
            'start': args.start,
            'end': args.end,
            'min_rows': args.min_rows,
            'max_end_gap_days': args.max_end_gap_days,
            'validation': [
                'raw_hfq_identical_date_sets',
                'raw_ohlcv_positive_and_consistent',
                'volume_amount_non_negative',
                'cumulative_hfq_factor_positive_and_monotonic',
                'intraday_hfq_over_raw_factor_constant',
                'history_start_not_later_than_start_plus_tolerance',
                'history_end_not_earlier_than_end_minus_gap',
            ],
        },
        'requested': args.count,
        'qualified': len(qualified_codes),
        'processed_this_run': processed,
        'stop_reason': stop_reason,
        'directories': {
            'raw': 'data_warehouse/kline_raw',
            'hfq': 'data_warehouse/kline_hfq',
            'factor': 'data_warehouse/adjust_factors',
            'note': 'data_warehouse/kline is qfq and is deliberately not used as raw',
        },
        'codes': qualified_codes,
        'symbols': {code: results[code] for code in qualified_codes},
        'symbols_discovered': len(codes),
        'failures': failures,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    if qualified_codes:
        pd.DataFrame(
            [
                {
                    'code': code,
                    'source': results[code].get('source', ''),
                    'start': results[code].get('start', ''),
                    'end': results[code].get('end', ''),
                    'rows': results[code].get('rows', 0),
                }
                for code in qualified_codes
            ]
        ).to_csv(root / 'data_warehouse' / 'raw_hfq_universe.csv', index=False)
    log(
        {
            'phase': 'done',
            'requested': args.count,
            'qualified': len(qualified_codes),
            'stop_reason': stop_reason,
            'failures': len(failures),
            'manifest': str(manifest_path.relative_to(root)),
        }
    )
    if len(qualified_codes) < args.count:
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
