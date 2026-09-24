"""Broad-sample low-frequency factor backtest using local RAW OHLCV.

This is a research companion to the frozen HFQ/RAW bundle. It uses a deterministic
sample of local kline files, computes all factors from information available at
close, and executes at the next available open with explicit costs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BroadCase:
    name: str
    factor: str
    direction: int
    quantile: float = 0.2
    rebalance: str = "monthly"
    market_filter: str = "none"
    drawdown_cut: float = 0.0
    drawdown_scale: float = 1.0
    volatility_target: float = 0.0
    max_single_weight: float = 0.0


DEFAULT_CASES = (
    BroadCase("lowvol_monthly_q20", "inv_vol20", 1),
    BroadCase("lowvol_monthly_q30", "inv_vol20", 1, .3),
    BroadCase("liquidity_monthly_q20", "liquidity20", 1),
    BroadCase("momentum63_monthly_q20", "mom63", 1),
    BroadCase("reversal5_monthly_q20", "reversal5", 1),
    BroadCase("quality_trend_monthly_q20", "quality_trend", 1),
    BroadCase("trend_stability_monthly_q20", "trend_stability", 1),
    BroadCase("range_breakout_monthly_q20", "range_breakout", 1),
    BroadCase("downside_lowvol_monthly_q20", "downside_vol20", -1),
    BroadCase("liquidity_stability_monthly_q20", "liquidity_stability", 1),
    BroadCase("lowvol_ddcut10_monthly_q20", "inv_vol20", 1, .2, "monthly", "none", .10),
    BroadCase("lowvol_weekly_q20", "inv_vol20", 1, .2, "weekly"),
    BroadCase("lowvol_voltarget10_monthly", "inv_vol20", 1, .2, "monthly", "none", 0.0, 1.0, .10, 0.0),
    BroadCase("lowvol_voltarget12_monthly", "inv_vol20", 1, .2, "monthly", "none", 0.0, 1.0, .12, 0.0),
    BroadCase("lowvol_ddtier_monthly", "inv_vol20", 1, .2, "monthly", "none", 0.08, 0.70, 0.0, 0.0),
    BroadCase("lowvol_singlecap2_monthly", "inv_vol20", 1, .2, "monthly", "none", 0.0, 1.0, 0.0, .02),
)


def _pick_files(kline_dir: Path, max_symbols: int) -> list[Path]:
    files = sorted(kline_dir.glob("*.parquet"))
    buckets = {"main": [], "growth": [], "star": [], "other": []}
    for path in files:
        code = path.stem.zfill(6)
        bucket = "star" if code.startswith("68") else "growth" if code.startswith(("30", "39")) else "main" if code.startswith(("00", "60")) else "other"
        buckets[bucket].append(path)
    quotas = {"main": int(max_symbols * .60), "growth": int(max_symbols * .20), "star": int(max_symbols * .15), "other": max_symbols}
    selected: list[Path] = []
    for bucket in ("main", "growth", "star", "other"):
        selected.extend(buckets[bucket][:quotas[bucket]])
    if len(selected) < max_symbols:
        used = set(selected); selected.extend(p for p in files if p not in used)
    return selected[:max_symbols]


def load_broad_panel(kline_dir: str | Path, *, max_symbols: int = 1000, min_days: int = 500) -> pd.DataFrame:
    frames = []
    for path in _pick_files(Path(kline_dir), max_symbols):
        frame = pd.read_parquet(path)
        required = {"date", "open", "high", "low", "close", "volume", "amount"}
        if not required.issubset(frame.columns):
            continue
        frame = frame[list(required)].copy()
        frame["code"] = path.stem.zfill(6)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        for col in ("open", "high", "low", "close", "volume", "amount"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=["date", "open", "high", "low", "close"])
        if len(frame) >= min_days:
            frames.append(frame)
    if not frames:
        raise ValueError("no broad local kline files satisfy schema/min_days")
    data = pd.concat(frames, ignore_index=True).sort_values(["code", "date"])
    data = data.drop_duplicates(["code", "date"], keep="last")
    close = data["close"]
    ret = close.groupby(data["code"]).pct_change()
    data["mom21"] = close.groupby(data["code"]).shift(21).rdiv(close) - 1
    data["mom63"] = close.groupby(data["code"]).shift(63).rdiv(close) - 1
    data["mom126"] = close.groupby(data["code"]).shift(126).rdiv(close) - 1
    data["reversal5"] = -ret.groupby(data["code"]).rolling(5).sum().reset_index(level=0, drop=True)
    data["inv_vol20"] = -ret.groupby(data["code"]).rolling(20).std().reset_index(level=0, drop=True)
    data["liquidity20"] = data["amount"].groupby(data["code"]).transform(lambda x: x.rolling(20, min_periods=10).mean())
    data["quality_trend"] = data["mom63"] / (ret.groupby(data["code"]).rolling(63).std().reset_index(level=0, drop=True) + 1e-8)
    # Additional cheap, point-in-time candidates. These are deliberately based
    # only on OHLCV; no current fundamentals or today's industry labels.
    data["trend_stability"] = data["mom126"] / (ret.groupby(data["code"]).rolling(126).std().reset_index(level=0, drop=True) + 1e-8)
    data["range_breakout"] = close / close.groupby(data["code"]).transform(lambda x: x.rolling(252, min_periods=60).max()) - 1.0
    data["downside_vol20"] = ret.where(ret < 0).groupby(data["code"]).rolling(20, min_periods=10).std().reset_index(level=0, drop=True)
    data["liquidity_stability"] = data["amount"].groupby(data["code"]).transform(lambda x: x.rolling(20, min_periods=10).mean()) / (data["amount"].groupby(data["code"]).transform(lambda x: x.rolling(60, min_periods=20).mean()) + 1e-8)
    return data.sort_values(["date", "code"])


def _schedule(dates: pd.DatetimeIndex, rebalance: str) -> set[pd.Timestamp]:
    if rebalance == "weekly":
        return set(pd.Series(dates, index=dates).groupby(dates.to_period("W")).max())
    if rebalance == "monthly":
        return set(pd.Series(dates, index=dates).groupby(dates.to_period("M")).max())
    return set(dates)


def _metrics(returns: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    curve = (1 + values).cumprod()
    years = len(values) / 252
    std = values.std(ddof=1)
    return {"observations": int(len(values)), "total_return": float(curve.iloc[-1] - 1),
            "annual_return": float(curve.iloc[-1] ** (1 / years) - 1) if years and curve.iloc[-1] > 0 else None,
            "sharpe": float(values.mean() / std * math.sqrt(252)) if len(values) > 1 and std > 0 else None,
            "max_drawdown": float((curve / curve.cummax() - 1).min()),
            "calmar": float((curve.iloc[-1] ** (1 / years) - 1) / abs((curve / curve.cummax() - 1).min())) if years and (curve / curve.cummax() - 1).min() < 0 else None}


def run_case(panel: pd.DataFrame, case: BroadCase, *, initial_capital: float = 1_000_000, cost_bps: float = 18.0) -> dict[str, Any]:
    dates = pd.DatetimeIndex(sorted(panel.date.unique())); schedule = _schedule(dates, case.rebalance)
    by_date = {date: frame.set_index("code") for date, frame in panel.groupby("date", sort=True)}
    holdings: dict[str, int] = {}; cash = float(initial_capital); peak = cash; pending: list[str] | None = None; ledger = []; trades = 0; previous_total = cash
    for index, date in enumerate(dates):
        frame = by_date[date]; open_prices = frame["open"]
        if pending is not None:
            available = [s for s in pending if s in open_prices.index and np.isfinite(open_prices[s]) and open_prices[s] > 0]
            equity = cash + sum(holdings.get(s, 0) * float(open_prices.get(s, 0)) for s in holdings)
            target_scale = 1.0
            if case.volatility_target > 0:
                recent = panel[(panel.date < date) & panel.code.isin(available)].groupby("date")["close"].mean().pct_change().tail(60)
                realized = float(recent.std() * math.sqrt(252)) if len(recent) > 10 else case.volatility_target
                target_scale = min(1.0, case.volatility_target / max(realized, 1e-8))
            if case.drawdown_scale < 1.0 and drawdown <= -abs(case.drawdown_cut):
                target_scale = min(target_scale, case.drawdown_scale)
            target_scale = max(0.0, min(1.0, target_scale))
            target_value = equity * .95 * target_scale / len(available) if available else 0.0
            raw_desired = {s: target_value / (float(open_prices[s]) * (1 + cost_bps / 10000)) for s in available}
            desired = {s: int(min(v, equity * case.max_single_weight / (float(open_prices[s]) * (1 + cost_bps / 10000))) // 100 * 100) if case.max_single_weight > 0 else int(v // 100 * 100) for s, v in raw_desired.items()}
            # Sell removed names and excess shares before buying. This makes every
            # rebalance target-weighted rather than accumulating old positions.
            for symbol in list(holdings):
                price = open_prices.get(symbol, np.nan)
                target_shares = desired.get(symbol, 0)
                if not np.isfinite(price) or price <= 0 or holdings[symbol] <= target_shares:
                    continue
                sold = holdings[symbol] - target_shares; fill = float(price) * (1 - cost_bps / 10000)
                cash += sold * fill * (1 - cost_bps / 10000); holdings[symbol] = target_shares; trades += 1
                if not holdings[symbol]: holdings.pop(symbol)
            for symbol, target_shares in desired.items():
                current = holdings.get(symbol, 0); buy = max(0, target_shares - current)
                fill = float(open_prices[symbol]) * (1 + cost_bps / 10000); fee = buy * fill * cost_bps / 10000
                if buy and buy * fill + fee <= cash:
                    cash -= buy * fill + fee; holdings[symbol] = current + buy; trades += 1
            pending = None
        close_prices = frame["close"].dropna(); position_value = sum(shares * float(close_prices.get(symbol, 0)) for symbol, shares in holdings.items()); total = cash + position_value; peak = max(peak, total); drawdown = total / peak - 1
        ledger.append({"date": date, "total_value": total, "cash": cash, "position_value": position_value, "drawdown": drawdown})
        if date in schedule and index + 1 < len(dates):
            if case.drawdown_cut and case.drawdown_scale >= 1.0 and drawdown <= -case.drawdown_cut:
                pending = []
            else:
                eligible = frame.dropna(subset=[case.factor]).copy(); ranked = (eligible[case.factor] * case.direction).sort_values(ascending=False)
                n = max(1, int(len(ranked) * case.quantile)); pending = ranked.head(n).index.astype(str).tolist()
    ledger_frame = pd.DataFrame(ledger); ledger_frame["return"] = ledger_frame.total_value.pct_change().fillna(0)
    return {"case": asdict(case), "metrics": _metrics(ledger_frame["return"]), "trades": trades, "symbols": int(panel.code.nunique()), "period": [str(dates[0].date()), str(dates[-1].date())], "negative_cash": int((ledger_frame.cash < -0.01).sum()), "audit": "PASS" if (ledger_frame.cash >= -0.01).all() else "BLOCK"}


def run_matrix(kline_dir: str | Path, output_dir: str | Path, *, max_symbols: int = 1000, cases=DEFAULT_CASES) -> dict:
    panel = load_broad_panel(kline_dir, max_symbols=max_symbols); output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    rows = [run_case(panel, case) for case in cases]
    rows.sort(key=lambda item: item["metrics"]["annual_return"] or -999, reverse=True)
    digest = hashlib.sha256(pd.util.hash_pandas_object(panel[["code", "date"]], index=False).values.tobytes()).hexdigest()
    bench = pd.read_parquet(Path(kline_dir).parent / "market" / "index_daily_沪深300.parquet"); bench["date"] = pd.to_datetime(bench["date"]); bench = bench[(bench.date >= panel.date.min()) & (bench.date <= panel.date.max())]
    benchmark = float(bench.close.iloc[-1] / bench.close.iloc[0] - 1) if len(bench) > 1 else None
    benchmark_years = len(bench) / 252 if len(bench) else 0
    benchmark_annual = ((1 + benchmark) ** (1 / benchmark_years) - 1) if benchmark is not None and benchmark_years > 0 and 1 + benchmark > 0 else None
    for item in rows:
        item["benchmark_total"] = benchmark
        item["benchmark_annual"] = benchmark_annual
        item["excess_annual"] = item["metrics"]["annual_return"] - benchmark_annual if item["metrics"]["annual_return"] is not None and benchmark_annual is not None else None
    result = {"schema": "broad-raw-matrix/v1", "sample_files": max_symbols, "actual_symbols": int(panel.code.nunique()), "rows": len(panel), "data_sha256": digest, "signal_price": "raw", "execution_price": "raw_next_open", "cost_bps": 18.0, "benchmark": {"name": "CSI300", "total_return": benchmark, "annual_return": benchmark_annual}, "cases": rows}
    (output / "broad_strategy_matrix.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = ["# Broad RAW Sample Low-Frequency Matrix", "", f"- Symbols: {result['actual_symbols']}; rows: {result['rows']}; period: {rows[0]['period']}", f"- Signal: RAW; execution: RAW next open; cost: 18 bps round-trip approximation; CSI300 annual: {benchmark_annual:.4f}", "", "| Strategy | Annual | Excess | Sharpe | Max DD | Calmar | Trades | Audit |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for item in rows:
        m = item["metrics"]; fmt = lambda x: "NA" if x is None else f"{x:.4f}"
        lines.append(f"| {item['case']['name']} | {fmt(m['annual_return'])} | {fmt(item['excess_annual'])} | {fmt(m['sharpe'])} | {fmt(m['max_drawdown'])} | {fmt(m['calmar'])} | {item['trades']} | {item['audit']} |")
    (output / "broad_strategy_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    panel.to_parquet(output / "broad_panel.parquet", index=False)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--kline-dir", required=True); parser.add_argument("--output-dir", required=True); parser.add_argument("--max-symbols", type=int, default=1000)
    args = parser.parse_args(argv); result = run_matrix(args.kline_dir, args.output_dir, max_symbols=args.max_symbols); print(json.dumps({"output": args.output_dir, "symbols": result["actual_symbols"], "cases": len(result["cases"]), "best": result["cases"][0]}, ensure_ascii=False, default=str)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
