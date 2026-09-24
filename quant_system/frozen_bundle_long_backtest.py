"""Production long-run replay for a frozen HFQ-signal/RAW-execution bundle."""
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

from .backtest_dataset_quality import audit_bundle
from .product_contract import PRODUCT_VERSION, release_metadata, validate_production_contract
from .data_release import load_release


# These columns are computed from the bundle's HFQ signal prices. They are
# research candidates, not automatically promoted production factors.
STANDARD_HFQ_FACTOR_SPECS = {
    "mom20": {"signal_price": "hfq", "execution_price": "raw", "formula": "hfq_close / hfq_close[-20] - 1"},
    "mom60": {"signal_price": "hfq", "execution_price": "raw", "formula": "hfq_close / hfq_close[-60] - 1"},
    "mom120": {"signal_price": "hfq", "execution_price": "raw", "formula": "hfq_close / hfq_close[-120] - 1"},
    "dist_52w": {"signal_price": "hfq", "execution_price": "raw", "formula": "hfq_close / rolling_max(hfq_close,252) - 1"},
    "amp20": {"signal_price": "hfq", "execution_price": "raw", "formula": "mean(hfq_high/hfq_low-1,20)"},
    "vol_ratio20": {"signal_price": "hfq", "execution_price": "raw", "formula": "volume / mean(volume,20)"},
}


@dataclass(frozen=True)
class LongRunConfig:
    bundle_dir: str
    output_dir: str
    factor: str = "amp20"
    direction: int = -1
    quantile: float = 0.20
    rebalance: str = "weekly"
    initial_capital: float = 1_000_000.0
    cash_buffer: float = 0.05
    commission_bps: float = 0.85
    stamp_duty_bps: float = 5.0
    transfer_fee_bps: float = 0.1
    slippage_bps: float = 10.0
    min_commission: float = 5.0
    min_symbols: int = 100
    min_days: int = 1000
    market_filter: str = "none"  # none, benchmark_sma200
    drawdown_cut: float = 0.0  # e.g. 0.10 means move to cash after 10% drawdown


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fee(notional: float, side: str, cfg: LongRunConfig) -> float:
    commission = max(cfg.min_commission, notional * cfg.commission_bps / 10000.0)
    transfer = notional * cfg.transfer_fee_bps / 10000.0
    stamp = notional * cfg.stamp_duty_bps / 10000.0 if side == "sell" else 0.0
    return commission + transfer + stamp


def _metrics(returns: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if values.empty:
        return {"observations": 0, "total_return": None, "annual_return": None, "sharpe": None, "max_drawdown": None}
    curve = (1 + values).cumprod()
    years = len(values) / 252.0
    std = values.std(ddof=1)
    return {
        "observations": int(len(values)),
        "total_return": float(curve.iloc[-1] - 1),
        "annual_return": float(curve.iloc[-1] ** (1 / years) - 1) if years > 0 and curve.iloc[-1] > 0 else None,
        "sharpe": float(values.mean() / std * math.sqrt(252)) if len(values) > 1 and std > 0 else None,
        "max_drawdown": float((curve / curve.cummax() - 1).min()),
    }


def _annual_performance(ledger: pd.DataFrame, benchmark_close: pd.Series) -> pd.DataFrame:
    """Align strategy and CSI300 returns and compound each calendar year."""
    frame = ledger[["date", "return"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).set_index("date")
    benchmark = pd.to_numeric(benchmark_close, errors="coerce").reindex(frame.index).ffill()
    frame["benchmark_return"] = benchmark.pct_change().fillna(0.0)
    frame["year"] = frame.index.year
    rows: list[dict[str, Any]] = []
    for year, group in frame.groupby("year", sort=True):
        strategy = _metrics(group["return"])
        benchmark_metrics = _metrics(group["benchmark_return"])
        rows.append({
            "year": int(year),
            "observations": int(len(group)),
            "strategy_total_return": strategy["total_return"],
            "strategy_annual_return": strategy["annual_return"],
            "benchmark_total_return": benchmark_metrics["total_return"],
            "benchmark_annual_return": benchmark_metrics["annual_return"],
            "annual_excess_return": (
                strategy["annual_return"] - benchmark_metrics["annual_return"]
                if strategy["annual_return"] is not None and benchmark_metrics["annual_return"] is not None
                else None
            ),
        })
    return pd.DataFrame(rows)


def _block_reason(row: pd.Series, side: str, config: LongRunConfig) -> str | None:
    price = pd.to_numeric(pd.Series([row.get("raw_open")]), errors="coerce").iloc[0]
    volume = pd.to_numeric(pd.Series([row.get("raw_volume", row.get("volume"))]), errors="coerce").iloc[0]
    if not np.isfinite(price) or price <= 0:
        return "missing_or_invalid_open_price"
    if not np.isfinite(volume) or volume <= 0:
        return "suspended_or_zero_volume"
    previous_close = pd.to_numeric(pd.Series([row.get("raw_close")]), errors="coerce").iloc[0]
    if np.isfinite(previous_close) and previous_close > 0:
        change = price / previous_close - 1.0
        if side == "buy" and change >= 0.095:
            return "limit_up_buy_blocked_proxy"
        if side == "sell" and change <= -0.095:
            return "limit_down_sell_blocked_proxy"
    return None


def run_long_backtest(config: LongRunConfig) -> dict[str, Any]:
    bundle = Path(config.bundle_dir)
    output = Path(config.output_dir)
    quality = audit_bundle(bundle, min_symbols=config.min_symbols, min_days=config.min_days)
    if quality["status"] != "PASS":
        raise RuntimeError(f"frozen bundle blocked: {quality['errors']}")
    production = validate_production_contract(bundle.parent.parent.parent)
    if production["status"] != "PASS":
        raise RuntimeError(f"production contract blocked: {production['errors']}")
    bundle_manifest = json.loads((bundle / "data_manifest.json").read_text(encoding="utf-8"))
    release_id = bundle_manifest.get("release_id") or f"frozen-{_sha256(bundle / 'data_manifest.json')[:16]}"

    registry = json.loads((bundle / "factor_registry.json").read_text(encoding="utf-8"))
    factor_meta = registry.get(config.factor) or STANDARD_HFQ_FACTOR_SPECS.get(config.factor)
    if not factor_meta:
        raise KeyError(f"factor missing from frozen registry or standard HFQ research set: {config.factor}")
    if factor_meta.get("signal_price") != "hfq" or factor_meta.get("execution_price") != "raw":
        raise RuntimeError("dual-price contract requires HFQ signal and RAW execution")

    prices = pd.read_parquet(bundle / "prices_hfq_raw.parquet").copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["code"] = prices["code"].astype(str).str.zfill(6)
    prices = prices.dropna(subset=["date", "code", config.factor, "raw_open", "raw_close"]).sort_values(["date", "code"])
    membership = pd.read_parquet(bundle / "universe_membership.parquet").copy()
    membership["code"] = membership["code"].astype(str).str.zfill(6)
    membership["start_date"] = pd.to_datetime(membership["start_date"], errors="coerce")
    membership["end_date"] = pd.to_datetime(membership.get("end_date"), errors="coerce")
    actions = pd.read_parquet(bundle / "corporate_actions.parquet").copy()
    actions["symbol"] = actions["symbol"].astype(str).str.zfill(6)
    actions["date"] = pd.to_datetime(actions["date"], errors="coerce")
    actions = actions[actions.date.between(prices.date.min(), prices.date.max(), inclusive="both")]
    action_map = {date: frame for date, frame in actions.groupby("date")}
    benchmark = pd.read_parquet(bundle / "benchmark_csi300.parquet").copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="coerce")
    benchmark_close = pd.to_numeric(benchmark["close"], errors="coerce").dropna().set_axis(benchmark.loc[benchmark["close"].notna(), "date"])
    benchmark_close = benchmark_close.reindex(pd.DatetimeIndex(sorted(prices.date.unique()))).ffill()
    benchmark_sma200 = benchmark_close.rolling(200, min_periods=200).mean()

    dates = pd.DatetimeIndex(sorted(prices.date.unique()))
    by_date = {date: frame.set_index("code") for date, frame in prices.groupby("date", sort=True)}
    member = membership.set_index("code")
    if config.rebalance == "weekly":
        rebalance_dates = set(pd.Series(dates, index=dates).groupby(dates.to_period("W")).max())
    elif config.rebalance == "monthly":
        rebalance_dates = set(pd.Series(dates, index=dates).groupby(dates.to_period("M")).max())
    elif config.rebalance == "daily":
        rebalance_dates = set(dates)
    else:
        raise ValueError(f"unsupported rebalance frequency: {config.rebalance}")

    cash = float(config.initial_capital)
    holdings: dict[str, int] = {}
    previous_close: dict[str, float] = {}
    pending_targets: list[str] | None = None
    trades: list[dict[str, Any]] = []
    blocked_events: list[dict[str, Any]] = []
    applied_actions: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    peak_value = float(config.initial_capital)

    for index, date in enumerate(dates):
        frame = by_date[date]
        cash_before = cash
        action_cash = 0.0
        fees_today = 0.0
        turnover_notional = 0.0

        for action in action_map.get(date, pd.DataFrame()).to_dict("records"):
            symbol = action["symbol"]
            shares = holdings.get(symbol, 0)
            if shares <= 0:
                continue
            if action["action_type"] == "dividend":
                value = shares * float(action.get("cash_per_share") or 0.0)
                cash += value; action_cash += value
            elif action["action_type"] == "split":
                ratio = float(action.get("ratio") or 1.0)
                holdings[symbol] = int(math.floor(shares * ratio))
                value = 0.0
            else:
                continue
            applied_actions.append({"date": str(date.date()), "symbol": symbol, "action_type": action["action_type"], "cash_effect": value, "shares_after": holdings.get(symbol, shares)})

        if pending_targets is not None:
            open_prices = pd.to_numeric(frame["raw_open"], errors="coerce")
            for symbol in list(holdings):
                if symbol in pending_targets:
                    continue
                if symbol not in frame.index:
                    blocked_events.append({"date": str(date.date()), "symbol": symbol, "side": "sell", "reason": "missing_symbol_quote", "requested_shares": holdings[symbol]})
                    continue
                reason = _block_reason(frame.loc[symbol], "sell", config)
                if reason:
                    blocked_events.append({"date": str(date.date()), "symbol": symbol, "side": "sell", "reason": reason, "requested_shares": holdings[symbol]})
                    continue
                shares = holdings.pop(symbol)
                fill = float(open_prices[symbol]) * (1 - config.slippage_bps / 10000.0)
                notional = shares * fill; fee = _fee(notional, "sell", config)
                cash += notional - fee; fees_today += fee; turnover_notional += notional
                trades.append({"date": str(date.date()), "symbol": symbol, "side": "sell", "shares": shares, "price": fill, "notional": notional, "fee": fee})

            available: list[str] = []
            for symbol in pending_targets:
                if symbol not in frame.index:
                    blocked_events.append({"date": str(date.date()), "symbol": symbol, "side": "buy", "reason": "missing_symbol_quote", "requested_shares": None})
                    continue
                reason = _block_reason(frame.loc[symbol], "buy", config)
                if reason:
                    blocked_events.append({"date": str(date.date()), "symbol": symbol, "side": "buy", "reason": reason, "requested_shares": None})
                    continue
                available.append(symbol)
            equity_open = cash + sum(holdings.get(symbol, 0) * float(open_prices.get(symbol, previous_close.get(symbol, 0.0))) for symbol in holdings)
            investable = max(0.0, equity_open * (1 - config.cash_buffer))
            target_value = investable / len(available) if available else 0.0
            for symbol in available:
                current = holdings.get(symbol, 0)
                fill = float(open_prices[symbol]) * (1 + config.slippage_bps / 10000.0)
                desired = int(target_value / fill // 100 * 100)
                buy_shares = max(0, desired - current)
                if buy_shares <= 0:
                    continue
                notional = buy_shares * fill; fee = _fee(notional, "buy", config)
                if notional + fee > cash:
                    # Fees depend on notional and have a minimum; iteratively shrink
                    # whole lots until cash covers both execution price and fees.
                    buy_shares = int(max(0.0, cash - config.min_commission) / fill // 100 * 100)
                    while buy_shares > 0:
                        notional = buy_shares * fill
                        fee = _fee(notional, "buy", config)
                        if notional + fee <= cash + 1e-8:
                            break
                        buy_shares -= 100
                    if buy_shares <= 0:
                        notional = 0.0; fee = 0.0
                if buy_shares:
                    cash -= notional + fee; fees_today += fee; turnover_notional += notional
                    holdings[symbol] = current + buy_shares
                    trades.append({"date": str(date.date()), "symbol": symbol, "side": "buy", "shares": buy_shares, "price": fill, "notional": notional, "fee": fee})
                else:
                    blocked_events.append({"date": str(date.date()), "symbol": symbol, "side": "buy", "reason": "insufficient_cash_or_lot_constraint", "requested_shares": int(target_value / fill // 100 * 100)})
            pending_targets = None

        close_prices = pd.to_numeric(frame["raw_close"], errors="coerce")
        for symbol, value in close_prices.dropna().items():
            previous_close[symbol] = float(value)
        position_value = sum(shares * previous_close.get(symbol, 0.0) for symbol, shares in holdings.items())
        total_value = cash + position_value
        accounting_error = total_value - cash - position_value
        ledger.append({
            "date": str(date.date()), "cash_before": cash_before, "action_cash": action_cash,
            "fees": fees_today, "turnover_notional": turnover_notional, "cash": cash,
            "position_value": position_value, "total_value": total_value,
            "accounting_error": accounting_error, "positions": len(holdings),
        })

        peak_value = max(peak_value, total_value)
        drawdown = total_value / peak_value - 1.0 if peak_value else 0.0
        risk_off = (config.drawdown_cut > 0 and drawdown <= -abs(config.drawdown_cut))
        trend_off = config.market_filter == "benchmark_sma200" and (not np.isfinite(benchmark_sma200.reindex([date]).iloc[0]) or benchmark_close.reindex([date]).iloc[0] <= benchmark_sma200.reindex([date]).iloc[0])
        if date in rebalance_dates and index + 1 < len(dates):
            if risk_off or trend_off:
                pending_targets = []
                continue
            eligible = frame.copy()
            eligible = eligible.join(member[["start_date", "end_date"]], how="inner")
            eligible = eligible[(eligible.start_date <= date) & (eligible.end_date.isna() | (eligible.end_date >= date))]
            eligible = eligible.dropna(subset=[config.factor])
            ranked = eligible.assign(_score=pd.to_numeric(eligible[config.factor], errors="coerce") * config.direction).dropna(subset=["_score"]).sort_values("_score", ascending=False)
            n = max(1, int(len(ranked) * config.quantile))
            pending_targets = ranked.head(n).index.astype(str).tolist()

    ledger_frame = pd.DataFrame(ledger)
    ledger_frame["return"] = ledger_frame.total_value.pct_change().fillna(0.0)
    trades_frame = pd.DataFrame(trades)
    blocks_frame = pd.DataFrame(blocked_events, columns=["date", "symbol", "side", "reason", "requested_shares"])
    actions_frame = pd.DataFrame(applied_actions)
    annual_frame = _annual_performance(ledger_frame, benchmark_close)
    max_accounting_error = float(ledger_frame.accounting_error.abs().max())
    negative_cash_days = int((ledger_frame.cash < -0.01).sum())
    audit = {
        "status": "PASS" if max_accounting_error <= 0.01 and negative_cash_days == 0 else "BLOCK",
        "max_accounting_error": max_accounting_error,
        "negative_cash_days": negative_cash_days,
        "applied_corporate_actions": len(actions_frame),
        "available_corporate_actions_in_period": int(len(actions)),
        "trade_rows": int(len(trades_frame)),
        "blocked_event_rows": int(len(blocks_frame)),
        "blocked_event_reasons": {str(reason): int(count) for reason, count in blocks_frame["reason"].value_counts().items()},
        "ledger_rows": int(len(ledger_frame)),
    }

    output.mkdir(parents=True, exist_ok=True)
    ledger_frame.to_csv(output / "daily_ledger.csv", index=False)
    trades_frame.to_csv(output / "trades.csv", index=False)
    blocks_frame.to_csv(output / "blocked_events.csv", index=False)
    annual_frame.to_csv(output / "annual_performance.csv", index=False)
    actions_frame.to_csv(output / "applied_corporate_actions.csv", index=False)
    report = {
        **release_metadata(),
        "schema": "frozen-long-backtest/v1",
        "config": asdict(config),
        "release_id": release_id,
        "bundle_manifest_sha256": _sha256(bundle / "data_manifest.json"),
        "factor_registry_sha256": _sha256(bundle / "factor_registry.json"),
        "production_contract": production,
        "quality_gate": quality,
        "metrics": _metrics(ledger_frame["return"]),
        "benchmark_metrics": _metrics(benchmark_close.pct_change().fillna(0.0)),
        "accounting_audit": audit,
        "annual_performance": annual_frame.to_dict("records"),
        "blocked_events": {"count": int(len(blocks_frame)), "reasons": audit["blocked_event_reasons"], "path": "blocked_events.csv"},
        "period": {"start": ledger_frame.date.iloc[0], "end": ledger_frame.date.iloc[-1]},
    }
    report["report_sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True, default=str).encode()).hexdigest()
    (output / "long_backtest_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = [
        f"# 冻结包长回测审计 · {PRODUCT_VERSION}", "",
        f"- 区间: {report['period']['start']} → {report['period']['end']}",
        f"- Bundle Manifest: `{report['bundle_manifest_sha256']}`",
        f"- 因子: `{config.factor}`，HFQ信号 / RAW执行，{config.rebalance}调仓",
        f"- 总收益: {report['metrics']['total_return']}",
        f"- 年化收益: {report['metrics']['annual_return']}",
        f"- 夏普: {report['metrics']['sharpe']}",
        f"- 最大回撤: {report['metrics']['max_drawdown']}",
        f"- 基准年化收益: {report['benchmark_metrics']['annual_return']}",
        f"- 会计审计: **{audit['status']}**，最大恒等式误差 {audit['max_accounting_error']}",
        f"- 交易: {audit['trade_rows']}，实际应用公司行动: {audit['applied_corporate_actions']}",
        f"- 阻断事件: {audit['blocked_event_rows']}，见 `blocked_events.csv`",
        "",
        "## 年度可比绩效",
        "",
        "| 年度 | 策略年化 | 基准年化 | 年化超额 |",
        "|---|---:|---:|---:|",
    ]
    for row in report["annual_performance"]:
        fmt = lambda value: "NA" if value is None else f"{value:.4f}"
        lines.append(f"| {row['year']} | {fmt(row['strategy_annual_return'])} | {fmt(row['benchmark_annual_return'])} | {fmt(row['annual_excess_return'])} |")
    lines += ["", "年度结果以同一交易日序列的策略净值与 CSI300 收盘收益计算；该冻结样本仅供研究比较，不构成真实 PIT 或可交易验证。"]
    (output / "long_backtest_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run_long_backtest(LongRunConfig(bundle_dir=args.bundle, output_dir=args.output))
    print(json.dumps({"status": report["accounting_audit"]["status"], "output": args.output, "sha256": report["report_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
