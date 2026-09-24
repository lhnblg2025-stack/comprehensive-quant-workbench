#!/usr/bin/env python3
"""Multi-market absolute-return overlay.

A-share factor sleeve (walk-forward artifact) + commodity trend sleeve
(gold/silver/hog/lithium/crude) with monthly point-in-time trend signals,
volatility targeting and cash, blended 70/30. Benchmarks: CSI300 ETF and
S&P500, reported separately per market.

Outputs generated/multi_asset_backtest_latest.json. Research-only; the
futures sleeve uses main continuous contracts, so roll costs are proxied
with an explicit per-rebalance roll cost and flagged in the manifest.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "generated" / "multi_asset_backtest_latest.json"
FACTOR_ARTIFACT = ROOT / "generated" / "factor_backtest_latest.json"

COMMODITY_ASSETS = (
    ("gold", "AU0", 0.20, 0.25),
    ("silver", "AG0", 0.20, 0.25),
    ("hog", "LH0", 0.15, 0.20),
    ("lithium_carbonate", "LC0", 0.15, 0.15),
    ("crude", "SC0", 0.15, 0.20),
)
COMMODITY_SLEEVE_CAP = 0.60
COMMODITY_TREND_DAYS = 60
COST_BPS = 10.0
ROLL_COST_BPS = 12.0
BLEND_ASHARE = 0.70
BLEND_COMMODITY = 0.30
REBALANCE = "monthly"


@dataclass
class SleeveResult:
    returns: pd.Series
    turnovers: list[float]
    windows: list[dict[str, Any]]


def _json_number(value: Any, digits: int = 6) -> Any:
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), digits) if math.isfinite(float(value)) else None
    return value


def _performance(series: pd.Series) -> dict[str, float]:
    from quant_system.backtest import _perf_metrics

    clean = series.dropna()
    if clean.empty:
        return {"annual_return": 0.0, "annual_vol": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    metrics = _perf_metrics(clean, (1.0 + clean).cumprod())
    return {key: _json_number(metrics.get(key, 0.0), 6)
            for key in ("annual_return", "annual_vol", "sharpe", "max_drawdown")}


def _rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    values = pd.Series(index=index, data=index)
    return pd.DatetimeIndex(values.groupby(index.to_period("M")).first().tolist())


def load_commodity_returns() -> dict[str, pd.Series]:
    from quant_web.market_history import get_market_history

    returns: dict[str, pd.Series] = {}
    for asset, _, _, _ in COMMODITY_ASSETS:
        result = get_market_history(asset, days=2000)
        if not result.get("ok") or not result.get("rows"):
            continue
        frame = pd.DataFrame(result["rows"])
        frame["date"] = pd.to_datetime(frame["date"])
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        series = frame.set_index("date")["close"].dropna().sort_index()
        returns[asset] = series.pct_change().dropna()
    return returns


def _clean_returns(series: pd.Series, sanity: float = 0.25) -> pd.Series:
    clean = series.copy()
    clean = clean[np.isfinite(clean)]
    clean = clean[(clean > -sanity) & (clean < sanity)]
    return clean


def load_benchmarks() -> dict[str, pd.Series]:
    from quant_web.market_history import get_market_history

    benchmarks: dict[str, pd.Series] = {}
    csi = get_market_history("a_share_etf", days=2000)
    if csi.get("ok"):
        frame = pd.DataFrame(csi["rows"])
        frame["date"] = pd.to_datetime(frame["date"])
        # etf_history 混存多只 ETF，必须先按 code=510300 过滤再计算收益。
        if "code" in frame.columns:
            frame = frame[frame["code"].astype(str).str.zfill(6) == "510300"]
        price_col = "close" if "close" in frame.columns else "price"
        frame[price_col] = pd.to_numeric(frame[price_col], errors="coerce")
        frame = frame[frame[price_col] > 0].drop_duplicates("date", keep="last")
        benchmarks["csi300_etf"] = _clean_returns(frame.set_index("date")[price_col].sort_index().pct_change().dropna())
    spx = get_market_history("sp500", days=2000)
    if spx.get("ok"):
        frame = pd.DataFrame(spx["rows"])
        frame["date"] = pd.to_datetime(frame["date"])
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame[frame["close"] > 0].drop_duplicates("date", keep="last")
        benchmarks["sp500"] = _clean_returns(frame.set_index("date")["close"].sort_index().pct_change().dropna())
    return benchmarks


def run_commodity_sleeve(returns: dict[str, pd.Series]) -> SleeveResult:
    panel = pd.DataFrame(returns).sort_index()
    common = panel.index
    rebalances = set(_rebalance_dates(common))
    weights: dict[str, float] = {}
    daily: list[tuple[pd.Timestamp, float]] = []
    windows: list[dict[str, Any]] = []
    turnovers: list[float] = []

    for position, date in enumerate(common):
        cost = 0.0
        if date in rebalances and position >= COMMODITY_TREND_DAYS + 1:
            history = panel.iloc[max(0, position - COMMODITY_TREND_DAYS):position]
            targets: dict[str, float] = {}
            details: list[dict[str, Any]] = []
            for asset in panel.columns:
                asset_hist = history[asset].dropna()
                if len(asset_hist) < 30:
                    details.append({"asset": asset, "signal": "insufficient_history"})
                    continue
                momentum = float((1.0 + asset_hist).prod() - 1.0)
                vol = float(asset_hist.std(ddof=1) * math.sqrt(252))
                target_vol = next((tv for a, _, tv, _ in COMMODITY_ASSETS if a == asset), 0.15)
                cap = next((c for a, _, _, c in COMMODITY_ASSETS if a == asset), 0.20)
                if momentum <= 0 or vol <= 1e-12:
                    details.append({"asset": asset, "momentum": _json_number(momentum), "signal": "cash"})
                    continue
                raw = min(target_vol / vol, cap)
                targets[asset] = raw
                details.append({"asset": asset, "momentum": _json_number(momentum), "vol": _json_number(vol),
                                "raw_weight": _json_number(raw), "signal": "trend"})
            total = sum(targets.values())
            if total > COMMODITY_SLEEVE_CAP:
                scale = COMMODITY_SLEEVE_CAP / total
                targets = {k: v * scale for k, v in targets.items()}
            turnover = sum(abs(targets.get(asset, 0.0) - weights.get(asset, 0.0)) for asset in set(targets) | set(weights)) / 2.0
            cost = turnover * COST_BPS / 10000.0 + (COMMODITY_SLEEVE_CAP * ROLL_COST_BPS / 10000.0 if targets else 0.0)
            weights = targets
            turnovers.append(turnover)
            windows.append({
                "rebalance_date": str(date.date()), "weights": {k: _json_number(v) for k, v in weights.items()},
                "turnover": _json_number(turnover), "details": details,
            })
        day = panel.loc[date].dropna()
        if weights and not day.empty:
            gross = float(sum(weights.get(asset, 0.0) * (day[asset] if asset in day else 0.0) for asset in weights))
        else:
            gross = 0.0
        daily.append((date, gross - cost))

    series = pd.Series(dict(daily), dtype=float).sort_index()
    return SleeveResult(returns=series, turnovers=turnovers, windows=windows)


def load_ashare_sleeve() -> tuple[pd.Series, str | None]:
    payload = json.loads(FACTOR_ARTIFACT.read_text(encoding="utf-8"))
    candidate_series = payload.get("candidate_series", {}) or {}
    # 混合层优先选择现金门控最强的绝对收益腿，而非分数最高的冠军腿。
    preferred = ["absolute_defensive", "absolute_balanced", "momentum_tilt", "quality_concentrated"]
    chosen = next((name for name in preferred if candidate_series.get(name)), None)
    if chosen is None:
        chosen = next(iter(candidate_series), None) if candidate_series else None
    if chosen is None:
        return pd.Series(dtype=float), None
    series = candidate_series[chosen] or {}
    dates = series.get("dates", [])
    returns = series.get("group_1_net", [])
    if not dates or not returns:
        return pd.Series(dtype=float), None
    window_count = len(payload.get("iteration", {}).get("windows", []))
    if window_count < 3:
        return pd.Series(dtype=float), None
    sleeve = pd.Series(
        [None if v is None else float(v) for v in returns],
        index=pd.to_datetime(dates), dtype=float,
    ).dropna()
    return sleeve, f"{chosen}:{payload.get('validation', {}).get('status')}"


def run_blend(ashare: pd.Series, commodity: pd.Series) -> tuple[pd.Series, list[dict[str, Any]]]:
    """Adaptive risk budget: weight sleeves by point-in-time trailing Sharpe.

    Both sleeves non-positive → full cash. No lookahead: only returns before
    each rebalance date are used to size the next month.
    """
    aligned = pd.concat([ashare.rename("a"), commodity.rename("c")], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float), []
    rebalances = set(_rebalance_dates(aligned.index))
    rows: list[tuple[pd.Timestamp, float]] = []
    weights_a = BLEND_ASHARE
    windows: list[dict[str, Any]] = []
    lookback = 120

    for position, (date, row) in enumerate(aligned.iterrows()):
        if date in rebalances and position >= lookback:
            history = aligned.iloc[max(0, position - lookback):position]
            sharpes: dict[str, float] = {}
            for name in ("a", "c"):
                hist = history[name].dropna()
                std = float(hist.std(ddof=1))
                sharpes[name] = float(hist.mean() / std * math.sqrt(252)) if len(hist) >= 30 and std > 1e-12 else 0.0
            positive = {k: max(v, 0.0) for k, v in sharpes.items()}
            total = sum(positive.values())
            weights_a = positive["a"] / total if total > 1e-12 else 0.0
            windows.append({
                "rebalance_date": str(date.date()),
                "trailing_sharpe": {k: _json_number(v) for k, v in sharpes.items()},
                "weight_ashare": _json_number(weights_a),
                "weight_commodity": _json_number(1.0 - weights_a),
                "signal": "blend" if total > 1e-12 else "cash",
            })
        rows.append((date, float(weights_a * row["a"] + (1.0 - weights_a) * row["c"])))
    return pd.Series(dict(rows), dtype=float).sort_index(), windows


def main() -> int:
    try:
        commodity_returns = load_commodity_returns()
        benchmarks = load_benchmarks()
        commodity = run_commodity_sleeve(commodity_returns)
        ashare, ashare_status = load_ashare_sleeve()
        ashare_available = not ashare.empty
        blend, blend_windows = run_blend(ashare, commodity.returns) if ashare_available else (commodity.returns, [])

        common_start = blend.index.min()
        benchmark_metrics: dict[str, Any] = {}
        for name, series in benchmarks.items():
            clipped = series.loc[series.index >= common_start]
            benchmark_metrics[name] = {
                **_performance(clipped),
                "start": str(clipped.index.min().date()) if not clipped.empty else None,
                "end": str(clipped.index.max().date()) if not clipped.empty else None,
            }

        payload = {
            "ok": True,
            "schema_version": "multi-asset.v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "engine": "scripts.multi_asset_backtest",
            "validation": {
                "status": "research_only",
                "deployable": False,
                "point_in_time": True,
                "flags": [
                    "期货采用主力连续合约，展期成本以每次调仓12bps近似，未逐笔建模",
                    "A股腿使用独立HFQ后复权滚动回测产物",
                    "商品趋势信号只用调仓日之前历史，无前视",
                ],
            },
            "provenance": {
                "blend_method": "adaptive_trailing_sharpe_120d",
                "ashare_sleeve_weight_reference": BLEND_ASHARE,
                "commodity_sleeve_weight_reference": BLEND_COMMODITY,
                "commodity_sleeve_cap": COMMODITY_SLEEVE_CAP,
                "trend_days": COMMODITY_TREND_DAYS,
                "cost_bps": COST_BPS,
                "roll_cost_bps": ROLL_COST_BPS,
                "rebalance": REBALANCE,
                "ashare_available": ashare_available,
                "ashare_sleeve_validation": ashare_status,
                "commodity_assets": [{"asset": a, "symbol": s, "target_vol": tv, "cap": c} for a, s, tv, c in COMMODITY_ASSETS],
                "commodity_assets_loaded": sorted(commodity_returns),
                "benchmarks": sorted(benchmarks),
                "period": {"start": str(blend.index.min().date()), "end": str(blend.index.max().date())},
            },
            "metrics": {
                "blend": _performance(blend),
                "commodity_sleeve": _performance(commodity.returns),
                "ashare_sleeve": _performance(ashare) if ashare_available else None,
                "benchmarks": benchmark_metrics,
            },
            "windows": commodity.windows,
            "blend_windows": blend_windows,
            "series": {
                "dates": [str(d.date()) for d in blend.index],
                "blend": [_json_number(v) for v in blend.tolist()],
                "commodity": [_json_number(v) for v in commodity.returns.reindex(blend.index).tolist()],
                "ashare": [_json_number(v) for v in (ashare.reindex(blend.index).tolist() if ashare_available else [])],
            },
        }
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"ok": True, "path": str(OUT), "metrics": payload["metrics"], "period": payload["provenance"]["period"]}, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
