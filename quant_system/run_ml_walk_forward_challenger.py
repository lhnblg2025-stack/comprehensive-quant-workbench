"""Walk-forward ML challenger on the dual-price OHLCV panel.

Uses per-stock chronological features and an embargoed expanding split. The
model output is a probability-ranked long-only portfolio; no random split and
no future labels are used. Results remain research-only until trade-state data
passes the production gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["mom_3_1", "mom_6_1", "mom_12_1", "trend_60", "trend_120", "rsi_14", "low_vol_60", "volatility_20", "gap_reversal", "volume_ratio_20", "price_efficiency_20", "amount_acceleration", "downside_vol_60"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--train-days", type=int, default=504); parser.add_argument("--test-days", type=int, default=126); parser.add_argument("--embargo", type=int, default=20); parser.add_argument("--hold-days", type=int, default=5); parser.add_argument("--cost-bps", type=float, default=15.0)
    args = parser.parse_args(argv)
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    panel = pd.read_parquet(args.panel)
    panel["date"] = pd.to_datetime(panel["date"]); panel = panel.sort_values(["code", "date"])
    panel["label"] = panel.groupby("code")["raw_open"].shift(-args.hold_days) / panel.groupby("code")["raw_open"].shift(-1) - 1.0
    panel["y"] = (panel["label"] > 0).astype(int)
    dates = pd.DatetimeIndex(sorted(panel.date.unique())); records = []; aucs = []
    starts = range(args.train_days, len(dates) - args.test_days, args.test_days)
    for start in starts:
        train_dates = dates[start - args.train_days:start - args.embargo]
        test_dates = dates[start:start + args.test_days]
        train = panel[panel.date.isin(train_dates)].dropna(subset=FEATURES + ["y"])
        test = panel[panel.date.isin(test_dates)].dropna(subset=FEATURES + ["y", "label"])
        if len(train) < 1000 or len(test) < 100:
            continue
        model = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15, learning_rate=.05, l2_regularization=1.0, random_state=42)
        model.fit(train[FEATURES], train.y)
        test = test.copy(); test["prob"] = model.predict_proba(test[FEATURES])[:, 1]
        if test.y.nunique() > 1: aucs.append(float(roc_auc_score(test.y, test.prob)))
        previous = set()
        for day_number, (day, group) in enumerate(test.groupby("date")):
            if day_number % max(1, args.hold_days):
                continue
            n = max(10, int(len(group) * .20)); selected = group.nlargest(n, "prob")
            current = set(selected.code.astype(str))
            turnover = 1.0 if not previous else 1.0 - len(current & previous) / max(1, len(current))
            net_return = float(selected.label.mean()) - turnover * args.cost_bps / 10000.0
            records.append({"date": day, "return": net_return, "gross_return": float(selected.label.mean()), "turnover": turnover, "n": int(len(selected)), "auc": aucs[-1] if aucs else None, "window_start": train_dates[0], "window_end": test_dates[-1]})
            previous = current
    result = pd.DataFrame(records).sort_values("date")
    if result.empty: raise ValueError("insufficient_ml_walk_forward_data")
    ret = result["return"].astype(float); curve = (1 + ret).cumprod(); sharpe = float(ret.mean() / ret.std(ddof=1) * np.sqrt(252)) if ret.std(ddof=1) else 0.0
    report = {"schema": "ml_walk_forward_challenger/v2", "status": "research_only", "features": FEATURES, "windows": int(result[["window_start", "window_end"]].drop_duplicates().shape[0]), "observations": int(len(result)), "annual_return_net": float(curve.iloc[-1] ** (252 / len(ret)) - 1), "annual_return_gross": float((1 + result["gross_return"]).prod() ** (252 / len(result)) - 1), "sharpe_net": sharpe, "max_drawdown_net": float((curve / curve.cummax() - 1).min()), "mean_auc": float(np.mean(aucs)) if aucs else None, "mean_turnover": float(result["turnover"].mean()), "mean_top_quantile": .20, "hold_days": args.hold_days, "cost_bps": args.cost_bps, "blocked_for_promotion": ["historical_trade_state_unverified", "corporate_actions_reconciliation_pending", "multiple_testing_family_incomplete"]}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); result.to_parquet(out.with_suffix(".parquet"), index=False); out.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=True, indent=2, default=str), encoding="utf-8"); print(json.dumps(report, ensure_ascii=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
