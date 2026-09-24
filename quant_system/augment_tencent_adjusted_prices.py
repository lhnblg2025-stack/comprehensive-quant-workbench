"""Add Tencent adjusted signal prices to a Tushare raw execution-price universe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--start", default="2016-05-19")
    parser.add_argument("--end", default="2026-08-31")
    parser.add_argument("--adjust", choices=("qfq", "hfq"), default="qfq")
    args = parser.parse_args(argv)
    from .sources_tencent import fetch_tencent_daily
    root = Path(args.root); price_dir = root / "raw" / "prices"
    failures = []; updated = 0; skipped = 0
    for index, path in enumerate(sorted(price_dir.glob("*.parquet")), 1):
        raw = pd.read_parquet(path)
        if {"hfq_open", "hfq_high", "hfq_low", "hfq_close"}.issubset(raw.columns) and raw["hfq_close"].notna().any():
            skipped += 1; continue
        code = path.stem.zfill(6)
        try:
            signal = fetch_tencent_daily(code, start=args.start.replace("-", ""), end=args.end.replace("-", ""), adjust=args.adjust)
            if signal.empty:
                failures.append({"code": code, "reason": "empty_tencent_adjusted"}); continue
            if set(signal.get("tencent_price_key", pd.Series(dtype=str)).dropna().unique()) - {f"{args.adjust}day"}:
                failures.append({"code": code, "reason": "adjusted_key_downgrade"}); continue
            signal = signal.rename(columns={name: f"hfq_{name}" for name in ("open", "high", "low", "close")})
            signal = signal[["date", "hfq_open", "hfq_high", "hfq_low", "hfq_close", "tencent_price_key"]]
            raw["date"] = pd.to_datetime(raw["date"]); signal["date"] = pd.to_datetime(signal["date"])
            out = raw.drop(columns=[c for c in ("hfq_open", "hfq_high", "hfq_low", "hfq_close") if c in raw.columns]).merge(signal, on="date", how="left")
            out["price_adjustment_source"] = out["tencent_price_key"].map(lambda key: f"tencent_{args.adjust}" if key == f"{args.adjust}day" else "tencent_adjustment_downgrade")
            out.to_parquet(path, index=False); updated += 1
        except Exception as exc:
            failures.append({"code": code, "reason": f"{type(exc).__name__}:{str(exc)[:160]}"})
        if index % 25 == 0:
            print(json.dumps({"processed": index, "updated": updated, "skipped": skipped, "total": len(list(price_dir.glob('*.parquet')))}), flush=True)
    report = {"schema": "tencent_adjusted_signal_prices/v1", "signal_adjust": args.adjust, "source": "tencent_direct", "updated": updated, "skipped": skipped, "failures": failures}
    (root / "adjusted_price_augmentation_manifest.json").write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
