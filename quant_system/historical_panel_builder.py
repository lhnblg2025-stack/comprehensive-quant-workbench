"""Build a reproducible multi-year factor panel from local OHLCV parquet data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_hfq_panel(symbols: list[str], output_dir: str | Path, *, start: str, end: str) -> dict:
    """Fetch aligned HFQ signal prices and raw execution prices."""
    from .sources_all import fetch_daily_unified

    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    frames = []; sources = []; failed_symbols = []
    for symbol in sorted({str(item).zfill(6) for item in symbols}):
        try:
            hfq = fetch_daily_unified(symbol, start=start.replace("-", ""), end=end.replace("-", ""), adjust="hfq")
            raw = fetch_daily_unified(symbol, start=start.replace("-", ""), end=end.replace("-", ""), adjust="")
        except Exception as exc:
            failed_symbols.append({"symbol": symbol, "error": str(exc)})
            continue
        if hfq is None or hfq.empty or raw is None or raw.empty:
            failed_symbols.append({"symbol": symbol, "error": "empty_hfq_or_raw"})
            continue
        hfq = hfq.copy(); raw = raw.copy(); hfq["date"] = pd.to_datetime(hfq["date"]); raw["date"] = pd.to_datetime(raw["date"])
        signal_cols = ["date", "open", "high", "low", "close"]
        signal = hfq[signal_cols].rename(columns={c: f"hfq_{c}" for c in signal_cols if c != "date"})
        execution = raw[["date", "open", "high", "low", "close", "volume", "amount"]].rename(columns={c: f"raw_{c}" for c in ("open", "high", "low", "close")})
        frame = signal.merge(execution, on="date", how="inner"); frame["code"] = symbol
        # Generic OHLC aliases intentionally point to HFQ for factor and label research.
        frame[["open", "high", "low", "close"]] = frame[["hfq_open", "hfq_high", "hfq_low", "hfq_close"]]
        frames.append(compute_price_factors(frame)); sources.append({"symbol": symbol, "signal_adjust": "hfq", "execution_adjust": "raw", "rows": int(len(frame))})
    if not frames:
        raise ValueError("no HFQ data returned")
    panel = pd.concat(frames, ignore_index=True).sort_values(["date", "code"])
    path = output / "hfq_panel.parquet"; panel.to_parquet(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"schema_version": "2.0", "signal_adjust": "hfq", "execution_adjust": "raw", "start": start, "end": end, "symbols": [x["symbol"] for x in sources], "failed_symbols": failed_symbols, "sources": sources, "rows": int(len(panel)), "dates": int(panel["date"].nunique()), "sha256": digest, "output": str(path.resolve())}
    (output / "panel_manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    return manifest


def compute_price_factors(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.sort_values("date").copy()
    close = pd.to_numeric(data["close"], errors="coerce")
    volume = pd.to_numeric(data.get("volume"), errors="coerce")
    for period in (20, 60, 120):
        data[f"mom{period}"] = close / close.shift(period) - 1.0
    rolling_high = close.rolling(252, min_periods=60).max()
    data["dist_52w"] = close / rolling_high - 1.0
    data["amp20"] = (pd.to_numeric(data["high"], errors="coerce") / pd.to_numeric(data["low"], errors="coerce") - 1.0).rolling(20, min_periods=10).mean()
    data["vol_ratio20"] = volume / volume.rolling(20, min_periods=10).mean()
    if "turnover" not in data:
        data["turnover"] = np.nan
    return data


def _select_paths(files: list[Path], max_symbols: int) -> list[Path]:
    """Choose a deterministic, board-aware sample without look-ahead selection."""
    buckets = {"main": [], "growth": [], "star": [], "other": []}
    for path in sorted(files):
        code = path.stem.zfill(6)
        bucket = "star" if code.startswith("68") else "growth" if code.startswith(("30", "39")) else "main" if code.startswith(("00", "60")) else "other"
        buckets[bucket].append(path)
    quotas = {"main": int(max_symbols * 0.60), "growth": int(max_symbols * 0.20), "star": int(max_symbols * 0.15), "other": max_symbols}
    selected = []
    for bucket in ("main", "growth", "star", "other"):
        selected.extend(buckets[bucket][:quotas[bucket]])
    if len(selected) < max_symbols:
        used = set(selected)
        selected.extend(path for path in sorted(files) if path not in used)
    return selected[:max_symbols]


def build_pit_panel(kline_dir: str | Path, output_dir: str | Path, *, start: str, end: str,
                    max_symbols: int = 500, min_rows: int = 252) -> dict:
    """Build a long local PIT panel with listing/delisting availability metadata.

    Local kline files are treated as point-in-time observations. No current index
    membership is inferred; each symbol is available only between its first and
    last observed date, which avoids silently filling pre-listing history.
    """
    manifest = build_panel(kline_dir, output_dir, start=start, end=end, max_symbols=max_symbols, min_rows=min_rows)
    panel_path = Path(manifest["output"])
    panel = pd.read_parquet(panel_path)
    bounds = panel.groupby("code")["date"].agg(first_seen="min", last_seen="max").reset_index()
    panel = panel.merge(bounds, on="code", how="left")
    panel["pit_available"] = panel["date"].between(panel["first_seen"], panel["last_seen"], inclusive="both")
    panel.to_parquet(panel_path, index=False)
    manifest.update({"schema_version": "pit-panel/v1", "pit_rule": "first_seen_to_last_seen",
                     "symbols": int(panel["code"].nunique()), "rows": int(len(panel)),
                     "sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest()})
    (panel_path.parent / "panel_manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    return manifest


def build_panel(kline_dir: str | Path, output_dir: str | Path, *, start: str, end: str, max_symbols: int = 100, min_rows: int = 800) -> dict:
    source = Path(kline_dir); output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    files = _select_paths(list(source.glob("*.parquet")), max_symbols)
    frames = []; selected = []
    for path in files:
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        required = {"date", "open", "high", "low", "close", "volume"}
        if not required.issubset(frame.columns) or len(frame) < min_rows:
            continue
        frame = frame.copy(); frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame[(frame["date"] >= pd.Timestamp(start)) & (frame["date"] <= pd.Timestamp(end))]
        if len(frame) < min_rows:
            continue
        frame["code"] = path.stem.zfill(6)
        frames.append(compute_price_factors(frame)); selected.append(path.stem.zfill(6))
    if not frames:
        raise ValueError("no kline files satisfy historical panel requirements")
    panel = pd.concat(frames, ignore_index=True).sort_values(["date", "code"])
    panel_path = output / "historical_panel.parquet"; panel.to_parquet(panel_path, index=False)
    digest = hashlib.sha256(panel_path.read_bytes()).hexdigest()
    manifest = {"schema_version": "1.0", "source": str(source.resolve()), "output": str(panel_path.resolve()), "start": start, "end": end, "max_symbols": max_symbols, "min_rows": min_rows, "selected_symbols": selected, "rows": len(panel), "dates": int(panel["date"].nunique()), "date_min": str(panel["date"].min().date()), "date_max": str(panel["date"].max().date()), "sha256": digest}
    (output / "panel_manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kline-dir", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start", default="2015-01-01"); parser.add_argument("--end", required=True)
    parser.add_argument("--pit", action="store_true", help="write PIT availability metadata")
    parser.add_argument("--max-symbols", type=int, default=100); parser.add_argument("--min-rows", type=int, default=800)
    args = parser.parse_args(argv)
    builder = build_pit_panel if args.pit else build_panel
    result = builder(args.kline_dir, args.output_dir, start=args.start, end=args.end, max_symbols=args.max_symbols, min_rows=args.min_rows)
    print(json.dumps({"output": result["output"], "rows": result["rows"], "dates": result["dates"], "symbols": len(result["selected_symbols"]), "sha256": result["sha256"]}, ensure_ascii=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
