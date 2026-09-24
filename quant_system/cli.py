from __future__ import annotations

import argparse
import json
from pathlib import Path

from .backtest import run_backtest
from .closed_loop import LoopConfig, run_loop
from .full_chain import FullChainConfig, run_full_chain
from .factor_backtest_runner import Config as FactorBacktestConfig, run as run_factor_backtest
from .config import DEFAULT_PORTFOLIO, DEFAULT_STRATEGY
from .data import fetch_many, load_market_state
from .risk import enrich_trade_plan
from .signals import latest_signal


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "generated" / "quant_system"


def cmd_scan(args: argparse.Namespace) -> int:
    try:
        market = load_market_state()
        data = fetch_many(args.symbols, start=args.start, end=args.end, use_cache=not args.no_cache)
        failures = _extract_failures(data)
        plans: list[dict] = []
        skipped: list[dict] = []
        for symbol, df in data.items():
            if symbol == "__failures__":
                continue
            sig = latest_signal(symbol, df, DEFAULT_STRATEGY)
            if sig.get("status") == "insufficient_data":
                # P2-Q25-fix(M300): 数据不足记录单独归入 skipped, 不再混入 plans
                skipped.append(sig)
                continue
            plans.append(enrich_trade_plan(sig, market, DEFAULT_STRATEGY, DEFAULT_PORTFOLIO))
        payload = {
            "market_state": market,
            "plans": plans,
            "skipped_insufficient_data": skipped,
            "failures": failures,
            "disclaimer": "Research only. Manual confirmation required. No auto trading.",
        }
        return _emit(payload, args.output)
    except Exception as e:
        # P2-Q25-fix(M300): 数据源/网络失败输出结构化错误并以非 0 退出码结束
        return _emit_error("scan", str(e), args.output)


def cmd_factor_backtest(args: argparse.Namespace) -> int:
    try:
        from .factor_backtest_runner import DEFAULT_FACTORS
        config = FactorBacktestConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            factors=tuple(args.factors or DEFAULT_FACTORS),
            quantile=args.quantile,
            cost_bps=args.cost_bps,
            min_stocks=args.min_stocks,
            oos_ratio=args.oos_ratio,
            annualization=args.annualization,
            rebalance=args.rebalance,
        )
        report = run_factor_backtest(config)
        print(json.dumps({"output_dir": args.output_dir, "manifest_sha256": report["manifest_sha256"], "factors": len(report["results"])}, ensure_ascii=False))
        return 0
    except Exception as e:
        return _emit_error("factor-backtest", str(e), None)


def cmd_full_chain(args: argparse.Namespace) -> int:
    try:
        result = run_full_chain(FullChainConfig(data=args.data, output_dir=args.output_dir, factors=tuple(args.factors),
                                                 rebalance=args.rebalance, top_n=args.top_n, release_id=args.release_id))
        print(json.dumps({"output_dir": args.output_dir, "quality_gate": result["quality_gate"], "execution": result["execution"], "sha256": result["sha256"]}, ensure_ascii=False))
        return 0
    except Exception as e:
        return _emit_error("full-chain", str(e), None)


def cmd_closed_loop(args: argparse.Namespace) -> int:
    try:
        config = LoopConfig(
            factors=tuple(args.factors),
            directions=tuple((name, int(direction)) for name, direction in (item.split(":", 1) for item in (args.directions or []))),
            top_n=args.top_n,
            horizons=tuple(args.horizons),
            quantiles=args.quantiles,
            cost_bps=args.cost_bps,
            time_stop=args.time_stop,
        )
        result = run_loop(args.data, args.output_dir, config, release_id=args.release_id)
        print(json.dumps({"output_dir": args.output_dir, **result["manifest"]}, ensure_ascii=False))
        return 0
    except Exception as e:
        return _emit_error("closed-loop", str(e), None)


def cmd_backtest(args: argparse.Namespace) -> int:
    try:
        data = fetch_many(args.symbols, start=args.start, end=args.end, use_cache=not args.no_cache)
        failures = _extract_failures(data)
        results = {symbol: run_backtest(df, DEFAULT_STRATEGY, DEFAULT_PORTFOLIO)
                   for symbol, df in data.items() if symbol != "__failures__"}
        payload = {"results": results, "failures": failures, "disclaimer": "Backtest is hypothetical and includes simple commission/slippage assumptions."}
        return _emit(payload, args.output)
    except Exception as e:
        # P2-Q25-fix(M300): 数据源/网络失败输出结构化错误并以非 0 退出码结束
        return _emit_error("backtest", str(e), args.output)


def _emit(payload: dict, output: str | None) -> int:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if output:
        path = Path(output)
        if not path.is_absolute():
            path = OUT_DIR / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        print(path)
    else:
        print(text)
    return 0


def _emit_error(command: str, error: str, output: str | None) -> int:
    """输出结构化错误并以非 0 退出码返回(供 main 透传)。

    # P2-Q25-fix(M300): cmd_scan/cmd_backtest 失败时不再裸抛异常, 而是
    输出 {ok:false, error:...} 结构并返回 1。
    """
    payload = {"ok": False, "command": command, "error": error}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if output:
        path = Path(output)
        if not path.is_absolute():
            path = OUT_DIR / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        print(path)
    else:
        print(text)
    return 1


def _extract_failures(data: dict) -> list[dict]:
    failures = data.get("__failures__")
    if failures is None:
        return []
    return failures.to_dict(orient="records")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A-share quant research system MVP")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="Generate latest signal and risk-controlled paper plan")
    scan.add_argument("--symbols", nargs="+", required=True)
    scan.add_argument("--start", default="20240101")
    scan.add_argument("--end")
    scan.add_argument("--output")
    scan.add_argument("--no-cache", action="store_true")
    scan.set_defaults(func=cmd_scan)

    backtest = sub.add_parser("backtest", help="Run simple trend strategy backtest")
    backtest.add_argument("--symbols", nargs="+", required=True)
    backtest.add_argument("--start", default="20240101")
    backtest.add_argument("--end")
    backtest.add_argument("--output")
    backtest.add_argument("--no-cache", action="store_true")
    backtest.set_defaults(func=cmd_backtest)

    full_chain = sub.add_parser("full-chain", help="Run PIT research, OOS gate, execution audit and paper ledger")
    full_chain.add_argument("--data", required=True)
    full_chain.add_argument("--output-dir", required=True)
    full_chain.add_argument("--factors", nargs="+", required=True)
    full_chain.add_argument("--rebalance", choices=("daily", "weekly", "monthly"), default="weekly")
    full_chain.add_argument("--top-n", type=int, default=10)
    full_chain.add_argument("--release-id", default="local")
    full_chain.set_defaults(func=cmd_full_chain)

    closed_loop = sub.add_parser("closed-loop", help="Build candidate library, diagnostics and paper plan from local panel")
    closed_loop.add_argument("--data", required=True, help="Parquet/CSV panel or directory")
    closed_loop.add_argument("--output-dir", default=str(ROOT / "generated" / "closed_loop"))
    closed_loop.add_argument("--factors", nargs="+", required=True)
    closed_loop.add_argument("--directions", nargs="*", help="Optional factor direction overrides, e.g. pe_pct252:-1")
    closed_loop.add_argument("--top-n", type=int, default=10)
    closed_loop.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 5, 10])
    closed_loop.add_argument("--quantiles", type=int, default=5)
    closed_loop.add_argument("--cost-bps", type=float, default=15.0)
    closed_loop.add_argument("--time-stop", type=int, default=5)
    closed_loop.add_argument("--release-id", default="local")
    closed_loop.set_defaults(func=cmd_closed_loop)

    factor_backtest = sub.add_parser("factor-backtest", help="Run cross-sectional factor quant backtest")
    factor_backtest.add_argument("--data-dir", default=str(ROOT / ".." / "data_warehouse" / "feature_store"))
    factor_backtest.add_argument("--output-dir", default=str(ROOT / "generated" / "factor_backtest"))
    factor_backtest.add_argument("--factors", nargs="+", default=None)
    factor_backtest.add_argument("--quantile", type=float, default=0.2)
    factor_backtest.add_argument("--cost-bps", type=float, default=15.0)
    factor_backtest.add_argument("--min-stocks", type=int, default=30)
    factor_backtest.add_argument("--oos-ratio", type=float, default=0.3)
    factor_backtest.add_argument("--annualization", type=float, default=252.0)
    factor_backtest.add_argument("--rebalance", choices=("daily", "weekly", "monthly"), default="daily")
    factor_backtest.set_defaults(func=cmd_factor_backtest)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
