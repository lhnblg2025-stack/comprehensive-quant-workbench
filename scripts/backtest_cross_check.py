#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""回测双引擎对照 + 偏差审计默认输出（W2.6）。

背景
----
系统内存在多个回测引擎（backtest.py / backtest_pro.py / combined_backtest.py /
signal_backtest.py / event_backtest.py 等），同一策略在不同引擎里口径容易不一致。
本脚本把**同一策略(参数) + 同一标的行情**分别交给两个引擎各跑一遍，自动对照
核心指标并输出「偏差审计」。

双引擎选型
----------
- 引擎A = ``quant_system.backtest.run_backtest``
  * 事件驱动：T+1 次日开盘成交、止损/止盈、per-board 涨跌停、公司行为记账，
    单标的 100 股整手、按 max_position_pct 分仓。是 cli.py 实际使用的口径。
- 引擎B = ``quant_system.backtest_pro.MultiAssetBacktest``
  * 向量化：价格×信号矩阵、T+1 shift(1)、日收盘收益、A股成本模型
    (佣金/最低佣金/印花税/过户费/滑点)、单标的上限 max_position。

  两个引擎都消费同一份 ``generate_signals(df, strategy)`` 产出的信号列，且费率
  参数均从 ``PortfolioConfig`` 映射，因此「同一策略/同一参数」可比。

为何不选 combined_backtest.py：其策略为内置 RSI/MA 硬编码、行情走 akshare 实时
拉取、股票池固定 watchlist，无法接受任意 StrategyConfig/参数/股票池，与「给定
策略对照」不可比。

四件套（默认核心指标）
----------------------
以代码里已有、且两引擎同口径/可归一化的指标为准：
  1. 收益   total_return（%）
  2. 回撤   max_drawdown（%）
  3. 夏普   sharpe（两引擎均已统一到 MetricsCalculator.sharpe, rf=0.02/ddof=1）
  4. 换手   turnover（单边换手次数 = 0/1 仓位切换的 Σ|Δpos|/2，即完整往返次数；
           引擎A 从 trades+equity_curve 重建持仓，引擎B 从 positions 派生）

说明：代码里两引擎均无原生「换手率」字段（backtest_pro 的 win_rate 为**日**胜率、
backtest.py 的 win_rate_pct 为**交易**胜率，口径不同，故不纳入四件套），因此换手
由本脚本按同一公式从两引擎输出派生，保持可比。

用法
----
  python3 scripts/backtest_cross_check.py                          # 合成数据冒烟
  python3 scripts/backtest_cross_check.py --symbol 000001          # 本地 kline
  python3 scripts/backtest_cross_check.py --symbol 000001 600519 --out /tmp/x.md
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutureTimeout
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_system.config import PortfolioConfig, StrategyConfig          # noqa: E402
from quant_system.backtest import run_backtest                           # noqa: E402
from quant_system.backtest_pro import MultiAssetBacktest                 # noqa: E402
from quant_system.signals import generate_signals                        # noqa: E402

ENGINE_A = "backtest.run_backtest"
ENGINE_B = "backtest_pro.MultiAssetBacktest"

# 四件套规格：key / 中文标签 / 单位 / （偏差 = A - B，阈值判 |偏差|）
FOUR_SUITE: list[tuple[str, str, str]] = [
    ("total_return_pct", "收益", "%"),
    ("max_drawdown_pct", "回撤", "%"),
    ("sharpe", "夏普", ""),
    ("turnover", "换手(单边次数)", ""),
]

DEFAULT_THRESHOLDS: dict[str, float] = {
    "total_return_pct": 2.0,   # 收益差 > 2 个百分点
    "max_drawdown_pct": 2.0,   # 回撤差 > 2 个百分点
    "sharpe": 0.3,             # 夏普差 > 0.3
    "turnover": 2.0,           # 换手差 > 2 次
}

DEFAULT_TIMEOUT_S = 60.0


# ════════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════════

def _to_float(value) -> float | None:
    """安全转 float；NaN/Inf/异常 → None（用于区分“缺失”与“0”）。"""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _position_turnover(pos) -> float:
    """单边换手次数 = Σ|Δpos|/2（前补 0 基线）。pos 为 0/1 或 0/比例 序列。"""
    arr = np.asarray(pos, dtype=float)
    if arr.size == 0:
        return 0.0
    seq = np.concatenate([[0.0], arr])
    return float(np.abs(np.diff(seq)).sum() / 2.0)


def _turnover_engine_a(result: dict) -> float:
    """引擎A：从 trades + equity_curve 重建每日 0/1 持仓 → 换手次数。"""
    equity_curve = result.get("equity_curve") or []
    if not equity_curve:
        return 0.0
    by_date: dict[str, list[dict]] = {}
    for t in result.get("trades") or []:
        if t.get("side") in ("BUY", "SELL"):
            by_date.setdefault(str(t.get("date")), []).append(t)
    shares = 0
    pos: list[int] = []
    for pt in equity_curve:
        d = str(pt.get("date"))
        for t in by_date.get(d, []):
            if t.get("side") == "BUY":
                shares = int(t.get("shares", shares) or 0)
            else:  # SELL
                shares = 0
        pos.append(1 if shares > 0 else 0)
    return _position_turnover(pos)


def _turnover_engine_b(result, symbol: str) -> float:
    """引擎B：从 BacktestResult.positions（shift 后的权重）派生 0/1 持仓 → 换手次数。"""
    positions = getattr(result, "positions", None)
    if positions is None or symbol not in positions.columns:
        return 0.0
    pos = (positions[symbol] > 0).astype(float)
    return _position_turnover(pos.values)


def _run_with_timeout(fn, timeout: float) -> dict:
    """软超时包装：线程内执行，超时/异常显式记录（不静默吞掉）。"""
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(fn)
    try:
        return {"ok": True, "result": fut.result(timeout=timeout)}
    except _FutureTimeout:
        return {"ok": False, "error": f"timeout > {timeout:.0f}s", "timed_out": True}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc(limit=6)}
    finally:
        # 软超时：不阻塞主流程等线程结束（纯 Python 计算，无原生阻塞调用）。
        ex.shutdown(wait=False, cancel_futures=True)


# ════════════════════════════════════════════════════════════════
# 引擎执行
# ════════════════════════════════════════════════════════════════

def _run_engine_a(df: pd.DataFrame, strategy: StrategyConfig,
                  portfolio: PortfolioConfig, timeout: float) -> dict:
    def _call() -> dict:
        return run_backtest(df, strategy, portfolio)

    out = _run_with_timeout(_call, timeout)
    if out.get("ok"):
        result = out["result"]
        if not isinstance(result, dict) or result.get("status") != "ok":
            out = {"ok": False,
                   "error": f"status={result.get('status') if isinstance(result, dict) else type(result).__name__}",
                   "result": result}
    return out


def _run_engine_b(df: pd.DataFrame, strategy: StrategyConfig,
                  portfolio: PortfolioConfig, symbol: str, timeout: float) -> dict:
    def _call():
        sig = generate_signals(df, strategy)
        prices = df.set_index("date")[["close"]].rename(columns={"close": symbol})
        signals = pd.DataFrame({symbol: sig.set_index("date")["signal"]})
        engine = MultiAssetBacktest(
            initial_capital=portfolio.initial_cash,
            commission=portfolio.commission_pct,
            slippage=portfolio.slippage_pct,
            stamp_tax_pct=portfolio.stamp_tax_pct,
            transfer_fee_pct=portfolio.transfer_fee_pct,
            min_commission=portfolio.min_commission,
        )
        return engine.run(prices, signals, max_position=portfolio.max_position_pct)

    return _run_with_timeout(_call, timeout)


# ════════════════════════════════════════════════════════════════
# 指标归一化
# ════════════════════════════════════════════════════════════════

def _metrics_a(result: dict) -> dict:
    m = result.get("metrics", result) if isinstance(result, dict) else {}
    return {
        "total_return_pct": _to_float(m.get("total_return_pct")),
        "max_drawdown_pct": _to_float(m.get("max_drawdown_pct")),
        "sharpe": _to_float(m.get("sharpe")),
        "turnover": _turnover_engine_a(result),
    }


def _metrics_b(result, symbol: str) -> dict:
    m = getattr(result, "metrics", {}) or {}
    total = _to_float(m.get("total_return"))
    dd = _to_float(m.get("max_drawdown"))
    return {
        "total_return_pct": total * 100.0 if total is not None else None,
        "max_drawdown_pct": dd * 100.0 if dd is not None else None,
        "sharpe": _to_float(m.get("sharpe_ratio")),
        "turnover": _turnover_engine_b(result, symbol),
    }


def _diagnostics_a(result: dict) -> dict:
    return {
        "trade_count": result.get("trade_count"),
        "win_rate_pct": result.get("win_rate_pct"),
        "final_equity": result.get("final_equity"),
    }


def _diagnostics_b(result) -> dict:
    m = getattr(result, "metrics", {}) or {}
    fc = _to_float(m.get("final_capital"))
    return {
        "n_trades": m.get("n_trades"),
        "final_capital": round(fc, 2) if fc is not None else None,
    }


# ════════════════════════════════════════════════════════════════
# 核心：双引擎对照
# ════════════════════════════════════════════════════════════════

def cross_check(df: pd.DataFrame,
                strategy: StrategyConfig | None = None,
                portfolio: PortfolioConfig | None = None,
                symbol: str | None = None,
                thresholds: dict | None = None,
                timeout: float = DEFAULT_TIMEOUT_S) -> dict:
    """同一策略/参数在两个引擎各跑一遍，返回对照报告 dict。

    返回结构：
      symbol, name, start, end, n_bars, strategy, thresholds,
      engine_a {name, ok, error, timed_out, metrics, diagnostics},
      engine_b {...},
      rows       四件套逐项（含偏差/阈值/判定）,
      over       超阈值项列表,
      failures   引擎失败记录（显式，不静默）,
      conclusion 审计结论文本。
    """
    strategy = strategy or StrategyConfig()
    portfolio = portfolio or PortfolioConfig()
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    df = df.copy()
    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if symbol is None:
        if "symbol" in df.columns and len(df):
            symbol = str(df["symbol"].iloc[0]).strip().zfill(6)
        else:
            symbol = "SYM"
    name = str(df["name"].iloc[0]) if "name" in df.columns and len(df) else symbol

    start = end = n_bars = None
    if "date" in df.columns and len(df):
        start = str(pd.Timestamp(df["date"].min()).date())
        end = str(pd.Timestamp(df["date"].max()).date())
        n_bars = int(len(df))

    # 两个引擎独立执行，互不短路。
    out_a = _run_engine_a(df, strategy, portfolio, timeout)
    out_b = _run_engine_b(df, strategy, portfolio, symbol, timeout)

    metrics_a = _metrics_a(out_a["result"]) if out_a.get("ok") else {
        k: None for k, _, _ in FOUR_SUITE}
    metrics_b = _metrics_b(out_b["result"], symbol) if out_b.get("ok") else {
        k: None for k, _, _ in FOUR_SUITE}

    rows: list[dict] = []
    over: list[dict] = []
    for key, label, unit in FOUR_SUITE:
        va = metrics_a.get(key)
        vb = metrics_b.get(key)
        thr = thresholds.get(key)
        if va is None or vb is None:
            deviation = None
            flag = "—"
        else:
            deviation = va - vb
            flag = "⚠️ 超阈值" if thr is not None and abs(deviation) > thr else "✅"
            if thr is not None and abs(deviation) > thr:
                over.append({"key": key, "label": label, "a": va, "b": vb,
                             "deviation": deviation, "threshold": thr})
        rows.append({"key": key, "label": label, "unit": unit,
                     "a": va, "b": vb, "deviation": deviation,
                     "threshold": thr, "flag": flag})

    failures = []
    for tag, out in (("A", out_a), ("B", out_b)):
        if not out.get("ok"):
            failures.append({"engine": tag,
                             "name": ENGINE_A if tag == "A" else ENGINE_B,
                             "error": out.get("error", "unknown"),
                             "timed_out": bool(out.get("timed_out"))})

    conclusion = _build_conclusion(rows, failures, over)

    return {
        "symbol": symbol,
        "name": name,
        "start": start,
        "end": end,
        "n_bars": n_bars,
        "strategy": asdict(strategy),
        "thresholds": thresholds,
        "engine_a": {
            "name": ENGINE_A,
            "ok": out_a.get("ok", False),
            "error": out_a.get("error"),
            "timed_out": bool(out_a.get("timed_out")),
            "metrics": metrics_a,
            "diagnostics": _diagnostics_a(out_a["result"]) if out_a.get("ok") else {},
        },
        "engine_b": {
            "name": ENGINE_B,
            "ok": out_b.get("ok", False),
            "error": out_b.get("error"),
            "timed_out": bool(out_b.get("timed_out")),
            "metrics": metrics_b,
            "diagnostics": _diagnostics_b(out_b["result"]) if out_b.get("ok") else {},
        },
        "rows": rows,
        "over": over,
        "failures": failures,
        "conclusion": conclusion,
    }


def _build_conclusion(rows: list[dict], failures: list[dict],
                      over: list[dict]) -> str:
    if failures:
        names = "、".join(f"{f['engine']}({f['name']}: {f['error']})" for f in failures)
        return f"❌ 存在引擎运行失败（未静默跳过）：{names}。四件套无法完整对照，需先处理失败引擎。"

    if not over:
        return "✅ 两引擎四件套全部落在阈值内，口径基本一致。"

    parts = []
    for o in over:
        unit = dict((r["key"], r["unit"]) for r in rows).get(o["key"], "")
        parts.append(f"{o['label']}差 {o['deviation']:+.2f}{unit}（阈值 ±{o['threshold']:.2f}{unit}）")
    return "⚠️ 存在口径偏差超阈值：" + "；".join(parts) + (
        "。建议审计两引擎执行口径（引擎A：次日开盘成交+止损止盈+整手；"
        "引擎B：收盘向量化权重、无止损），确认哪一者更贴近实盘。"
    )


# ════════════════════════════════════════════════════════════════
# Markdown 输出
# ════════════════════════════════════════════════════════════════

def _fmt(v, nd: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v:.{nd}f}"


def _fmt_dev(v) -> str:
    if v is None:
        return "—"
    return f"{v:+.2f}"


def render_markdown(report: dict) -> str:
    L: list[str] = []
    L.append("## 🔬 回测双引擎对照 · 偏差审计（W2.6）")
    L.append("")
    L.append(f"- 标的: `{report['symbol']}` ({report['name']})")
    L.append(f"- 区间: {report['start']} ~ {report['end']}（{report['n_bars']} 根K线）")
    strat = report["strategy"]
    L.append("- 策略: " + ", ".join(f"{k}={v}" for k, v in strat.items()))
    L.append(f"- 引擎A: `{report['engine_a']['name']}`（事件驱动 / 次日开盘成交 / 止损止盈）")
    L.append(f"- 引擎B: `{report['engine_b']['name']}`（收盘向量化权重矩阵）")
    L.append("")

    L.append("### 四件套（默认核心指标）")
    L.append("")
    L.append("| 指标 | 引擎A | 引擎B | 偏差 (A−B) | 阈值 (±) | 判定 |")
    L.append("|---|---|---|---|---|---|")
    for r in report["rows"]:
        unit = f" ({r['unit']})" if r["unit"] else ""
        thr = f"{r['threshold']:.2f}" if r["threshold"] is not None else "—"
        L.append(
            f"| {r['label']}{unit} | {_fmt(r['a'])} | {_fmt(r['b'])} "
            f"| {_fmt_dev(r['deviation'])} | {thr} | {r['flag']} |"
        )
    L.append("")

    L.append("### 偏差审计结论")
    L.append("")
    L.append(report["conclusion"])
    L.append("")

    L.append("### 引擎运行状态")
    L.append("")
    for key in ("engine_a", "engine_b"):
        e = report[key]
        tag = "A" if key == "engine_a" else "B"
        if e["ok"]:
            diag = ", ".join(f"{k}={v}" for k, v in e["diagnostics"].items())
            L.append(f"- 引擎{tag} `{e['name']}`: ✅ ok" + (f"（{diag}）" if diag else ""))
        else:
            mark = "⏱️" if e["timed_out"] else "❌"
            L.append(f"- 引擎{tag} `{e['name']}`: {mark} {e['error']}")
    return "\n".join(L)


# ════════════════════════════════════════════════════════════════
# 数据：合成冒烟 / 本地 kline
# ════════════════════════════════════════════════════════════════

def synthetic_df(n: int = 140, seed: int = 7) -> pd.DataFrame:
    """极小的确定性合成日线（用于冒烟，不要求真实行情）。"""
    dates = pd.bdate_range("2023-01-02", periods=n)
    rng = np.random.default_rng(seed)
    base = 20.0 + 0.02 * np.arange(n) + 2.0 * np.sin(np.arange(n) / 8.0)
    close = np.maximum(base + rng.normal(0.0, 0.3, n), 1.0)
    open_ = close * (1.0 + rng.normal(0.0, 0.004, n))
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    volume = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame({
        "date": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "symbol": "600000",
        "name": "浦发银行(合成)",
    })


def load_kline(symbol: str, kline_dir: Path | None = None) -> pd.DataFrame | None:
    code = str(symbol).strip().replace(".", "").zfill(6)
    kd = kline_dir or (ROOT / "data_warehouse" / "kline")
    p = kd / f"{code}.parquet"
    if not p.is_file():
        return None
    try:
        df = pd.read_parquet(p)
    except Exception:  # noqa: BLE001
        return None
    if df is None or df.empty or "close" not in df.columns:
        return None
    df = df.copy()
    df["symbol"] = code
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # 需要当日成交量/价（止损可成交判定用），缺失时补 NaN，引擎内部已兜底。
    for col in ("open", "high", "low", "volume"):
        if col not in df.columns:
            df[col] = np.nan
    return df


def _build_strategy(args) -> StrategyConfig:
    return StrategyConfig(
        fast_ma=args.fast, slow_ma=args.slow, trend_ma=args.trend,
        long_trend_ma=args.long, volume_ma=args.volume,
        stop_loss_pct=args.stop_loss, take_profit_pct=args.take_profit,
    )


def _parse_thresholds(args) -> dict:
    return {
        "total_return_pct": args.thr_return,
        "max_drawdown_pct": args.thr_drawdown,
        "sharpe": args.thr_sharpe,
        "turnover": args.thr_turnover,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="回测双引擎对照 + 偏差审计（W2.6）")
    ap.add_argument("--symbol", action="append", default=[],
                    help="标的代码（可多次）；缺省跑合成数据冒烟")
    ap.add_argument("--out", default=None, help="把 Markdown 写入文件（同时打印）")
    # 策略参数
    ap.add_argument("--fast", type=int, default=5)
    ap.add_argument("--slow", type=int, default=10)
    ap.add_argument("--trend", type=int, default=15)
    ap.add_argument("--long", type=int, default=20)
    ap.add_argument("--volume", type=int, default=5)
    ap.add_argument("--stop-loss", type=float, default=0.08)
    ap.add_argument("--take-profit", type=float, default=0.24)
    # 阈值覆盖
    ap.add_argument("--thr-return", type=float, default=2.0)
    ap.add_argument("--thr-drawdown", type=float, default=2.0)
    ap.add_argument("--thr-sharpe", type=float, default=0.3)
    ap.add_argument("--thr-turnover", type=float, default=2.0)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    strategy = _build_strategy(args)
    thresholds = _parse_thresholds(args)

    blocks: list[str] = []
    if args.symbol:
        for sym in args.symbol:
            df = load_kline(sym)
            if df is None or len(df) < 30:
                blocks.append(f"## ⚠️ 标的 {sym}: 本地 kline 缺失或不足 30 根，跳过。\n")
                continue
            report = cross_check(df, strategy=strategy,
                                 portfolio=PortfolioConfig(),
                                 symbol=sym, thresholds=thresholds,
                                 timeout=args.timeout)
            blocks.append(render_markdown(report))
    else:
        report = cross_check(synthetic_df(), strategy=strategy,
                             portfolio=PortfolioConfig(),
                             thresholds=thresholds, timeout=args.timeout)
        blocks.append(render_markdown(report))

    md = "\n\n".join(blocks)
    print(md)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md + "\n", encoding="utf-8")
        print(f"\n📁 已写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
