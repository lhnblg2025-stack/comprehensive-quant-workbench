#!/usr/bin/env python3
"""域 X：压力基线（性能 + 内存基线建立，不改被测模块逻辑）。

建立三个轻量基线：

- ``temp_replay_60f``: 复用 :mod:`quant_system.temperature_replay`，
  默认只扫描 kline 目录排序后的前 60 个文件并回放最近 250 个交易日，
  避免线上全量 kline I/O。
- ``kline_io``: 抽样读取 ``data_warehouse/kline`` 下 parquet 的
  ``date``/``close`` 两列，近似 realtime_snapshot 的盘中读取 I/O。
- ``backtest_2000bar``: 用合成 2000 根单标的 K 线跑
  :mod:`quant_system.backtest_engine`，得到事件驱动回测耗时。

首次运行用 ``--record`` 写入 ``generated/stress_baseline.json``；
之后用 ``--check`` 与基线比较。超过基线 ``2.5`` 倍只告警不失败，
退出码 ``1`` 表示退化，``0`` 表示正常。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import resource
import sys
import tempfile
import time
import tracemalloc
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

# 支持 `python quant_system/stress_baseline.py` 或 `python -m quant_system.stress_baseline`。
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from quant_system import backtest_engine as be  # noqa: E402
from quant_system import temperature_replay as tr  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_BASELINE_PATH = _REPO_ROOT / "generated" / "stress_baseline.json"
DEFAULT_TEMP_LIMIT = 250
DEFAULT_TEMP_MAX_FILES = 60
DEFAULT_KLINE_SAMPLE = 500
DEFAULT_BACKTEST_BARS = 2000
# 经验阈值：单次采样环境噪声下 2.5x 平衡误报/漏报；可用环境变量 STRESS_DEGRADE_THRESHOLD 覆盖
DEGRADE_THRESHOLD = 2.5

PERF_FIELDS = ("sec", "mem_mb")


# ---------------------------------------------------------------------------
# 计时 / 内存测量
# ---------------------------------------------------------------------------


def _measure_call(func: Callable[[], Any]) -> tuple[dict[str, float], Any]:
    """运行 ``func``，返回 ``{sec, mem_mb}`` 与函数结果。

    耗时采用 ``time.perf_counter``。``mem_mb`` 使用
    ``resource.getrusage(resource.RUSAGE_SELF).ru_maxrss``（峰值 RSS，
    KB → MB，含 C 层分配）；tracemalloc 结果保留为 ``py_mem_mb``，
    仅作为 Python 分配峰值的补充字段，回归比较只使用 ``mem_mb``。
    """
    tracemalloc.start()
    start = time.perf_counter()
    try:
        result = func()
    finally:
        elapsed = time.perf_counter() - start
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        ru_maxrss_kb = float(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        )

    metrics = {
        "sec": max(float(elapsed), 0.0),
        "mem_mb": max(ru_maxrss_kb / 1024.0, 0.0),
        "py_mem_mb": max(float(peak_bytes) / (1024.0 * 1024.0), 0.0),
    }
    return metrics, result


# ---------------------------------------------------------------------------
# 三类基线测量
# ---------------------------------------------------------------------------


def _make_limited_data_root(
    data_root: Path | str | None,
    max_files: int,
) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    """构造只含前 ``max_files`` 个 kline 文件的轻量回放数据根。

    ``data_warehouse/market`` 使用符号链接复用原数据目录，kline 目录则只
    链接 ``sorted(kline_dir.glob("*.parquet"))`` 的前 ``max_files`` 个文件，
    从而让 temperature_replay 只扫描轻量文件子集。
    """
    root = (
        Path(data_root) if data_root is not None else _REPO_ROOT
    ).resolve()
    warehouse = root / "data_warehouse"
    kline_dir = warehouse / "kline"

    max_files_int = int(max_files)
    if max_files_int <= 0 or not kline_dir.is_dir():
        return root, None

    files = sorted(kline_dir.glob("*.parquet"))[:max_files_int]
    tmp_dir = tempfile.TemporaryDirectory(prefix="stress_temperature_replay_")
    mirror_root = Path(tmp_dir.name)
    mirror_warehouse = mirror_root / "data_warehouse"
    mirror_kline_dir = mirror_warehouse / "kline"
    mirror_kline_dir.mkdir(parents=True, exist_ok=True)

    market_dir = warehouse / "market"
    if market_dir.exists():
        (mirror_warehouse / "market").symlink_to(market_dir, target_is_directory=True)

    for source_file in files:
        (mirror_kline_dir / source_file.name).symlink_to(source_file)

    return mirror_root, tmp_dir


def measure_temperature_replay(
    data_root: Path | str | None = None,
    limit: int = DEFAULT_TEMP_LIMIT,
    max_files: int = DEFAULT_TEMP_MAX_FILES,
    full: bool = False,
    out_path: Path | str | None = None,
) -> dict[str, float]:
    """测量温度回放耗时与峰值内存。

    ``full=False`` 时使用小规模 ``limit``（默认 250 个交易日）；仅显式
    传入 ``full=True`` 时才回放全量交易日。轻量基线只扫描 kline 目录
    sorted 后的前 ``max_files``（默认 60）个文件：轻量基线 = 60 文件 I/O
    + 回放计算，全量 kline I/O 由 ``measure_realtime_snapshot(kline_io)``
    项单独覆盖。回放结果写入临时 parquet，避免污染
    ``generated/temperature_history.parquet``。
    """
    effective_limit = None if full else int(limit)
    out = Path(out_path) if out_path is not None else Path(
        tempfile.gettempdir()
    ) / f"stress_temperature_replay_{uuid.uuid4().hex}.parquet"
    replay_root, tmp_dir = _make_limited_data_root(data_root, max_files)

    try:
        def _run() -> dict[str, int]:
            history = tr.replay(
                data_root=replay_root,
                limit=effective_limit,
                rebuild=True,
                out_path=out,
            )
            return {"rows": int(len(history))}

        metrics, _extra = _measure_call(_run)
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()
        if out_path is None:
            try:
                out.unlink(missing_ok=True)
            except OSError:
                logger.debug("清理临时回放文件失败: %s", out, exc_info=True)
    return metrics


def measure_realtime_snapshot(
    data_root: Path | str | None = None,
    sample_n: int = DEFAULT_KLINE_SAMPLE,
    kline_dir: Path | str | None = None,
) -> dict[str, float | int]:
    """测量 kline parquet 的抽样 I/O 基线。

    不调用真实 realtime_snapshot（会扫网络并写盘），而是读取 kline 文件
    的 ``date``/``close`` 两列，近似快照数据读取成本。``sample_n`` 为
    ``None`` 或 ``<=0`` 时读取全部文件。
    """
    if kline_dir is None:
        root = Path(data_root) if data_root is not None else _REPO_ROOT
        kline_dir = root / "data_warehouse" / "kline"
    kline_dir = Path(kline_dir)

    def _read_sample() -> int:
        files = sorted(kline_dir.glob("*.parquet"))
        sample_n_int = 0 if sample_n is None else int(sample_n)
        if sample_n_int > 0:
            files = random.Random(42).sample(
                files,
                min(sample_n_int, len(files)),
            )

        read_count = 0
        for path in files:
            try:
                pd.read_parquet(path, columns=["date", "close"])
                read_count += 1
            except Exception:
                logger.debug("读取 kline 失败: %s", path, exc_info=True)
        return read_count

    metrics, read_count = _measure_call(_read_sample)
    return {"sec": metrics["sec"], "files": int(read_count)}


class _BuyHoldBaselineStrategy(be.StrategyTemplate):
    """单标的买入持有的最小回测策略，只为产生真实事件驱动负载。"""

    def __init__(self) -> None:
        super().__init__()
        self._entered = False

    def on_bar(self, bar: be.BarData) -> None:
        if self.bg is None or self._entered:
            return
        self.buy(bar.close, self.bg.size, symbol=bar.symbol)
        self._entered = True


def _synthetic_ohlcv(bar_count: int) -> pd.DataFrame:
    """构造确定性的单标的 OHLCV 数据，避免真实数据依赖。"""
    dates = pd.date_range("2025-01-01", periods=bar_count, freq="D")
    close = 10.0 + np.linspace(0.0, 5.0, bar_count) + np.sin(
        np.arange(bar_count) / 20.0
    ) * 0.5
    close = pd.Series(close, dtype="float64")
    volume = np.full(bar_count, 100000.0)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": volume,
            "amount": volume * close.to_numpy(),
        }
    )


def measure_backtest(bar_count: int = DEFAULT_BACKTEST_BARS) -> dict[str, float | int]:
    """用 2000 bar 合成单标的回测测量事件驱动引擎耗时。"""
    bar_count = int(bar_count)

    def _run() -> dict[str, int]:
        engine = be.BacktestEngine()
        engine.set_capital(1_000_000.0)
        engine.set_commission(0.000085)
        engine.set_slippage(0.001, "percent")
        engine.set_fill_mode("next_open")
        engine.add_data("SYNTHETIC", _synthetic_ohlcv(bar_count))
        engine.add_strategy(_BuyHoldBaselineStrategy)
        result = engine.run()
        return {"bar_count": bar_count, "trades": len(result.trades)}

    metrics, extra = _measure_call(_run)
    return {"sec": metrics["sec"], "bar_count": int(extra["bar_count"])}


# ---------------------------------------------------------------------------
# 基线存取与比较
# ---------------------------------------------------------------------------


def _item_keys(
    *,
    full: bool,
    limit: int,
    sample_n: int = DEFAULT_KLINE_SAMPLE,
    max_files: int = DEFAULT_TEMP_MAX_FILES,
) -> tuple[str, str]:
    """生成当前测量项 key。默认配置与任务约定结构完全一致。"""
    temp_key = (
        f"temp_replay_full_{int(max_files)}f"
        if full
        else f"temp_replay_{int(max_files)}f"
    )
    kline_key = "kline_io"
    return temp_key, kline_key


def build_baseline(items: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": 1,
        "baseline_date": datetime.now().strftime("%Y-%m-%d"),
        "items": items,
    }


def write_baseline(baseline: dict[str, Any], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _normalize_legacy_item_keys(items: dict[str, Any]) -> dict[str, Any]:
    """把历史基线 key 映射到当前稳定 key，兼容旧基线文件读取。"""
    normalized = dict(items)

    legacy_kline_keys = [
        key for key in normalized if key.startswith("kline_io") and key != "kline_io"
    ]
    if legacy_kline_keys:
        if "kline_io" not in normalized:
            normalized["kline_io"] = normalized.pop(legacy_kline_keys[0])
        for key in legacy_kline_keys:
            normalized.pop(key, None)

    legacy_temp_keys = {
        "temp_replay_250d": "temp_replay_60f",
        "temp_replay_full": "temp_replay_full_60f",
    }
    for legacy_key, current_key in legacy_temp_keys.items():
        if current_key not in normalized and legacy_key in normalized:
            normalized[current_key] = normalized.pop(legacy_key)
        else:
            normalized.pop(legacy_key, None)

    return normalized


def load_baseline(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"基线文件不存在: {path}")
    with path.open("r", encoding="utf-8") as fh:
        baseline = json.load(fh)
    if not isinstance(baseline, dict) or "items" not in baseline:
        raise ValueError(f"基线结构无效: {path}")
    if not isinstance(baseline["items"], dict):
        raise ValueError(f"基线结构无效: {path}")
    baseline["items"] = _normalize_legacy_item_keys(baseline["items"])
    return baseline


def record_baseline(
    path: Path | str = DEFAULT_BASELINE_PATH,
    *,
    data_root: Path | str | None = None,
    limit: int = DEFAULT_TEMP_LIMIT,
    max_files: int = DEFAULT_TEMP_MAX_FILES,
    sample_n: int = DEFAULT_KLINE_SAMPLE,
    full: bool = False,
) -> dict[str, Any]:
    """执行三类小规模测量并写入基线 JSON。

    ``limit`` 为温度回放天数，``max_files`` 为回放扫描的 kline 文件数；
    轻量回放只扫描排序后的前 ``max_files`` 个 kline 文件。
    """
    temp_metric = measure_temperature_replay(
        data_root=data_root,
        limit=limit,
        max_files=max_files,
        full=full,
    )
    kline_metric = measure_realtime_snapshot(
        data_root=data_root,
        sample_n=sample_n,
    )
    backtest_metric = measure_backtest(DEFAULT_BACKTEST_BARS)

    temp_key, kline_key = _item_keys(
        full=full,
        limit=limit,
        sample_n=sample_n,
        max_files=max_files,
    )
    items = {
        temp_key: temp_metric,
        kline_key: kline_metric,
        "backtest_2000bar": backtest_metric,
    }
    baseline = build_baseline(items)
    write_baseline(baseline, path)
    return baseline


def check_baseline(
    current: dict[str, Any],
    baseline: dict[str, Any],
    threshold: float = DEGRADE_THRESHOLD,
) -> list[dict[str, Any]]:
    """对比 current 与 baseline，返回超过阈值倍数的告警列表。

    仅比较两边都存在的指标；耗时或内存超过 ``baseline * threshold`` 时
    产生告警。该函数不抛异常、不写文件。
    """
    warnings: list[dict[str, Any]] = []
    base_items = baseline.get("items", {})
    current_items = current.get("items", {})
    threshold = float(threshold)

    for item_name, base_metric in base_items.items():
        current_metric = current_items.get(item_name)
        if current_metric is None:
            warnings.append(
                {
                    "item": item_name,
                    "metric": "<missing>",
                    "baseline": None,
                    "current": None,
                    "ratio": None,
                    "message": "当前运行未产出该基线项",
                }
            )
            continue

        for metric_name in PERF_FIELDS:
            if metric_name not in base_metric or metric_name not in current_metric:
                continue
            base_value = float(base_metric[metric_name] or 0.0)
            current_value = float(current_metric[metric_name] or 0.0)
            if base_value <= 0:
                continue
            ratio = current_value / base_value
            if ratio > threshold:
                warnings.append(
                    {
                        "item": item_name,
                        "metric": metric_name,
                        "baseline": base_value,
                        "current": current_value,
                        "ratio": ratio,
                        "message": (
                            f"{item_name}.{metric_name} 为基线 {ratio:.2f} 倍，"
                            f"超过阈值 {threshold:.2f}"
                        ),
                    }
                )
    return warnings


def _degrade_threshold() -> float:
    """读取退化阈值，允许 ``STRESS_DEGRADE_THRESHOLD`` 环境变量覆盖。"""
    raw_value = os.environ.get("STRESS_DEGRADE_THRESHOLD")
    if raw_value is None:
        return DEGRADE_THRESHOLD
    try:
        value = float(raw_value)
    except ValueError:
        logger.warning("无效 STRESS_DEGRADE_THRESHOLD=%s，使用默认值", raw_value)
        return DEGRADE_THRESHOLD
    if value <= 0:
        logger.warning("非正 STRESS_DEGRADE_THRESHOLD=%s，使用默认值", raw_value)
        return DEGRADE_THRESHOLD
    return value


def _print_check_result(warnings: list[dict[str, Any]], threshold: float) -> None:
    if not warnings:
        print("stress baseline OK")
        return
    for warning in warnings:
        if warning["metric"] == "<missing>":
            print(f"WARN {warning['item']}: {warning['message']}")
        else:
            print(
                f"WARN {warning['item']}.{warning['metric']}: "
                f"current={warning['current']:.6f} baseline={warning['baseline']:.6f} "
                f"ratio={warning['ratio']:.2f} (threshold={threshold:.2f})"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="量化系统压力基线记录/检查")
    parser.add_argument(
        "--record",
        action="store_true",
        help="首次运行：写入 generated/stress_baseline.json",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="与已有基线比较，退化时退出码为 1",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="触发全量温度回放；默认关闭，避免测试过慢",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=DEFAULT_KLINE_SAMPLE,
        dest="sample_n",
        help=f"kline I/O 抽样文件数，默认 {DEFAULT_KLINE_SAMPLE}",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE_PATH,
        help="基线 JSON 路径",
    )
    args = parser.parse_args(argv)

    if args.record and args.check:
        parser.error("--record 与 --check 不能同时使用")
    if not args.record and not args.check:
        parser.error("请指定 --record（首次写基线）或 --check（回归检查）")

    if args.record:
        baseline = record_baseline(
            path=args.baseline,
            limit=DEFAULT_TEMP_LIMIT,
            sample_n=args.sample_n,
            full=args.full,
        )
        print(f"baseline written: {args.baseline}")
        print(json.dumps(baseline, ensure_ascii=False, indent=2))
        return 0

    try:
        baseline = load_baseline(args.baseline)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    current = record_baseline(
        path=Path(tempfile.gettempdir()) / f"stress_baseline_check_{uuid.uuid4().hex}.json",
        limit=DEFAULT_TEMP_LIMIT,
        sample_n=args.sample_n,
        full=args.full,
    )
    threshold = _degrade_threshold()
    warnings = check_baseline(current, baseline, threshold=threshold)
    _print_check_result(warnings, threshold)
    return 1 if warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
