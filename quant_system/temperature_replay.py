#!/usr/bin/env python3
"""温度→仓位映射网格校准 — 历史回放器。

用 data_warehouse 的 kline/index_daily/margin 数据，为每个历史交易日构造
``market_temperature.classify_cycle`` 所需的 data dict，并计算下一交易日全 A
等权收益，落盘为 generated/temperature_history.parquet。

口径说明：
- pct_above_ma300 使用“上证指数相对自身 MA300 的偏离%”，与线上 fetch_market_data 同口径
- advance_pct 使用 up_n/(up_n+down_n)*100，排除平盘/停牌，对齐线上 meta.json 上涨占比%
- ret_next 使用不复权日线 close.pct_change() 的等权均值，作为日收益近似
- 北向 2024-08 后停更，回放统一 north_net_yi=0
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# 允许 `python quant_system/temperature_replay.py` 或 `python -m ...` 两种运行方式
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in os.environ.get("PYTHONPATH", "").split(os.pathsep):
    import sys
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from quant_system.market_temperature import classify_cycle  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = _REPO_ROOT
DEFAULT_OUT_PATH = _REPO_ROOT / "generated" / "temperature_history.parquet"

REQUIRED_COLUMNS = [
    "date",
    "cycle_segment",
    "temperature",
    "risk_posture",
    "ret_next",
    "margin_available",
]


def _warehouse(data_root: Path | str | None) -> Path:
    root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
    # 支持 MACRO_DATA_DIR 指向仓库根目录（该目录下仍有 data_warehouse/）。
    env_root = os.environ.get("MACRO_DATA_DIR")
    if data_root is None and env_root:
        root = Path(env_root)
    return root / "data_warehouse"


def _num(value: Any) -> float:
    """把可能为 NaN 的标量安全转成 float，缺失按 0 处理。"""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    return x if np.isfinite(x) else 0.0


def load_market_daily(
    kline_dir: Path | str,
    index_df: pd.DataFrame,
    limit: int | None = None,
) -> pd.DataFrame:
    """从全市场 kline 聚合出日频市场统计。

    返回以交易日为索引的 DataFrame：
    ret_mean, advance_pct, ret_n

    advance_pct = up_n / (up_n + down_n) * 100，排除平盘/停牌，与线上
    meta.json 的“上涨占比%”口径一致。
    """
    kline_dir = Path(kline_dir)
    idx = index_df.copy()
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date").reset_index(drop=True)
    if limit is not None:
        idx = idx.tail(max(1, int(limit)))
    date_index = pd.DatetimeIndex(idx["date"])
    n_days = len(date_index)

    # 每列：ret_sum, ret_n, up_n, down_n
    acc = np.zeros((n_days, 4), dtype=np.float64)

    files = sorted(kline_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"kline 目录无 parquet 文件: {kline_dir}")

    for f in files:
        try:
            df = pd.read_parquet(f, columns=["date", "close"])
        except Exception:
            logger.debug("读取 kline 失败: %s", f, exc_info=True)
            continue
        if df.empty:
            continue

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        if df.empty:
            continue

        close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=np.float64)

        ret = np.full_like(close, np.nan, dtype=np.float64)
        if len(close) > 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                ret[1:] = close[1:] / close[:-1] - 1.0

        ret_valid = ~np.isnan(ret)
        up = ret > 0
        down = ret < 0

        pos = date_index.get_indexer(df["date"])
        valid = pos >= 0
        if not valid.any():
            continue
        pos = pos[valid]

        np.add.at(acc[:, 0], pos, np.nan_to_num(ret[valid], nan=0.0))
        np.add.at(acc[:, 1], pos, ret_valid[valid].astype(np.float64))
        np.add.at(acc[:, 2], pos, up[valid].astype(np.float64))
        np.add.at(acc[:, 3], pos, down[valid].astype(np.float64))

    daily = pd.DataFrame(
        acc,
        index=date_index,
        columns=["ret_sum", "ret_n", "up_n", "down_n"],
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        daily["ret_mean"] = daily["ret_sum"] / daily["ret_n"]
        daily["advance_pct"] = (
            daily["up_n"] / (daily["up_n"] + daily["down_n"]) * 100.0
        )

    daily = daily[["ret_mean", "advance_pct", "ret_n"]]
    return daily


def load_margin(
    sh_path: Path | str,
    sz_path: Path | str,
) -> pd.DataFrame:
    """读取沪深融资余额并计算日变化(亿元)。

    返回以 date 为索引的 DataFrame，列含 margin_change_yi / margin_available。
    """
    sh_path = Path(sh_path)
    sz_path = Path(sz_path)
    parts: list[pd.DataFrame] = []

    for path in (sh_path, sz_path):
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
        except Exception:
            logger.debug("读取融资余额失败: %s", path, exc_info=True)
            continue
        if df.empty:
            continue

        date_col = "日期" if "日期" in df.columns else "date"
        if date_col not in df.columns:
            continue
        balance_col = next(
            (c for c in ("融资余额", "margin_balance", "fin_balance") if c in df.columns),
            None,
        )
        if balance_col is None:
            continue

        tmp = pd.DataFrame(
            {
                "date": pd.to_datetime(df[date_col]),
                "balance": pd.to_numeric(df[balance_col], errors="coerce"),
            }
        )
        parts.append(tmp)

    if not parts:
        return pd.DataFrame(
            index=pd.DatetimeIndex([], name="date"),
            columns=["margin_change_yi", "margin_available"],
        )

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.dropna(subset=["date", "balance"])
    if combined.empty:
        return pd.DataFrame(
            index=pd.DatetimeIndex([], name="date"),
            columns=["margin_change_yi", "margin_available"],
        )

    balance = combined.groupby("date")["balance"].sum().sort_index()
    # 数据文件为“元”，除以 1e8 转成亿元。
    change_yi = balance.diff() / 1e8
    # margin_available 必须按“当日两融行数 > 0”判断；groupby.sum() 对全 NaN 组
    # 会返回 0，导致 balance.notna() 恒为 True。
    balance_count = combined.groupby("date")["balance"].count().sort_index()
    margin = pd.DataFrame(
        {
            "margin_change_yi": change_yi,
            "margin_available": balance_count > 0,
        }
    )
    margin.index.name = "date"
    return margin


def build_temperature_history(
    daily: pd.DataFrame,
    index_df: pd.DataFrame,
    margin_df: pd.DataFrame,
) -> pd.DataFrame:
    """基于聚合好的日频统计逐日调用 classify_cycle，生成历史温度表。"""
    daily = daily.copy()
    daily = daily[daily["ret_n"] > 0].copy()
    if daily.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    sh = index_df.copy()
    sh["date"] = pd.to_datetime(sh["date"])
    sh = sh.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    sh = sh.set_index("date")

    close = pd.to_numeric(sh["close"], errors="coerce")
    # 与线上 fetch_market_data 同口径：上证指数相对自身 MA300 的偏离%。
    # 滚动窗口按历史升序 close 计算，窗口内不含未来数据。
    ma300 = close.rolling(300, min_periods=300).mean()
    sh["pct_above_ma300"] = (close / ma300 - 1.0) * 100.0
    sh["sh_pct"] = close.pct_change(fill_method=None) * 100.0

    # 线上 ma60_direction = ma60 vs 60 日前的 MA60（等价于最近 60 日与前 60 日均值比较）。
    # 前 60 行 prev60 为 NaN，属预热期；按任务约定统一标为 "up"。
    ma60 = close.rolling(60, min_periods=60).mean()
    prev60 = ma60.shift(60)
    sh["ma60_direction"] = pd.Series(
        np.where(prev60.notna() & (ma60 <= prev60), "down", "up"),
        index=sh.index,
    )
    sh["roc_60"] = (close / close.shift(60) - 1.0) * 100.0

    # 历史无沪+深成交额，用上证 index_daily.volume 近似
    # （线上为沪+深指数成交额，total/avg 比值口径一致）。
    volume = pd.to_numeric(
        sh["volume"] if "volume" in sh.columns else np.nan,
        errors="coerce",
    )
    sh["total_amount_yi"] = volume

    merged = daily.join(
        sh[["sh_pct", "pct_above_ma300", "ma60_direction", "roc_60", "total_amount_yi"]],
        how="left",
    )

    # 标签列（shift(-1) 下一交易日收益），仅用于校准目标，禁止作为特征输入任何模型。
    merged["ret_next"] = merged["ret_mean"].shift(-1)
    merged["total_amount_yi"] = merged["total_amount_yi"].fillna(0.0)
    merged["avg_amount_yi_60"] = (
        merged["total_amount_yi"].rolling(60, min_periods=1).mean()
    )
    merged["avg_amount_yi_20"] = (
        merged["total_amount_yi"].rolling(20, min_periods=1).mean()
    )

    if not margin_df.empty:
        merged = merged.join(
            margin_df[["margin_change_yi", "margin_available"]],
            how="left",
        )
    else:
        merged["margin_change_yi"] = np.nan
        merged["margin_available"] = False

    merged["margin_change_yi"] = merged["margin_change_yi"].fillna(0.0)
    merged["margin_available"] = np.where(
        merged["margin_available"].isna(), False, merged["margin_available"]
    ).astype(bool)

    rows: list[dict[str, Any]] = []
    for ts, row in merged.iterrows():
        data = {
            "timestamp": ts.strftime("%Y-%m-%d"),
            "pct_above_ma300": _num(row.get("pct_above_ma300")),
            "advance_pct": _num(row.get("advance_pct")),
            "advance_pct_estimate": False,
            "sh_pct": _num(row.get("sh_pct")),
            "ma60_direction": (
                row.get("ma60_direction")
                if isinstance(row.get("ma60_direction"), str)
                else "up"
            ),
            "roc_60": _num(row.get("roc_60")),
            "total_amount_yi": _num(row.get("total_amount_yi")),
            "avg_amount_yi_60": _num(row.get("avg_amount_yi_60")),
            "avg_amount_yi_20": _num(row.get("avg_amount_yi_20")),
            "margin_change_yi": _num(row.get("margin_change_yi")),
            "north_net_yi": 0.0,
        }
        cycle = classify_cycle(data)
        rows.append(
            {
                "date": ts,
                "cycle_segment": cycle.get("cycle_segment", ""),
                "temperature": cycle.get("temperature", 50),
                "risk_posture": cycle.get("risk_posture", ""),
                "ret_next": row.get("ret_next"),
                "margin_available": bool(row.get("margin_available", False)),
            }
        )

    out = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    out = out.sort_values("date").reset_index(drop=True)
    return out


def _is_fresh(
    out_path: Path,
    kline_dir: Path,
    index_path: Path,
    margin_paths: list[Path],
) -> bool:
    """简单新鲜度检查：缓存 mtime 不得早于任一数据源。"""
    if not out_path.exists():
        return False
    sources = [p for p in [index_path, *margin_paths] if p.exists()]
    if not sources:
        return False
    latest = max(p.stat().st_mtime for p in sources)
    kline_files = list(kline_dir.glob("*.parquet"))
    if kline_files:
        latest = max(latest, max(p.stat().st_mtime for p in kline_files))
    return out_path.stat().st_mtime >= latest


def replay(
    data_root: Path | str | None = None,
    limit: int | None = None,
    rebuild: bool = False,
    out_path: Path | str | None = None,
) -> pd.DataFrame:
    """回放历史温度，默认落盘 generated/temperature_history.parquet。

    参数 data_root 用于测试/冒烟时指向包含 data_warehouse/ 的临时根目录。
    """
    root = Path(data_root) if data_root is not None else Path(
        os.environ.get("MACRO_DATA_DIR", str(DEFAULT_DATA_ROOT))
    )
    warehouse = _warehouse(root)
    kline_dir = warehouse / "kline"
    index_path = warehouse / "market" / "index_daily.parquet"
    margin_paths = [
        warehouse / "market" / "market_margin_sh.parquet",
        warehouse / "market" / "market_margin_sz.parquet",
    ]
    out = Path(out_path) if out_path is not None else (
        root / "generated" / "temperature_history.parquet"
    )

    if (
        not rebuild
        and out.exists()
        and _is_fresh(out, kline_dir, index_path, margin_paths)
    ):
        logger.info("命中缓存: %s", out)
        return pd.read_parquet(out)

    if not index_path.exists():
        raise FileNotFoundError(f"缺少指数日线: {index_path}")

    index_df = pd.read_parquet(index_path)
    daily = load_market_daily(kline_dir, index_df, limit=limit)
    margin_df = load_margin(*margin_paths)
    history = build_temperature_history(daily, index_df, margin_df)

    out.parent.mkdir(parents=True, exist_ok=True)
    history.to_parquet(out, index=False)
    logger.info("历史温度已落盘: %s (rows=%d)", out, len(history))
    return history


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="回放历史市场温度")
    parser.add_argument("--rebuild", action="store_true", help="强制重算并覆盖缓存")
    parser.add_argument("--limit", type=int, default=None, help="仅回放最近 N 个交易日（测试用）")
    parser.add_argument("--data-root", type=Path, default=None, help="数据根目录，默认仓库根")
    parser.add_argument("--out", type=Path, default=None, help="输出 parquet 路径")
    args = parser.parse_args(argv)

    df = replay(
        data_root=args.data_root,
        limit=args.limit,
        rebuild=args.rebuild,
        out_path=args.out,
    )
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
