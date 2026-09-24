#!/usr/bin/env python3
"""温度→仓位映射网格校准器。

读 generated/temperature_history.parquet，枚举 7 个周期阶段各 3 档仓位系数
（3^7=2187 组合），按 7:3 时间切分训练/OOS，以年化夏普选优。

选优规则：
- 先取训练夏普最高的组合；
- 在这些组合中，只接受 OOS 夏普 >= 原手拍档位 OOS 基线的组合；
- 若全部低于基线，则回退原手拍档位并在 note 中说明。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

PHASES = ["恐慌", "底部", "早期", "成熟", "后期", "熊市", "其他"]

# 原手拍档位对应的“中点系数”。
BASE_COEFFICIENTS: dict[str, float] = {
    "恐慌": 0.4,
    "底部": 0.25,
    "早期": 0.7,
    "成熟": 0.5,
    "后期": 0.25,
    "熊市": 0.075,
    "其他": 0.3,
}

# 任务规定的 3 档候选系数。
CANDIDATES: dict[str, list[float]] = {
    "恐慌": [0.2, 0.4, 0.6],
    "底部": [0.15, 0.25, 0.4],
    "早期": [0.5, 0.7, 0.9],
    "成熟": [0.3, 0.5, 0.7],
    "后期": [0.15, 0.25, 0.4],
    "熊市": [0.0, 0.075, 0.15],
    "其他": [0.2, 0.3, 0.5],
}

DEFAULT_HISTORY = _REPO_ROOT / "generated" / "temperature_history.parquet"
DEFAULT_OUTPUT = _REPO_ROOT / "config" / "position_map.json"


def phase_key(cycle_segment: str) -> str:
    """把 classify_cycle 的 cycle_segment 归一到 7 个校准阶段。"""
    segment = str(cycle_segment or "")
    if "恐慌" in segment:
        return "恐慌"
    if "底部" in segment:
        return "底部"
    if "早期" in segment:
        return "早期"
    if "成熟" in segment:
        return "成熟"
    if "后期" in segment or "狂热" in segment:
        return "后期"
    if "熊市" in segment:
        return "熊市"
    return "其他"


def _sharpe_vectorized(returns: np.ndarray, periods_per_year: int = 252) -> np.ndarray:
    """按列计算年化夏普。输入 (n_days, n_combos)。"""
    if returns.size == 0:
        return np.zeros(0, dtype=np.float64)
    mean = np.mean(returns, axis=0)
    std = np.std(returns, axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = mean / std * np.sqrt(periods_per_year)
    sharpe = np.where(np.isfinite(sharpe) & (std > 1e-12), sharpe, 0.0)
    return sharpe


def _single_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    if returns.size < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if not np.isfinite(std) or std <= 1e-12:
        return 0.0
    sharpe = float(np.mean(returns) / std * np.sqrt(periods_per_year))
    return sharpe if np.isfinite(sharpe) else 0.0


def calibrate(
    history: pd.DataFrame,
    split_frac: float = 0.7,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """执行网格校准，返回可落盘的 position_map 结构。"""
    if history is None or history.empty:
        raise ValueError("temperature_history 为空，无法校准")

    df = history.copy()
    required = {"cycle_segment", "ret_next"}
    if not required.issubset(df.columns):
        raise ValueError(f"缺少必要列: {required - set(df.columns)}")

    df = df.dropna(subset=["cycle_segment", "ret_next"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("无有效 ret_next，无法校准")

    # ret_next 是回放器用 shift(-1) 生成的下一交易日收益，仅作为校准目标标签；
    # 禁止将其作为特征输入任何模型。
    ret = df["ret_next"].astype(float).to_numpy(dtype=np.float64)
    phase_idx = df["cycle_segment"].map(phase_key).map(
        {k: i for i, k in enumerate(PHASES)}
    ).to_numpy(dtype=np.int64)
    assert phase_idx.min() >= 0 and phase_idx.max() < len(PHASES)

    n = len(df)
    split = max(1, min(n - 1, int(n * split_frac)))
    train_slice = slice(0, split)
    oos_slice = slice(split, n)

    # 每行：该日 ret_next 乘以对应阶段的系数。未出现的阶段列保持 0。
    design = np.zeros((n, len(PHASES)), dtype=np.float64)
    design[np.arange(n), phase_idx] = ret
    design_train = design[train_slice, :]
    design_oos = design[oos_slice, :]

    # 列顺序必须与 PHASES 顺序严格对应（[CANDIDATES[p] for p in PHASES]），禁止乱序。
    candidate_arrays = [CANDIDATES[p] for p in PHASES]
    mesh = np.stack(
        np.meshgrid(*candidate_arrays, indexing="ij"),
        axis=-1,
    ).reshape(-1, len(PHASES))

    train_returns = design_train @ mesh.T
    oos_returns = design_oos @ mesh.T
    train_sharpes = _sharpe_vectorized(train_returns, periods_per_year)
    oos_sharpes = _sharpe_vectorized(oos_returns, periods_per_year)

    base = np.array([BASE_COEFFICIENTS[p] for p in PHASES], dtype=np.float64)
    base_train_returns = design_train @ base
    base_oos_returns = design_oos @ base
    base_train_sharpe = _single_sharpe(base_train_returns, periods_per_year)
    base_oos_sharpe = _single_sharpe(base_oos_returns, periods_per_year)

    max_train = float(np.max(train_sharpes))
    top_idx = np.where(np.isclose(train_sharpes, max_train))[0]
    eligible = top_idx[oos_sharpes[top_idx] >= base_oos_sharpe - 1e-12]

    best_idx = -1
    if eligible.size > 0:
        best_idx = int(eligible[np.argmax(oos_sharpes[eligible])])
        selected = mesh[best_idx]
        calibrated = True
        note = "训练夏普最高组合中，OOS 夏普不低于原档位基线，已采用校准档位"
    else:
        selected = base
        calibrated = False
        note = "训练夏普最高组合的 OOS 夏普低于原档位基线，回退原手拍档位"

    selected_sharpe_train = (
        float(train_sharpes[best_idx]) if calibrated else base_train_sharpe
    )
    selected_sharpe_oos = float(oos_sharpes[best_idx]) if calibrated else base_oos_sharpe

    phases = {phase: float(selected[i]) for i, phase in enumerate(PHASES)}

    _print_table(selected, calibrated, base_oos_sharpe, base_train_sharpe)

    return {
        "phases": phases,
        "baseline_sharpe": float(base_oos_sharpe),
        "oos_sharpe": selected_sharpe_oos,
        "train_sharpe": selected_sharpe_train,
        "note": note,
    }


def _print_table(
    selected: np.ndarray,
    calibrated: bool,
    base_oos_sharpe: float,
    base_train_sharpe: float,
) -> None:
    print("阶段     原档位    候选档位                  选中档位")
    for i, phase in enumerate(PHASES):
        cands = "{" + ",".join(f"{c:g}" for c in CANDIDATES[phase]) + "}"
        print(
            f"{phase:<8} {BASE_COEFFICIENTS[phase]:>6.3f}  {cands:<24} {selected[i]:>8.3f}"
        )
    print(f"原档位 OOS 夏普: {base_oos_sharpe:.4f}  (训练夏普: {base_train_sharpe:.4f})")
    print(f"选中结果: {'校准档位' if calibrated else '回退原档位'}")


def write_json(result: dict[str, Any], output: Path | str) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="校准温度→仓位映射")
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    if not args.history.exists():
        print(f"[失败] 历史温度不存在: {args.history}", file=sys.stderr)
        return 1

    history = pd.read_parquet(args.history)
    result = calibrate(history)
    write_json(result, args.output)
    print(f"position_map 已生成: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
