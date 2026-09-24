"""walk_forward — 滚动样本外验证框架测试 (V12.3 补覆盖, 原 0%)。

小数据跑通 + 输出契约 + 参数校验。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_system.walk_forward import WalkForwardValidator  # noqa: E402


def _synthetic_prices(n: int = 400) -> pd.Series:
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2024-01-01", periods=n)
    rets = rng.normal(0.0005, 0.02, n)
    return pd.Series(100 * np.cumprod(1 + rets), index=idx, name="close")


def _ma_cross(prices: pd.Series, fast: int, slow: int) -> pd.Series:
    f = prices.rolling(fast).mean()
    s = prices.rolling(slow).mean()
    pos = pd.Series(0.0, index=prices.index)
    pos[f > s] = 1.0
    return pos.shift(1).fillna(0.0)


def test_init_validation():
    with pytest.raises(ValueError, match="过小"):
        WalkForwardValidator(train_window=5)
    with pytest.raises(ValueError, match="必须 > 0"):
        WalkForwardValidator(step=0)
    with pytest.raises(ValueError, match=">= 0"):
        WalkForwardValidator(embargo=-1)


def test_run_contract():
    """小数据跑通: 输出契约字段齐全, OOS 收益与净值长度合理。"""
    prices = _synthetic_prices(400)
    wfv = WalkForwardValidator(train_window=120, test_window=40, step=40, embargo=5)
    res = wfv.run(
        prices,
        strategy_fn=_ma_cross,
        param_grid={"fast": [5, 10], "slow": [20, 40]},
        score_func=lambda rets: float(rets.mean()) / (float(rets.std()) + 1e-9),
        verbose=False,
    )
    assert res["num_windows"] >= 2
    assert isinstance(res["oos_returns"], pd.Series) and len(res["oos_returns"]) > 0
    assert isinstance(res["oos_equity"], pd.Series)
    assert isinstance(res["aggregate_sharpe"], float)
    assert isinstance(res["total_return_pct"], float)
    assert isinstance(res["best_params_per_window"], list)
    assert res["param_grid_size"] == 4
    assert isinstance(res["config"], dict)


def test_run_insufficient_data():
    wfv = WalkForwardValidator(train_window=120, test_window=40)
    with pytest.raises(ValueError, match="不足以支撑"):
        wfv.run(_synthetic_prices(100), strategy_fn=_ma_cross,
                param_grid={"fast": [5]}, verbose=False)


def test_empty_param_grid_raises():
    """无效参数网格(空列表值) → ValueError(sklearn 或框架自身均可)。"""
    wfv = WalkForwardValidator(train_window=120, test_window=40)
    with pytest.raises(ValueError):
        wfv.run(_synthetic_prices(300), strategy_fn=_ma_cross,
                param_grid={"fast": []}, verbose=False)


def test_non_series_prices_raises():
    wfv = WalkForwardValidator(train_window=120, test_window=40)
    with pytest.raises(TypeError, match="pd.Series"):
        wfv.run([1, 2, 3], strategy_fn=_ma_cross, param_grid={"fast": [5]}, verbose=False)


def test_overlap_warns():
    with pytest.warns(RuntimeWarning, match="重叠"):
        WalkForwardValidator(train_window=120, test_window=40, step=20)
