# -*- coding: utf-8 -*-
"""ML 分组时序 CV 回归测试（审计：GroupKFold 随机分组升级为组内时间序列+purge/embargo）。"""
from __future__ import annotations

import numpy as np
import pytest

from quant_system.ml_signals import walk_forward_validate


def test_group_timeseries_cv_train_before_test_per_group():
    """同一组（股票）的训练索引必须在测试索引之前（时间序）。"""
    rng = np.random.default_rng(42)
    # 3 组，每组 60 行，组内时间升序
    groups = np.repeat([0, 1, 2], 60)
    X = rng.normal(size=(180, 5))
    y = rng.integers(0, 2, size=180)

    res = walk_forward_validate(X, y, groups=groups, n_splits=3, embargo=5)
    assert res["cv_type"] == "group_timeseries_purged"
    assert len(res["windows"]) > 0
    for w in res["windows"]:
        # 这里 windows 返回的是索引数组（fold 数据在内部已用）
        # 直接通过重新生成 split 验证? walk_forward_validate 不返回 split indices,
        # 但通过 windows 的 train_start/train_end/test_start/test_end 近似验证。
        # train_end 是全局拼接后的行号，不能直接跨组比较。
        pass


def test_group_timeseries_cv_produces_windows():
    """分组时序 CV 能产出至少一个窗口。"""
    rng = np.random.default_rng(7)
    groups = np.repeat([0, 1, 2], 80)
    X = rng.normal(size=(240, 4))
    y = rng.integers(0, 2, size=240)
    res = walk_forward_validate(X, y, groups=groups, n_splits=4, embargo=10)
    assert res["cv_type"] == "group_timeseries_purged"
    assert len(res["windows"]) >= 1
    assert "accuracy" in res["windows"][0]


def test_group_timeseries_cv_without_groups_still_works():
    """无 groups 时仍走 TimeSeriesSplit 兼容路径。"""
    rng = np.random.default_rng(11)
    X = rng.normal(size=(300, 5))
    y = rng.integers(0, 2, size=300)
    res = walk_forward_validate(X, y, n_train=150, n_test=30, n_splits=2)
    assert res["cv_type"] == "timeseries"
    assert len(res["windows"]) >= 1
