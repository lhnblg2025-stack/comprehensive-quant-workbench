#!/usr/bin/env python3
"""P1-2 因子健康监控测试（quant_system/ic_factors/monitor.py）。

覆盖审计_因子层.md P1-2 的核心承诺：
  1. 滚动 IC / t 值公式与手算一致。
  2. 因子收益波动率 = 多空组合收益滚动 std 与手算一致。
  3. 多空换手率公式正确（新增+移出）/腿规模，与手算一致。
  4. 同类因子相关均值正确反映冗余/拥挤。
  5. monitor_factor / monitor_panel 产出 schema 正确、缺失输入时诚实 NaN。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_system.ic_factors.monitor import (
    factor_correlation_mean,
    factor_return_volatility,
    long_short_turnover,
    monitor_factor,
    monitor_panel,
    rolling_ic,
    rolling_ic_tvalue,
)


def _ic_series(n=40, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(0.05, 0.1, n), index=dates)


class TestRollingIC:
    def test_rolling_ic_mean_matches_hand(self):
        ic = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
        out = rolling_ic(ic, window=3, min_periods=3)
        # window=3, min_periods=3: 暖窗前 2 个 NaN，之后均值
        assert pd.isna(out.iloc[0]) and pd.isna(out.iloc[1])
        assert out.iloc[2] == pytest.approx(0.2)
        assert out.iloc[3] == pytest.approx(0.3)
        assert out.iloc[4] == pytest.approx(0.4)

    def test_rolling_ic_tvalue_matches_formula(self):
        ic = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
        t = rolling_ic_tvalue(ic, window=5)  # 全窗口
        mean = ic.mean()
        std = ic.std(ddof=1)
        expected = mean / (std / np.sqrt(len(ic)))
        assert t.iloc[-1] == pytest.approx(expected)

    def test_tvalue_increasing_with_signal_strength(self):
        dates = pd.date_range("2024-01-01", periods=40, freq="B")
        rng = np.random.default_rng(1)
        noise = rng.normal(0, 0.2, 40)
        weak = pd.Series(0.01 + noise, index=dates)
        strong = pd.Series(0.20 + noise, index=dates)  # 同噪声，均值偏移更大
        assert abs(rolling_ic_tvalue(strong).dropna().iloc[-1]) > \
            abs(rolling_ic_tvalue(weak).dropna().iloc[-1])

    def test_rolling_ic_empty(self):
        assert rolling_ic(pd.Series(dtype=float)).empty
        assert rolling_ic_tvalue(pd.Series(dtype=float)).empty


class TestFactorReturnVolatility:
    def test_vol_matches_hand(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        stocks = [f"S{i}" for i in range(30)]
        # 构造固定横截面：因子值=行号/gap，收益与因子正相关 → 多空收益稳定
        f_rows = np.zeros((10, 30))
        r_rows = np.zeros((10, 30))
        for d in range(10):
            f_rows[d] = np.arange(30)
            # 收益 = 0.5*(z值) + 噪声 → 多空收益≈0.5*E[z^2]，波动小
            z = np.arange(30)
            z = (z - z.mean()) / z.std()
            r_rows[d] = 0.5 * z + np.random.default_rng(d).normal(0, 0.01, 30)
        fw = pd.DataFrame(f_rows, index=dates, columns=stocks, dtype=float)
        rw = pd.DataFrame(r_rows, index=dates, columns=stocks, dtype=float)

        vol = factor_return_volatility(fw, rw, window=5)
        # 手工重算最后一个非 NaN 值：近5日多空收益的 std
        ls = []
        for d in dates[-5:]:
            f = fw.loc[d].astype(float)
            r = rw.loc[d].astype(float)
            valid = f.notna() & r.notna()
            fv = f[valid]; rv = r[valid]
            z = (fv - fv.mean()); z = z / z.std()
            ls.append((z * rv).mean())
        expected = pd.Series(ls).std(ddof=1)
        assert vol.dropna().iloc[-1] == pytest.approx(expected, rel=1e-6)

    def test_vol_missing_data_returns_empty(self):
        assert factor_return_volatility(pd.DataFrame(), pd.DataFrame()).empty


class TestLongShortTurnover:
    def test_turnover_formula(self):
        # 20 只股票、值全不同（0..19，S19 最大）。top 20% = {S16..S19}，bottom 20% = {S00..S03}。
        # 第2日把 top 换成 {S15,S17,S18,S19}（移出 S16、移入 S15），bottom 不变。
        dates = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
        stocks = [f"S{i}" for i in range(20)]
        day1 = {f"S{i}": float(i) for i in range(20)}
        day2 = dict(day1)
        day2["S16"] = 10.0  # 把 S16 从 top 撤下（原 16 提到 lower，与 mid 同层）
        day2["S15"] = 30.0  # 把 S15 提进 top
        day2 = {k: (float(v) + 100 if k in ("S15", "S17", "S18", "S19") else v)
                for k, v in day2.items()}  # top 重新确立为 S15,S17,S18,S19
        fw = pd.DataFrame([day1, day2], index=dates, columns=stocks, dtype=float)
        to = long_short_turnover(fw, long_q=0.2, short_q=0.2)
        assert len(to) == 1
        d2 = to.iloc[0]
        # long: 前日 {S16,S17,S18,S19}, 今日 {S15,S17,S18,S19} -> 对称差 {S15,S16} (2), 腿规模 4 -> 0.5
        # short: 两日均 {S00,S01,S02,S03} -> 0
        expected = (0.5 + 0.0) / 2.0
        assert d2 == pytest.approx(expected)

    def test_turnover_empty(self):
        assert long_short_turnover(pd.DataFrame()).empty


class TestPeerCorrelation:
    def test_returns_high_for_correlated_peers(self):
        rng = np.random.default_rng(5)
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        stocks = [f"S{i}" for i in range(50)]
        base = pd.DataFrame(rng.normal(0, 1, (10, 50)), index=dates, columns=stocks)
        corr_peer = base + rng.normal(0, 0.001, base.shape)  # 高度相关
        uncorr = pd.DataFrame(rng.normal(0, 1, (10, 50)), index=dates, columns=stocks)
        panel_dict = {"a": base, "b": corr_peer, "c": uncorr}
        hi = factor_correlation_mean(base, ["b"], panel_dict)
        lo = factor_correlation_mean(base, ["c"], panel_dict)
        assert hi > 0.9
        assert lo < 0.3

    def test_no_peers_returns_nan(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        fw = pd.DataFrame(np.random.rand(5, 20), index=dates)
        assert np.isnan(factor_correlation_mean(fw, [], {}))


class TestMonitor:
    def test_monitor_factor_schema_and_alert(self):
        ic = pd.Series(np.linspace(-0.05, -0.08, 30),
                       index=pd.date_range("2024-01-01", periods=30, freq="B"))
        rng = np.random.default_rng(2)
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        stocks = [f"S{i}" for i in range(30)]
        fw = pd.DataFrame(rng.normal(0, 1, (30, 30)), index=dates, columns=stocks)
        rw = fw.copy()  # 因子与"收益"无关 → 多空收益≈0
        row = monitor_factor(ic, fw, rw, "test_f", category="reversal")
        assert set(["factor", "category", "rolling_ic", "ic_tvalue",
                    "factor_ret_vol", "peer_corr_mean", "ls_turnover",
                    "alerts", "status"]).issubset(row.keys())
        # 负 IC 稳定 → t 值负且 |t| 大，不应判 ic_tvalue 低；但 peer(无/低)相关/换手可能触发
        assert pd.notna(row["ic_tvalue"])

    def test_monitor_missing_inputs_honest_nan(self):
        row = monitor_factor(None, None, None, "no_input")
        assert np.isnan(row["rolling_ic"])
        assert np.isnan(row["ic_tvalue"])
        assert np.isnan(row["factor_ret_vol"])

    def test_monitor_panel_batch(self):
        rng = np.random.default_rng(6)
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        stocks = [f"S{i}" for i in range(40)]
        panels = {
            "mom1": pd.DataFrame(rng.normal(0, 1, (30, 40)), index=dates, columns=stocks),
            "mom2": pd.DataFrame(rng.normal(0, 1, (30, 40)), index=dates, columns=stocks),
        }
        ic_map = {"mom1": pd.Series(rng.normal(0, 1, 30), index=dates)}
        df, alerted = monitor_panel(panels, ic_map=ic_map,
                                    ret_wide=panels["mom2"],
                                    category_map={"mom1": "momentum", "mom2": "momentum"})
        assert list(df.columns) == list(monitor_factor(None, None, None, "x").keys())
        assert len(df) == 2
