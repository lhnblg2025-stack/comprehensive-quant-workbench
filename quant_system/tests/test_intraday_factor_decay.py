"""analysis_core/intraday_factor_decay 因子实时衰减热力图测试（V12.1 域M）。

覆盖:
  - 快照 mock（tmp parquet，8 窗口 × 100 只）
  - 因子派生 / 同名列优先
  - 窗口分组（30min 桶、合并、末窗口 open）
  - 滚动 IC：手工构造已知相关（±1）、防前视（t+1 收益）、只到最新已收盘窗口
  - 下岗熔断：连续 3 负且 |IC| 递增；不递增/有正间隔不触发；下岗不回魂
  - 权重再分配：下岗归 0 + 剩余归一；等权兜底；全下岗全 0；实时排名
  - 输出 json/md 结构、无快照降级、CLI、缺省最新日期
全 mock、无网络。
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent


def _import():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import importlib
    return importlib.import_module("quant_system.analysis_core.intraday_factor_decay")


m = _import()

DATE = "2026-08-11"
DAY = "20260811"
TS8 = ["100505", "103505", "110505", "113005", "130505", "133505", "140505", "143505"]
N_STOCKS = 100
CODES = [f"{i:06d}" for i in range(N_STOCKS)]


# ── mock 快照构造 ─────────────────────────────────────────────────────────
def make_mock_snapshots(root: Path, date: str = DATE, ts_list: list[str] | None = None,
                        n_stocks: int = N_STOCKS, with_factor_cols: bool = True,
                        factor_fn=None, price_fn=None) -> Path:
    """构造 tmp/realtime_snapshot/YYYYMMDD/*.parquet 快照目录并返回该目录。

    factor_fn(t, i) -> dict[str, np.ndarray] 覆盖因子列；price_fn(t, i) 覆盖价格。
    默认：因子列 = mom_5=i / rev_5=-i / lowvol=i / turnover=i（rank 严格单调），
    价格 base 逐窗口下降 → 收益 rank 与 i 同向 → 默认 IC 为 ±1，可控已知。
    """
    ts_list = ts_list or TS8
    d = root / "realtime_snapshot" / date.replace("-", "")
    d.mkdir(parents=True, exist_ok=True)
    rng = np.arange(n_stocks)
    for t, ts in enumerate(ts_list):
        price = (100 - 2 * t) + rng if price_fn is None else price_fn(t, rng)
        data = {
            "code": CODES[:n_stocks],
            "name": [f"stock_{i}" for i in range(n_stocks)],
            "price": price.astype(float),
            "pre_close": price.astype(float),
            "open": price.astype(float),
            "high": price.astype(float),
            "low": price.astype(float),
            "pct_chg": rng.astype(float) / 10.0,
            "turnover": np.full(n_stocks, 0.5 + t * 0.1),
            "ts": [date.replace("-", "") + ts] * n_stocks,
        }
        if with_factor_cols:
            default = {"mom_5": rng, "rev_5": -rng, "lowvol": rng, "turnover": rng}
            cols = factor_fn(t, rng) if factor_fn else default
            data.update(cols)
        pd.DataFrame(data).to_parquet(d / f"{ts}.parquet")
    return d


def _factor_panel(n_windows: int = 8, ret_ascending: bool = True) -> tuple[list[pd.DataFrame], dict[int, pd.Series]]:
    """手工构造 panel + 收益：rank 严格单调 → 截面相关精确 ±1。"""
    i = np.arange(N_STOCKS)
    panel = [
        pd.DataFrame({"mom_5": i, "rev_5": -i, "lowvol": i, "turnover": i}, index=CODES)
        for _ in range(n_windows)
    ]
    rets = {t: pd.Series(i if ret_ascending else -i, index=CODES) for t in range(n_windows - 1)}
    return panel, rets


# ── 因子计算 ──────────────────────────────────────────────────────────────
def test_derive_factors_formulas():
    df = pd.DataFrame({
        "code": ["000001", "000002", "000003", "000004"],
        "pct_chg": [1.0, -2.0, 3.0, 0.0],
        "turnover": [0.1, 0.2, 0.3, 0.4],
    })
    out = m.derive_factors(df)
    assert list(out.columns) == ["mom_5", "rev_5", "lowvol", "turnover"]
    assert out["mom_5"].tolist() == [1.0, -2.0, 3.0, 0.0]
    assert out["rev_5"].tolist() == [-1.0, 2.0, -3.0, 0.0]
    assert out["lowvol"].tolist() == [-1.0, -2.0, -3.0, 0.0]
    assert out["turnover"].tolist() == [0.1, 0.2, 0.3, 0.4]


def test_derive_factors_explicit_column_precedence():
    df = pd.DataFrame({
        "code": ["000001", "000002"],
        "pct_chg": [10.0, 20.0],
        "turnover": [0.5, 0.6],
        "mom_5": [1.0, 2.0],
    })
    out = m.derive_factors(df)
    assert out["mom_5"].tolist() == [1.0, 2.0]  # 同名列优先于派生
    assert out["rev_5"].tolist() == [-10.0, -20.0]
    assert out["turnover"].tolist() == [0.5, 0.6]


# ── 窗口构建 ──────────────────────────────────────────────────────────────
def test_build_windows_bucketing_and_merge():
    i = np.arange(10)
    def snap(ts, factor):
        return pd.DataFrame({"code": [f"{x:06d}" for x in range(10)], "price": i + factor,
                             "ts": [DATE.replace("-", "") + ts] * 10, "mom_5": i})
    w = m.build_windows([snap("103300", 0), snap("103700", 10), snap("110500", 0), snap("145000", 0)])
    assert [x.start.strftime("%H:%M") for x in w] == ["10:30", "11:00", "14:30"]
    assert len(w) == 3
    # 同一 30min 桶内两份快照合并，因子取均值
    merged = w[0].df
    assert len(merged) == 20
    assert w[0].is_open is False
    assert w[-1].is_open is True


def test_build_windows_bad_ts_dropped():
    df = pd.DataFrame({"code": ["000001", "000002"], "price": [1.0, 2.0],
                       "ts": ["bad_ts", DATE.replace("-", "") + "103500"]})
    w = m.build_windows([df])
    assert len(w) == 1
    assert len(w[0].df) == 1


# ── 滚动 IC ───────────────────────────────────────────────────────────────
def test_ic_perfect_positive_known():
    panel, rets = _factor_panel()
    ic = m.rolling_ic(panel, rets, method="spearman")
    assert len(ic) == len(panel) - 2  # 只到最新已收盘窗口
    assert ic["mom_5"].tolist() == pytest.approx([1.0] * 6, abs=1e-9)
    assert ic["lowvol"].tolist() == pytest.approx([1.0] * 6, abs=1e-9)
    assert ic["turnover"].tolist() == pytest.approx([1.0] * 6, abs=1e-9)


def test_ic_perfect_negative_known():
    panel, rets = _factor_panel()
    ic = m.rolling_ic(panel, rets, method="spearman")
    assert ic["rev_5"].tolist() == pytest.approx([-1.0] * 6, abs=1e-9)


def test_ic_uses_next_window_return_no_lookahead():
    """防前视：IC[t] 必须用 t+1 窗口收益，而非同窗口。

    构造：因子 rank 升序；窗口0 价格 rank 降序、窗口1 价格 rank 升序。
    同窗口相关 = -1，下一窗口收益相关 = +1 → 断言 +1 即证明用了 t+1 收益。
    """
    i = np.arange(N_STOCKS)
    panel = [pd.DataFrame({"mom_5": i, "rev_5": -i, "lowvol": i, "turnover": i}, index=CODES)
             for _ in range(3)]
    rets = {0: pd.Series(i, index=CODES)}  # 窗口0→1：因子0 升序 vs 收益 升序 → +1
    ic = m.rolling_ic(panel, rets, method="spearman")
    assert ic.loc[0, "mom_5"] == pytest.approx(1.0, abs=1e-9)
    # 反向收益 → 必为 -1（若误用同窗口则相反）
    rets2 = {0: pd.Series(-i, index=CODES)}
    ic2 = m.rolling_ic(panel, rets2, method="spearman")
    assert ic2.loc[0, "mom_5"] == pytest.approx(-1.0, abs=1e-9)


def test_ic_stops_at_latest_closed_window(tmp_path):
    """8 窗口 → 6 个 IC（t=0..5），最后两窗口 None，latest_closed_window=n-2。"""
    d = make_mock_snapshots(tmp_path)
    res = m.analyze(date=DATE, snapshot_dir=d.parent, out_dir=tmp_path / "out")
    f = res["factors"][0]
    assert len(f["ic"]) == 8
    assert f["ic"][:6] == pytest.approx([1.0] * 6, abs=1e-9)
    assert f["ic"][6] is None and f["ic"][7] is None
    assert res["latest_closed_window"] == 6
    assert res["windows"][-1]["is_open"] is True


def test_ic_insufficient_pairs_none():
    codes = ["000001", "000002"]
    i = np.arange(2)
    panel = [pd.DataFrame({"mom_5": i, "rev_5": -i, "lowvol": i, "turnover": i}, index=codes)
             for _ in range(3)]
    rets = {0: pd.Series(i, index=codes), 1: pd.Series(i, index=codes)}
    ic = m.rolling_ic(panel, rets, method="spearman")
    assert ic["mom_5"].isna().all()  # 有效对数 < MIN_PAIRS → None，不崩溃


def test_pearson_method_option():
    i = np.arange(N_STOCKS)
    panel = [pd.DataFrame({"mom_5": np.exp(i * 0.05), "rev_5": -i, "lowvol": i, "turnover": i}, index=CODES)
             for _ in range(3)]
    rets = {0: pd.Series(i, index=CODES)}
    ic_s = m.rolling_ic(panel, rets, method="spearman")
    ic_p = m.rolling_ic(panel, rets, method="pearson")
    assert ic_s.loc[0, "mom_5"] == pytest.approx(1.0, abs=1e-9)
    assert 0.0 < ic_p.loc[0, "mom_5"] < 1.0  # 非线性单调：Pearson 弱于 Spearman
    assert ic_p.loc[0, "rev_5"] == pytest.approx(-1.0, abs=1e-9)


# ── 下岗熔断 ──────────────────────────────────────────────────────────────
def test_decay_state_triggers_three_neg_increasing():
    status = m.decay_state({"f": [-0.4, -0.6, -1.0, -1.0, -1.0, -1.0, None, None]}, 8)
    assert status["f"] == ["up", "up", "up", "down", "down", "down", "down", "down"]


def test_decay_state_no_trigger_without_increasing_magnitude():
    status = m.decay_state({"f": [-1.0, -0.6, -0.4, -1.0, -1.0, -1.0, None, None]}, 8)
    assert status["f"] == ["up"] * 8  # |IC| 递减 → 不触发


def test_decay_state_no_trigger_when_positive_gap():
    status = m.decay_state({"f": [-0.4, 0.5, -0.6, -1.0, -1.0, -1.0, None, None]}, 8)
    assert status["f"] == ["up"] * 8  # 中间有正 IC 打断连续性


def test_decay_state_stays_down_after_trigger():
    # 触发后即便 IC 转正也保持 down（今日下岗不回魂）
    status = m.decay_state({"f": [-0.4, -0.6, -1.0, 1.0, 1.0, 1.0, None, None]}, 8)
    assert status["f"] == ["up", "up", "up", "down", "down", "down", "down", "down"]


def test_decay_state_ignores_unsettled_tail():
    status = m.decay_state({"f": [-0.4, -0.6, None, None, None, None, None, None]}, 8)
    assert status["f"] == ["up"] * 8  # 未结算窗口不参与连续判定


# ── 权重再分配 ────────────────────────────────────────────────────────────
def _ic_status_harness():
    ic_full = {
        "mom_5": [0.5] * 6 + [None, None],
        "rev_5": [-0.5] * 6 + [None, None],
        "lowvol": [0.3] * 6 + [None, None],
        "turnover": [-0.4, -0.6, -1.0] + [-1.0] * 3 + [None, None],
    }
    status = m.decay_state(ic_full, 8)
    return ic_full, status


def test_reallocate_weights_initial_equal():
    ic_full, status = _ic_status_harness()
    w, r = m.reallocate_weights(ic_full, status, 8)
    for name in ic_full:
        assert w[name][0] == pytest.approx(0.25)
    assert r["mom_5"][0] == 1  # 并列按因子顺序


def test_reallocate_weights_down_zero_remaining_normalized():
    ic_full, status = _ic_status_harness()
    w, r = m.reallocate_weights(ic_full, status, 8)
    for t in range(3, 8):
        assert w["turnover"][t] == 0.0  # 下岗归 0
        total = sum(w[n][t] for n in ic_full)
        assert total == pytest.approx(1.0)  # 剩余归一
    # 窗口3 按 |ic[2]| = [0.5, 0.5, 0.3, 0] 比例
    t3 = [w[n][3] for n in ["mom_5", "rev_5", "lowvol", "turnover"]]
    assert t3[:3] == pytest.approx([0.5 / 1.3, 0.5 / 1.3, 0.3 / 1.3])
    # 实时排名：mom_5/rev_5 并列第一（按因子顺序），lowvol 第三，turnover 第四
    assert [r[n][3] for n in ["mom_5", "rev_5", "lowvol", "turnover"]] == [1, 2, 3, 4]


def test_reallocate_weights_all_down_all_zero():
    # 四因子均连续 3 负且 |IC| 递增 → 全部下岗 → 权重全 0
    ic_full = {n: [-0.3, -0.5, -1.0] + [-1.0] * 3 + [None, None] for n in ["mom_5", "rev_5", "lowvol", "turnover"]}
    status = m.decay_state(ic_full, 8)
    assert all(s == "down" for s in status["mom_5"][3:])
    w, r = m.reallocate_weights(ic_full, status, 8)
    for t in range(3, 8):
        assert all(w[n][t] == 0.0 for n in ic_full)  # 全下岗 → 全 0


def test_reallocate_weights_fallback_equal_no_ic():
    ic_full = {n: [None] * 8 for n in ["mom_5", "rev_5", "lowvol", "turnover"]}
    status = m.decay_state(ic_full, 8)
    w, _ = m.reallocate_weights(ic_full, status, 8)
    for t in range(8):
        assert all(w[n][t] == pytest.approx(0.25) for n in ic_full)  # 无 IC → 等权


# ── 端到端与输出 ───────────────────────────────────────────────────────────
def test_output_json_structure_full_pipeline(tmp_path):
    d = make_mock_snapshots(tmp_path)
    res = m.analyze(date=DATE, snapshot_dir=d.parent, out_dir=tmp_path / "out")
    assert res["degraded"] is False
    assert res["date"] == DATE
    assert res["n_windows"] == 8
    assert res["method"] == "spearman"
    assert len(res["factors"]) == 4
    for f in res["factors"]:
        assert set(f) >= {"name", "ic", "status", "weight", "rank",
                          "trigger_window", "current_weight", "current_rank"}
        assert len(f["ic"]) == len(f["status"]) == len(f["weight"]) == len(f["rank"]) == 8
    jp = tmp_path / "out" / "factor_decay" / DAY / f"factor_decay_{DATE}.json"
    assert jp.exists()
    saved = json.loads(jp.read_text(encoding="utf-8"))
    assert saved == res


def test_output_md_written(tmp_path):
    d = make_mock_snapshots(tmp_path)
    m.analyze(date=DATE, snapshot_dir=d.parent, out_dir=tmp_path / "out")
    mp = tmp_path / "out" / "factor_decay" / DAY / f"factor_decay_{DATE}.md"
    assert mp.exists()
    txt = mp.read_text(encoding="utf-8")
    assert "因子实时衰减热力图" in txt and "mom_5" in txt and "turnover" in txt


def test_no_snapshot_degraded_json(tmp_path):
    res = m.analyze(date="2020-01-01", snapshot_dir=tmp_path / "none", out_dir=tmp_path / "out")
    assert res["degraded"] is True
    assert res["reason"] == "no_snapshot"
    assert res["factors"] == []
    jp = tmp_path / "out" / "factor_decay" / "20200101" / "factor_decay_2020-01-01.json"
    assert jp.exists()
    saved = json.loads(jp.read_text(encoding="utf-8"))
    assert saved["degraded"] is True and saved["factors"] == []


def test_cli_writes_files_return_zero(tmp_path):
    d = make_mock_snapshots(tmp_path)
    code = m.main(["--date", DATE, "--snapshot-dir", str(d.parent),
                   "--out-dir", str(tmp_path / "cli_out")])
    assert code == 0
    jp = tmp_path / "cli_out" / "factor_decay" / DAY / f"factor_decay_{DATE}.json"
    assert jp.exists()


def test_cli_degraded_no_snapshot_zero(tmp_path):
    code = m.main(["--date", "2020-01-01", "--snapshot-dir", str(tmp_path / "none"),
                   "--out-dir", str(tmp_path / "cli_out")])
    assert code == 0
    jp = tmp_path / "cli_out" / "factor_decay" / "20200101" / "factor_decay_2020-01-01.json"
    assert json.loads(jp.read_text(encoding="utf-8"))["degraded"] is True


def test_default_latest_snapshot_date(tmp_path):
    make_mock_snapshots(tmp_path, date="2026-08-10", ts_list=TS8[:3])  # 3 窗口旧日期
    make_mock_snapshots(tmp_path, date=DATE, ts_list=TS8)             # 8 窗口新日期
    res = m.analyze(snapshot_dir=tmp_path / "realtime_snapshot")
    assert res["date"] == DATE
    assert res["n_windows"] == 8


def test_end_to_end_down_factor_weight_zero(tmp_path):
    """集成：构造因子前 3 窗口 IC 为负且 |IC| 递增 → 下岗；后续转正也不回魂。"""
    i = np.arange(N_STOCKS)
    wob = np.sin(2 * np.pi * 7 * i / N_STOCKS)
    amps = [6.0, 2.5, 0.0]  # |IC| 递增（已数值验证 0.984 < 0.999 < 1.0）

    def factor_fn(t, rng):
        cols = {"mom_5": rng, "rev_5": -rng, "lowvol": rng}
        if t < 3:
            cols["turnover"] = -(rng + amps[t] * wob)  # 负 IC，|IC| 递增
        else:
            cols["turnover"] = rng  # 转正，验证不回魂
        return cols

    d = make_mock_snapshots(tmp_path, factor_fn=factor_fn)
    res = m.analyze(date=DATE, snapshot_dir=d.parent, out_dir=tmp_path / "out")
    f = next(x for x in res["factors"] if x["name"] == "turnover")
    neg_ics = [v for v in f["ic"][:3]]
    assert all(v is not None and v < 0 for v in neg_ics)
    assert abs(neg_ics[0]) < abs(neg_ics[1]) < abs(neg_ics[2])
    assert f["trigger_window"] == 3
    assert f["status"][-1] == "down"
    assert f["current_weight"] == 0.0
    others = [x for x in res["factors"] if x["name"] != "turnover"]
    assert sum(x["current_weight"] for x in others) == pytest.approx(1.0, abs=1e-5)  # 剩余归一（6位四舍五入容差）


def test_end_to_end_ic_matches_hand_computed(tmp_path):
    """集成：输出 IC 与手工 Spearman（scipy 独立实现）一致。"""
    from scipy.stats import spearmanr
    d = make_mock_snapshots(tmp_path)
    res = m.analyze(date=DATE, snapshot_dir=d.parent)
    # 从 mock 快照独立计算各窗口均价 → 收益 → scipy spearman 交叉验证
    prices = []
    for ts in TS8:
        df = pd.read_parquet(d / f"{ts}.parquet")
        prices.append(df.groupby("code")["price"].mean().reindex(CODES))
    for t in range(6):
        ret = prices[t + 1] / prices[t] - 1.0
        expected, _ = spearmanr(np.arange(N_STOCKS), ret.to_numpy())
        assert res["factors"][0]["ic"][t] == pytest.approx(expected, abs=1e-9)
        assert expected == pytest.approx(1.0, abs=1e-9)  # mock 构造：完美正相关
