"""ic_oos_report — 因子方向样本外验证报告测试 (V12.3 补覆盖)。

用合成日K验证: 抽样/时段切分/IC统计/方向保持率报告结构/输出落盘。
重计算部分(面板构建)以 monkeypatch 注入合成 IC 统计, 避免跑全量。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
for p in (ROOT, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import ic_oos_report as oos  # noqa: E402


def _synthetic_kline(n: int = 600, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = 100 * np.cumprod(1 + rng.normal(0.0002, 0.015, n))
    return pd.DataFrame({
        "date": dates,
        "open": close * 0.99, "high": close * 1.02, "low": close * 0.98,
        "close": close, "volume": rng.integers(1e5, 1e6, n).astype(float),
    })


def _synthetic_kl_dict(n_stocks: int = 5, n_days: int = 600) -> dict[str, pd.DataFrame]:
    # 每只股票独立 seed → 收益截面有差异(否则行内秩全同, IC 恒 NaN)
    return {f"{600000+i:06d}": _synthetic_kline(n_days, seed=3 + i) for i in range(n_stocks)}


def test_load_sample_returns_valid(monkeypatch, tmp_path):
    """抽样返回足够历史的日K(列裁剪)。"""
    monkeypatch.setattr(oos, "KLINE_DIR", tmp_path / "kline")
    (tmp_path / "kline").mkdir()
    for i in range(3):
        _synthetic_kline(600).to_parquet(tmp_path / "kline" / f"{600001+i:06d}.parquet", index=False)
    _synthetic_kline(50).to_parquet(tmp_path / "kline" / "short.parquet", index=False)  # 历史不足被跳过
    kl = oos._load_sample(5, seed=1)
    assert len(kl) == 3  # 短历史被过滤
    assert "date" in next(iter(kl.values())).columns


def test_split_dates_ratio():
    kl = _synthetic_kl_dict(3, 400)
    train, test = oos._split_dates(kl, test_frac=0.4)
    assert len(train) == pytest.approx(240, abs=3)
    assert len(test) == pytest.approx(160, abs=3)
    assert train[-1] < test[0]  # 无重叠


def test_ic_stats_synthetic(monkeypatch, tmp_path):
    """注入合成面板: _ic_stats 产出 ic_mean/icir/winrate 行。"""
    import ic_vectorized as icv_mod  # 真实模块, 函数内 import 会拿到它
    monkeypatch.setenv("QV6_LOG_FILE", str(tmp_path / "logs" / "quantv6.log"))
    kl = _synthetic_kl_dict(40, 300)  # 40 只对齐面板宽度(>min_n=30)
    dates = pd.DatetimeIndex(sorted(set().union(*[set(d["date"]) for d in kl.values()])))[:200]

    codes = [f"60{i:04d}" for i in range(40)]  # 40 只 > min_n=30
    fake_panels = {"f1": pd.DataFrame(
        {c: np.sin(np.arange(len(dates)) / 10) + i for i, c in enumerate(codes)},
        index=dates)}

    def fake_build(kl_sub, names, series_fn, common, codes, spill, label):
        return fake_panels

    def fake_prepare(kl_sub, dates):
        return dates, list(kl_sub.keys())

    monkeypatch.setattr(icv_mod, "_build_panels", fake_build)
    monkeypatch.setattr(icv_mod, "_prepare_common", fake_prepare)
    df = oos._ic_stats(kl, dates, ["f1"], None, 5)
    assert not df.empty
    row = df.iloc[0]
    assert row["factor"] == "f1"
    assert isinstance(row["ic_mean"], float)
    assert isinstance(row["icir"], float)
    assert 0 <= row["winrate"] <= 1


def test_main_report_structure(monkeypatch, tmp_path):
    """main() 用注入的 train/test IC 统计 → 报告含保持率/翻转/稳定Top, 落盘。"""
    monkeypatch.setenv("QV6_LOG_FILE", str(tmp_path / "logs" / "quantv6.log"))
    monkeypatch.setattr(oos, "OUT_DIR", tmp_path / "ic_report")
    monkeypatch.setattr(oos, "_load_sample", lambda n, seed: _synthetic_kl_dict(3, 600))
    monkeypatch.setattr(oos, "_split_dates", lambda kl, frac: (
        pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=300)),
        pd.DatetimeIndex(pd.bdate_range("2025-01-01", periods=200))))

    def fake_ic_stats(kl, dates, names, series_fn, forward):
        rng = np.random.default_rng(len(dates))
        n = len(names)
        rows = []
        for i, name in enumerate(names):
            ic = rng.normal(0.01 if i % 2 == 0 else -0.01, 0.05, 50)
            rows.append({"factor": name, "ic_mean": float(ic.mean()),
                         "icir": float(ic.mean() / (ic.std() + 1e-9) * 7),
                         "winrate": float((ic > 0).mean()),
                         "n_days": 50})
        return pd.DataFrame(rows)

    monkeypatch.setattr(oos, "_ic_stats", fake_ic_stats)
    # 因子清单注入(真实 ic_vectorized 模块的 ZOO_FACTOR_NAMES)
    import ic_vectorized as icv_mod2
    monkeypatch.setattr(icv_mod2, "ZOO_FACTOR_NAMES", [f"f{i}" for i in range(6)])
    # main() 内 parse_args 解析 sys.argv → 注入脚本参数
    monkeypatch.setattr(sys, "argv", ["ic_oos_report.py", "--n", "5"])
    rc = oos.main()
    assert rc == 0
    jf = tmp_path / "ic_report" / "IC_OOS_REPORT.json"
    mf = tmp_path / "ic_report" / "IC_OOS_REPORT.md"
    assert jf.exists() and mf.exists()
    rep = json.loads(jf.read_text(encoding="utf-8"))
    assert 0 <= rep["sign_keep_rate"] <= 1
    assert rep["n_factors"] == 6
    assert "stable_top" in rep and "flipped_factors" in rep
    assert "样本外验证报告" in mf.read_text(encoding="utf-8")
    assert "符号保持率" in mf.read_text(encoding="utf-8")
