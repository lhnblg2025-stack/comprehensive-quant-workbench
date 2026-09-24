"""
test_macro_learner — macro_learner._betainc_approx 兜底数值回归（域G）

运行: python3 -m pytest tests/test_macro_learner.py -q --no-header
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quant_system"))

scipy = pytest.importorskip("scipy")
from scipy.special import betainc  # noqa: E402

from quant_system.analysis_core.macro_learner import _betainc_approx  # noqa: E402


def test_betainc_fallback_matches_scipy_five_groups():
    """兜底近似与 scipy.special.betainc 对比 5 组 (a,b,x)，相对误差 < 1e-3。"""
    cases = [
        (5.0, 0.5, 0.5),      # 审计复现：旧实现漏 ÷a → 约 5 倍偏差
        (2.5, 0.5, 1e-6),     # 小 x：旧实现溢出风险点，不溢出且相对误差达标
        (10.0, 0.5, 0.1),
        (20.0, 0.5, 0.05),    # 大 df/2 场景（Welch 回退典型参数）
        (2.0, 3.0, 0.7),      # 走对称分支 I_x(a,b)=1-I_(1-x)(b,a)
    ]
    for a, b, x in cases:
        got = _betainc_approx(a, b, x)
        ref = float(betainc(a, b, x))
        assert 0.0 <= got <= 1.0, f"a={a} b={b} x={x}: 结果 {got} 越界 [0,1]"
        assert got == got, f"a={a} b={b} x={x}: 结果 NaN"
        rel = abs(got - ref) / ref if ref else abs(got - ref)
        assert rel < 1e-3, f"a={a} b={b} x={x}: got={got} ref={ref} relerr={rel}"


def test_betainc_fallback_bug_repro_in_unit_range():
    """a=5, x=0.5 输出必须 ∈[0,1]（旧实现输出 2.5>1）。"""
    got = _betainc_approx(5.0, 0.5, 0.5)
    assert 0.0 <= got <= 1.0
    assert abs(got - float(betainc(5.0, 0.5, 0.5))) / float(betainc(5.0, 0.5, 0.5)) < 1e-3


def test_betainc_fallback_small_x_no_overflow():
    """x=1e-6 不溢出：有限值、∈[0,1]，与 scipy 相对误差 < 1e-3。"""
    for a in (2.5, 5.0):
        got = _betainc_approx(a, 0.5, 1e-6)
        ref = float(betainc(a, 0.5, 1e-6))
        assert got == got and 0.0 <= got <= 1.0
        assert abs(got - ref) / ref < 1e-3
