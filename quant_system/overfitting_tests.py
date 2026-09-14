"""
overfitting_tests.py — 过拟合检验工具集 (Deflated Sharpe Ratio, Bailey & Lopez de Prado 2014)
V4.1 feature

提供三个核心函数：
1. deflated_sharpe_ratio — DSR，校正多重比较偏误后评估策略夏普的统计显著性
2. num_false_strategies    — 给定夏普阈值下，预期出现的虚假正阳策略数
3. min_backtest_length     — 达到指定置信水平所需的最小回测样本量

参考文献
--------
Bailey, D. H., & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio:
Correcting for Multiple Testing. Journal of Portfolio Management, 40(1), 94-107.
"""

import warnings

import numpy as np
from scipy import stats

# 欧拉-马歇罗尼常数，用于 E[max Z] 近似
_EULER_GAMMA = 0.577215664901532860606512090082402431042


def _expected_max_normal(num_trials: int) -> float:
    """
    计算 N 个 i.i.d. 标准正态随机变量的期望最大值 E[max{Z}]。

    使用 Bailey & Lopez de Prado (2014) 的近似公式：
        E[max Z] ≈ (1-γ) · Φ⁻¹(1-1/N) + γ · Φ⁻¹(1-1/(N·e))
    其中 γ 为欧拉常数，e 为自然底数。

    当 N ≤ 1 时直接返回 0（单次试验的期望最大值就是均值 0）。

    Parameters
    ----------
    num_trials : int
        独立试验次数 N，即调参 / 策略搜索空间大小。

    Returns
    -------
    float
        E[max{Z}] 近似值。
    """
    if num_trials <= 1:
        return 0.0

    inv1 = stats.norm.ppf(1.0 - 1.0 / num_trials)
    inv2 = stats.norm.ppf(1.0 - 1.0 / (num_trials * np.e))
    return (1.0 - _EULER_GAMMA) * inv1 + _EULER_GAMMA * inv2


def deflated_sharpe_ratio(
    sharpe: float,
    num_trials: int,
    num_observations: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    target_sharpe: float = 0.0,
) -> tuple[float, float]:
    """
    计算 Deflated Sharpe Ratio (DSR) 及对应 p 值。

    DSR 在经典夏普比率检验的基础上做了两项校正：
    1. 多重比较校正 —— 通过 E[max{SR_null}] 惩罚试验次数；
    2. 非正态校正 —— 通过偏度 γ 和峰度 κ 修正方差估计。

    统计量定义：
        DSR = (SR_hat - SR_target - E[max{SR_null}]) / σ(SR)

    其中：
        σ²(SR) = (1 - γ·SR_hat + (κ-1)/4·SR_hat²) / (T - 1)
        σ²_null = 1 / (T - 1)
        E[max{SR_null}] = sqrt(σ²_null) · E[max{Z}]

    Parameters
    ----------
    sharpe : float
        策略的夏普比率估计值（请使用与观测频率一致的 SR，不要传入年化值）。
        例如：日频数据则用 SR_daily = mean(r) / std(r)；
        年化转换：SR_daily = SR_annual / sqrt(252)。
    num_trials : int
        独立试验次数，即回测中尝试的参数 / 策略组合数。
    num_observations : int
        样本观测数（如交易日数）。
    skew : float, optional
        收益序列偏度，默认 0（正态分布）。
    kurtosis : float, optional
        收益序列峰度，默认 3（正态分布）。
    target_sharpe : float, optional
        基准夏普比率，默认 0（检验策略夏普是否显著为正）。

    Returns
    -------
    dsr : float
        Deflated Sharpe Ratio 统计量（标准正态 z-score 尺度）。
    prob : float
        单侧上尾 p 值 P(Z ≥ DSR | H₀)，即 H₀: SR ≤ target_sharpe 的显著性。
        p 值越小，策略越显著优于基准。
    """
    if num_observations <= 1 or num_trials <= 0:
        return 0.0, 1.0

    # P2-Q9-fix (Q9-M552): 输入校验——任何分布的峰度均 >= 1；非法/非有限
    # skew 回退默认值（可见告警，不静默）。
    if not np.isfinite(kurtosis) or kurtosis < 1.0:
        warnings.warn(
            f"kurtosis={kurtosis} 非法（任何分布峰度≥1），回退正态峰度 3",
            stacklevel=2,
        )
        kurtosis = 3.0
    if not np.isfinite(skew):
        warnings.warn(f"skew={skew} 非有限，回退 0", stacklevel=2)
        skew = 0.0

    T = num_observations
    N = num_trials

    # 夏普比率方差（含偏度-峰度校正）
    variance = (
        1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe ** 2
    ) / (T - 1)
    # P2-Q9-fix (Q9-M552): 极端偏度/峰度可使方差估计为负，原实现钳制到
    # 1e-12 使 std_dev≈1e-6、DSR 爆炸到荒谬大值。现回退标准正态方差估计
    # 并告警（失败可见）。同时移除被钳制逻辑掩盖的 `std_dev < 1e-12`
    # 死分支（Q9-L553）。
    if not np.isfinite(variance) or variance <= 0:
        warnings.warn(
            f"DSR 方差估计为负/非有限 ({variance})，回退标准正态方差 1/(T-1)",
            stacklevel=2,
        )
        variance = 1.0 / (T - 1)
    std_dev = np.sqrt(variance)

    # 零假设下（SR=0）的方差：偏度=0，峰度=3 → V_null = 1 / (T-1)
    std_dev_null = np.sqrt(1.0 / (T - 1))

    # 多重比较校正项：E[max{SR_null}]
    expected_max_sharpe_null = std_dev_null * _expected_max_normal(N)

    dsr = (sharpe - target_sharpe - expected_max_sharpe_null) / std_dev
    prob = 1.0 - stats.norm.cdf(dsr)  # 单侧上尾 p 值：H₁: SR > target

    return dsr, prob


def num_false_strategies(
    sharpe: float,
    num_trials: int,
    num_observations: int,
) -> float:
    """
    估算给定观测数 T 和搜索空间 N 下，预期会出现的虚假正阳策略数量。

    零假设假定所有策略的真实夏普为零，此时观测夏普 ~ N(0, σ²_null)。
    对给定的夏普阈值，虚假正阳的期望数 = N · P(Z ≥ threshold / σ_null)。

    Parameters
    ----------
    sharpe : float
        夏普比率阈值（同频，非年化）。超过该值即视为"显著"。
    num_trials : int
        独立试验次数（参数 / 策略搜索数量）。
    num_observations : int
        样本观测数。

    Returns
    -------
    float
        预期虚假正阳策略数量（非整数，可用于后续 FDR 调整）。
    """
    if num_trials <= 0 or num_observations <= 1:
        return 0.0

    sigma_null = np.sqrt(1.0 / (num_observations - 1))

    if sigma_null < 1e-12:
        return float(num_trials)

    z_val = sharpe / sigma_null
    p_false = 1.0 - stats.norm.cdf(z_val)  # 单侧上尾

    return num_trials * p_false


def min_backtest_length(
    target_sharpe: float,
    confidence: float = 0.95,
    num_trials: int = 1,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> int:
    """
    计算达到目标置信水平所需的最小回测样本量 T_min。

    基于 DSR 框架，迭代求解最小的 T 使得：
        DSR(SR_target, T, N, γ, κ) ≥ Φ⁻¹(confidence)

    使用二分搜索在 [2, 10⁷] 区间内寻找精确解。

    Parameters
    ----------
    target_sharpe : float
        目标夏普比率（同频，非年化），须为正数。
        例如日频数据：SR_target = SR_annual_target / sqrt(252)。
    confidence : float, optional
        目标置信水平，默认 0.95。
    num_trials : int, optional
        独立试验次数（多重比较校正），默认 1（单一策略）。
    skew : float, optional
        收益偏度，默认 0。
    kurtosis : float, optional
        收益峰度，默认 3。

    Returns
    -------
    int
        所需最小观测数 T_min。
    """
    # P2-Q9-fix (Q9-L554): 目标夏普 <= 0 时"最小回测长度"语义上不需要样本，
    # 原返回 1（单样本）具有误导性，改抛 ValueError；confidence 越界时
    # norm.ppf 直接抛无友好提示的异常，现提前校验。
    if not np.isfinite(target_sharpe) or target_sharpe <= 0.0:
        raise ValueError("target_sharpe 必须为正有限数（非正目标夏普无需最小回测长度）")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence 必须在 (0,1) 区间内，got {confidence}")

    # 目标 Z 临界值
    z_target = stats.norm.ppf(confidence)

    lo, hi = 2, 10_000_000

    # 检查上界是否充分
    dsr_hi, _ = deflated_sharpe_ratio(
        target_sharpe, num_trials, hi, skew, kurtosis, target_sharpe=0.0,
    )
    if dsr_hi < z_target:
        return hi  # 即使 10⁷ 个观测也不够

    # 二分搜索最小 T
    while lo < hi:
        mid = (lo + hi) // 2
        dsr_mid, _ = deflated_sharpe_ratio(
            target_sharpe, num_trials, mid, skew, kurtosis, target_sharpe=0.0,
        )
        if dsr_mid >= z_target:
            hi = mid
        else:
            lo = mid + 1

    return lo
