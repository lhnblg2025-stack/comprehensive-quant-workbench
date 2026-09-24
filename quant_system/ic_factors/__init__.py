"""
QuantV6 因子层。
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。
"""
from quant_system.ic_factors.base import Factor, FactorResult
from quant_system.ic_factors.zoo import (
    register_defaults, register_factor, get_factor, list_factors,
    compute_factor_frame, factor_meta,
)
from quant_system.ic_factors.processor import process_factor_frame
from quant_system.ic_factors.combination import combine_factors, top_n
from quant_system.ic_factors.evaluator import (
    evaluate_factor, evaluate_over_time, quantile_returns,
    long_short_return, ic_summary,
)
