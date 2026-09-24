"""
factor_system — 完整因子体系
V4.1 feature

包含：
- registry: 因子注册表（60+因子元数据）
- engine: 因子计算引擎
- evaluation: 因子评估（IC/ICIR/衰减）
- combination: 因子合成
- monitor: 因子监控
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。
"""

from .registry import FactorRegistry, FactorDef, FactorCategory, FactorHorizon, register_default_factors
from .engine import FactorEngine
from .evaluation import FactorEvaluator
from .combination import FactorCombiner
from .monitor import FactorMonitor

__all__ = [
    'FactorRegistry', 'FactorDef', 'FactorCategory', 'FactorHorizon', 'register_default_factors',
    'FactorEngine',
    'FactorEvaluator',
    'FactorCombiner',
    'FactorMonitor',
]
