"""
deep_factors — V4.1 深度学习因子模块包

对外统一导出 LSTM、横截面 Transformer、自编码器因子发现和集成器。
DeepFactorEnsemble 是主要入口，适合在 Alpha 生产流程中统一调度各类深度因子模型。
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。
"""

from .autoencoder import FactorAutoEncoder
from .ensemble import DeepFactorEnsemble
from .lstm import LSTMFactorModel
from .transformer import CrossSectionalTransformer

__all__ = [
    "LSTMFactorModel",
    "CrossSectionalTransformer",
    "FactorAutoEncoder",
    "DeepFactorEnsemble",
]

__version__ = "4.1"
