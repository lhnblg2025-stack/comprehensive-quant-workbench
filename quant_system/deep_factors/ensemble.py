"""
deep_factors/ensemble.py — 深度学习模型集成
V4.1 feature: 深度学习因子
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。
"""

from __future__ import annotations

import pandas as pd

from .autoencoder import FactorAutoEncoder
from .lstm import LSTMFactorModel
from .transformer import CrossSectionalTransformer


class DeepFactorEnsemble:
    """深度学习因子集成器。

    整合 LSTM、横截面 Transformer 和 Autoencoder 的输出，通过 Rank 平均或等权 Z-score
    合成最终 Alpha。该类是 deep_factors 包的主要对外入口。
    """

    def __init__(self):
        self._models = {}
        self._weights = None

    def add_lstm(self, **kwargs):
        """添加 LSTM 收益预测模型。"""
        self._models["lstm"] = LSTMFactorModel(**kwargs)
        return self._models["lstm"]

    def add_transformer(self, **kwargs):
        """添加横截面 Transformer 模型。"""
        self._models["transformer"] = CrossSectionalTransformer(**kwargs)
        return self._models["transformer"]

    def add_autoencoder(self, input_dim: int, **kwargs):
        """添加自编码器因子发现模型。"""
        self._models["autoencoder"] = FactorAutoEncoder(input_dim, **kwargs)
        return self._models["autoencoder"]

    def predict(self, data: dict) -> pd.DataFrame:
        """所有模型分别预测，返回 stocks x models 的 DataFrame。"""
        results = {}
        for name, model in self._models.items():
            pred = self._predict_single(name, model, data)
            if pred is not None and len(pred) > 0:
                results[name] = pred
        return pd.DataFrame(results)

    def _predict_single(self, name: str, model, data: dict) -> pd.Series:
        """按模型类型分发预测逻辑，并保持输出为一维 Alpha 序列。"""
        if name == "lstm":
            return model.predict_alpha(data.get("prices"))
        if name == "transformer":
            return model.predict_alpha(data.get("X"), data.get("symbols"))
        if name == "autoencoder":
            # P2-Q6-fix (Q6-L491): 缺 X_raw 键时原实现 extract_factors(None) 直接崩溃；
            #   加 key 检查给出清晰报错
            x_raw = data.get("X_raw")
            if x_raw is None:
                raise KeyError("autoencoder 模型需要 data['X_raw']（原始特征矩阵）作为输入")
            factors = model.extract_factors(x_raw, data.get("feature_names"))
            return factors.mean(axis=1)
        return pd.Series(dtype=float)

    def ensemble_alpha(self, data: dict, method: str = "rank") -> pd.Series:
        """集成合成：支持 Rank 平均或等权 Z-score。

        P2-Q6-fix (Q6-M490): 原实现用全样本均值/标准差做 zscore、在全样本池上做 rank
        （多日期生产使用时即前视/截面排名失真）。现按日期分组，**逐截面**标准化/排名
        后再平均；单日期输入退化为原行为。
        """
        preds = self.predict(data)
        if preds.empty:
            return pd.Series(dtype=float)

        def _standardize_block(block: pd.DataFrame) -> pd.DataFrame:
            if method == "rank":
                return block.rank(pct=True)
            std = block.std()
            return (block - block.mean()) / std.replace(0, pd.NA)

        if isinstance(preds.index, pd.MultiIndex):
            # 逐截面（每个日期）标准化/排名后写回，保持原始行序；groupby.apply 在
            #   MultiIndex 下重建索引不可靠，故用显式 mask 循环。
            grouped = preds.copy()
            for d in pd.unique(preds.index.get_level_values(0)):
                mask = preds.index.get_level_values(0) == d
                grouped.loc[mask] = _standardize_block(preds.loc[mask]).values
        else:
            grouped = _standardize_block(preds)

        if method in {"rank", "zscore", "z_score"}:
            return grouped.mean(axis=1)
        raise ValueError("method 仅支持 'rank'、'zscore' 或 'z_score'")
