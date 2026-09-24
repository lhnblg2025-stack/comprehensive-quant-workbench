"""
deep_factors/lstm.py — LSTM 时序收益预测模型
V4.1 feature: 深度学习因子
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class LSTMFactorModel:
    """LSTM 收益预测模型。

    使用 LSTM 网络从历史价格序列中学习时间序列模式，预测未来收益，
    输出值可直接作为 Alpha 因子。TensorFlow/Keras 是可选依赖，只有在 build/fit 时才需要。
    """

    def __init__(
        self,
        seq_length: int = 60,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
    ):
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self._model = None
        self._is_trained = False

    @staticmethod
    def _returns_matrix(prices: pd.DataFrame) -> np.ndarray:
        """将价格矩阵转为日收益矩阵（处理 inf/nan）。"""
        if prices is None or prices.empty:
            return np.empty((0, 0))
        return prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0).values

    def _prepare_sequences(self, prices: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """将价格数据转换为 LSTM 序列，形状为 (样本数, 序列长度, 股票数)。

        训练语义：窗口 [i-L, i) 预测已实现的收益 returns[i]（窗口后一日）。
        """
        if prices is None or prices.empty:
            return np.empty((0, self.seq_length, 0)), np.empty((0, 0))
        returns = self._returns_matrix(prices)
        n_obs = len(returns)
        x_seq, y_seq = [], []
        for i in range(self.seq_length, n_obs):
            x_seq.append(returns[i - self.seq_length:i])
            y_seq.append(returns[i])  # 预测窗口后一日的横截面收益。
        return np.array(x_seq), np.array(y_seq)

    def _last_sequence(self, prices: pd.DataFrame) -> np.ndarray:
        """构建预测**下一交易日**收益的输入窗口（含最后一行收益）。

        P1-Q6-fix: 原 predict() 直接取 _prepare_sequences 的最后一个样本，
        其窗口为 returns[n_obs-1-L : n_obs-1]（不含最后一行），模型输出的是
        对**最后已实现交易日**收益的预测，Alpha 信号滞后一日。这里改为
        returns[n_obs-L : n_obs]（含最后一行），与训练时"窗口预测窗口后一日"的
        语义对齐。
        """
        returns = self._returns_matrix(prices)
        n_obs = len(returns)
        n_stocks = returns.shape[1] if returns.ndim > 1 else 0
        if n_obs < self.seq_length:
            return np.empty((0, self.seq_length, n_stocks))
        return np.array([returns[n_obs - self.seq_length:n_obs]])

    def build(self, n_features: int = 1):
        """构建 LSTM 网络；未安装 TensorFlow 时给出清晰错误。"""
        try:
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.optimizers import Adam
        except ImportError as exc:
            self._model = None
            raise ImportError("需要安装 tensorflow 或 keras 才能构建 LSTMFactorModel") from exc

        model = Sequential()
        model.add(
            LSTM(
                self.hidden_size,
                return_sequences=(self.num_layers > 1),
                input_shape=(self.seq_length, n_features),
            )
        )
        if self.num_layers > 1:
            for i in range(1, self.num_layers):
                model.add(LSTM(self.hidden_size, return_sequences=(i < self.num_layers - 1)))
                model.add(Dropout(self.dropout))
        model.add(Dropout(self.dropout))
        model.add(Dense(n_features))  # 每只股票输出一个预测收益，和 prices.columns 对齐。
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mse")
        self._model = model
        return model

    def fit(
        self,
        prices: pd.DataFrame,
        validation_split: float = 0.2,
        epochs: int = 50,
        batch_size: int = 64,
        verbose: int = 0,
    ):
        """训练 LSTM 模型，输入为日期 x 股票的价格矩阵。"""
        x_seq, y_seq = self._prepare_sequences(prices)
        if len(x_seq) == 0:
            raise ValueError("价格数据长度不足，无法生成 LSTM 训练序列")
        if self._model is None:
            self.build(n_features=x_seq.shape[2])

        history = self._model.fit(
            x_seq.astype(np.float32),
            y_seq.astype(np.float32),
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self._is_trained = True
        return history

    def predict(self, prices: pd.DataFrame) -> pd.Series:
        """预测最近一期横截面收益，作为 Alpha 信号。"""
        if self._model is None:
            raise ValueError("模型尚未构建或训练，请先调用 build/fit")

        # P1-Q6-fix: 用含最后一行收益的窗口预测下一交易日（原实现取最后一个
        # 训练样本，窗口不含最后一行 → 预测的是最后已实现日，信号滞后一日）。
        x_pred = self._last_sequence(prices)
        if len(x_pred) == 0:
            return pd.Series(dtype=float)
        preds = self._model.predict(x_pred.astype(np.float32), verbose=0)[0]
        return pd.Series(np.asarray(preds).flatten(), index=prices.columns)

    def predict_alpha(self, prices: pd.DataFrame) -> pd.Series:
        """便捷方法：返回预测收益作为 Alpha 因子。"""
        return self.predict(prices)
