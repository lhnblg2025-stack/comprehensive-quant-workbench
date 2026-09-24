"""
deep_factors/autoencoder.py — 自编码器无监督因子发现
V4.1 feature: 深度学习因子
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class FactorAutoEncoder:
    """自编码器因子发现。

    从大量原始特征中学习紧凑潜在表示，每个潜在维度都可作为一个数据驱动因子。
    TensorFlow/Keras 是可选依赖，只有在 build/fit/extract_factors 时才需要。
    """

    def __init__(self, input_dim: int, latent_dim: int = 16, encoding_layers: list[int] | None = None):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoding_layers = encoding_layers or [64, 32]
        self._encoder = None
        self._autoencoder = None
        # P2-Q6-fix (Q6-M492): 输入标准化参数（fit 时拟合、extract_factors 时复用）
        self._input_mean = None
        self._input_std = None

    def build(self):
        """构建自编码器，Encoder 输出潜在因子，Decoder 重构原始特征。"""
        try:
            from tensorflow.keras import Model, layers
            import tensorflow as tf
        except ImportError as exc:
            self._encoder = None
            self._autoencoder = None
            raise ImportError("需要安装 tensorflow 才能构建 FactorAutoEncoder") from exc
        # P2-Q6-fix (Q6-L493): 固定随机种子，保证结果可复现
        tf.random.set_seed(42)

        # Encoder：逐层压缩原始特征，学习稳健的潜在表示。
        inputs = layers.Input(shape=(self.input_dim,))
        x = inputs
        for units in self.encoding_layers:
            x = layers.Dense(units, activation="relu")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
        latent = layers.Dense(self.latent_dim, activation="linear", name="latent")(x)

        # Decoder：从潜在因子重构输入，用重构误差驱动无监督训练。
        x = latent
        for units in reversed(self.encoding_layers):
            x = layers.Dense(units, activation="relu")(x)
        outputs = layers.Dense(self.input_dim, activation="linear")(x)

        self._autoencoder = Model(inputs, outputs)
        self._encoder = Model(inputs, latent)
        self._autoencoder.compile(optimizer="adam", loss="mse")
        return self._autoencoder

    def fit(self, X: np.ndarray, epochs: int = 100, batch_size: int = 64, verbose: int = 0):
        """无监督训练：目标值等于输入特征本身。

        P2-Q6-fix (Q6-M492): 训练前对 X 做 z-score（拟合参数保存，提取时复用），
        避免高量纲特征主导重构损失、潜在因子被尺度绑架。
        """
        if self._autoencoder is None:
            self.build()
        X = np.asarray(X, dtype=np.float32)
        self._input_mean = X.mean(axis=0, keepdims=True)
        self._input_std = X.std(axis=0, keepdims=True)
        self._input_std[self._input_std < 1e-8] = 1.0
        Xs = (X - self._input_mean) / self._input_std
        return self._autoencoder.fit(
            Xs,
            Xs,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=verbose,
        )

    def extract_factors(self, X: np.ndarray, feature_names: list[str] | None = None) -> pd.DataFrame:
        """提取潜在因子矩阵，行顺序与输入样本一致。

        P2-Q6-fix (Q6-M492): 用 fit 保存的标准化参数对 X 做同样变换后再预测。
        P2-Q6-fix (Q6-L493): feature_names 为输入特征名（长度=input_dim），与潜在维度
        不对应，仅作留档保留（不用于列名）；潜在列名用 ae_factor_N。
        """
        if self._encoder is None:
            raise ValueError("模型尚未构建或训练，请先调用 build/fit")
        X = np.asarray(X, dtype=np.float32)
        if self._input_mean is not None:
            X = (X - self._input_mean) / self._input_std
        latent = self._encoder.predict(X, verbose=0)
        cols = [f"ae_factor_{i + 1}" for i in range(self.latent_dim)]
        return pd.DataFrame(latent, columns=cols)
