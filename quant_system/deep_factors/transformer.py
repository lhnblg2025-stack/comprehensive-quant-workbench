"""
deep_factors/transformer.py — 横截面 Transformer 因子合成模型
V4.1 feature: 深度学习因子
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class CrossSectionalTransformer:
    """横截面 Attention 模型。

    将股票视为横截面序列（可按行业、市值或其他稳定规则排序），用 Transformer Encoder
    捕捉股票间相对关系，输出每只股票的预测 Alpha。TensorFlow/Keras 是可选依赖。
    """

    def __init__(self, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self._model = None
        self._n_features = 0

    def build(self, n_features: int):
        """构建横截面 Transformer；输入形状为 (批次, 股票数, 特征数)。"""
        try:
            from tensorflow.keras import Model, layers
            import tensorflow as tf
        except ImportError as exc:
            self._model = None
            raise ImportError("需要安装 tensorflow 才能构建 CrossSectionalTransformer") from exc
        # P2-Q6-fix (Q6-L493): 固定随机种子，保证结果可复现
        tf.random.set_seed(42)

        inputs = layers.Input(shape=(None, n_features))
        x = layers.Dense(self.d_model)(inputs)
        x = layers.LayerNormalization()(x)

        for _ in range(self.num_layers):
            # 多头注意力学习股票之间的横截面相互作用。
            attn = layers.MultiHeadAttention(
                num_heads=self.nhead,
                key_dim=max(1, self.d_model // self.nhead),
                dropout=self.dropout,
            )(x, x)
            attn = layers.Dropout(self.dropout)(attn)
            x = layers.Add()([x, attn])
            x = layers.LayerNormalization()(x)

            # 前馈网络对每只股票的上下文表示做非线性变换。
            ffn = layers.Dense(self.d_model * 4, activation="relu")(x)
            ffn = layers.Dropout(self.dropout)(ffn)
            ffn = layers.Dense(self.d_model)(ffn)
            x = layers.Add()([x, ffn])
            x = layers.LayerNormalization()(x)

        outputs = layers.Dense(1)(x)  # 每只股票输出一个预测值。
        self._model = Model(inputs=inputs, outputs=outputs)
        self._model.compile(optimizer="adam", loss="mse")
        self._n_features = n_features
        return self._model

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 32,
            verbose: int = 0, validation_split: float = 0.2, patience: int = 5):
        """训练 Transformer，X/y 均按 (批次, 股票数, 特征/目标) 组织。

        P2-Q6-fix (Q6-M489): 原实现全样本 in-sample 训练（无验证集/早停）→ 过拟合。
        现按批次尾部切出**时间序列验证集**（保持时序），加 EarlyStopping 并恢复
        最优权重；样本批次过少时只留最后一批做验证。
        """
        if self._model is None:
            self.build(X.shape[2])
        n_batch = X.shape[0]
        if n_batch <= 2:
            raise ValueError(f"训练批次不足（batch={n_batch}），无法切出验证集")
        split = max(int(n_batch * (1 - validation_split)), 1)
        if split >= n_batch:
            split = max(n_batch - 1, 1)
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]
        from tensorflow.keras.callbacks import EarlyStopping
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
        ]
        return self._model.fit(
            X_tr, y_tr,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=verbose,
        )

    def predict_alpha(self, X: np.ndarray, symbols: list[str]) -> pd.Series:
        """输出横截面 Alpha 序列，并按 symbols 对齐。

        P2-Q6-fix (Q6-M488): 原实现只返回 preds[0,:,0] 静默丢弃其余样本，且不校验
        symbols 数量与 X 股票维。现校验形状/长度，且只接受单截面（batch=1）输入；
        多日期输入显式抛错，避免张冠李戴。
        """
        if self._model is None:
            raise ValueError("模型尚未构建或训练，请先调用 build/fit")
        if X is None or X.ndim != 3:
            raise ValueError(f"predict_alpha 需要 (batch, stocks, features) 3 维输入，实际 {0 if X is None else X.ndim} 维")
        if symbols is None:
            raise ValueError("predict_alpha 需要传入 symbols 列表（与 X 股票维对齐）")
        if X.shape[1] != len(symbols):
            raise ValueError(f"symbols 数量 {len(symbols)} 与 X 股票维 {X.shape[1]} 不一致")
        preds = self._model.predict(X, verbose=0)
        if preds.shape[0] != 1:
            raise ValueError(
                f"predict_alpha 仅支持单截面输入（batch=1），实际 batch={preds.shape[0]}；多日期请逐日调用"
            )
        return pd.Series(preds[0, :, 0], index=symbols)
