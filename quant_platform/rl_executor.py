# -*- coding: utf-8 -*-
"""
rl_executor.py — 强化学习执行代理 (V6.1, 推理模式)

问题背景: 规则式下单不考虑流动性碎片化。"怎么买"比"买什么"更重要:
    挂限价单等 30 秒 vs 直接扫盘口吃掉卖一, 不同场景成本差异巨大。

设计原则 (用户约束: 本地算力受限):
    - 训练在本机完成 (GTX 可跑 DQN/SAC), 导出 model 权重
    - 本模块只做**推理 (Inference)**: 加载权重, 根据盘口状态选动作
    - 推理时 CPU 消耗极低, Vultr 1 核完全扛得住
    - 无权重文件时自动降级为规则式策略 (被动限价/主动市价按盘口深度)

动作空间 (离散 4 动作):
    0 = PASS        不动作
    1 = LIMIT_BUY   挂限价单 (吃买一附近)
    2 = AGGRESSIVE  扫盘口主动买入
    3 = CANCEL      撤销挂单/等待

状态特征 (可扩展):
    [spread_ratio, depth_imbalance, price_vs_vwap, volatility, remaining_ratio, time_left]
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RLDecision:
    """RL 执行决策。

    Attributes:
        action: 动作名 (pass/limit_buy/aggressive/cancel)
        action_id: 动作编号 0~3
        confidence: 动作置信度 (0~1)
        reason: 决策依据
        q_values: 各动作 Q 值 (dict)
    """
    action: str = "pass"
    action_id: int = 0
    confidence: float = 0.0
    reason: str = "rule_fallback"
    q_values: dict = field(default_factory=dict)


#: 动作编号 → 名称
ACTION_NAMES = ["pass", "limit_buy", "aggressive", "cancel"]


class RLExecutor:
    """RL 执行代理 (推理模式, 支持降级)。"""

    def __init__(self, model_path: Optional[str] = None,
                 state_dim: int = 6, aggressive_threshold: float = 0.55):
        """初始化。

        Args:
            model_path: 训练好的模型权重路径 (.npy/.npz)。None 表示纯规则模式。
            state_dim: 状态特征维度 (与训练时一致)。
            aggressive_threshold: 主动扫盘口动作的 Q 值置信阈值。
        """
        self.state_dim = state_dim
        self.aggressive_threshold = aggressive_threshold
        self._weights: Optional[np.ndarray] = None
        self._bias: Optional[np.ndarray] = None
        self._loaded_path: Optional[str] = None
        if model_path:
            self.load(model_path)

    # ── 模型加载/保存 ──
    def load(self, path: str) -> bool:
        """加载推理权重。

        支持两种格式:
            - .npz: 含键 "w" (n_actions × state_dim) 与 "b" (n_actions,)
            - .json: {"w": [...], "b": [...]}

        Args:
            path: 权重文件路径。

        Returns:
            是否加载成功。
        """
        p = Path(path)
        if not p.exists():
            logger.warning("RL 权重不存在: %s, 使用规则模式", path)
            return False
        try:
            if p.suffix == ".npz":
                data = np.load(p)
                w = data["w"]
                b = data.get("b", np.zeros(w.shape[0]))
            elif p.suffix == ".json":
                data = json.loads(p.read_text(encoding="utf-8"))
                w = np.array(data["w"])
                b = np.array(data.get("b", [0.0] * w.shape[0]))
            else:
                logger.warning("不支持的权重格式: %s", p.suffix)
                return False
            if w.ndim != 2 or w.shape[1] != self.state_dim:
                logger.warning("权重维度不匹配: %s (期望 %d×%d)", w.shape, 4, self.state_dim)
                return False
            self._weights = w.astype(float)
            self._bias = b.astype(float)
            self._loaded_path = str(p)
            logger.info("RL 权重加载成功: %s", path)
            return True
        except Exception as exc:
            logger.warning("RL 权重加载失败 %s: %s", path, exc)
            return False

    @property
    def using_rl(self) -> bool:
        """是否使用 RL 模型 (True) 还是规则降级 (False)。"""
        return self._weights is not None

    # ── 状态构造 ──
    @staticmethod
    def build_state(order_book_snapshot: dict) -> np.ndarray:
        """从盘口快照构造状态向量 (6 维)。

        Args:
            order_book_snapshot: 含以下键 (缺失时用中性值填充):
                best_bid / best_ask / last_price / vwap / depth_bid_qty /
                depth_ask_qty / remaining_qty / target_qty / time_left / max_time

        Returns:
            (6,) 状态向量。
        """
        bb = float(order_book_snapshot.get("best_bid", 0.0) or 0.0)
        ba = float(order_book_snapshot.get("best_ask", 0.0) or 0.0)
        last = float(order_book_snapshot.get("last_price", 0.0) or 0.0)
        vwap = float(order_book_snapshot.get("vwap", last or 1.0) or 1.0)
        spread = (ba - bb) / (last if last else 1.0) if ba > 0 and bb > 0 else 0.0
        bid_q = float(order_book_snapshot.get("depth_bid_qty", 0.0) or 0.0)
        ask_q = float(order_book_snapshot.get("depth_ask_qty", 0.0) or 0.0)
        imbalance = (bid_q - ask_q) / (bid_q + ask_q + 1e-9) if (bid_q + ask_q) > 0 else 0.0
        price_vs_vwap = (last - vwap) / (vwap if vwap else 1.0)
        rem = float(order_book_snapshot.get("remaining_qty", 0.0) or 0.0)
        tgt = float(order_book_snapshot.get("target_qty", 1.0) or 1.0)
        remaining_ratio = rem / tgt if tgt > 0 else 0.0
        t_left = float(order_book_snapshot.get("time_left", 0.0) or 0.0)
        t_max = float(order_book_snapshot.get("max_time", 1.0) or 1.0)
        time_left = t_left / t_max if t_max > 0 else 0.0
        vol = float(order_book_snapshot.get("volatility", 0.01) or 0.01)
        return np.array([spread, imbalance, price_vs_vwap, vol, remaining_ratio, time_left],
                        dtype=float)

    # ── 决策 ──
    def decide(self, order_book_snapshot: dict) -> RLDecision:
        """根据盘口状态输出执行动作。

        Args:
            order_book_snapshot: 盘口快照 (见 build_state)。

        Returns:
            RLDecision。
        """
        state = self.build_state(order_book_snapshot)
        if self.using_rl:
            q = self._weights @ state + self._bias
            action_id = int(np.argmax(q))
            q_vals = {ACTION_NAMES[i]: round(float(q[i]), 4) for i in range(len(q))}
            # 置信度 = 最优动作与次优动作的 softmax 差距
            exp_q = np.exp(q - q.max())
            probs = exp_q / exp_q.sum()
            conf = float(probs[action_id])
            return RLDecision(
                action=ACTION_NAMES[action_id], action_id=action_id,
                confidence=round(conf, 4), reason="rl_model", q_values=q_vals,
            )

        # ── 规则降级: 无 RL 权重时的启发式 ──
        return self._rule_decide(state, order_book_snapshot)

    def _rule_decide(self, state: np.ndarray, snap: dict) -> RLDecision:
        """规则式降级策略。

        逻辑:
            - 剩余量少 → 主动扫盘口 (时间压力大)
            - 盘口深度不平衡 (买单厚) → 挂限价等成交
            - 价差过大 → 挂限价
            - 否则 PASS
        """
        spread, imbalance, price_vs_vwap, vol, remaining_ratio, time_left = state
        q_approx = {}
        if remaining_ratio > 0.8 and time_left < 0.2:
            q_approx = {"pass": 0.1, "limit_buy": 0.2, "aggressive": 0.9, "cancel": 0.0}
            return RLDecision("aggressive", 2, 0.8, "rule: time_pressure", q_approx)
        if imbalance > 0.3 and spread < 0.01:
            q_approx = {"pass": 0.2, "limit_buy": 0.8, "aggressive": 0.4, "cancel": 0.1}
            return RLDecision("limit_buy", 1, 0.7, "rule: bid_depth", q_approx)
        if spread > 0.02:
            q_approx = {"pass": 0.4, "limit_buy": 0.7, "aggressive": 0.1, "cancel": 0.2}
            return RLDecision("limit_buy", 1, 0.6, "rule: wide_spread", q_approx)
        if remaining_ratio < 0.3:
            q_approx = {"pass": 0.6, "limit_buy": 0.3, "aggressive": 0.2, "cancel": 0.1}
            return RLDecision("pass", 0, 0.5, "rule: almost_done", q_approx)
        q_approx = {"pass": 0.7, "limit_buy": 0.2, "aggressive": 0.1, "cancel": 0.0}
        return RLDecision("pass", 0, 0.5, "rule: neutral", q_approx)

    # ── 工具 ──
    def save_weights_template(self, path: str) -> None:
        """生成权重模板文件（供本机训练后填充）。

        Args:
            path: 输出路径 (.npz)。
        """
        w = np.zeros((4, self.state_dim))
        b = np.zeros(4)
        np.savez(path, w=w, b=b)
        logger.info("权重模板已生成: %s (4×%d)", path, self.state_dim)


if __name__ == "__main__":  # pragma: no cover - 手工调试入口
    snap = {
        "best_bid": 10.00, "best_ask": 10.02, "last_price": 10.01, "vwap": 10.00,
        "depth_bid_qty": 50000, "depth_ask_qty": 20000,
        "remaining_qty": 800, "target_qty": 1000,
        "time_left": 30, "max_time": 300, "volatility": 0.015,
    }
    ex = RLExecutor()  # 规则模式
    d = ex.decide(snap)
    print("规则模式:", d.action, d.reason, d.confidence)

    # 加载模板权重 → RL 模式
    import tempfile, os
    tmp = os.path.join(tempfile.mkdtemp(), "rl_weights.npz")
    ex.save_weights_template(tmp)
    ex2 = RLExecutor(model_path=tmp)
    print("RL 模式:", ex2.using_rl)
    d2 = ex2.decide(snap)
    print("RL 决策:", d2.action, d2.reason, d2.q_values)
