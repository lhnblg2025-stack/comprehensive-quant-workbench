"""
signal_generator.py — 综合信号生成引擎
V4.1 feature

整合多策略、多因子、多模型信号，通过集成/融合生成最终交易信号。
"""

import numpy as np
import pandas as pd
from typing import Optional
from datetime import datetime
from enum import Enum

logger = __import__('logging').getLogger(__name__)


class SignalMethod(Enum):
    """信号生成方法"""
    FACTOR_RANK = "factor_rank"           # 因子排名
    MEAN_REVERSION = "mean_reversion"     # 均值回复
    MOMENTUM = "momentum"                 # 动量
    ML_PREDICT = "ml_predict"             # ML预测
    ENSEMBLE = "ensemble"                 # 集成
    CUSTOM = "custom"                     # 用户自定义


class Signal:
    """信号数据结构"""
    def __init__(self, name: str = "", values: Optional[pd.Series] = None,
                 method: str = "", weight: float = 1.0):
        self.name = name
        self.values = values or pd.Series(dtype=float)
        self.method = method
        self.weight = weight
        self.created_at = datetime.now()

    def __repr__(self):
        return f"Signal({self.name}, method={self.method}, w={self.weight})"


# ══════════════════════════════════════
# 1. 信号工厂
# ══════════════════════════════════════

class SignalFactory:
    """信号工厂 - 从不同输入生成信号"""

    @staticmethod
    def from_factor_rank(factors: pd.DataFrame, top_pct: float = 0.2,
                          reverse: bool = False) -> pd.Series:
        """从因子排名生成信号：Top -> +1, Bottom -> -1"""
        rank = factors.rank(pct=True, axis=1)
        signals = pd.DataFrame(0, index=factors.index, columns=factors.columns)
        if reverse:
            signals[rank <= top_pct] = 1
            signals[rank >= 1 - top_pct] = -1
        else:
            signals[rank >= 1 - top_pct] = 1
            signals[rank <= top_pct] = -1
        return signals.mean(axis=1) if factors.shape[1] > 1 else signals.iloc[:, 0]

    @staticmethod
    def from_mean_reversion(close: pd.Series, window: int = 20,
                             n_std: float = 2.0) -> pd.Series:
        """均值回复信号：布林带法"""
        ma = close.rolling(window).mean()
        std = close.rolling(window).std(ddof=0)
        upper = ma + n_std * std
        lower = ma - n_std * std
        signals = pd.Series(0, index=close.index)
        signals[close > upper] = -1
        signals[close < lower] = 1
        return signals

    @staticmethod
    def from_momentum(close: pd.Series, lookback: int = 20) -> pd.Series:
        """动量信号：最近N日收益方向"""
        ret = close.pct_change(lookback)
        signals = pd.Series(0, index=close.index)
        signals[ret > 0] = 1
        signals[ret < 0] = -1
        return signals

    @staticmethod
    def from_moving_average_cross(close: pd.Series, fast: int = 5,
                                    slow: int = 20) -> pd.Series:
        """均线交叉信号"""
        fast_ma = close.rolling(fast).mean()
        slow_ma = close.rolling(slow).mean()
        signals = pd.Series(0, index=close.index)
        signals[fast_ma > slow_ma] = 1
        signals[fast_ma < slow_ma] = -1
        return signals

    @staticmethod
    def from_rsi(close: pd.Series, window: int = 14,
                  oversold: float = 30, overbought: float = 70) -> pd.Series:
        """RSI信号"""
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_g = gain.rolling(window).mean()
        avg_l = loss.rolling(window).mean()
        rs = avg_g / avg_l.clip(lower=1e-12)
        rsi = 100 - 100 / (1 + rs)
        signals = pd.Series(0, index=close.index)
        signals[rsi < oversold] = 1
        signals[rsi > overbought] = -1
        return signals

    @staticmethod
    def from_macd(close: pd.Series, fast: int = 12, slow: int = 26,
                   signal_window: int = 9) -> pd.Series:
        """MACD 金叉死叉信号"""
        ema_f = close.ewm(span=fast).mean()
        ema_s = close.ewm(span=slow).mean()
        macd = ema_f - ema_s
        signal = macd.ewm(span=signal_window).mean()
        signals = pd.Series(0, index=close.index)
        signals[macd > signal] = 1
        signals[macd < signal] = -1
        return signals


# ══════════════════════════════════════
# 2. SignalIntegrator
# ══════════════════════════════════════

class SignalIntegrator:
    """信号集成器 - 多信号融合"""

    def __init__(self):
        self._signals: list[Signal] = []

    def add(self, signal: Signal):
        """添加信号"""
        self._signals.append(signal)

    def add_many(self, signals: list[Signal]):
        self._signals.extend(signals)

    def clear(self):
        self._signals.clear()

    def vote(self, threshold: float = 0.0) -> pd.Series:
        """投票集成：多数表决

        # P2-Q9-fix (Q9-M532): 原实现未使用 threshold 参数，且平票时
        # value_counts().index[0] 取首次出现值，结果任意。现实现阈值投票：
        # ① |信号值| < threshold 视为弃权（不投票）；
        # ② 净票数 = 做多票 - 做空票，净票>0 → +1，<0 → -1，=0 → 0（弃权）。
        """
        if not self._signals:
            return pd.Series(dtype=float)
        df = pd.DataFrame({s.name: s.values for s in self._signals if s.values is not None})
        if df.empty:
            return pd.Series(dtype=float)
        pos_votes = (df > threshold).sum(axis=1)
        neg_votes = (df < -threshold).sum(axis=1)
        net = pos_votes - neg_votes
        return pd.Series(np.sign(net), index=df.index, dtype=float)

    def weighted_average(self) -> pd.Series:
        """加权平均集成

        # P2-Q9-fix (Q9-L533): 原实现以 pd.Series(0.0)（索引 [0]）为起点逐信号
        # add(fill_value=0)，依赖索引并集对齐且逐次对齐效率低。现先统一对齐到
        # 并集索引（缺失按 0 填充），再向量化加权求和。
        """
        if not self._signals:
            return pd.Series(dtype=float)
        sigs = [s for s in self._signals if s.values is not None]
        total_w = sum(s.weight for s in sigs)
        if total_w == 0:
            return pd.Series(dtype=float)
        df = pd.DataFrame({s.name: s.values for s in sigs}).fillna(0.0)
        w = pd.Series({s.name: s.weight / total_w for s in sigs})
        return df.dot(w)

    def rank_average(self) -> pd.Series:
        """排名平均集成"""
        if not self._signals:
            return pd.Series(dtype=float)
        df = pd.DataFrame({s.name: s.values.rank(pct=True)
                          for s in self._signals if s.values is not None})
        if df.empty:
            return pd.Series(dtype=float)
        return df.mean(axis=1)

    @staticmethod
    def _score_signal(sig: pd.Series, r: pd.Series, metric: str) -> float:
        """计算单个信号在给定标签窗口上的得分（sharpe 或 corr），NaN 安全。"""
        if metric == "sharpe":
            prod = sig * r
            sr = prod.mean() / max(prod.std(), 1e-12) * np.sqrt(252)
        else:
            sr = sig.corr(r)
        return max(sr, 0) if not np.isnan(sr) else 0.0

    def best_signal(self, metric: str = "sharpe",
                     forward_returns: Optional[pd.Series] = None,
                     lookback: Optional[int] = 120) -> Signal:
        """选择最近 lookback 窗口内得分最高的信号 —— **仅限离线诊断**。

        # P1-Q9-fix (H04): 原实现用全样本 forward_returns 评分后 argmax，
        # 属 in-sample selection / 前视，用于上线决策或参数选择会高估信号
        # 有效性。现默认只在最近 lookback 个已实现标签内滚动评分，并在每次
        # 调用输出可见告警；线上安全选择请使用 best_signal_walk_forward()。
        """
        if not self._signals:
            raise ValueError("没有可用信号")
        if forward_returns is None:
            return self._signals[0]
        logger.warning(
            "best_signal() 为离线诊断工具（最近 %s 个样本滚动评分），"
            "禁止用于上线决策/参数选择；线上请用 best_signal_walk_forward()",
            lookback if lookback is not None else "全部",
        )
        fr = forward_returns.dropna()
        if lookback is not None and len(fr) > lookback:
            fr = fr.iloc[-lookback:]
        scores = []
        for s in self._signals:
            if s.values is None:
                continue
            common = s.values.dropna().index.intersection(fr.index)
            if len(common) < 10:
                scores.append(0)
            else:
                scores.append(self._score_signal(s.values.loc[common],
                                                 fr.loc[common], metric))
        best_idx = np.argmax(scores)
        return self._signals[best_idx]

    def best_signal_walk_forward(
        self,
        forward_returns: pd.Series,
        lookback: int = 120,
        embargo: int = 20,
        min_samples: int = 10,
        metric: str = "sharpe",
    ) -> pd.DataFrame:
        """Walk-forward 无前视最佳信号选择（上线决策安全版）。

        # P1-Q9-fix (H04): 在每个决策日 t，只用 t 之前（滚动窗口，且剔除
        # 最近 embargo 个尚未完全实现的标签，防前视）的已实现收益评分，
        # 取得分最高信号在 t 日的值作为决策，杜绝用未来数据选择信号。

        Args:
            forward_returns: 前向收益序列（标签），index 为交易日
            lookback: 滚动评分窗口大小（样本数）
            embargo: 决策日前剔除的标签样本数，须 >= 标签前视期
                     （如 20 日标签须 >= 20），避免未实现收益进入评分窗
            min_samples: 评分所需最少样本
            metric: "sharpe" | "corr"

        Returns:
            DataFrame(index=决策日, columns=[chosen_signal, chosen_value, score]
                      及每个信号的评分列)
        """
        if not self._signals:
            raise ValueError("没有可用信号")
        fr = forward_returns.dropna()
        if len(fr) < min_samples:
            raise ValueError(f"forward_returns 样本不足: {len(fr)} < {min_samples}")

        idx = list(fr.index)
        rows = []
        for p, t in enumerate(idx):
            hist_end = p - embargo
            if hist_end < min_samples:
                continue  # 历史标签不足，不做决策
            hist_idx = idx[max(0, hist_end - lookback):hist_end]

            scores: dict[str, float] = {}
            for s in self._signals:
                if s.values is None:
                    continue
                common = s.values.dropna().index.intersection(hist_idx)
                if len(common) < min_samples:
                    scores[s.name] = 0.0
                    continue
                sc = self._score_signal(s.values.loc[common], fr.loc[common], metric)
                scores[s.name] = round(float(sc), 6)

            if not scores:
                continue
            best_name = max(scores, key=lambda k: scores[k])
            # 取该信号在决策日 t 的值（缺失记为 0）
            chosen_val = 0.0
            for s in self._signals:
                if s.name == best_name and s.values is not None:
                    v = s.values.get(t, None)
                    if pd.notna(v):
                        chosen_val = float(v)
                    break
            rows.append({"date": t, "chosen_signal": best_name,
                         "chosen_value": chosen_val, "score": scores[best_name],
                         **scores})

        if not rows:
            return pd.DataFrame(columns=["date", "chosen_signal",
                                         "chosen_value", "score"])
        return pd.DataFrame(rows).set_index("date")

    def list_signals(self) -> list[Signal]:
        return self._signals


# ══════════════════════════════════════
# 3. SignalOptimizer
# ══════════════════════════════════════

class SignalOptimizer:
    """信号优化器 - 信号后处理"""

    @staticmethod
    def threshold(signals: pd.Series, entry_threshold: float = 0.5,
                   exit_threshold: float = 0.2) -> pd.Series:
        """阈值过滤（滞回）

        # P2-Q9-fix (Q9-L531): exit_threshold 原先定义了但从未使用，只按
        # entry_threshold 双向截断。现实现滞回逻辑：空仓时信号越过
        # ±entry_threshold 才开仓；持仓后信号回落到 ±exit_threshold 以内才平仓，
        # 消除信号在阈值附近反复横跳。exit_threshold 应 <= entry_threshold，
        # 否则钳制为 entry_threshold。
        """
        if exit_threshold > entry_threshold:
            exit_threshold = entry_threshold
        optimized = pd.Series(0, index=signals.index, dtype=float)
        pos = 0
        for i in range(len(signals)):
            s = signals.iloc[i]
            if pd.isna(s):
                pos = 0  # 缺失信号视为中性（平仓），保持原向量化语义
            elif pos == 0:
                if s > entry_threshold:
                    pos = 1
                elif s < -entry_threshold:
                    pos = -1
            else:
                if pos == 1 and s < exit_threshold:
                    pos = 0
                elif pos == -1 and s > -exit_threshold:
                    pos = 0
            optimized.iloc[i] = float(pos)
        return optimized

    @staticmethod
    def smooth(signals: pd.Series, window: int = 3) -> pd.Series:
        """信号平滑

        Q9-fix: center=True 的中心窗口均值会引入未来 window//2 根 K 线数据
        （前视偏差，实测信号 [0,0,0,1,0,0,0] 平滑后索引 2 出现 0.333）。
        改为 center=False（只用 t 及之前的数据）。
        """
        return signals.rolling(window, min_periods=1, center=False).mean()

    @staticmethod
    def min_holding(signals: pd.Series, min_days: int = 5) -> pd.Series:
        """最小持仓期约束

        # P1-Q9-fix (H02): 修复两个逻辑错误（输入
        # [1,1,1,1,1,0,0,0,-1,-1,0,1,1] 旧输出
        # [1,1,1,1,1,1,1,1,0,-1,-1,-1,-1]）：
        # ① 中性信号(0)在持仓期满后视为平仓信号，不再被无限持有；
        # ② 反向信号日输出 -pos（立即反手），不再被压成 0 丢失反转 1 天；
        # ③ 入口日显式写 optimized[i] = pos，不依赖 copy 的初始值。
        """
        optimized = signals.copy()
        pos = 0
        count = 0  # 当前持仓已持续的天数（进入后逐日累加）
        for i in range(len(signals)):
            sig = signals.iloc[i]
            if pos == 0:
                if sig != 0:
                    pos = sig
                    count = 0
                    optimized.iloc[i] = pos  # 入口日显式写入
            else:
                count += 1
                if count >= min_days and sig != pos:
                    # 持仓期满后：中性(0)=平仓；反向(-pos)=反手
                    pos = sig
                optimized.iloc[i] = pos
        return optimized

    @staticmethod
    def max_open(signals: pd.DataFrame, max_signals: int = 10) -> pd.DataFrame:
        """最大同时持仓数限制

        # P1-Q9-fix (H03): 原实现用 signals.cumsum()（逐资产按时间累计）
        # 判断超限，而"最大同时持仓数"应为同一时点横截面持仓计数——
        # 对 ±1 持仓信号逐列累计无法反映当日持仓数量，限制形同虚设。
        # 改为逐行统计非零持仓数（多头/空头均为持仓），超限时 mask 当日信号。
        """
        if max_signals <= 0:
            raise ValueError(f"max_signals 必须为正整数，got {max_signals}")
        # 同一时点横截面持仓计数：非零信号（±1 均计为持仓）数
        if isinstance(signals, pd.Series):
            open_count = signals.ne(0)
            open_count = open_count.astype(int)
        else:
            open_count = signals.ne(0).sum(axis=1)
        over = open_count > max_signals
        return signals.mask(over, 0)

    @staticmethod
    def combine_conditions(signals_list: list[pd.Series],
                            operator: str = "and") -> pd.Series:
        """多条件组合"""
        df = pd.concat(signals_list, axis=1)
        if operator == "and":
            return df.all(axis=1).astype(int)
        elif operator == "or":
            return df.any(axis=1).astype(int)
        elif operator == "majority":
            return (df.sum(axis=1) > df.shape[1] / 2).astype(int)
        return df.iloc[:, 0]


# ══════════════════════════════════════
# 4. SignalGenerator — 顶层整合
# ══════════════════════════════════════

class SignalGenerator:
    """综合信号生成器"""

    def __init__(self):
        self.factory = SignalFactory()
        self.integrator = SignalIntegrator()
        self.optimizer = SignalOptimizer()

    def generate(self, close: pd.Series, method: str = "momentum",
                  **kwargs) -> pd.Series:
        """生成信号

        # P1-Q9-fix (H01): **kwargs 原先整体透传给工厂方法，导致后处理参数
        # (entry_threshold/min_holding/smooth_window) 传入时直接 TypeError
        # （实测 generate(close,'momentum',entry_threshold=0.5) 崩溃）。
        # 现在先按 method 白名单提取工厂参数，再单独消费后处理参数。
        """
        # 每个 method 的工厂参数白名单
        method_map = {
            "momentum": (self.factory.from_momentum, {"lookback"}),
            "mean_reversion": (self.factory.from_mean_reversion, {"window", "n_std"}),
            "ma_cross": (self.factory.from_moving_average_cross, {"fast", "slow"}),
            "rsi": (self.factory.from_rsi, {"window", "oversold", "overbought"}),
            "macd": (self.factory.from_macd, {"fast", "slow", "signal_window"}),
        }

        if method not in method_map:
            raise ValueError(f"未知方法: {method}，可选: {list(method_map.keys())}")

        factory_fn, factory_params = method_map[method]

        # 1) 仅透传该 method 白名单内的工厂参数
        factory_kwargs = {k: v for k, v in kwargs.items() if k in factory_params}
        raw = factory_fn(close, **factory_kwargs)

        # 2) 单独消费后处理参数（不再进入工厂方法）
        if "entry_threshold" in kwargs or "exit_threshold" in kwargs:
            raw = self.optimizer.threshold(
                raw,
                entry_threshold=kwargs.get("entry_threshold", 0.5),
                exit_threshold=kwargs.get("exit_threshold", 0.2),
            )
        if "min_holding" in kwargs:
            raw = self.optimizer.min_holding(raw, kwargs["min_holding"])
        if "smooth_window" in kwargs:
            raw = self.optimizer.smooth(raw, kwargs["smooth_window"])

        # 3) 未消费参数：可见告警，不静默吞掉
        known = factory_params | {"entry_threshold", "exit_threshold",
                                  "min_holding", "smooth_window"}
        unknown = set(kwargs) - known
        if unknown:
            logger.warning("generate(%s) 收到未消费参数: %s（已忽略）",
                           method, sorted(unknown))

        return raw

    def multi_signal_alpha(self, close: pd.Series, methods: list[str]) -> pd.Series:
        """多方法合成Alpha"""
        self.integrator.clear()
        for i, method in enumerate(methods):
            sig = self.generate(close, method)
            self.integrator.add(Signal(name=method, values=sig, method=method))
        return self.integrator.rank_average()

    def from_factor_model(self, factors: pd.DataFrame, top_pct: float = 0.2) -> pd.Series:
        """因子模型信号"""
        return self.factory.from_factor_rank(factors, top_pct)

    def ensemble(self, methods: list[str], close: pd.Series,
                  weights: Optional[list[float]] = None) -> pd.Series:
        """集成信号"""
        self.integrator.clear()
        for i, method in enumerate(methods):
            sig = self.generate(close, method)
            w = weights[i] if weights else 1.0
            self.integrator.add(Signal(name=method, values=sig, method=method, weight=w))
        return self.integrator.weighted_average()

    def pipeline(self, close: pd.Series,
                  stages: list[dict]) -> pd.Series:
        """信号管道：多阶段处理
        
        stages: [
            {"method": "momentum", "args": {"lookback": 20}},
            {"method": "optimize", "name": "threshold",
             "args": {"entry_threshold": 0.3, "exit_threshold": 0.1}},
            {"method": "optimize", "name": "min_holding", "args": {"min_days": 3}},
        ]
        # P2-Q9-fix (Q9-M530): 文档示例原先用 {"entry": 0.3}，与 threshold()
        # 签名 (entry_threshold/exit_threshold) 不匹配，按文档调用会
        # TypeError。已统一为 entry_threshold/exit_threshold。
        """
        signals = close.copy() * 0
        for stage in stages:
            if stage["method"] == "optimize":
                opt_name = stage.get("name", "threshold")
                opt_fn = getattr(self.optimizer, opt_name, None)
                if opt_fn:
                    signals = opt_fn(signals, **stage.get("args", {}))
            else:
                signals = self.generate(close, stage["method"],
                                        **stage.get("args", {}))
        return signals


__all__ = [
    "Signal", "SignalMethod",
    "SignalFactory", "SignalIntegrator",
    "SignalOptimizer", "SignalGenerator",
]
