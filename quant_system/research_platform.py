"""
research_platform.py — 策略研究与回测管理平台
V4.1 feature

提供：策略模板库、参数优化框架、策略对比、模拟部署、一站式研究入口
"""

import os
import re
import json
import uuid
import copy
import logging
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from sklearn.model_selection import ParameterGrid
    _HAS_SKL = True
except ImportError:
    _HAS_SKL = False

try:
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ══════════════════════════════════════
# 1. StrategyTemplate
# ══════════════════════════════════════

class StrategyTemplate:
    """策略模板基类"""

    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__
        self.params = {}

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成交易信号，返回 -1/0/1 的 Series"""
        raise NotImplementedError

    def clone(self, **new_params) -> "StrategyTemplate":
        """克隆并更新参数"""
        obj = copy.deepcopy(self)
        obj.params.update(new_params)
        return obj

    def __repr__(self):
        return f"{self.name}(params={self.params})"


class MovingAverageCross(StrategyTemplate):
    """双均线交叉策略"""

    def __init__(self, fast: int = 5, slow: int = 20):
        super().__init__("MA_Cross")
        self.params = {"fast": fast, "slow": slow}

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"] if "close" in data else data.iloc[:, 0]
        fast_ma = close.rolling(self.params["fast"]).mean()
        slow_ma = close.rolling(self.params["slow"]).mean()
        signals = pd.Series(0, index=data.index)
        signals[fast_ma > slow_ma] = 1
        signals[fast_ma < slow_ma] = -1
        return signals


class MeanReversion(StrategyTemplate):
    """均值回归策略 (布林带)"""

    def __init__(self, window: int = 20, n_std: float = 2.0):
        super().__init__("Mean_Reversion")
        self.params = {"window": window, "n_std": n_std}

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"] if "close" in data else data.iloc[:, 0]
        ma = close.rolling(self.params["window"]).mean()
        std = close.rolling(self.params["window"]).std(ddof=0)
        upper = ma + self.params["n_std"] * std
        lower = ma - self.params["n_std"] * std
        signals = pd.Series(0, index=data.index)
        signals[close > upper] = -1  # 上轨卖出
        signals[close < lower] = 1   # 下轨买入
        return signals


class MomentumStrategy(StrategyTemplate):
    """截面动量策略"""

    def __init__(self, lookback: int = 20, top_pct: float = 0.2):
        super().__init__("Momentum")
        self.params = {"lookback": lookback, "top_pct": top_pct}

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        ret = data["close"].pct_change(self.params["lookback"]) if "close" in data else data.iloc[:, 0].pct_change(self.params["lookback"])
        # P2-Q25-fix(M297): 原实现 `signals[ret < -threshold]` 假设收益分布对称,
        # 非对称分布下多空信号不对等。现独立计算正负两端分位数。
        long_threshold = ret.quantile(1 - self.params["top_pct"])
        short_threshold = ret.quantile(self.params["top_pct"])
        signals = pd.Series(0, index=data.index)
        signals[ret > long_threshold] = 1
        signals[ret < short_threshold] = -1
        return signals


class PairTrading(StrategyTemplate):
    """配对交易策略"""

    def __init__(self, symbol_a: str = "", symbol_b: str = "",
                 window: int = 60, entry_z: float = 2.0, exit_z: float = 0.5):
        super().__init__("Pair_Trading")
        self.params = {"window": window, "entry_z": entry_z, "exit_z": exit_z}
        self.symbol_a = symbol_a
        self.symbol_b = symbol_b

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # 需要两列: symbol_a 和 symbol_b
        if self.symbol_a not in data.columns or self.symbol_b not in data.columns:
            return pd.Series(0, index=data.index)
        a, b = data[self.symbol_a], data[self.symbol_b]
        ratio = a / b
        z = (ratio - ratio.rolling(self.params["window"]).mean()) / ratio.rolling(self.params["window"]).std(ddof=0).clip(lower=1e-12)
        signals = pd.Series(0, index=data.index)
        signals[z > self.params["entry_z"]] = -1  # ratio高，空A多B
        signals[z < -self.params["entry_z"]] = 1   # ratio低，多A空B
        signals[abs(z) < self.params["exit_z"]] = 0  # 回归平仓
        return signals


# ══════════════════════════════════════
# 2. ParameterOptimizer
# ══════════════════════════════════════

class ParameterOptimizer:
    """参数优化器"""

    @staticmethod
    def grid_search(strategy_class, param_grid: dict, data: pd.DataFrame,
                    verbose: bool = True, metric: str = "sharpe") -> dict:
        """网格搜索"""
        best_score = -np.inf
        best_params = {}
        results = []

        for values in ParameterGrid(param_grid):
            try:
                strategy = strategy_class(**values)
                signals = strategy.generate_signals(data)
                score = _evaluate_strategy(signals, data, metric)
                results.append({**values, metric: score})
                if score > best_score:
                    best_score = score
                    best_params = values
            except Exception as e:
                logger.warning(f"参数 {values} 评估失败: {e}")

        result_df = pd.DataFrame(results).sort_values(metric, ascending=False)
        return {
            "best_params": best_params,
            "best_score": best_score,
            "results": result_df,
            "n_combinations": len(results),
        }

    @staticmethod
    def random_search(strategy_class, param_dist: dict, data: pd.DataFrame,
                      n_iter: int = 100, metric: str = "sharpe",
                      random_state: int = 42) -> dict:
        """随机搜索"""
        rng = np.random.RandomState(random_state)
        best_score = -np.inf
        best_params = {}
        results = []

        for _ in range(n_iter):
            # P2-Q25-fix(L306): tuple 也识别为候选列表(原仅 list/ndarray,
            # tuple 被静默按常值处理不参与搜索)。
            params = {k: rng.choice(v) if isinstance(v, (list, tuple, np.ndarray))
                      else v for k, v in param_dist.items()}
            try:
                strategy = strategy_class(**params)
                signals = strategy.generate_signals(data)
                score = _evaluate_strategy(signals, data, metric)
                results.append({**params, metric: score})
                if score > best_score:
                    best_score = score
                    best_params = params
            except Exception as e:
                logger.error(f"[research_platform] 操作失败: {e}", exc_info=True)
                continue

        return {
            "best_params": best_params,
            "best_score": best_score,
            "results": pd.DataFrame(results).sort_values(metric, ascending=False),
            "n_combinations": len(results),
        }

    @staticmethod
    def bayesian_optimize(strategy_class, param_space: dict, data: pd.DataFrame,
                          metric: str = "sharpe", n_iter: int = 50) -> dict:
        """贝叶斯优化（使用简单 GP：随机初始化 -> GP拟合 -> EI采集）"""
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern
        except ImportError:
            logger.warning("sklearn GP不可用，回退到随机搜索")
            # P1-Q25-fix: H01 回退调用改为关键字传参，修复参数错位
            # （原调用把 n_iter 传给 data、data 传给 n_iter，DataFrame 被当作整数引发 TypeError）
            return ParameterOptimizer.random_search(strategy_class, param_space,
                                                    data=data, n_iter=n_iter, metric=metric)

        # P1-Q25-fix: H02 补齐 GP 拟合 + EI 采集循环（原实现仅初始随机采样即返回，n_iter 大部分被忽略）
        rng = np.random.RandomState(42)

        # 参数空间解析：区间[low,high]按均匀采样，离散列表按choice采样，标量为固定值
        param_defs = []
        for k, v in param_space.items():
            if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
                param_defs.append((k, "range", (float(v[0]), float(v[1]))))
            elif isinstance(v, (list, tuple, np.ndarray)):
                param_defs.append((k, "choice", list(v)))
            else:
                param_defs.append((k, "fixed", v))

        def _sample_params() -> dict:
            params = {}
            for k, kind, dom in param_defs:
                if kind == "range":
                    params[k] = rng.uniform(dom[0], dom[1])
                elif kind == "choice":
                    params[k] = rng.choice(dom)
                else:
                    params[k] = dom
            return params

        def _encode(params: dict) -> list:
            vec = []
            for k, kind, dom in param_defs:
                if kind == "range":
                    vec.append(float(params[k]))
                elif kind == "choice":
                    vec.append(dom.index(params[k]))
                else:
                    vec.append(0.0)
            return vec

        def _decode(vec: list) -> dict:
            params = {}
            for (k, kind, dom), x in zip(param_defs, vec):
                if kind == "range":
                    params[k] = x
                elif kind == "choice":
                    params[k] = dom[int(round(x))]
                else:
                    params[k] = dom
            return params

        def _evaluate(params: dict) -> float:
            strategy = strategy_class(**params)
            signals = strategy.generate_signals(data)
            return _evaluate_strategy(signals, data, metric)

        # 初始随机采样
        n_initial = min(10, n_iter // 5)
        X_obs: list = []
        y_obs: list = []
        for _ in range(n_initial):
            params = _sample_params()
            try:
                y_obs.append(_evaluate(params))
                X_obs.append(_encode(params))
            except Exception as e:
                logger.warning(f"bayesian_optimize 初始采样参数 {params} 评估失败: {e}")

        if not y_obs:
            return {"best_score": 0.0, "best_params": {}, "n_observations": 0}

        # EI 采集函数（期望改进）；scipy.stats 不可用时降级为置信下界并给出可见告警
        try:
            from scipy.stats import norm as _norm
        except ImportError:
            logger.warning("bayesian_optimize: scipy.stats 不可用，EI 降级为置信下界")
            _norm = None

        def _expected_improvement(mu, sigma, y_best):
            sigma = np.maximum(sigma, 1e-12)
            z = (y_best - mu) / sigma
            if _norm is not None:
                return (y_best - mu) * _norm.cdf(z) + sigma * _norm.pdf(z)
            return -sigma  # 退化兜底：优先低不确定度区域

        # GP 拟合 + EI 采集循环，消耗全部 n_iter
        for _ in range(n_initial, n_iter):
            gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-6,
                                          normalize_y=True, random_state=42)
            gp.fit(np.array(X_obs), np.array(y_obs))

            # 随机候选池上计算 EI，选取期望改进最大的候选点评估
            candidates = [_sample_params() for _ in range(100)]
            X_cand = np.array([_encode(c) for c in candidates])
            mu, sigma = gp.predict(X_cand, return_std=True)
            ei = _expected_improvement(mu, sigma, float(max(y_obs)))
            ei_max = float(np.nanmax(ei))
            if np.isnan(ei_max) or ei_max <= 0:
                break  # 无可改进候选（或 EI 退化），提前终止
            params = candidates[int(np.nanargmax(ei))]
            try:
                y_obs.append(_evaluate(params))
                X_obs.append(_encode(params))
            except Exception as e:
                logger.warning(f"bayesian_optimize 迭代参数 {params} 评估失败: {e}")
                continue

        best_i = int(np.argmax(y_obs))
        return {
            "best_score": float(y_obs[best_i]),
            "best_params": _decode(X_obs[best_i]),
            "n_observations": len(y_obs),
        }

    @staticmethod
    def parameter_stability(strategy_class, param_grid: dict,
                            data: pd.DataFrame, subperiods: int = 6) -> pd.DataFrame:
        """参数稳定性测试"""
        n = len(data)
        chunk = n // subperiods
        rows = []
        for i in range(subperiods):
            start = i * chunk
            end = start + chunk if i < subperiods - 1 else n
            sub_data = data.iloc[start:end]
            result = ParameterOptimizer.grid_search(strategy_class, param_grid, sub_data, verbose=False)
            rows.append({
                "period": i + 1,
                "start": data.index[start],
                "end": data.index[min(end, n) - 1],
                "best_params": str(result["best_params"]),
                "best_score": result["best_score"],
            })
        return pd.DataFrame(rows)


# ══════════════════════════════════════
# 3. StrategyComparison
# ══════════════════════════════════════

class StrategyComparison:
    """策略对比框架"""

    @staticmethod
    def compare(strategies: list, data: pd.DataFrame,
                metrics: Optional[list] = None) -> pd.DataFrame:
        """多策略横向对比"""
        if metrics is None:
            metrics = ["sharpe", "calmar", "total_return", "max_dd", "win_rate", "n_trades"]

        results = []
        for strategy in strategies:
            try:
                signals = strategy.generate_signals(data)
                stats = _full_stats(signals, data)
                row = {"name": strategy.name}
                for m in metrics:
                    row[m] = stats.get(m, 0)
                results.append(row)
            except Exception as e:
                logger.error(f"{strategy.name} 评估失败: {e}")

        return pd.DataFrame(results).sort_values("sharpe", ascending=False) if results else pd.DataFrame()

    @staticmethod
    def correlation_report(results: dict) -> pd.DataFrame:
        """策略收益相关性"""
        returns = pd.DataFrame(results)
        return returns.corr()

    @staticmethod
    def rank_strategies(comparison_df: pd.DataFrame,
                        weights: dict = None) -> pd.Series:
        """综合排名"""
        if weights is None:
            weights = {"sharpe": 0.4, "calmar": 0.3, "max_dd": 0.3}

        scores = pd.Series(0.0, index=comparison_df.index)
        for metric, w in weights.items():
            if metric in comparison_df.columns:
                # 归一化
                col = comparison_df[metric]
                norm = (col - col.min()) / max(col.max() - col.min(), 1e-12)
                scores += w * norm

        return scores

    @staticmethod
    def portfolio_of_strategies(results: dict, method: str = "equal",
                                target_vol: float = 0.15) -> dict:
        """策略组合构建"""
        returns = pd.DataFrame(results)
        # P2-Q25-fix(M301): 空数据校验, 避免 n=0 除零与 sharpe 为 NaN。
        if returns.empty or len(returns.columns) == 0:
            return {"weights": {}, "portfolio_return": pd.Series(dtype=float), "sharpe": 0.0}
        n = len(returns.columns)

        if method == "equal":
            weights = {col: 1.0 / n for col in returns.columns}
        elif method == "risk_parity":
            vols = (returns.std() * np.sqrt(252)).replace(0, np.nan)
            valid = vols.dropna()
            # P2-Q25-fix(M301): 全 0/NaN 波动(无有效波动)时回退等权, 避免权重全 0。
            if valid.empty or not np.isfinite(valid).all():
                weights = {col: 1.0 / n for col in returns.columns}
            else:
                inv_vol = 1.0 / valid.clip(lower=1e-12)
                total = float(inv_vol.sum())
                weights = {col: float(inv_vol.get(col, 0.0) / total) for col in returns.columns}
        else:
            weights = {col: 1.0 / n for col in returns.columns}

        portfolio_ret = returns.dot(pd.Series(weights))
        sharpe = float(portfolio_ret.mean() / max(portfolio_ret.std(), 1e-12) * np.sqrt(252))
        # P2-Q25-fix(M301): 空/单点收益序列时 sharpe 非有限, 回退为 0。
        if not np.isfinite(sharpe):
            sharpe = 0.0
        return {
            "weights": weights,
            "portfolio_return": portfolio_ret,
            "sharpe": sharpe,
        }


# ══════════════════════════════════════
# 4. SimulationDeployer
# ══════════════════════════════════════

class SimulationDeployer:
    """模拟部署器"""

    def __init__(self, sim_dir: str = ""):
        self.sim_dir = sim_dir or os.path.expanduser("~/.quant_system/simulations/")
        os.makedirs(self.sim_dir, exist_ok=True)
        self._running: dict[str, dict] = {}

    def deploy(self, strategy: StrategyTemplate, name: str,
               start_date: str, end_date: str, initial_cash: float = 1e6) -> str:
        """部署策略到模拟环境"""
        sim_id = f"sim_{uuid.uuid4().hex[:8]}"
        config = {
            "sim_id": sim_id,
            "name": name,
            "strategy": strategy.__class__.__name__,
            "params": strategy.params,
            "start_date": start_date,
            "end_date": end_date,
            "initial_cash": initial_cash,
            "created_at": datetime.now().isoformat(),
            "status": "running",
        }
        self._save_config(sim_id, config)
        self._running[sim_id] = config
        logger.info(f"[SimDeploy] 部署 {name} -> {sim_id}")
        return sim_id

    def stop(self, sim_id: str) -> bool:
        """停止模拟"""
        if sim_id in self._running:
            self._running[sim_id]["status"] = "stopped"
            self._save_config(sim_id, self._running[sim_id])
            return True
        return False

    def status(self, sim_id: str) -> dict:
        """模拟运行状态"""
        config = self._load_config(sim_id)
        if config is None:
            return {"error": "not_found"}
        return {
            "sim_id": sim_id,
            "name": config.get("name", ""),
            "status": config.get("status", "unknown"),
            "created_at": config.get("created_at", ""),
        }

    def compare_live_vs_backtest(self, sim_id: str, live_returns: pd.Series,
                                  bt_returns: pd.Series) -> dict:
        """实盘 vs 回测差异分析"""
        common = live_returns.index.intersection(bt_returns.index)
        if len(common) < 5:
            return {"error": "数据不足"}
        corr = live_returns.loc[common].corr(bt_returns.loc[common])
        tracking_error = (live_returns.loc[common] - bt_returns.loc[common]).std() * np.sqrt(252)
        return {
            "correlation": corr,
            "tracking_error": tracking_error,
            "live_sharpe": live_returns.mean() / max(live_returns.std(), 1e-12) * np.sqrt(252),
            "bt_sharpe": bt_returns.mean() / max(bt_returns.std(), 1e-12) * np.sqrt(252),
        }

    def _save_config(self, sim_id: str, config: dict):
        path = os.path.join(self.sim_dir, f"{sim_id}.json")
        with open(path, "w") as f:
            json.dump(config, f, indent=2)

    def _load_config(self, sim_id: str) -> Optional[dict]:
        path = os.path.join(self.sim_dir, f"{sim_id}.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)


# ══════════════════════════════════════
# 5. ResearchPlatform — 顶层整合
# ══════════════════════════════════════

class ResearchPlatform:
    """策略研究平台顶层整合"""

    def __init__(self):
        self.templates = {
            "ma_cross": MovingAverageCross,
            "mean_reversion": MeanReversion,
            "momentum": MomentumStrategy,
            "pair_trading": PairTrading,
        }
        self.optimizer = ParameterOptimizer()
        self.comparison = StrategyComparison()
        self.deployer = SimulationDeployer()

    def run_research(self, template_name: str, param_grid: dict,
                     data: pd.DataFrame, method: str = "grid") -> dict:
        """一站式策略研究"""
        if template_name not in self.templates:
            return {"error": f"未知模板: {template_name}"}

        strategy_class = self.templates[template_name]

        if method == "grid":
            result = self.optimizer.grid_search(strategy_class, param_grid, data)
        elif method == "random":
            result = self.optimizer.random_search(strategy_class, param_grid, data)
        else:
            result = self.optimizer.grid_search(strategy_class, param_grid, data)

        # 用最优参数做完整评估
        best_strategy = strategy_class(**result["best_params"])
        signals = best_strategy.generate_signals(data)
        stats = _full_stats(signals, data)

        result["strategy"] = template_name
        result["stats"] = stats
        return result

    def list_strategies(self) -> list[dict]:
        """列出所有可用策略模板"""
        return [{"name": k, "class": v.__name__} for k, v in self.templates.items()]

    def compare_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """所有模板对比"""
        strategies = [cls() for cls in self.templates.values()]
        return self.comparison.compare(strategies, data)

    def generate_research_report(self, strategy_name: str,
                                  result: dict) -> str:
        """生成可读的研究报告"""
        stats = result.get("stats", {})
        lines = [
            "=" * 55,
            f"策略研究报告: {strategy_name}",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 55,
            "",
            f"最优参数: {result.get('best_params', {})}",
            f"搜索方法: {result.get('strategy', 'grid')}",
            f"搜索组合数: {result.get('n_combinations', 0)}",
            "",
            "【绩效指标】",
            f"  年化夏普: {stats.get('sharpe', 0):.3f}",
            f"  总收益: {stats.get('total_return', 0)*100:.2f}%",
            f"  年化波动: {stats.get('ann_vol', 0)*100:.2f}%",
            f"  最大回撤: {stats.get('max_dd', 0)*100:.2f}%",
            f"  Calmar比: {stats.get('calmar', 0):.2f}",
            f"  胜率: {stats.get('win_rate', 0)*100:.1f}%",
        ]
        return "\n".join(lines)


# ══════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════

def _evaluate_strategy(signals: pd.Series, data: pd.DataFrame,
                        metric: str = "sharpe") -> float:
    """评估策略绩效"""
    stats = _full_stats(signals, data)
    return stats.get(metric, 0)


def _board_daily_limit(data: pd.DataFrame, close: pd.Series) -> Optional[float]:
    """推断 A 股日涨跌幅限制：主板10% / 创业板·科创板20% / 北交所30% / ST5%。

    从数据中解析证券代码（常见代码列，或 close/索引名中的6位数字）；无法确定代码
    时返回 None，表示不做收益截断（真实涨跌停约束由交易/回测引擎层处理）。
    """
    code = None
    for col in ("symbol", "code", "ticker", "stock_code", "sec_code", "security_code"):
        if col in data.columns and len(data):
            v = data[col].iloc[0]
            if v is not None:
                code = str(v)
                break
    if code is None:
        for name in (getattr(close, "name", None), getattr(data.index, "name", None)):
            if name is None:
                continue
            m = re.search(r"(\d{6})", str(name))
            if m:
                code = m.group(1)
                break
    if code is None:
        return None

    # ST 股 5%（若名称列携带 ST 标记）
    if "name" in data.columns and len(data) and "ST" in str(data["name"].iloc[0]).upper():
        return 0.05

    if code.startswith(("688", "689")):            # 科创板 20%
        return 0.20
    if code.startswith(("300", "301", "302")):     # 创业板 20%
        return 0.20
    if code.startswith(("4", "8", "92")):          # 北交所 30%
        return 0.30
    return 0.10                                    # 主板 10%


def _full_stats(signals: pd.Series, data: pd.DataFrame) -> dict:
    """计算策略完整绩效"""
    close = data["close"] if "close" in data else data.iloc[:, 0]
    ret = close.pct_change().fillna(0)

    # 策略收益 = 信号 * 收益（信号滞后一期避免前视）
    strategy_ret = signals.shift(1).fillna(0) * ret
    # P1-Q25-fix: H03 按板块动态限制日收益（主板10%/创业·科创20%/北交所30%/ST5%），
    # 无法识别代码时不截断——原固定 ±10% 会把创业板/科创板 20% 涨跌幅收益人为截断
    _limit = _board_daily_limit(data, close)
    if _limit is not None:
        strategy_ret = strategy_ret.clip(-_limit, _limit)

    ann_ret = strategy_ret.mean() * 252
    ann_vol = strategy_ret.std() * np.sqrt(252)
    sharpe = ann_ret / max(ann_vol, 1e-12)

    cum = (1 + strategy_ret).cumprod()
    peak = cum.expanding().max()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    calmar = ann_ret / max(abs(max_dd), 1e-12)
    win_rate = (strategy_ret > 0).mean()
    # P2-Q25-fix(L307): 按信号变化幅度计数 — 0→±1 计 1 笔, ±1→∓1 直接翻转
    # 计 2 笔(卖出+买入)。原 `abs()>0` 布尔计数把翻转仅计 1 次。
    n_trades = int(signals.diff().abs().fillna(0).sum())

    return {
        "sharpe": sharpe,
        "annualized_return": ann_ret,
        "ann_vol": ann_vol,
        "total_return": cum.iloc[-1] - 1,
        "max_dd": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
        "n_trades": n_trades,
    }


__all__ = [
    "StrategyTemplate", "MovingAverageCross", "MeanReversion",
    "MomentumStrategy", "PairTrading",
    "ParameterOptimizer", "StrategyComparison",
    "SimulationDeployer", "ResearchPlatform",
]
