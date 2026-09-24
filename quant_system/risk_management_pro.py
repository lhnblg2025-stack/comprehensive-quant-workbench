"""
risk_management_pro.py — V4.1 feature: 专业风控管理引擎

本模块面向 A 股量化研究与交易系统，提供组合层面的压力测试、流动性风险、
合规检查、实时风控监控与日报生成能力。设计目标是“可直接落地、可降级运行、
可被上层 CLI / Web / 交易执行模块复用”。

核心设计原则（V4.1 feature）:
  1. 风控指标统一以“损失为正数”的口径输出，便于限额比较与告警。
  2. 所有外部依赖均做 try/except 保护，缺少行情或 akshare 时使用保守默认值。
  3. scipy 作为基础科学计算依赖；sklearn 仅用于可选 PCA，加速失败不影响主流程。
  4. 返回值保留机器可读字段，同时报告生成器输出中文文本，便于人工复核。
  5. 每个关键计算块均有中文注释，避免后续维护人员误读风险方向与单位。

注意:
  - 本模块不直接下单，不修改账户状态，只提供风险识别、估算、限额与建议。
  - 历史场景为模板化冲击框架；若 returns_df 覆盖场景日期，会优先使用真实历史收益。
  - 流动性模型默认按成交额参与率估算，实际交易应结合盘口深度与执行算法复核。

D6收敛登记 (2026-08-11): 风控域收敛（保守策略）——
  1. VaR/CVaR 历史模拟族跨模块异名异签名异口径，全部标注保留、不强迁：
     _historical_var_cvar(损失数组→(var,cvar)小数, np.isfinite过滤) vs
     portfolio_risk.compute_var(akshare自取数+日期对齐联合分布+百分数多档输出) vs
     portfolio_v2._historical_cvar_from_returns(收益序列→cvar, np.nan_to_num) vs
     portfolio_optimizer.cvar_optimize(RU优化, α=尾部概率)。
  2. 本模块 StressTestEngine / LiquidityRiskManager / ComplianceChecker /
     RealTimeRiskMonitor / RiskReportGenerator = 独特能力（压力测试/流动性/合规/
     实时状态机与分级预警/日报），无等价实现，不强迁。
"""

from __future__ import annotations
import logging

import json
import math
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from quant_system.utils import safe_float as _safe_float_impl

# scipy 是基础依赖，但仍保留降级分支，方便在极简环境下做语法检查或离线演示。
try:  # V4.1 feature: scipy 基础依赖保护
    from scipy import optimize, stats

    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover - 仅在缺少 scipy 的运行环境触发
    linalg = None  # type: ignore[assignment]
    optimize = None  # type: ignore[assignment]
    stats = None  # type: ignore[assignment]
    SCIPY_AVAILABLE = False

# sklearn 是可选依赖；反压力测试中若可用则使用 PCA，否则回退到 numpy 特征分解。
try:  # V4.1 feature: sklearn 可选依赖保护
    from sklearn.decomposition import PCA

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - 缺少 sklearn 时正常降级
    PCA = None  # type: ignore[assignment]
    SKLEARN_AVAILABLE = False

# market_impact.py 是同项目模块，可能在脚本模式或包模式下导入路径不同。
try:  # V4.1 feature: 优先复用 Almgren-Chriss 市场冲击模型
    from quant_system.market_impact import ACParams, MarketImpact

    MARKET_IMPACT_AVAILABLE = True
except Exception:  # pragma: no cover - 单文件调试时可能触发
    try:
        from market_impact import ACParams, MarketImpact  # type: ignore

        MARKET_IMPACT_AVAILABLE = True
    except Exception:
        ACParams = None  # type: ignore[assignment]
        MarketImpact = None  # type: ignore[assignment]
        MARKET_IMPACT_AVAILABLE = False


# ---------------------------------------------------------------------------
# 全局常量与基础配置
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
CST = timezone(timedelta(hours=8))
TRADING_DAYS = 242
EPS = 1e-12

# A 股交易费用（保守口径，Q20-M208）：
# 印花税 0.05% 仅卖出、过户费 0.001% 双边、佣金 万 2.5（最低 5 元/笔）。
STAMP_DUTY_RATE = 0.0005
TRANSFER_FEE_RATE = 0.00001
COMMISSION_RATE = 0.00025
MIN_COMMISSION = 5.0

# V4.1 feature: 风险等级枚举以字符串保留，避免引入额外 enum 依赖。
SEVERITY_INFO = "info"
SEVERITY_LOW = "low"
SEVERITY_MEDIUM = "medium"
SEVERITY_HIGH = "high"
SEVERITY_CRITICAL = "critical"


# ---------------------------------------------------------------------------
# 数据结构定义
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HistoricalScenario:
    """历史压力场景定义。"""

    name: str
    aliases: tuple[str, ...]
    start: str
    end: str
    benchmark_loss: float
    vol_multiplier: float
    factor_shocks: dict[str, float]
    description: str


@dataclass(slots=True)
class LiquiditySnapshot:
    """单标的流动性快照。"""

    symbol: str
    price: float = 10.0
    avg_daily_amount: float = 100_000_000.0
    avg_daily_volume: float = 10_000_000.0
    turnover_rate: float = 0.01
    bid_ask_spread: float = 0.003
    amihud: float = 1e-8
    name: str = ""
    board: str = "主板"
    pct_change: float | None = None
    listed_days: int | None = None
    limit_status: str = "unknown"
    source: str = "fallback"
    updated_at: str = field(default_factory=lambda: datetime.now(CST).isoformat())


@dataclass(slots=True)
class RiskAlert:
    """实时风控告警。"""

    severity: str
    metric: str
    value: float | str
    threshold: float | str
    message: str
    symbol: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(CST).isoformat())
    action: str = "review"

    def to_dict(self) -> dict[str, Any]:
        """转换为普通 dict，便于 JSON 序列化或前端展示。"""
        return {
            "severity": self.severity,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "message": self.message,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "action": self.action,
        }


@dataclass(slots=True)
class RiskThresholds:
    """默认风控阈值。"""

    var_95_limit: float = 0.03
    var_99_limit: float = 0.05
    cvar_99_limit: float = 0.08
    max_drawdown_warn: float = 0.08
    max_drawdown_stop: float = 0.15
    max_single_weight: float = 0.20
    max_industry_weight: float = 0.35
    min_liquidity_score: float = 40.0
    max_liquidation_days_20pct: float = 5.0
    max_gross_exposure: float = 1.20
    max_net_exposure_abs: float = 1.00
    hhi_warn: float = 0.18
    effective_holdings_min: float = 6.0


@dataclass(slots=True)
class PortfolioRiskState:
    """实时风控状态缓存。"""

    timestamp: str
    total_value: float
    weights: dict[str, float]
    gross_exposure: float
    net_exposure: float
    daily_pnl: float
    daily_return: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    top_position_weight: float
    top_industry_weight: float
    hhi: float
    effective_holdings: float
    liquidity: dict[str, Any]
    concentration: dict[str, Any]
    stress_results: dict[str, Any]
    factor_exposure: dict[str, float]
    market_snapshot: dict[str, Any]


# ---------------------------------------------------------------------------
# 通用工具函数
# ---------------------------------------------------------------------------


def _now() -> str:
    """返回东八区 ISO 时间戳。"""
    return datetime.now(CST).isoformat(timespec="seconds")


def _safe_float(value: Any, default: float = 0.0) -> float:
    """安全转换为 float，兼容中文行情字段、百分号与空值。

    D1 收敛: 转发 quant_system.utils.safe_float（原语义：bool 按数值参与转换）。
    """
    # 原语义保留: 字符串 'NaN'（清洗后）原样返回 nan；
    # 'inf'/'Infinity'/'-inf'（清洗后，大小写敏感）原样返回 ±inf
    # （旧字符串路径无有限性过滤），其余交给公共实现。
    if isinstance(value, str):
        text = value.strip().replace(",", "").replace("%", "")
        if text == "NaN":
            return float("nan")
        if text in ("inf", "Infinity"):
            return float("inf")
        if text == "-inf":
            return float("-inf")
    return _safe_float_impl(value, default=default, allow_bool=True)


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """安全除法，避免风控链路出现 ZeroDivisionError。"""
    if abs(denominator) <= EPS:
        return default
    return numerator / denominator


def _clip(value: float, low: float, high: float) -> float:
    """裁剪数值到指定区间。"""
    return float(max(low, min(high, value)))


def _fmt_pct(value: float, digits: int = 2) -> str:
    """将小数格式化为百分比。"""
    return f"{value * 100:.{digits}f}%"


def _fmt_money(value: float) -> str:
    """将金额格式化为中文报告友好的字符串。"""
    abs_v = abs(value)
    if abs_v >= 1e8:
        return f"{value / 1e8:,.2f} 亿"
    if abs_v >= 1e4:
        return f"{value / 1e4:,.2f} 万"
    return f"{value:,.2f}"


def _as_numeric_frame(data: pd.DataFrame | pd.Series | Mapping[str, Any]) -> pd.DataFrame:
    """将输入收益率对象转换为纯数值 DataFrame。"""
    if isinstance(data, pd.Series):
        frame = data.to_frame()
    elif isinstance(data, pd.DataFrame):
        frame = data.copy()
    elif isinstance(data, Mapping):
        frame = pd.DataFrame(data)
    else:
        raise TypeError("returns 必须是 pandas DataFrame/Series 或可转 DataFrame 的 mapping")

    # 只保留数值列；非数值行情字段不参与协方差与压力测试。
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return numeric


def _normalize_symbol(symbol: Any) -> str:
    """标准化证券代码，统一剥离交易所前缀并对齐 6 位数字代码。"""
    text = str(symbol).strip().lower()
    # P2-Q20-fix: 剥离 sh/sz/bj 前缀与 .SH/.SZ/.BJ 后缀，保证与纯 6 位代码对齐；
    # 否则带前缀持仓在 _align_weights_returns / akshare 匹配中落入 missing_in_returns 被丢弃。
    for prefix in ("sh", "sz", "bj"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    for suffix in (".sh", ".sz", ".bj", ".ss"):
        if text.endswith(suffix):
            text = text[:-len(suffix)]
            break
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if text.isdigit() and len(text) < 6:
        text = text.zfill(6)
    return text


def _normalize_weights(weights: Mapping[str, float] | pd.Series) -> pd.Series:
    """清洗组合权重，保留多空方向，不强行归一到 100%。"""
    if isinstance(weights, pd.Series):
        series = weights.copy()
    else:
        series = pd.Series(dict(weights), dtype="float64")
    if series.empty:
        return pd.Series(dtype="float64")
    series.index = [_normalize_symbol(idx) for idx in series.index]
    series = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    series = series[series.abs() > EPS]
    return series


def _align_weights_returns(
    weights: Mapping[str, float] | pd.Series,
    returns: pd.DataFrame | pd.Series | Mapping[str, Any],
) -> tuple[pd.Series, pd.DataFrame, dict[str, Any]]:
    """对齐组合权重和收益率矩阵。"""
    w = _normalize_weights(weights)
    r = _as_numeric_frame(returns)
    r.columns = [_normalize_symbol(c) for c in r.columns]

    common = [symbol for symbol in w.index if symbol in r.columns]
    missing_in_returns = [symbol for symbol in w.index if symbol not in r.columns]
    extra_return_cols = [symbol for symbol in r.columns if symbol not in w.index]

    if not common:
        raise ValueError("权重与收益率矩阵没有可匹配的标的代码")

    aligned_w = w.reindex(common).fillna(0.0)
    aligned_r = r.loc[:, common].fillna(0.0)
    meta = {
        "matched_symbols": common,
        "missing_in_returns": missing_in_returns,
        "extra_return_cols": extra_return_cols,
        "matched_weight_sum": float(aligned_w.sum()),
        "matched_gross_weight": float(aligned_w.abs().sum()),
    }
    return aligned_w, aligned_r, meta


def _portfolio_returns(weights: pd.Series, returns: pd.DataFrame) -> pd.Series:
    """计算组合收益率序列。"""
    aligned = returns.reindex(columns=weights.index).fillna(0.0)
    return aligned.dot(weights).astype(float)


def _max_drawdown_from_returns(returns: pd.Series | np.ndarray) -> float:
    """由收益率序列计算最大回撤，损失以正数输出。

    D6收敛: 异名/近名异口径保留 —— 损失正数输出, 与 portfolio_v2._max_drawdown(负向)/
    portfolio_optimizer 内联(负向)/portfolio_risk.compute_drawdown(净值百分数) 方向与量纲不同。
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    nav = np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(nav)
    drawdown = 1.0 - nav / np.maximum(peak, EPS)
    return float(np.max(drawdown))


def _historical_var_cvar(losses: np.ndarray, confidence: float) -> tuple[float, float]:
    """历史模拟法 VaR/CVaR，输入 losses 为损失正数。

    D6收敛: 异名异签名异口径保留 —— 与 portfolio_v2._historical_cvar_from_returns 实现近同
    但输入(损失vs收益)/返回(tuple vs float)/NaN处理(isfinite过滤 vs nan_to_num)不同；
    与 portfolio_risk.compute_var(百分数报告)/portfolio_optimizer.cvar_optimize(尾部概率α) 亦异口径。
    """
    clean = losses[np.isfinite(losses)]
    if clean.size == 0:
        return 0.0, 0.0
    var = float(np.quantile(clean, confidence))
    tail = clean[clean >= var]
    cvar = float(tail.mean()) if tail.size else var
    return max(var, 0.0), max(cvar, 0.0)


def _nearest_psd(cov: np.ndarray) -> np.ndarray:
    """将协方差矩阵修正为近似半正定，避免 Monte Carlo Cholesky 失败。"""
    cov = np.asarray(cov, dtype=float)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov = (cov + cov.T) / 2.0
    if cov.size == 0:
        return cov
    try:
        eig_val, eig_vec = np.linalg.eigh(cov)
        eig_val = np.maximum(eig_val, 1e-10)
        fixed = eig_vec @ np.diag(eig_val) @ eig_vec.T
        return (fixed + fixed.T) / 2.0
    except Exception:
        diag = np.diag(np.maximum(np.diag(cov), 1e-8))
        return diag


def _shrink_covariance(returns: pd.DataFrame, shrink: float = 0.08) -> np.ndarray:
    """对样本协方差做简单收缩，减少小样本病态矩阵风险。"""
    cov = returns.cov().values.astype(float)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    diag = np.diag(np.diag(cov))
    shrunk = (1.0 - shrink) * cov + shrink * diag
    return _nearest_psd(shrunk)


def _hhi(weights: Iterable[float]) -> float:
    """计算 Herfindahl-Hirschman Index。

    D6收敛: 异名/近名异口径保留 —— abs口径(按Σ|w|归一, 支持多空), 与
    portfolio_v2._effective_n/portfolio_optimizer.portfolio_summary 平方口径(满仓多头)不同。
    """
    arr = np.asarray(list(weights), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    total = np.sum(np.abs(arr))
    if total <= EPS:
        return 0.0
    share = np.abs(arr) / total
    return float(np.sum(share ** 2))


def _effective_number(weights: Iterable[float]) -> float:
    """由 HHI 推导有效持仓数量。"""
    hhi = _hhi(weights)
    return float(1.0 / hhi) if hhi > EPS else 0.0


def _to_date_index(frame: pd.DataFrame) -> pd.DataFrame:
    """尝试将收益率矩阵索引转换为日期索引。"""
    out = frame.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        return out
    try:
        converted = pd.to_datetime(out.index, errors="coerce")
        if converted.notna().sum() >= max(3, int(len(out) * 0.5)):
            out.index = converted
    except Exception as e:
        logging.getLogger(__name__).error(f"[risk_management_pro] 操作失败: {e}", exc_info=True)
    return out


def _extract_position_value(item: Any) -> float:
    """从持仓对象中提取市值。"""
    if isinstance(item, Mapping):
        for key in ("market_value", "value", "position_value", "amount", "市值"):
            if key in item:
                return _safe_float(item.get(key), 0.0)
        qty = _safe_float(item.get("qty", item.get("shares", item.get("持仓数量", 0))), 0.0)
        price = _safe_float(item.get("price", item.get("current_price", item.get("最新价", 0))), 0.0)
        return qty * price
    return _safe_float(item, 0.0)


def _position_symbol(item_key: Any, item_value: Any) -> str:
    """从持仓 key/value 中提取证券代码。"""
    if isinstance(item_value, Mapping):
        for key in ("symbol", "code", "stock", "证券代码"):
            if item_value.get(key):
                return _normalize_symbol(item_value.get(key))
    return _normalize_symbol(item_key)


def _positions_to_frame(positions: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """将多种持仓输入格式标准化为 DataFrame。"""
    records: list[dict[str, Any]] = []
    if isinstance(positions, Mapping):
        iterable = positions.items()
        for key, item in iterable:
            symbol = _position_symbol(key, item)
            value = _extract_position_value(item)
            record: dict[str, Any] = {
                "symbol": symbol,
                "market_value": value,
                "name": "",
                "industry": "未知",
                "current_drawdown": 0.0,
            }
            if isinstance(item, Mapping):
                record.update({
                    "name": str(item.get("name", item.get("stock_name", item.get("名称", "")))),
                    "industry": str(item.get("industry", item.get("sector", item.get("行业", "未知")))),
                    "current_drawdown": _safe_float(item.get("current_drawdown", item.get("drawdown", 0.0))),
                    "qty": _safe_float(item.get("qty", item.get("shares", 0.0))),
                    "price": _safe_float(item.get("price", item.get("current_price", 0.0))),
                })
            records.append(record)
    else:
        for idx, item in enumerate(positions):
            symbol = _position_symbol(idx, item)
            value = _extract_position_value(item)
            records.append({
                "symbol": symbol,
                "market_value": value,
                "name": str(item.get("name", item.get("stock_name", ""))) if isinstance(item, Mapping) else "",
                "industry": str(item.get("industry", item.get("sector", "未知"))) if isinstance(item, Mapping) else "未知",
                "current_drawdown": _safe_float(item.get("current_drawdown", item.get("drawdown", 0.0))) if isinstance(item, Mapping) else 0.0,
                "qty": _safe_float(item.get("qty", item.get("shares", 0.0))) if isinstance(item, Mapping) else 0.0,
                "price": _safe_float(item.get("price", item.get("current_price", 0.0))) if isinstance(item, Mapping) else 0.0,
            })

    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "market_value", "weight", "industry"])
    frame["market_value"] = pd.to_numeric(frame["market_value"], errors="coerce").fillna(0.0)
    total = float(frame["market_value"].abs().sum())
    frame["weight"] = frame["market_value"] / total if total > EPS else 0.0
    return frame


def _load_stock_name(symbol: str) -> str:
    """从本地 stock_name_map.json 尝试读取股票名称。"""
    path = ROOT / "stock_name_map.json"
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return str(data.get(symbol, data.get(symbol.lstrip("0"), "")))
    except Exception:
        return ""
    return ""


def _board_from_symbol(symbol: str) -> str:
    """根据代码前缀判断交易板块。"""
    s = _normalize_symbol(symbol)
    if s.startswith(("688", "689")):
        return "科创板"
    if s.startswith(("300", "301")):
        return "创业板"
    if s.startswith(("8", "4")):
        return "北交所"
    if s.startswith(("600", "601", "603", "605", "000", "001", "002", "003")):
        return "主板"
    return "未知"


def _limit_threshold_by_board(board: str) -> float:
    """返回不同板块涨跌停阈值。"""
    if board in {"科创板", "创业板"}:
        return 0.198
    if board == "北交所":
        return 0.298
    return 0.098


def _dict_without_none(data: Mapping[str, Any]) -> dict[str, Any]:
    """移除值为 None 的字段，便于报告和接口输出。"""
    return {k: v for k, v in data.items() if v is not None}


# ---------------------------------------------------------------------------
# 1. StressTestEngine — 压力测试
# ---------------------------------------------------------------------------


class StressTestEngine:
    """
    V4.1 feature: 专业压力测试引擎。

    D6收敛登记: 独立能力保留（历史场景复盘/模板冲击/MC厚尾VaR·CVaR/反压力/敏感性，无等价实现）。

    支持四类核心压力测试:
      - 历史场景复盘：如果收益率覆盖场景窗口，直接计算真实组合损益。
      - 模板场景冲击：缺少历史窗口时，用场景冲击 + 组合 beta 估算损失。
      - Monte Carlo 极端损失：使用厚尾 t 分布模拟 VaR / CVaR。
      - 反压力测试：反推达到目标亏损所需的系统性冲击与因子路径。
    """

    def __init__(
        self,
        scenario_library: Mapping[str, HistoricalScenario] | None = None,
        random_seed: int | None = 42,
    ) -> None:
        self.scenario_library = dict(scenario_library or self._default_scenarios())
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

    @staticmethod
    def _default_scenarios() -> dict[str, HistoricalScenario]:
        """内置历史压力场景。"""
        return {
            "2008金融危机": HistoricalScenario(
                name="2008金融危机",
                aliases=("2008", "金融危机", "global financial crisis", "gfc"),
                start="2008-01-01",
                end="2008-12-31",
                benchmark_loss=-0.62,
                vol_multiplier=2.8,
                factor_shocks={
                    "market": -0.62,
                    "equity": -0.62,
                    "financial": -0.72,
                    "real_estate": -0.58,
                    "small_cap": -0.68,
                    "credit_spread": 0.08,
                    "liquidity": -0.45,
                },
                description="全球金融危机，权益资产大幅下跌、信用利差扩大、流动性折价显著。",
            ),
            "2015股灾": HistoricalScenario(
                name="2015股灾",
                aliases=("2015", "股灾", "a股股灾", "china crash"),
                start="2015-06-12",
                end="2015-08-26",
                benchmark_loss=-0.45,
                vol_multiplier=3.2,
                factor_shocks={
                    "market": -0.45,
                    "equity": -0.45,
                    "small_cap": -0.55,
                    "growth": -0.52,
                    "broker": -0.60,
                    "liquidity": -0.50,
                    "momentum": -0.35,
                },
                description="A股杠杆出清与流动性踩踏，中小盘和高 beta 资产承压更重。",
            ),
            "2020新冠": HistoricalScenario(
                name="2020新冠",
                # P2-Q20-fix: 别名表补充 "2020_新冠"，兼容下划线写法，避免调用方抛未知场景；
                # tests.py 侧此前已由 P1-Q28 改为使用 "2020新冠"，这里做防御性兼容。
                aliases=("2020", "新冠", "2020_新冠", "covid", "疫情"),
                start="2020-01-20",
                end="2020-03-23",
                benchmark_loss=-0.25,
                vol_multiplier=2.5,
                factor_shocks={
                    "market": -0.25,
                    "equity": -0.25,
                    "consumer": -0.22,
                    "travel": -0.45,
                    "healthcare": 0.08,
                    "online": 0.05,
                    "liquidity": -0.25,
                },
                description="疫情冲击实体活动，短期风险偏好急降，结构上医疗和线上经济相对抗跌。",
            ),
            "2022美联储加息": HistoricalScenario(
                name="2022美联储加息",
                aliases=("2022", "美联储加息", "fed hike", "加息"),
                start="2022-01-01",
                end="2022-10-31",
                benchmark_loss=-0.30,
                vol_multiplier=2.0,
                factor_shocks={
                    "market": -0.30,
                    "equity": -0.30,
                    "growth": -0.42,
                    "duration": -0.38,
                    "value": -0.15,
                    "usd": 0.12,
                    "rate": 0.035,
                },
                description="全球加息与估值压缩，久期较长的成长资产和高估值资产更脆弱。",
            ),
        }

    def _resolve_scenario(self, scenario_name: str) -> HistoricalScenario:
        """根据名称或别名匹配场景。"""
        text = scenario_name.strip().lower()
        for name, scenario in self.scenario_library.items():
            if text == name.lower() or text in {alias.lower() for alias in scenario.aliases}:
                return scenario
        available = ", ".join(self.scenario_library.keys())
        raise ValueError(f"未知压力场景: {scenario_name}. 可选: {available}")

    def historical_scenario(
        self,
        scenario_name: str,
        portfolio_weights: dict[str, float],
        returns_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        计算历史场景下的组合损益。

        Args:
            scenario_name: 场景名称，支持“2008金融危机/2015股灾/2020新冠/2022美联储加息”。
            portfolio_weights: 组合权重，key 为证券代码，value 为权重。
            returns_df: 日收益率矩阵，index 建议为日期，columns 为证券代码。

        Returns:
            dict: 场景损益、最大回撤、分标的贡献和数据来源。
        """
        scenario = self._resolve_scenario(scenario_name)
        weights, returns, meta = _align_weights_returns(portfolio_weights, returns_df)
        dated_returns = _to_date_index(returns)

        # V4.1 feature: 如果收益率矩阵覆盖历史窗口，使用真实场景收益。
        scenario_returns = pd.DataFrame()
        if isinstance(dated_returns.index, pd.DatetimeIndex):
            start = pd.Timestamp(scenario.start)
            end = pd.Timestamp(scenario.end)
            scenario_returns = dated_returns.loc[(dated_returns.index >= start) & (dated_returns.index <= end)]

        if len(scenario_returns) >= 3:
            port_ret = _portfolio_returns(weights, scenario_returns)
            total_return = float(np.prod(1.0 + port_ret.values) - 1.0)
            total_loss = max(-total_return, 0.0)
            max_dd = _max_drawdown_from_returns(port_ret)
            min_daily_return = float(port_ret.min())
            worst_day = str(port_ret.idxmin().date()) if isinstance(port_ret.index, pd.DatetimeIndex) else str(port_ret.idxmin())
            asset_total_returns = (1.0 + scenario_returns).prod(axis=0) - 1.0
            contribution = (weights * asset_total_returns.reindex(weights.index).fillna(0.0)).sort_values()
            source = "historical_window"
            period = {
                "start": str(scenario_returns.index.min().date()) if isinstance(scenario_returns.index, pd.DatetimeIndex) else scenario.start,
                "end": str(scenario_returns.index.max().date()) if isinstance(scenario_returns.index, pd.DatetimeIndex) else scenario.end,
                "observations": int(len(scenario_returns)),
            }
        else:
            # V4.1 feature: 历史窗口缺失时，使用场景模板 + 历史 beta 估算冲击。
            port_ret_all = _portfolio_returns(weights, returns)
            port_var = float(np.var(port_ret_all.values, ddof=1)) if len(port_ret_all) > 1 else 0.0
            shocks: dict[str, float] = {}
            for symbol in weights.index:
                series = returns[symbol].fillna(0.0)
                if port_var > EPS and len(series) == len(port_ret_all):
                    beta = float(np.cov(series.values, port_ret_all.values, ddof=1)[0, 1] / port_var)
                    beta = _clip(beta, 0.25, 1.80)
                else:
                    beta = 1.0
                shocks[symbol] = scenario.benchmark_loss * beta
            shock_series = pd.Series(shocks)
            contribution = (weights * shock_series.reindex(weights.index).fillna(scenario.benchmark_loss)).sort_values()
            total_return = float(contribution.sum())
            total_loss = max(-total_return, 0.0)
            max_dd = total_loss
            # P2-Q20-fix: 去掉 max(10.0, ...) 下限——sqrt(242)/vol_multiplier 最大约 4.86 < 10，
            # max 恒取 10 使 min_daily_return ≡ benchmark_loss/10、vol_multiplier 完全失效；
            # 直接用 sqrt(TRADING_DAYS)/vol_multiplier 作分母，让波动倍数真正参与日损失估算。
            min_daily_return = scenario.benchmark_loss / (math.sqrt(TRADING_DAYS) / scenario.vol_multiplier)
            worst_day = "template"
            source = "scenario_template"
            period = {"start": scenario.start, "end": scenario.end, "observations": 0}

        top_losses = contribution.head(10)
        top_hedges = contribution.tail(5).sort_values(ascending=False)

        return {
            "scenario": scenario.name,
            "description": scenario.description,
            "source": source,
            "period": period,
            "portfolio_return": round(total_return, 6),
            "portfolio_loss": round(total_loss, 6),
            "loss_amount_per_100m": round(total_loss * 100_000_000, 2),
            "max_drawdown": round(max_dd, 6),
            "min_daily_return": round(min_daily_return, 6),
            "worst_day": worst_day,
            "vol_multiplier": scenario.vol_multiplier,
            "factor_shocks": scenario.factor_shocks,
            "top_loss_contributors": {k: round(float(v), 6) for k, v in top_losses.items()},
            "top_hedge_contributors": {k: round(float(v), 6) for k, v in top_hedges.items()},
            # P1-Q20-fix: 输出全标的贡献序列，供 scenario_loss_matrix 直接使用，
            # 避免其回退到仅 head(10) 的 top_loss_contributors 导致每场景丢失约 5 个标的贡献。
            "asset_contributions": {k: round(float(v), 6) for k, v in contribution.items()},
            "alignment": meta,
            "interpretation": self._scenario_interpretation(total_loss),
        }

    @staticmethod
    def _scenario_interpretation(loss: float) -> str:
        """按场景亏损给出中文解释。"""
        if loss >= 0.30:
            return "极端亏损，组合在该场景下可能触发系统性降仓或硬止损。"
        if loss >= 0.15:
            return "重大亏损，应提前准备对冲、减仓或流动性缓冲。"
        if loss >= 0.08:
            return "中等压力，建议检查集中度和高 beta 暴露。"
        return "压力损失可控，但仍需关注尾部相关性上升。"

    def monte_carlo_stress(
        self,
        weights: Mapping[str, float],
        returns: pd.DataFrame,
        n_simulations: int = 10000,
        confidence: float = 0.99,
    ) -> dict[str, Any]:
        """
        Monte Carlo 模拟极端损失。

        模型说明（V4.1 feature）:
          - 使用历史均值与收缩协方差作为基础分布参数。
          - 默认使用自由度 5 的 t 分布生成厚尾冲击，较正态分布更保守。
          - VaR/CVaR 输出均为“损失正数”，即 0.05 表示 -5% 亏损。
        """
        if not (0.5 < confidence < 0.9999):
            raise ValueError("confidence 应位于 (0.5, 0.9999) 区间")
        if n_simulations <= 100:
            raise ValueError("n_simulations 建议大于 100，才能估计尾部损失")

        w, r, meta = _align_weights_returns(weights, returns)
        mean = r.mean().values.astype(float)
        cov = _shrink_covariance(r)
        n_assets = len(w)

        # V4.1 feature: 厚尾冲击，df 越小尾部越厚；df=5 兼顾稳定性与保守性。
        df = 5
        try:
            if SCIPY_AVAILABLE and stats is not None:
                z = stats.t.rvs(df=df, size=(n_simulations, n_assets), random_state=self.random_seed)
            else:
                z = self.rng.standard_t(df=df, size=(n_simulations, n_assets))
            z = z / math.sqrt(df / (df - 2.0))
        except Exception:
            z = self.rng.standard_normal(size=(n_simulations, n_assets))

        try:
            chol = np.linalg.cholesky(cov)
        except Exception:
            chol = np.linalg.cholesky(_nearest_psd(cov) + np.eye(n_assets) * 1e-10)

        simulated_returns = z @ chol.T + mean
        port_returns = simulated_returns @ w.values
        losses = -port_returns
        var, cvar = _historical_var_cvar(losses, confidence)
        var_95, cvar_95 = _historical_var_cvar(losses, 0.95)
        var_99, cvar_99 = _historical_var_cvar(losses, 0.99)

        positive_losses = losses[losses > 0]
        expected_loss = float(positive_losses.mean()) if positive_losses.size else 0.0
        max_loss = float(np.max(losses)) if losses.size else 0.0

        return {
            "method": "monte_carlo_t_distribution",
            "n_simulations": int(n_simulations),
            "confidence": float(confidence),
            "var": round(var, 6),
            "cvar": round(cvar, 6),
            "var_95": round(var_95, 6),
            "cvar_95": round(cvar_95, 6),
            "var_99": round(var_99, 6),
            "cvar_99": round(cvar_99, 6),
            "expected_loss": round(max(expected_loss, 0.0), 6),
            "expected_return": round(float(np.mean(port_returns)), 6),
            "volatility_daily": round(float(np.std(port_returns, ddof=1)), 6),
            "volatility_annualized": round(float(np.std(port_returns, ddof=1) * math.sqrt(TRADING_DAYS)), 6),
            "max_simulated_loss": round(max(max_loss, 0.0), 6),
            "prob_loss_gt_5pct": round(float(np.mean(losses > 0.05)), 6),
            "prob_loss_gt_10pct": round(float(np.mean(losses > 0.10)), 6),
            "prob_loss_gt_20pct": round(float(np.mean(losses > 0.20)), 6),
            "tail_sample_size": int(np.sum(losses >= var)),
            "alignment": meta,
        }

    def reverse_stress_test(
        self,
        weights: Mapping[str, float],
        returns: pd.DataFrame,
        target_loss: float = 0.2,
    ) -> dict[str, Any]:
        """
        反压力测试：推导什么市场条件会导致目标亏损。

        输出包括:
          - 达到 target_loss 所需的一日波动倍数。
          - 沿组合风险梯度方向的资产冲击。
          - PCA / 特征分解下的主因子冲击分析。
        """
        if target_loss <= 0:
            raise ValueError("target_loss 必须为正数，例如 0.2 表示 20% 目标亏损")

        w, r, meta = _align_weights_returns(weights, returns)
        cov = _shrink_covariance(r)
        w_vec = w.values.reshape(-1, 1)
        port_var = float(w_vec.T @ cov @ w_vec)
        port_vol = math.sqrt(max(port_var, EPS))
        required_sigma = float(target_loss / max(port_vol, EPS))

        # V4.1 feature: 风险梯度方向是最容易伤害当前组合的线性冲击方向。
        gradient = cov @ w.values
        denom = float(w.values @ gradient)
        if abs(denom) <= EPS:
            shock_vector = -target_loss * w.values / max(float(w.values @ w.values), EPS)
        else:
            shock_vector = -target_loss * gradient / denom
        asset_shocks = pd.Series(shock_vector, index=w.index).sort_values()

        # 主成分因子冲击分析：优先 sklearn PCA，失败则 numpy eig。
        factor_records: list[dict[str, Any]] = []
        centered = r - r.mean()
        try:
            if SKLEARN_AVAILABLE and PCA is not None and len(r) > len(w):
                n_components = min(5, len(w), len(r) - 1)
                pca = PCA(n_components=n_components)
                pca.fit(centered.values)
                components = pca.components_
                variances = pca.explained_variance_
            else:
                eig_val, eig_vec = np.linalg.eigh(cov)
                order = np.argsort(eig_val)[::-1]
                variances = eig_val[order][: min(5, len(w))]
                components = eig_vec[:, order].T[: min(5, len(w))]
            for i, (var_i, comp_i) in enumerate(zip(variances, components), start=1):
                exposure = float(w.values @ comp_i)
                factor_vol = math.sqrt(max(float(var_i), EPS))
                required_move = _safe_div(target_loss, abs(exposure) * factor_vol, default=float("inf"))
                loadings = pd.Series(comp_i, index=w.index).sort_values(key=lambda x: x.abs(), ascending=False)
                factor_records.append({
                    "factor": f"PC{i}",
                    "portfolio_exposure": round(exposure, 6),
                    "factor_vol_daily": round(factor_vol, 6),
                    "required_sigma_move": round(required_move, 3) if np.isfinite(required_move) else "inf",
                    "top_loaded_assets": {k: round(float(v), 4) for k, v in loadings.head(5).items()},
                })
        except Exception as exc:
            factor_records.append({
                "factor": "fallback",
                "error": str(exc),
                "message": "PCA 失败，已保留资产级风险梯度冲击。",
            })

        # 使用 scipy optimize 进一步求解最小二乘冲击：min ||shock||, s.t. w·shock = -target_loss。
        optimized: dict[str, Any]
        if SCIPY_AVAILABLE and optimize is not None:
            try:
                n = len(w)

                def objective(x: np.ndarray) -> float:
                    return float(np.sum(x ** 2))

                constraints = ({"type": "eq", "fun": lambda x: float(w.values @ x + target_loss)},)
                bounds = [(-0.9, 0.9) for _ in range(n)]
                result = optimize.minimize(objective, shock_vector, bounds=bounds, constraints=constraints, method="SLSQP")
                if result.success:
                    opt_series = pd.Series(result.x, index=w.index).sort_values()
                    optimized = {
                        "success": True,
                        "loss_check": round(float(-(w.values @ result.x)), 6),
                        "l2_norm": round(float(np.linalg.norm(result.x)), 6),
                        "asset_shocks": {k: round(float(v), 6) for k, v in opt_series.items()},
                    }
                else:
                    optimized = {"success": False, "message": str(result.message)}
            except Exception as exc:
                optimized = {"success": False, "message": str(exc)}
        else:
            optimized = {"success": False, "message": "scipy.optimize 不可用，跳过约束优化。"}

        return {
            "target_loss": round(float(target_loss), 6),
            "portfolio_vol_daily": round(port_vol, 6),
            "portfolio_vol_annualized": round(port_vol * math.sqrt(TRADING_DAYS), 6),
            "required_one_day_sigma_move": round(required_sigma, 3),
            "risk_gradient_asset_shocks": {k: round(float(v), 6) for k, v in asset_shocks.items()},
            "largest_required_declines": {k: round(float(v), 6) for k, v in asset_shocks.head(10).items()},
            "factor_shock_analysis": factor_records,
            "optimized_min_norm_shock": optimized,
            "conditions": self._reverse_conditions(required_sigma, target_loss),
            "alignment": meta,
        }

    @staticmethod
    def _reverse_conditions(required_sigma: float, target_loss: float) -> list[str]:
        """将反压力测试数值翻译为风控语言。"""
        notes: list[str] = []
        if required_sigma <= 2.0:
            notes.append("目标亏损可能由常见二倍波动冲击触发，应视为高概率尾部风险。")
        elif required_sigma <= 4.0:
            notes.append("目标亏损需要显著市场冲击，通常伴随相关性上升和流动性恶化。")
        else:
            notes.append("目标亏损需要极端冲击，但若组合杠杆或流动性不足，实际门槛会降低。")
        if target_loss >= 0.2:
            notes.append("亏损目标达到或超过 20%，建议同步检查硬止损、融资约束和赎回压力。")
        return notes

    def sensitivity_analysis(
        self,
        weights: Mapping[str, float],
        factor_exposures: pd.DataFrame,
        factor_shocks: Mapping[str, float] | pd.Series | pd.DataFrame,
    ) -> pd.DataFrame:
        """
        因子敏感性分析：估算各因子变动 1σ 对组合的影响。

        Args:
            weights: 组合权重。
            factor_exposures: 行为证券代码、列为因子的暴露矩阵。
            factor_shocks: 因子 1σ 冲击幅度，正负均可；若为 DataFrame，会读取 shock/sigma 列。

        Returns:
            DataFrame: 每个因子的组合暴露、冲击收益、逆向损失和主要贡献标的。
        """
        w = _normalize_weights(weights)
        exposures = factor_exposures.copy()
        exposures.index = [_normalize_symbol(i) for i in exposures.index]
        exposures = exposures.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        common = [symbol for symbol in w.index if symbol in exposures.index]
        if not common:
            raise ValueError("权重与因子暴露矩阵没有可匹配证券")
        w = w.reindex(common).fillna(0.0)
        exposures = exposures.loc[common]

        if isinstance(factor_shocks, pd.DataFrame):
            shock_col = "shock" if "shock" in factor_shocks.columns else "sigma"
            shocks = factor_shocks[shock_col].copy() if shock_col in factor_shocks.columns else factor_shocks.iloc[:, 0].copy()
        elif isinstance(factor_shocks, pd.Series):
            shocks = factor_shocks.copy()
        else:
            shocks = pd.Series(dict(factor_shocks), dtype="float64")
        shocks = pd.to_numeric(shocks, errors="coerce").fillna(0.0)

        records: list[dict[str, Any]] = []
        for factor in exposures.columns:
            shock = float(shocks.get(factor, 0.0))
            exposure_vec = exposures[factor]
            portfolio_exposure = float(w.dot(exposure_vec))
            pnl_impact = portfolio_exposure * shock
            symbol_contrib = (w * exposure_vec * shock).sort_values(key=lambda x: x.abs(), ascending=False)
            records.append({
                "factor": factor,
                "shock_1sigma": shock,
                "portfolio_exposure": portfolio_exposure,
                "pnl_impact": pnl_impact,
                "loss_if_adverse": abs(pnl_impact),
                "direction": "收益" if pnl_impact >= 0 else "亏损",
                "top_contributors": json.dumps({k: round(float(v), 6) for k, v in symbol_contrib.head(5).items()}, ensure_ascii=False),
            })
        result = pd.DataFrame(records)
        if not result.empty:
            result = result.sort_values("loss_if_adverse", ascending=False).reset_index(drop=True)
        return result

    def run_all_historical_scenarios(
        self,
        portfolio_weights: Mapping[str, float],
        returns_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        V4.1 feature: 批量运行内置历史压力场景。

        用于风控日报和看板，不改变 historical_scenario 的单场景 dict 输出。
        即使个别场景失败，也会保留错误行，便于审计数据质量。
        """
        records: list[dict[str, Any]] = []
        for scenario_name in self.scenario_library:
            try:
                result = self.historical_scenario(scenario_name, dict(portfolio_weights), returns_df)
                records.append({
                    "scenario": result.get("scenario"),
                    "source": result.get("source"),
                    "portfolio_loss": _safe_float(result.get("portfolio_loss"), 0.0),
                    "portfolio_return": _safe_float(result.get("portfolio_return"), 0.0),
                    "max_drawdown": _safe_float(result.get("max_drawdown"), 0.0),
                    "min_daily_return": _safe_float(result.get("min_daily_return"), 0.0),
                    "worst_day": result.get("worst_day"),
                    "description": result.get("description"),
                    "interpretation": result.get("interpretation"),
                })
            except Exception as exc:
                records.append({
                    "scenario": scenario_name,
                    "source": "error",
                    "portfolio_loss": np.nan,
                    "portfolio_return": np.nan,
                    "max_drawdown": np.nan,
                    "min_daily_return": np.nan,
                    "worst_day": "",
                    "description": "",
                    "interpretation": f"场景运行失败: {exc}",
                })
        frame = pd.DataFrame(records)
        if not frame.empty:
            frame = frame.sort_values("portfolio_loss", ascending=False, na_position="last").reset_index(drop=True)
        return frame

    def scenario_loss_matrix(
        self,
        portfolio_weights: Mapping[str, float],
        returns_df: pd.DataFrame,
        custom_shocks: Mapping[str, Mapping[str, float]] | None = None,
    ) -> pd.DataFrame:
        """
        V4.1 feature: 生成场景-标的损益贡献矩阵。

        返回值中 portfolio_total 为组合场景收益，portfolio_loss 统一按损失为正数。
        custom_shocks 可传入自定义冲击，格式为 {场景名: {证券代码: 冲击收益率}}。
        """
        w = _normalize_weights(portfolio_weights)
        r = _as_numeric_frame(returns_df)
        r.columns = [_normalize_symbol(c) for c in r.columns]
        rows: list[pd.Series] = []
        names: list[str] = []

        for scenario_name in self.scenario_library:
            result = self.historical_scenario(scenario_name, dict(w), r)
            contributions = pd.Series(result.get("asset_contributions", {}), dtype="float64")
            if contributions.empty:
                # P1-Q20-fix: 兼容旧返回结构（无 asset_contributions）时，合并损失与对冲两组
                # 贡献，避免仅用 top_loss_contributors（head 10）丢失正贡献标的使 portfolio_total 失真。
                merged: dict[str, float] = {}
                merged.update(result.get("top_hedge_contributors", {}) or {})
                merged.update(result.get("top_loss_contributors", {}) or {})
                contributions = pd.Series(merged, dtype="float64")
            rows.append(contributions.reindex(w.index).fillna(0.0))
            names.append(scenario_name)

        if custom_shocks:
            for scenario_name, shocks in custom_shocks.items():
                shock_series = pd.Series({_normalize_symbol(k): _safe_float(v) for k, v in shocks.items()}, dtype="float64")
                rows.append((w * shock_series.reindex(w.index).fillna(0.0)).reindex(w.index).fillna(0.0))
                names.append(str(scenario_name))

        if not rows:
            return pd.DataFrame(columns=list(w.index))
        matrix = pd.DataFrame(rows, index=names)
        matrix["portfolio_total"] = matrix.sum(axis=1)
        matrix["portfolio_loss"] = (-matrix["portfolio_total"]).clip(lower=0.0)
        return matrix.sort_values("portfolio_loss", ascending=False)

    def tail_diagnostics(
        self,
        weights: Mapping[str, float],
        returns: pd.DataFrame,
        confidence_levels: Sequence[float] = (0.95, 0.975, 0.99),
    ) -> dict[str, Any]:
        """
        V4.1 feature: 历史收益尾部诊断。

        输出偏度、峰度、最差日期、多置信度 VaR/CVaR，辅助判断 Monte Carlo
        假设是否低估厚尾或左偏风险。
        """
        w, r, meta = _align_weights_returns(weights, returns)
        port_ret = _portfolio_returns(w, r)
        losses = -port_ret.values
        metrics: dict[str, Any] = {
            "observations": int(len(port_ret)),
            "mean_daily_return": round(float(port_ret.mean()), 6) if len(port_ret) else 0.0,
            "vol_daily": round(float(port_ret.std(ddof=1)), 6) if len(port_ret) > 1 else 0.0,
            "vol_annualized": round(float(port_ret.std(ddof=1) * math.sqrt(TRADING_DAYS)), 6) if len(port_ret) > 1 else 0.0,
            "max_drawdown": round(_max_drawdown_from_returns(port_ret), 6),
            "alignment": meta,
        }
        if len(port_ret) > 2:
            if SCIPY_AVAILABLE and stats is not None:
                metrics["skew"] = round(float(stats.skew(port_ret.values, nan_policy="omit")), 6)
                metrics["kurtosis"] = round(float(stats.kurtosis(port_ret.values, fisher=True, nan_policy="omit")), 6)
            else:
                centered = port_ret.values - float(port_ret.mean())
                vol = float(np.std(centered, ddof=1))
                metrics["skew"] = round(float(np.mean(centered ** 3) / max(vol ** 3, EPS)), 6)
                metrics["kurtosis"] = round(float(np.mean(centered ** 4) / max(vol ** 4, EPS) - 3.0), 6)

        metrics["var_table"] = []
        for level in confidence_levels:
            var, cvar = _historical_var_cvar(losses, float(level))
            metrics["var_table"].append({
                "confidence": float(level),
                "var": round(var, 6),
                "cvar": round(cvar, 6),
                "tail_count": int(np.sum(losses >= var)),
            })
        if len(port_ret):
            worst = port_ret.sort_values().head(10)
            metrics["worst_days"] = {str(k): round(float(v), 6) for k, v in worst.items()}
        return metrics


# ---------------------------------------------------------------------------
# 2. LiquidityRiskManager — 流动性风险管理
# ---------------------------------------------------------------------------


class LiquidityRiskManager:
    """
    V4.1 feature: 流动性风险管理器。

    D6收敛登记: 独立能力保留（流动性评分/清仓成本/参与率/执行风险计划，无等价实现）。

    评分权重:
      - 日均成交额 40%
      - 换手率 30%
      - 买卖价差 20%
      - Amihud 非流动性 10%

    输入行情可以来自:
      - 构造函数传入的 dict / DataFrame。
      - akshare 实时快照（可选依赖）。
      - 保守 fallback，保证风控流程不中断。
    """

    # P2-Q20-fix: 全市场快照（60s TTL）与个股上市天数（按日缓存）进程内复用，
    # 避免实时监控逐标的重复拉取全市场行情，也避免新股检查恒退化为"缺少上市天数"。
    _AKSHARE_SPOT_CACHE: dict[str, Any] = {}
    _AKSHARE_SPOT_TTL = 60.0
    _AKSHARE_LISTED_DAYS_CACHE: dict[str, Any] = {}
    _AKSHARE_LISTED_DAYS_TTL = 86400.0

    def __init__(
        self,
        market_data: Mapping[str, Any] | pd.DataFrame | None = None,
        snapshot_loader: Callable[[str], LiquiditySnapshot | None] | None = None,
    ) -> None:
        self.market_data = market_data
        self.snapshot_loader = snapshot_loader
        self._snapshot_cache: dict[str, LiquiditySnapshot] = {}

    def assess_liquidity(self, symbol: str, position_value: float) -> dict[str, Any]:
        """评估单个标的流动性。"""
        snapshot = self._get_snapshot(symbol)
        adv = max(snapshot.avg_daily_amount, EPS)
        turnover_rate = self._normalize_turnover(snapshot.turnover_rate)
        spread = max(snapshot.bid_ask_spread, 0.0)
        amihud = max(snapshot.amihud, 0.0)

        # V4.1 feature: 日均成交额采用对数评分，避免大票分数过度挤压中等流动性标的。
        adv_score = _clip((math.log10(max(adv, 1.0)) - 7.0) / 3.0 * 100.0, 0.0, 100.0)
        turnover_score = _clip(turnover_rate / 0.05 * 100.0, 0.0, 100.0)
        spread_score = _clip((0.02 - spread) / (0.02 - 0.0005) * 100.0, 0.0, 100.0)
        # P2-Q20-fix: Amihud 改用对数标尺（1e-10→100 分、1e-6→0 分），
        # 原线性标尺下 amihud≤1e-7 恒在 90-100 分区间、10% 权重近似常数项失去区分度；
        # akshare 分支用 1/成交额 作代理口径（常见 1e-9~1e-8），对数映射可拉开档位。
        amihud_log = math.log10(max(amihud, 1e-10))
        amihud_score = _clip(100.0 * (-6.0 - amihud_log) / 4.0, 0.0, 100.0)

        score = (
            adv_score * 0.40
            + turnover_score * 0.30
            + spread_score * 0.20
            + amihud_score * 0.10
        )

        liquidation_days = {
            "participation_10pct": _safe_div(position_value, adv * 0.10, default=float("inf")),
            "participation_20pct": _safe_div(position_value, adv * 0.20, default=float("inf")),
            "participation_50pct": _safe_div(position_value, adv * 0.50, default=float("inf")),
        }
        adv_ratio = _safe_div(position_value, adv, default=float("inf"))

        return {
            "symbol": _normalize_symbol(symbol),
            "name": snapshot.name,
            "board": snapshot.board,
            "liquidity_score": round(score, 2),
            "liquidity_level": self._score_level(score),
            "position_value": float(position_value),
            "adv_ratio": round(float(adv_ratio), 4) if np.isfinite(adv_ratio) else "inf",
            "estimated_liquidation_days": {k: round(float(v), 2) if np.isfinite(v) else "inf" for k, v in liquidation_days.items()},
            "score_breakdown": {
                "avg_daily_amount_40pct": round(adv_score, 2),
                "turnover_rate_30pct": round(turnover_score, 2),
                "bid_ask_spread_20pct": round(spread_score, 2),
                "amihud_10pct": round(amihud_score, 2),
            },
            "metrics": {
                "price": snapshot.price,
                "avg_daily_amount": snapshot.avg_daily_amount,
                "avg_daily_volume": snapshot.avg_daily_volume,
                "turnover_rate": turnover_rate,
                "bid_ask_spread": spread,
                "amihud": amihud,
                "source": snapshot.source,
                "updated_at": snapshot.updated_at,
            },
            "warnings": self._liquidity_warnings(score, liquidation_days, spread, adv_ratio),
        }

    @staticmethod
    def _normalize_turnover(value: float) -> float:
        """统一换手率为小数口径：akshare/行情源换手率为百分数（0.5 表示 0.5%）。"""
        value = _safe_float(value, 0.0)
        # P1-Q20-fix: 换手率按百分数统一归一（akshare 口径），并处理边界 1.0（1.0%→0.01）；
        # 修正此前 0.5 被当作 50% 小数导致 turnover_score≈100、流动性评分系统性虚高、
        # min_liquidity_score<40 告警漏报的问题。
        return max(value / 100.0, 0.0)

    @staticmethod
    def _score_level(score: float) -> str:
        """根据流动性分数输出中文等级。"""
        if score >= 80:
            return "优秀"
        if score >= 60:
            return "良好"
        if score >= 40:
            return "一般"
        if score >= 20:
            return "较差"
        return "极差"

    @staticmethod
    def _liquidity_warnings(
        score: float,
        liquidation_days: Mapping[str, float],
        spread: float,
        adv_ratio: float,
    ) -> list[str]:
        """生成流动性风险提示。"""
        warnings_list: list[str] = []
        if score < 40:
            warnings_list.append("流动性评分低于 40，建议降低单票仓位或延长执行周期。")
        if liquidation_days.get("participation_20pct", 0.0) > 5:
            warnings_list.append("按 20% 成交额参与率清仓超过 5 天，存在退出拥挤风险。")
        if spread > 0.01:
            warnings_list.append("买卖价差超过 1%，交易成本可能显著抬升。")
        if adv_ratio > 1.0:
            warnings_list.append("持仓市值超过 1 日成交额，压力市况下可能难以退出。")
        return warnings_list

    def liquidation_cost(self, symbol: str, shares_to_sell: int, urgency: str = "normal") -> float:
        """基于 market_impact.py 计算清仓成本（含 A 股交易费用）。"""
        if shares_to_sell <= 0:
            return 0.0
        snapshot = self._get_snapshot(symbol)
        urgency_key = urgency.lower().strip()
        horizon_days = {
            "low": 5.0,
            "slow": 5.0,
            "normal": 2.0,
            "high": 1.0,
            "urgent": 0.25,
            "immediate": 0.10,
        }.get(urgency_key, 2.0)
        trade_value = shares_to_sell * max(snapshot.price, 0.01)

        # P2-Q20-fix: 在冲击+半价差基础上叠加 A 股交易费用（印花税卖出 0.05%、
        # 过户费 0.001% 双边、佣金 max(5 元, 万 2.5)），满足"保守估计"设计原则；
        # 原实现仅含市场冲击+半价差，卖出 100 万实测仅约 15.16bps，明显低估清仓成本。
        stamp_duty = trade_value * STAMP_DUTY_RATE
        transfer_fee = trade_value * TRANSFER_FEE_RATE
        commission = max(MIN_COMMISSION, trade_value * COMMISSION_RATE)
        explicit_fees = stamp_duty + transfer_fee + commission

        # V4.1 feature: 优先复用 Almgren-Chriss 模型，保持执行成本口径一致。
        if MARKET_IMPACT_AVAILABLE and MarketImpact is not None and ACParams is not None:
            try:
                params = ACParams(
                    sigma=0.35,
                    spread=max(snapshot.bid_ask_spread, 0.0005),
                    daily_volume=max(snapshot.avg_daily_amount, 1.0),
                )
                model = MarketImpact(params=params)
                result = model.total_cost(
                    shares=float(shares_to_sell),
                    arrival_price=max(snapshot.price, 0.01),
                    daily_vol=max(snapshot.avg_daily_amount, 1.0),
                    horizon_days=max(horizon_days, 0.05),
                    side="sell",
                )
                return float(result.total_cost_value) + explicit_fees
            except Exception as e:
                logging.getLogger(__name__).error(f"[risk_management_pro] 操作失败: {e}", exc_info=True)

        # fallback: 平方根冲击 + 半价差成本。
        horizon_adv = snapshot.avg_daily_amount * max(horizon_days, 0.05)
        participation = _safe_div(trade_value, horizon_adv, default=1.0)
        spread_cost_bps = snapshot.bid_ask_spread * 0.5 * 1e4
        impact_bps = 50.0 * math.sqrt(max(participation, 0.0))
        urgency_multiplier = {"low": 0.8, "slow": 0.8, "normal": 1.0, "high": 1.4, "urgent": 2.0, "immediate": 2.5}.get(urgency_key, 1.0)
        return float(trade_value * (spread_cost_bps + impact_bps * urgency_multiplier) / 1e4) + explicit_fees

    def liquidity_check(
        self,
        portfolio: Mapping[str, Any],
        threshold: float = 0.2,
        max_adv_ratio: float | None = None,
        score_floor: float | None = None,
    ) -> list[dict[str, Any]]:
        """检查组合中流动性不足的持仓。

        Args:
            threshold: 兼容旧调用的统一阈值；当 max_adv_ratio/score_floor 未显式给出时，
                按 threshold 拆分（score_floor = threshold*100，max_adv_ratio = threshold）。
            max_adv_ratio: 持仓市值/日均成交额上限，缺省时取 threshold。
            score_floor: 流动性评分下限，缺省时取 threshold*100（threshold<=1 时）。
        """
        warnings_list: list[dict[str, Any]] = []
        # P2-Q20-fix: 拆分参数语义——原 threshold 同时兼作评分下限(×100)与 adv_ratio 上限，
        # 语义混淆易误用；显式提供 score_floor 与 max_adv_ratio，未指定时按旧口径从 threshold 推导。
        if score_floor is None:
            score_floor = threshold * 100.0 if threshold <= 1.0 else threshold
        if max_adv_ratio is None:
            max_adv_ratio = threshold
        for symbol, item in portfolio.items():
            norm_symbol = _position_symbol(symbol, item)
            value = _extract_position_value(item)
            if value <= 0:
                continue
            assessment = self.assess_liquidity(norm_symbol, value)
            metrics = assessment.get("metrics", {})
            adv = _safe_float(metrics.get("avg_daily_amount"), 0.0)
            adv_ratio = _safe_div(value, adv, default=float("inf"))
            days20 = assessment.get("estimated_liquidation_days", {}).get("participation_20pct", "inf")
            days20_float = _safe_float(days20, float("inf"))

            reasons: list[str] = []
            if _safe_float(assessment.get("liquidity_score"), 0.0) < score_floor:
                reasons.append(f"流动性评分低于阈值 {score_floor:.1f}")
            if adv_ratio > max_adv_ratio:
                reasons.append(f"持仓/日均成交额 {adv_ratio:.2f} 超过阈值 {max_adv_ratio:.2f}")
            if days20_float > 5:
                reasons.append("20% 参与率清仓超过 5 天")
            if reasons:
                warnings_list.append({
                    "symbol": norm_symbol,
                    "position_value": value,
                    "liquidity_score": assessment.get("liquidity_score"),
                    "liquidity_level": assessment.get("liquidity_level"),
                    "adv_ratio": round(float(adv_ratio), 4) if np.isfinite(adv_ratio) else "inf",
                    "days_to_liquidate_20pct": days20,
                    "reasons": reasons,
                    "suggestion": "降低仓位、拆单执行，或将参与率控制在 10%-20% 区间。",
                })
        warnings_list.sort(key=lambda x: _safe_float(x.get("days_to_liquidate_20pct"), 0.0), reverse=True)
        return warnings_list

    def portfolio_liquidity_summary(
        self,
        portfolio: Mapping[str, Any],
        participation_rates: Sequence[float] = (0.10, 0.20, 0.50),
    ) -> dict[str, Any]:
        """
        V4.1 feature: 组合级流动性画像。

        输出市值加权流动性评分、最差评分、组合成交额覆盖倍数、不同参与率下
        最慢清仓天数，以及最需要警惕的流动性瓶颈标的。
        """
        frame = _positions_to_frame(portfolio)
        if frame.empty:
            return {
                "portfolio_value": 0.0,
                "weighted_liquidity_score": 100.0,
                "min_liquidity_score": 100.0,
                "avg_daily_amount_coverage": 0.0,
                "liquidation_days": {},
                "bottlenecks": [],
                "assessments": [],
            }

        total_abs_value = float(frame["market_value"].abs().sum())
        assessments: list[dict[str, Any]] = []
        for _, row in frame.iterrows():
            symbol = str(row["symbol"])
            value = abs(float(row["market_value"]))
            assessment = self.assess_liquidity(symbol, value)
            assessment["portfolio_weight_abs"] = _safe_div(value, total_abs_value, default=0.0)
            assessments.append(assessment)

        weighted_score = sum(
            _safe_float(a.get("liquidity_score"), 0.0) * _safe_float(a.get("portfolio_weight_abs"), 0.0)
            for a in assessments
        )
        min_score = min((_safe_float(a.get("liquidity_score"), 0.0) for a in assessments), default=100.0)
        total_adv = sum(_safe_float(a.get("metrics", {}).get("avg_daily_amount"), 0.0) for a in assessments)
        coverage = _safe_div(total_abs_value, total_adv, default=float("inf"))

        liquidation_days: dict[str, float | str] = {}
        for rate in participation_rates:
            if rate <= 0:
                continue
            slowest_days = 0.0
            for item in assessments:
                value = _safe_float(item.get("position_value"), 0.0)
                adv = _safe_float(item.get("metrics", {}).get("avg_daily_amount"), 0.0)
                slowest_days = max(slowest_days, _safe_div(value, adv * rate, default=float("inf")))
            key = f"participation_{int(rate * 100)}pct"
            liquidation_days[key] = round(float(slowest_days), 2) if np.isfinite(slowest_days) else "inf"

        bottlenecks = sorted(
            assessments,
            key=lambda item: (_safe_float(item.get("liquidity_score"), 0.0), -_safe_float(item.get("adv_ratio"), 0.0)),
        )[:10]
        compact_bottlenecks = [
            {
                "symbol": item.get("symbol"),
                "name": item.get("name"),
                "liquidity_score": item.get("liquidity_score"),
                "liquidity_level": item.get("liquidity_level"),
                "adv_ratio": item.get("adv_ratio"),
                "days_20pct": item.get("estimated_liquidation_days", {}).get("participation_20pct"),
                "warnings": item.get("warnings", []),
            }
            for item in bottlenecks
        ]
        return {
            "portfolio_value": total_abs_value,
            "weighted_liquidity_score": round(float(weighted_score), 2),
            "min_liquidity_score": round(float(min_score), 2),
            "avg_daily_amount_coverage": round(float(coverage), 4) if np.isfinite(coverage) else "inf",
            "liquidation_days": liquidation_days,
            "bottlenecks": compact_bottlenecks,
            "assessments": assessments,
        }

    def execution_risk_plan(
        self,
        symbol: str,
        shares_to_sell: int,
        max_participation: float = 0.20,
    ) -> dict[str, Any]:
        """
        V4.1 feature: 大额减仓执行风险计划。

        该方法只估算不同紧急程度下的成本和建议交易天数，不产生任何下单动作。
        """
        snapshot = self._get_snapshot(symbol)
        shares = max(int(shares_to_sell), 0)
        trade_value = shares * max(snapshot.price, 0.01)
        daily_capacity_value = snapshot.avg_daily_amount * max(max_participation, 0.01)
        suggested_days = max(1.0, math.ceil(_safe_div(trade_value, daily_capacity_value, default=1.0)))
        urgency_costs = {
            urgency: self.liquidation_cost(snapshot.symbol, shares, urgency=urgency)
            for urgency in ("slow", "normal", "high", "urgent")
        }
        cost_bps = {
            urgency: round(_safe_div(cost, trade_value, default=0.0) * 1e4, 2)
            for urgency, cost in urgency_costs.items()
        }
        return {
            "symbol": snapshot.symbol,
            "name": snapshot.name,
            "trade_value": trade_value,
            "max_participation": max_participation,
            "suggested_days": suggested_days,
            "daily_capacity_value": daily_capacity_value,
            "urgency_cost_value": {k: round(float(v), 2) for k, v in urgency_costs.items()},
            "urgency_cost_bps": cost_bps,
            "recommendation": self._execution_recommendation(suggested_days, cost_bps),
        }

    @staticmethod
    def _execution_recommendation(suggested_days: float, cost_bps: Mapping[str, float]) -> str:
        """根据成本和建议天数生成执行建议。"""
        urgent_cost = _safe_float(cost_bps.get("urgent"), 0.0)
        normal_cost = _safe_float(cost_bps.get("normal"), 0.0)
        if suggested_days > 10:
            return "建议分阶段退出，必要时先用指数或行业工具对冲，避免单日高参与率。"
        if normal_cost > 0 and urgent_cost > normal_cost * 1.8:
            return "紧急执行成本显著高于普通执行，除非触发硬风控，不建议一次性退出。"
        return "可按普通节奏执行，并实时监控成交额、盘口价差和涨跌停状态。"

    def _get_snapshot(self, symbol: str) -> LiquiditySnapshot:
        """获取流动性快照，按自定义 loader、缓存、传入行情、akshare、fallback 顺序尝试。"""
        norm_symbol = _normalize_symbol(symbol)
        if norm_symbol in self._snapshot_cache:
            return self._snapshot_cache[norm_symbol]

        snapshot: LiquiditySnapshot | None = None
        if self.snapshot_loader is not None:
            try:
                snapshot = self.snapshot_loader(norm_symbol)
            except Exception:
                snapshot = None
        if snapshot is None:
            snapshot = self._snapshot_from_market_data(norm_symbol)
        if snapshot is None:
            snapshot = self._snapshot_from_akshare(norm_symbol)
        if snapshot is None:
            snapshot = self._fallback_snapshot(norm_symbol)

        self._snapshot_cache[norm_symbol] = snapshot
        return snapshot

    def _snapshot_from_market_data(self, symbol: str) -> LiquiditySnapshot | None:
        """从构造函数传入的行情对象提取快照。"""
        data = self.market_data
        if data is None:
            return None
        try:
            raw: Any = None
            if isinstance(data, pd.DataFrame):
                frame = data.copy()
                if symbol in frame.index.astype(str):
                    raw = frame.loc[symbol].to_dict()
                elif "symbol" in frame.columns or "code" in frame.columns:
                    col = "symbol" if "symbol" in frame.columns else "code"
                    matched = frame[frame[col].astype(str).map(_normalize_symbol) == symbol]
                    if not matched.empty:
                        raw = matched.iloc[0].to_dict()
            elif isinstance(data, Mapping):
                if symbol in data and isinstance(data[symbol], Mapping):
                    raw = data[symbol]
                elif _normalize_symbol(data.get("symbol", data.get("code", ""))) == symbol:
                    raw = data
            if raw is None:
                return None
            return self._snapshot_from_mapping(symbol, raw, source="provided")
        except Exception:
            return None

    @staticmethod
    def _snapshot_from_mapping(symbol: str, raw: Mapping[str, Any], source: str) -> LiquiditySnapshot:
        """将行情 mapping 转为 LiquiditySnapshot。"""
        price = _safe_float(raw.get("price", raw.get("current_price", raw.get("最新价", 10.0))), 10.0)
        amount = _safe_float(
            raw.get("avg_daily_amount", raw.get("daily_amount", raw.get("amount", raw.get("成交额", 100_000_000.0)))),
            100_000_000.0,
        )
        volume = _safe_float(
            raw.get("avg_daily_volume", raw.get("daily_volume", raw.get("volume", raw.get("成交量", 0.0)))),
            0.0,
        )
        if volume <= 0 and price > 0:
            volume = amount / price
        turnover = _safe_float(raw.get("turnover_rate", raw.get("换手率", 0.01)), 0.01)
        spread = _safe_float(raw.get("bid_ask_spread", raw.get("spread", raw.get("买卖价差", 0.003))), 0.003)
        amihud = _safe_float(raw.get("amihud", raw.get("amihud_illiq", 1e-8)), 1e-8)
        return LiquiditySnapshot(
            symbol=symbol,
            price=max(price, 0.01),
            avg_daily_amount=max(amount, 1.0),
            avg_daily_volume=max(volume, 1.0),
            turnover_rate=turnover,
            bid_ask_spread=max(spread, 0.0001),
            amihud=max(amihud, 0.0),
            name=str(raw.get("name", raw.get("名称", _load_stock_name(symbol)))),
            board=str(raw.get("board", _board_from_symbol(symbol))),
            pct_change=_safe_float(raw.get("pct_change", raw.get("涨跌幅", None)), np.nan),
            listed_days=int(_safe_float(raw.get("listed_days"), -1)) if raw.get("listed_days") is not None else None,
            limit_status=str(raw.get("limit_status", "unknown")),
            source=source,
        )

    @classmethod
    def _snapshot_from_akshare(cls, symbol: str) -> LiquiditySnapshot | None:
        """从 akshare 获取实时快照（全市场快照按 60s TTL 缓存），失败时返回 None。"""
        try:
            spot = cls._akshare_spot()
            if spot is None or spot.empty:
                return None
            code_col = "代码" if "代码" in spot.columns else "code"
            matched = spot[spot[code_col].astype(str).map(_normalize_symbol) == symbol]
            if matched.empty:
                return None
            row = matched.iloc[0]
            raw = {
                "price": row.get("最新价", 10.0),
                "amount": row.get("成交额", 100_000_000.0),
                "volume": row.get("成交量", 0.0),
                "turnover_rate": row.get("换手率", 0.01),
                "pct_change": row.get("涨跌幅", None),
                "name": row.get("名称", _load_stock_name(symbol)),
                "board": _board_from_symbol(symbol),
            }
            snapshot = LiquidityRiskManager._snapshot_from_mapping(symbol, raw, source="akshare")
            # akshare 快照通常没有盘口价差和 Amihud，这里用成交额做保守估计。
            snapshot.bid_ask_spread = 0.0015 if snapshot.avg_daily_amount > 1e9 else 0.0035
            snapshot.amihud = 1.0 / max(snapshot.avg_daily_amount, 1.0)
            # P2-Q20-fix: 补齐上市天数（按日缓存），避免新股检查恒退化出"缺少上市天数"告警。
            listed_days = cls._akshare_listed_days(symbol)
            if listed_days is not None:
                snapshot.listed_days = listed_days
            return snapshot
        except Exception:
            return None

    @classmethod
    def _akshare_spot(cls) -> pd.DataFrame | None:
        """拉取并缓存全市场快照（60s TTL），供多标的复用，避免 N 标的 N 次全市场拉取。"""
        now = time.time()
        cached = cls._AKSHARE_SPOT_CACHE
        if cached.get("frame") is not None and now - cached.get("fetched_at", 0.0) < cls._AKSHARE_SPOT_TTL:
            return cached["frame"]
        frame: pd.DataFrame | None = None
        try:
            import akshare as ak  # type: ignore

            spot = ak.stock_zh_a_spot_em()
            frame = spot if spot is not None and not spot.empty else None
        except Exception:
            frame = None
        cls._AKSHARE_SPOT_CACHE = {"fetched_at": now, "frame": frame}
        return frame

    @classmethod
    def _akshare_listed_days(cls, symbol: str) -> int | None:
        """从 akshare 个股信息接口获取上市天数（按日缓存），失败返回 None 保持原退化路径。"""
        now = time.time()
        cache = cls._AKSHARE_LISTED_DAYS_CACHE
        if now - cache.get("fetched_at", 0.0) > cls._AKSHARE_LISTED_DAYS_TTL:
            cache.clear()
            cache["fetched_at"] = now
        data = cache.get("data", {})
        if symbol in data:
            return data[symbol]
        listed_days: int | None = None
        try:
            import akshare as ak  # type: ignore

            info = ak.stock_individual_info_em(symbol=symbol)
            if info is not None and not info.empty and "item" in info.columns:
                item_row = info[info["item"] == "上市时间"]
                if not item_row.empty:
                    listing_ts = pd.Timestamp(str(item_row.iloc[0]["value"]))
                    if listing_ts.tzinfo is None:
                        listing_ts = listing_ts.tz_localize("Asia/Shanghai")
                    today = pd.Timestamp.now(tz="Asia/Shanghai").normalize()
                    # 用工作日近似交易日（忽略法定节假日），满足新股首日/限售期粗判。
                    listed_days = int(len(pd.bdate_range(start=listing_ts.normalize(), end=today)))
        except Exception:
            listed_days = None
        data[symbol] = listed_days
        cache["data"] = data
        return listed_days

    @staticmethod
    def _fallback_snapshot(symbol: str) -> LiquiditySnapshot:
        """行情缺失时的保守流动性快照。"""
        board = _board_from_symbol(symbol)
        name = _load_stock_name(symbol)
        return LiquiditySnapshot(
            symbol=symbol,
            price=10.0,
            avg_daily_amount=100_000_000.0,
            avg_daily_volume=10_000_000.0,
            turnover_rate=0.01,
            bid_ask_spread=0.003,
            amihud=1e-8,
            name=name,
            board=board,
            source="fallback",
        )


# ---------------------------------------------------------------------------
# 3. ComplianceChecker — 合规检查
# ---------------------------------------------------------------------------


class ComplianceChecker:
    """V4.1 feature: 组合与交易合规检查器。

    D6收敛登记: 独立能力保留（限额/交易限制/分散度合规，无等价实现）。
    """

    def __init__(
        self,
        market_data: Mapping[str, Any] | pd.DataFrame | None = None,
        liquidity_manager: LiquidityRiskManager | None = None,
    ) -> None:
        self.market_data = market_data
        self.liquidity_manager = liquidity_manager or LiquidityRiskManager(market_data=market_data)

    def check_position_limits(
        self,
        positions: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        limits: Mapping[str, Any],
    ) -> dict[str, Any]:
        """检查个股权重、行业集中度和回撤硬止损。"""
        frame = _positions_to_frame(positions)
        if frame.empty:
            return {"passed": True, "breaches": [], "warnings": [], "metrics": {}, "note": "无持仓"}

        max_single = _safe_float(limits.get("max_single_weight", 0.20), 0.20)
        max_industry = _safe_float(limits.get("max_industry_weight", 0.35), 0.35)
        hard_drawdown = _safe_float(limits.get("max_drawdown", limits.get("hard_stop_drawdown", 0.15)), 0.15)
        industry_map = limits.get("industry_map", {}) or {}

        if isinstance(industry_map, Mapping) and industry_map:
            # P2-Q20-fix: 先建 symbol→industry 映射再一次性赋值，避免逐行全表扫描 O(n²)；
            # industry_map 值为 None/空串时回退到持仓自带行业，避免 str(None) 变成 "None"。
            industry_lookup = {str(row["symbol"]): str(row.get("industry", "未知")) for _, row in frame.iterrows()}
            industry_lookup.update(
                {_normalize_symbol(str(k)): str(v) for k, v in industry_map.items() if v is not None and str(v).strip()}
            )
            frame["industry"] = frame["symbol"].map(lambda s: industry_lookup.get(str(s), "未知"))

        breaches: list[dict[str, Any]] = []
        warnings_list: list[dict[str, Any]] = []

        for _, row in frame.iterrows():
            symbol = str(row["symbol"])
            weight = abs(float(row["weight"]))
            if weight > max_single:
                breaches.append({
                    "type": "single_position_limit",
                    "symbol": symbol,
                    "value": round(weight, 6),
                    "limit": max_single,
                    "message": f"{symbol} 个股权重 {_fmt_pct(weight)} 超过上限 {_fmt_pct(max_single)}",
                })
            elif weight > max_single * 0.8:
                warnings_list.append({
                    "type": "single_position_near_limit",
                    "symbol": symbol,
                    "value": round(weight, 6),
                    "limit": max_single,
                    "message": f"{symbol} 个股权重接近上限",
                })

            drawdown = abs(_safe_float(row.get("current_drawdown"), 0.0))
            if drawdown > hard_drawdown:
                breaches.append({
                    "type": "hard_stop_drawdown",
                    "symbol": symbol,
                    "value": round(drawdown, 6),
                    "limit": hard_drawdown,
                    "message": f"{symbol} 当前回撤 {_fmt_pct(drawdown)} 超过硬止损 {_fmt_pct(hard_drawdown)}",
                })

        industry_weights = frame.groupby("industry")["weight"].apply(lambda x: float(np.sum(np.abs(x)))).sort_values(ascending=False)
        for industry, weight in industry_weights.items():
            if weight > max_industry:
                breaches.append({
                    "type": "industry_concentration_limit",
                    "industry": str(industry),
                    "value": round(float(weight), 6),
                    "limit": max_industry,
                    "message": f"{industry} 行业权重 {_fmt_pct(float(weight))} 超过上限 {_fmt_pct(max_industry)}",
                })
            elif weight > max_industry * 0.85:
                warnings_list.append({
                    "type": "industry_near_limit",
                    "industry": str(industry),
                    "value": round(float(weight), 6),
                    "limit": max_industry,
                    "message": f"{industry} 行业权重接近上限",
                })

        metrics = {
            "total_abs_value": float(frame["market_value"].abs().sum()),
            "max_single_weight": round(float(frame["weight"].abs().max()), 6),
            "top_industry_weight": round(float(industry_weights.iloc[0]) if not industry_weights.empty else 0.0, 6),
            "industry_weights": {str(k): round(float(v), 6) for k, v in industry_weights.items()},
            "position_count": int(len(frame)),
        }

        return {
            "passed": len(breaches) == 0,
            "breaches": breaches,
            "warnings": warnings_list,
            "metrics": metrics,
            "checked_at": _now(),
        }

    def check_trading_restrictions(self, symbol: str) -> dict[str, Any]:
        """检查 ST、涨跌停、新股限售和板块准入门槛。"""
        norm_symbol = _normalize_symbol(symbol)
        snapshot = self.liquidity_manager._get_snapshot(norm_symbol)
        name = snapshot.name or _load_stock_name(norm_symbol)
        board = snapshot.board or _board_from_symbol(norm_symbol)

        restrictions: list[dict[str, Any]] = []
        warnings_list: list[dict[str, Any]] = []

        upper_name = name.upper()
        is_st = "ST" in upper_name or "＊ST" in upper_name or "*ST" in upper_name
        is_delist_warning = any(token in name for token in ("退", "退市", "终止上市"))
        if is_st:
            restrictions.append({"type": "st_stock", "message": f"{norm_symbol} {name} 为 ST/*ST 标的，需按策略禁买或降低限额。"})
        if is_delist_warning:
            restrictions.append({"type": "delisting_warning", "message": f"{norm_symbol} {name} 存在退市警示，禁止新开仓。"})

        threshold = _limit_threshold_by_board(board)
        # P2-Q20-fix: ST/*ST 主板股涨跌停阈值为 5%，按 0.048 复核接近涨跌停；
        # 否则 ST 股涨 4.9% 不会被识别为接近涨停/跌停，仅靠 is_st 阻断买入、卖出侧告警缺失。
        if is_st and board in {"主板", "未知"}:
            threshold = 0.048
        pct_change = _safe_float(snapshot.pct_change, np.nan)
        if pct_change is not None and np.isfinite(pct_change):
            # P1-Q20-fix: 涨跌幅统一按百分数口径解析（akshare 0.35 = 0.35%），
            # 与 _safe_float 去百分号逻辑保持一致；避免 0.5% 被误判为 50% 而误报涨停、阻断买入。
            pct = pct_change / 100.0
            if pct >= threshold:
                restrictions.append({"type": "limit_up", "message": f"{norm_symbol} 接近或达到涨停，买入成交与滑点风险高。"})
            elif pct <= -threshold:
                restrictions.append({"type": "limit_down", "message": f"{norm_symbol} 接近或达到跌停，卖出流动性风险高。"})
        else:
            warnings_list.append({"type": "limit_status_unknown", "message": "缺少实时涨跌幅，无法确认涨跌停状态。"})

        if snapshot.listed_days is not None:
            if snapshot.listed_days < 5:
                restrictions.append({"type": "new_stock_first_days", "message": "上市不足 5 个交易日，波动与交易规则风险高。"})
            elif snapshot.listed_days < 60:
                warnings_list.append({"type": "new_stock_lockup", "message": "上市不足 60 日，需关注新股限售、流动性与异常波动。"})
        else:
            warnings_list.append({"type": "listed_days_unknown", "message": "缺少上市天数，无法检查新股限售期。"})

        access_required: list[str] = []
        if board == "科创板":
            access_required.append("科创板权限与适当性门槛")
        elif board == "创业板":
            access_required.append("创业板权限")
        elif board == "北交所":
            access_required.append("北交所权限与适当性门槛")
        if access_required:
            warnings_list.append({"type": "board_access", "message": " / ".join(access_required)})

        # P2-Q20-fix: 拆分"执行可行性"与"政策允许"两维度——跌停时卖出排队堆积、承接有限
        # （sell 不可行），但买入仍有承接量、可以成交，此前 limit_down 连带禁买过严；
        # 涨停只影响买入可行性；ST/新股首日属于政策禁买，不属执行可行性问题。
        execution_feasible = not any(item["type"] in {"delisting_warning"} for item in restrictions)
        buy_feasible = execution_feasible and not any(item["type"] in {"limit_up"} for item in restrictions)
        sell_feasible = execution_feasible and not any(item["type"] in {"limit_down"} for item in restrictions)
        policy_buy_allowed = not any(item["type"] in {"st_stock", "new_stock_first_days"} for item in restrictions)
        buy_allowed = buy_feasible and policy_buy_allowed
        sell_allowed = sell_feasible
        tradable = buy_allowed or sell_allowed

        return {
            "symbol": norm_symbol,
            "name": name,
            "board": board,
            "tradable": bool(tradable),
            "buy_allowed": bool(buy_allowed),
            "sell_allowed": bool(sell_allowed),
            "execution_feasible": bool(execution_feasible),
            "policy_buy_allowed": bool(policy_buy_allowed),
            "restrictions": restrictions,
            "warnings": warnings_list,
            "checks": {
                "st_or_star_st": is_st,
                "delisting_warning": is_delist_warning,
                "limit_threshold": threshold,
                "pct_change": pct_change,
                "listed_days": snapshot.listed_days,
                "requires_special_access": bool(access_required),
            },
            "checked_at": _now(),
        }

    def check_diversification(
        self,
        weights: Mapping[str, float],
        industry_map: Mapping[str, str],
    ) -> dict[str, Any]:
        """检查组合分散度，输出 HHI、有效持仓数和行业分散评分。

        D6收敛登记: 独立能力保留 —— 与 portfolio_risk.compute_concentration 异名异实现
        (输入 weights+industry_map, 输出 HHI/有效持仓/行业分散评分)。
        """
        w = _normalize_weights(weights)
        if w.empty:
            return {"passed": True, "hhi": 0.0, "effective_holdings": 0.0, "score": 100.0, "warnings": []}

        abs_w = w.abs()
        total = float(abs_w.sum())
        norm_w = abs_w / total if total > EPS else abs_w
        hhi_value = _hhi(norm_w)
        effective = _effective_number(norm_w)
        top5 = float(norm_w.sort_values(ascending=False).head(5).sum())

        industry_series = pd.Series({symbol: industry_map.get(symbol, "未知") for symbol in norm_w.index})
        industry_weights = norm_w.groupby(industry_series).sum().sort_values(ascending=False)
        industry_hhi = _hhi(industry_weights)

        # V4.1 feature: 分散度评分同时惩罚单票和行业集中。
        position_score = _clip((effective / max(len(norm_w), 1)) * 100.0, 0.0, 100.0)
        industry_score = _clip((1.0 - industry_hhi) * 125.0, 0.0, 100.0)
        top5_penalty = _clip((top5 - 0.50) * 100.0, 0.0, 40.0)
        score = _clip(position_score * 0.45 + industry_score * 0.45 + (100.0 - top5_penalty) * 0.10, 0.0, 100.0)

        warnings_list: list[str] = []
        if hhi_value > 0.18:
            warnings_list.append("个股 HHI 高于 0.18，组合集中度偏高。")
        if effective < 6:
            warnings_list.append("有效持仓数低于 6，分散度不足。")
        if not industry_weights.empty and float(industry_weights.iloc[0]) > 0.35:
            warnings_list.append("第一大行业权重超过 35%，存在行业拥挤风险。")
        if top5 > 0.60:
            warnings_list.append("前五大持仓合计超过 60%，需关注单票事件风险。")

        return {
            "passed": len(warnings_list) == 0,
            "hhi": round(hhi_value, 6),
            "effective_holdings": round(effective, 2),
            "top5_weight": round(top5, 6),
            "industry_hhi": round(industry_hhi, 6),
            "industry_diversification_score": round(industry_score, 2),
            "diversification_score": round(score, 2),
            "industry_weights": {str(k): round(float(v), 6) for k, v in industry_weights.items()},
            "warnings": warnings_list,
        }

    def run_full_compliance_check(
        self,
        positions: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        limits: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        V4.1 feature: 一次性执行持仓限制、交易限制和分散度检查。

        适合盘前风控或日报，统一输出 passed、summary 和各子模块明细。
        """
        frame = _positions_to_frame(positions)
        if frame.empty:
            return {
                "passed": True,
                "position_limits": {},
                "trading_restrictions": [],
                "diversification": {},
                "summary": "无持仓",
                "checked_at": _now(),
            }

        position_limits = self.check_position_limits(positions, limits)
        industry_map = limits.get("industry_map", {}) if isinstance(limits, Mapping) else {}
        if not industry_map:
            industry_map = {str(row["symbol"]): str(row.get("industry", "未知")) for _, row in frame.iterrows()}
        weights = {str(row["symbol"]): float(row["weight"]) for _, row in frame.iterrows()}
        diversification = self.check_diversification(weights, industry_map)

        trading_restrictions: list[dict[str, Any]] = []
        for symbol in frame["symbol"].astype(str).tolist():
            try:
                result = self.check_trading_restrictions(symbol)
                if result.get("restrictions") or result.get("warnings"):
                    trading_restrictions.append(result)
            except Exception as exc:
                trading_restrictions.append({
                    "symbol": symbol,
                    "tradable": False,
                    "buy_allowed": False,
                    "sell_allowed": False,
                    "restrictions": [{"type": "check_failed", "message": str(exc)}],
                    "warnings": [],
                })

        hard_blocks = [item for item in trading_restrictions if item.get("restrictions") and not item.get("buy_allowed", True)]
        passed = bool(position_limits.get("passed")) and bool(diversification.get("passed")) and len(hard_blocks) == 0
        return {
            "passed": passed,
            "position_limits": position_limits,
            "trading_restrictions": trading_restrictions,
            "diversification": diversification,
            "summary": {
                "position_breaches": len(position_limits.get("breaches", [])),
                "position_warnings": len(position_limits.get("warnings", [])),
                "restriction_symbols": len(trading_restrictions),
                "hard_block_symbols": len(hard_blocks),
                "diversification_score": diversification.get("diversification_score"),
            },
            "checked_at": _now(),
        }


# ---------------------------------------------------------------------------
# 4. RealTimeRiskMonitor — 实时风控监控
# ---------------------------------------------------------------------------


class RealTimeRiskMonitor:
    """V4.1 feature: 实时组合风控监控。

    D6收敛登记: 独立能力保留（实时风控状态机 + 分级预警/历史告警/限额检查，无等价实现）。
    """

    def __init__(
        self,
        thresholds: RiskThresholds | None = None,
        stress_engine: StressTestEngine | None = None,
        liquidity_manager: LiquidityRiskManager | None = None,
        compliance_checker: ComplianceChecker | None = None,
    ) -> None:
        self.thresholds = thresholds or RiskThresholds()
        self.stress_engine = stress_engine or StressTestEngine()
        self.liquidity_manager = liquidity_manager or LiquidityRiskManager()
        self.compliance_checker = compliance_checker or ComplianceChecker(liquidity_manager=self.liquidity_manager)
        self.state: PortfolioRiskState | None = None
        self.alert_history: list[dict[str, Any]] = []
        self.nav_history: list[float] = [1.0]
        self.last_portfolio: Mapping[str, Any] | None = None
        self.last_market_data: Mapping[str, Any] | pd.DataFrame | None = None

    def update(self, portfolio: Mapping[str, Any], market_data: Mapping[str, Any] | pd.DataFrame) -> None:
        """实时更新组合风险指标。"""
        self.last_portfolio = portfolio
        self.last_market_data = market_data
        frame = _positions_to_frame(portfolio)
        if frame.empty:
            self.state = PortfolioRiskState(
                timestamp=_now(),
                total_value=0.0,
                weights={},
                gross_exposure=0.0,
                net_exposure=0.0,
                daily_pnl=0.0,
                daily_return=0.0,
                var_95=0.0,
                var_99=0.0,
                cvar_95=0.0,
                cvar_99=0.0,
                max_drawdown=0.0,
                top_position_weight=0.0,
                top_industry_weight=0.0,
                hhi=0.0,
                effective_holdings=0.0,
                liquidity={"warnings": []},
                concentration={},
                stress_results={},
                factor_exposure={},
                market_snapshot={},
            )
            return

        total_value = float(frame["market_value"].sum())
        gross_abs = float(frame["market_value"].abs().sum())
        if abs(total_value) > EPS:
            gross_exposure = gross_abs / abs(total_value)
        else:
            # P1-Q20-fix: 净敞口≈0（多空对冲组合）时不能以净值为分母（会得到 0.0 低估总敞口，
            # 绕过 max_gross_exposure 去杠杆告警）。改为以单边最大市值为基准，多空各 100 万 → 2.0x。
            long_abs = float(frame.loc[frame["market_value"] > 0, "market_value"].sum())
            short_abs = float((-frame.loc[frame["market_value"] < 0, "market_value"]).sum())
            gross_exposure = gross_abs / max(long_abs, short_abs, EPS)
        net_exposure = float(frame["weight"].sum())
        weights = {str(row["symbol"]): float(row["weight"]) for _, row in frame.iterrows()}

        returns_df = self._extract_returns_frame(market_data)
        symbol_returns = self._extract_symbol_returns(market_data, frame["symbol"].astype(str).tolist())
        daily_return = float(sum(weights.get(symbol, 0.0) * ret for symbol, ret in symbol_returns.items())) if symbol_returns else 0.0
        daily_pnl = daily_return * total_value
        if np.isfinite(daily_return):
            self.nav_history.append(self.nav_history[-1] * (1.0 + daily_return))
            self.nav_history = self.nav_history[-5000:]

        var_95 = var_99 = cvar_95 = cvar_99 = 0.0
        stress_results: dict[str, Any] = {}
        if returns_df is not None and not returns_df.empty:
            try:
                w, r, _ = _align_weights_returns(weights, returns_df)
                port_ret = _portfolio_returns(w, r)
                losses = -port_ret.values
                var_95, cvar_95 = _historical_var_cvar(losses, 0.95)
                var_99, cvar_99 = _historical_var_cvar(losses, 0.99)
                for scenario_name in ("2015股灾", "2020新冠", "2022美联储加息"):
                    stress_results[scenario_name] = self.stress_engine.historical_scenario(scenario_name, weights, returns_df)
            except Exception as exc:
                stress_results["error"] = str(exc)

        nav_returns = pd.Series(self.nav_history).pct_change().dropna()
        max_dd = _max_drawdown_from_returns(nav_returns) if len(nav_returns) else 0.0

        industry_map = {str(row["symbol"]): str(row.get("industry", "未知")) for _, row in frame.iterrows()}
        concentration = self.compliance_checker.check_diversification(weights, industry_map)
        industry_weights = concentration.get("industry_weights", {})
        top_industry_weight = max([_safe_float(v) for v in industry_weights.values()], default=0.0)

        # P2-Q20-fix: 使用拆分后的显式参数，避免 threshold 同时兼作评分下限与 adv_ratio 上限的歧义。
        liquidity_warnings = self.liquidity_manager.liquidity_check(portfolio, max_adv_ratio=0.2, score_floor=20.0)
        liquidity_scores: dict[str, float] = {}
        for _, row in frame.iterrows():
            symbol = str(row["symbol"])
            try:
                assessment = self.liquidity_manager.assess_liquidity(symbol, abs(float(row["market_value"])))
                liquidity_scores[symbol] = _safe_float(assessment.get("liquidity_score"), 0.0)
            except Exception:
                liquidity_scores[symbol] = 0.0
        liquidity = {
            "warnings": liquidity_warnings,
            "scores": liquidity_scores,
            "min_score": min(liquidity_scores.values()) if liquidity_scores else 0.0,
            "avg_score": float(np.mean(list(liquidity_scores.values()))) if liquidity_scores else 0.0,
        }

        factor_exposure = self._extract_factor_exposure(market_data, weights)
        self.state = PortfolioRiskState(
            timestamp=_now(),
            total_value=total_value,
            weights=weights,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            daily_pnl=daily_pnl,
            daily_return=daily_return,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown=max_dd,
            top_position_weight=max([abs(v) for v in weights.values()], default=0.0),
            top_industry_weight=top_industry_weight,
            hhi=_safe_float(concentration.get("hhi"), 0.0),
            effective_holdings=_safe_float(concentration.get("effective_holdings"), 0.0),
            liquidity=liquidity,
            concentration=concentration,
            stress_results=stress_results,
            factor_exposure=factor_exposure,
            market_snapshot=self._summarize_market_data(market_data),
        )

    def get_alerts(self) -> list[dict[str, Any]]:
        """检查所有风控阈值，触发时产生告警。"""
        if self.state is None:
            return []
        s = self.state
        t = self.thresholds
        alerts: list[RiskAlert] = []

        if s.var_95 > t.var_95_limit:
            alerts.append(self._alert(SEVERITY_MEDIUM, "VaR95", s.var_95, t.var_95_limit, "95% VaR 超过限额", action="reduce_risk"))
        if s.var_99 > t.var_99_limit:
            alerts.append(self._alert(SEVERITY_HIGH, "VaR99", s.var_99, t.var_99_limit, "99% VaR 超过限额", action="reduce_risk"))
        if s.cvar_99 > t.cvar_99_limit:
            alerts.append(self._alert(SEVERITY_HIGH, "CVaR99", s.cvar_99, t.cvar_99_limit, "99% CVaR 超过限额", action="hedge_or_cut"))
        if s.max_drawdown > t.max_drawdown_stop:
            alerts.append(self._alert(SEVERITY_CRITICAL, "max_drawdown", s.max_drawdown, t.max_drawdown_stop, "最大回撤超过硬止损", action="stop_loss"))
        elif s.max_drawdown > t.max_drawdown_warn:
            alerts.append(self._alert(SEVERITY_MEDIUM, "max_drawdown", s.max_drawdown, t.max_drawdown_warn, "最大回撤超过预警线", action="review_drawdown"))
        if s.top_position_weight > t.max_single_weight:
            alerts.append(self._alert(SEVERITY_HIGH, "top_position_weight", s.top_position_weight, t.max_single_weight, "单票权重超过上限", action="trim_position"))
        if s.top_industry_weight > t.max_industry_weight:
            alerts.append(self._alert(SEVERITY_HIGH, "top_industry_weight", s.top_industry_weight, t.max_industry_weight, "行业集中度超过上限", action="rebalance_industry"))
        if s.hhi > t.hhi_warn:
            alerts.append(self._alert(SEVERITY_MEDIUM, "hhi", s.hhi, t.hhi_warn, "HHI 集中度偏高", action="diversify"))
        if 0 < s.effective_holdings < t.effective_holdings_min:
            alerts.append(self._alert(SEVERITY_MEDIUM, "effective_holdings", s.effective_holdings, t.effective_holdings_min, "有效持仓数不足", action="diversify"))
        if _safe_float(s.liquidity.get("min_score"), 100.0) < t.min_liquidity_score:
            alerts.append(self._alert(SEVERITY_HIGH, "liquidity_score", _safe_float(s.liquidity.get("min_score"), 0.0), t.min_liquidity_score, "最低流动性评分低于阈值", action="slow_liquidation"))
        for item in s.liquidity.get("warnings", []):
            raw_days = item.get("days_to_liquidate_20pct", "unknown")
            # P2-Q20-fix: 仅当 20% 参与率清仓天数超过阈值时才生成 medium/high 告警并按天数分级；
            # 其余（如仅评分低、与天数无关）降为 info 仅入报告，阈值字段不再纯装饰。
            days20 = float("inf") if isinstance(raw_days, str) and raw_days.strip().lower() == "inf" else _safe_float(raw_days, 0.0)
            if days20 > t.max_liquidation_days_20pct:
                severity = SEVERITY_HIGH if (not np.isfinite(days20) or days20 > t.max_liquidation_days_20pct * 2) else SEVERITY_MEDIUM
                alerts.append(RiskAlert(
                    severity=severity,
                    metric="liquidity_warning",
                    value="inf" if not np.isfinite(days20) else round(days20, 2),
                    threshold=t.max_liquidation_days_20pct,
                    message="; ".join(item.get("reasons", [])),
                    symbol=item.get("symbol"),
                    action="check_execution_plan",
                ))
            else:
                alerts.append(RiskAlert(
                    severity=SEVERITY_INFO,
                    metric="liquidity_note",
                    value=raw_days,
                    threshold=t.max_liquidation_days_20pct,
                    message="; ".join(item.get("reasons", [])),
                    symbol=item.get("symbol"),
                    action="check_execution_plan",
                ))
        if s.gross_exposure > t.max_gross_exposure:
            alerts.append(self._alert(SEVERITY_HIGH, "gross_exposure", s.gross_exposure, t.max_gross_exposure, "总敞口超过限额", action="delever"))
        if abs(s.net_exposure) > t.max_net_exposure_abs:
            alerts.append(self._alert(SEVERITY_MEDIUM, "net_exposure", s.net_exposure, t.max_net_exposure_abs, "净敞口超过限额", action="rebalance_net"))

        output = [alert.to_dict() for alert in alerts]
        # P2-Q20-fix: 历史记录收敛到 commit_alerts() 并按内容去重，避免
        # risk_dashboard / generate_daily_report / generate_json_report 多次调用
        # get_alerts 导致 alert_history 4→8→12 重复累计、下游告警入库/统计重复计数。
        self.commit_alerts(output)
        return output

    @staticmethod
    def _alert_key(alert: Mapping[str, Any]) -> tuple[Any, ...]:
        """告警去重键（不含 timestamp），内容一致即视为同一条告警。"""
        return (
            alert.get("severity"),
            alert.get("metric"),
            alert.get("symbol"),
            alert.get("message"),
            str(alert.get("value")),
            str(alert.get("threshold")),
        )

    def commit_alerts(self, alerts: Sequence[Mapping[str, Any]] | None = None) -> None:
        """显式将告警写入历史（按内容去重），供需要持久化告警的调用方使用。"""
        if alerts is None:
            alerts = self.get_alerts()
        seen = {self._alert_key(a) for a in self.alert_history}
        for alert in alerts:
            key = self._alert_key(alert)
            if key not in seen:
                self.alert_history.append(dict(alert))
                seen.add(key)
        self.alert_history = self.alert_history[-1000:]

    @staticmethod
    def _alert(
        severity: str,
        metric: str,
        value: float,
        threshold: float,
        message: str,
        action: str,
    ) -> RiskAlert:
        """构造统一告警对象。"""
        return RiskAlert(
            severity=severity,
            metric=metric,
            value=round(float(value), 6),
            threshold=round(float(threshold), 6),
            message=message,
            action=action,
        )

    def check_limits(self, exposure_limits: dict[str, Any]) -> list[dict[str, Any]]:
        """风险限额检查，包括因子暴露限额和 VaR 限额。"""
        if self.state is None:
            return []
        s = self.state
        breaches: list[dict[str, Any]] = []

        scalar_map = {
            "gross_exposure": s.gross_exposure,
            "net_exposure_abs": abs(s.net_exposure),
            "var_95": s.var_95,
            "var_99": s.var_99,
            "cvar_99": s.cvar_99,
            "max_drawdown": s.max_drawdown,
            "top_position_weight": s.top_position_weight,
            "top_industry_weight": s.top_industry_weight,
        }
        for metric, value in scalar_map.items():
            if metric in exposure_limits:
                limit = _safe_float(exposure_limits[metric], float("inf"))
                if value > limit:
                    breaches.append({
                        "metric": metric,
                        "value": round(float(value), 6),
                        "limit": limit,
                        "severity": SEVERITY_HIGH,
                        "message": f"{metric} 超过限额",
                    })

        factor_limits = exposure_limits.get("factor_exposure", {}) or exposure_limits.get("factors", {})
        if isinstance(factor_limits, Mapping):
            for factor, limit in factor_limits.items():
                value = _safe_float(s.factor_exposure.get(str(factor)), 0.0)
                limit_value = abs(_safe_float(limit, float("inf")))
                if abs(value) > limit_value:
                    breaches.append({
                        "metric": "factor_exposure",
                        "factor": str(factor),
                        "value": round(value, 6),
                        "limit": limit_value,
                        "severity": SEVERITY_MEDIUM,
                        "message": f"因子 {factor} 暴露超过限额",
                    })
        return breaches

    def risk_dashboard(self) -> dict[str, Any]:
        """
        V4.1 feature: 输出轻量级实时风险看板。

        dashboard 仅返回前端、调度任务和告警系统常用字段，避免直接传递完整 state。
        """
        if self.state is None:
            return {"status": "empty", "alerts": [], "metrics": {}, "top_risks": []}
        alerts = self.get_alerts()
        state = self.state
        severity_rank = {SEVERITY_CRITICAL: 5, SEVERITY_HIGH: 4, SEVERITY_MEDIUM: 3, SEVERITY_LOW: 2, SEVERITY_INFO: 1}
        max_severity = max((severity_rank.get(str(alert.get("severity")), 0) for alert in alerts), default=0)
        status = "critical" if max_severity >= 5 else "warning" if max_severity >= 3 else "ok"
        return {
            "status": status,
            "timestamp": state.timestamp,
            "alerts": alerts[:20],
            "metrics": {
                "total_value": state.total_value,
                "daily_return": state.daily_return,
                "daily_pnl": state.daily_pnl,
                "var_95": state.var_95,
                "var_99": state.var_99,
                "cvar_99": state.cvar_99,
                "max_drawdown": state.max_drawdown,
                "gross_exposure": state.gross_exposure,
                "net_exposure": state.net_exposure,
                "top_position_weight": state.top_position_weight,
                "top_industry_weight": state.top_industry_weight,
                "liquidity_min_score": state.liquidity.get("min_score"),
                "effective_holdings": state.effective_holdings,
            },
            "top_risks": self._top_risk_items(state, alerts),
        }

    @staticmethod
    def _top_risk_items(state: PortfolioRiskState, alerts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """提取看板顶部风险事项。"""
        items: list[dict[str, Any]] = []
        for alert in alerts[:10]:
            items.append({
                "type": "alert",
                "severity": alert.get("severity"),
                "metric": alert.get("metric"),
                "message": alert.get("message"),
            })
        if state.stress_results:
            stress_losses: list[tuple[str, float]] = []
            for name, result in state.stress_results.items():
                if isinstance(result, Mapping) and "portfolio_loss" in result:
                    stress_losses.append((name, _safe_float(result.get("portfolio_loss"), 0.0)))
            if stress_losses:
                name, loss = max(stress_losses, key=lambda item: item[1])
                items.append({
                    "type": "stress",
                    "severity": SEVERITY_MEDIUM,
                    "metric": name,
                    "message": f"最差压力场景损失 {_fmt_pct(loss)}",
                })
        return items

    def reset_history(self, keep_last_state: bool = True) -> None:
        """
        V4.1 feature: 重置监控历史。

        Args:
            keep_last_state: True 时保留最新 state，只清告警与净值历史；False 时清空 state。
        """
        self.alert_history.clear()
        self.nav_history = [1.0]
        if not keep_last_state:
            self.state = None

    @staticmethod
    def _extract_returns_frame(market_data: Mapping[str, Any] | pd.DataFrame) -> pd.DataFrame | None:
        """从 market_data 提取历史收益率矩阵。"""
        if isinstance(market_data, pd.DataFrame):
            return _as_numeric_frame(market_data)
        if isinstance(market_data, Mapping):
            for key in ("returns", "returns_df", "historical_returns"):
                value = market_data.get(key)
                if isinstance(value, (pd.DataFrame, pd.Series, Mapping)):
                    try:
                        return _as_numeric_frame(value)
                    except Exception as e:
                        logging.getLogger(__name__).error(f"[risk_management_pro] 操作失败: {e}", exc_info=True)
                        continue
        return None

    @staticmethod
    def _extract_symbol_returns(market_data: Mapping[str, Any] | pd.DataFrame, symbols: Sequence[str]) -> dict[str, float]:
        """提取当日单标的收益率。"""
        result: dict[str, float] = {}
        if isinstance(market_data, pd.DataFrame):
            if "symbol" in market_data.columns and ("return" in market_data.columns or "pct_change" in market_data.columns):
                ret_col = "return" if "return" in market_data.columns else "pct_change"
                for _, row in market_data.iterrows():
                    sym = _normalize_symbol(row.get("symbol"))
                    val = _safe_float(row.get(ret_col), 0.0)
                    result[sym] = val / 100.0 if abs(val) > 1 else val
            elif len(market_data) >= 1:
                last = market_data.iloc[-1]
                for sym in symbols:
                    if sym in last.index:
                        result[sym] = _safe_float(last[sym], 0.0)
            return {s: result.get(s, 0.0) for s in symbols if s in result}
        if isinstance(market_data, Mapping):
            direct = market_data.get("daily_returns", market_data.get("returns_today", None))
            if isinstance(direct, Mapping):
                for sym, val in direct.items():
                    x = _safe_float(val, 0.0)
                    result[_normalize_symbol(sym)] = x / 100.0 if abs(x) > 1 else x
            else:
                for sym in symbols:
                    raw = market_data.get(sym, {})
                    if isinstance(raw, Mapping):
                        val = _safe_float(raw.get("return", raw.get("pct_change", raw.get("涨跌幅", 0.0))), 0.0)
                        result[sym] = val / 100.0 if abs(val) > 1 else val
        return result

    @staticmethod
    def _extract_factor_exposure(market_data: Mapping[str, Any] | pd.DataFrame, weights: Mapping[str, float]) -> dict[str, float]:
        """从 market_data 中提取或汇总组合因子暴露。"""
        if not isinstance(market_data, Mapping):
            return {}
        raw = market_data.get("factor_exposure", market_data.get("factor_exposures", {}))
        if isinstance(raw, Mapping) and all(isinstance(v, (int, float, np.number)) for v in raw.values()):
            return {str(k): float(v) for k, v in raw.items()}
        if isinstance(raw, pd.DataFrame):
            try:
                w = _normalize_weights(weights)
                exposures = raw.copy()
                exposures.index = [_normalize_symbol(i) for i in exposures.index]
                common = [s for s in w.index if s in exposures.index]
                if common:
                    combo = exposures.loc[common].T.dot(w.reindex(common).fillna(0.0))
                    return {str(k): float(v) for k, v in combo.items()}
            except Exception:
                return {}
        return {}

    @staticmethod
    def _summarize_market_data(market_data: Mapping[str, Any] | pd.DataFrame) -> dict[str, Any]:
        """生成行情快照摘要，避免在 state 中存储过大的原始数据。"""
        if isinstance(market_data, pd.DataFrame):
            return {"type": "DataFrame", "rows": int(len(market_data)), "columns": list(map(str, market_data.columns[:20]))}
        if isinstance(market_data, Mapping):
            summary = {"type": "Mapping", "keys": list(map(str, list(market_data.keys())[:20]))}
            for key in ("market_return", "index_return", "volatility", "regime"):
                if key in market_data:
                    summary[key] = market_data[key]
            return summary
        return {"type": type(market_data).__name__}


# ---------------------------------------------------------------------------
# 5. RiskReportGenerator — 风控报告生成
# ---------------------------------------------------------------------------


class RiskReportGenerator:
    """V4.1 feature: 中文风控日报生成器。

    D6收敛登记: 独立能力保留 —— 日报/JSON快照, 与 portfolio_risk.full_risk_report 异名异实现。
    """

    def __init__(
        self,
        monitor: RealTimeRiskMonitor | None = None,
        stress_engine: StressTestEngine | None = None,
        liquidity_manager: LiquidityRiskManager | None = None,
        compliance_checker: ComplianceChecker | None = None,
    ) -> None:
        self.monitor = monitor or RealTimeRiskMonitor()
        self.stress_engine = stress_engine or self.monitor.stress_engine
        self.liquidity_manager = liquidity_manager or self.monitor.liquidity_manager
        self.compliance_checker = compliance_checker or self.monitor.compliance_checker

    def generate_daily_report(self) -> str:
        """生成风控日报。"""
        state = self.monitor.state
        if state is None:
            return "风控日报 (V4.1 feature)\n\n暂无实时风控状态。请先调用 RealTimeRiskMonitor.update(portfolio, market_data)。"

        alerts = self.monitor.get_alerts()
        lines: list[str] = []
        lines.append("=" * 72)
        lines.append("专业风控日报 (V4.1 feature)")
        lines.append("=" * 72)
        lines.append(f"生成时间: {_now()}")
        lines.append(f"状态时间: {state.timestamp}")
        lines.append("")

        lines.extend(self._section_overview(state))
        lines.extend(self._section_risk_metrics(state))
        lines.extend(self._section_concentration_liquidity(state))
        lines.extend(self._section_stress(state))
        lines.extend(self._section_alerts(alerts))
        lines.extend(self._section_recommendations(state, alerts))
        lines.append("=" * 72)
        return "\n".join(lines)

    def generate_json_report(self) -> dict[str, Any]:
        """
        V4.1 feature: 生成机器可读 JSON 风控报告。

        与中文日报互补，适合 API 返回、告警系统入库和审计归档。
        """
        state = self.monitor.state
        if state is None:
            return {"status": "empty", "generated_at": _now(), "message": "暂无实时风控状态"}
        alerts = self.monitor.get_alerts()
        return {
            "status": "ok" if not alerts else "warning",
            "generated_at": _now(),
            "state_timestamp": state.timestamp,
            "overview": {
                "total_value": state.total_value,
                "daily_return": state.daily_return,
                "daily_pnl": state.daily_pnl,
                "gross_exposure": state.gross_exposure,
                "net_exposure": state.net_exposure,
                "position_count": len(state.weights),
            },
            "risk_metrics": {
                "var_95": state.var_95,
                "var_99": state.var_99,
                "cvar_95": state.cvar_95,
                "cvar_99": state.cvar_99,
                "max_drawdown": state.max_drawdown,
            },
            "concentration": state.concentration,
            "liquidity": state.liquidity,
            "stress_results": state.stress_results,
            "factor_exposure": state.factor_exposure,
            "alerts": alerts,
            "recommendations": self._recommendation_list(state, alerts),
        }

    def save_daily_report(self, path: str | Path, fmt: str = "txt") -> Path:
        """
        V4.1 feature: 保存风控日报到本地文件。

        Args:
            path: 输出路径。
            fmt: txt 或 json。其他值按 txt 处理。
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if fmt.lower().strip() == "json":
            output_path.write_text(json.dumps(self.generate_json_report(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        else:
            output_path.write_text(self.generate_daily_report(), encoding="utf-8")
        return output_path

    @staticmethod
    def _recommendation_list(state: PortfolioRiskState, alerts: Sequence[Mapping[str, Any]]) -> list[str]:
        """将中文建议章节转成 JSON 列表。"""
        lines = RiskReportGenerator._section_recommendations(state, alerts)
        recommendations: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("-"):
                recommendations.append(stripped.lstrip("- "))
        return recommendations

    @staticmethod
    def _section_overview(state: PortfolioRiskState) -> list[str]:
        """组合总览章节。"""
        lines = ["一、组合风险总览", "-" * 72]
        lines.append(f"组合市值: {_fmt_money(state.total_value)}")
        lines.append(f"当日收益率: {_fmt_pct(state.daily_return)}")
        lines.append(f"当日盈亏: {_fmt_money(state.daily_pnl)}")
        lines.append(f"总敞口: {state.gross_exposure:.2f}x")
        lines.append(f"净敞口: {state.net_exposure:.2f}x")
        lines.append(f"持仓数量: {len(state.weights)}")
        lines.append("")
        return lines

    @staticmethod
    def _section_risk_metrics(state: PortfolioRiskState) -> list[str]:
        """VaR/CVaR/最大回撤章节。"""
        lines = ["二、VaR / CVaR / 回撤", "-" * 72]
        lines.append(f"VaR 95%: {_fmt_pct(state.var_95)}")
        lines.append(f"VaR 99%: {_fmt_pct(state.var_99)}")
        lines.append(f"CVaR 95%: {_fmt_pct(state.cvar_95)}")
        lines.append(f"CVaR 99%: {_fmt_pct(state.cvar_99)}")
        lines.append(f"最大回撤: {_fmt_pct(state.max_drawdown)}")
        lines.append("")
        return lines

    @staticmethod
    def _section_concentration_liquidity(state: PortfolioRiskState) -> list[str]:
        """集中度与流动性章节。"""
        lines = ["三、集中度 / 流动性", "-" * 72]
        lines.append(f"最大单票权重: {_fmt_pct(state.top_position_weight)}")
        lines.append(f"最大行业权重: {_fmt_pct(state.top_industry_weight)}")
        lines.append(f"HHI 集中度: {state.hhi:.4f}")
        lines.append(f"有效持仓数: {state.effective_holdings:.2f}")
        lines.append(f"流动性最低分: {_safe_float(state.liquidity.get('min_score'), 0.0):.2f}")
        lines.append(f"流动性平均分: {_safe_float(state.liquidity.get('avg_score'), 0.0):.2f}")
        warnings_list = state.liquidity.get("warnings", [])
        if warnings_list:
            lines.append("流动性关注标的:")
            for item in warnings_list[:8]:
                reasons = "; ".join(item.get("reasons", []))
                lines.append(f"  - {item.get('symbol')}: {reasons}")
        else:
            lines.append("流动性关注标的: 无")
        lines.append("")
        return lines

    @staticmethod
    def _section_stress(state: PortfolioRiskState) -> list[str]:
        """压力测试章节。"""
        lines = ["四、压力测试结果", "-" * 72]
        if not state.stress_results:
            lines.append("暂无压力测试结果；需要在 market_data 中提供 returns / returns_df。")
        else:
            for name, result in state.stress_results.items():
                if not isinstance(result, Mapping):
                    continue
                if "portfolio_loss" in result:
                    lines.append(
                        f"{name}: 组合损失 {_fmt_pct(_safe_float(result.get('portfolio_loss')))}, "
                        f"最大回撤 {_fmt_pct(_safe_float(result.get('max_drawdown')))}, "
                        f"来源 {result.get('source')}"
                    )
                elif "error" in result:
                    lines.append(f"{name}: 压力测试失败 - {result.get('error')}")
        lines.append("")
        return lines

    @staticmethod
    def _section_alerts(alerts: Sequence[Mapping[str, Any]]) -> list[str]:
        """告警汇总章节。"""
        lines = ["五、告警汇总", "-" * 72]
        if not alerts:
            lines.append("当前无触发告警。")
        else:
            severity_order = {SEVERITY_CRITICAL: 0, SEVERITY_HIGH: 1, SEVERITY_MEDIUM: 2, SEVERITY_LOW: 3, SEVERITY_INFO: 4}
            sorted_alerts = sorted(alerts, key=lambda x: severity_order.get(str(x.get("severity")), 9))
            for alert in sorted_alerts[:20]:
                symbol = f"[{alert.get('symbol')}] " if alert.get("symbol") else ""
                lines.append(
                    f"  - {alert.get('severity')} | {symbol}{alert.get('metric')}: "
                    f"{alert.get('message')} (值={alert.get('value')}, 阈值={alert.get('threshold')})"
                )
        lines.append("")
        return lines

    @staticmethod
    def _section_recommendations(state: PortfolioRiskState, alerts: Sequence[Mapping[str, Any]]) -> list[str]:
        """改进建议章节。"""
        lines = ["六、改进建议", "-" * 72]
        recommendations: list[str] = []
        alert_metrics = {str(a.get("metric")) for a in alerts}
        if {"VaR95", "VaR99", "CVaR99"} & alert_metrics:
            recommendations.append("降低高 beta 或高相关持仓，必要时使用指数期货/ETF 对冲系统性风险。")
        if "max_drawdown" in alert_metrics:
            recommendations.append("执行回撤分层预案：暂停新增风险、复核止损线、逐笔检查亏损来源。")
        if "top_position_weight" in alert_metrics:
            recommendations.append("将最大单票仓位拆分至策略上限以内，避免单一事件主导组合净值。")
        if "top_industry_weight" in alert_metrics:
            recommendations.append("降低第一大行业暴露，补充低相关行业或现金缓冲。")
        if "liquidity_score" in alert_metrics or "liquidity_warning" in alert_metrics:
            recommendations.append("对低流动性标的设置更长清仓周期，交易参与率建议控制在 10%-20%。")
        if state.effective_holdings < 6:
            recommendations.append("提高有效持仓数量，优先增加低相关、基本面质量稳定的标的。")
        if not recommendations:
            recommendations.append("当前核心指标未触发硬性限制，继续维持日内监控并复核压力测试假设。")
        for item in recommendations:
            lines.append(f"  - {item}")
        lines.append("")
        return lines


# ---------------------------------------------------------------------------
# 便捷函数与模块导出
# ---------------------------------------------------------------------------


def quick_risk_snapshot(
    portfolio: Mapping[str, Any],
    market_data: Mapping[str, Any] | pd.DataFrame,
    thresholds: RiskThresholds | None = None,
) -> dict[str, Any]:
    """
    V4.1 feature: 一行式组合风控快照。

    该函数便于 CLI、任务调度或 notebook 快速调用，不要求调用方手动实例化多个类。
    """
    monitor = RealTimeRiskMonitor(thresholds=thresholds)
    monitor.update(portfolio, market_data)
    alerts = monitor.get_alerts()
    state = monitor.state
    if state is None:
        return {"state": None, "alerts": alerts}
    return {
        "state": {
            "timestamp": state.timestamp,
            "total_value": state.total_value,
            "daily_return": state.daily_return,
            "var_95": state.var_95,
            "var_99": state.var_99,
            "cvar_99": state.cvar_99,
            "max_drawdown": state.max_drawdown,
            "top_position_weight": state.top_position_weight,
            "top_industry_weight": state.top_industry_weight,
            "liquidity_min_score": state.liquidity.get("min_score"),
        },
        "alerts": alerts,
    }


def build_default_risk_stack(
    thresholds: RiskThresholds | None = None,
    market_data: Mapping[str, Any] | pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    V4.1 feature: 构建默认专业风控组件栈。

    返回五大模块实例，确保共享阈值、行情源与缓存，减少上层服务重复初始化。
    """
    liquidity = LiquidityRiskManager(market_data=market_data)
    compliance = ComplianceChecker(market_data=market_data, liquidity_manager=liquidity)
    stress = StressTestEngine()
    monitor = RealTimeRiskMonitor(
        thresholds=thresholds,
        stress_engine=stress,
        liquidity_manager=liquidity,
        compliance_checker=compliance,
    )
    reporter = RiskReportGenerator(
        monitor=monitor,
        stress_engine=stress,
        liquidity_manager=liquidity,
        compliance_checker=compliance,
    )
    return {
        "stress_engine": stress,
        "liquidity_manager": liquidity,
        "compliance_checker": compliance,
        "risk_monitor": monitor,
        "report_generator": reporter,
    }


def export_risk_snapshot_json(
    portfolio: Mapping[str, Any],
    market_data: Mapping[str, Any] | pd.DataFrame,
    path: str | Path,
    thresholds: RiskThresholds | None = None,
) -> Path:
    """
    V4.1 feature: 快速生成并导出 JSON 风控快照。

    适合定时任务或回测后批量审计，内部只执行一次 update 和一次报告生成。
    """
    stack = build_default_risk_stack(thresholds=thresholds, market_data=market_data)
    monitor: RealTimeRiskMonitor = stack["risk_monitor"]
    reporter: RiskReportGenerator = stack["report_generator"]
    monitor.update(portfolio, market_data)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(reporter.generate_json_report(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return output_path


__all__ = [
    "HistoricalScenario",
    "LiquiditySnapshot",
    "RiskAlert",
    "RiskThresholds",
    "PortfolioRiskState",
    "StressTestEngine",
    "LiquidityRiskManager",
    "ComplianceChecker",
    "RealTimeRiskMonitor",
    "RiskReportGenerator",
    "quick_risk_snapshot",
    "build_default_risk_stack",
    "export_risk_snapshot_json",
]


if __name__ == "__main__":  # pragma: no cover - 手工 smoke test
    # V4.1 feature: 简单演示，避免模块被直接运行时没有输出。
    demo_returns = pd.DataFrame(
        {
            "000001": [0.01, -0.02, 0.003, -0.015, 0.004],
            "600000": [-0.004, -0.01, 0.008, -0.02, 0.005],
        },
        index=pd.date_range("2022-01-01", periods=5),
    )
    demo_portfolio = {
        "000001": {"market_value": 600_000, "industry": "银行", "current_drawdown": 0.03},
        "600000": {"market_value": 400_000, "industry": "银行", "current_drawdown": 0.02},
    }
    monitor = RealTimeRiskMonitor()
    monitor.update(demo_portfolio, {"returns": demo_returns, "daily_returns": {"000001": 0.002, "600000": -0.001}})
    print(RiskReportGenerator(monitor).generate_daily_report())
