"""
registry.py — V7.0 因子统一注册表（FactorMeta）
=================================================
多因子七步法第 1 步：统一注册。
- 因子元数据：name/category/freq/universe/handler/data_deps/params
- 与 legacy zoo.py 兼容：import_from_zoo() 将 zoo 的 40+ 因子导入本注册表
- 自动发现：autodiscover() 扫描 factors 包内 @register_factor 装饰器
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。 V7 统一注册表(元数据层)，不承载因子实现。
"""
from __future__ import annotations

import importlib
import pkgutil
import sys
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from quant_system.market_forecast._support.common.logger import get_logger

log = get_logger("qv6.factor.registry")

# 15 大类（与 V7.0 方案一致）
CATEGORIES = (
    "momentum",   # 动量/趋势
    "reversal",   # 反转
    "volatility",  # 波动
    "liquidity",  # 流动性
    "quality",    # 质量
    "growth",     # 成长
    "value",      # 估值
    "flow",       # 资金
    "position",   # 筹码
    "event",      # 事件
    "sentiment",  # 情绪
    "industry",   # 行业/风格
    "macro",      # 宏观映射
    "bond",       # 债市映射
    "commodity",  # 商品映射
)

# 频率
FREQ_DAILY = "daily"
FREQ_WEEKLY = "weekly"
FREQ_QUARTERLY = "quarterly"
FREQ_EVENT = "event"


@dataclass
class FactorMeta:
    """因子元数据。"""
    name: str
    category: str
    freq: str = FREQ_DAILY
    universe: str = "all_ashare"   # all_ashare / hs300 / zz500 / cb / etf / index
    handler: Callable | None = None
    data_deps: list = field(default_factory=list)   # warehouse 表名
    default_params: dict = field(default_factory=dict)
    description: str = ""
    direction: int = 1        # 1=越大越看多, -1=越小越看多
    active: bool = True
    source: str = ""          # 来源模块（zoo/baostock/...）
    added_at: str = ""
    health_rules: dict = field(default_factory=dict)  # 健康断言规则 {min,max,max_missing_ratio,constant_ok}
    base_direction: int | None = None  # 定义时基准方向（校准覆盖只改 direction，不改 base）

    def __post_init__(self) -> None:
        # 2026-08-14: 方向校准以 base_direction 为基线，否则覆盖叠加导致振荡
        if self.base_direction is None:
            self.base_direction = self.direction

    def compute(self, data: dict[str, pd.DataFrame], **kwargs) -> pd.Series:
        """计算因子值（横截面 Series，index=股票代码）。"""
        if self.handler is None:
            raise ValueError(f"因子 {self.name} 无 handler")
        params = {**self.default_params, **kwargs}
        return self.handler(data, **params)


_REGISTRY: dict[str, FactorMeta] = {}


def register_factor(
    name: str | None = None,
    *,
    category: str = "momentum",
    freq: str = FREQ_DAILY,
    universe: str = "all_ashare",
    data_deps: list | None = None,
    params: dict | None = None,
    description: str = "",
    direction: int = 1,
    active: bool = True,
    source: str = "",
    health_rules: dict | None = None,
):
    """装饰器：注册因子函数（handler 签名: fn(data: dict, **params) -> pd.Series）。
    D4收敛登记: 跨模块同名异签名-不强迁保留
    """
    def decorator(func):
        fname = name or func.__name__
        meta = FactorMeta(
            name=fname, category=category, freq=freq, universe=universe,
            handler=func, data_deps=data_deps or [], default_params=params or {},
            description=description, direction=direction, active=active,
            source=source or func.__module__.split(".")[-1],
            health_rules=health_rules or {},
        )
        _REGISTRY[fname] = meta
        return func
    return decorator


def get_factor(name: str) -> FactorMeta | None:
    """D4收敛登记: 跨模块同名异签名-不强迁保留"""
    return _REGISTRY.get(name)


def list_factors(category: str | None = None, active_only: bool = True,
                 freq: str | None = None) -> list[FactorMeta]:
    """D4收敛登记: 跨模块同名异签名-不强迁保留"""
    out = []
    for m in _REGISTRY.values():
        if active_only and not m.active:
            continue
        if category and m.category != category:
            continue
        if freq and m.freq != freq:
            continue
        out.append(m)
    return out


def info(name: str) -> dict:
    m = _REGISTRY.get(name)
    if m is None:
        return {}
    return {
        "name": m.name, "category": m.category, "freq": m.freq,
        "universe": m.universe, "direction": m.direction,
        "active": m.active, "data_deps": m.data_deps,
        "description": m.description, "source": m.source,
    }


def summary() -> pd.DataFrame:
    """因子池总览（按大类统计）。"""
    rows = [
        {"name": m.name, "category": m.category, "freq": m.freq,
         "universe": m.universe, "direction": m.direction, "active": m.active,
         "source": m.source}
        for m in _REGISTRY.values()
    ]
    return pd.DataFrame(rows)


def count_by_category() -> dict[str, int]:
    out = {}
    for m in _REGISTRY.values():
        if m.active:
            out[m.category] = out.get(m.category, 0) + 1
    return out


def import_from_zoo() -> int:
    """将 legacy zoo.py 的因子导入本注册表（幂等，不覆盖已有）。"""
    from quant_system.ic_factors import zoo
    zoo.register_defaults()
    # zoo 内建因子 → 按名称猜测大类
    guess = {
        "mom": "momentum", "rev": "reversal", "ma_": "momentum",
        "macd": "momentum", "boll": "volatility", "cci": "reversal",
        "rsi": "reversal", "high_low": "momentum", "bias": "reversal",
        "new_high": "momentum", "new_low": "momentum",
        "volume": "liquidity", "vol_": "volatility", "obv": "liquidity",
        "realized": "volatility", "downside": "volatility", "max_dd": "volatility",
        "beta": "volatility", "atr": "volatility",
        "amount": "liquidity", "amihud": "liquidity", "turnover": "liquidity",
        "spread": "liquidity", "kdj": "reversal", "williams": "reversal",
        "dmi": "momentum", "donchian": "momentum", "ma_trend": "momentum",
    }
    # V10 审计 Low-1 修复：vol_ 前缀先排除 volume_（volume_trend 等是量能），
    # vol_change/vol_20d/vol_ratio 归 volatility 更合理（combination.py 口径）
    guess_vol_override = {"vol_change": "volatility", "vol_20d": "volatility",
                          "volume_trend": "liquidity", "volume_surge": "liquidity",
                          "volume_ratio": "liquidity", "vol_ratio": "volatility"}
    n = 0
    for name in zoo.list_factors(include_inactive=True):
        if name in _REGISTRY:
            continue
        f = zoo.get_factor(name)
        cat = guess_vol_override.get(name, "momentum")
        for k, v in guess.items():
            if name.startswith(k):
                cat = v
                break
        _REGISTRY[name] = FactorMeta(
            name=name, category=cat, freq=FREQ_DAILY, universe="all_ashare",
            handler=f.func, description=f.description,
            direction=f.direction, active=f.active, source="zoo",
            base_direction=getattr(f, "base_direction", f.direction),
        )
        n += 1
    log.info(f"从 zoo 导入因子 {n} 个，注册表总计 {len(_REGISTRY)}")
    return n


def autodiscover(package: str = "quant_system.ic_factors") -> int:
    """扫描包内所有模块，收集 @register_factor 装饰器注册的因子。"""
    pkg = importlib.import_module(package)
    before = len(_REGISTRY)
    for mod in pkgutil.iter_modules(pkg.__path__):
        if mod.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{package}.{mod.name}")
        except Exception as e:  # noqa: BLE001 - 单模块失败不影响整体
            log.warning(f"因子模块加载失败 {mod.name}: {e}")
    return len(_REGISTRY) - before


_OVERRIDES_APPLIED = False


def ensure_loaded() -> None:
    """确保注册表就绪（幂等）：先 autodiscover，再补 zoo 导入，最后应用方向校准覆盖。

    2026-08-14 审计修复: 旧版 `if _REGISTRY: return` 早退——注册表非空时（多数场景，
    因 autodiscover 已提前执行）①方向覆盖永不应用（65 个 v7 因子覆盖静默失效）、
    ②import_from_zoo 被跳过（40 个 zoo 因子缺失）。
    现拆分: autodiscover 按需执行；import_from_zoo 幂等必执行（只补不覆盖）；
    覆盖独立于加载执行一次（_OVERRIDES_APPLIED 防重复）。"""
    if not _REGISTRY:
        autodiscover()
    import_from_zoo()
    _apply_direction_overrides_once()


def _apply_direction_overrides_once() -> None:
    """应用 config/factor_direction_overrides.json（进程内一次，幂等）。"""
    global _OVERRIDES_APPLIED
    if _OVERRIDES_APPLIED:
        return
    _OVERRIDES_APPLIED = True
    try:
        from quant_system.validate_apply_direction import apply_overrides
        from quant_system.ic_factors import zoo
        n = apply_overrides(registry_module=sys.modules[__name__], zoo_module=zoo)
        if n:
            log.info(f"方向校准覆盖已应用: {n} 个因子")
    except Exception as e:  # noqa: BLE001 - 覆盖失败不阻断主流程
        log.warning(f"方向校准覆盖未应用: {e}")
