"""analysis_core.pipeline_tier — 冷热分离注册表与查询工具（V12.1 阶段5）。

职责边界：
  本模块只提供 tier 标签注册/查询/冷模块清单，不改变任何现有调度。
  运行模式语义：
    hot  — 在线核心决策（fusion / battle_map / execution）
    warm — 分钟级日终融合（情绪 / 宏观 / 资金面）
    cold — 小时级按需触发（公告 / 概念 / 季节性），不阻塞 hot

查询接口：
  tier_of(module) / is_hot(module) / hot_modules() / warm_modules() /
  cold_modules() / modules_by_tier() / register_module()
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional

VALID_TIERS = ("hot", "warm", "cold")
DEFAULT_TIER = "cold"

# 模块注册表（模块名 → tier）。模块名统一使用小写短横线/下划线别名。
# 这里覆盖了 analysis_core 中与冷热分离语义相关的核心模块；
# 未注册模块按冷模块处理，避免把未知的新模块误判为 hot 阻塞链路。
MODULE_TIERS: Dict[str, str] = {
    # hot：盘中实时决策链，任何调度都不应被 cold 模块阻塞。
    "fusion": "hot",
    "battle_map": "hot",
    "order_dispatcher": "hot",
    "trade_executor": "hot",
    "execution": "hot",
    "risk_system": "hot",
    # warm：日终/分钟级融合，不参与逐笔执行决策。
    "emotion_system": "warm",
    "emotion_cycle": "warm",
    "macro_system": "warm",
    "macro_veto": "warm",
    "macro_learner": "warm",
    "macro_overseas": "warm",
    "social_sentiment": "warm",
    "retail_sentiment": "warm",
    "fund_forces": "warm",
    "theme_cycle": "warm",
    # cold：按需触发，默认不进入 hot 调度阻塞路径。
    "announcement_arbitrage": "cold",
    "concept_lifecycle": "cold",
    "seasonality": "cold",
    "calendar_effects": "cold",
    "alternative_data": "cold",
}


def _key(module: str) -> str:
    """模块名归一化。"""
    if not isinstance(module, str):
        raise TypeError(f"module 必须为 str，收到 {type(module).__name__}")
    return module.strip().lower().replace("-", "_")


def _validate_tier(tier: str) -> str:
    key = tier.strip().lower()
    if key not in VALID_TIERS:
        raise ValueError(f"非法 tier={tier!r}，可选: {', '.join(VALID_TIERS)}")
    return key


def register_module(module: str, tier: str,
                    registry: Optional[Dict[str, str]] = None) -> str:
    """注册/覆盖一个模块的 tier，返回归一化后的模块名。

    传入 registry 时只更新该副本；不传时更新模块级 MODULE_TIERS。
    """
    key = _key(module)
    tier_key = _validate_tier(tier)
    (registry if registry is not None else MODULE_TIERS)[key] = tier_key
    return key


def tier_of(module: str, registry: Optional[Mapping[str, str]] = None,
            default: str = DEFAULT_TIER) -> str:
    """查询模块 tier；未知模块返回 default（默认 cold）。"""
    key = _key(module)
    source = registry if registry is not None else MODULE_TIERS
    return source.get(key, _validate_tier(default))


def is_hot(module: str, registry: Optional[Mapping[str, str]] = None) -> bool:
    """该模块是否属于 hot 在线链路。"""
    return tier_of(module, registry) == "hot"


def _modules_for_tier(tier: str,
                      registry: Optional[Mapping[str, str]] = None) -> List[str]:
    tier_key = _validate_tier(tier)
    source = registry if registry is not None else MODULE_TIERS
    return sorted(name for name, value in source.items() if value == tier_key)


def hot_modules(registry: Optional[Mapping[str, str]] = None) -> List[str]:
    """hot 模块清单（按名称排序）。"""
    return _modules_for_tier("hot", registry)


def warm_modules(registry: Optional[Mapping[str, str]] = None) -> List[str]:
    """warm 模块清单（按名称排序）。"""
    return _modules_for_tier("warm", registry)


def cold_modules(registry: Optional[Mapping[str, str]] = None) -> List[str]:
    """cold 模块清单（按名称排序），即可按需触发、不阻塞 hot 的集合。"""
    return _modules_for_tier("cold", registry)


def modules_by_tier(registry: Optional[Mapping[str, str]] = None) -> Dict[str, List[str]]:
    """按 tier 分组的模块清单，键固定为 hot/warm/cold。"""
    return {
        "hot": hot_modules(registry),
        "warm": warm_modules(registry),
        "cold": cold_modules(registry),
    }


def cold_schedule(registry: Optional[Mapping[str, str]] = None) -> Dict[str, object]:
    """调度建议：输出 cold 模块清单及建议触发策略。"""
    cold = cold_modules(registry)
    return {
        "cold_modules": cold,
        "count": len(cold),
        "suggestion": "cold 模块可按需/低优先级触发，不阻塞 hot 在线决策链",
        "trigger_policy": "on_demand",
    }


def validate_registry(registry: Mapping[str, str]) -> List[str]:
    """校验注册表，返回非法 tier 条目列表（模块名→值）供排查。"""
    errors: List[str] = []
    for module, tier in registry.items():
        if tier not in VALID_TIERS:
            errors.append(f"{module}={tier}")
    return errors


def report(registry: Optional[Mapping[str, str]] = None) -> Dict[str, object]:
    """生成冷热分离标签总览（纯查询，不调度）。"""
    grouped = modules_by_tier(registry)
    return {
        "valid_tiers": list(VALID_TIERS),
        "default_tier": DEFAULT_TIER,
        "counts": {tier: len(names) for tier, names in grouped.items()},
        "modules_by_tier": grouped,
        "cold_schedule": cold_schedule(registry),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(report(), ensure_ascii=False, indent=2))
