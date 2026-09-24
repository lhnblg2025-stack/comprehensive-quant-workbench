"""
zoo.py — QuantV6 因子注册表
管理所有因子：注册/查询/工厂创建/计算矩阵。
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。 legacy zoo 注册表：因子包装层，实际因子实现引用 price/volume/volatility/liquidity/technical。
"""
from __future__ import annotations


import pandas as pd

from quant_system.market_forecast._support.common.logger import get_logger
from quant_system.ic_factors.base import Factor

log = get_logger("qv6.zoo")

_REGISTRY: dict[str, Factor] = {}


def register_factor(name: str, direction: int = 1, description: str = "",
                    active: bool = True):
    """装饰器：注册因子函数。
    D4收敛登记: 跨模块同名异签名-不强迁保留
    """
    def decorator(func):
        fac = Factor(name, func, direction=direction,
                     description=description, active=active)
        _REGISTRY[name] = fac
        return func
    return decorator


def get_factor(name: str) -> Factor | None:
    """按名称获取因子（不存在返回 None）。
    D4收敛登记: 跨模块同名异签名-不强迁保留
    """
    return _REGISTRY.get(name)


def list_factors(include_inactive: bool = False) -> list[str]:
    """因子名称列表。include_inactive=False 时只返回 active 因子。
    D4收敛登记: 跨模块同名异签名-不强迁保留
    """
    return [n for n, f in _REGISTRY.items() if include_inactive or f.active]


_DEFAULTS_REGISTERED = False


def register_defaults() -> None:
    """注册内置因子（幂等，可重入）。

    2026-08-14 修复: 旧版 `if _REGISTRY: return` 早退——autodiscover 期间
    v7 模块向 zoo._REGISTRY 预注册少数因子后，内置 40 因子被跳过（只注册 4 个），
    方向校准/import_from_zoo 因此丢失大量因子。改用独立标记保证内置因子
    至少注册一次（同名覆盖为相同定义，幂等）。
    """
    global _DEFAULTS_REGISTERED
    if _DEFAULTS_REGISTERED:
        return
    _DEFAULTS_REGISTERED = True
    from quant_system.ic_factors import price, volume, volatility, liquidity, technical
    specs = [
        ("mom20", price.momentum, 1, "20日动量", True),
        ("mom1m", price.momentum_1m, 1, "1月动量", True),
        ("mom3m", price.momentum_3m, 1, "3月动量", True),
        ("mom6m", price.momentum_6m, 1, "6月动量", True),
        ("mom12m", price.momentum_12m, 1, "12月动量", True),
        ("rev5", price.reversal, 1, "5日反转", True),
        ("ma_cross", price.ma_cross, 1, "均线金叉", True),
        ("macd_hist", price.macd_hist_factor, 1, "MACD柱", True),
        ("macd_cross", price.macd_cross, 1, "MACD金叉(归一化)", True),
        ("boll_pos", price.boll_pos_factor, -1, "布林位置(超买反向)", True),
        ("cci_20", price.cci_20, -1, "CCI(超买反向)", True),
        ("rsi14", price.rsi_factor, -1, "RSI(超买反向)", True),
        ("high_low_pos", price.high_low_position, 1, "52周高低位置", True),
        ("bias12", price.bias_factor, -1, "BIAS乖离(反向)", True),
        ("new_high_dist", price.new_high_distance, 1, "距60日新高", True),
        ("new_low_dist", price.new_low_distance, 1, "距60日新低", True),
        ("volume_trend", volume.volume_trend, 1, "量能趋势", True),
        ("volume_surge", volume.volume_surge, 1, "量能放大", True),
        ("vol_20d", volume.vol_20d, 1, "20日均量", True),
        ("volume_ratio", volume.volume_ratio, 1, "量比", True),
        ("obv_slope", volume.obv_slope, 1, "OBV斜率", True),
        ("volume_price_fit", volume.volume_price_fit, 1, "量价配合", True),
        ("realized_vol", volatility.realized_volatility, -1, "已实现波动率", True),
        ("downside_vol", volatility.downside_volatility, -1, "下行波动率", True),
        ("max_dd_12m", volatility.max_dd_12m, 1, "12月最大回撤", True),
        ("beta_60d", volatility.beta_60d, -1, "60日Beta", True),
        ("downside_beta", volatility.downside_beta, -1, "下行Beta", True),
        ("vol_change", volatility.vol_change, 1, "波动收敛", True),
        ("atr_ratio", volatility.atr_ratio, -1, "ATR比率", True),
        ("amount_liquidity", liquidity.amount_liquidity, 1, "成交额流动性", True),
        ("amihud_illiq", liquidity.amihud_illiq, -1, "Amihud非流动性", True),
        ("turnover_stability", liquidity.turnover_stability, 1, "换手稳定", True),
        ("spread_approx", liquidity.spread_approx, -1, "价差近似", True),
        ("macd_div", technical.macd_divergence, 1, "MACD背离", True),
        ("kdj", technical.kdj_factor, -1, "KDJ(J超买反向)", True),
        ("williams", technical.williams_factor, 1, "威廉WR", True),
        ("dmi", technical.dmi_factor, 1, "DMI趋向", True),
        ("donchian", technical.donchian_breakout, 1, "唐奇安突破", True),
        ("boll_width", technical.boll_width, -1, "布林带宽", True),
        ("ma_trend", technical.ma_trend_strength, 1, "均线趋势", True),
    ]
    for name, func, direction, desc, active in specs:
        _REGISTRY[name] = Factor(name, func, direction=direction,
                                 description=desc, active=active)
    # GTJA 因子（2026-08-14: 国泰君安 Alpha 移植, 日线近似, 18个, 方向由IC校准覆盖）
    try:
        from quant_system.ic_factors.gtja import register_into as _gtja_reg
        n_gtja = _gtja_reg(
            _REGISTRY,
            lambda name, func, direction=1, description="":
                _REGISTRY.__setitem__(name, Factor(name, func, direction=direction,
                                                   description=description, active=True)))
        if n_gtja:
            log.info(f"GTJA 因子注册: {n_gtja} 个")
    except Exception as e:  # noqa: BLE001
        log.warning(f"GTJA 因子注册失败: {e}")
    log.info(f"因子注册完成: {len(_REGISTRY)} 个")
    # 应用方向校准覆盖（持久化，重启仍生效）
    try:
        import quant_system.ic_factors.zoo as _zoo_mod
        from quant_system.validate_apply_direction import apply_overrides
        from quant_system.ic_factors import registry as _qv6_registry
        _n = apply_overrides(registry_module=_qv6_registry, zoo_module=_zoo_mod)
        if _n:
            log.info(f"方向校准覆盖已应用: {_n} 个因子")
    except Exception as e:  # noqa: BLE001
        log.warning(f"方向校准覆盖未应用: {e}")


def compute_factor_frame(data: dict, factors: list[str] | None = None,
                         names: list[str] | None = None,
                         apply_direction: bool = True) -> pd.DataFrame:
    """
    计算因子矩阵（截面）：行=股票代码，列=因子名。
    data: {code: DataFrame} K线数据
    factors: 因子名列表（None 用全部 active）
    apply_direction: True（默认）时按 zoo 元数据 direction 校正原始值
      （vals = vals * fac.direction），输出列语义统一为“越大越看多”。
      这是 direction 全链路的单一入口；combination/evaluator 对原始帧
      可选传入 apply_direction 参数保持同口径。
    """
    register_defaults()
    selected = factors or list_factors()
    cols: dict[str, pd.Series] = {}
    for name in selected:
        fac = _REGISTRY.get(name)
        if fac is None or not fac.active:
            continue
        try:
            vals = fac.func(data)
            if isinstance(vals, dict):
                vals = pd.Series(vals, dtype=float)
            vals = vals.astype(float).replace([float("inf"), float("-inf")], float("nan"))
            # P0-2：方向校正——反向因子（direction=-1）参与合成/评价前翻转
            if apply_direction and fac.direction != 1:
                vals = vals * fac.direction
            cols[name] = vals
        except Exception as e:
            log.warning(f"因子 {name} 计算失败: {str(e)[:80]}")
            cols[name] = pd.Series(dtype=float)
    return pd.DataFrame(cols)


def compute_factor_matrix(data: dict, factors: list[str] | None = None) -> pd.DataFrame:
    """别名：计算因子矩阵。"""
    return compute_factor_frame(data, factors)


def factor_meta() -> dict:
    """全部因子元数据。"""
    register_defaults()
    return {n: {"direction": f.direction, "description": f.description,
                "active": f.active} for n, f in _REGISTRY.items()}
