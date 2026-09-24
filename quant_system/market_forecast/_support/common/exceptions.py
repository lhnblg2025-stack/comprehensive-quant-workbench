"""
exceptions.py — QuantV6 统一异常体系
分层异常：数据层 → 预测层 → 策略层 → 执行层 → 风控层，全部继承 QuantV6Error。
"""
from __future__ import annotations


class QuantV6Error(Exception):
    """系统根异常。"""


class ConfigError(QuantV6Error):
    """配置加载/校验错误。"""


class DataFetchError(QuantV6Error):
    """数据获取失败（网络/接口异常）。"""


class DataQualityError(QuantV6Error):
    """数据质量不合格（缺失率过高/停牌/异常值）。"""


class PredictionError(QuantV6Error):
    """预测失败（模型未训练/特征缺失）。"""


class StrategyError(QuantV6Error):
    """策略执行失败（信号非法/仓位溢出）。"""


class ExecutionError(QuantV6Error):
    """交易执行失败（资金不足/T+1违规/手数非法）。"""


class RiskError(QuantV6Error):
    """风控拦截（触发风控阈值）。"""


class WallTimeoutError(QuantV6Error):
    """墙钟超时（网络调用/计算超时保护）。"""


class InsufficientDataError(QuantV6Error):
    """历史数据不足（无法训练/无法计算）。"""
