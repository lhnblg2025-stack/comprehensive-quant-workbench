"""
seasonality — A股季节效应 (V5)

不基于实时数据，基于历史统计规律：
  - month_effect:     各月/各季表现
  - holiday_effect:   节前节后效应
  - earnings_season:  财报季效应
  - policy_window:    政策窗口效应
  - weekday_effect:   周内效应
  - annual_pattern:   年度模式（春季躁动/五穷六绝等）

核心价值：知道什么时候大概率涨/跌，是一种先验概率。
"""

from .month_effect import MonthEffect
from .holiday_effect import HolidayEffect
from .earnings_season import EarningsSeason
from .policy_window import PolicyWindow
from .weekday_effect import WeekdayEffect
from .annual_pattern import AnnualPattern

__all__ = [
    "MonthEffect", "HolidayEffect", "EarningsSeason",
    "PolicyWindow", "WeekdayEffect", "AnnualPattern",
]
