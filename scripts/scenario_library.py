#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""压力测试场景库·扩展（2026-08-23 —— 知识库/skills 驱动的 20+ 极端行情场景）

场景源（skills/知识库，非凭空捏造）:
  - skills/market-liquidity-risk-crisis  《Market Liquidity》(Foucault): 流动性螺旋/挤兑踩踏/价差扩大10-100倍/相关性趋同/A股跌停封死
  - skills/greenspan-crisis-liquidity-transmission: 流动性vs偿付性/传染渠道/保证金追缴/赎回潮/美元融资/央行后盾
  - skills/macro-four-driver-asset-map  (Dalio): 增长/通胀/风险溢价/贴现率 四驱
  - 宏观记忆库: 2013 taper tantrum / 2020 熔断 / 2022 英镑LDI / 2023 SVB / Volcker / 亚洲金融危机 / 卢布制裁 / 负油价等

每个场景: 注入 blocks + 断言(不fail-open/仓位受控/特定风险预案触发)。
用法: 被 extreme_market_drill.py 导入合并进 SCENES。
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # 使 extreme_market_drill 可直接导入

from extreme_market_drill import (_base_blocks, _directions, _factor, _fund,  # noqa: E402
                                  _ladder, _mk, _overseas, _veto, _emo,
                                  _fresh_crash, _MAX_POS_CRASH)

# ── 常用海外极端组合 ──────────────────────────────
_US_CRASH = [{"label": "纳指", "chg_pct": -4.5}, {"label": "道指", "chg_pct": -3.8},
             {"label": "标普500", "chg_pct": -4.0}, {"label": "美10年债殖", "chg_pct": 1.9},
             {"label": "美元指数", "chg_pct": 1.2}]
_US_TAPER = [{"label": "纳指", "chg_pct": -4.0}, {"label": "美10年债殖", "chg_pct": 2.8},
             {"label": "美元指数", "chg_pct": 1.5}, {"label": "伦敦金", "chg_pct": -3.2},
             {"label": "美元兑人民币", "chg_pct": 1.8}]
_FED_HAWK = [{"label": "纳指", "chg_pct": -4.2}, {"label": "美10年债殖", "chg_pct": 3.0},
             {"label": "美元指数", "chg_pct": 1.6}, {"label": "伦敦金", "chg_pct": -2.5},
             {"label": "离岸人民币", "chg_pct": -1.8}]
_FED_DOVE = [{"label": "纳指", "chg_pct": 1.2}, {"label": "美10年债殖", "chg_pct": -1.5},
             {"label": "伦敦金", "chg_pct": 3.5}, {"label": "美元指数", "chg_pct": -1.2}]
_GEOPOLITIC = [{"label": "WTI原油", "chg_pct": 8.5}, {"label": "伦敦金", "chg_pct": 5.2},
               {"label": "纳指", "chg_pct": -3.0}, {"label": "VIX", "chg_pct": 60.0}]
_OIL_SHOCK = [{"label": "WTI原油", "chg_pct": 10.5}, {"label": "伦敦金", "chg_pct": 1.5},
              {"label": "纳指", "chg_pct": -2.2}, {"label": "美元指数", "chg_pct": 0.8}]
_FX_CRASH = [{"label": "离岸人民币", "chg_pct": -2.2}, {"label": "美元指数", "chg_pct": 1.4},
             {"label": "纳指", "chg_pct": -3.5}, {"label": "美10年债殖", "chg_pct": 1.2}]
_LDI_CRISIS = [{"label": "英债30Y", "chg_pct": 30.0}, {"label": "英镑", "chg_pct": -5.0},
               {"label": "美10年债殖", "chg_pct": 2.0}, {"label": "标普500", "chg_pct": -2.5},
               {"label": "道指", "chg_pct": -2.0}]
_DECOUPLE = [{"label": "纳指", "chg_pct": -3.0}, {"label": "伦敦金", "chg_pct": -2.5},
             {"label": "美10年债殖", "chg_pct": -1.8}, {"label": "美元指数", "chg_pct": -1.2}]  # 股债金同跌


def _liquidity_spiral() -> dict:
    """流动性螺旋(Foucault): 价格跌→保证金升→被迫卖出→再跌; 资金枯竭+跌停封死。"""
    return _mk(_base_blocks(),
               market_temperature=_fund(2.0, -2.5e10, -1.2e10),
               leader_sentiment=_ladder(zt=12, mb=2, zb=35, dt=1300, zb_rate=0.85),
               overseas=_overseas(_US_CRASH),
               strong_direction=_directions([]), factor_signal=_factor(0.22),
               macro_veto=_veto("hard", 0.2))


def _margin_call() -> dict:
    """保证金追缴式崩盘(Greenspan): 高杠杆板块清盘, 融资余额骤降, 跌停潮。"""
    return _mk(_base_blocks(),
               market_temperature=_fund(5.0, -3.0e10, -1.0e10),
               leader_sentiment=_ladder(zt=20, mb=2, zb=40, dt=900, zb_rate=0.75),
               overseas=_overseas(_US_CRASH), factor_signal=_factor(0.25),
               strong_direction=_directions([{"name": "融资盘", "level": "单独", "score": 0}]),
               macro_veto=_veto("hard", 0.25))


def _flash_crash() -> dict:
    """闪崩: 指数几分钟内-9%(2010/2020式), 盘中技术分骤降, 情绪从修复直接恐慌。"""
    return _mk(_base_blocks(),
               market_temperature=_emo("恐慌", 25, 2),
               leader_sentiment=_ladder(zt=25, mb=2, zb=45, dt=400, zb_rate=0.7),
               overseas=_overseas([{"label": "纳指", "chg_pct": -2.8}, {"label": "VIX", "chg_pct": 80.0}]),
               factor_signal=_factor(0.2), macro_veto=_veto("hard", 0.15))


def _treasury_tantrum() -> dict:
    """2013 taper tantrum 式: 美债殖利率飙升, 新兴市场资金外逃, 汇率贬值。"""
    return _mk(_base_blocks(),
               market_temperature=_fund(20.0, -8.0e9, -6.0e9),
               leader_sentiment=_ladder(zt=35, mb=3, zb=20, dt=120, zb_rate=0.4),
               overseas=_overseas(_US_TAPER), factor_signal=_factor(0.3),
               macro_veto=_veto("soft", 0.45))


def _hawkish_shock() -> dict:
    """突然极端鹰派(Volcker 式): 加息超预期, 股债双杀, 高位资产重挫。"""
    return _mk(_base_blocks(),
               market_temperature=_fund(15.0, -1.2e10, -9.0e9),
               leader_sentiment=_ladder(zt=28, mb=2, zb=30, dt=240, zb_rate=0.6),
               overseas=_overseas(_FED_HAWK), factor_signal=_factor(0.28),
               macro_veto=_veto("soft", 0.4))


def _dovish_trap() -> dict:
    """衰退恐慌式鸽派: 股市脉冲反弹后转跌(利好出尽), 金大涨, 债殖急跌。"""
    return _mk(_base_blocks(),
               market_temperature=_emo("分歧", 48, 3),
               leader_sentiment=_ladder(zt=48, mb=3, zb=28, dt=150, zb_rate=0.5),
               overseas=_overseas(_FED_DOVE), factor_signal=_factor(0.35),
               macro_veto=_veto("soft", 0.5))


def _fx_plunge() -> dict:
    """汇率急贬(亚洲金融危机式): 人民币快速贬值+北向大撤+外资重仓补跌。"""
    return _mk(_base_blocks(),
               market_temperature=_fund(18.0, -1.8e10, -1.5e10),
               leader_sentiment=_ladder(zt=30, mb=2, zb=25, dt=300, zb_rate=0.55),
               overseas=_overseas(_FX_CRASH), factor_signal=_factor(0.26),
               stock_picks={"date": "2026-08-21", "pools": {
                   "mid_long": [{"name": "外资重仓", "code": "600519", "dist52w": -5}]}},
               macro_veto=_veto("soft", 0.35))


def _volcker_shock() -> dict:
    """央行意外大幅加息(Volcker): 全球risk-off, 债券抛售, 利率体系重定价。"""
    return _mk(_base_blocks(),
               market_temperature=_fund(8.0, -2.0e10, -1.3e10),
               leader_sentiment=_ladder(zt=18, mb=2, zb=35, dt=500, zb_rate=0.72),
               overseas=_overseas(_FED_HAWK + [{"label": "美2年债殖", "chg_pct": 4.0}]),
               factor_signal=_factor(0.24), macro_veto=_veto("hard", 0.2))


def _geopolitical_shock() -> dict:
    """地缘冲击(俄乌式): 油价暴涨+避险+股市大跌+VIX 60+。"""
    return _mk(_base_blocks(),
               market_temperature=_emo("恐慌", 22, 2),
               leader_sentiment=_ladder(zt=22, mb=2, zb=38, dt=350, zb_rate=0.68),
               overseas=_overseas(_GEOPOLITIC), factor_signal=_factor(0.27),
               macro_veto=_veto("soft", 0.3))


def _oil_price_shock() -> dict:
    """大宗商品暴涨(1973石油式): 滞胀恐慌, 成本推动通胀。"""
    return _mk(_base_blocks(),
               market_temperature=_fund(22.0, -6.0e9, -5.0e9),
               leader_sentiment=_ladder(zt=40, mb=3, zb=18, dt=80, zb_rate=0.35),
               overseas=_overseas(_OIL_SHOCK), factor_signal=_factor(0.33),
               macro_veto=_veto("soft", 0.5))


def _double_circuit_breaker() -> dict:
    """2020 式连环熔断: 两周两度熔断+流动性枯竭+回购市场紧张。"""
    return _mk(_base_blocks(),
               market_temperature=_emo("恐慌", 10, 1),
               leader_sentiment=_ladder(zt=10, mb=1, zb=50, dt=1600, zb_rate=0.95),
               overseas=_overseas(_US_CRASH + [{"label": "美10年债殖", "chg_pct": -2.0}]),
               factor_signal=_factor(0.18), macro_veto=_veto("hard", 0.1),
               freshness=_fresh_crash())


def _sanctions_like() -> dict:
    """制裁式卢布危机: 汇率暴跌+外资冻结传闻+银行挤兑+商品出口波动。"""
    return _mk(_base_blocks(),
               market_temperature=_emo("恐慌", 15, 1),
               leader_sentiment=_ladder(zt=15, mb=1, zb=42, dt=700, zb_rate=0.8),
               overseas=_overseas([{"label": "美元兑卢布", "chg_pct": 30.0},
                                   {"label": "WTI原油", "chg_pct": 7.0},
                                   {"label": "伦敦金", "chg_pct": 4.0},
                                   {"label": "纳指", "chg_pct": -2.5}]),
               factor_signal=_factor(0.22), macro_veto=_veto("hard", 0.15))


def _pension_ldi() -> dict:
    """2022 英镑养老金 LDI 危机: 英债殖急升30bp级+英镑急贬+全球债市波动。"""
    return _mk(_base_blocks(),
               market_temperature=_fund(25.0, -9.0e9, -8.0e9),
               leader_sentiment=_ladder(zt=32, mb=3, zb=22, dt=160, zb_rate=0.45),
               overseas=_overseas(_LDI_CRISIS), factor_signal=_factor(0.31),
               macro_veto=_veto("soft", 0.4))


def _short_squeeze() -> dict:
    """轧空(2021 GameStop 式): 情绪极端狂热+高波动+空头挤压, 泡沫+回撤双重风险。"""
    return _mk(_base_blocks(),
               market_temperature=_emo("高潮", 138, 8),
               leader_sentiment=_ladder(zt=138, mb=8, zb=50, dt=8, zb_rate=0.42),
               strong_direction=_directions([{"name": "轧空标的", "level": "主线", "score": 3}]),
               overseas=_overseas([{"label": "纳指", "chg_pct": 0.6}, {"label": "VIX", "chg_pct": 35.0}]),
               factor_signal=_factor(0.4),
               macro_veto=_veto("soft", 0.6))


def _crypto_depeg() -> dict:
    """稳定币脱锚(UST 2022 式): 加密崩+风险偏好骤降+流动性抽离跨市场传染。"""
    return _mk(_base_blocks(),
               market_temperature=_fund(12.0, -1.5e10, -1.0e10),
               leader_sentiment=_ladder(zt=20, mb=2, zb=34, dt=420, zb_rate=0.66),
               overseas=_overseas([{"label": "BTC", "chg_pct": -25.0},
                                   {"label": "纳指", "chg_pct": -3.2},
                                   {"label": "美10年债殖", "chg_pct": -1.5}]),
               factor_signal=_factor(0.26), macro_veto=_veto("soft", 0.3))


def _negative_oil() -> dict:
    """负油价(2020 CL 挤仓式): 商品价格发现失效, 多头爆仓, 跨市场恐慌。"""
    return _mk(_base_blocks(),
               market_temperature=_emo("恐慌", 18, 1),
               leader_sentiment=_ladder(zt=18, mb=1, zb=44, dt=600, zb_rate=0.85),
               overseas=_overseas([{"label": "WTI原油", "chg_pct": -40.0},
                                   {"label": "纳指", "chg_pct": -2.0},
                                   {"label": "伦敦金", "chg_pct": 1.8}]),
               factor_signal=_factor(0.23), macro_veto=_veto("soft", 0.25))


def _correlation_converge() -> dict:
    """相关性趋同危机(2020.3/2022): 股债金同跌, 资产配置失效, 现金为王。"""
    return _mk(_base_blocks(),
               market_temperature=_emo("恐慌", 20, 2),
               leader_sentiment=_ladder(zt=20, mb=2, zb=40, dt=520, zb_rate=0.7),
               overseas=_overseas(_DECOUPLE), factor_signal=_factor(0.21),
               macro_veto=_veto("soft", 0.2))


def _svb_banking() -> dict:
    """银行挤兑(2023 SVB 式): 金融股崩+机构赎回潮+隔夜拆借紧张+信用利差走阔。"""
    return _mk(_base_blocks(),
               market_temperature=_fund(10.0, -2.2e10, -1.4e10),
               leader_sentiment=_ladder(zt=16, mb=2, zb=36, dt=480, zb_rate=0.73),
               overseas=_overseas([{"label": "KBW银行", "chg_pct": -12.0},
                                   {"label": "纳指", "chg_pct": -2.8},
                                   {"label": "美10年债殖", "chg_pct": -2.2}]),
               factor_signal=_factor(0.25), macro_veto=_veto("soft", 0.3))


def _redemption_loop() -> dict:
    """赎回潮螺旋(Greenspan 传染渠道): 基金赎回→被迫卖→净值跌→再赎回。"""
    return _mk(_base_blocks(),
               market_temperature=_fund(4.0, -2.8e10, -1.1e10),
               leader_sentiment=_ladder(zt=10, mb=1, zb=48, dt=1500, zb_rate=0.92),
               overseas=_overseas(_US_CRASH), factor_signal=_factor(0.2),
               macro_veto=_veto("hard", 0.15))


def _china_trust_contagion() -> dict:
    """影子/信托兑付危机(A股传导): 信用事件→流动性抽离→小票踩踏。"""
    return _mk(_base_blocks(),
               market_temperature=_fund(14.0, -1.9e10, -1.3e10),
               leader_sentiment=_ladder(zt=22, mb=2, zb=40, dt=1300, zb_rate=0.78),
               overseas=_overseas([{"label": "纳指", "chg_pct": -1.5},
                                   {"label": "离岸人民币", "chg_pct": -1.2},
                                   {"label": "美10年债殖", "chg_pct": 0.8}]),
               factor_signal=_factor(0.24), macro_veto=_veto("soft", 0.3))


def _sector_rotation_flash() -> dict:
    """风格急转闪崩: 主题抱团瓦解(白酒/新能源2021-22式), 高估值踩踏+低估值防御。"""
    return _mk(_base_blocks(),
               market_temperature=_emo("退潮", 30, 2),
               leader_sentiment=_ladder(zt=30, mb=2, zb=32, dt=260, zb_rate=0.6),
               strong_direction=_directions([{"name": "防御红利", "level": "次主线", "score": 1}]),
               overseas=_overseas([{"label": "纳指", "chg_pct": 0.2}]),
               factor_signal=_factor(0.36), macro_veto=_veto("soft", 0.5))


# 知识库场景注册表: 名称 → (构造函数, 断言)
EXTRA_SCENES: list[dict] = [
    {"name": "流动性螺旋(Foucault)", "blocks": _liquidity_spiral(),
     "exp_posture": "防守观望", "exp_pos_max": _MAX_POS_CRASH,
     "source": "skills/market-liquidity-risk-crisis·《Market Liquidity》"},
    {"name": "保证金追缴崩盘", "blocks": _margin_call(),
     "exp_posture": "防守观望", "exp_pos_max": _MAX_POS_CRASH,
     "source": "skills/greenspan-crisis-liquidity-transmission"},
    {"name": "闪崩(2010/2020式)", "blocks": _flash_crash(),
     "exp_posture": "防守观望", "exp_pos_max": _MAX_POS_CRASH,
     "source": "Flash Crash 2010/COVID 2020"},
    {"name": "美债taper tantrum", "blocks": _treasury_tantrum(),
     "exp_posture_banned": ("积极进攻",), "exp_pos_max": 0.35,
     "source": "2013 缩减恐慌·skills/macro-four-driver-asset-map"},
    {"name": "突然极端鹰派(Volcker)", "blocks": _hawkish_shock(),
     "exp_posture_banned": ("积极进攻",), "exp_pos_max": _MAX_POS_CRASH + 0.10,
     "source": "Volcker 式加息"},
    {"name": "衰退恐慌式鸽派", "blocks": _dovish_trap(),
     "exp_posture_banned": ("积极进攻",), "exp_pos_max": 0.45,
     "source": "鸽派但衰退恐慌·宏观四驱"},
    {"name": "汇率急贬(亚洲危机式)", "blocks": _fx_plunge(),
     "exp_posture_banned": ("积极进攻",), "exp_pos_max": 0.35,
     "source": "亚洲金融危机 1997"},
    {"name": "央行意外加息", "blocks": _volcker_shock(),
     "exp_posture": "防守观望", "exp_pos_max": _MAX_POS_CRASH,
     "source": "Volcker 1980/风险溢价冲击"},
    {"name": "地缘冲击(俄乌式)", "blocks": _geopolitical_shock(),
     "exp_posture_banned": ("积极进攻",), "exp_pos_max": _MAX_POS_CRASH + 0.10,
     "source": "2022 俄乌·greenspan 传染渠道"},
    {"name": "石油危机滞胀", "blocks": _oil_price_shock(),
     "exp_posture_banned": ("积极进攻",), "exp_pos_max": 0.45,
     "source": "1973/滞胀·Dalio 四驱"},
    {"name": "连环熔断(2020式)", "blocks": _double_circuit_breaker(),
     "exp_posture": "防守观望", "exp_pos_max": _MAX_POS_CRASH,
     "source": "2020.3 两度熔断"},
    {"name": "制裁式卢布危机", "blocks": _sanctions_like(),
     "exp_posture": "防守观望", "exp_pos_max": _MAX_POS_CRASH,
     "source": "2022 卢布·制裁冲击"},
    {"name": "英镑LDI养老金危机", "blocks": _pension_ldi(),
     "exp_posture_banned": ("积极进攻",), "exp_pos_max": 0.35,
     "source": "2022 英国 LDI"},
    {"name": "轧空狂热(GameStop)", "blocks": _short_squeeze(),
     "exp_posture_banned": ("积极进攻",), "expect_risk": True,
     "source": "2021 GameStop·动物精神"},
    {"name": "稳定币脱锚传染", "blocks": _crypto_depeg(),
     "exp_posture_banned": ("积极进攻",), "exp_pos_max": 0.35,
     "source": "2022 UST depeg"},
    {"name": "负油价挤仓", "blocks": _negative_oil(),
     "exp_posture_banned": ("积极进攻",), "exp_pos_max": 0.40,
     "source": "2020 CL 负油价"},
    {"name": "相关性趋同危机", "blocks": _correlation_converge(),
     "exp_posture_banned": ("积极进攻",), "exp_pos_max": 0.45,
     "source": "2020.3/2022 股债金同跌·Foucault"},
    {"name": "银行挤兑(2023 SVB)", "blocks": _svb_banking(),
     "exp_posture_banned": ("积极进攻",), "exp_pos_max": 0.40,
     "source": "2023 SVB·融资流动性"},
    {"name": "赎回潮螺旋", "blocks": _redemption_loop(),
     "exp_posture": "防守观望", "exp_pos_max": _MAX_POS_CRASH,
     "source": "greenspan 传染·赎回压力"},
    {"name": "信用事件踩踏(A股)", "blocks": _china_trust_contagion(),
     "exp_posture_banned": ("积极进攻",), "exp_pos_max": 0.35,
     "source": "A股信用传导·Foucault 危机特征"},
    {"name": "风格急转闪崩", "blocks": _sector_rotation_flash(),
     "exp_posture_banned": ("积极进攻",), "exp_pos_max": 0.45,
     "source": "2021-22 抱团瓦解"},
]