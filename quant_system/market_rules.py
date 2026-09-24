"""
A股市场交易规则硬编码库 — 上交所/深交所/北交所 + 证监会规则

规则来源（2024-2025 现行有效版本，已联网核验）：
  - 上海证券交易所《交易规则》《科创板股票交易特别规定》
  - 深圳证券交易所《交易规则》《创业板股票交易特别规定》
  - 北京证券交易所《交易规则（试行）》
  - 证监会《上市公司重大资产重组管理办法》《退市新规》(2024-04)、《减持新规》等
  - 财政部/税务总局：印花税 2023-08-28 起减半至 0.05%（卖出单边）
  - 中国结算：过户费 2022-04-29 起 0.001%（双向）

设计目标：所有与市场制度相关的硬编码（涨跌幅/交易时间/费用/数量单位/停牌/退市/
两融/适当性）集中于此，供回测引擎、风控、下单模块统一引用，禁止散落硬编码。

用法:
  from market_rules import get_price_limit_pct, is_in_call_auction, calc_trade_fee, ...
  python3 -m quant_system.market_rules --self-test   # 自检
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Optional

CST = timezone(timedelta(hours=8))

# ══════════════════════════════════════════════════════════════════
# 一、板块识别
# ══════════════════════════════════════════════════════════════════

# P2-Q4-fix(L426): 删除未使用的 BOARD_PREFIX 死字典（Q4#33）——
# 板块识别统一走 detect_board() 内的显式分支，避免两份规则漂移。


def detect_board(code: str) -> str:
    """根据股票代码识别所属板块。

    Args:
        code: 6位数字代码（可带 .SH/.SZ/.BJ 后缀，可带交易所前缀如 sh600000）

    Returns:
        板块名: 沪主板/科创板/深主板/创业板/北交所/深B股/沪B股/未知
    """
    if not code:
        return "未知"
    c = re.sub(r"[^0-9]", "", code)  # 去除非数字字符
    if len(c) < 6:
        return "未知"
    c = c[:6]
    if c.startswith("688") or c.startswith("689"):
        return "科创板"
    if c.startswith("60"):
        return "沪主板"
    if c.startswith("30"):
        return "创业板"
    if c.startswith("00"):
        return "深主板"
    if c.startswith("20"):
        return "深B股"
    if c.startswith("90"):
        return "沪B股"
    if c.startswith(("8", "4", "92")):
        return "北交所"
    return "未知"


def is_st_name(name: str) -> bool:
    """判断证券简称是否带 ST/*ST 风险警示。

    Args:
        name: 证券简称，如 'ST康美'、'*ST中安'
    """
    if not name:
        return False
    n = name.upper().strip()
    return n.startswith("ST") or n.startswith("*ST")


def is_risk_warning_star(name: str) -> bool:
    """是否 *ST（退市风险警示，区别于普通 ST）。"""
    if not name:
        return False
    return name.upper().strip().startswith("*ST")


# ══════════════════════════════════════════════════════════════════
# 二、涨跌幅限制
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PriceLimitRule:
    board: str
    normal_pct: float          # 正常状态日涨跌幅（%）
    st_pct: float              # ST/*ST 状态日涨跌幅（%）
    new_stock_no_limit_days: int  # 上市前 N 个交易日不设涨跌幅
    note: str = ""


PRICE_LIMIT_RULES = {
    "沪主板": PriceLimitRule("沪主板", 10.0, 5.0, 5,
                             "2023-02-17 全面注册制：前5日无涨跌幅，第6日起±10%"),
    "深主板": PriceLimitRule("深主板", 10.0, 5.0, 5,
                             "2023-02-17 全面注册制：前5日无涨跌幅，第6日起±10%"),
    "科创板": PriceLimitRule("科创板", 20.0, 20.0, 5,
                             "前5日无涨跌幅，第6日起±20%；科创板无ST制度(但有*ST,同20%)"),
    "创业板": PriceLimitRule("创业板", 20.0, 20.0, 5,
                             "2020-08-24 注册制：前5日无涨跌幅，第6日起±20%；ST股仍±20%"),
    "北交所": PriceLimitRule("北交所", 30.0, 30.0, 1,
                             "新股首日不设涨跌幅，次日起±30%；ST股同为30%"),
    "深B股": PriceLimitRule("深B股", 10.0, 5.0, 0, "B股维持±10%"),
    "沪B股": PriceLimitRule("沪B股", 10.0, 5.0, 0, "B股维持±10%"),
    "未知": PriceLimitRule("未知", 10.0, 5.0, 0, "默认按主板处理"),
}


def get_price_limit_pct(code: str, name: str = "", is_new_stock: bool = False,
                        trade_day: Optional[int] = None,
                        delisting_period: bool = False) -> float:
    """获取某只股票当日涨跌幅限制（百分比，如 10 表示 ±10%）。

    Args:
        code: 股票代码
        name: 证券简称（用于 ST 判断）
        is_new_stock: 是否新股（上市前N个交易日）
        trade_day: 上市第几个交易日（1=上市首日）；与 is_new_stock 二选一
        delisting_period: 是否处于退市整理期（首日不设涨跌幅）

    Returns:
        涨跌幅百分比，0 表示不设涨跌幅限制

    规则要点:
      - 主板 ±10%；ST/*ST 主板 ±5%
      - 创业板/科创板 ±20%；ST 仍 ±20%
      - 北交所 ±30%
      - 新股上市前5日（主板/双创）/首日（北交所）不设涨跌幅
      - 退市整理期首日不设涨跌幅，其后 10%（主板）/20%（双创）/30%（北交所）
    """
    board = detect_board(code)
    rule = PRICE_LIMIT_RULES[board]

    # 退市整理期首日不设涨跌幅
    if delisting_period and _is_first_delisting_day(trade_day):
        return 0.0

    # 新股无涨跌幅窗口
    if trade_day is not None:
        if 1 <= trade_day <= rule.new_stock_no_limit_days:
            return 0.0
    elif is_new_stock:
        return 0.0

    # ST 折扣（仅主板/B股适用；双创与北交所 ST 不额外收窄）
    if is_st_name(name) and rule.st_pct < rule.normal_pct:
        return rule.st_pct
    return rule.normal_pct


def _is_first_delisting_day(trade_day: Optional[int]) -> bool:
    """退市整理期首日判断（trade_day=1 视为首日）。"""
    return trade_day == 1


def price_limit_px(code: str, prev_close: float, name: str = "",
                   is_new_stock: bool = False, trade_day: Optional[int] = None,
                   delisting_period: bool = False) -> tuple[float, float]:
    """计算当日涨停价/跌停价（四舍五入到 0.01）。

    Returns:
        (跌停价, 涨停价)；无涨跌幅限制时返回 (None, None)
    """
    pct = get_price_limit_pct(code, name, is_new_stock, trade_day, delisting_period)
    if pct <= 0:
        return None, None
    lo = round(prev_close * (1 - pct / 100.0) + 1e-9, 2)
    hi = round(prev_close * (1 + pct / 100.0) - 1e-9, 2)
    return lo, hi


# ══════════════════════════════════════════════════════════════════
# 三、交易时间 / 竞价阶段
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SessionPhase:
    """交易阶段定义（沪/深/北交所 A 股通用主体结构）。"""
    name: str
    start: time
    end: time
    can_submit: bool = True   # 可否申报
    can_cancel: bool = True   # 可否撤单


# 沪市 A 股（主板/双创）交易日结构
PHASES_SH = [
    SessionPhase("开盘集合竞价(可撤)", time(9, 15), time(9, 20), True, True),
    SessionPhase("开盘集合竞价(不可撤)", time(9, 20), time(9, 25), True, False),
    SessionPhase("开盘撮合/静默", time(9, 25), time(9, 30), False, False),
    SessionPhase("连续竞价-上午", time(9, 30), time(11, 30), True, True),
    # P2-Q4-fix(M414): 沪市午间 11:30-13:00 不接受申报与撤单（can_submit/can_cancel=False）
    SessionPhase("午间休市(沪:不可申报)", time(11, 30), time(13, 0), False, False),
    SessionPhase("连续竞价-下午", time(13, 0), time(14, 57), True, True),
    SessionPhase("收盘集合竞价(不可撤)", time(14, 57), time(15, 0), True, False),
    SessionPhase("盘后(大宗/盘后定价)", time(15, 0), time(15, 30), True, True),
]
# P2-Q4-fix(M414): 深市午间 11:30-13:00 可申报可撤单（深交所特有），与沪市拆分为两套阶段表
PHASES_SZ = [
    SessionPhase("开盘集合竞价(可撤)", time(9, 15), time(9, 20), True, True),
    SessionPhase("开盘集合竞价(不可撤)", time(9, 20), time(9, 25), True, False),
    SessionPhase("开盘撮合/静默", time(9, 25), time(9, 30), False, False),
    SessionPhase("连续竞价-上午", time(9, 30), time(11, 30), True, True),
    SessionPhase("午间休市(深:可申报可撤)", time(11, 30), time(13, 0), True, True),
    SessionPhase("连续竞价-下午", time(13, 0), time(14, 57), True, True),
    SessionPhase("收盘集合竞价(不可撤)", time(14, 57), time(15, 0), True, False),
    SessionPhase("盘后(大宗/盘后定价)", time(15, 0), time(15, 30), True, True),
]
# P2-Q4-fix(M414): 北交所午间同样不接受申报与撤单，阶段结构与沪市一致
# V11 审计修复（Medium）: 原 `PHASES_BJ = PHASES_SH` 是别名共享——注释声称"不再借用
# 沪表别名"但实际共享同一列表对象，将来调整沪市阶段会连带北交所。
# 修正: 独立副本（内容相同，对象独立）。
PHASES_BJ = [
    SessionPhase("开盘集合竞价(可撤)", time(9, 15), time(9, 20), True, True),
    SessionPhase("开盘集合竞价(不可撤)", time(9, 20), time(9, 25), True, False),
    SessionPhase("开盘撮合/静默", time(9, 25), time(9, 30), False, False),
    SessionPhase("连续竞价-上午", time(9, 30), time(11, 30), True, True),
    SessionPhase("午间休市(北交:不可申报)", time(11, 30), time(13, 0), False, False),
    SessionPhase("连续竞价-下午", time(13, 0), time(14, 57), True, True),
    SessionPhase("收盘集合竞价(不可撤)", time(14, 57), time(15, 0), True, False),
    SessionPhase("盘后(大宗/盘后定价)", time(15, 0), time(15, 30), True, True),
]

PHASE_NOTES = {
    "午间-沪": "沪市 11:30-13:00 不接受申报与撤单",
    "午间-深": "深市 11:30-13:00 可申报、可撤单（深交所特有）",
    "北交所午间": "北交所 11:30-13:00 不接受申报与撤单",
}


def _phases_for_market(market: str = "沪深") -> list:
    """按市场选择阶段表（P2-Q4-fix M414）。

    深市午间可申报可撤；沪市/北交所午间不可。默认"沪深"按沪市规则保守取值。
    """
    if market == "北交所":
        return PHASES_BJ
    if market == "深市":
        return PHASES_SZ
    return PHASES_SH

AUC_OPEN_START = time(9, 15)
AUC_OPEN_END = time(9, 25)
AUC_CLOSE_START = time(14, 57)
AUC_CLOSE_END = time(15, 0)
CONTINUOUS_START = time(9, 30)
CONTINUOUS_END = time(14, 57)
LUNCH_START = time(11, 30)
LUNCH_END = time(13, 0)


def is_trading_day_now(dt: Optional[datetime] = None) -> bool:
    """判断当前是否为交易日（仅周末粗筛；精确日历见 market_clock.get_trade_calendar）。"""
    dt = dt or datetime.now(CST)
    return dt.weekday() < 5


def current_phase(dt: Optional[datetime] = None,
                  market: str = "沪深") -> Optional[str]:
    """返回当前所处交易阶段名；非交易时段返回 None。

    market: '沪深'/'沪市'/'深市'/'北交所'（午间申报规则不同；
    深市午间可申报可撤，沪市/北交所午间不可，P2-Q4-fix M414）
    """
    dt = dt or datetime.now(CST)
    if not is_trading_day_now(dt):
        return None
    t = dt.time()
    for ph in _phases_for_market(market):
        if ph.start <= t < ph.end:
            return ph.name
    return None


def is_in_call_auction(dt: Optional[datetime] = None) -> bool:
    """是否处于集合竞价时段（开盘 9:15-9:25 或收盘 14:57-15:00）。"""
    dt = dt or datetime.now(CST)
    t = dt.time()
    return (AUC_OPEN_START <= t < AUC_OPEN_END) or (AUC_CLOSE_START <= t < AUC_CLOSE_END)


def is_in_continuous_auction(dt: Optional[datetime] = None) -> bool:
    """是否处于连续竞价时段（9:30-11:30 / 13:00-14:57）。"""
    dt = dt or datetime.now(CST)
    t = dt.time()
    return (CONTINUOUS_START <= t < LUNCH_START) or (LUNCH_END <= t < CONTINUOUS_END)


def is_trading_session(dt: Optional[datetime] = None, market: str = "沪深") -> bool:
    """是否处于可交易时段（集合竞价+连续竞价+收盘竞价）。"""
    return current_phase(dt, market) is not None


def can_cancel_now(dt: Optional[datetime] = None, market: str = "沪深") -> bool:
    """当前时段是否允许撤单（用于模拟撤单行为）。

    P2-Q4-fix(M414): 按市场选择阶段表——沪市/北交所午间不可撤单，
    深市午间可撤单。默认"沪深"按沪市规则保守返回。
    """
    dt = dt or datetime.now(CST)
    ph = current_phase(dt, market)
    if ph is None:
        return False
    for p in _phases_for_market(market):
        if p.name == ph:
            return p.can_cancel
    return True


# ══════════════════════════════════════════════════════════════════
# 四、申报数量单位（手数/股数）
# ══════════════════════════════════════════════════════════════════

def min_buy_share(code: str) -> int:
    """最小买入单位（股）。科创板 200 股起，其余 100 股起。"""
    return 200 if detect_board(code) == "科创板" else 100


def share_increment(code: str) -> int:
    """买入申报递增单位（股）。科创板 1 股递增，其余 100 股整数倍。"""
    return 1 if detect_board(code) == "科创板" else 100


def validate_order_qty(code: str, qty: int, is_buy: bool = True) -> bool:
    """校验委托数量是否合规（数量规则，不含价格笼子）。

    规则:
      - 买入: 科创板 ≥200股且1股递增；其余 ≥100股且100股整数倍
      - 卖出: 余额不足100股（科创200股）部分须一次性申报卖出
      - 北交所: 100股起，1股递增（买入卖出均可零股递增）
    """
    board = detect_board(code)
    if qty <= 0:
        return False
    if board == "北交所":
        return qty >= 100  # 100股起，1股递增
    if board == "科创板":
        return qty >= 200
    # 沪深主板/创业板
    if is_buy:
        return qty >= 100 and qty % 100 == 0
    # V11 审计修复（Medium）: 原 `qty >= 100` 拒绝 <100 的零股卖出，
    # 但 A 股允许零股（持仓不足 100 股）一次性申报卖出。修正: 卖出 ≥1 即合法
    #（零股清仓；100 整数倍部分正常卖出；混合时整手+零股需拆单由上层处理）。
    return qty >= 1


# ══════════════════════════════════════════════════════════════════
# 五、交易费用（写死现行标准，2024-2025）
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FeeSpec:
    commission_rate: float   # 券商佣金（双边）
    commission_min: float    # 单笔最低佣金（元）
    stamp_tax_rate: float    # 印花税（仅卖出，单边）
    transfer_fee_rate: float # 过户费（双边）
    note: str = ""


# 2023-08-28 起印花税减半 0.05%；过户费 2022-04-29 起 0.001%（双边）
# ⚠️ 佣金口径：万0.85 为用户实际账户费率（系统回测默认值），
#    万2.5 为市场普遍标准（仅参考，勿作系统默认）。
FEE_DEFAULT = FeeSpec(
    commission_rate=0.000085,  # 万0.85（用户实际账户费率 = 系统回测默认）
    commission_min=5.0,        # 单笔最低 5 元
    stamp_tax_rate=0.0005,     # 卖出 0.05%
    transfer_fee_rate=0.00001, # 0.001% 双边
    note="佣金万0.85(用户账户实际费率,最低5元)/印花税卖出0.05%/过户费双边0.001%",
)

# 市场普遍标准佣金（参考，非系统默认）
FEE_MARKET_STANDARD = FeeSpec(
    commission_rate=0.00025,   # 万2.5
    commission_min=5.0,
    stamp_tax_rate=0.0005,
    transfer_fee_rate=0.00001,
    note="市场普遍标准：佣金万2.5(最低5元)，仅供参考",
)


def calc_trade_fee(amount: float, side: str = "buy",
                   spec: FeeSpec = FEE_DEFAULT) -> dict:
    """计算一笔成交的费用明细。

    Args:
        amount: 成交金额（元）
        side: 'buy' 或 'sell'
        spec: 费率规格

    Returns:
        {"commission": ..., "stamp_tax": ..., "transfer_fee": ...,
         "total": ..., "note": ...}

    注: 回测引擎 backtest_engine 默认佣金万0.85 即用户实际账户费率，
    与 FEE_DEFAULT 一致；如需模拟市场普遍标准可传入 FEE_MARKET_STANDARD。
    """
    commission = max(amount * spec.commission_rate, spec.commission_min)
    stamp_tax = amount * spec.stamp_tax_rate if side == "sell" else 0.0
    transfer_fee = amount * spec.transfer_fee_rate
    return {
        "commission": round(commission, 4),
        "stamp_tax": round(stamp_tax, 4),
        "transfer_fee": round(transfer_fee, 4),
        "total": round(commission + stamp_tax + transfer_fee, 4),
        "note": spec.note,
    }


# ══════════════════════════════════════════════════════════════════
# 六、价格笼子（连续竞价有效申报价格范围）
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PriceCageRule:
    pct: float            # 申报价相对基准价的最大偏离（%）
    ticks: int            # 或相对基准价 ±N 个最小变动单位（取孰高/孰低）
    tick_size: float = 0.01
    note: str = ""


# 全面注册制（2023-02-17）起，主板/创业板/科创板统一 2% 价格笼子
PRICE_CAGE = PriceCageRule(
    pct=2.0,
    ticks=10,
    tick_size=0.01,
    note="买入申报≤基准价×102%与基准价+10个最小变动单位孰高；卖出申报≥基准价×98%与基准价-10个最小变动单位孰低",
)


def effective_price_range(code: str, ref_price: float,
                          side: str = "buy") -> tuple[float, float]:
    """连续竞价阶段有效申报价格范围（价格笼子）。

    Args:
        code: 股票代码（北交所无价格笼子，返回 (0, inf)）
        ref_price: 基准价（买入=卖一价，卖出=买一价；简化可用最新价）
        side: 'buy'/'sell'

    Returns:
        (下限, 上限)
    """
    if detect_board(code) == "北交所":
        return 0.0, float("inf")  # 北交所无价格笼子
    cage = PRICE_CAGE
    tick_abs = cage.ticks * cage.tick_size
    # V11 审计修复（High）: 原实现买卖边界语义反了——买入返回(下限,inf)、卖出返回(0,上限)，
    # 约束的是不受监管的那一侧（会放行"买得过高/卖得过低"的违规申报）。
    # 监管规则: 买入受上限约束（≤基准×102% 与 +10 ticks 孰高），卖出受下限约束（≥98% 与 -10 ticks 孰低）。
    if side == "buy":
        hi = max(ref_price * (1 + cage.pct / 100), ref_price + tick_abs)
        return 0.0, round(hi, 2)
    lo = min(ref_price * (1 - cage.pct / 100), ref_price - tick_abs)
    return round(lo, 2), float("inf")


# ══════════════════════════════════════════════════════════════════
# 七、临时停牌（无涨跌幅限制股票盘中临停）
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CircuitBreakerRule:
    thresholds: tuple = (30.0, 60.0)   # 较开盘价 ±30%/±60%
    duration_minutes: int = 10
    note: str = "无涨跌幅限制股票(新股前5日/退市整理首日/重新上市首日)较开盘价首次±30%、±60%触发，停牌10分钟"


def circuit_breaker_trigger(open_px: float, cur_px: float) -> Optional[float]:
    """判断是否触发盘中临时停牌。

    Returns:
        已触及的最高阈值（30或60），未触发返回 None。
        语义：较开盘价首次±30%停牌10分钟，复牌后首次±60%再停10分钟；
        当前涨跌幅若已超60%，则30%与60%两级均已触发过。
    """
    if not open_px:
        return None
    chg = abs(cur_px / open_px - 1) * 100
    hit = None
    for th in CircuitBreakerRule.thresholds:
        if chg >= th:
            hit = th
    return hit


# ══════════════════════════════════════════════════════════════════
# 八、大宗交易门槛
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BlockTradeRule:
    min_shares: int     # 单笔最低股数
    min_amount: float   # 或最低金额（元）
    note: str = ""


BLOCK_TRADE_RULES = {
    "沪主板": BlockTradeRule(300000, 2_000_000, "30万股或200万元"),
    "深主板": BlockTradeRule(300000, 2_000_000, "30万股或200万元"),
    "科创板": BlockTradeRule(300000, 2_000_000, "30万股或200万元"),
    "创业板": BlockTradeRule(300000, 2_000_000, "30万股或200万元"),
    "北交所": BlockTradeRule(100000, 1_000_000, "10万股或100万元（北交所）"),
    "未知": BlockTradeRule(300000, 2_000_000, "默认按沪深标准"),
}


def block_trade_min(code: str) -> BlockTradeRule:
    return BLOCK_TRADE_RULES.get(detect_board(code), BLOCK_TRADE_RULES["未知"])


# ══════════════════════════════════════════════════════════════════
# 九、T+1 交易制度
# ══════════════════════════════════════════════════════════════════

T_PLUS_1 = True  # A股实行 T+1：当日买入的股票次一交易日方可卖出（北交所同为T+1）


def can_sell_shares(buy_date: datetime, sell_date: datetime) -> bool:
    """判断 buy_date 买入的股票是否可在 sell_date 卖出（T+1，基于交易日历）。

    P2-Q4-fix(L431): 原实现按自然日粗判（周五买入周日即视为可卖），改为
    挂靠 market_clock 交易日历——最早可卖日 = buy_date 的下一个交易日；
    market_clock 不可用时降级为原自然日粗判（可见降级，不静默吞异常）。
    """
    try:
        from market_clock import next_trading_day  # 懒加载，避免 market_rules↔market_clock 循环导入
        earliest = next_trading_day(buy_date, n=1)
        if earliest is None:
            return False
        return sell_date.date() >= earliest
    except Exception:
        return (sell_date.date() - buy_date.date()).days >= 1


# ══════════════════════════════════════════════════════════════════
# 十、融资融券（证监会/交易所规则）
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MarginRule:
    asset_threshold: float   # 前20交易日日均资产门槛（元）
    experience_months: int   # 交易经验要求（月）
    note: str = ""


MARGIN_RULES = {
    "沪深A股": MarginRule(500_000, 6, "开通融资融券：前20交易日日均资产≥50万+6个月交易经验"),
    "创业板": MarginRule(100_000, 24, "创业板权限：前20交易日日均资产≥10万+24个月经验"),
    "科创板": MarginRule(500_000, 24, "科创板权限：前20交易日日均资产≥50万+24个月经验"),
    "北交所": MarginRule(500_000, 24, "北交所权限：前20交易日日均资产≥50万+24个月经验"),
}
# P2-Q4-fix(M412): 2023-09-08 起融资保证金最低比例由 100% 降至 80%（沪深北同步）
MARGIN_RATIO_DEFAULT = 0.8      # 融资保证金最低比例 80%（2023-09-08 起生效）
MAINTENANCE_RATIO_DEFAULT = 1.3  # 维持担保比例警戒线 130%


# ══════════════════════════════════════════════════════════════════
# 十一、退市规则（2024-04-30 退市新规后现行）
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DelistingRule:
    face_value_days: int        # 连续N日收盘价低于面值(1元) → 交易类退市
    face_value: float = 1.0
    market_cap_days: int = 20   # 连续N日总市值低于阈值 → 交易类退市
    market_cap_threshold: float = 5e8  # 主板 5 亿元（2024新规）；双创 3 亿见 MARKET_CAP_THRESHOLD_STAR
    delisting_period_days: int = 15    # 退市整理期 15 个交易日
    note: str = "交易类退市(面值/市值)无整理期直接退市；财务/规范/重大违法类退市整理期15个交易日"


DELISTING_RULE = DelistingRule(
    face_value_days=20,
    face_value=1.0,
    market_cap_days=20,
    market_cap_threshold=5e8,
    delisting_period_days=15,
    note="2024-04-30 退市新规：主板市值退市标准5亿/科创、创业板3亿；北交所适用面值退市",
)

# P2-Q4-fix(M413): 市值退市板块阈值（2024-04-30 退市新规）——主板 5 亿 / 科创、创业板 3 亿。
# 主板阈值直接接线 DelistingRule.market_cap_threshold，消除该字段"死字段"问题。
MARKET_CAP_THRESHOLD_MAIN = DELISTING_RULE.market_cap_threshold
MARKET_CAP_THRESHOLD_STAR = 3e8


def is_face_value_delisting(code: str, close_prices: list[float]) -> bool:
    """面值退市判断：连续20个交易日收盘价低于1元。

    P2-Q4-fix(M415): 删除冗余双分支（原两分支阈值相同造成死分支），统一为
    单一面值阈值；主板/创业板/科创板面值标准一致（均为 1 元）。
    """
    threshold = DELISTING_RULE.face_value
    recent = close_prices[-DELISTING_RULE.face_value_days:]
    if len(recent) < DELISTING_RULE.face_value_days:
        return False
    return all(p < threshold for p in recent)


def is_market_cap_delisting(code: str, market_caps: list[float]) -> bool:
    """市值退市判断（P2-Q4-fix M413，2024-04-30 退市新规）。

    连续 market_cap_days（默认20）个交易日总市值均低于对应板块阈值：
      - 沪/深主板: 5 亿元（MARKET_CAP_THRESHOLD_MAIN）
      - 科创板/创业板: 3 亿元（MARKET_CAP_THRESHOLD_STAR）
      - 北交所: 不适用市值退市 → 恒 False

    Args:
        code: 股票代码（6位）
        market_caps: 每日总市值序列（元），按时间升序

    Returns:
        True=触发市值退市；False=未触发/数据不足/板块不适用
    """
    board = detect_board(code)
    if board == "北交所":
        return False
    if board in ("科创板", "创业板"):
        threshold = MARKET_CAP_THRESHOLD_STAR
    else:
        threshold = MARKET_CAP_THRESHOLD_MAIN  # 沪/深主板（及未知默认）
    days = DELISTING_RULE.market_cap_days
    recent = market_caps[-days:]
    if len(recent) < days:
        return False
    return all(c < threshold for c in recent)


# ══════════════════════════════════════════════════════════════════
# 十二、新股/特别标识
# ══════════════════════════════════════════════════════════════════

# 上市首日简称加 "N"，上市次日至第5个交易日加 "C"（主板/双创）
NEW_STOCK_PREFIX = "N"
NEW_STOCK_C_PREFIX = "C"


def listing_day_label(trade_day: int) -> str:
    """上市第 N 个交易日的简称前缀。"""
    if trade_day <= 1:
        return "N"
    if trade_day <= 5:
        return "C"
    return ""


# ══════════════════════════════════════════════════════════════════
# 十三、汇总导出（供文档/RAG/审计引用）
# ══════════════════════════════════════════════════════════════════

RULES_SUMMARY = {
    "涨跌幅": {b: f"±{r.normal_pct}% (ST: ±{r.st_pct}%)" for b, r in PRICE_LIMIT_RULES.items()},
    "新股无涨跌幅": {b: f"前{r.new_stock_no_limit_days}日" for b, r in PRICE_LIMIT_RULES.items()},
    "交易时间": "9:15-9:25集合竞价(9:20后不可撤)/9:30-11:30、13:00-14:57连续竞价/14:57-15:00收盘竞价/15:00-15:30大宗(深市盘后定价15:05-15:30)",
    "价格笼子": "2% 与 10个最小变动单位孰高/孰低（北交所无）",
    "交易费用": FEE_DEFAULT.note,
    "T+1": "当日买入次一交易日方可卖出",
    "大宗交易": {b: r.note for b, r in BLOCK_TRADE_RULES.items()},
    "临时停牌": CircuitBreakerRule.note,
    "退市": DELISTING_RULE.note,
    "两融门槛": {k: v.note for k, v in MARGIN_RULES.items()},
}


def print_summary() -> str:
    """输出规则汇总（供审计/文档使用）。"""
    lines = ["═══ A股市场交易规则硬编码库（market_rules.py）═══"]
    for k, v in RULES_SUMMARY.items():
        if isinstance(v, dict):
            lines.append(f"▸ {k}:")
            for kk, vv in v.items():
                lines.append(f"    {kk}: {vv}")
        else:
            lines.append(f"▸ {k}: {v}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# 自检
# ══════════════════════════════════════════════════════════════════

def self_test() -> list[str]:
    ok: list[str] = []
    # 板块识别
    assert detect_board("600519") == "沪主板"
    assert detect_board("688981") == "科创板"
    assert detect_board("000001") == "深主板"
    assert detect_board("300750") == "创业板"
    assert detect_board("830799") == "北交所"
    assert detect_board("920001") == "北交所"
    # 涨跌幅
    assert get_price_limit_pct("600519") == 10.0
    assert get_price_limit_pct("000001", "ST康美") == 5.0
    assert get_price_limit_pct("300750") == 20.0
    assert get_price_limit_pct("688981") == 20.0
    assert get_price_limit_pct("830799") == 30.0
    assert get_price_limit_pct("688981", "", trade_day=3) == 0.0   # 新股前5日
    assert get_price_limit_pct("600519", "", trade_day=8) == 10.0
    assert get_price_limit_pct("600519", "", delisting_period=True, trade_day=1) == 0.0
    lo, hi = price_limit_px("600519", 100.0)
    assert lo == 90.0 and hi == 110.0
    # 数量规则
    assert validate_order_qty("600519", 100) is True
    assert validate_order_qty("600519", 150) is False
    assert validate_order_qty("688981", 200) is True
    assert validate_order_qty("688981", 201) is True
    assert validate_order_qty("830799", 101) is True
    # 费用
    fee = calc_trade_fee(10_000, "sell")
    assert fee["stamp_tax"] == 5.0
    assert fee["transfer_fee"] == 0.1
    assert fee["commission"] >= 5.0
    # 价格笼子（V11 审计修复: 买入受上限约束、卖出受下限约束）
    lo2, hi2 = effective_price_range("600519", 10.0, "buy")
    assert hi2 == 10.2  # 买入上限=10×(1+2%)=10.2 vs 10+0.1=10.1 → 孰高 10.2
    lo3, hi3 = effective_price_range("600519", 10.0, "sell")
    assert lo3 == 9.8  # 卖出下限=10×(1-2%)=9.8 vs 10-0.1=9.9 → 孰低 9.8
    assert effective_price_range("830799", 10.0)[0] == 0.0  # 北交所无笼子
    # 临停
    assert circuit_breaker_trigger(10.0, 13.2) == 30.0
    assert circuit_breaker_trigger(10.0, 16.5) == 60.0
    assert circuit_breaker_trigger(10.0, 11.0) is None
    # 市值退市（P2-Q4-fix M413）
    assert is_market_cap_delisting("600000", [6e8] * 20) is False   # 主板 ≥5亿 不触发
    assert is_market_cap_delisting("600000", [4e8] * 20) is True    # 主板 <5亿 触发
    assert is_market_cap_delisting("300750", [4e8] * 20) is False   # 创业板 <5亿但 ≥3亿 不触发
    assert is_market_cap_delisting("300750", [2.5e8] * 20) is True  # 创业板 <3亿 触发
    assert is_market_cap_delisting("830799", [1e8] * 20) is False   # 北交所不适用市值退市
    # 午间阶段（P2-Q4-fix M414）：沪市不可申报/撤单，深市可
    _lunch = datetime(2024, 6, 3, 12, 0, tzinfo=CST)  # 2024-06-03 周一
    assert current_phase(_lunch, "沪市") == "午间休市(沪:不可申报)"
    assert can_cancel_now(_lunch, "沪市") is False
    assert can_cancel_now(_lunch, "深市") is True
    ok.append("market_rules self-test: ALL PASS ✅")
    return ok


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        for line in self_test():
            print(line)
    else:
        print(print_summary())
