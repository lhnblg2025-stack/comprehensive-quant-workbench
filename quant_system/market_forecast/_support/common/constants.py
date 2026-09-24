"""
constants.py — A股量化系统常量定义
QuantV6 公共层：所有模块共享的常量，避免魔法数字散落各处。
"""
from __future__ import annotations

from datetime import time, timedelta, timezone

# ── 时区 ──────────────────────────────────────────────
CST = timezone(timedelta(hours=8), name="CST")          # 中国标准时间 UTC+8
TRADING_DAYS_PER_YEAR = 252                               # 年化交易日

# ── 交易时段（北京时间）────────────────────────────────
COLLECTION_START = time(9, 15)     # 集合竞价开始
CONTINUOUS_START = time(9, 30)     # 连续竞价开始（可交易）
MORNING_END = time(11, 30)         # 早盘收盘
LUNCH_START = time(11, 30)         # 午休开始
AFTERNOON_START = time(13, 0)      # 下午开盘
CLOSE = time(15, 0)                # 收盘

# ── 交易成本 ──────────────────────────────────────────
COMMISSION_RATE = 0.00085          # 佣金 万0.85
MIN_COMMISSION = 5.0               # 最低佣金 5 元（单笔最低；全链路唯一取值点）
STAMP_TAX_RATE = 0.0005            # 印花税 万5（仅卖出）
TRANSFER_FEE_RATE = 0.00001        # 过户费 0.001%（沪深双向收取）
SLIPPAGE_RATE = 0.001              # 默认滑点 0.1%

# ── 涨跌停幅度 ────────────────────────────────────────
# 判断优先级：先板块（双创 20%/北交所 30%），非双创再判断 ST（5%），否则主板 10%。
LIMIT_UP_PCT = 0.10                # 主板 10%（非 ST）
LIMIT_UP_PCT_CHINEXT = 0.20        # 创业板(300/301) 20%（ST 同样 20%）
LIMIT_UP_PCT_STAR = 0.20           # 科创板(688/689) 20%（ST 同样 20%）
LIMIT_UP_PCT_BSE = 0.30            # 北交所(8xxxxx/4xxxxx/920xxx) 30%
LIMIT_UP_PCT_ST = 0.05             # 主板 ST 5%

# ── 最小交易单位 ──────────────────────────────────────
LOT_SIZE_MAIN = 100                # 主板最小 100 股
LOT_SIZE_STAR = 200                # 科创板最小 200 股

# ── 指数代码映射（新浪接口 endswith 匹配）────────────
INDEX_MAP = {
    "上证指数": "000001",
    "深证成指": "399001",
    "创业板指": "399006",
    "科创50": "000688",
    "沪深300": "000300",
    "中证500": "000905",
    "中证1000": "000852",
    "中证2000": "932000",
}

# ── 板块/概念 ─────────────────────────────────────────
TMT_INDUSTRIES = ("C39", "I63", "I64", "I65", "R86", "R87")  # 证监会TMT口径

# ── 风控阈值 ──────────────────────────────────────────
STOP_LOSS_PCT = 0.08               # 个股止损 -8%
TAKE_PROFIT_PCT = 0.15             # 个股止盈 +15%
TRAILING_STOP_PCT = 0.05           # 移动止盈回撤 5%
MAX_POSITION_PCT = 0.30            # 单票仓位上限 30%
MAX_DAILY_LOSS_PCT = 0.02          # 日亏损上限 -2%
PAUSE_AFTER_LOSS_DAYS = 3          # 连续亏损暂停天数
MAX_DRAWDOWN_PCT = 0.10            # 最大回撤触发线 -10%

# ── 情绪过热阈值（预测层使用）───────────────────────
HEAT_LIMIT_UP = 100                # 涨停家数过热阈值
HEAT_RISE_RATIO = 0.85             # 上涨家数比过热阈值
FROZEN_LIMIT_UP = 15               # 涨停家数冰点阈值

# ── ST/退市 识别模式 ──────────────────────────────────
ST_PATTERNS = ("ST", "*ST", "退")

# ── 数据缓存 TTL（秒）────────────────────────────────
TTL_INTRADAY = 30                  # 盘中数据 30s
TTL_DAILY = 86400                  # 日频 1天
TTL_WEEKLY = 604800                # 低频 7天

# ── 默认持仓配置 ──────────────────────────────────────
MAX_POSITIONS = 4                  # 最大持仓数
CANDIDATE_LIMIT = 12               # 候选池上限
MIN_SCORED = 5                     # 最少有效评分数，不足则 hold
INITIAL_CASH = 100_000.0           # 模拟盘初始资金 10万

# ── 因子窗口 ──────────────────────────────────────────
HIST_DAYS = 420                    # K线历史窗口（≥252交易日）
Z_CLIP = 3.0                       # 因子zscore截断
MIN_VALID_FACTORS = 3              # 最少有效因子数，不足跳过该股

# ── 状态文件 ──────────────────────────────────────────
STATE_DIR_DEFAULT = "/root/quant/state"
SNAPSHOT_KEEP_DAYS = 30            # 快照保留天数

# P2-Q12-fix: 限定 import * 导出，避免 datetime.time/timedelta/timezone 泄漏到 common 命名空间
__all__ = [
    "CST", "TRADING_DAYS_PER_YEAR",
    "COLLECTION_START", "CONTINUOUS_START", "MORNING_END", "LUNCH_START", "AFTERNOON_START", "CLOSE",
    "COMMISSION_RATE", "MIN_COMMISSION", "STAMP_TAX_RATE", "TRANSFER_FEE_RATE", "SLIPPAGE_RATE",
    "LIMIT_UP_PCT", "LIMIT_UP_PCT_CHINEXT", "LIMIT_UP_PCT_STAR", "LIMIT_UP_PCT_BSE", "LIMIT_UP_PCT_ST",
    "LOT_SIZE_MAIN", "LOT_SIZE_STAR", "INDEX_MAP", "TMT_INDUSTRIES",
    "STOP_LOSS_PCT", "TAKE_PROFIT_PCT", "TRAILING_STOP_PCT", "MAX_POSITION_PCT", "MAX_DAILY_LOSS_PCT",
    "PAUSE_AFTER_LOSS_DAYS", "MAX_DRAWDOWN_PCT", "HEAT_LIMIT_UP", "HEAT_RISE_RATIO", "FROZEN_LIMIT_UP",
    "ST_PATTERNS", "TTL_INTRADAY", "TTL_DAILY", "TTL_WEEKLY", "MAX_POSITIONS", "CANDIDATE_LIMIT",
    "MIN_SCORED", "INITIAL_CASH", "HIST_DAYS", "Z_CLIP", "MIN_VALID_FACTORS", "STATE_DIR_DEFAULT", "SNAPSHOT_KEEP_DAYS",
]
