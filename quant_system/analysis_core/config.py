"""
analysis_core.config — 路径与阈值常量（V11 分析包统一配置）

数据契约:
  zt_pool_history.parquet    K线重建的涨停/炸板/跌停长表（2020→今，历史地基）
  zt_pool_em_daily.parquet   东财涨停/炸板/跌停三池精确日更（含封板资金/炸板次数/行业）
  stock_names.parquet        A股代码-名称-ST标记映射（月度刷新）
  zt_daily_stats.parquet     每日聚合统计（情绪周期/天梯/晋级率的地基）
  predictions.parquet        预测记录库（进化闭环）
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # workspace
MARKET_DIR = ROOT / "data_warehouse" / "market"
PRED_DIR = ROOT / "data_warehouse" / "predictions"

ZT_HISTORY = MARKET_DIR / "zt_pool_history.parquet"
ZT_EM_DAILY = MARKET_DIR / "zt_pool_em_daily.parquet"
ZT_DAILY_STATS = MARKET_DIR / "zt_daily_stats.parquet"
NAME_MAP = MARKET_DIR / "stock_names.parquet"
PREDICTIONS = PRED_DIR / "predictions.parquet"

# ── 涨停规则（按代码前缀）───────────────────────────────
# 主板 10%，创业板(30x/301) 与科创板(688) 20%，北交所(8x/4x/92x) 30%
LIMIT_RULES: list[tuple[tuple[str, ...], float]] = [
    (("30", "68"), 0.20),
    (("8", "4", "92"), 0.30),
]
DEFAULT_LIMIT = 0.10

# 判定容差（涨跌幅与理论涨停价的接近程度）
ZT_EPS = 1e-6          # 收盘价与涨停价比较容差
PCT_TOL = 0.3          # 涨跌幅容差（百分点）：如 9.7% 即视为触及 10% 涨停

# ── 情绪周期六阶段阈值（V1 经验值，随预测记录库校准）────
EMOTION_THRESHOLDS = {
    "ice":        {"zt_cnt_max": 35,  "max_board_max": 2,  "premium_max": 0.0},
    "repair":     {"zt_cnt_min": 30,  "zt_cnt_max": 65,    "max_board_min": 2, "max_board_max": 3},
    "ferment":    {"zt_cnt_min": 55,  "max_board_min": 3,  "jr1_min": 0.28, "premium_min": 0.3},
    "climax":     {"zt_cnt_min": 95,  "max_board_min": 5},
    "divergence": {"zb_rate_min": 0.38},
    "ebb":        {"max_board_max": 2, "zt_cnt_max": 45,   "premium_max": -1.0},
}

# ── 决策层硬规则（防手痒）───────────────────────────────
DECISION_RULES = {
    "ice_ebb_force_light": True,   # 冰点/退潮只允许 空仓/试错 档
    "max_position_ferment": 0.8,
    "max_position_climax": 0.6,
    "max_position_divergence": 0.4,
    "max_position_ice": 0.2,
    "max_position_ebb": 0.1,
}
