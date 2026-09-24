"""
日内决策引擎 — 盘中交互式股票分析与交易建议。

用法（我在对话中调用）：
    from quant_system.intraday_decision import decide_stock
    result = decide_stock("600519")   # 茅台全量分析

    from quant_system.intraday_decision import scan_market
    result = scan_market()            # 当天市场扫描+信号
"""

from __future__ import annotations
import logging

from datetime import timedelta, timezone
from typing import Any, Optional

import pandas as pd

from .config import DEFAULT_STRATEGY
from .data import aggregate_timeframe, fetch_daily, resolve_symbol, search_stock_name
from .global_market import (
    fetch_global_quotes,
    HK_KNOWN,
    US_KNOWN,
)
from .indicators import add_technical_indicators
from .margin import fetch_margin_summary, fetch_margin_individual
from .realtime import fetch_realtime, RealtimeQuote
from .signals import latest_signal

# Sina 分钟K线（60分钟 CCI + 均线系统）
try:
    from .sources_sina_intraday import analyze_intraday_hourly as _intra_hourly
except Exception:
    _intra_hourly = None

# 市场背景数据
try:
    from .market_context import get_market_context as _ctx
except Exception:
    _ctx = None

# 尝试加载股票池（可选，用于名称解析）
try:
    from scripts.stock_screener import search_stock as _ss_search
except Exception:
    _ss_search = None

from quant_system.utils import now_cst

CST = timezone(timedelta(hours=8))


# ═══════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════

def is_trading_hours() -> bool:
    n = now_cst()
    if n.weekday() >= 5:
        return False
    t = n.hour * 60 + n.minute
    # P2-P2-Q10-M024-fix: 原 9:30<=t<=15:00 把午休(11:30-13:00)与尾盘竞价(14:57-15:00)
    # 都算交易时段, 与 monitor(930-1130|1300-1457) 口径不一致。现对齐。
    return (9 * 60 + 30 <= t <= 11 * 60 + 30) or (13 * 60 <= t <= 14 * 60 + 57)


def is_premarket() -> bool:
    """集合竞价窗口 9:15-9:25"""
    n = now_cst()
    t = n.hour * 60 + n.minute
    return 9 * 60 + 15 <= t <= 9 * 60 + 25


def seconds_to_close() -> int:
    """距收盘秒数（仅统计剩余交易时段，午休不计入；盘前/盘后/周末=0）"""
    n = now_cst()
    hm = n.hour * 100 + n.minute
    if n.weekday() >= 5:
        return 0
    if 930 <= hm <= 1130:
        # 上午剩余交易时间 + 整个下午(120分钟)
        morning_remain = int((n.replace(hour=11, minute=30, second=0, microsecond=0) - n).total_seconds())
        return max(0, morning_remain) + 120 * 60
    if 1300 <= hm < 1500:
        return max(0, int((n.replace(hour=15, minute=0, second=0, microsecond=0) - n).total_seconds()))
    if 1130 < hm < 1300:
        # P2-P2-Q10-M032-fix: 午休 11:30-13:00 不显示"距收盘 2 小时"误导, 只算剩余下午
        return 120 * 60
    return 0


# ═══════════════════════════════════════════════════════════════════
# 第一重: 市场背景扫描 (Triple Screen Level 1)
# ═══════════════════════════════════════════════════════════════════

def assess_market_triple_screen() -> dict[str, Any]:
    """三重滤网市场评估。"""
    result = {"ok": True, "time": now_cst().isoformat()}

    # 第一重：指数趋势
    indices = {
        "sh000001": "上证指数",
        "sz399001": "深证成指",
        "sz399006": "创业板指",
    }
    index_data = {}
    for sym, name in indices.items():
        try:
            # Q10-fix: fetch_realtime 签名是 list[str]，传字符串会被逐字符迭代；
            # 返回 dict{短代码: quote}，需先取列表再按规范化键取值。
            quotes = fetch_realtime([sym], timeout=6)
            q = quotes.get(sym.lstrip("shszbj")) if quotes else None
            if q and q.price:
                trend = "up" if q.price >= q.prev_close else "down"
                index_data[name] = {
                    "price": q.price,
                    "change_pct": (q.price - q.prev_close) / q.prev_close * 100,
                    "trend": trend,
                }
        except Exception as e:
            logging.getLogger(__name__).error(f"[intraday_decision] 操作失败: {e}", exc_info=True)
    result["indices"] = index_data

    # 两融变化
    try:
        margin = fetch_margin_summary()
        if margin and margin.get("ok"):
            m = margin
            result["margin_total"] = m.get("total", 0)
            result["margin_change"] = m.get("change", 0)
    except Exception as e:
        logging.getLogger(__name__).error(f"[intraday_decision] 操作失败: {e}", exc_info=True)

    # 市场状态判断
    up_count = sum(1 for v in index_data.values() if v.get("trend") == "up")
    # Q10-fix: 指数取数全部失败时不能再判 "bearish"（原逻辑 up_count<=0 恒成立
    # -> 三重滤网第一重恒看空）。数据缺失时应显式标记 unknown。
    if not index_data:
        result["market_bias"] = "unknown"
    elif up_count >= 2:
        result["market_bias"] = "bullish"
    elif up_count <= 0:
        result["market_bias"] = "bearish"
    else:
        result["market_bias"] = "mixed"

    if result.get("margin_change", 0) < -20:
        result["margin_signal"] = "caution"       # 两融大幅减少→谨慎
    elif result.get("margin_change", 0) > 20:
        result["margin_signal"] = "active"        # 两融大增→活跃
    else:
        result["margin_signal"] = "neutral"

    return result


# ═══════════════════════════════════════════════════════════════════
# 第二重: 个股三重滤网判断
# ═══════════════════════════════════════════════════════════════════

def triple_screen_stock(df: pd.DataFrame) -> dict[str, Any]:
    """对某只股票执行三重滤网分析。"""
    if df.empty or len(df) < 50:
        return {"ok": False, "error": "数据不足", "screen_1": "unknown"}

    # 确保指标已计算
    df_i = add_technical_indicators(df, DEFAULT_STRATEGY)
    latest = df_i.iloc[-1]

    # P2-P2-Q10-M025-fix: Elder 三重滤网第一重要求**独立周期**的周线 MACD 趋势,
    # 原用最近 5 根日线冒充周线差异大。改用 data.aggregate_timeframe 聚合真实周线。
    weekly_agg_failed = False
    try:
        weekly_df = aggregate_timeframe(df, "weekly")
        weekly_i = add_technical_indicators(weekly_df, DEFAULT_STRATEGY)
        weekly_close = weekly_i["close"].values.astype(float)
        macd_hist_5d = weekly_i["macd_hist"].dropna().values
    except Exception:
        # 聚合失败时降级为原日线近似(降级可见: 结果标记 weekly_agg_failed=True)
        weekly_agg_failed = True
        weekly = df_i.iloc[-5:]
        weekly_close = weekly["close"].values
        macd_hist_5d = weekly["macd_hist"].values if "macd_hist" in weekly.columns else []

    # --- 第一重: 周线MACD趋势 (真实周线) ---
    # 周线MACD方向：取MACD柱近期方向
    if len(macd_hist_5d) >= 2:
        screen1 = "up" if macd_hist_5d[-1] > macd_hist_5d[0] else "down" if macd_hist_5d[-1] < macd_hist_5d[0] else "flat"
    else:
        screen1 = "unknown"

    # 用当前价格与MA60/MA144的比较验证
    close = latest.get("close", 0)
    ma60 = latest.get("ma_slow", 0)
    ma144 = latest.get("ma_trend", 0)
    if screen1 == "up" and close < ma60:
        screen1 = "weak_up"  # 均线上是上升趋势但价格已跌破短期均线

    # --- 第二重: 日线振荡指标 ---
    rsi = latest.get("rsi_14", 50)
    kdj_j = latest.get("kdj_j", 50)
    macd_hist_now = latest.get("macd_hist", 0)
    volume_ratio = latest.get("volume_ratio", 1.0)
    atr_pct = latest.get("atr_pct", 1.0)

    # 振荡指标状态
    if rsi > 70:
        oscillator_state = "overbought"
    elif rsi < 30:
        oscillator_state = "oversold"
    elif rsi > 60:
        oscillator_state = "strong"
    elif rsi < 40:
        oscillator_state = "weak"
    else:
        oscillator_state = "neutral"

    # 是否有背离（简化：看MACD柱与价格方向是否一致）
    # 取最近5根判断
    prices_5d = weekly_close
    if len(prices_5d) >= 3 and len(macd_hist_5d) >= 3:
        price_dir = prices_5d[-1] > prices_5d[-3]  # 价格方向(涨/跌)
        macd_dir = macd_hist_5d[-1] > macd_hist_5d[-3]  # MACD柱方向
        divergence = "none"
        if price_dir and not macd_dir:
            divergence = "bearish_divergence"  # 顶背离
        elif not price_dir and macd_dir:
            divergence = "bullish_divergence"  # 底背离
    else:
        divergence = "unknown"

    # --- 第三重: 盘中入场条件（点位分析）---
    # 计算当前价格在BOLL通道中的位置
    boll_upper = latest.get("boll_upper", close)
    boll_lower = latest.get("boll_lower", close)
    boll_mid = latest.get("boll_mid", close)
    boll_width = boll_upper - boll_lower
    boll_pos = (close - boll_lower) / boll_width * 100 if boll_width > 0 else 50

    # 计算近10日最高最低
    last_10 = df_i.tail(10)
    high_10 = last_10["high"].max()
    low_10 = last_10["low"].min()
    range_10 = high_10 - low_10
    range_pos = (close - low_10) / range_10 * 100 if range_10 > 0 else 50

    # 综合判断
    result = {
        "ok": True,
        "screen_1": screen1,               # 周线趋势方向
        "weekly_agg_failed": weekly_agg_failed,  # P2-P2-Q10-M025-fix: 周线聚合失败降级标记
        "screen_2": {
            "oscillator": oscillator_state, # 振荡指标状态
            "rsi_14": round(rsi, 1),
            "rsi_state": "超买" if rsi > 70 else "超卖" if rsi < 30 else "正常",
            "macd_hist": round(macd_hist_now, 3),
            "macd_signal": "多头" if macd_hist_now > 0 else "空头" if macd_hist_now < 0 else "中性",
            "divergence": divergence,       # 背离信号
            "volume_ratio": round(volume_ratio, 2),
            "volume_state": "放量" if volume_ratio > 1.5 else "缩量" if volume_ratio < 0.7 else "正常",
        },
        "screen_3": {
            "close": round(close, 2),
            "boll_pos_pct": round(boll_pos, 1),      # BOLL位置(0-100)
            "boll_zone": "上轨" if boll_pos > 80 else "下轨" if boll_pos < 20 else "中轨区",
            "range_pos_pct": round(range_pos, 1),     # 10日价格位置
            "atr_pct": round(atr_pct, 2),             # 波动率
            "atr_zone": "高波动" if atr_pct > 3 else "低波动" if atr_pct < 1 else "正常波动",
        },
        "combined_signal": "",
    }

    # 综合信号
    signals = []
    if screen1 in ("up", "weak_up") and oscillator_state in ("oversold", "weak", "neutral") and divergence != "bearish_divergence":
        signals.append("做多偏多")
    if screen1 == "down" and oscillator_state in ("overbought", "strong", "neutral") and divergence != "bullish_divergence":
        signals.append("做空偏空")
    if divergence == "bullish_divergence":
        signals.append("底背离(看涨)")
    if divergence == "bearish_divergence":
        signals.append("顶背离(看跌)")
    if oscillator_state == "oversold" and screen1 == "up":
        signals.append("回调至超卖(参照三重滤网做多)")
    if oscillator_state == "overbought" and screen1 == "down":
        signals.append("反弹至超买(参照三重滤网做空)")
    if not signals:
        signals.append("无明显信号")

    result["combined_signal"] = " | ".join(signals)
    return result


# ═══════════════════════════════════════════════════════════════════
# 第三重: 策略匹配 — 当前哪9种日内策略适用
# ═══════════════════════════════════════════════════════════════════

def match_intraday_strategies(
    triple_screen: dict,
    realtime: Optional[RealtimeQuote] = None,
    df: Optional[pd.DataFrame] = None,
) -> list[dict[str, Any]]:
    """根据当前市场状态匹配适用的Aziz/Carter交易策略。"""
    s1 = triple_screen.get("screen_1", "unknown")
    s2 = triple_screen.get("screen_2", {})
    s3 = triple_screen.get("screen_3", {})

    strategies = []

    # --- Aziz 9策略 ---
    # 1. ABCD 模式 (需要趋势+candle形态, 盘中判断)
    if s1 in ("up", "weak_up") and s2.get("oscillator") not in ("overbought",):
        strategies.append({
            "name": "ABCD 模式",
            "author": "Aziz",
            "condition": "上升趋势中有回调+再次突破前高",
            "suitability": "中等" if s1 == "weak_up" else "高",
            "entry_trigger": "价格突破近期回调的高点",
            "stop": "回调低点下方",
            "target": "A→B等同涨幅",
        })

    # 2. Bull Flag 动量 (需要陡峭拉升+缩量整理)
    if s2.get("volume_state") == "缩量" and s2.get("oscillator") in ("strong", "neutral"):
        strategies.append({
            "name": "牛市旗形 (Bull Flag)",
            "author": "Aziz",
            "condition": "陡峭拉升后缩量整理",
            "suitability": "中(需盘中观察旗形结构)",
            "entry_trigger": "价格突破旗面上沿+放量",
            "stop": "旗面下沿",
            "target": "+1旗杆高度",
        })

    # 3-4. 反转交易 (需要趋势+反转K线)
    if s2.get("divergence") == "bullish_divergence":
        strategies.append({
            "name": "底部反转",
            "author": "Aziz",
            "condition": "底背离+出现看涨反转K线(锤子线/看涨吞没)",
            "suitability": "高",
            "entry_trigger": "反转K线确认",
            "stop": "近期低点下方",
            "target": "前高或阻力位",
        })
    if s2.get("divergence") == "bearish_divergence":
        strategies.append({
            "name": "顶部反转",
            "author": "Aziz",
            "condition": "顶背离+出现看跌反转K线(射击之星/看跌吞没)",
            "suitability": "高",
            "entry_trigger": "反转K线确认",
            "stop": "近期高点上方",
            "target": "前低或支撑位",
        })

    # 5. 均线趋势交易
    if s1 in ("up", "weak_up"):
        strategies.append({
            "name": "均线趋势(做多)",
            "author": "Aziz",
            "condition": "价格回踩EMA9/EMA20/EMA50",
            "suitability": "高",
            "entry_trigger": "价格触及均线+企稳",
            "stop": "EMA50下方",
            "target": "前高或BOLL上轨",
        })
    if s1 == "down":
        strategies.append({
            "name": "均线趋势(做空)",
            "author": "Aziz",
            "condition": "价格反弹至EMA9/EMA20/EMA50遇阻",
            "suitability": "高",
            "entry_trigger": "价格触及均线+回落",
            "stop": "EMA50上方",
            "target": "前低或BOLL下轨",
        })

    # 6. VWAP交易 (需盘中数据)
    if realtime:
        strategies.append({
            "name": "VWAP 交易",
            "author": "Aziz",
            "condition": f"当前{realtime.price} vs VWAP(需实时计算)",
            "suitability": "中(需实时VWAP数据)",
            "entry_trigger": "价格突破VWAP站稳/跌破VWAP无法收复",
            "stop": "VWAP反方向",
            "target": "下一技术位",
        })

    # 7. 支撑/阻力交易
    strategies.append({
        "name": "支撑/阻力交易",
        "author": "Aziz",
        "condition": "价格触及关键支撑/阻力位",
        "suitability": "高",
        "entry_trigger": "触及支撑反弹/触及阻力回落",
        "stop": "支撑下方/阻力上方",
        "target": "反向关键位",
    })

    # 8. Red-to-Green (需盘中观察跳空)
    if realtime and realtime.prev_close > 0:
        gap_pct = (realtime.price - realtime.prev_close) / realtime.prev_close * 100
        if gap_pct < -1:
            strategies.append({
                "name": "Red-to-Green",
                "author": "Aziz",
                "condition": f"低开{gap_pct:.1f}%，观察能否翻红",
                "suitability": "中(需实时观察)",
                "entry_trigger": "价格回升至昨日收盘价以上",
                "stop": "当日最低点下方",
                "target": "近期阻力位",
            })
        elif gap_pct > 1:
            strategies.append({
                "name": "Red-to-Green(反向做空)",
                "author": "Aziz",
                "condition": f"高开{gap_pct:.1f}%，观察能否维持",
                "suitability": "中(需实时观察)",
                "entry_trigger": "价格回落至昨日收盘价以下",
                "stop": "当日最高点上方",
                "target": "近期支撑位",
            })

    # 9. ORB (开盘区间突破, 需开盘后5分钟数据)
    strategies.append({
        "name": "开盘区间突破 (ORB)",
        "author": "Aziz",
        "condition": "今日前5/15分钟形成区间，突破入场",
        "suitability": "高(开盘后使用)",
        "entry_trigger": "价格突破前15分钟区间",
        "stop": "VWAP或区间边界",
        "target": "前日关键位",
    })

    # --- Carter 特有策略 ---
    # 挤牌 (BOLL收缩)
    boll_width_ratio = s3.get("boll_pos_pct", 50)

    # 均值回归
    if s2.get("oscillator") in ("overbought", "oversold"):
        strategies.append({
            "name": "均值回归 (Mean Reversion)",
            "author": "Carter",
            "condition": f"RSI={s2.get('rsi_14', 50)}, 超买/超卖区域",
            "suitability": "高(震荡市)/低(趋势市)",
            "entry_trigger": "反向入场",
            "stop": "ATR×1.5",
            "target": "ATR×1-1.5",
        })

    # 轴心点交易
    strategies.append({
        "name": "轴心点交易 (Pivot Points)",
        "author": "Carter",
        "condition": "价格触及前日计算的Pivot S1/R1",
        "suitability": "高",
        "entry_trigger": "S1反弹做多 / R1回落做空",
        "stop": "S1/R1下方/上方少许",
        "target": "枢轴点(P) → R1/S1",
    })

    return strategies


# ═══════════════════════════════════════════════════════════════════
# 仓位计算器
# ═══════════════════════════════════════════════════════════════════

def calc_position_size(
    entry: float,
    stop: float,
    account_size: float,
    risk_pct: float = 1.0,
) -> dict[str, Any]:
    """计算建议持仓规模。

    P2-Q10-L041-fix: 移除从未使用的 price_step 参数与恒为空串的 reward_risk_target。
    P2-Q10-L042-fix: 补充 A股交易费用估算(佣金最低5元/笔、卖出印花税0.05%、过户费
    0.001%双边、滑点0.05%), max_loss 为毛利口径, 另给 max_loss_net 净亏口径。
    """
    risk_per_share = abs(entry - stop)
    if risk_per_share <= 0:
        return {"ok": False, "error": "入场价与止损价相同"}

    max_loss = account_size * risk_pct / 100
    shares_raw = max_loss / risk_per_share

    # 取整到百股（A股一手=100股）
    shares = int(shares_raw // 100 * 100)
    if shares <= 0:
        shares = 100

    total_value = shares * entry
    risk_ratio = risk_per_share / entry * 100

    # P2-P2-Q10-L042-fix: A股费用估算 —— 佣金最低5元/笔(双边, 按万3)、卖出印花税0.05%、
    # 过户费0.001%(双边)、滑点按入场价0.05%估算。卖出额按止损价估算(保守)。
    # V11 审计修复（Medium）: 原硬编码万3 与 config 万0.85 不一致 → 统一读 config。
    try:
        from quant_system.config import PortfolioConfig
        _comm = float(PortfolioConfig().commission_pct)
    except Exception:
        _comm = 0.000085
    buy_commission = max(5.0, shares * entry * _comm)
    sell_value = shares * stop
    sell_commission = max(5.0, sell_value * _comm)
    stamp_duty = sell_value * 0.0005             # 卖出印花税 0.05%
    transfer_fee = (total_value + sell_value) * 0.00001  # 过户费双边 0.001%
    slippage = total_value * 0.0005              # 入场滑点估算 0.05%
    estimated_costs = round(buy_commission + sell_commission + stamp_duty + transfer_fee + slippage, 2)

    return {
        "ok": True,
        "entry": entry,
        "stop": stop,
        "risk_per_share": round(risk_per_share, 2),
        "risk_pct_of_price": round(risk_ratio, 2),
        "max_loss": round(max_loss, 2),
        "estimated_costs": estimated_costs,
        "max_loss_net": round(max_loss + estimated_costs, 2),
        "suggested_shares": shares,
        "total_value": round(total_value, 2),
        "position_pct": round(total_value / account_size * 100, 1),
    }


# ═══════════════════════════════════════════════════════════════════
# 主决策入口
# ═══════════════════════════════════════════════════════════════════

def decide_stock(
    symbol: str,
    account_size: float = 1_000_000,
) -> dict[str, Any]:
    """
    单只股票全量决策分析。

    返回:
        - 基础信息
        - 实时行情（盘中）
        - 日线技术指标全量
        - 三重滤网评估
        - 适用的策略列表
        - 建议仓位
        - 综合信号
    """
    sym = resolve_symbol(symbol)

    # 1. 基础信息
    # 解析股票名称：优先用股票池，其次用data.py的search
    name = sym
    try:
        r = search_stock_name(sym)
        name = r[0]["name"] if r else sym
    except Exception as e:
        logging.getLogger(__name__).error(f"[intraday_decision] 操作失败: {e}", exc_info=True)
    if name == sym and _ss_search:
        try:
            r = _ss_search(sym)
            if r:
                name = r[0].get("name", sym)
        except Exception as e:
            logging.getLogger(__name__).error(f"[intraday_decision] 操作失败: {e}", exc_info=True)

    result = {
        "ok": True,
        "symbol": sym,
        "name": name,
        "time": now_cst().isoformat(),
        "trading_hours": is_trading_hours(),
        "premarket": is_premarket(),
        "seconds_to_close": seconds_to_close(),
        # P2-P2-Q10-L041-fix: 记录实际账户规模, 供 format_decision 显示(不再硬编码100万)
        "account_size": account_size,
    }

    # 2. 实时行情
    try:
        # Q10-fix: fetch_realtime 签名是 list[str]，传字符串会逐字符迭代；
        # 返回 dict{短代码: quote}。
        quotes = fetch_realtime([sym], timeout=6)
        rt = quotes.get(sym.lstrip("shszbj")) if quotes else None
        if rt:
            result["realtime"] = {
                "price": rt.price,
                "prev_close": rt.prev_close,
                "open": rt.open,
                "high": rt.high,
                "low": rt.low,
                "volume": rt.volume,
                "amount": rt.amount,
                "change_pct": round((rt.price - rt.prev_close) / rt.prev_close * 100, 2) if rt.prev_close else 0,
                "bid": rt.bid,
                "ask": rt.ask,
                "time": rt.time,
            }
    except Exception as e:
        logging.getLogger(__name__).error(f"[intraday_decision] 操作失败: {e}", exc_info=True)

    # 3. 日线技术分析
    from datetime import datetime as _dt, timedelta as _td
    end_str = _dt.now().strftime("%Y%m%d")
    # 动态起点（近 2 年），避免固定日期导致回看窗口随时间缩窄
    start_str = (_dt.now() - _td(days=365 * 2)).strftime("%Y%m%d")
    # P1-Q10-fix: fetch_daily 在无数据时抛 ValueError（data.py 无数据即 raise，不返回空 df），
    # 原 `if df.empty` 分支永不触发 → decide_stock 直接 traceback 而非返回错误 dict。
    # 用 try/except 捕获，失败时返回 {ok:False, error:...} 保持调用方契约。
    try:
        df = fetch_daily(sym, start=start_str, end=end_str, use_cache=True)
        if df.empty:
            df = fetch_daily(sym, start=(_dt.now() - _td(days=365 * 3)).strftime("%Y%m%d"), end=end_str, use_cache=True)
    except Exception as e:
        result["ok"] = False
        result["error"] = f"无法获取{sym}日线数据({e})"
        return result
    if df.empty:
        result["ok"] = False
        result["error"] = f"无法获取{sym}日线数据"
        return result

    df_i = add_technical_indicators(df, DEFAULT_STRATEGY)
    # P2-P2-Q10-L043-fix: snapshot = latest_indicator_snapshot(...) 计算后从未使用(死代码), 删除
    signal = latest_signal(sym, df_i, DEFAULT_STRATEGY)

    result["daily"] = {
        "latest_bar": {
            "date": str(df_i.iloc[-1]["date"]),
            "close": float(df_i.iloc[-1]["close"]),
            "volume": float(df_i.iloc[-1]["volume"]),
            "amount": float(df_i.iloc[-1]["amount"]),
        },
        "trend": {
            "ma20": float(df_i.iloc[-1].get("ma_fast", 0)),
            "ma60": float(df_i.iloc[-1].get("ma_slow", 0)),
            "ma144": float(df_i.iloc[-1].get("ma_trend", 0)),
            "ma300": float(df_i.iloc[-1].get("ma_long_trend", 0)),
        },
        "bollinger": {
            "upper": float(df_i.iloc[-1].get("boll_upper", 0)),
            "mid": float(df_i.iloc[-1].get("boll_mid", 0)),
            "lower": float(df_i.iloc[-1].get("boll_lower", 0)),
        },
        "indicators": {
            "rsi_14": float(df_i.iloc[-1].get("rsi_14", 0)),
            "macd_dif": float(df_i.iloc[-1].get("macd_dif", 0)),
            "macd_dea": float(df_i.iloc[-1].get("macd_dea", 0)),
            "macd_hist": float(df_i.iloc[-1].get("macd_hist", 0)),
            "kdj_k": float(df_i.iloc[-1].get("kdj_k", 0)),
            "kdj_d": float(df_i.iloc[-1].get("kdj_d", 0)),
            "kdj_j": float(df_i.iloc[-1].get("kdj_j", 0)),
            "volume_ratio": float(df_i.iloc[-1].get("volume_ratio", 1.0)),
            "atr_pct": float(df_i.iloc[-1].get("atr_pct", 0)),
            "trend_score": float(df_i.iloc[-1].get("trend_score", 0)),
            "momentum_score": float(df_i.iloc[-1].get("momentum_score", 0)),
            "risk_score": float(df_i.iloc[-1].get("risk_score", 0)),
            "composite_score": float(df_i.iloc[-1].get("composite_score", 0)),
        },
        "signal": signal,
        "n_bars": len(df_i),
    }

    # 4. 三重滤网评估
    ts = triple_screen_stock(df_i)
    result["triple_screen"] = ts

    # 4b. 小时级(60分钟) CCI + 均线分析
    if _intra_hourly:
        try:
            hourly = _intra_hourly(sym, datalen=200)
            if hourly and hourly.get("ok"):
                result["hourly_cci"] = hourly.get("cci")
                result["hourly_ma_signal"] = hourly.get("ma_signal")
                result["hourly_rsi"] = hourly.get("rsi_14")
                result["hourly_macd"] = hourly.get("macd")
                result["hourly_volume"] = hourly.get("volume")
                result["hourly_signals"] = hourly.get("hourly_signals", "")
                result["hourly_atr_pct"] = hourly.get("atr_pct", 0)
                result["hourly_trend"] = hourly.get("last_5_trend", "N/A")
        except Exception as e:
            logging.getLogger(__name__).error(f"[intraday_decision] 操作失败: {e}", exc_info=True)

    # 5. 策略匹配
    rt_obj = None
    try:
        rt_data = result.get("realtime", {})
        if rt_data:
            rt_obj = RealtimeQuote(
                symbol=sym,
                name=name,
                open=rt_data.get("open", 0),
                prev_close=rt_data.get("prev_close", 0),
                price=rt_data.get("price", 0),
                high=rt_data.get("high", 0),
                low=rt_data.get("low", 0),
                volume=rt_data.get("volume", 0),
                amount=rt_data.get("amount", 0),
                bid=rt_data.get("bid", 0),
                ask=rt_data.get("ask", 0),
                time=rt_data.get("time", ""),
            )
    except Exception as e:
        logging.getLogger(__name__).error(f"[intraday_decision] 操作失败: {e}", exc_info=True)

    strategies = match_intraday_strategies(ts, realtime=rt_obj, df=df_i)
    result["applicable_strategies"] = strategies

    # 6. 仓位建议（使用实时价+BOLL上下轨）
    close_price = result.get("realtime", {}).get("price") or result["daily"]["latest_bar"]["close"]
    boll = result["daily"]["bollinger"]

    # P2-P2-Q10-M028-fix: 止损取 -3% 和 BOLL 下轨中的"更严者"=更接近入场价者。
    # 原 `min(close*0.97, boll_lower)` 取的是更低(更松)的价位, 风险口径与注释矛盾;
    # 做多更严=取 max(做空相反取 min)。
    long_pos = calc_position_size(
        entry=close_price,
        stop=max(close_price * 0.97, boll["lower"]),
        account_size=account_size,
    )
    long_pos["direction"] = "做多"
    long_pos["stop_type"] = "BOLL下轨" if boll["lower"] > close_price * 0.97 else "固定3%"
    result["position_size_long"] = long_pos

    short_pos = calc_position_size(
        entry=close_price,
        stop=min(close_price * 1.03, boll["upper"]),
        account_size=account_size,
    )
    short_pos["direction"] = "做空"
    short_pos["stop_type"] = "BOLL上轨" if boll["upper"] < close_price * 1.03 else "固定3%"
    result["position_size_short"] = short_pos

    # 7. 两融
    try:
        margin = fetch_margin_individual(sym)
        if margin:
            result["margin"] = margin
    except Exception as e:
        logging.getLogger(__name__).error(f"[intraday_decision] 操作失败: {e}", exc_info=True)

    # 8. 市场背景
    if _ctx:
        try:
            result["market_context"] = _ctx()
        except Exception as e:
            logging.getLogger(__name__).error(f"[intraday_decision] 操作失败: {e}", exc_info=True)

    return result


# ═══════════════════════════════════════════════════════════════════
# 市场扫描
# ═══════════════════════════════════════════════════════════════════

def scan_market() -> dict[str, Any]:
    """当天市场全景扫描 + 大信号。"""
    result = {
        "ok": True,
        "time": now_cst().isoformat(),
        "trading_hours": is_trading_hours(),
    }

    # 市场背景
    result["market"] = assess_market_triple_screen()

    # 两融摘要
    try:
        margin = fetch_margin_summary()
        if margin and margin.get("ok"):
            result["margin_total"] = margin.get("total", 0)
            result["margin_change"] = margin.get("change", 0)
    except Exception as e:
        logging.getLogger(__name__).error(f"[intraday_decision] 操作失败: {e}", exc_info=True)

    # 港股行情（快速）
    try:
        # 限3秒超时，只取前5个
        top_global = list({**HK_KNOWN, **US_KNOWN}.keys())[:5]
        hk_quotes = fetch_global_quotes(top_global)
        hk_list = []
        for sym, q in (hk_quotes or {}).items():
            if q and hasattr(q, 'price') and q.price:
                hk_list.append({
                    "name": q.name or sym,
                    "price": q.price,
                    "change_pct": q.change_pct,
                })
        result["global_quotes"] = hk_list
    except Exception as e:
        logging.getLogger(__name__).error(f"[intraday_decision] 操作失败: {e}", exc_info=True)

    return result


# ═══════════════════════════════════════════════════════════════════
# 格式化输出（给助手自己用）
# ═══════════════════════════════════════════════════════════════════

def format_decision(decision: dict[str, Any]) -> str:
    """将决策结果格式化为可读文本（供我回答用户时使用）。"""
    if not decision.get("ok"):
        return f"❌ 错误: {decision.get('error', '未知')}"

    lines = []
    sym = decision["symbol"]
    name = decision["name"]

    # 标题
    now_str = decision.get("time", "")[11:19]
    th = "🟢 交易时段" if decision.get("trading_hours") else "🔴 盘后"
    lines.append(f"📊 *{name} ({sym})* — {now_str} {th}")

    # 市场背景（如果可用）
    ctx = decision.get("market_context", {})
    ctx_summary = ctx.get("_summary", "") if ctx else ""
    if ctx_summary:
        lines.append(f"\n*市场背景*")
        for cl in ctx_summary.split("\n"):
            lines.append(f"  {cl}")

    # 实时行情
    rt = decision.get("realtime")
    if rt:
        arrow = "📈" if rt["change_pct"] >= 0 else "📉"
        lines.append(f"\n*实时行情*")
        lines.append(f"  最新: {rt['price']:.2f} {arrow} {rt['change_pct']:+.2f}%")
        lines.append(f"  昨收: {rt['prev_close']:.2f}  开盘: {rt['open']:.2f}")
        lines.append(f"  最高: {rt['high']:.2f}  最低: {rt['low']:.2f}")
        lines.append(f"  量: {rt.get('volume', 0):.0f}  额: {rt.get('amount', 0) / 1e8:.2f}亿")

    # 日线技术指标
    daily = decision.get("daily", {})
    ind = daily.get("indicators", {})
    trend = daily.get("trend", {})
    boll = daily.get("bollinger", {})
    sig = daily.get("signal", {})

    lines.append(f"\n*技术指标*")
    lines.append(f"  MA: 20={trend.get('ma20', 0):.1f}  60={trend.get('ma60', 0):.1f}  144={trend.get('ma144', 0):.1f}  300={trend.get('ma300', 0):.1f}")
    lines.append(f"  BOLL: 上={boll.get('upper', 0):.1f} 中={boll.get('mid', 0):.1f} 下={boll.get('lower', 0):.1f}")
    lines.append(f"  RSI(14)={ind.get('rsi_14', 0):.1f}  MACD柱={ind.get('macd_hist', 0):.3f}  KDJ_J={ind.get('kdj_j', 0):.1f}")
    lines.append(f"  量比={ind.get('volume_ratio', 0):.2f}  ATR={ind.get('atr_pct', 0):.2f}%")
    lines.append(f"  综合分={ind.get('composite_score', 0):.1f}  信号={sig.get('action', 'N/A')}")

    # 三重滤网
    ts = decision.get("triple_screen", {})
    lines.append(f"\n*三重滤网评估*")
    s1 = ts.get("screen_1", "?")
    s1_emoji = "⬆️" if "up" in s1 else "⬇️" if "down" in s1 else "➡️"
    lines.append(f"  第一重(周线趋势): {s1_emoji} {s1}")

    s2 = ts.get("screen_2", {})
    osc = s2.get("oscillator", "?")
    osc_emoji = "🔥" if osc == "overbought" else "🧊" if osc == "oversold" else "✅"
    lines.append(f"  第二重(日线振荡): {osc_emoji} {osc} RSI={s2.get('rsi_14', 0)} "
                 f"背离={s2.get('divergence', '无')}")

    s3 = ts.get("screen_3", {})
    lines.append(f"  第三重(点位): BOLL位置={s3.get('boll_pos_pct', 0):.0f}%({s3.get('boll_zone', '?')}) "
                 f"ATR={s3.get('atr_pct', 0):.2f}%({s3.get('atr_zone', '?')})")

    cs = ts.get("combined_signal", "")
    lines.append(f"  📌 综合: {cs}")

    # 小时级 CCI + 均线
    h_signals = decision.get("hourly_signals", "")
    h_cci = decision.get("hourly_cci", {})
    h_ma = decision.get("hourly_ma_signal", {})
    h_trend = decision.get("hourly_trend", "")
    if h_signals:
        cci_val = h_cci.get("value", "?")
        cci_zone = h_cci.get("zone", "")
        ema5 = h_ma.get("ema_5", "")
        ema21 = h_ma.get("ema_21", "")
        cross = h_ma.get("cross", "")
        lines.append(f"\n*小时级(60min)*")
        lines.append(f"  CCI={cci_val} ({cci_zone}) | EMA5={ema5} EMA21={ema21} | {cross}")
        lines.append(f"  近5小时: {h_trend} | 信号: {h_signals}")

    # 适用策略（取Top 3）
    strategies = decision.get("applicable_strategies", [])
    if strategies:
        lines.append(f"\n*推荐策略 (Top {min(3, len(strategies))})*")
        for s in strategies[:3]:
            lines.append(f"  📍 {s['name']} [{s['author']}] — 适合度:{s['suitability']}")
            lines.append(f"     触发: {s['entry_trigger']}")
            lines.append(f"     止损: {s['stop']}  目标: {s['target']}")

    # 仓位建议
    # P2-P2-Q10-L041-fix: 仓位显示不再硬编码 100 万, 使用 decision.account_size(缺省 100万)
    acct = decision.get("account_size", 1_000_000)
    lines.append(f"\n*仓位建议 (假设账户{acct/10000:.0f}万)*")
    pos_long = decision.get("position_size_long", {})
    if pos_long.get("ok"):
        lines.append(f"  做多: {pos_long['suggested_shares']}股(约{pos_long['total_value']/10000:.1f}万, {pos_long['position_pct']:.0f}%仓)")
        lines.append(f"      止损{pos_long['stop']:.2f}(跌{pos_long['risk_pct_of_price']:.1f}%, 亏{pos_long['max_loss']:.0f}元)")
        # P2-P2-Q10-L042-fix: 明示交易费用对净亏损的影响
        if pos_long.get("estimated_costs") is not None:
            lines.append(f"      含费净亏≈{pos_long.get('max_loss_net', pos_long['max_loss']):.0f}元(费用≈{pos_long['estimated_costs']:.0f}元)")
    pos_short = decision.get("position_size_short", {})
    if pos_short.get("ok"):
        lines.append(f"  做空: {pos_short['suggested_shares']}股(约{pos_short['total_value']/10000:.1f}万)")

    # 两融
    margin = decision.get("margin")
    if margin:
        rzye = margin.get("rzye")
        rqye = margin.get("rqye")
        if rzye is not None or rqye is not None:
            lines.append(f"\n*两融*")
            if rzye:
                lines.append(f"  融资余额: {rzye:.2f}亿")
            if rqye:
                lines.append(f"  融券余额: {rqye:.2f}亿")

    # 底部总结
    sec = decision.get("seconds_to_close", 0)
    if sec > 0:
        lines.append(f"\n⏱ 距收盘还有 {sec // 60} 分钟")

    return "\n".join(lines)


__all__ = [
    "decide_stock",
    "scan_market",
    "assess_market_triple_screen",
    "calc_position_size",
    "format_decision",
    "is_trading_hours",
]
