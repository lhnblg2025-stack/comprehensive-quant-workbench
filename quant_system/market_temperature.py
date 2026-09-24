"""
量化交易系统 — 市场温度计 (基于 Gerald Appel 市场周期分段法)。

综合指数位置/涨跌比/成交量/融资余额/波动率 →
输出当前市场状态和交易策略基调。

用法:
  python3 -m quant_system.market_temperature          # 完整报告
  python3 -m quant_system.market_temperature --brief  # 简要版
"""

from __future__ import annotations

import functools
import json
import logging

import sys
import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

CST = timezone(timedelta(hours=8))
SINA_HEADERS = {"Referer": "https://finance.sina.com.cn", "User-Agent": "Mozilla/5.0"}
# P2-Q27-fix(L343): 移除未使用的 TENCENT_HEADERS（死代码，本模块仅用新浪接口）


# ════════════════════════════════════════════════════════════════
# 1. 基础数据获取
# ════════════════════════════════════════════════════════════════

def _fetch_sina_index(code: str) -> dict:
    """Fetch a single A-share index from Sina (s_ format for simple fields)."""
    url = f"http://hq.sinajs.cn/list=s_{code}"
    r = requests.get(url, headers=SINA_HEADERS, timeout=6)
    text = r.text.strip().split('"')[1] if '"' in r.text else ""
    parts = text.split(",")
    if len(parts) >= 6:
        try:
            return {
                "name": parts[0],
                "index": float(parts[1]),
                "change": float(parts[2]),
                "change_pct": float(parts[3]),
                "volume_wan": float(parts[4]),
                "amount_yi": float(parts[5]) / 10000,
            }
        except (ValueError, IndexError):
            pass
    return {}


def _fetch_index_daily(code: str = "sh000001", days: int = 400):
    """获取指数日线数据（akshare 新浪指数通道，与 market_regime._get_index_bars 一致）。

    P1-Q27-fix: 原实现 fetch_daily("000001") 会把 000001 解析为 sz000001=平安银行(个股)，
    导致 MA300/MA144/MA60/MA20/roc_60 全部基于个股价格，方向性判定错误。
    """
    try:
        import akshare as ak
        df = ak.stock_zh_index_daily(symbol=code)
        if df is None or df.empty:
            return None
        return df.tail(days)
    except Exception:
        return None


_AMOUNT_HIST_CACHE: dict = {}
_AMOUNT_HIST_TTL = 3600  # 历史成交额缓存 1 小时


def _fetch_index_amount_history(code: str, days: int = 60) -> list[float]:
    """获取指数历史成交额（亿元），失败返回 []。

    P2-Q27-fix(M341): 用于动态计算最近 20/60 日成交额均值，替代硬编码 avg_volume=15000。
    优先东财接口（含 amount 字段），带模块级缓存避免高频调用打网络。
    """
    cache_key = f"{code}:{days}"
    hit = _AMOUNT_HIST_CACHE.get(cache_key)
    if hit and _time.time() - hit[0] < _AMOUNT_HIST_TTL:
        return hit[1]
    try:
        import akshare as ak
        symbol = f"sh{code}" if not str(code).startswith("399") else f"sz{code}"
        df = ak.stock_zh_index_daily_em(symbol=symbol)
        if df is None or df.empty or "amount" not in df.columns:
            return []
        amounts = pd.to_numeric(df["amount"], errors="coerce").dropna().tail(days)
        out = [round(a / 1e8, 2) for a in amounts.tolist()]  # 元 → 亿元
        _AMOUNT_HIST_CACHE[cache_key] = (_time.time(), out)
        return out
    except Exception:
        return []


def _load_real_breadth() -> dict | None:
    """读取最近一日的全市场真实宽度 (generated/a_share_data/*-meta.json)。

    P1-Q27-fix: 优先用真实涨跌家数替代 market_context 的估算值。
    """
    try:
        meta_dir = Path(__file__).resolve().parent.parent / "generated" / "a_share_data"
        files = sorted(meta_dir.glob("*-meta.json"))
        if not files:
            return None
        with files[-1].open(encoding="utf-8") as fh:
            meta = json.load(fh)
        bread = meta.get("breadth") or {}
        if any(k in bread for k in ("股票数", "上涨", "下跌", "上涨占比%")):
            return bread
        return None
    except Exception:
        return None


def fetch_market_data() -> dict[str, Any]:
    """Get all data needed for temperature calculation."""
    result = {"timestamp": datetime.now(CST).strftime("%Y-%m-%d %H:%M")}
    
    # 1. Major indices (P2-Q27-fix L342: 补充北交所 bj899050 北证50)
    for code, key in [("sh000001", "sh"), ("sz399001", "sz"), ("sz399006", "cy"), ("bj899050", "bj")]:
        d = _fetch_sina_index(code)
        if d:
            result[f"{key}_index"] = d["index"]
            result[f"{key}_pct"] = d["change_pct"]
            result[f"{key}_amount_yi"] = d["amount_yi"]

    # 2. Total volume (沪+深+北；北交所获取失败时降级并标注口径)
    sh_amt = result.get("sh_amount_yi", 0)
    sz_amt = result.get("sz_amount_yi", 0)
    bj_amt = result.get("bj_amount_yi", 0)
    result["total_amount_yi"] = sh_amt + sz_amt + bj_amt
    # P2-Q27-fix(L342): 注明成交额口径（沪深北 / 沪深）
    result["amount_scope"] = "沪深北" if bj_amt else "沪深(北交所未获取)"

    # P2-Q27-fix(M341): 动态计算最近 20/60 日成交额均值（与 total_amount_yi 同口径），
    # 替代硬编码 avg_volume=15000 亿；获取失败时由 classify_cycle 回退并可见标注。
    try:
        sh_hist = _fetch_index_amount_history("000001", days=60)
        sz_hist = _fetch_index_amount_history("399001", days=60)
        if sh_hist and sz_hist:
            n = min(len(sh_hist), len(sz_hist))
            sums = [sh_hist[i] + sz_hist[i] for i in range(n)]
            result["avg_amount_yi_20"] = sum(sums[-20:]) / min(20, len(sums))
            result["avg_amount_yi_60"] = sum(sums[-60:]) / min(60, len(sums))
    except Exception as e:
        logging.getLogger(__name__).error(f"[market_temperature] 操作失败: {e}", exc_info=True)
    
    # 3. Advance/decline — 优先真实全市场宽度(meta)，否则用 market_context 估算并标注
    result["advance_pct_estimate"] = True  # 默认标记估算
    try:
        real = _load_real_breadth()
        if real:
            result["advance"] = int(real.get("上涨", 0))
            result["decline"] = int(real.get("下跌", 0))
            result["advance_pct"] = round(float(real.get("上涨占比%", 50)), 1)
            result["advance_pct_estimate"] = False  # P1-Q27-fix: 真实宽度
        else:
            from quant_system.market_context import get_market_context
            ctx = get_market_context()
            ad = ctx.get("advance_decline", {})
            result["advance"] = ad.get("advance", 0)
            result["decline"] = ad.get("decline", 0)
            result["advance_pct"] = round(ad.get("advance", 0) / max(ad.get("total", 1), 1) * 100, 1)
            result["advance_pct_estimate"] = True  # P1-Q27-fix: 估算宽度
    except Exception as e:
        logging.getLogger(__name__).error(f"[market_temperature] 操作失败: {e}", exc_info=True)

    # 4. 300MA position for sh index (long-term trend)
    # P1-Q27-fix: 用 sh000001 上证指数日线，不再经 fetch_daily 解析成平安银行
    try:
        df_sh = _fetch_index_daily("sh000001", days=400)
        if df_sh is not None and len(df_sh) > 300:
            closes = df_sh["close"].values.astype(float)
            ma300 = sum(closes[-300:]) / 300
            ma144 = sum(closes[-144:]) / 144
            ma60 = sum(closes[-60:]) / 60
            ma20 = sum(closes[-20:]) / 20
            current = closes[-1]
            result["ma20"] = round(ma20, 0)
            result["ma60"] = round(ma60, 0)
            result["ma144"] = round(ma144, 0)
            result["ma300"] = round(ma300, 0)
            result["pct_above_ma300"] = round((current / ma300 - 1) * 100, 1)
            # Rate of change (60-day)
            result["roc_60"] = round((closes[-1] / closes[-61] - 1) * 100, 1) if len(closes) > 61 else 0
            # MACD on monthly equivalent (60-day MA direction)
            result["ma60_direction"] = "up" if ma60 > sum(closes[-120:-60])/60 else "down"
    except Exception as e:
        logging.getLogger(__name__).error(f"[market_temperature] 操作失败: {e}", exc_info=True)
    
    # 5. 融资余额 from existing margin module
    try:
        from quant_system.margin import fetch_margin_summary
        margin = fetch_margin_summary()
        if margin and "total_margin_balance" in margin:
            result["margin_yi"] = margin["total_margin_balance"]
            result["margin_change_yi"] = margin.get("total_margin_change", 0)
    except Exception as e:
        logging.getLogger(__name__).error(f"[market_temperature] 操作失败: {e}", exc_info=True)
    
    # 6. North-bound flow
    try:
        from quant_system.north_flow import fetch_north_summary
        north = fetch_north_summary()
        if north and "total_net_yi" in north:
            result["north_net_yi"] = north["total_net_yi"]
    except Exception as e:
        logging.getLogger(__name__).error(f"[market_temperature] 操作失败: {e}", exc_info=True)
    
    # 7. Bond yield spread (can't get easily, skip for now)
    
    return result


# ════════════════════════════════════════════════════════════════
# 2. Appel Cycle Classification
# ════════════════════════════════════════════════════════════════

# Market temperature score 0-100
_TEMPERATURE_LABELS = [
    (0, "❄️ 冰点 — 恐慌底部区"),
    (20, "🌧️ 寒冷 — 超卖修复区"),
    (40, "🌥️ 温和 — 正常波动区"),
    (60, "☀️ 温暖 — 上涨趋势中"),
    (80, "🔥 过热 — 高风险区"),
]


def classify_cycle(data: dict) -> dict[str, Any]:
    """
    Appel market cycle classification.
    
    Uses: primary trend (MA300/MA144 position), breadth, momentum, volume
    
    Returns: cycle_segment, risk_posture, temperature, signals
    """
    result = {"timestamp": data.get("timestamp", "")}
    
    # === Step 1: Identify primary trend ===
    pct_above_ma300 = data.get("pct_above_ma300", 0)
    sh_pct = data.get("sh_pct", 0)
    ma60_direction = data.get("ma60_direction", "up")
    
    if pct_above_ma300 > 10:
        primary_trend = "strong_up"
    elif pct_above_ma300 > 0:
        primary_trend = "moderate_up"
    elif pct_above_ma300 > -10:
        primary_trend = "sideways"
    else:
        primary_trend = "down"
    
    result["primary_trend"] = primary_trend
    
    # === Step 2: Measure breadth ===
    adv_pct = data.get("advance_pct", 50)
    # P1-Q27-fix: 估算宽度(advance_pct_estimate=True)是合成数据，
    # 不触发极端广度判定，避免让估算值主导周期分段。
    breadth_estimate = bool(data.get("advance_pct_estimate", False))
    if breadth_estimate:
        breadth_strong = False
        breadth_weak = False
        breadth_crash = False
    else:
        breadth_strong = adv_pct > 60
        breadth_weak = adv_pct < 30
        breadth_crash = adv_pct < 15
    
    # === Step 3: Measure momentum ===
    roc_60 = data.get("roc_60", 0)
    volume_yi = data.get("total_amount_yi", 0)
    
    # Volume surge (vs historical average ~1.5万亿 for normal)
    # P2-Q27-fix(M341): 原硬编码 15000 亿在 2026 年两市常态 2 万亿+ 下导致
    # vol_ratio 系统性偏低、"后期狂热/派发"分支难触发。改用最近 20/60 日实际
    # 成交额均值动态计算；历史数据不可用时回退硬编码并在结果中标注。
    avg_volume = data.get("avg_amount_yi_60") or data.get("avg_amount_yi_20")
    if avg_volume is None or avg_volume <= 0:
        avg_volume = 15000.0  # 回退值（仅当历史成交额不可用）
        result["volume_baseline"] = "hardcoded_15000"
    else:
        result["volume_baseline"] = "dynamic_20_60d"
    result["avg_volume_yi"] = round(float(avg_volume), 0)
    vol_ratio = volume_yi / avg_volume if avg_volume > 0 else 1
    
    # === Step 4: Classify cycle segment ===
    if breadth_crash and sh_pct < -5:
        cycle_segment = "恐慌/衰竭(Panic)"
        temperature = 5
    elif breadth_weak and primary_trend in ("down", "sideways") and roc_60 < -5:
        cycle_segment = "熊市下跌(Bear Decline)"
        temperature = 15
    elif breadth_weak and primary_trend == "down":
        cycle_segment = "底部积累(Bottoming)"
        temperature = 25
    elif (primary_trend == "strong_up" and breadth_strong and vol_ratio > 1.3):
        cycle_segment = "后期狂热/派发(Late-cycle)"
        temperature = 85
    elif (primary_trend in ("strong_up", "moderate_up") and breadth_strong):
        cycle_segment = "成熟牛市(Mature Bull)"
        temperature = 65
    elif primary_trend == "moderate_up" and not breadth_weak:
        cycle_segment = "早期牛市(Early Bull)"
        temperature = 50
    elif primary_trend == "sideways" and not breadth_weak:
        cycle_segment = "早期牛市(Early Bull)"
        temperature = 45
    elif primary_trend in ("sideways", "down") and breadth_weak:
        cycle_segment = "底部积累(Bottoming)"
        temperature = 25
    else:
        cycle_segment = "温和正常"
        temperature = 50
    
    # === Step 5: Set risk posture ===
    if "恐慌" in cycle_segment:
        risk_posture = "分批回场(Staged Re-entry)"
    elif "底部" in cycle_segment:
        risk_posture = "选择性做多(Selective Long)"
    elif "早期" in cycle_segment:
        risk_posture = "积极做多(Aggressive Long)"
    elif "成熟" in cycle_segment:
        risk_posture = "选择性做多(Selective Long)"
    elif "后期" in cycle_segment or "狂热" in cycle_segment:
        risk_posture = "防御/现金(Defensive/Cash)"
    elif "熊市" in cycle_segment:
        risk_posture = "防御/现金(Defensive/Cash)"
    else:
        risk_posture = "中性/对冲(Neutral)"
    
    # Adjust posture based on specific signals
    margin_change = data.get("margin_change_yi", 0)
    if margin_change < -200:
        risk_posture = "防御/现金(Defensive/Cash)"  # margin dropping = risk off
    
    north = data.get("north_net_yi", 0)
    if north > 50 and "防御" in risk_posture:
        risk_posture = "中性/对冲(Neutral)"  # north buying offsets caution
    
    result["cycle_segment"] = cycle_segment
    result["risk_posture"] = risk_posture
    result["temperature"] = min(100, max(0, temperature))
    result["temperature_label"] = next(
        (lbl for threshold, lbl in reversed(_TEMPERATURE_LABELS) if temperature >= threshold),
        "❄️ 冰点"
    )
    
    # === Step 6: Trading strategy recommendations ===
    result["strategy"] = _get_strategy(cycle_segment, risk_posture, temperature)
    
    return result


_POSITION_MAP_PATHS = (
    Path(__file__).resolve().parents[1] / "config" / "position_map.json",
    Path(__file__).resolve().parent / "config" / "position_map.json",
)


def _cycle_phase(cycle: str) -> str:
    """将 cycle_segment 归一为 calibration 使用的 7 个阶段之一。"""
    if "恐慌" in cycle:
        return "恐慌"
    if "底部" in cycle:
        return "底部"
    if "早期" in cycle:
        return "早期"
    if "成熟" in cycle:
        return "成熟"
    if "后期" in cycle or "狂热" in cycle:
        return "后期"
    if "熊市" in cycle:
        return "熊市"
    return "其他"


@functools.lru_cache(maxsize=1)
def _load_position_phases() -> dict:
    """读取 position_map.json 的 phases 映射，失败返回空 dict。"""
    for path in _POSITION_MAP_PATHS:
        try:
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as fh:
                payload = json.load(fh)
            phases = payload.get("phases", {}) if isinstance(payload, dict) else {}
            if isinstance(phases, dict):
                return phases
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "[market_temperature] 解析 position_map 失败: %s (%s)", path, exc
            )
            continue
    return {}


def _coefficient_to_position(coefficient: float) -> str:
    """把仓位系数转成区间字符串（上界为系数*100，下界按 -20% 带宽取整）。

    统一带宽规则：upper = round(c*100)，lower = max(0, upper-20)。
    c==0 单独返回 "0-0%"，避免 -20% 带宽出现负数下界；其余系数均走同一规则，
    保证映射随系数单调。
    """
    c = max(0.0, min(1.0, float(coefficient)))
    if c <= 0.0:
        return "0-0%"
    upper = round(c * 100)
    lower = max(0, upper - 20)
    return f"{lower}-{upper}%"


def _get_strategy(cycle: str, posture: str, temp: int) -> dict:
    """Get trading strategy recommendations based on cycle phase.

    优先读取 config/position_map.json 的校准仓位；缺配置/缺键时回退原手拍档位。
    """
    strategy = _get_strategy_default(cycle, posture, temp)

    phases = _load_position_phases()
    phase = _cycle_phase(cycle)
    try:
        coefficient = phases.get(phase)
        if coefficient is not None:
            strategy["position"] = _coefficient_to_position(float(coefficient))
            strategy["position_source"] = "calibrated"
            return strategy
    except (TypeError, ValueError):
        pass

    strategy["position_source"] = "default"
    return strategy


def _get_strategy_default(cycle: str, posture: str, temp: int) -> dict:
    """原手拍档位，作为无 position_map.json 时的兜底。"""
    
    if "恐慌" in cycle:
        return {
            "action": "🟢 分批建仓",
            "position": "30-50%",
            "focus": "超跌大盘蓝筹+宽基ETF",
            "stop": "跌破前低止损",
            "note": "恐慌=机会,但不要一把梭,分3批建仓"
        }
    elif "底部" in cycle:
        return {
            "action": "🟢 试探性建仓",
            "position": "20-30%",
            "focus": "价值股+高股息+被错杀的优质股",
            "stop": "买入价的-5%",
            "note": "底部区域需要耐心,做好反复震荡的准备"
        }
    elif "早期" in cycle:
        return {
            "action": "🟢 积极做多",
            "position": "60-80%",
            "focus": "成长股+周期股+券商",
            "stop": "MA60下方",
            "note": "早期牛市最好的策略是持有不动"
        }
    elif "成熟" in cycle:
        return {
            "action": "🟡 选择性做多",
            "position": "40-60%",
            "focus": "龙头股+从成长切换至价值",
            "stop": "动态止盈",
            "note": "涨多了要舍得卖,追高要谨慎"
        }
    elif "后期" in cycle:
        return {
            "action": "🔴 减仓防御",
            "position": "20-30%",
            "focus": "银行+公用事业+黄金+现金",
            "stop": "跌破MA144减仓半",
            "note": "最后一波往往最疯狂,也是最容易亏钱的时候"
        }
    elif "熊市" in cycle:
        return {
            "action": "🔴 空仓/极轻仓",
            "position": "0-15%",
            "focus": "现金+货基+债券ETF",
            "stop": "站上MA60前不进场",
            "note": "不要接飞刀,等右侧信号出现"
        }
    else:
        return {
            "action": "⚪ 观望",
            "position": "20-40%",
            "focus": "低估值+高股息",
            "note": "方向不明,减少操作频率"
        }


# ════════════════════════════════════════════════════════════════
# 3. 格式化输出 & CLI
# ════════════════════════════════════════════════════════════════

def format_report(data: dict = None, brief: bool = False) -> str:
    """Full market temperature report."""
    if data is None:
        data = fetch_market_data()
    
    cycle = classify_cycle(data)
    lines = []
    
    # Header
    now = data.get("timestamp", datetime.now(CST).strftime("%H:%M"))
    lines.append(f"🌡️ **市场温度计** — {now}")
    lines.append(f"{'='*45}")
    lines.append(f"")
    
    # Temperature gauge
    temp = cycle["temperature"]
    gauge = "█" * (temp // 5) + "░" * (20 - temp // 5)
    lines.append(f"[{gauge}] {temp}/100")
    lines.append(f"📊 周期阶段: {cycle['cycle_segment']}")
    lines.append(f"🎯 交易姿态: {cycle['risk_posture']}")
    lines.append(f"")
    
    if brief:
        strat = cycle.get("strategy", {})
        lines.append(f"💡 建议: {strat.get('action', '')}")
        lines.append(f"   仓位: {strat.get('position', '')}")
        lines.append(f"   关注: {strat.get('focus', '')}")
        return "\n".join(lines)
    
    # Detailed data
    lines.append(f"**📈 核心数据**")
    if "sh_index" in data:
        lines.append(f"  上证: {data['sh_index']:.0f} ({data.get('sh_pct',0):+.2f}%)")
    if "cy_pct" in data:
        lines.append(f"  创业板: {data.get('cy_pct',0):+.2f}%")
    if "total_amount_yi" in data:
        lines.append(f"  成交额: {data['total_amount_yi']:.0f}亿")
    if "advance_pct" in data:
        # P1-Q27-fix: 估算宽度显式标注，避免被当作真实数据
        est_tag = " (估算)" if data.get("advance_pct_estimate", False) else ""
        lines.append(f"  上涨占比: {data['advance_pct']}%{est_tag}")
    if "pct_above_ma300" in data:
        lines.append(f"  距MA300: {data['pct_above_ma300']:+.1f}%")
    if "roc_60" in data:
        lines.append(f"  60日涨幅: {data['roc_60']:+.1f}%")
    if "margin_yi" in data:
        lines.append(f"  融资余额: {data['margin_yi']:.0f}亿 ({data.get('margin_change_yi',0):+.0f}亿)")
    if "north_net_yi" in data:
        lines.append(f"  北向净买入: {data['north_net_yi']:+.2f}亿")
    lines.append(f"")
    
    # Cycle evidence
    lines.append(f"**🔍 周期判定依据**")
    lines.append(f"  主趋势: {cycle['primary_trend']}")
    lines.append(f"  MA60方向: {data.get('ma60_direction','?')}")
    lines.append(f"  温度: {cycle['temperature_label']}")
    lines.append(f"")
    
    # Strategy
    strat = cycle.get("strategy", {})
    lines.append(f"**💡 交易策略**")
    lines.append(f"  操作: {strat.get('action', '')}")
    lines.append(f"  仓位: {strat.get('position', '')}")
    lines.append(f"  关注: {strat.get('focus', '')}")
    if "stop" in strat:
        lines.append(f"  风控: {strat['stop']}")
    if "note" in strat:
        lines.append(f"  📝 {strat['note']}")
    lines.append(f"")
    
    # Transition triggers to watch
    lines.append(f"**⚠️ 关注信号**")
    if "后期" in cycle.get("cycle_segment", "") or temp >= 80:
        lines.append(f"  1. 涨跌比连续走弱 (顶背离)")
        lines.append(f"  2. 成交量萎缩 (动能衰竭)")
        lines.append(f"  3. 融资余额大幅下降 (杠杆撤退)")
    elif "底部" in cycle.get("cycle_segment", "") or temp <= 30:
        lines.append(f"  1. 指数站上MA20 (第一入场信号)")
        lines.append(f"  2. 涨跌比回升到50%以上 (广度改善)")
        lines.append(f"  3. 北向资金连续净买入 (聪明钱进场)")
    elif "早期" in cycle.get("cycle_segment", ""):
        lines.append(f"  1. MA20上穿MA60 (金叉确认)")
        lines.append(f"  2. 成交量温和放大 (资金入场)")
        lines.append(f"  3. 板块轮动健康 (非一枝独秀)")
    
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="市场温度计")
    parser.add_argument("--brief", action="store_true", help="简要版")
    args = parser.parse_args()
    
    data = fetch_market_data()
    print(format_report(data, brief=args.brief))
