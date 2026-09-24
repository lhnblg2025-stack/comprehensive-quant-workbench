"""
量化交易系统 — 市场状态识别 (Market Regime Detection)

识别当前市场所处的状态：趋势 / 波动 / 流动性 / 情绪。

用法:
  from quant_system.market_regime import get_current_regime
  regime = get_current_regime()
  print(regime_to_text(regime))
"""

from __future__ import annotations
import logging

import math
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from quant_system.utils import to_float as _to_float

ROOT = Path(__file__).resolve().parent
CST = timezone(timedelta(hours=8))
SINA_HEADERS = {"Referer": "https://finance.sina.com.cn", "User-Agent": "Mozilla/5.0"}
TENCENT_HEADERS = {"Referer": "https://gu.qq.com", "User-Agent": "Mozilla/5.0"}


# ════════════════════════════════════════════════════════════════
#  内部辅助
# ════════════════════════════════════════════════════════════════

def _get_index_bars(code: str = "000001", days: int = 300) -> list[list]:
    """获取指数 K 线数据.

    Args:
        code: 指数代码, 默认 000001 (上证指数).
        days: 需要的交易日数.

    Returns:
        [[date, open, high, low, close, volume], ...]
    """
    try:
        import akshare as ak
        df = ak.stock_zh_index_daily(symbol=f"sh{code}")
        if df is None or df.empty:
            return []
        df.columns = [str(c).strip() for c in df.columns]
        date_col = "date" if "date" in df.columns else df.columns[0]
        bars = []
        for _, row in df.tail(days).iterrows():
            bars.append([
                str(row[date_col])[:10],
                _to_float(row.get("open", 0)),
                _to_float(row.get("high", 0)),
                _to_float(row.get("low", 0)),
                _to_float(row.get("close", 0)),
                _to_float(row.get("volume", 0)),
            ])
        return bars
    except Exception as e:
        logging.getLogger(__name__).error(f"[market_regime] 操作失败: {e}", exc_info=True)

    # Fallback: 从新浪获取
    try:
        # V11 审计修复（Medium）: 原实现硬编码 sh000001，忽略 code 参数
        # （传 399001 等仍取上证）；且 scale=60 是 60 分钟线，用于日线分析错误。
        # 修正: 动态拼 symbol + 用日线 scale=240。
        _sym = f"sh{code}" if str(code).startswith(("0", "1", "9")) else f"sz{code}"
        url = (f"http://money.finance.sina.com.cn/quotes_service/api/jsonp_v2.php/"
               f"var%20_{_sym}_20=/CN_MarketData.getKLineData?"
               f"symbol={_sym}&datalen={days}&scale=240&ma=no")
        r = requests.get(url, headers=SINA_HEADERS, timeout=6)
        import json as _json
        text = r.text.strip()
        if "(" in text and text.endswith(")"):
            text = text[text.index("(") + 1 : -1]
        data = _json.loads(text)
        bars = []
        for item in data:
            bars.append([
                str(item.get("day", ""))[:10],
                _to_float(item.get("open", 0)),
                _to_float(item.get("high", 0)),
                _to_float(item.get("low", 0)),
                _to_float(item.get("close", 0)),
                _to_float(item.get("volume", 0)),
            ])
        return bars[-days:]
    except Exception:
        return []


def _get_index_amount_history(code: str = "000001", days: int = 20) -> list[float]:
    """获取最近 N 个交易日的指数成交额（亿元）。

    P1-Q27-fix: avg_20d_amount_yi 必须用历史成交额，不能拿 volume(手) 当亿元。
    优先东财接口（含 amount 字段），失败返回 [] 由调用方降级估算。
    """
    try:
        import akshare as ak
        end = datetime.now(CST).strftime("%Y%m%d")
        start = (datetime.now(CST) - timedelta(days=int(days * 1.6) + 30)).strftime("%Y%m%d")
        df = ak.stock_zh_index_daily_em(symbol=f"sh{code}", start_date=start, end_date=end)
        if df is None or df.empty or "amount" not in df.columns:
            return []
        amounts = pd.to_numeric(df["amount"], errors="coerce").dropna().tail(days)
        return [round(a / 1e8, 2) for a in amounts.tolist()]  # 元 → 亿元
    except Exception:
        return []


def _load_real_limit_counts() -> tuple[int, int] | None:
    """从最近一日的 meta breadth 读取近似涨停/近似跌停家数。

    P1-Q27-fix: market_context 无 limit_up/涨停 键，涨跌停家数需从真实全市场宽度读取；
    读取失败返回 None，由调用方标记情绪置信度低。
    """
    try:
        import json
        meta_dir = Path(__file__).resolve().parent.parent / "generated" / "a_share_data"
        files = sorted(meta_dir.glob("*-meta.json"))
        if not files:
            return None
        with files[-1].open(encoding="utf-8") as fh:
            meta = json.load(fh)
        bread = meta.get("breadth") or {}
        if not any(k in bread for k in ("近似涨停", "近似跌停")):
            return None
        zt = int(_to_float(bread.get("近似涨停", 0)))
        dt = int(_to_float(bread.get("近似跌停", 0)))
        return zt, dt
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════
#  1. 趋势状态识别
# ════════════════════════════════════════════════════════════════

def detect_regime(index_bars: list[list]) -> dict[str, Any]:
    """识别指数趋势状态.

    判断方法:
      - 价格在 MA144 上方 = bull, 下方 = bear, 附近 = sideways
      - MA20 > MA60 > MA144 = 多头排列
      - 从低位反弹 >10% = recovery
      - 从高位下跌 >10% = correction

    Args:
        index_bars: K 线列表 [[date, open, high, low, close, volume], ...].

    Returns:
        {trend, score(1-5), description, ma20, ma60, ma144, ...}
    """
    if not index_bars or len(index_bars) < 144:
        return {"trend": "unknown", "score": 3, "description": "数据不足", "ma20": 0, "ma60": 0, "ma144": 0}

    closes = [float(b[4]) for b in index_bars]

    def ma(period: int) -> float:
        if len(closes) < period:
            return closes[-1]
        return sum(closes[-period:]) / period

    ma20 = ma(20)
    ma60 = ma(60)
    ma144 = ma(144)
    current = closes[-1]

    # 确认类型
    above_ma144 = current > ma144
    below_ma144 = current < ma144
    near_ma144 = abs(current / ma144 - 1) < 0.03  # 3% 范围内

    bullish_aligned = ma20 > ma60 > ma144
    bearish_aligned = ma20 < ma60 < ma144

    # 检查底部反弹 / 顶部下跌
    lowest_60 = min(closes[-60:])
    highest_60 = max(closes[-60:])
    bounce_pct = (current / lowest_60 - 1) * 100 if lowest_60 > 0 else 0
    drop_pct = (current / highest_60 - 1) * 100 if highest_60 > 0 else 0

    # P2-Q27-fix(M353): 原分支顺序中 current 在 MA144 ±3% 内时必先落入 above/below
    # 分支（> 或 < 其一恒成立），near_ma144 仅在 current == ma144 时可达 → "横盘"
    # 描述从不输出（平坦序列误判 bull）。现将 near_ma144 提前为最优先判定。
    if near_ma144:
        if bullish_aligned:
            trend = "bull"
            score = 2
            desc = "横盘偏多 — 价格在MA144附近且均线多头排列"
        elif bearish_aligned:
            trend = "bear"
            score = 4
            desc = "横盘偏空 — 价格在MA144附近且均线空头排列"
        else:
            trend = "sideways"
            score = 3
            desc = "横盘 — 价格在MA144附近震荡"
    elif above_ma144 and bullish_aligned:
        trend = "bull"
        score = 1  # 低风险
        desc = "牛市 — MA多头排列, 价格在MA144上方"
    elif below_ma144 and bearish_aligned:
        trend = "bear"
        score = 5  # 高风险
        desc = "熊市 — MA空头排列, 价格在MA144下方"
    elif above_ma144 and not bullish_aligned:
        if drop_pct <= -10:
            trend = "correction"
            score = 4
            desc = f"回调 — 价格在MA144上方但从高点下跌{drop_pct:.1f}%"
        else:
            trend = "bull"
            score = 2
            desc = "偏牛 — 价格在MA144上方但均线未完全多头"
    elif below_ma144 and not bearish_aligned:
        if bounce_pct >= 10:
            trend = "recovery"
            score = 3
            desc = f"修复 — 价格在MA144下方但从低点反弹{bounce_pct:.1f}%"
        else:
            trend = "bear"
            score = 4
            desc = "偏熊 — 价格在MA144下方但均线未完全空头"
    else:
        trend = "sideways"
        score = 3
        desc = "震荡 — 无明显趋势"

    return {
        "trend": trend,
        "score": score,
        "description": desc,
        "ma20": round(ma20, 2),
        "ma60": round(ma60, 2),
        "ma144": round(ma144, 2),
        "current": round(current, 2),
        "above_ma144": above_ma144,
        "bullish_aligned": bullish_aligned,
        "bearish_aligned": bearish_aligned,
        "bounce_pct": round(bounce_pct, 2),
        "drop_pct": round(drop_pct, 2),
    }


# ════════════════════════════════════════════════════════════════
#  2. 波动状态识别
# ════════════════════════════════════════════════════════════════

def detect_volatility(index_bars: list[list]) -> dict[str, Any]:
    """识别市场波动状态.

    计算: 20日年化波动率 vs 历史分位.

    Args:
        index_bars: K 线列表.

    Returns:
        {state, vol_20d, vol_60d, percentile}
        state: "low" / "normal" / "high" / "extreme"
    """
    if not index_bars or len(index_bars) < 21:
        return {"state": "unknown", "vol_20d": 0, "vol_60d": 0, "percentile": 0}

    closes = [float(b[4]) for b in index_bars]

    def _annualized_vol(prices: list[float], period: int) -> float:
        if len(prices) < period + 1:
            return 0.0
        rets = []
        for i in range(len(prices) - period, len(prices)):
            if prices[i - 1] > 0:
                rets.append(math.log(prices[i] / prices[i - 1]))
        if len(rets) < 2:
            return 0.0
        std = statistics.stdev(rets) if len(rets) > 1 else 0.0
        return std * math.sqrt(252) * 100  # 年化百分比

    vol_20d = _annualized_vol(closes, 20)
    vol_60d = _annualized_vol(closes, 60)

    # P2-Q27-fix(M352): 分位基准仅统计"近 300 交易日"（_get_index_bars 上限，
    # 约 1.2 年），且排除与当前 20 日窗口重叠的历史窗口（自包含偏差）。
    all_rets = []
    for i in range(1, len(closes)):
        if closes[i - 1] > 0:
            all_rets.append(math.log(closes[i] / closes[i - 1]))

    if len(all_rets) < 60:
        return {"state": "normal", "vol_20d": round(vol_20d, 2), "vol_60d": round(vol_60d, 2), "percentile": 50}

    # 滚动 20 日波动率（严格早于当前窗口，即排除末尾与当前窗口重叠/相等的窗口，
    # 否则当前 vol_20d 会与自身（或近乎相同的窗口）比较，percentile 偏高）
    rolling_vols = []
    for i in range(20, len(all_rets) - 19):
        seg = all_rets[i - 20 : i]
        std = statistics.stdev(seg) if len(seg) > 1 else 0.0
        rolling_vols.append(std * math.sqrt(252) * 100)

    # 当前波动率在历史中的百分位
    below = sum(1 for v in rolling_vols if v <= vol_20d)
    percentile = below / len(rolling_vols) * 100 if rolling_vols else 50

    if percentile >= 90:
        state = "extreme"
    elif percentile >= 70:
        state = "high"
    elif percentile >= 30:
        state = "normal"
    else:
        state = "low"

    return {
        "state": state,
        "vol_20d": round(vol_20d, 2),
        "vol_60d": round(vol_60d, 2),
        "percentile": round(percentile, 1),
    }


# ════════════════════════════════════════════════════════════════
#  3. 流动性状态识别
# ════════════════════════════════════════════════════════════════

def detect_liquidity(margin_data: dict | None = None, volume_data: dict | None = None) -> dict[str, Any]:
    """识别市场流动性状态.

    Args:
        margin_data: 两融数据, 含 total_margin_balance, total_margin_change.
        volume_data: 成交量数据, 含 amount_yi, avg_20d_amount_yi.

    Returns:
        {state: "abundant" / "normal" / "tight", detail}
    """
    result: dict[str, Any] = {"state": "normal", "detail": "默认 (数据不足)", "margin_ok": False, "volume_ok": False}

    score = 0  # 0=正常, +1=充裕, -1=紧缩

    # 两融判断
    if margin_data and margin_data.get("ok", False):
        balance = _to_float(margin_data.get("total_margin_balance", 0))
        change = _to_float(margin_data.get("total_margin_change", 0))
        result["margin_yi"] = balance
        if balance > 20000:  # >2万亿 充裕
            score += 1
            result["margin_detail"] = f"两融余额{balance:.0f}亿 (充裕)"
        elif balance > 15000:
            result["margin_detail"] = f"两融余额{balance:.0f}亿 (正常)"
        else:
            score -= 1
            result["margin_detail"] = f"两融余额{balance:.0f}亿 (偏低)"
        result["margin_ok"] = True

    # 成交量判断
    if volume_data:
        amount = _to_float(volume_data.get("amount_yi", 0))
        avg_20d = _to_float(volume_data.get("avg_20d_amount_yi", 0))
        result["volume_yi"] = amount
        if avg_20d > 0 and amount > avg_20d * 1.3:
            score += 1
            result["volume_detail"] = f"成交{amount:.0f}亿 (放量)"
        elif avg_20d > 0 and amount < avg_20d * 0.7:
            score -= 1
            result["volume_detail"] = f"成交{amount:.0f}亿 (缩量)"
        elif avg_20d > 0:
            result["volume_detail"] = f"成交{amount:.0f}亿 (正常)"
        else:
            result["volume_detail"] = f"成交{amount:.0f}亿"
        result["volume_ok"] = True

    if score > 0:
        result["state"] = "abundant"
    elif score < 0:
        result["state"] = "tight"
    else:
        result["state"] = "normal"

    result["detail"] = f"流动性: {result['state']} (score={score})"
    return result


# ════════════════════════════════════════════════════════════════
#  4. 情绪状态识别
# ════════════════════════════════════════════════════════════════

def detect_sentiment(
    advance_decline_ratio: float = 1.0,
   涨停_count: int = 0,
    跌停_count: int = 0,
    etf_flow: float = 0.0,
) -> dict[str, Any]:
    """识别市场情绪状态.

    Args:
        advance_decline_ratio: 涨跌比 (上涨家数 / 下跌家数).
        涨停_count: 涨停家数.
        跌停_count: 跌停家数.
        etf_flow: ETF 资金净流入 (亿元).

    Returns:
        {state: "greed" / "normal" / "fear", score, detail}
    """
    score = 0  # 0=正常, +1~+2=贪婪, -1~-2=恐惧

    # 涨跌比判断
    if advance_decline_ratio > 2.0:
        score += 1
        ad_note = f"AD比率 {advance_decline_ratio:.2f} (>2.0, 贪婪)"
    elif advance_decline_ratio < 0.5:
        score -= 1
        ad_note = f"AD比率 {advance_decline_ratio:.2f} (<0.5, 恐惧)"
    else:
        ad_note = f"AD比率 {advance_decline_ratio:.2f} (正常)"

    # 涨跌停判断
    if 涨停_count > 50 and 跌停_count < 5:
        score += 1
        zt_note = f"涨停{涨停_count} 跌停{跌停_count} (多头强势)"
    elif 跌停_count > 20:
        score -= 1
        zt_note = f"跌停{跌停_count} (恐慌)"
    else:
        zt_note = f"涨停{涨停_count} 跌停{跌停_count} (正常)"

    # ETF资金流判断
    if etf_flow > 50:
        score += 1
        etf_note = f"ETF净流入{etf_flow:.0f}亿 (积极)"
    elif etf_flow < -50:
        score -= 1
        etf_note = f"ETF净流出{abs(etf_flow):.0f}亿 (避险)"
    else:
        etf_note = f"ETF流{etf_flow:+.0f}亿 (中性)"

    if score >= 2:
        state = "greed"
    elif score <= -1:
        state = "fear"
    else:
        state = "normal"

    # 调整: 如果同时出现矛盾信号, 降为 normal
    if score >= 1 and 跌停_count > 10:
        state = "normal"

    return {
        "state": state,
        "score": score,
        "detail": f"情绪: {state} (score={score})",
        "advance_decline_ratio": round(advance_decline_ratio, 2),
        "涨停": 涨停_count,
        "跌停": 跌停_count,
        "etf_flow_yi": round(etf_flow, 2),
        "ad_note": ad_note,
        "zt_note": zt_note,
        "etf_note": etf_note,
    }


# ════════════════════════════════════════════════════════════════
#  5. 一站式市场状态获取
# ════════════════════════════════════════════════════════════════

def get_current_regime() -> dict[str, Any]:
    """一站式获取当前全面市场状态.

    自动收集各维度的实时数据并综合分析.

    Returns:
        {trend, volatility, liquidity, sentiment, composite_risk_level, timestamp}
        其中 composite_risk_level: 0-10 (0=极低风险, 10=极高风险)
    """
    ts = datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S")

    # 1. 趋势 & 波动 — 用上证指数
    index_bars = _get_index_bars("000001", 300)
    trend = detect_regime(index_bars)
    volatility = detect_volatility(index_bars)

    # 2. 流动性 — 使用 margin 和成交量
    margin_data: dict | None = None
    volume_data: dict | None = None

    try:
        from quant_system.margin import fetch_margin_summary
        margin_data = fetch_margin_summary()
    except Exception as e:
        logging.getLogger(__name__).error(f"[market_regime] 操作失败: {e}", exc_info=True)

    # 成交量: 从新浪获取
    try:
        r = requests.get("http://hq.sinajs.cn/list=s_sh000001", headers=SINA_HEADERS, timeout=6)
        text = r.text.strip().split('"')[1] if '"' in r.text else ""
        parts = text.split(",")
        if len(parts) >= 6:
            sh_amt = _to_float(parts[5]) / 10000  # 亿元
            volume_data = {"amount_yi": sh_amt, "avg_20d_amount_yi": sh_amt}
            # P1-Q27-fix: 20日均额必须用历史成交额(亿元)，不能拿 volume(手) 当亿元。
            # 原实现 amounts=[float(b[5])...] 取 stock_zh_index_daily 的 volume(手)，
            # 与 amount_yi(亿元) 量纲混用 → 恒满足 amount < avg*0.7 → 恒判"缩量"。
            amount_hist = _get_index_amount_history("000001", 20)
            if amount_hist:
                volume_data["avg_20d_amount_yi"] = round(sum(amount_hist) / len(amount_hist), 1)
                volume_data["avg_20d_source"] = "amount_history"
            else:
                # 降级：以今日价格水平把 volume 换算为金额（今日成交额 × 20日均量/今日量）
                volumes = [float(b[5]) for b in index_bars[-20:] if len(b) > 5]
                today_vol = volumes[-1] if volumes else 0.0
                avg_vol = sum(volumes) / len(volumes) if volumes else 0.0
                if today_vol > 0 and avg_vol > 0:
                    volume_data["avg_20d_amount_yi"] = round(sh_amt * avg_vol / today_vol, 1)
                    volume_data["avg_20d_estimate"] = True  # P1-Q27-fix: 可见的降级标注
                else:
                    volume_data["avg_20d_amount_yi"] = sh_amt
                    volume_data["avg_20d_estimate"] = True
    except Exception as e:
        logging.getLogger(__name__).error(f"[market_regime] 操作失败: {e}", exc_info=True)

    liquidity = detect_liquidity(margin_data, volume_data)

    # 3. 情绪
    ad_ratio = 1.0
    zt_count = 0
    dt_count = 0
    etf_flow = 0.0
    ad_estimate = True   # P1-Q27-fix: 涨跌家数是否估算
    limit_data = None    # P1-Q27-fix: 真实涨跌停是否可用
    try:
        from quant_system.market_context import get_market_context
        ctx = get_market_context()
        ad = ctx.get("advance_decline", {})
        adv = _to_float(ad.get("advance", 0))
        dec = _to_float(ad.get("decline", 0))
        ad_ratio = adv / dec if dec > 0 else (adv / max(adv, 1))
        ad_estimate = bool(ad.get("estimate", True))
        # P1-Q27-fix: market_context 无 limit_up/涨停 键（实测 keys 只有
        # indices/advance_decline/commodities/global_indices），从真实宽度 meta 读取。
        limit_data = _load_real_limit_counts()
        if limit_data is not None:
            zt_count, dt_count = limit_data
    except Exception as e:
        logging.getLogger(__name__).error(f"[market_regime] 操作失败: {e}", exc_info=True)

    sentiment = detect_sentiment(ad_ratio, zt_count, dt_count, etf_flow)
    # P1-Q27-fix: 涨跌停数据缺失或涨跌家数为估算时，显式标注情绪置信度低
    if limit_data is None or ad_estimate:
        sentiment["confidence"] = "low"
        missing_parts = []
        if limit_data is None:
            missing_parts.append("涨跌停数据缺失(未取到近似涨停/跌停)")
        if ad_estimate:
            missing_parts.append("涨跌家数为估算值")
        sentiment["zt_note"] = f"{'; '.join(missing_parts)} → 情绪判定置信度低 (AD={ad_ratio:.2f})"

    # 4. 综合风险评分 (0-10)
    risk_score = _compute_composite_risk(trend, volatility, liquidity, sentiment)

    return {
        "trend": trend,
        "volatility": volatility,
        "liquidity": liquidity,
        "sentiment": sentiment,
        "composite_risk_level": risk_score,
        "timestamp": ts,
    }


def _compute_composite_risk(
    trend: dict[str, Any],
    volatility: dict[str, Any],
    liquidity: dict[str, Any],
    sentiment: dict[str, Any],
) -> int:
    """综合风险评分 0-10.

    评分规则 (各维度满分):
      - 趋势: 0-4 (bull=0, sideways=2, bear=4)
      - 波动: 0-3 (low=0, normal=1, high=2, extreme=3)
      - 流动性: 0-2 (abundant=0, normal=1, tight=2)
      - 情绪: 0-1 (normal=0, greed/fear=1)
    """
    score = 0

    # 趋势风险
    t = trend.get("trend", "sideways")
    score += {"bull": 0, "recovery": 1, "sideways": 2, "correction": 3, "bear": 4, "unknown": 2}.get(t, 2)

    # 波动风险
    v = volatility.get("state", "normal")
    score += {"low": 0, "normal": 1, "high": 2, "extreme": 3, "unknown": 1}.get(v, 1)

    # 流动性风险
    l = liquidity.get("state", "normal")
    score += {"abundant": 0, "normal": 1, "tight": 2}.get(l, 1)

    # 情绪风险 (极端的贪婪/恐惧都加风险)
    s = sentiment.get("state", "normal")
    if s in ("greed", "fear"):
        score += 1

    return min(score, 10)


# ════════════════════════════════════════════════════════════════
#  6. 状态转文本
# ════════════════════════════════════════════════════════════════

def regime_to_text(regime: dict[str, Any]) -> str:
    """将市场状态转换为可读文本.

    Args:
        regime: get_current_regime() 返回的字典.

    Returns:
        格式化字符串.
    """
    trend = regime.get("trend", {})
    volatility = regime.get("volatility", {})
    liquidity = regime.get("liquidity", {})
    sentiment = regime.get("sentiment", {})
    risk = regime.get("composite_risk_level", 0)

    # 趋势图标
    trend_icons = {
        "bull": "📈",
        "bear": "📉",
        "sideways": "➡️",
        "recovery": "🔄",
        "correction": "🔻",
    }
    trend_icon = trend_icons.get(trend.get("trend", ""), "❓")
    trend_text = trend.get("trend", "未知").upper()

    # 波动图标
    vol_icons = {"low": "💤", "normal": "📊", "high": "⚡", "extreme": "🔥"}
    vol_icon = vol_icons.get(volatility.get("state", ""), "❓")
    vol_text = {
        "low": "低波动",
        "normal": "正常波动",
        "high": "高波动",
        "extreme": "极端波动",
    }.get(volatility.get("state", ""), "未知")

    # 流动性图标
    liq_icons = {"abundant": "💧", "normal": "🌊", "tight": "💀"}
    liq_icon = liq_icons.get(liquidity.get("state", ""), "❓")
    liq_text = {
        "abundant": "流动性充裕",
        "normal": "流动性正常",
        "tight": "流动性紧缩",
    }.get(liquidity.get("state", ""), "未知")

    # 情绪图标
    sent_icons = {"greed": "🟢", "normal": "⚪", "fear": "🔴"}
    sent_icon = sent_icons.get(sentiment.get("state", ""), "❓")
    sent_text = {
        "greed": "贪婪",
        "normal": "情绪适中",
        "fear": "恐惧",
    }.get(sentiment.get("state", ""), "未知")

    # 风险等级
    if risk <= 2:
        risk_label = "🟢 低风险"
    elif risk <= 4:
        risk_label = "🟡 中低风险"
    elif risk <= 6:
        risk_label = "🟠 中等风险"
    elif risk <= 8:
        risk_label = "🔴 高风险"
    else:
        risk_label = "⛔ 极高风险"

    lines = [
        f"{trend_icon} {trend_text} · {vol_icon}{vol_text} · {liq_icon}{liq_text} · {sent_icon}{sent_text}",
        f"  风险等级: {risk_label} ({risk}/10)",
        f"  MA20={trend.get('ma20',0):.0f} MA60={trend.get('ma60',0):.0f} MA144={trend.get('ma144',0):.0f}",
        f"  年化波动: {volatility.get('vol_20d',0):.1f}% (历史{volatility.get('percentile',0):.0f}分位)",
    ]

    # 流动性明细
    liq_detail = liquidity.get("detail", "")
    if liquidity.get("margin_detail"):
        lines.append(f"  {liquidity['margin_detail']}")
    if liquidity.get("volume_detail"):
        lines.append(f"  {liquidity['volume_detail']}")

    # 情绪明细
    if sentiment.get("ad_note"):
        lines.append(f"  {sentiment['ad_note']}")
    if sentiment.get("zt_note"):
        lines.append(f"  {sentiment['zt_note']}")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
#  CLI 入口
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="市场状态识别")
    parser.add_argument("--terse", action="store_true", help="简洁输出")
    args = parser.parse_args()

    regime = get_current_regime()
    if args.terse:
        t = regime.get("trend", {}).get("trend", "?")
        v = regime.get("volatility", {}).get("state", "?")
        l = regime.get("liquidity", {}).get("state", "?")
        s = regime.get("sentiment", {}).get("state", "?")
        r = regime.get("composite_risk_level", "?")
        print(f"{t} V:{v} L:{l} S:{s} R:{r}/10")
    else:
        print(regime_to_text(regime))
