"""
量化交易系统 — 买入机会扫描器。

对 409 只主板≥300亿股票全量扫描，检测技术性买入机会。

核心逻辑:
  1. 第一轮过滤 (腾讯实时数据, 无K线): 排除明显不适合买入的
  2. 第二轮检测 (腾讯日K线 + 技术指标): 逐一检查买入信号
  3. 评分排序: 每个信号加总, 按分数排序输出

买入信号清单 (每个+1分):
  S1. RSI < 35 (超卖)
  S2. CCI < -80 (超卖)
  S3. Price near MA60 (价格在MA60附近, 强支撑)
  S4. Price near MA144 (价格在MA144附近, 长期强支撑)
  S5. MACD柱由负转正 (MACD金叉)
  S6. MACD柱底背离 (price新低, MACD柱不创新低)
  S7. 量比 > 1.5 (放量企稳)
  S8. 今日收涨 (确认企稳)
  S9. BOLL下轨附近 (price < BOLL下轨*1.02)
  S10. PB < 1.5 (低估, 安全边际)

用法:
  python3 -m quant_system.opportunity                     # 全量扫描
  python3 -m quant_system.opportunity --top 20            # TOP 20
  python3 -m quant_system.opportunity --brief             # 简要版
  python3 -m quant_system.opportunity --sector 银行       # 只看某行业
  python3 -m quant_system.opportunity --deep 601288       # 某只完整评估
"""

from __future__ import annotations
import logging

import os
import sys
import threading
import time as _time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from quant_system.financial_data import get_financial_summary
from quant_system.watchlist import fetch_quotes, _cap_tier

# ── Config ──
CST = timezone(timedelta(hours=8))

# Cache: symbol -> (timestamp, bars_list)
_KLINE_CACHE: dict[str, tuple[float, list]] = {}
_KLINE_CACHE_TTL = 1800  # 30 min
# 并发 K 线抓取数: 4 → 8 (V6.3: 云端 300 只 15-30 分钟太慢; 8 并发东财通常能扛住, 失败有退避兜底; 可用 KLINE_MAX_WORKERS 覆盖)
_MAX_WORKERS = int(os.environ.get("KLINE_MAX_WORKERS", "8"))

# ── K线抓取风控统计 (东财限流退避) ──
_KLINE_STATS_LOCK = threading.Lock()
_KLINE_OK = 0            # 三源任一成功的股票数
_KLINE_FAIL = 0          # 三源全失败的股票数
_KLINE_DEGRADED = 0      # em 失败、由 sina/tencent 降级成功的股票数
_KLINE_CONSEC_FAIL = 0   # 连续失败数(>10 触发限流退避)
_KLINE_ATTEMPTS = 0      # 抓取尝试总数(用于周期性探测 em 是否恢复)
# 各源最近 20 次抓取结果, 用于 _kline_source_order() 按"当前最可靠源"动态排序
_SOURCE_RESULTS: dict[str, deque[bool]] = {name: deque(maxlen=20)
                                           for name in ("em", "sina", "tencent")}


def _em_success_rate() -> float:
    """最近 N 次 em 成功率(无样本时按 1.0, 默认 em 优先)。"""
    with _KLINE_STATS_LOCK:
        results = list(_SOURCE_RESULTS["em"])
    if not results:
        return 1.0
    return sum(results) / len(results)


def _record_source_result(name: str, ok: bool) -> None:
    """累计单个源最近抓取结果(供动态源排序采样)。"""
    with _KLINE_STATS_LOCK:
        _SOURCE_RESULTS[name].append(ok)


def _kline_source_order() -> list[tuple[Any, str]]:
    """按各源最近成功率降序返回 K 线抓取源列表(当前最可靠源优先)。

    V6.3-fix: em 失败时不再逐源完整尝试——全局 em 失败率 > 70% 时直接跳过 em、
    优先 sina; sina 失败率同样高时优先 tencent。无样本源按 1.0 参与排序
    (稳定排序下保持 em → sina → tencent 默认优先); 每 20 次抓取保留一次
    em 探测, 恢复后自动切回。
    """
    with _KLINE_STATS_LOCK:
        rates = {name: (sum(v) / len(v) if v else 1.0)
                 for name, v in _SOURCE_RESULTS.items()}
        em_has_sample = bool(_SOURCE_RESULTS["em"])
        attempts = _KLINE_ATTEMPTS
    order = ["em", "sina", "tencent"]
    order.sort(key=lambda n: rates[n], reverse=True)
    if em_has_sample and rates["em"] < 0.3 and attempts % 20 != 0:
        order.remove("em")  # em 失败率>70%: 直接跳过, 不再等完整三源尝试
    fns = {"em": _fetch_kline_em, "sina": _fetch_kline_sina,
           "tencent": _fetch_kline_tencent}
    return [(fns[n], n) for n in order]


def _kline_backoff() -> None:
    """连续失败 > 10 次 → 每次抓取前 sleep 1s, 限流退避避免三源雪崩。"""
    with _KLINE_STATS_LOCK:
        consec = _KLINE_CONSEC_FAIL
    if consec > 10:
        _time.sleep(1.0)


def _record_kline_result(success: bool, em_ok: Optional[bool] = None,
                         degraded: bool = False) -> None:
    """累计全局 K 线抓取统计。em_ok=None 表示本轮未尝试 em(不采样)。"""
    global _KLINE_OK, _KLINE_FAIL, _KLINE_DEGRADED
    global _KLINE_CONSEC_FAIL, _KLINE_ATTEMPTS
    with _KLINE_STATS_LOCK:
        _KLINE_ATTEMPTS += 1
        if em_ok is not None:
            _SOURCE_RESULTS["em"].append(em_ok)
        if success:
            _KLINE_OK += 1
            _KLINE_CONSEC_FAIL = 0
            if degraded:
                _KLINE_DEGRADED += 1
        else:
            _KLINE_FAIL += 1
            _KLINE_CONSEC_FAIL += 1


def _kline_stats() -> tuple[int, int, float]:
    """返回 (成功数, 失败数, 降级源占比%)。"""
    with _KLINE_STATS_LOCK:
        ok, fail, degraded = _KLINE_OK, _KLINE_FAIL, _KLINE_DEGRADED
    pct = degraded / ok * 100 if ok else 0.0
    return ok, fail, pct


# ════════════════════════════════════════════════════════════════
# 1. Fetch daily K-line from Tencent
# ════════════════════════════════════════════════════════════════

def _tencent_prefix(symbol: str) -> str:
    return f"sh{symbol}" if symbol.startswith("6") else f"sz{symbol}"


def fetch_kline(symbol: str, days: int = 160, force: bool = False) -> list[list] | None:
    """
    Fetch daily K-line via AKShare（数据源自动回退）。
    Returns list of [date, open, close, high, low, volume]

    V6.1-fix: Vultr 海外服务器访问东方财富接口被风控(RemoteDisconnected),
    导致 K 线拉取成功率 0% → 扫描永远 0 机会。
    回退链: 东财 stock_zh_a_hist → 新浪 stock_zh_a_daily → 腾讯 stock_zh_a_hist_tx。

    V6.2-fix: 并发风控修复 (云端 RemoteDisconnected 全失败):
      - 连续失败 > 10 次 → 抓取前 sleep 1s 限流退避, 避免三源雪崩;
      - em 最近成功率 < 50% → 源顺序轮换为 sina 优先 (每 4 次仍探测一次 em 恢复);
      - 全局累计成功/失败/降级占比, 在 [opp] 进度行输出。

    V6.3-fix: 速度优化 (300 只 15-30 分钟太慢):
      - _kline_source_order() 按各源最近成功率排序, 直接跳过高失败率源,
        不再逐源完整尝试 (em 失败率>70% 跳过 em 优先 sina, 每 20 次探测恢复);
      - _MAX_WORKERS 默认 4 → 8, 失败有退避兜底。
    """
    now = _time.time()
    if not force and symbol in _KLINE_CACHE:
        ts, data = _KLINE_CACHE[symbol]
        if now - ts < _KLINE_CACHE_TTL:
            return data

    # P2-P2-Q10-L034-fix: 起始日期此前硬编码 "20250101", days 参数(67/936行)被忽略。
    # 按 days 计算起始日期: 1.6 倍日历日≈覆盖 days 个交易日, 再加 40 天缓冲
    # 确保 MA144 等长周期指标有足够 bar。
    start_dt = datetime.now(CST) - timedelta(days=int(days * 1.6) + 40)
    start_date = start_dt.strftime("%Y%m%d")

    _kline_backoff()

    # V6.3-fix: 源顺序按最近成功率动态排序 (当前最可靠源优先),
    # em 失败率>70% 时直接跳过 em → sina/tencent, 不再逐源完整尝试
    sources = _kline_source_order()

    bars = None
    em_ok: Optional[bool] = None
    source_ok = ""
    for fn, name in sources:
        bars = fn(symbol, start_date)
        _record_source_result(name, bars is not None)
        if name == "em":
            em_ok = bars is not None
        if bars is not None:
            source_ok = name
            break

    if bars is None:
        # Q10-fix: 回退一律返回 None，严禁用分钟线冒充日线
        _record_kline_result(success=False, em_ok=em_ok)
        return None
    _record_kline_result(success=True, em_ok=em_ok, degraded=source_ok != "em")
    _KLINE_CACHE[symbol] = (now, bars)
    return bars


def _fetch_kline_em(symbol: str, start_date: str) -> list[list] | None:
    """东方财富日 K（Vultr 上常被风控）。"""
    try:
        import akshare as ak
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                start_date=start_date, adjust="qfq")
        if df is None or len(df) < 30:
            return None
        bars = []
        for _, row in df.iterrows():
            bars.append([
                str(row["日期"]), float(row["开盘"]), float(row["收盘"]),
                float(row["最高"]), float(row["最低"]), float(row["成交量"]),
            ])
        return bars if len(bars) >= 30 else None
    except Exception:
        return None


def _fetch_kline_sina(symbol: str, start_date: str) -> list[list] | None:
    """新浪日 K: ak.stock_zh_a_daily(symbol='sh600519', ...)。"""
    try:
        import akshare as ak
        prefix = f"sh{symbol}" if symbol.startswith("6") else (
            f"bj{symbol}" if symbol.startswith(("4", "8")) else f"sz{symbol}")
        df = ak.stock_zh_a_daily(symbol=prefix, start_date=start_date, adjust="qfq")
        if df is None or len(df) < 30:
            return None
        bars = []
        for _, row in df.iterrows():
            bars.append([
                str(row["date"]), float(row["open"]), float(row["close"]),
                float(row["high"]), float(row["low"]), float(row["volume"]),
            ])
        return bars if len(bars) >= 30 else None
    except Exception:
        return None


def _fetch_kline_tencent(symbol: str, start_date: str) -> list[list] | None:
    """腾讯日 K: ak.stock_zh_a_hist_tx(symbol='sh600519', ...)。"""
    try:
        import akshare as ak
        prefix = f"sh{symbol}" if symbol.startswith("6") else (
            f"bj{symbol}" if symbol.startswith(("4", "8")) else f"sz{symbol}")
        df = ak.stock_zh_a_hist_tx(symbol=prefix, start_date=start_date, adjust="qfq")
        if df is None or len(df) < 30:
            return None
        bars = []
        for _, row in df.iterrows():
            bars.append([
                str(row["date"]), float(row["open"]), float(row["close"]),
                float(row["high"]), float(row["low"]), float(row["volume"]),
            ])
        return bars if len(bars) >= 30 else None
    except Exception:
        return None


# 行业分类缓存
_SECTOR_CACHE: dict[str, tuple[float, str]] = {}  # symbol -> (timestamp, sector)
_SECTOR_CACHE_TTL = 86400  # 24h

# ── sector 抓取风控统计 (东财 stock_individual_info_em 限流退避) ──
_SECTOR_STATS_LOCK = threading.Lock()
_SECTOR_FAIL_CONSEC = 0   # 连续失败数(>5 触发轻退避 sleep 0.3s)
_SECTOR_FAIL_TOTAL = 0    # 累计失败数(每 20 次打一条汇总, 避免刷屏)
_SECTOR_SKIP = False      # 连续失败>10 次后置 True: 本进程剩余扫描跳过全部 sector 查询


def _get_stock_sector(symbol: str) -> str:
    """获取股票所属行业 (缓存24小时)。

    P2-Q10-M026-fix: 原 `ak.stock_board_industry_cons_em(symbol=股票代码)` 的 symbol 参数
    语义是**板块名称**(如"银行")，传 6 位股票代码恒取不到 → sector 恒 "", 行业分布/
    --sector 过滤失效。改用 `ak.stock_individual_info_em(symbol)` 单股查询: 返回
    item/value 两列 DataFrame, 行业在 item=="行业"。

    云端限流修复 (Pass3 卡死): 不重试(单次查询, 失败即返回 "" 不影响机会判定);
    连续失败 > 5 次后每次失败前 sleep 0.3s 轻退避; 不再逐只打 warning, 改为
    每累计 20 次失败打一条汇总。

    V6.3-fix: 连续失败 > 10 次 → _SECTOR_SKIP=True, 本进程剩余扫描直接跳过
    全部 sector 查询 (sector 仅展示字段, 失败不影响机会判定), 不再逐只尝试。
    """
    global _SECTOR_FAIL_CONSEC, _SECTOR_FAIL_TOTAL, _SECTOR_SKIP
    now = _time.time()
    if symbol in _SECTOR_CACHE:
        ts, sector = _SECTOR_CACHE[symbol]
        if now - ts < _SECTOR_CACHE_TTL:
            return sector
    if _SECTOR_SKIP:
        return ""
    # 连续失败 > 5 次 → 轻退避 0.3s, 避免东财限流时几十只候选排队卡死
    with _SECTOR_STATS_LOCK:
        consec = _SECTOR_FAIL_CONSEC
    if consec > 5:
        _time.sleep(0.3)
    import akshare as ak
    try:
        df = ak.stock_individual_info_em(symbol=symbol)
        if df is not None and len(df) > 0:
            for _, row in df.iterrows():
                if str(row.get("item", "")).strip() == "行业":
                    sector = str(row.get("value", "") or "").strip()
                    if sector:
                        _SECTOR_CACHE[symbol] = (now, sector)
                        with _SECTOR_STATS_LOCK:
                            _SECTOR_FAIL_CONSEC = 0
                        return sector
    except Exception as e:
        with _SECTOR_STATS_LOCK:
            _SECTOR_FAIL_CONSEC += 1
            _SECTOR_FAIL_TOTAL += 1
            consec, total = _SECTOR_FAIL_CONSEC, _SECTOR_FAIL_TOTAL
        if consec > 10:
            _SECTOR_SKIP = True
            logging.getLogger(__name__).warning(
                f"[opportunity] _get_stock_sector 连续失败 {consec} 次 > 10 — "
                "本进程剩余扫描跳过全部 sector 查询 (sector 仅展示字段, 不影响机会判定)"
            )
        elif total % 20 == 0:
            logging.getLogger(__name__).warning(
                f"[opportunity] _get_stock_sector 累计失败 {total} 次"
                f"(连续 {consec} 次, 最近 {symbol}: {e}) — sector 返回空, 不影响机会判定"
            )
    return ""


def _limit_state(q: dict) -> str:
    """检测涨跌停状态。

    A股规则: 主板 ±10%, 创业板(300/301)/科创板(688) ±20%, ST ±5%。
    涨跌停价按昨收精确取整到分; 返回:
      "limit_up"/"limit_down"       — 已封板(无法成交/无法卖出, 应剔除)
      "near_limit_up"/"near_limit_down" — 距涨跌停 5% 以内(风险标注)
      ""                            — 正常
    """
    try:
        price = float(q.get("price", 0))
        prev_close = float(q.get("prev_close", 0))
    except (TypeError, ValueError):
        return ""
    if price <= 0 or prev_close <= 0:
        return ""
    sym = str(q.get("symbol", ""))
    name = str(q.get("name", ""))
    # P2-P2-Q10-M023-fix: 涨跌停限制 — ST ±5%、创业板/科创板 ±20%、其余主板 ±10%
    # V11 审计修复（Medium）: 北交所（4/8/920 开头）原按主板 10% 处理 → 20% 误差，
    # 实际北交所涨跌幅 ±30%（上市首日无限制）。
    if "ST" in name.upper():
        limit_pct = 0.05
    elif sym.startswith(("3", "688")):
        limit_pct = 0.20
    elif sym.startswith(("4", "8", "920")):
        limit_pct = 0.30
    else:
        limit_pct = 0.10
    limit_up = round(prev_close * (1 + limit_pct), 2)
    limit_down = round(prev_close * (1 - limit_pct), 2)
    if price >= limit_up - 0.001:
        return "limit_up"
    if price <= limit_down + 0.001:
        return "limit_down"
    near_price = prev_close * limit_pct * 0.05  # 距涨跌停 0.5%(按昨收, 单位元)
    if price >= limit_up - near_price:
        return "near_limit_up"
    if price <= limit_down + near_price:
        return "near_limit_down"
    return ""


def _weighted_score(signals: dict) -> int:
    """信号加权评分(含支撑汇聚加成)。

    P2-Q10-M027-fix: 与 compute_indicators 内部评分逻辑保持一致, 供扫描层在注入
    low_pb 等扫描层信号后重算分数(compute_indicators 无 pb 入参, 无法在其内部评分)。
    """
    raw = sum(v for v in signals.values() if v > 0)
    if signals.get("downtrend", 0) < 0:
        raw += signals["downtrend"]
    support_count = sum(1 for k in signals if "support" in k)
    if support_count >= 2:
        raw *= 1.15
    if support_count >= 3:
        raw *= 1.1
    return int(round(raw))


# ── 辅助函数 ──
def _sma(data_vals, n):
    if len(data_vals) < n:
        return [data_vals[-1]] if data_vals else []
    return [sum(data_vals[i-n:i])/n for i in range(n, len(data_vals)+1)]

def _ema(data_vals, n):
    if not data_vals:
        return []
    result = [data_vals[0]]
    k = 2 / (n + 1)
    for v in data_vals[1:]:
        result.append(v * k + result[-1] * (1 - k))
    return result

def _rsi(data_vals, n=14):
    if len(data_vals) < n + 1:
        return data_vals[-1] if data_vals else 50
    gains, losses = 0, 0
    for i in range(-n, 0):
        diff = data_vals[i] - data_vals[i-1]
        if diff > 0:
            gains += diff
        else:
            losses -= diff
    avg_gain = gains / n
    avg_loss = losses / n
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def _cci(data_h, data_l, data_c, n=20):
    if len(data_c) < n:
        return 0
    tp = [(data_h[i] + data_l[i] + data_c[i]) / 3 for i in range(-n, 0)]
    mean_tp = sum(tp) / n
    md = sum(abs(t - mean_tp) for t in tp) / n
    if md == 0:
        return 0
    return (tp[-1] - mean_tp) / (0.015 * md)


def _find_pivots(closes: list[float], window: int = 5) -> tuple[list[int], list[int]]:
    """
    识别波段高点和低点 (摆动点检测)
    返回 (高点下标列表, 低点下标列表)
    """
    highs = []
    lows = []
    for i in range(window, len(closes) - window):
        # 波段高点: 左右各 window 根 K 线的最高
        if all(closes[i] >= closes[j] for j in range(i - window, i + window + 1) if j != i):
            highs.append(i)
        # 波段低点
        if all(closes[i] <= closes[j] for j in range(i - window, i + window + 1) if j != i):
            lows.append(i)
    return highs, lows


def _wave_analysis(closes: list[float], highs_idx: list[int], lows_idx: list[int]) -> dict:
    """
    基础的波浪结构分析
    返回: {"trend": "up"|"down"|"side", "wave_count": int, "last_swing_high", "last_swing_low",
           "fib_retrace_382", "fib_retrace_500", "fib_retrace_618", "fib_retrace_786"}
    """
    result = {"trend": "side", "wave_count": 0}

    if len(highs_idx) < 2 or len(lows_idx) < 2:
        return result

    # 最近的几个摆动点
    recent_highs = [closes[i] for i in highs_idx[-5:]]
    recent_lows = [closes[i] for i in lows_idx[-5:]]

    latest_price = closes[-1]
    last_high = max(recent_highs) if recent_highs else latest_price
    last_low = min(recent_lows) if recent_lows else latest_price

    result["last_swing_high"] = round(last_high, 2)
    result["last_swing_low"] = round(last_low, 2)

    # 趋势判断: 更高高点+更高低点 = 上升趋势
    if len(recent_highs) >= 2 and len(recent_lows) >= 2:
        higher_highs = recent_highs[-1] > recent_highs[-2]
        higher_lows = recent_lows[-1] > recent_lows[-2]
        if higher_highs and higher_lows:
            result["trend"] = "up"
        elif not higher_highs and not higher_lows:
            result["trend"] = "down"

    # 斐波那契回撤位 (基于最近的一个完整摆动区间)
    swing_range = last_high - last_low
    if swing_range > 0:
        result["fib_382"] = round(last_high - swing_range * 0.382, 2)
        result["fib_500"] = round(last_high - swing_range * 0.5, 2)
        result["fib_618"] = round(last_high - swing_range * 0.618, 2)
        result["fib_786"] = round(last_high - swing_range * 0.786, 2)

    # 价格相对于斐波那契的位置
    fibs = [result.get(k, 0) for k in ["fib_382", "fib_500", "fib_618", "fib_786"]]
    if fibs:
        nearest_fib = min(fibs, key=lambda f: abs(latest_price - f) if f > 0 else float('inf'))
        idx = fibs.index(nearest_fib) if nearest_fib in fibs else -1
        labels = ["0.382", "0.5", "0.618", "0.786"]
        result["nearest_fib"] = labels[idx] if idx >= 0 else ""
        result["nearest_fib_price"] = nearest_fib

    return result

# ── 模块级辅助函数（避免 computed_indicators 内部重复定义） ──

def _calc_slope(arr: list[float], n: int = 20) -> float:
    """
    用线性回归计算序列斜率。n=20相比原n=5大幅降低噪声。
    返回回归系数，反映最近n期的趋势方向与强度。
    """
    if not arr or len(arr) < n:
        return 0.0
    y = arr[-n:]
    x = list(range(n))
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    den = sum((x[i] - mean_x) ** 2 for i in range(n))
    return num / den if den != 0 else 0.0


def _get_price_band_thresh(px: float) -> tuple[float, float]:
    """根据股价分档返回自适应接近阈值 (pct_thresh, abs_pct)。

    P2-Q10-M029-fix: 原 abs_thresh 为固定元数(0.05/0.10/0.20), 千元股
    max_allowed_pct=min(pct_thresh, 0.2/1000)≈0.02%, MA60/MA144/Fib/摆动低点
    支撑对高价股几乎永不触发。现改为**占价格的比例**(0.003=0.3%)。
    """
    if px < 10:
        return 0.005, 0.005   # 低价股: 0.5% 绝对容差
    elif px <= 50:
        return 0.003, 0.003   # 中价股: 0.3% 绝对容差
    else:
        return 0.002, 0.003   # 高价股: 0.3% 绝对容差(不再被固定元数压死)


def _near(px: float, target: float, pct_thresh: float = None, abs_thresh: float = None) -> tuple[bool, float]:
    """是否接近: 自适应价格分档。返回 (is_near, dist_pct)。

    P2-Q10-M029-fix: abs_thresh 语义改为**占 target 的比例**(0.003=0.3%)，
    与 pct_thresh 同量纲直接取 min；修复高价股支撑信号系统性失效。
    """
    if target <= 0:
        return False, 999.0
    if pct_thresh is None:
        pct_thresh, abs_thresh = _get_price_band_thresh(target)
    pct_dist = abs(px - target) / target
    max_allowed_pct = min(pct_thresh, abs_thresh)
    return pct_dist <= max_allowed_pct, round(pct_dist * 100, 2)


def _get_factor_scores(symbols: list[str]) -> dict[str, float]:
    """
    对一组标的批量计算 factor_zoo 因子复合评分，截面归一化到 0-10。
    返回 {symbol: factor_score}；factor_zoo 不可用或无数据时返回空字典。

    P1-Q10-fix: 原 `_get_factor_composite_score(symbol)` 逐股调用，存在三重必挂：
      ① fetch_quotes 返回 list[dict]，`symbol not in quotes` 恒为真 → 直接 return {}；
      ② compute_factors 签名是 (symbol, bars, ...)，原 `compute_factors({symbol: quote})`
         把 dict 当 symbol 且缺 bars → TypeError；
      ③ compute_composite_score 对单标的 {factor: z} 生成单行 DataFrame，0-10 归一化
         时 min==max → 恒 0。因子分必须跨标截面计算才有意义。
    因此改为对候选集批量计算：{factor: {symbol: value}} → 截面归一化 → 每只一档分数。
    """
    try:
        from quant_system.factor_zoo import compute_composite_score, compute_factors, valid_factors_from_registry
        valid = set(valid_factors_from_registry())
        ic_dict: dict[str, dict[str, float]] = {}
        try:
            import json as _json
            from pathlib import Path as _Path
            p = _Path(__file__).resolve().parent.parent / "generated" / "factor_quality_registry.json"
            for x in _json.loads(p.read_text(encoding="utf-8")).get("factors", []):
                if x.get("tier") in ("core", "candidate"):
                    ic_dict[str(x.get("factor"))] = {"ic": x.get("ic_mean") or 0, "icir": x.get("icir") or 0}
        except Exception:
            ic_dict = {}
        by_factor: dict[str, dict[str, float]] = {}
        for sym in symbols:
            # compute_factors(symbol, bars) 需要日线 bars；扫描时 _KLINE_CACHE 已命中，
            # 无额外网络请求。
            bars = fetch_kline(sym)
            if not bars or len(bars) < 30:
                continue
            f = compute_factors(sym, bars)
            if not f:
                continue
            for fname, val in f.items():
                by_factor.setdefault(fname, {})[sym] = float(val)
        if not by_factor:
            return {}
        # 有效因子池过滤 + icir 加权；若账本缺失则退化为等权但不伪造有效名单。
        if valid:
            by_factor = {k: v for k, v in by_factor.items() if k in valid}
        if ic_dict:
            ic_dict = {k: v for k, v in ic_dict.items() if k in by_factor}
        # compute_composite_score 接受 {factor: {symbol: value}} → DataFrame(index=symbols)
        scores = compute_composite_score(by_factor, weighting="icir_weighted" if ic_dict else "equal", ic_dict=ic_dict or None)
        if scores is None or len(scores) == 0:
            return {}
        return {sym: float(val) for sym, val in scores.items()}
    except Exception as e:
        # P1-Q10-fix: 降级必须可见 —— 不再静默吞异常
        print(f"[opp] 因子评分失败: {e}", flush=True)
        return {}


def compute_indicators(bars: list[list], fin_data: Optional[dict] = None) -> dict[str, Any]:
    """
    综合分析: 技术指标 + 波浪结构 + 斐波那契 + 量价配合 + 多维度信号加权

    Args:
        bars: K-line data
        fin_data: Optional fundamental data dict with roe, profit_growth,
                  revenue_growth, debt_ratio, gross_margin
    """
    if not bars or len(bars) < 30:
        return {}

    closes = [float(b[2]) for b in bars]
    highs = [float(b[3]) for b in bars]
    lows = [float(b[4]) for b in bars]
    volumes = [float(b[5]) for b in bars]

    price = closes[-1]
    prev_close = closes[-2] if len(closes) > 1 else price
    pct_chg = (price / prev_close - 1) * 100 if prev_close > 0 else 0

    # ── 均线 ──
    ma20_arr = _sma(closes, 20)
    ma60_arr = _sma(closes, 60)
    ma144_arr = _sma(closes, 144) if len(closes) >= 144 else None
    ma20_val = ma20_arr[-1] if ma20_arr else 0
    ma60_val = ma60_arr[-1] if ma60_arr else 0
    ma144_val = ma144_arr[-1] if ma144_arr else 0

    # 均线斜率 (平滑版，n=20)
    ma60_slope = _calc_slope(ma60_arr) if ma60_arr else 0.0
    ma144_slope = _calc_slope(ma144_arr) if ma144_arr else 0.0

    # ── RSI ──
    rsi_val = _rsi(closes, 14)
    rsi_prev = _rsi(closes[:-1], 14) if len(closes) >= 15 else rsi_val

    # ── CCI ──
    cci_val = _cci(highs, lows, closes, 20)

    # ── MACD ──
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    dif = ema12[-1] - ema26[-1] if ema12 and ema26 else 0
    dif_prev = (ema12[-2] - ema26[-2]) if len(ema12) > 1 and len(ema26) > 1 else dif
    dif_list = [(ema12[i] - ema26[i]) for i in range(min(len(ema12), len(ema26)))]
    if dif_list:
        dea_list = _ema(dif_list, 9)
        dea = dea_list[-1] if dea_list else 0
        macd_hist = (dif - dea) * 2
        dea_prev = dea_list[-2] if len(dea_list) > 1 else dea
        macd_hist_prev = (dif_prev - dea_prev) * 2
        # 完整 MACD 柱序列 —— P2-Q10-M022-fix: 底背离需要"创新低那根 bar"的 MACD,
        # 仅最后两根(未对齐)无法正确比较。
        macd_hist_list = [(dif_list[i] - dea_list[i]) * 2
                          for i in range(min(len(dif_list), len(dea_list)))]
    else:
        macd_hist = macd_hist_prev = 0
        macd_hist_list = []

    # ── BOLL ──
    if len(closes) >= 20:
        last20 = closes[-20:]
        mid20 = sum(last20) / 20
        std = (sum((x - mid20) ** 2 for x in last20) / 20) ** 0.5
        boll_upper = mid20 + 2 * std
        boll_lower = mid20 - 2 * std
    else:
        boll_upper = price * 1.1
        boll_lower = price * 0.9

    # ── 量比 ──
    if len(volumes) >= 21:
        avg_vol = sum(volumes[-21:-1]) / 20
        vol_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 0
    else:
        vol_ratio = 0

    # ── 波浪结构 + 斐波那契 ──
    pivots_h, pivots_l = _find_pivots(closes, window=5)
    wave = _wave_analysis(closes, pivots_h, pivots_l)

    # ── 信号检测 (加权, 非二值) ──
    signals: dict[str, float] = {}
    signal_reasons: dict[str, str] = {}

    # S1: 均线支撑 (自适应阈值) — 使用模块级 _near
    is_near60, dist60 = _near(price, ma60_val) if ma60_val > 0 else (False, 999)
    is_near144, dist144 = _near(price, ma144_val) if ma144_val > 0 else (False, 999)

    # 均线趋势不是下降的才视为有效支撑。
    # P2-P2-Q10-M029-fix: 斜率按价格归一化(元/日 → %/日), 否则 5 元股 -0.1 元/日已很陡、
    # 千元股 -0.1 元/日却极缓, 同一阈值跨价格不可比。MA60 允许 -0.2%/日, MA144 更严 -0.1%/日。
    ma60_slope_pct = (ma60_slope / ma60_val * 100) if ma60_val > 0 else 0.0
    ma144_slope_pct = (ma144_slope / ma144_val * 100) if ma144_val > 0 else 0.0
    if is_near60 and ma60_slope_pct >= -0.2:
        signals["ma60_support"] = 1.0
        signal_reasons["ma60_support"] = f"MA60=¥{ma60_val:.2f}, 距离{dist60}%"

    if is_near144 and ma144_slope_pct >= -0.1:
        signals["ma144_support"] = 1.5  # 144均线支撑权重更高
        signal_reasons["ma144_support"] = f"MA144=¥{ma144_val:.2f}, 距离{dist144}%"

    # S2: 斐波那契回撤位支撑
    nearest_fib = wave.get("nearest_fib_price", 0)
    if nearest_fib > 0:
        is_near_fib, dist_fib = _near(price, nearest_fib, pct_thresh=0.005, abs_thresh=0.003)
        if is_near_fib:
            fib_label = wave.get("nearest_fib", "")
            signals["fib_support"] = 1.5
            signal_reasons["fib_support"] = f"{fib_label}回撤=¥{nearest_fib:.2f}, 距离{dist_fib}%"

    # S3: 波浪支撑 (价格在最近摆动低点附近)
    last_swing_low = wave.get("last_swing_low", 0)
    if last_swing_low > 0:
        is_near_swing, dist_swing = _near(price, last_swing_low, pct_thresh=0.01, abs_thresh=0.003)
        if is_near_swing:
            signals["swing_low_support"] = 1.0
            signal_reasons["swing_low_support"] = f"前低¥{last_swing_low}"

    # S4: RSI 超卖 + 回升
    if rsi_val < 35:
        signals["rsi_oversold"] = 1.0
        signal_reasons["rsi_oversold"] = f"RSI={rsi_val:.1f}"
    elif rsi_val < 45:
        signals["rsi_near_oversold"] = 0.5
        signal_reasons["rsi_near_oversold"] = f"RSI={rsi_val:.1f}"

    if rsi_val > rsi_prev and rsi_val < 50:
        signals["rsi_turning_up"] = 0.8
        signal_reasons["rsi_turning_up"] = f"RSI回升{rsi_prev:.0f}→{rsi_val:.0f}"

    # S5: CCI 超卖 + 回升
    if cci_val < -80:
        signals["cci_oversold"] = 1.0
        signal_reasons["cci_oversold"] = f"CCI={cci_val:.0f}"

    # S6: MACD
    if macd_hist > 0 and macd_hist_prev <= 0:
        signals["macd_bullish_cross"] = 1.2  # 金叉权重较高
        signal_reasons["macd_bullish_cross"] = f"MACD柱转正{macd_hist:.4f}"
    elif macd_hist > macd_hist_prev and macd_hist < 0:
        signals["macd_bullish_diverging"] = 0.6  # 绿柱缩脚
        signal_reasons["macd_bullish_diverging"] = f"MACD柱缩脚{macd_hist_prev:.4f}→{macd_hist:.4f}"

    # S7: MACD底背离 (price新低但MACD柱没新低)
    # P2-P2-Q10-M022-fix: 原 `low_3<low_5 and macd_hist>macd_hist_prev` 只比较最后两根 MACD
    # (未对齐创新低那根 bar), 大量误标。现取 low_3/low_5 窗口内实际新低 bar 的 MACD 对齐比较。
    if len(closes) >= 8 and len(macd_hist_list) >= 5:
        seg3 = closes[-3:]
        seg5 = closes[-5:-2]
        idx3 = len(closes) - 3 + seg3.index(min(seg3))  # low_3 对应 bar
        idx5 = len(closes) - 5 + seg5.index(min(seg5))  # low_5 对应 bar
        macd3 = macd_hist_list[idx3] if idx3 < len(macd_hist_list) else 0
        macd5 = macd_hist_list[idx5] if idx5 < len(macd_hist_list) else 0
        if closes[idx3] < closes[idx5] and macd3 > macd5:
            signals["bullish_divergence"] = 1.5  # 底背离高权重
            signal_reasons["bullish_divergence"] = (
                f"底背离(价新低{closes[idx3]:.2f}<{closes[idx5]:.2f}, MACD升{macd5:.3f}→{macd3:.3f})"
            )

    # S8: 今日收涨 (确认企稳) —— P2-Q10-M027-fix: docstring 承诺的 S8 此前未实现
    if pct_chg > 0:
        signals["today_up"] = 0.4
        signal_reasons["today_up"] = f"今日收涨{pct_chg:.1f}%"

    # S8b: 量价配合
    if vol_ratio > 1.5 and pct_chg > 0:
        signals["volume_price_up"] = 1.0
        signal_reasons["volume_price_up"] = f"放量{vol_ratio:.1f}x上涨{pct_chg:.1f}%"
    elif vol_ratio < 0.7 and pct_chg < 0:
        signals["volume_price_down"] = 0.5  # 缩量下跌=抛压衰竭
        signal_reasons["volume_price_down"] = f"缩量{vol_ratio:.2f}x下跌"

    # S9: BOLL下轨附近
    if price <= boll_lower * 1.01:
        signals["boll_lower_band"] = 1.0
        signal_reasons["boll_lower_band"] = f"BOLL下轨=¥{boll_lower:.2f}"
    elif price <= boll_lower * 1.03:
        signals["boll_near_lower"] = 0.5
        signal_reasons["boll_near_lower"] = f"BOLL下轨=¥{boll_lower:.2f}"

    # S10: 趋势背景
    trend = wave.get("trend", "side")
    if trend == "up":
        signals["uptrend"] = 0.8  # 上升趋势中的回调买入加分
        signal_reasons["uptrend"] = "上升趋势"
    elif trend == "down":
        signals["downtrend"] = -1.0  # 下降趋势减分

    # S11-S14: 基本面加权 (仅当 fin_data 提供时)
    if fin_data:
        roe = fin_data.get("roe", 0)
        profit_growth = fin_data.get("profit_growth", 0)
        gross_margin = fin_data.get("gross_margin", 0)
        debt_ratio = fin_data.get("debt_ratio", 100)

        # ROE: 分级评分
        if roe > 15:
            signals["fundamental_roe"] = 0.5
            signal_reasons["fundamental_roe"] = f"ROE={roe:.1f}%"
        elif roe > 8:
            signals["fundamental_roe"] = 0.2
            signal_reasons["fundamental_roe"] = f"ROE={roe:.1f}%"

        # 利润增长: 分级
        if profit_growth > 10:
            signals["fundamental_growth"] = 0.5
            signal_reasons["fundamental_growth"] = f"利润增长{profit_growth:.1f}%"
        elif profit_growth > 0:
            signals["fundamental_growth"] = 0.2  # 正增长但不高
            signal_reasons["fundamental_growth"] = f"利润增{profit_growth:.1f}%"

        # 毛利率: 分级
        if gross_margin > 30:
            signals["fundamental_margin"] = 0.3
            signal_reasons["fundamental_margin"] = f"毛利率{gross_margin:.1f}%"
        elif gross_margin > 15:
            signals["fundamental_margin"] = 0.15
            signal_reasons["fundamental_margin"] = f"毛利率{gross_margin:.1f}%"

        # 资产负债率: 分级
        if debt_ratio < 40:
            signals["fundamental_safety"] = 0.3
            signal_reasons["fundamental_safety"] = f"负债率{debt_ratio:.1f}%"
        elif debt_ratio < 60:
            signals["fundamental_safety"] = 0.15
            signal_reasons["fundamental_safety"] = f"负债率{debt_ratio:.1f}%"

        # 综合质量分: 基础盈利能力
        profitable = roe > 0 and profit_growth > -30  # 亏损收窄也算
        if profitable:
            signals["fundamental_profitable"] = 0.2
            signal_reasons["fundamental_profitable"] = "盈利"

    # ── 综合评分 (加权) ──
    raw_score = sum(v for v in signals.values() if v > 0)
    # 扣分项
    if signals.get("downtrend", 0) < 0:
        raw_score += signals["downtrend"]

    # 汇聚效应: 当多个支撑汇聚时加分
    support_count = sum(1 for k in signals if "support" in k)
    if support_count >= 2:
        raw_score *= 1.15  # 多支撑汇聚 +15%
    if support_count >= 3:
        raw_score *= 1.1  # 三重支撑再加

    # Q10-fix: signal_count 必须是 int，否则后续 "+"*signal_count 抛 TypeError
    final_score = int(round(raw_score))

    # ── 可读的信号摘要 ──
    signal_labels = []
    for k, v in signals.items():
        if v > 0:
            r = signal_reasons.get(k, k)
            signal_labels.append(r)

    # ── 因子评分（factor_zoo 复合分，由 scan_opportunities 传入 symbol 后重算） ──
    factor_score = 0.0

    return {
        "price": round(price, 2),
        "change_pct": round(pct_chg, 2),
        "ma20": round(ma20_val, 2),
        "ma60": round(ma60_val, 2),
        "ma144": round(ma144_val, 2),
        "ma60_slope": round(ma60_slope, 4),
        "ma144_slope": round(ma144_slope, 4),
        "ma60_slope_pct": round(ma60_slope_pct, 4),
        "ma144_slope_pct": round(ma144_slope_pct, 4),
        "rsi": round(rsi_val, 1),
        "cci": round(cci_val, 1),
        "macd_hist": round(macd_hist, 4),
        "boll_upper": round(boll_upper, 2),
        "boll_lower": round(boll_lower, 2),
        "vol_ratio": round(vol_ratio, 2),
        "trend": wave.get("trend", "side"),
        "fib_382": wave.get("fib_382"),
        "fib_618": wave.get("fib_618"),
        "last_swing_low": wave.get("last_swing_low"),
        "last_swing_high": wave.get("last_swing_high"),
        "signals": signals,
        "signal_reasons": signal_reasons,
        "signal_count": final_score,
        "factor_score": 0.0,
        "signal_summary": " | ".join(signal_labels[:6]),  # 最多6条
        # 基本面数据 (透传)
        "roe": fin_data.get("roe") if fin_data else None,
        "profit_growth": fin_data.get("profit_growth") if fin_data else None,
        "revenue_growth": fin_data.get("revenue_growth") if fin_data else None,
        "debt_ratio": fin_data.get("debt_ratio") if fin_data else None,
        "gross_margin": fin_data.get("gross_margin") if fin_data else None,
    }


# ════════════════════════════════════════════════════════════════
# 2. Full opportunity scan
# ════════════════════════════════════════════════════════════════

def scan_opportunities(top_n: int = 30, min_score: float = 2.5) -> list[dict]:
    """
    Full scan: 409 stocks → filter → K-line → signals → rank.

    Args:
        top_n: 返回前N个买入机会
        min_score: 最低信号分 (0-10)

    Returns:
        List of dicts sorted by signal_count desc:
        {symbol, name, price, change_pct, market_cap_yi, tier,
         signal_count, signal_summary, ...}
    """
    t0 = _time.time()

    # === Pass 1: Fetch all 409 quotes (fast ~5s) ===
    quotes = fetch_quotes(force=True)
    valid = [q for q in quotes if q.get("price", 0) > 0]
    print(f"[opp] {len(valid)}只行情 ({_time.time()-t0:.1f}s)", flush=True)

    # === Pass 2: Aggressive filter using Tencent quote data only ===
    candidates = []
    for q in valid:
        price = q.get("price", 0)
        pb = q.get("pb", 0)
        pe = q.get("pe_ttm", 0)
        pct = q.get("change_pct", 0)
        high52 = q.get("high_52w", 0)
        low52 = q.get("low_52w", 0)
        turnover = q.get("turnover", 0)

        # Hard filters
        if price < 3:
            continue
        if pb <= 0:
            continue
        if high52 > 0 and price / high52 > 0.95:
            continue  # don't chase highs
        if pe > 80:
            continue
        # P2-P2-Q10-M023-fix: 涨跌停状态过滤 —— 跌停无法成交、涨停买不进, 均剔除候选
        if _limit_state(q) in ("limit_up", "limit_down"):
            continue

        # Must have at least ONE reason to be a buy candidate:
        reasons = []
        if pct < -2:
            reasons.append("today_drop")
        if 0 < pe < 20:
            reasons.append("low_pe")
        if pb < 1.5:
            reasons.append("low_pb")
        if low52 > 0 and price / low52 < 1.10:
            reasons.append("near_low52")
        if turnover > 0.5:
            reasons.append("active")

        if len(reasons) < 2:
            continue  # need at least 2 indicators of potential

        candidates.append(q)

    print(f"[opp] 初筛后 {len(candidates)}只 (需满足≥2个信号)", flush=True)

    if not candidates:
        print("[opp] 无候选股票", flush=True)
        return []

    # === Pass 3: Fetch K-lines concurrently ===
    results = []
    symbols_to_fetch = [(q["symbol"], q) for q in candidates]

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        future_map = {}
        for sym, q in symbols_to_fetch:
            future = executor.submit(fetch_kline, sym, 160, False)
            future_map[future] = (sym, q)

        done = 0
        for future in as_completed(future_map):
            sym, q = future_map[future]
            done += 1
            try:
                bars = future.result()
                if bars is None:
                    continue

                # 获取基本面数据 (静默跳过失败)
                fin_summary: Optional[dict] = None
                try:
                    fin_summary = get_financial_summary(sym, use_cache=True)
                except Exception as e:
                    logging.getLogger(__name__).error(f"[opportunity] 操作失败: {e}", exc_info=True)

                ind = compute_indicators(bars, fin_data=fin_summary)
                if not ind:
                    continue

                # P2-P2-Q10-M027-fix: S10 "PB<1.5" 此前仅初筛使用未入信号体系;
                # 在扫描层注入 signals/reasons (compute_indicators 无 pb 入参),
                # 并重算加权分, 使其真正计入 min_score 过滤。
                sig = dict(ind.get("signals", {}))
                sig_reasons = dict(ind.get("signal_reasons", {}))
                pb_now = q.get("pb", 0)
                if 0 < pb_now < 1.5:
                    sig["low_pb"] = 1.0
                    sig_reasons["low_pb"] = f"PB={pb_now:.2f}"
                final_score = _weighted_score(sig)
                if final_score < min_score:
                    continue
                signal_labels = [sig_reasons.get(k, k) for k, v in sig.items() if v > 0]
                signal_summary = " | ".join(signal_labels[:6])

                # 因子复合评分（factor_zoo）—— 延后到候选集批量截面计算（见排序前）
                ind["factor_score"] = 0.0

                # 获取行业
                sector = _get_stock_sector(sym)
                # 五视角评估(2026-08-14): 归类信号到 防御/短线/价值/资金/动量,
                # 输出主导视角+置信度+一句话侧重, 供飞书/前端分类展示
                try:
                    from quant_system.opportunity_angles import evaluate_angles
                    angle_res = evaluate_angles(
                        sig, sig_reasons,
                        extra={
                            "pe_ttm": q.get("pe_ttm", 0),
                            "pb": pb_now,
                            "roe": ind.get("roe"),
                            "vol_ratio": ind.get("vol_ratio", 0),
                            "factor_score": 0.0,  # 截面后回填
                            "price": ind["price"],
                            "ma60": ind.get("ma60", 0),
                        })
                except Exception:
                    angle_res = None
                results.append({
                    "symbol": sym,
                    "name": q["name"],
                    "price": ind["price"],
                    "change_pct": q.get("change_pct", 0),
                    "market_cap_yi": q.get("market_cap_yi", 0),
                    "pe_ttm": q.get("pe_ttm", 0),
                    "pb": pb_now,
                    "tier": _cap_tier(q.get("market_cap_yi", 0)),
                    # 五视角评估元数据 (2026-08-14 新增)
                    "angles": angle_res.get("angles") if angle_res else None,
                    "dominant_angle": angle_res.get("dominant_angle") if angle_res else None,
                    "dominant_label": angle_res.get("dominant_label") if angle_res else None,
                    "angle_summary": angle_res.get("summary") if angle_res else "",
                    "sector": sector,
                    "signal_count": final_score,
                    "signal_summary": signal_summary,
                    "signal_reasons": sig_reasons,
                    "rsi": ind.get("rsi", 0),
                    "cci": ind.get("cci", 0),
                    "macd_hist": ind.get("macd_hist", 0),
                    "ma20": ind.get("ma20", 0),
                    "ma60": ind.get("ma60", 0),
                    "ma144": ind.get("ma144", 0),
                    "boll_lower": ind.get("boll_lower", 0),
                    "vol_ratio": ind.get("vol_ratio", 0),
                    "signals": sig,
                    "limit_state": _limit_state(q),
                    "trend": ind.get("trend", "side"),
                    "fib_382": ind.get("fib_382"),
                    "fib_618": ind.get("fib_618"),
                    "last_swing_low": ind.get("last_swing_low"),
                    "last_swing_high": ind.get("last_swing_high"),
                    "ma60_slope": ind.get("ma60_slope", 0),
                    "ma144_slope": ind.get("ma144_slope", 0),
                    "factor_score": ind.get("factor_score", 0.0),
                    # 基本面数据
                    "roe": ind.get("roe"),
                    "profit_growth": ind.get("profit_growth"),
                    "revenue_growth": ind.get("revenue_growth"),
                    "debt_ratio": ind.get("debt_ratio"),
                    "gross_margin": ind.get("gross_margin"),
                })
            except Exception as e:
                logging.getLogger(__name__).error(f"[opportunity] 操作失败: {e}", exc_info=True)

            if done % 20 == 0 or done == len(symbols_to_fetch):
                elapsed = _time.time() - t0
                ok, fail, deg_pct = _kline_stats()
                print(f"[opp] K线 {done}/{len(symbols_to_fetch)} ({elapsed:.0f}s) {len(results)}个机会"
                      f" | 成功 {ok}/失败 {fail} (降级 {deg_pct:.0f}%)", flush=True)

    # 因子复合评分（factor_zoo）—— 批量截面归一化到 0-10。
    # P1-Q10-fix: 原逐股 `_get_factor_composite_score` 三重必挂（list 查找/签名错误/
    # 单标的截面归一化归零），factor_score 恒 0。改为对全部候选批量截面计算。
    fs_map = _get_factor_scores([r["symbol"] for r in results])
    for r in results:
        if r["symbol"] in fs_map:
            r["factor_score"] = fs_map[r["symbol"]]

    # Rank
    results.sort(key=lambda r: (-r["signal_count"], r.get("rsi", 50)))

    if top_n > 0:
        results = results[:top_n]

    elapsed = _time.time() - t0
    print(f"[opp] 完成 {elapsed:.0f}s — {len(results)}个买入机会", flush=True)

    return results


_HOLDINGS = None
_HOLDINGS_REFRESH = 0

def _load_holdings() -> dict:
    """Load current holdings for portfolio-aware display."""
    global _HOLDINGS, _HOLDINGS_REFRESH
    now = _time.time()
    if _HOLDINGS is not None and now - _HOLDINGS_REFRESH < 60:
        return _HOLDINGS
    try:
        from quant_system.trade_db import get_positions
        positions = get_positions()
        _HOLDINGS = {p["symbol"]: p for p in positions}
        _HOLDINGS_REFRESH = now
    except Exception:
        _HOLDINGS = {}
    return _HOLDINGS


def format_opportunities(opps: list[dict], brief: bool = False) -> str:
    """Format opportunity list for display."""
    if not opps:
        return "当前没有符合条件的买入机会"

    lines = []
    now = datetime.now(CST)
    h = now.hour
    session = "盘中" if (9 <= h < 11 or 13 <= h < 15) else ("盘前" if h < 9 else "盘后")
    lines.append(f"**买入机会扫描** — {now.strftime('%Y-%m-%d %H:%M')} {session}")
    lines.append(f"发现 {len(opps)} 个机会 (信号分≥2, 满分10)")
    lines.append("")

    if brief:
        # One-liner per opportunity
        for r in opps[:10]:
            name = r["name"]
            # P2-P2-Q10-L039-fix: score_dots 计算后从未使用(且本身触发 C5 崩溃), 删除
            # 基本面摘要
            fin_parts = []
            roe = r.get('roe')
            if roe is not None and roe > 0:
                fin_parts.append(f"ROE={roe:.1f}%")
            gm = r.get('gross_margin')
            if gm is not None and gm > 0:
                fin_parts.append(f"毛利{gm:.0f}%")
            dr = r.get('debt_ratio')
            if dr is not None and dr > 0:
                fin_parts.append(f"负债{dr:.0f}%")
            fin_brief = " | ".join(fin_parts) if fin_parts else ""
            # P2-P2-Q10-M023-fix: brief 模式同样标注近涨跌停
            _ltag = {"near_limit_up": "近涨停", "near_limit_down": "近跌停"}.get(r.get("limit_state", ""), "")
            lines.append(
                f"  {r['symbol']} {name:<8} ¥{r['price']:<8.2f} {r['change_pct']:>+6.2f}% {_ltag}"
                f"  | 得分 {r['signal_count']}/10"
                f"  | RSI={r['rsi']} CCI={r['cci']:.0f}"
                f"  | {r['market_cap_yi']:>5.0f}亿  {fin_brief}"
            )
            # Portfolio awareness
            holdings = _load_holdings()
            if r['symbol'] in holdings:
                h = holdings[r['symbol']]
                hp = h.get('pnl_pct')
                if hp is not None:
                    lines.append(f"    已持仓{h['shares']}股 {'+'+str(round(hp,2))+'%' if hp>=0 else '-'+str(round(hp,2))+'%'}")
                else:
                    lines.append(f"    已持仓{h['shares']}股")
            lines.append(f"    信号: {r['signal_summary']}")
        return "\n".join(lines)

    # Full version
    for r in opps:
        name = r["name"]
        signals = r.get("signals", {})
        score = int(r["signal_count"])

        # P2-P2-Q10-M027-fix: 键名对齐 compute_indicators 实际输出
        # (原 near_ma60/volume_confirmation/today_up/near_boll_lower 等键不存在,
        #  全量版永远显示空标签; today_up 信号现已在 compute_indicators 实现,
        #  low_pb 在扫描层注入)
        signal_labels = []
        if signals.get("rsi_oversold"): signal_labels.append("RSI超卖")
        if signals.get("cci_oversold"): signal_labels.append("CCI超卖")
        if signals.get("ma60_support"): signal_labels.append("MA60支撑")
        if signals.get("ma144_support"): signal_labels.append("MA144支撑")
        if signals.get("macd_bullish_cross"): signal_labels.append("MACD金叉")
        if signals.get("bullish_divergence"): signal_labels.append("底背离")
        if signals.get("volume_price_up"): signal_labels.append("放量上涨")
        if signals.get("today_up"): signal_labels.append("今日收涨")
        if signals.get("boll_lower_band") or signals.get("boll_near_lower"):
            signal_labels.append("BOLL下轨")
        if signals.get("rsi_turning_up"): signal_labels.append("RSI回升")
        if signals.get("low_pb"): signal_labels.append("PB<1.5")

        tier = r["tier"]
        risk_line = []
        if r["pb"] < 1.5:
            risk_line.append(f"PB={r['pb']:.2f}(低估)")
        if r["pe_ttm"] < 15 and r["pe_ttm"] > 0:
            risk_line.append(f"PE={r['pe_ttm']:.1f}(低估)")
        risk_str = " | ".join(risk_line) if risk_line else ""

        lines.append(f"")
        lines.append(f"  {'='*55}")
        # P2-P2-Q10-M023-fix: 结果标注涨跌停状态(近涨停/近跌停), 已封板的在初筛被剔除
        limit_tag = {"limit_up": "涨停", "limit_down": "跌停",
                     "near_limit_up": "近涨停", "near_limit_down": "近跌停"}.get(r.get("limit_state", ""), "")
        lines.append(f"  {r['symbol']} {name:<8}  {tier}  {limit_tag}")
        lines.append(f"  {'='*55}")
        lines.append(f"  现价: ¥{r['price']:<8.2f}  {r['change_pct']:>+.2f}%  |  {risk_str}")
        lines.append(f"  技术: RSI={r['rsi']}  CCI={r['cci']:.0f}  MACD柱={r['macd_hist']:.3f}")
        lines.append(f"  均线: MA20={r['ma20']}  MA60={r['ma60']}  MA144={r['ma144']}")
        if r["boll_lower"]:
            # P2-P2-Q10-L040-fix: 外层 if 已保证 >0, 内层三元冗余; 且原 f-string 未替换 r['price']
            lines.append(f"  BOLL: 下轨={r['boll_lower']}  现价/下轨={r['price']/r['boll_lower']:.2%}")
        lines.append(f"  量比: {r['vol_ratio']:.1f}x")

        # 基本面信号
        fin_parts = []
        roe = r.get('roe')
        if roe is not None and roe > 0:
            fin_parts.append(f"ROE={roe:.1f}%")
        pg = r.get('profit_growth')
        if pg is not None and pg > 0:
            fin_parts.append(f"利润增长{pg:.1f}%")
        gm = r.get('gross_margin')
        if gm is not None and gm > 0:
            fin_parts.append(f"毛利率{gm:.1f}%")
        dr = r.get('debt_ratio')
        if dr is not None and dr > 0:
            fin_parts.append(f"负债率{dr:.1f}%")
        if fin_parts:
            lines.append(f"  基本面: {' | '.join(fin_parts)}")

        lines.append(f"  信号: {' | '.join(signal_labels)}")
        lines.append(f"  得分: {'+'*score}{'-'*(10-score)} {score}/10")

        # Portfolio awareness
        holdings = _load_holdings()
        if r['symbol'] in holdings:
            h = holdings[r['symbol']]
            hp = h['pnl_pct'] if h.get('pnl_pct') is not None else None
            if hp is not None and hp >= 0:
                lines.append(f"  已持仓: {h['shares']}股 成本{h['cost_price']:.2f} "
                             f"盈亏+{hp:.2f}%")
            elif hp is not None:
                lines.append(f"  已持仓: {h['shares']}股 成本{h['cost_price']:.2f} "
                             f"盈亏{hp:.2f}%")
            else:
                lines.append(f"  已持仓: {h['shares']}股 成本{h['cost_price']:.2f} ")

    lines.append(f"\n{'='*55}")

    # 行业聚类
    sectors = Counter(r.get('sector', '') for r in opps if r.get('sector'))
    if sectors:
        sec_lines = []
        for sec, cnt in sectors.most_common(8):
            sec_lines.append(f"{sec}({cnt})")
        lines.append(f"行业分布: {' | '.join(sec_lines)}")

    # 风格分布
    tiers = Counter(r.get('tier', '') for r in opps)
    tier_parts = [f"{t}({c})" for t, c in tiers.most_common()]
    if tier_parts:
        lines.append(f"市值风格: {' | '.join(tier_parts)}")

    lines.append(f"提示: --deep 代码 查看完整决策")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# 3. CLI
# ════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════
# 5. 实时盘中扫描（快速版, 仅腾讯报价 + 缓存日线指标）
# ════════════════════════════════════════════════════════════════

# Cache for daily indicators (pre-computed, reused across real-time scans)
_DAILY_INDICATORS_CACHE: dict[str, tuple[float, dict]] = {}  # symbol -> (timestamp, indicators)
_DAILY_CACHE_TTL = 600  # 10 min refresh


def _load_daily_indicators(symbol: str, force: bool = False) -> dict | None:
    """Load and cache daily indicators for a symbol (含基本面)."""
    now = _time.time()
    if not force and symbol in _DAILY_INDICATORS_CACHE:
        ts, ind = _DAILY_INDICATORS_CACHE[symbol]
        if now - ts < _DAILY_CACHE_TTL:
            return ind
    bars = fetch_kline(symbol, days=160, force=force)
    if bars is None:
        return None
    # 加载基本面 (静默跳过)
    fin_data = None
    try:
        fin_data = get_financial_summary(symbol, use_cache=True)
    except Exception as e:
        logging.getLogger(__name__).error(f"[opportunity] 操作失败: {e}", exc_info=True)
    ind = compute_indicators(bars, fin_data=fin_data)
    if ind:
        _DAILY_INDICATORS_CACHE[symbol] = (now, ind)
    return ind


def _preload_daily_indicators(symbols: list[str]):
    """Preload daily indicators in parallel for a list of symbols."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        futs = {ex.submit(_load_daily_indicators, sym, True): sym for sym in symbols}
        for i, f in enumerate(as_completed(futs)):
            if (i + 1) % 100 == 0:
                print(f"[opp-rt] 预热K线 {i+1}/{len(symbols)}", flush=True)
            try:
                f.result()
            except Exception as e:
                logging.getLogger(__name__).error(f"[opportunity] 操作失败: {e}", exc_info=True)


def scan_real_time(min_score: float = 1.5, preload: bool = True,
                   quotes: list[dict] | None = None) -> dict:
    """
    V3 实时盘中扫描 — 统一加权评分体系。

    策略:
      1. 缓存日线指标 (10min TTL via _load_daily_indicators)
      2. 实时报价叠加实时信号 (今日跌幅、异动)
      3. 基本面数据 (小时级缓存)
      4. 加权评分与 scan_opportunities 统一

    Args:
        min_score: 最低信号分
        preload: 是否预热日线缓存
        quotes: 复用调用方已拉取的行情(如 intraday_monitor 每轮行情),
                避免 Tier-3 每轮二次全量拉取 409 只。

    返回: {results: [...], summary: {...}}
    """
    t0 = _time.time()
    # P2-P2-Q10-M031-fix: 支持复用调用方行情; None 时才自行拉取
    if quotes is None:
        quotes = fetch_quotes(force=True)
    valid = [q for q in quotes if q.get("price", 0) > 0]

    # Pre-cache daily indicators on first run
    symbols_to_preload = [q["symbol"] for q in valid]
    uncached = [s for s in symbols_to_preload if s not in _DAILY_INDICATORS_CACHE]
    if preload and uncached and len(uncached) > 50:
        # P2-P2-Q10-M031-fix: 预热改后台线程, 原同步预热 409 根 K 线(分钟级)阻塞监控循环
        print(f"[opp-rt] 后台预热 {len(uncached)}只K线...", flush=True)
        def _bg_preload():
            try:
                _preload_daily_indicators(uncached)
                print(f"[opp-rt] 预热完成", flush=True)
            except Exception as exc:
                print(f"[opp-rt] 预热失败: {exc}", flush=True)
        threading.Thread(target=_bg_preload, daemon=True).start()

    # 基本面缓存 (小时级)
    if not hasattr(scan_real_time, "_fin_cache"):
        scan_real_time._fin_cache = {}  # symbol -> (timestamp, dict)
    fin_cache = scan_real_time._fin_cache
    fin_ttl = 3600  # 1 hour

    results = []
    for q in valid:
        sym = q["symbol"]
        price = q["price"]
        pct = q.get("change_pct", 0)
        pe = q.get("pe_ttm", 0)
        pb = q.get("pb", 0)
        high52 = q.get("high_52w", 0)
        low52 = q.get("low_52w", 0)
        mv = q.get("market_cap_yi", 0)

        # Quick filters
        if price < 3 or pb <= 0 or pe > 80:
            continue
        if high52 > 0 and price / high52 > 0.93:
            continue
        # P2-P2-Q10-M023-fix: 涨跌停剔除 —— 跌停无法成交、涨停买不进
        if _limit_state(q) in ("limit_up", "limit_down"):
            continue

        # Get cached daily indicators (含基本面如果已缓存)
        ind = _load_daily_indicators(sym)
        if not ind:
            continue

        # ── 基本面 (小时级缓存后注入) ──
        now_ts = _time.time()
        needs_fin_refresh = (sym not in fin_cache or
                             now_ts - fin_cache[sym][0] > fin_ttl)
        fin_data_available = False
        if needs_fin_refresh:
            try:
                fin_summary = get_financial_summary(sym, use_cache=True)
                if fin_summary and any(v for k, v in fin_summary.items() if v is not None):
                    fin_cache[sym] = (now_ts, fin_summary)
                    fin_data_available = True
            except Exception as e:
                logging.getLogger(__name__).error(f"[opportunity] 操作失败: {e}", exc_info=True)
        else:
            if fin_cache.get(sym) and fin_cache[sym][1]:
                fin_data_available = True

        # 重新计算含基本面的评分
        if fin_data_available:
            ind = compute_indicators(
                _KLINE_CACHE.get(sym, (0, None))[1],
                fin_data=fin_cache[sym][1]
            ) or ind

        # ── 实时信号 (叠加在日线评分之上) ──
        real_signals: dict[str, float] = {}
        real_reasons: dict[str, str] = {}

        # RT1: 今日跌幅较大 (回调买入机会)
        if pct < -2:
            real_signals["rt_today_drop"] = 0.8
            real_reasons["rt_today_drop"] = f"今日跌{pct:.1f}%"
        elif pct < -1:
            real_signals["rt_today_slight_drop"] = 0.4
            real_reasons["rt_today_slight_drop"] = f"今日跌{pct:.1f}%"

        # RT2: 接近52周低位
        if low52 > 0 and price / low52 < 1.08:
            real_signals["rt_near_52w_low"] = 0.6
            real_reasons["rt_near_52w_low"] = f"距52周低{((price/low52-1)*100):.0f}%"

        # RT3: 换手率活跃
        turnover = q.get("turnover", 0)
        if turnover > 1.0 and pct > -3:
            real_signals["rt_active"] = 0.5
            real_reasons["rt_active"] = f"换手{turnover:.1f}%"

        # Merge real-time signals into daily indicator result
        merged_signals = dict(ind.get("signals", {}))
        merged_signals.update(real_signals)
        merged_reasons = dict(ind.get("signal_reasons", {}))
        merged_reasons.update(real_reasons)

        # P2-P2-Q10-M027-fix: S10 "PB<1.5" 此前仅初筛使用未入信号体系, 注入实时信号
        if 0 < pb < 1.5:
            merged_signals["low_pb"] = 1.0
            merged_reasons["low_pb"] = f"PB={pb:.2f}"

        # Recompute weighted score (与 compute_indicators 内部逻辑一致)
        final_score = _weighted_score(merged_signals)

        if final_score < min_score:
            continue

        signal_labels = []
        for k, v in merged_signals.items():
            if v > 0:
                r = merged_reasons.get(k, k)
                signal_labels.append(r)

        # 五视角评估注入 (2026-08-14 审计G2: 盘中 scan_real_time 未调用视角评估)
        try:
            from quant_system.opportunity_angles import evaluate_angles
            angle_res = evaluate_angles(merged_signals, merged_reasons, extra={
                "pe_ttm": pe, "pb": pb, "roe": ind.get("roe"),
                "vol_ratio": ind.get("vol_ratio", 0),
                "price": price, "ma60": ind.get("ma60", 0),
            })
        except Exception:
            angle_res = None

        results.append({
            "symbol": sym,
            "name": q["name"],
            "price": price,
            "change_pct": pct,
            "market_cap_yi": mv,
            "pe_ttm": pe,
            "pb": pb,
            "tier": _cap_tier(mv),
            "sector": _get_stock_sector(sym),
            "signal_count": final_score,
            "signal_summary": " | ".join(signal_labels[:6]),
            "signal_reasons": merged_reasons,
            "signals": merged_signals,
            # 五视角元数据 (2026-08-14)
            "angles": angle_res.get("angles") if angle_res else None,
            "dominant_angle": angle_res.get("dominant_angle") if angle_res else None,
            "dominant_label": angle_res.get("dominant_label") if angle_res else None,
            "angle_summary": angle_res.get("summary") if angle_res else "",
            "limit_state": _limit_state(q),
            "rsi": ind.get("rsi", 0),
            "cci": ind.get("cci", 0),
            "macd_hist": ind.get("macd_hist", 0),
            "ma60": ind.get("ma60", 0),
            "ma144": ind.get("ma144", 0),
            "boll_lower": ind.get("boll_lower", 0),
            "vol_ratio": ind.get("vol_ratio", 0),
            "roe": ind.get("roe"),
            "profit_growth": ind.get("profit_growth"),
            "gross_margin": ind.get("gross_margin"),
            "debt_ratio": ind.get("debt_ratio"),
        })

    results.sort(key=lambda r: (-r["signal_count"], r.get("rsi", 50)))
    elapsed = _time.time() - t0

    summary = {
        "total_scanned": len(valid),
        "results_count": len(results),
        "elapsed_s": round(elapsed, 1),
        "top_scores": [r["signal_count"] for r in results[:5]],
    }

    # P2-P2-Q10-M031-fix: 预热已改后台线程, 提示语相应调整
    refresh_note = " (后台预热中)" if preload and len(uncached) > 50 else ""
    print(f"[opp-rt] {len(results)}个机会 ({elapsed:.1f}s{refresh_note})", flush=True)
    return {"results": results, "summary": summary}


def scan_hourly_cci(symbols: list[str]) -> dict[str, dict]:
    """
    检测小时级CCI信号（需新浪60分钟K线）。
    用于盘中每20-30分钟批量检测。
    """
    try:
        from quant_system.sources_sina_intraday import fetch_sina_intraday
        results = {}
        for sym in symbols:
            try:
                df = fetch_sina_intraday(sym, scale=60, datalen=30)
                if df is None or len(df) < 5:
                    continue
                closes = df["close"].values.astype(float)
                highs = df["high"].values.astype(float)
                lows = df["low"].values.astype(float)

                # Compute CCI on last 20 hourly bars
                if len(closes) >= 20:
                    tp = [(highs[i] + lows[i] + closes[i]) / 3 for i in range(-20, 0)]
                    mean_tp = sum(tp) / len(tp)
                    md = sum(abs(t - mean_tp) for t in tp) / len(tp)
                    cci = (tp[-1] - mean_tp) / (0.015 * md) if md > 0 else 0

                    # CCI from previous bar
                    tp_prev = [(highs[i] + lows[i] + closes[i]) / 3 for i in range(-21, -1)]
                    if len(tp_prev) >= 20:
                        mean_prev = sum(tp_prev) / len(tp_prev)
                        md_prev = sum(abs(t - mean_prev) for t in tp_prev) / len(tp_prev)
                        cci_prev = (tp_prev[-1] - mean_prev) / (0.015 * md_prev) if md_prev > 0 else 0
                    else:
                        cci_prev = cci

                    # Detect CCI turning up from oversold
                    latest_close = float(df["close"].iloc[-1])
                    results[sym] = {
                        "hourly_cci": round(cci, 1),
                        "hourly_cci_prev": round(cci_prev, 1),
                        "latest_close": latest_close,
                    }
            except Exception as e:
                logging.getLogger(__name__).error(f"[opportunity] 操作失败: {e}", exc_info=True)
        return results
    except ImportError:
        return {}


def format_real_time(opps: list[dict], full: bool = False) -> str:
    """Format real-time scan results."""
    if not opps:
        return "当前没有实时买入机会"

    now = datetime.now(CST)
    lines = [f"**实时机会扫描** — {now.strftime('%H:%M:%S')} | {len(opps)}个机会"]
    lines.append("")

    for r in opps[:20 if full else 10]:
        name = r["name"]
        score = int(r["signal_count"])
        rsi = r.get("rsi", 0)
        ma60 = r.get("ma60", 0)
        near_support = ""
        if ma60 > 0:
            pct_from_ma60 = (r["price"] / ma60 - 1) * 100
            near_support = f" (距MA60 {pct_from_ma60:+.1f}%)"

        arrows = "+" * min(score, 5) + "-" * (5 - min(score, 5))
        lines.append(
            f"  {arrows} {r['symbol']} {name:<8} ¥{r['price']:<8.2f} {r['change_pct']:>+6.2f}%"
            f"  | RSI={rsi:.0f} | {r['market_cap_yi']:>5.0f}亿{near_support}"
        )
        lines.append(f"     信号: {r['signal_summary']}")

    lines.append(f"\n--deep 代码 看完整决策")
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="买入机会扫描器")
    parser.add_argument("--top", type=int, default=30, help="返回前N个机会 (默认30)")
    parser.add_argument("--min-score", type=int, default=2, help="最低信号分 (默认2)")
    parser.add_argument("--brief", action="store_true", help="简要输出")
    parser.add_argument("--deep", type=str, help="某只股票完整决策")
    parser.add_argument("--cache-clear", action="store_true", help="清除K线缓存")
    parser.add_argument("--realtime", action="store_true", help="快速实时扫描(仅报价+缓存)")
    parser.add_argument("--hourly", nargs="*", default=[],
                        help="小时级CCI检测, 例: --hourly 601288 600519")
    parser.add_argument("--sector", type=str, default=None,
                        help="只看某行业 (需已获取行业数据), 例: --sector 银行")
    args = parser.parse_args()

    if args.deep:
        from quant_system.intraday_decision import decide_stock, format_decision
        result = decide_stock(args.deep)
        print(format_decision(result))
        return

    if args.realtime:
        t0 = _time.time()
        opps = scan_real_time(min_score=1)
        print(format_real_time(opps, full=True))
        print(f"\n耗时 {_time.time()-t0:.1f}s")
        return

    if args.hourly is not None and len(args.hourly) > 0:
        symbols = args.hourly
        t0 = _time.time()
        hourly = scan_hourly_cci(symbols)
        print(f"\n小时级CCI ({len(hourly)}只, {_time.time()-t0:.1f}s):")
        for sym, h in sorted(hourly.items(), key=lambda x: x[1].get("hourly_cci", 0)):
            cci = h["hourly_cci"]
            prev = h["hourly_cci_prev"]
            arrow = "+" if cci > prev else "-"
            zone = "超买" if cci > 100 else ("超卖" if cci < -100 else "正常")
            print(f"  {sym}  {arrow} 当前CCI={cci:.1f} 上轮={prev:.1f}  {zone}  ¥{h['latest_close']:.2f}")
        return

    if args.cache_clear:
        global _KLINE_CACHE
        _KLINE_CACHE.clear()
        print("[opp] 缓存已清除", flush=True)
        return

    t0 = _time.time()
    opps = scan_opportunities(top_n=args.top, min_score=args.min_score)
    # P2-P2-Q10-M026-fix: 实现 docstring 承诺的 --sector 过滤(此前无该参数, 行业恒空)
    if args.sector:
        opps = [r for r in opps
                if args.sector in (r.get("sector") or "") or (r.get("sector") or "") in args.sector]
        print(f"[opp] 按行业过滤: {args.sector} → {len(opps)}只", flush=True)
    elapsed = _time.time() - t0

    # Try to get market context
    try:
        from quant_system.market_context import get_market_context
        ctx = get_market_context()
        ctx_summary = ctx.get("_summary", "")
        first_lines = "\n".join(ctx_summary.split("\n")[:3]) if ctx_summary else ""
        if first_lines:
            print(first_lines)
            print()
    except Exception as e:
        logging.getLogger(__name__).error(f"[opportunity] 操作失败: {e}", exc_info=True)

    print(format_opportunities(opps, brief=args.brief))
    print(f"\n总耗时 {elapsed:.0f}s")


if __name__ == "__main__":
    main()
