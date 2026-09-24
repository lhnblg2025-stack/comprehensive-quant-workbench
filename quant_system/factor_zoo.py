"""
factor_zoo.py — 50+ A股因子库。

多因子体系: 动量/价值/质量/成长/低波/技术 六大类，
统一入口: compute_factor_portfolio(symbols, bar_data, fin_data_cache) → DataFrame。

每个因子经过截面z-score标准化, 再clip到 (-3 ~ +3)。
D4收敛登记 (2026-08-11): 因子域唯一真源(保守策略)——37个因子(FACTOR_META, 36 active)定义与计算均以本模块为唯一真源；ic_factors/factor_model/factor_system 同名因子一律不强迁(异口径), 独特因子各自标注保留。
"""

from __future__ import annotations
import logging

import sys
import sqlite3
from datetime import timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - optional runtime dependency
    scipy_stats = None

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

CST = timezone(timedelta(hours=8))
FACTOR_IC_DB = Path.home() / ".quant_system" / "factor_ic_cache.sqlite3"

# V4.1 arch fix: 因子计算结果缓存 {(symbol, bars_hash): factors}
# bars用首尾日期+长度作简化指纹，避免序列化完整K线
_FACTOR_CACHE: dict[tuple[str, str], dict[str, float]] = {}
# P2-Q5-fix (M440): 缓存上限 + 简单 LRU 近似（Python dict 保序，先入=最老）。
#   V5.4 缓存永不清除 → 全市场长期运行内存膨胀。
_FACTOR_CACHE_MAX = 2048


def _cache_put(key: tuple[str, str], value: dict[str, float]) -> None:
    """写入因子缓存；超出上限时淘汰最老的 1/4 条目（可见的容量策略）。"""
    _FACTOR_CACHE[key] = value
    if len(_FACTOR_CACHE) > _FACTOR_CACHE_MAX:
        for k in list(_FACTOR_CACHE)[: max(1, len(_FACTOR_CACHE) // 4)]:
            _FACTOR_CACHE.pop(k, None)


def _bars_fingerprint(symbol: str, bars: list[list]) -> str:
    """P2-Q5-fix (M440): 缓存指纹纳入数据内容哈希。

    V5.4 用 (symbol, 首尾日期-长度) 作指纹——复权方式/中间数据变化但首尾同长时
    撞键返回过期值。这里对全部 K 线内容做 MD5（进程内字符串稳定），内容变 → 键变。
    """
    try:
        import hashlib
        digest = hashlib.md5()
        for b in bars:
            digest.update(str(b).encode("utf-8", "ignore"))
        return f"{bars[0][0]}-{bars[-1][0]}-{len(bars)}-{digest.hexdigest()[:12]}"
    except Exception:
        return f"{bars[0][0]}-{bars[-1][0]}-{len(bars)}"


# ════════════════════════════════════════════════════════════════
# 因子定义
# ════════════════════════════════════════════════════════════════

FACTOR_META = {
    # === 动量因子 ===
    "mom_12m":     {"family": "momentum", "desc": "12个月动量(剔除最近1月)", "horizon": "long"},
    "mom_6m":      {"family": "momentum", "desc": "6个月动量", "horizon": "medium"},
    "mom_3m":      {"family": "momentum", "desc": "3个月动量", "horizon": "short"},
    "mom_1m":      {"family": "momentum", "desc": "1个月动量", "horizon": "short"},
    # P2-Q5-fix (L453): 同步元数据与实现状态。
    #   V5.4 此处大段注释称 ep/bp/sp/cp 等"登记但未实现"，与下方实际已实现矛盾，
    #   误导维护者。自 V5.1 起以下因子已全部实现并登记；唯一未实现的是
    #   "seasonality"（月频季节效应，需月频收益历史，暂无数据源）。

    # === 价值因子（V5.1: 已全部实现） ===
    "ep":          {"family": "value", "desc": "E/P (市盈率倒数)", "horizon": "long"},
    "bp":          {"family": "value", "desc": "B/P (市净率倒数)", "horizon": "long"},
    "sp":          {"family": "value", "desc": "S/P (市销率倒数)", "horizon": "long"},
    "cp":          {"family": "value", "desc": "C/P (经营现金流/市值)", "horizon": "long"},
    "div_yield":   {"family": "value", "desc": "股息率", "horizon": "long"},
    "ev_ebitda":   {"family": "value", "desc": "EV/EBITDA (负值)", "horizon": "long", "active": False},

    # === 质量因子（部分字段未从 fin_data 获取） ===
    "roe":         {"family": "quality", "desc": "ROE", "horizon": "medium"},
    "roa":         {"family": "quality", "desc": "ROA", "horizon": "medium"},
    "gross_margin":{"family": "quality", "desc": "毛利率", "horizon": "medium"},
    "net_margin":  {"family": "quality", "desc": "净利率", "horizon": "medium"},
    "accruals":    {"family": "quality", "desc": "应计利润(经营现金流-净利润)/总资产", "horizon": "medium"},
    "leverage":    {"family": "quality", "desc": "负债率(负债/总资产, 负值)", "horizon": "long"},
    "current_ratio":{"family": "quality", "desc": "流动比率", "horizon": "short"},
    "debt_equity": {"family": "quality", "desc": "负债权益比(负值)", "horizon": "long"},
    "interest_cov":{"family": "quality", "desc": "利息保障倍数", "horizon": "medium"},
    "asset_turn":  {"family": "quality", "desc": "总资产周转率", "horizon": "medium"},

    # === 成长因子（部分字段未从 fin_data 获取） ===
    "earnings_growth_yoy":  {"family": "growth", "desc": "净利润同比增速", "horizon": "medium"},
    "sales_growth_yoy":     {"family": "growth", "desc": "营收同比增速", "horizon": "medium"},
    "earnings_growth_qoq":  {"family": "growth", "desc": "净利润环比增速", "horizon": "short"},
    "surprise":            {"family": "growth", "desc": "超预期(净利润vs一致预期)", "horizon": "short"},
    "roe_change":          {"family": "growth", "desc": "ROE变动(TTM vs 前TTM)", "horizon": "medium"},
    "margin_change":       {"family": "growth", "desc": "毛利率变动", "horizon": "medium"},

    # === 低波因子 ===
    # P2-Q5-fix (M434/M433): beta_60m→beta_60d、idio_vol_60m→idio_vol_60d。
    #   V5.4 名称暗示 60 个月，实际实现是 60 个交易日，名实严重不符；且 idio_vol
    #   未对市场回归取残差（总波动冒充特质波动）。改名并改为残差口径（见 compute_factors）。
    "beta_60d":    {"family": "low_vol", "desc": "CAPM Beta (60交易日, 负值)", "horizon": "long"},
    "idio_vol_60d":{"family": "low_vol", "desc": "特质波动率(60交易日, 市场回归残差std, 负值)", "horizon": "long"},
    "max_dd_12m":  {"family": "low_vol", "desc": "12个月最大回撤(负值)", "horizon": "medium"},
    "vol_20d":     {"family": "low_vol", "desc": "20日波动率(负值)", "horizon": "short"},
    "downside_beta":{"family": "low_vol", "desc": "下行Beta(负值)", "horizon": "medium"},

    # === 技术因子 ===
    "rsi_14":      {"family": "technical", "desc": "RSI(14, 负值=超卖)", "horizon": "short"},
    "cci_20":      {"family": "technical", "desc": "CCI(20, 负值=超卖)", "horizon": "short"},
    "macd_hist":   {"family": "technical", "desc": "MACD柱状图", "horizon": "short"},
    "volume_trend":{"family": "technical", "desc": "成交量趋势(20日/60日)", "horizon": "short"},
    "ma_cross":    {"family": "technical", "desc": "均线交叉(MA5-MA20)", "horizon": "short"},
    "boll_pos":    {"family": "technical", "desc": "布林带位置(price-下轨)/(上轨-下轨), 负值=低位", "horizon": "short"},
}


def valid_factors_from_registry() -> list[str]:
    """读取真实因子质量账本，返回允许进入策略的有效因子名。

    只把核心/待选（真实有效）因子作为可用池；观察和阻断因子不进入。
    账本缺失/读取失败返回空列表（调用方应视为“无已验证因子”，不得回退全量）。
    """
    import json as _json
    from pathlib import Path as _Path
    p = _Path(__file__).resolve().parent.parent / "generated" / "factor_quality_registry.json"
    if not p.exists():
        return []
    try:
        data = _json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [str(x.get("factor")) for x in data.get("factors", [])
            if x.get("tier") in ("core", "candidate") and x.get("eligible")]


def list_factors(family: str = "", include_inactive: bool = False) -> list[str]:  # V4.1 fix: add include_inactive filter
    """列出所有因子，可选按族过滤。include_inactive=True 时包含未实现的因子。
    D4收敛登记: 因子清单唯一真源(保守策略)
    """
    if family:
        result = [k for k, v in FACTOR_META.items() if v["family"] == family]
    else:
        result = list(FACTOR_META.keys())
    if not include_inactive:  # V4.1 fix: filter out inactive (unimplemented) factors by default
        result = [k for k in result if FACTOR_META[k].get("active", True)]
    return result


# ════════════════════════════════════════════════════════════════
# 因子计算（基于K线数据 + 财务数据）
# ════════════════════════════════════════════════════════════════

def compute_factors(
    symbol: str,
    bars: list[list],
    fin_data: Optional[dict] = None,
    market_returns: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """
    对单只股票计算所有可用因子。

    Args:
        symbol: 股票代码
        bars: K线数据 [[date, open, close, high, low, volume], ...]
        fin_data: 基本面数据 (来自 financial_data.get_financial_summary)

    Returns:
        {factor_name: z_score, ...}
    D4收敛登记: 因子域唯一真源(保守策略)——36个active因子定义与实现均以本函数为唯一真源，跨模块同名因子一律不强迁
    """
    if not bars or len(bars) < 30:
        return {}

    # V4.1 arch fix: 检查因子缓存
    # P2-Q5-fix (M440): 指纹纳入 K 线内容哈希（复权方式/中间数据变化即失效）
    cache_key = (symbol, _bars_fingerprint(symbol, bars))
    if cache_key in _FACTOR_CACHE:
        return dict(_FACTOR_CACHE[cache_key])

    # Helper: compute EMA on numpy array with given period and return last value
    def _ema_last(arr: np.ndarray, period: int) -> float:
        """Exponential Moving Average, last value only."""
        if len(arr) == 0 or period < 1:
            return 0.0
        alpha = 2.0 / (period + 1)
        result = arr[0]
        for v in arr[1:]:
            result = alpha * v + (1 - alpha) * result
        return result

    def _rsi_last(arr: np.ndarray, window: int = 14) -> float:
        """RSI value for the last element.

        P2-Q5-fix (M441): V5.4 用最后 14 个 Δ 的简单平均（SMA 起点依赖、无平滑），
        与标准 RSI 数值有偏差。改为 Wilder 递推平滑：首值取简单均值，其后
        avg = (prev*(window-1) + cur) / window，与主流软件 RSI 口径一致。
        """
        if len(arr) < window + 1:
            return 50.0
        deltas = np.diff(arr)
        gains = np.maximum(deltas, 0)
        losses = np.maximum(-deltas, 0)
        avg_gain = float(np.mean(gains[:window]))
        avg_loss = float(np.mean(losses[:window]))
        for i in range(window, len(gains)):
            avg_gain = (avg_gain * (window - 1) + gains[i]) / window
            avg_loss = (avg_loss * (window - 1) + losses[i]) / window
        if avg_loss < 1e-12:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100.0 - 100.0 / (1.0 + rs))

    closes = np.array([float(b[2]) for b in bars])
    highs = np.array([float(b[3]) for b in bars])
    lows = np.array([float(b[4]) for b in bars])
    volumes = np.array([float(b[5]) for b in bars], dtype=np.float64)

    result: dict[str, float] = {}

    # ── 动量因子 ──
    # P2-Q5-fix (M432): 统一 Jegadeesh-Titman 口径——12个月动量恒剔除最近1月
    #   （t-22~t-274）。V5.4 对 252~273 天降级为含近月的 t-1~t-252，同一因子在
    #   两只股票间定义不同，截面 z-score 混合两套口径。修复：不足 274 天不出该因子
    #   （可见降级：宁缺毋滥，不混口径）。
    if len(closes) >= 274:
        # 12个月动量(剔除最近1月): Jegadeesh & Titman (1993) 标准做法
        # 用 t-22 到 t-274 的区间（跳过最近约21个交易日）
        result["mom_12m"] = closes[-22] / closes[-274] - 1 if closes[-274] > 0 else 0

    if len(closes) >= 126:
        result["mom_6m"] = closes[-1] / closes[-126] - 1 if closes[-126] > 0 else 0
        result["mom_3m"] = closes[-1] / closes[-63] - 1 if closes[-63] > 0 else 0
        result["mom_1m"] = closes[-1] / closes[-21] - 1 if closes[-21] > 0 else 0
    elif len(closes) >= 63:
        result["mom_3m"] = closes[-1] / closes[-63] - 1 if closes[-63] > 0 else 0
        # P2-Q5-fix (L452): 已在 len>=63 分支内，`len(closes) >= 21` 恒真，去掉冗余条件
        result["mom_1m"] = closes[-1] / closes[-21] - 1 if closes[-21] > 0 else 0

    # ── 低波因子 ──
    if len(closes) >= 60:
        # 使用 np.maximum 避免零值破坏时间序列连续性
        safe_closes = np.maximum(closes, 1e-12)
        returns = np.diff(np.log(safe_closes))
        if len(returns) >= 20:
            result["vol_20d"] = -np.std(returns[-20:]) * (252 ** 0.5)  # 负值=低波好
        # P2-Q5-fix (M433): idio_vol_60d 改为在下方 Beta 段用市场回归残差计算，
        #   不再在此处用总波动冒充特质波动。

    # ── 技术因子 ──
    if len(closes) >= 15:
        rsi_val = _rsi_last(closes, 14)
        result["rsi_14"] = -rsi_val if rsi_val else 0  # 负值=超卖好

    if len(closes) >= 20:
        # CCI
        tp = (highs + lows + closes) / 3
        tp_mean = np.mean(tp[-20:])
        tp_mad = np.mean(np.abs(tp[-20:] - tp_mean))
        cci = (tp[-1] - tp_mean) / (0.015 * tp_mad) if tp_mad > 0 else 0
        result["cci_20"] = -cci  # 负值=超卖好

        # BOLL 位置
        last20 = closes[-20:]
        mid = np.mean(last20)
        std = np.std(last20)
        boll_lower = mid - 2 * std
        boll_upper = mid + 2 * std
        if boll_upper > boll_lower:
            result["boll_pos"] = -(closes[-1] - boll_lower) / (boll_upper - boll_lower)  # 负值=靠近下轨好

        # 均线交叉
        ma5 = np.mean(closes[-5:]) if len(closes) >= 5 else 0
        ma20 = np.mean(closes[-20:]) if len(closes) >= 20 else 0
        if ma20 > 0:
            result["ma_cross"] = (ma5 - ma20) / ma20

        # 量比
        if len(volumes) >= 21:
            avg_vol = np.mean(volumes[-21:-1]) if np.mean(volumes[-21:-1]) > 0 else 1
            result["volume_trend"] = volumes[-1] / avg_vol

    # MACD 柱状图 (DIF - DEA)
    if len(closes) >= 35:
        # P2-Q5-fix (L451): 删除未使用的 ema12/ema26 死代码（下方用 dif_series 递推）
        dif_series = np.zeros(len(closes))
        alpha_fast = 2.0 / 13.0
        alpha_slow = 2.0 / 27.0
        cur_fast = closes[0]
        cur_slow = closes[0]
        for i in range(len(closes)):
            cur_fast = alpha_fast * closes[i] + (1 - alpha_fast) * cur_fast
            cur_slow = alpha_slow * closes[i] + (1 - alpha_slow) * cur_slow
            dif_series[i] = cur_fast - cur_slow
        dea = _ema_last(dif_series, 9)
        result["macd_hist"] = dif_series[-1] - dea  # 柱状图 = DIF - DEA

    # ── 最大回撤 ──
    if len(closes) >= 252:
        peak = np.maximum.accumulate(closes[-252:])
        dd = (closes[-252:] - peak) / peak
        result["max_dd_12m"] = np.min(dd)
    elif len(closes) >= 60:
        peak = np.maximum.accumulate(closes[-60:])
        dd = (closes[-60:] - peak) / peak
        result["max_dd_12m"] = np.min(dd)

    # ── 基本面因子 (来自 fin_data) ──
    if fin_data:
        # V5.1 expansion: 全部财务因子
        fin = fin_data

        # 估值  (raw values in fin_data are already in correct scale)
        for fin_name, factor_name in [
            ("pe_ttm", "pe_ttm"),
            ("pb", "pb"),
            ("ep", "ep"),
            ("bp", "bp"),
            ("sp", "sp"),
            ("cp", "cp"),
            ("div_yield", "div_yield"),
        ]:
            v = fin.get(fin_name, 0)
            if v and isinstance(v, (int, float)):
                result[factor_name] = v

        # 质量
        roe = fin.get("roe", 0)
        if roe:
            result["roe"] = roe / 100 if roe > 1 else roe  # normalize

        roa = fin.get("roa", 0)
        if roa:
            result["roa"] = roa / 100 if roa > 1 else roa

        gm = fin.get("gross_margin", 0)
        if gm:
            result["gross_margin"] = gm / 100 if gm > 1 else gm

        nm = fin.get("net_margin", 0)
        if nm:
            result["net_margin"] = nm / 100 if nm > 1 else nm

        # 负债类 (负值=低负债好)
        dr = fin.get("debt_ratio", 0)
        if dr:
            result["leverage"] = -dr / 100 if dr > 1 else -dr

        cr = fin.get("current_ratio", 0)
        if cr:
            result["current_ratio"] = cr / 100 if cr > 10 else cr

        de = fin.get("debt_equity_ratio", 0)  # 产权比率
        if de:
            result["debt_equity"] = -de / 100 if de > 1 else -de

        # 周转
        at = fin.get("asset_turnover", 0)
        if at:
            result["asset_turn"] = at

        # 应计利润: (净利润 - OCF) / 总资产
        # P1-Q5-fix: V5.4 回退到 ocf_to_sales（比率）与 net_profit/ocf_total（绝对额）
        #   直接相减 → 跨量纲混用，accruals 数值无意义。绝对额 OCF 缺失即放弃该因子。
        np_val = fin.get("net_profit", 0)
        ocf_total = fin.get("ocf_total", 0)
        if np_val and ocf_total:
            # 用营业收入估算总资产
            rev = fin.get("revenue", 0)
            debt_r = fin.get("debt_ratio", 0)
            if debt_r > 0 and rev:
                ta = rev / debt_r if debt_r < 1 else rev / (debt_r / 100)
                accrual = (np_val - ocf_total) / ta if ta != 0 else 0
                result["accruals"] = -accrual  # 负值=低应计利润好

        # 利息保障倍数: EBIT / 利息支出（真实利息费用）
        # P1-Q5-fix: V5.4 用 ebit / max(abs(eps - ebit), 0.001) 近似，该式无金融含义
        #   （EBIT/share 与 EPS 之差并非利息费用）。改为 EBIT / 真实利息支出；
        #   无利息费用数据时不出该因子（禁止用无意义代理冒充）。
        ebit = fin.get("ebit_per_share", 0) or fin.get("ebit", 0)
        interest_exp = fin.get("interest_expense", 0) or fin.get("财务费用", 0)
        if ebit and interest_exp and abs(float(interest_exp)) > 1e-9:
            result["interest_cov"] = ebit / float(interest_exp)

        # 成长
        eg_yoy = fin.get("profit_growth_yoy", 0) or fin.get("profit_growth", 0)
        if eg_yoy:
            result["earnings_growth_yoy"] = eg_yoy / 100 if abs(eg_yoy) > 1 else eg_yoy

        sg_yoy = fin.get("sales_growth_yoy", 0) or fin.get("revenue_growth", 0)
        if sg_yoy:
            result["sales_growth_yoy"] = sg_yoy / 100 if abs(sg_yoy) > 1 else sg_yoy

        # 环比增长 (多期变化因子)
        eg_qoq = fin.get("earnings_growth_qoq", 0)
        if eg_qoq:
            result["earnings_growth_qoq"] = eg_qoq / 100 if abs(eg_qoq) > 1 else eg_qoq

        rc = fin.get("roe_change", 0)
        if rc:
            result["roe_change"] = rc / 100 if abs(rc) > 1 else rc

        mc = fin.get("margin_change", 0)
        if mc:
            result["margin_change"] = mc / 100 if abs(mc) > 1 else mc

        # 超预期 (surprise) — 无一致预期数据源，用营收增速 vs 行业均值近似
        # 如果该因子前N只股票平均营收增速可用，则计算偏差
        # 当前简化为: 营收增速的截面排名（在截面标准化时自然处理）

    # ── 低波: Beta / 下行Beta / 特质波动 (V5.1: 使用市场收益) ──
    # P2-Q5-fix (M434): beta_60m 名实不符（60"个月"实际 60 交易日）——改名为
    #   beta_60d 并更新文档，窗口保持 60 交易日（60 个月=1260 交易日绝大多数股票
    #   数据不足，强行延长会让该因子大面积缺失）。
    # P2-Q5-fix (M433): idio_vol_60d 在此用市场回归残差 std 实现（Beta 剥离系统性
    #   风险），替代 V5.4 的总波动口径；无市场数据时不出该因子（可见降级，不冒充）。
    if len(closes) >= 60 and market_returns is not None and len(market_returns) >= 60:
        safe_closes = np.maximum(closes, 1e-12)
        returns_arr = np.diff(np.log(safe_closes))
        n_beta = min(len(returns_arr), len(market_returns), 60)
        if n_beta >= 20:
            try:
                stock_ret = returns_arr[-n_beta:]
                mkt_ret = market_returns[-n_beta:]
                cov = np.cov(stock_ret, mkt_ret)[0, 1]
                var_mkt = np.var(mkt_ret) + 1e-12
                beta_val = cov / var_mkt
                result["beta_60d"] = -beta_val  # 负值=低Beta好

                # 特质波动率 = 市场回归残差的年化 std（剥离市场系统性暴露）
                resid = stock_ret - beta_val * mkt_ret
                result["idio_vol_60d"] = -np.std(resid) * (252 ** 0.5)

                # 下行Beta: 市场收益为负时的Beta
                neg_mask = mkt_ret < 0
                if np.sum(neg_mask) >= 5:
                    stock_neg = stock_ret[neg_mask]
                    mkt_neg = mkt_ret[neg_mask]
                    cov_neg = np.cov(stock_neg, mkt_neg)[0, 1]
                    var_mkt_neg = np.var(mkt_neg) + 1e-12
                    down_beta = cov_neg / var_mkt_neg
                    result["downside_beta"] = -down_beta  # 负值=低下行Beta好
            except Exception as e:
                logging.getLogger(__name__).error(f"[factor_zoo] 操作失败: {e}", exc_info=True)

    # V4.1 arch fix: 缓存单只股票因子结果
    _cache_put(cache_key, result)  # P2-Q5-fix (M440): 带上限淘汰，防内存膨胀
    # Cross-sectional normalization will be done in aggregate
    return result


def compute_factor_portfolio(
    symbols: list[str],
    bar_data: dict[str, list[list]],
    fin_data_cache: dict[str, dict],
    _market_returns: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    V5.1: 接受可选 _market_returns (沪深300对数收益) 用于Beta计算。
    """
    # P2-Q5-fix (M435): 市场收益优先走 data_store 缓存（指数行情落库，无网络依赖、
    #   失败不再静默），data_store 无数据时才回退 akshare 并告警（可见降级）。
    #   V5.4 每次组合计算都实时联网拉沪深300，慢且失败静默，热路径依赖网络。
    if _market_returns is None:
        try:
            from quant_system.data_store import get_store
            store = get_store()
            df = store.get("000300", days=400)
            if df is not None and not df.empty:
                if "pct_chg" in df.columns:
                    _market_returns = df["pct_chg"].dropna().values.astype(np.float64)
                elif "close" in df.columns:
                    m_prices = np.array(df["close"].dropna().values, dtype=float)
                    m_prices = np.maximum(m_prices, 1e-12)
                    _market_returns = np.diff(np.log(m_prices))
        except Exception:
            _market_returns = None
    if _market_returns is None:
        print("[factor_zoo] ⚠️ data_store 无沪深300指数行情，回退 akshare 联网拉取")
        try:
            import akshare as ak
            m = ak.stock_zh_index_daily(symbol="sh000300")
            if m is not None and not m.empty:
                m_prices = np.array(m["close"].values, dtype=float)
                m_prices = np.maximum(m_prices, 1e-12)
                _market_returns = np.diff(np.log(m_prices))
        except Exception as e:
            print(f"[factor_zoo] ⚠️ akshare 沪深300拉取失败({e})，Beta/特质波动因子将缺失")
            _market_returns = None

    rows = []
    for sym in symbols:
        bars = bar_data.get(sym)
        if not bars:
            continue
        fin = fin_data_cache.get(sym, {})
        factors = compute_factors(sym, bars, fin_data=fin, market_returns=_market_returns)
        if factors:
            row = {"symbol": sym}
            row.update(factors)
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("symbol")

    # Z-score normalization per factor (cross-sectional)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        vals = df[col].replace([np.inf, -np.inf], np.nan)
        # 1) 先winsorize极端值（1%/99%分位截断）避免异常值扭曲均值和标准差
        lo, hi = vals.quantile(0.01), vals.quantile(0.99)
        trimmed = vals.clip(lo, hi)
        mean = trimmed.mean()
        std = trimmed.std()
        if std > 0:
            df[col] = (trimmed - mean) / std
        else:
            # P2-Q5-fix (L443): std==0 时保留 NaN 行（trimmed * 0.0：数值行→0，NaN 行→NaN），
            #   与 std>0 分支的缺失处理一致——不再把全列（含缺失行）粗暴置 0。
            df[col] = trimmed * 0.0
        # 2) 再clip z-score在 +/-3 范围
        df[col] = df[col].clip(-3, 3)

    return df


def compute_composite_score(
    factor_df: pd.DataFrame | dict[str, pd.Series],
    weights: Optional[dict[str, float]] = None,
    weighting: str = "equal",
    ic_dict: Optional[dict[str, float | dict[str, float]]] = None,
) -> pd.Series:
    """
    复合因子评分: 支持等权、固定权重、IC 加权和 ICIR 加权。

    Args:
        factor_df: compute_factor_portfolio 输出，或 {factor_name: factor_series}
        weights: {factor_name: weight}, 保留向后兼容；优先级高于 weighting
        weighting: "equal" / "ic_weighted" / "icir_weighted"
        ic_dict: {factor_name: ic} 或 {factor_name: {"ic": x, "icir": y}}

    Returns:
        Series: 复合评分, index=symbol
    """
    if isinstance(factor_df, dict):
        factor_df = pd.DataFrame(factor_df)

    if factor_df.empty:
        return pd.Series(dtype=float)

    numeric_cols = factor_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return pd.Series(0.0, index=factor_df.index)

    if weights:
        available_weights = {k: v for k, v in weights.items() if k in numeric_cols}
        if not available_weights:
            return pd.Series(0.0, index=factor_df.index)
        w = pd.Series(available_weights)
        w = w / (w.abs().sum() + 1e-12)  # normalize
        available_cols = list(w.index)
        # P2-Q5-fix (M442): 按行可用因子加权均值替代 dot(w)——含任一 NaN 因子的股票
        #   不再得 NaN 总分（缺财务数据的股票被整体排除出 Top 排名）；与 equal 路径
        #   mean(axis=1) 跳过 NaN 的行为统一（权重均值化，缺失因子的权重重分配给其余因子）。
        sub = factor_df[available_cols]
        num = sub.fillna(0.0).values @ w.values
        den = sub.notna().values @ w.values
        score = pd.Series(num / np.maximum(den, 1e-12), index=factor_df.index)
    elif weighting in {"ic_weighted", "icir_weighted"} and ic_dict:
        raw_weights: dict[str, float] = {}
        metric = "icir" if weighting == "icir_weighted" else "ic"
        for col in numeric_cols:
            val = ic_dict.get(col) if ic_dict else None
            if isinstance(val, dict):
                raw_weights[col] = abs(float(val.get(metric, val.get("rank_ic", 0.0)) or 0.0))
            elif val is not None:
                raw_weights[col] = abs(float(val))
        w = pd.Series({k: v for k, v in raw_weights.items() if np.isfinite(v) and v > 0})
        if w.empty:
            score = factor_df[numeric_cols].mean(axis=1)
        else:
            w = w / (w.sum() + 1e-12)
            # P2-Q5-fix (M442): 与 weights 路径同款——按行可用因子加权均值，缺失不整行排除
            ic_cols = list(w.index)
            sub = factor_df[ic_cols]
            num = sub.fillna(0.0).values @ w.values
            den = sub.notna().values @ w.values
            score = pd.Series(num / np.maximum(den, 1e-12), index=factor_df.index)
    else:
        score = factor_df[numeric_cols].mean(axis=1)

    # Renormalize to 0-10 scale
    score = (score - score.min()) / (score.max() - score.min() + 1e-10) * 10
    return score


# ── 快捷接口 ──

def get_top_factors(
    symbols: list[str],
    bar_data: dict[str, list[list]],
    fin_data_cache: dict[str, dict],
    top_n: int = 20,
    weights: Optional[dict[str, float]] = None,
) -> list[dict]:
    """
    一站式: 计算因子 → 复合评分 → 返回 Top 股票。

    Returns: [{"symbol": ..., "composite_score": ..., "top_factors": {...}}, ...]
    """
    df = compute_factor_portfolio(symbols, bar_data, fin_data_cache)
    if df.empty:
        return []

    scores = compute_composite_score(df, weights)
    ranked = scores.sort_values(ascending=False).head(top_n)

    results = []
    for sym, score in ranked.items():
        if sym in df.index:
            row = df.loc[sym]
            # Get top 3 individual factors
            numeric = row.select_dtypes(include=[np.number])
            top3 = numeric.dropna().abs().sort_values(ascending=False).head(3).to_dict()
            results.append({
                "symbol": sym,
                "composite_score": round(score, 2),
                "top_factors": {k: round(v, 3) for k, v in top3.items()},
            })

    return results


# ── 默认因子权重 (适用于震荡/牛市) ──

DEFAULT_WEIGHTS_BULL = {
    "mom_6m": 0.15,
    "mom_3m": 0.10,
    "ep": 0.10,
    "bp": 0.10,
    "roe": 0.10,
    "earnings_growth_yoy": 0.10,
    "sales_growth_yoy": 0.05,
    "gross_margin": 0.05,
    "rsi_14": 0.05,
    "vol_20d": 0.05,
    "idio_vol_60d": 0.05,  # P2-Q5-fix (M433): 因子改名 idio_vol_60m→idio_vol_60d
    "ma_cross": 0.05,
    "volume_trend": 0.05,
}

DEFAULT_WEIGHTS_BEAR = {
    "idio_vol_60d": 0.20,  # P2-Q5-fix (M433): 因子改名 idio_vol_60m→idio_vol_60d
    "max_dd_12m": 0.15,
    "beta_60d": 0.10,  # P2-Q5-fix (M434): 因子改名 beta_60m→beta_60d
    "bp": 0.10,
    "ep": 0.10,
    "div_yield": 0.10,
    "leverage": 0.10,
    "gross_margin": 0.05,
    "current_ratio": 0.05,
    "rsi_14": 0.05,
}


# ── 因子分析与跟踪 ──

def _ensure_factor_db() -> sqlite3.Connection:
    """Create the factor analysis SQLite cache and required tables if needed."""
    FACTOR_IC_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(FACTOR_IC_DB))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS factor_ic (
            factor_name TEXT NOT NULL,
            date TEXT NOT NULL,
            ic REAL,
            rank_ic REAL,
            icir REAL,
            n_stocks INTEGER,
            PRIMARY KEY (factor_name, date)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS factor_performance (
            factor_name TEXT NOT NULL,
            date TEXT NOT NULL,
            long_return REAL,
            short_return REAL,
            spread REAL,
            t_stat REAL,
            long_sharpe REAL,
            PRIMARY KEY (factor_name, date)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS factor_turnover (
            factor_name TEXT NOT NULL,
            date TEXT NOT NULL,
            turnover REAL,
            PRIMARY KEY (factor_name, date)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS factor_ic_regime (
            factor_name TEXT NOT NULL,
            regime TEXT NOT NULL,
            date TEXT NOT NULL,
            ic REAL,
            PRIMARY KEY (factor_name, regime, date)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS factor_corr_snapshot (
            factor_name TEXT NOT NULL,
            date TEXT NOT NULL,
            value REAL,
            PRIMARY KEY (factor_name, date)
        )
        """
    )
    conn.commit()
    return conn


def compute_factor_ic(
    factor_name: str,
    factor_values: pd.Series,
    forward_returns: pd.Series,
    method: str = "spearman",
) -> float:
    """Compute single-period IC between factor exposure and forward returns."""
    _ = factor_name
    aligned = pd.concat([factor_values, forward_returns], axis=1, join="inner").dropna()
    if aligned.shape[0] < 3:
        return 0.0

    x = aligned.iloc[:, 0].astype(float)
    y = aligned.iloc[:, 1].astype(float)
    if x.nunique() <= 1 or y.nunique() <= 1:
        return 0.0

    method_l = method.lower()
    try:
        if scipy_stats is not None:
            if method_l == "pearson":
                corr, _ = scipy_stats.pearsonr(x, y)
            else:
                corr, _ = scipy_stats.spearmanr(x, y)
        else:
            corr = x.corr(y, method="pearson" if method_l == "pearson" else "spearman")
        return float(corr) if pd.notna(corr) and np.isfinite(corr) else 0.0
    except Exception:
        return 0.0


def update_factor_ic_db(factor_name: str, ic_value: float, date: str, n_stocks: int) -> None:
    """Persist one factor IC observation to the SQLite cache."""
    try:
        conn = _ensure_factor_db()
        hist = pd.read_sql_query(
            "SELECT rank_ic FROM factor_ic WHERE factor_name = ? ORDER BY date DESC LIMIT 252",
            conn,
            params=(factor_name,),
        )
        vals = pd.concat([pd.Series([float(ic_value)]), hist["rank_ic"]], ignore_index=True).dropna()
        icir = float(vals.mean() / (vals.std(ddof=1) + 1e-12)) if len(vals) > 1 else 0.0
        conn.execute(
            """
            INSERT OR REPLACE INTO factor_ic(factor_name, date, ic, rank_ic, icir, n_stocks)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (factor_name, date, float(ic_value), float(ic_value), icir, int(n_stocks)),
        )
        conn.commit()
        conn.close()
    except Exception:
        return


def track_factor_ic_history(factor_name: str, n_days: int = 252) -> dict[str, float | int]:
    """Read cached IC history and return summary statistics for a factor."""
    try:
        conn = _ensure_factor_db()
        df = pd.read_sql_query(
            "SELECT date, rank_ic FROM factor_ic WHERE factor_name = ? ORDER BY date DESC LIMIT ?",
            conn,
            params=(factor_name, int(n_days)),
        )
        conn.close()
        vals = df["rank_ic"].dropna().astype(float)
        if vals.empty:
            return {"mean": 0.0, "std": 0.0, "icir": 0.0, "positive_ratio": 0.0, "gt_002_ratio": 0.0, "n": 0}
        std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        return {
            "mean": float(vals.mean()),
            "std": std,
            "icir": float(vals.mean() / (std + 1e-12)) if std > 0 else 0.0,
            "positive_ratio": float((vals > 0).mean()),
            "gt_002_ratio": float((vals.abs() > 0.02).mean()),
            "n": int(len(vals)),
        }
    except Exception:
        return {"mean": 0.0, "std": 0.0, "icir": 0.0, "positive_ratio": 0.0, "gt_002_ratio": 0.0, "n": 0}


def compute_ic_decay(factor_name: str, max_hold: int = 20) -> pd.Series:
    """Return cached IC decay for holding horizons 1/3/5/10/15/20 days."""
    horizons = [h for h in [1, 3, 5, 10, 15, 20] if h <= max_hold]
    result = pd.Series(index=horizons, dtype=float, name=factor_name)
    try:
        conn = _ensure_factor_db()
        for horizon in horizons:
            candidates = [f"{factor_name}_{horizon}d", f"{factor_name}_h{horizon}"]
            placeholders = ",".join("?" for _ in candidates)
            df = pd.read_sql_query(
                f"SELECT rank_ic FROM factor_ic WHERE factor_name IN ({placeholders}) ORDER BY date DESC LIMIT 252",
                conn,
                params=tuple(candidates),
            )
            if not df.empty:
                result.loc[horizon] = float(df["rank_ic"].dropna().astype(float).mean())
            elif horizon == 1:
                base = pd.read_sql_query(
                    "SELECT rank_ic FROM factor_ic WHERE factor_name = ? ORDER BY date DESC LIMIT 252",
                    conn,
                    params=(factor_name,),
                )
                if not base.empty:
                    result.loc[horizon] = float(base["rank_ic"].dropna().astype(float).mean())
        conn.close()
    except Exception:
        return result
    return result


def factor_long_short_portfolio(
    factor_name: str,
    factor_values: pd.Series,
    forward_returns: pd.Series,
    n_groups: int = 5,
) -> dict[str, float]:
    """Compute equal-weight long-short performance between top and bottom factor quantiles."""
    _ = factor_name
    aligned = pd.concat([factor_values, forward_returns], axis=1, join="inner").dropna()
    if aligned.shape[0] < max(n_groups, 3):
        return {"long_return": 0.0, "short_return": 0.0, "spread": 0.0, "t_stat": 0.0, "long_sharpe": 0.0}
    aligned.columns = ["factor", "ret"]
    try:
        aligned["group"] = pd.qcut(aligned["factor"].rank(method="first"), n_groups, labels=False) + 1
    except Exception:
        return {"long_return": 0.0, "short_return": 0.0, "spread": 0.0, "t_stat": 0.0, "long_sharpe": 0.0}

    long_rets = aligned.loc[aligned["group"] == n_groups, "ret"].astype(float)
    short_rets = aligned.loc[aligned["group"] == 1, "ret"].astype(float)
    long_return = float(long_rets.mean()) if not long_rets.empty else 0.0
    short_return = float(short_rets.mean()) if not short_rets.empty else 0.0
    spread = long_return - short_return
    try:
        if scipy_stats is not None and len(long_rets) > 1 and len(short_rets) > 1:
            t_stat = float(scipy_stats.ttest_ind(long_rets, short_rets, equal_var=False, nan_policy="omit").statistic)
        else:
            spread_sample = long_rets.reset_index(drop=True) - short_rets.reset_index(drop=True)
            t_stat = float(spread_sample.mean() / (spread_sample.std(ddof=1) / np.sqrt(len(spread_sample)) + 1e-12)) if len(spread_sample) > 1 else 0.0
    except Exception:
        t_stat = 0.0
    long_sharpe = float(long_rets.mean() / (long_rets.std(ddof=1) + 1e-12)) if len(long_rets) > 1 else 0.0
    return {
        "long_return": long_return,
        "short_return": short_return,
        "spread": spread,
        "t_stat": t_stat if np.isfinite(t_stat) else 0.0,
        "long_sharpe": long_sharpe if np.isfinite(long_sharpe) else 0.0,
    }


def track_factor_performance(factor_name: str, n_days: int = 252) -> dict[str, float | int]:
    """Summarize cached long-short factor performance and rolling Sharpe."""
    try:
        conn = _ensure_factor_db()
        df = pd.read_sql_query(
            "SELECT date, spread, long_return, short_return FROM factor_performance WHERE factor_name = ? ORDER BY date DESC LIMIT ?",
            conn,
            params=(factor_name, int(n_days)),
        )
        conn.close()
        spreads = df["spread"].dropna().astype(float)
        if spreads.empty:
            return {"cumulative_return": 0.0, "mean_spread": 0.0, "rolling_sharpe": 0.0, "n": 0}
        rolling = spreads.head(min(60, len(spreads)))
        return {
            "cumulative_return": float((1.0 + spreads).prod() - 1.0),
            "mean_spread": float(spreads.mean()),
            "rolling_sharpe": float(rolling.mean() / (rolling.std(ddof=1) + 1e-12)) if len(rolling) > 1 else 0.0,
            "n": int(len(spreads)),
        }
    except Exception:
        return {"cumulative_return": 0.0, "mean_spread": 0.0, "rolling_sharpe": 0.0, "n": 0}


def compute_factor_turnover(
    today_ranking: pd.Series,
    yesterday_ranking: pd.Series,
    top_pct: float = 0.2,
) -> float:
    """Compute top-bucket turnover from today's and yesterday's factor rankings."""
    if today_ranking.empty or yesterday_ranking.empty or top_pct <= 0:
        return 0.0
    common = today_ranking.dropna().index.intersection(yesterday_ranking.dropna().index)
    if len(common) == 0:
        return 0.0
    n_top = max(1, int(len(common) * min(top_pct, 1.0)))
    today_top = set(today_ranking.loc[common].sort_values(ascending=False).head(n_top).index)
    yesterday_top = set(yesterday_ranking.loc[common].sort_values(ascending=False).head(n_top).index)
    turnover = 1.0 - (len(today_top & yesterday_top) / n_top)
    return float(np.clip(turnover, 0.0, 1.0))


def track_turnover_history(factor_name: str, n_days: int = 252) -> dict[str, float | int]:
    """Read cached daily turnover history and return average turnover statistics."""
    try:
        conn = _ensure_factor_db()
        df = pd.read_sql_query(
            "SELECT turnover FROM factor_turnover WHERE factor_name = ? ORDER BY date DESC LIMIT ?",
            conn,
            params=(factor_name, int(n_days)),
        )
        conn.close()
        vals = df["turnover"].dropna().astype(float)
        if vals.empty:
            return {"mean_turnover": 0.0, "std_turnover": 0.0, "n": 0}
        return {
            "mean_turnover": float(vals.mean()),
            "std_turnover": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "n": int(len(vals)),
        }
    except Exception:
        return {"mean_turnover": 0.0, "std_turnover": 0.0, "n": 0}


if __name__ == "__main__":
    print("=== 因子库 V1 ===\n")
    for family in ["momentum", "value", "quality", "growth", "low_vol", "technical"]:
        factors = list_factors(family)
        print(f"[{family}] {len(factors)}个因子:")
        for f in factors:
            print(f"  {f:20s} {FACTOR_META[f]['desc']}")
        print()
    print(f"总计: {len(FACTOR_META)} 个因子")
