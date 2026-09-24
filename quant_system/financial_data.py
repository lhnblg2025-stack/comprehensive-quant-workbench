"""
financial_data.py — V5.1 财务因子数据接入 (全面扩展: 80项指标×多期)

从 akshare stock_financial_abstract 获取 A 股全部80项财务指标（含多期历史），
存入 SQLite 缓存。支持:
  1) 全部80项指标的批量存储与读取
  2) 多期历史数据（用于环比/同比变化因子）
  3) 替换现有5字段摘要 → 完整因子矩阵

用法:
  python3 -m quant_system.financial_data --cache    # 缓存全部
  python3 -m quant_system.financial_data --show 600519  # 查看

D5收敛登记 (2026-08-11): 基本面域收敛（保守策略）——与 fundamental.py /
fundamental_analysis.py 的数值转换重复已由 D1(utils.safe_float) 收敛；本模块
fetch_financial_indicators（akshare financial_abstract 80指标×多期 + SQLite 缓存）
为独立能力保留；派生估值因子 pe_ttm/pb/ps_ttm 与 fundamental.py(东财行情字段)、
fundamental_analysis.py(市值/年报) 为同名异口径保留。
"""

from __future__ import annotations
import logging

import atexit
import re
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from quant_system.utils import safe_float as _safe_float_impl

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path.home() / ".quant_system" / "financial.db"

# V5.1 expansion: 完整 80 项财务指标映射 (内部名 → 中文名)
FINANCIAL_METRICS = {
    # 盈利
    "roe": "净资产收益率(ROE)",
    "roa": "总资产报酬率(ROA)",
    "gross_margin": "毛利率",
    "net_margin": "销售净利率",
    "profit_margin": "营业利润率",
    "cost_profit_ratio": "成本费用利润率",
    "ebit_margin": "息税前利润率",
    "roic": "投入资本回报率",
    "roa_avg": "总资产净利率_平均",
    # 估值
    "pe_ttm": "pe_ttm",
    "pb": "pb",
    "ps_ttm": "ps_ttm",
    "pcf_ttm": "pcf_ttm",
    "div_yield": "dividend_yield",
    # 每股
    "eps": "基本每股收益",
    "bvps": "每股净资产",
    "ocf_per_share": "每股经营现金流",
    "sales_per_share": "每股营业收入",
    "ebit_per_share": "每股息税前利润",
    "fcff_per_share": "每股企业自由现金流量",
    # 资产负债
    "debt_ratio": "资产负债率",
    "current_ratio": "流动比率",
    "quick_ratio": "速动比率",
    "debt_equity_ratio": "产权比率",
    "equity_multiplier": "权益乘数",
    # 周转
    "asset_turnover": "总资产周转率",
    "inventory_turnover": "存货周转率",
    "receivables_turnover": "应收账款周转率",
    # 现金流
    "ocf_to_sales": "经营性现金净流量/营业总收入",
    "ocf_to_net_profit": "经营活动净现金/归属母公司的净利润",
    # 成长
    "sales_growth_yoy": "营业总收入增长率",
    "profit_growth_yoy": "归属母公司净利润增长率",
    # 稀释
    "diluted_eps": "稀释每股收益",
    "retained_eps": "每股未分配利润",
    "capital_reserve_ps": "每股资本公积金",
}

# D5收敛登记: 独立能力保留 —— 本映射表服务于 akshare financial_abstract 中文列名，
# 与 fundamental.py 的 THS 摘要列名映射、fundamental_analysis.py 的 ratios dict
# 键名(roe/gross_margin/net_margin 等) 同名但源列名/量纲不同，不强迁。


# P2-Q3-fix(M379): 模块级连接复用。每线程一个连接，不再每次调用新建连接；
# 全部连接登记到 _CONNECTIONS，进程退出时由 atexit 显式 close，
# 不再依赖 CPython 引用计数回收。synchronous=OFF 仅用于缓存库，掉电丢数风险已知悉。
_LOCAL = threading.local()
_CONNECTIONS: list[sqlite3.Connection] = []
_CONNECTIONS_LOCK = threading.Lock()


def _db() -> sqlite3.Connection:
    """Get or create the financial data database (线程内复用连接)."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = getattr(_LOCAL, "conn", None)
    if conn is None:
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _LOCAL.conn = conn
        with _CONNECTIONS_LOCK:
            _CONNECTIONS.append(conn)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS financial_indicators (
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                indicator TEXT NOT NULL,
                value REAL,
                updated TEXT NOT NULL,
                PRIMARY KEY (symbol, date, indicator)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS financial_cache_log (
                symbol TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                n_indicators INTEGER
            )
        """)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=OFF")
        conn.commit()
    return conn


def _close_db() -> None:
    """关闭所有已打开的缓存库连接（atexit 调用）。"""
    with _CONNECTIONS_LOCK:
        conns = list(_CONNECTIONS)
        _CONNECTIONS.clear()
    for conn in conns:
        try:
            conn.close()
        except Exception as e:
            logging.getLogger(__name__).error(f"[financial_data] 操作失败: {e}", exc_info=True)
    _LOCAL.conn = None


atexit.register(_close_db)


# ── 财务指标获取（按最新方式） ──

# V5.1 fix: 从 financial_abstract 提取全部80项指标的中文名到内部名的映射
# 运行时自动构建
_ABSTRACT_KEYS: dict[str, str] = {}

# 可用的中文指标名（全部）
_ABSTRACT_CHINESE_FACTORS = [
    "归母净利润", "营业总收入", "营业成本", "净利润", "扣非净利润",
    "股东权益合计(净资产)", "商誉", "经营现金流量净额",
    "基本每股收益", "每股净资产", "每股现金流",
    "净资产收益率(ROE)", "总资产报酬率(ROA)",
    "毛利率", "销售净利率", "期间费用率", "资产负债率",
    "基本每股收益", "稀释每股收益",
    "摊薄每股收益_最新股数", "摊薄每股净资产_期末股数",
    "调整每股净资产_期末股数", "每股净资产_最新股数",
    "每股经营现金流", "每股现金流量净额",
    "每股企业自由现金流量", "每股股东自由现金流量",
    "每股未分配利润", "每股资本公积金", "每股盈余公积金",
    "每股留存收益", "每股营业收入", "每股营业总收入",
    "每股息税前利润",
    "净资产收益率(ROE)", "摊薄净资产收益率",
    "净资产收益率_平均", "净资产收益率_平均_扣除非经常损益",
    "摊薄净资产收益率_扣除非经常损益", "息税前利润率",
    "总资产报酬率", "总资本回报率", "投入资本回报率",
    "息前税后总资产报酬率_平均",
    "毛利率", "销售净利率", "成本费用利润率", "营业利润率",
    "总资产净利率_平均", "总资产净利率_平均(含少数股东损益)",
    "归母净利润", "营业总收入", "净利润", "扣非净利润",
    "营业总收入增长率", "归属母公司净利润增长率",
    "经营活动净现金/销售收入", "经营性现金净流量/营业总收入",
    "成本费用率", "期间费用率", "销售成本率",
    "经营活动净现金/归属母公司的净利润", "所得税/利润总额",
    "流动比率", "速动比率", "保守速动比率",
    "资产负债率", "权益乘数", "权益乘数(含少数股权的净资产)",
    "产权比率", "现金比率",
    "应收账款周转率", "应收账款周转天数",
    "存货周转率", "存货周转天数",
    "总资产周转率", "总资产周转天数",
    "流动资产周转率", "流动资产周转天数",
    "应付账款周转率",
]


def _build_abstract_keys() -> dict[str, str]:
    """反向映射: 中文指标名 → 我们用的内部名。"""
    if _ABSTRACT_KEYS:
        return _ABSTRACT_KEYS
    # INI: 内部名 → 中文名的正向查找
    rev = {v: k for k, v in FINANCIAL_METRICS.items() if isinstance(v, str)}
    # 额外补充没有在 FINANCIAL_METRICS 里注册的中文名
    extra = {
        "归母净利润": "net_profit_parent",
        "营业总收入": "revenue",
        "营业成本": "cost_of_sales",
        "净利润": "net_profit",
        "扣非净利润": "net_profit_excl",
        "股东权益合计(净资产)": "equity",
        "经营现金流量净额": "ocf_total",
        "每股现金流": "cash_flow_per_share",
        "每股留存收益": "retained_earnings_ps",
        "每股盈余公积金": "surplus_reserve_ps",
        "营业总收入增长率": "sales_growth_yoy",
        "归属母公司净利润增长率": "profit_growth_yoy",
        "经营现金流量净额": "ocf_total",
        "保守速动比率": "conservative_quick_ratio",
        "现金比率": "cash_ratio",
        "应收账款周转天数": "receivables_turnover_days",
        "存货周转天数": "inventory_turnover_days",
        "总资产周转天数": "asset_turnover_days",
        "流动资产周转天数": "current_asset_turnover_days",
        "应付账款周转率": "payables_turnover",
        "流动资产周转率": "current_asset_turnover",
    }
    rev.update(extra)
    _ABSTRACT_KEYS.update(rev)
    return _ABSTRACT_KEYS


def _write_indicator(db: sqlite3.Connection, symbol: str, period_date: str,
                     indicator: str, value: float | None) -> None:
    """Write one raw or derived financial indicator row.

    P2-Q3-fix(M377): 缺失/非数值指标以 NULL 入库，与真实 0 区分（未知 vs 极差）。
    读取端（get_all_for_date / _cli_show）已按 ``value IS NOT NULL`` 过滤。
    """
    if value is None:
        db.execute(
            """INSERT OR REPLACE INTO financial_indicators
               (symbol, date, indicator, value, updated)
               VALUES (?, ?, ?, NULL, ?)""",
            (symbol, period_date, indicator, datetime.now().isoformat()),
        )
        return
    try:
        val = float(value)
    except (TypeError, ValueError):
        val = None
    if val is not None and not np.isfinite(val):
        val = None
    db.execute(
        """INSERT OR REPLACE INTO financial_indicators
           (symbol, date, indicator, value, updated)
           VALUES (?, ?, ?, ?, ?)""",
        (symbol, period_date, indicator, val, datetime.now().isoformat()),
    )


def _same_report_period_previous(latest_date: str, date_cols: list[str]) -> str | None:
    """Find prior report with the same MMDD, avoiding Q1-vs-annual cumulative comparisons."""
    if len(str(latest_date)) != 8:
        return None
    suffix = str(latest_date)[4:]
    for candidate in date_cols[1:]:
        cand = str(candidate)
        if len(cand) == 8 and cand[4:] == suffix:
            return cand
    return None


def _profit_value(idx: dict[str, Any]) -> float:
    return _safe_float(idx.get("净利润", idx.get("归母净利润", 0)))


def _dividend_per_share_em(row: Any) -> float:
    """解析东财分红送配单行为每股派息(元)。

    stock_fhps_detail_em 的"现金分红-现金分红比例"为每10股口径(如 10派1元 → 1.0),
    优先解析"现金分红-现金分红比例描述"(如 "10派1元"), 解析不到时按比例/10 兜底。
    """
    desc = str(row.get("现金分红-现金分红比例描述", "") or "")
    m = re.search(r"派\s*([\d.]+)\s*元", desc)
    if m:
        return _safe_float(m.group(1)) / 10.0
    ratio = _safe_float(row.get("现金分红-现金分红比例", 0))
    return ratio / 10.0 if ratio > 0 else 0.0


def _recent_dividend_per_share(symbol: str) -> float:
    """最近一期已实施分红的每股派息(元); 获取失败返回 0 (DCF 股息折现降级跳过)。

    akshare 1.18.64 兼容: 旧股息接口已删除, 优先东财 stock_fhps_detail_em,
    失败/无数据时回退新浪 stock_history_dividend_detail(symbol, indicator="分红")。
    """
    try:
        import akshare as ak
        df = ak.stock_fhps_detail_em(symbol=symbol)
        if df is not None and not df.empty and "除权除息日" in df.columns:
            recent = df[df["除权除息日"].notna()].sort_values("除权除息日", ascending=False)
            for _, row in recent.iterrows():
                if "实施" not in str(row.get("方案进度", "")):
                    continue
                dps = _dividend_per_share_em(row)
                if dps > 0:
                    return dps
    except Exception as e:
        logging.getLogger(__name__).warning(f"[financial_data] 东财分红接口失败, 回退新浪: {e}")

    try:
        import akshare as ak
        df = ak.stock_history_dividend_detail(symbol=symbol, indicator="分红")
        if df is not None and not df.empty and "除权除息日" in df.columns:
            recent = df[df["除权除息日"].notna()].sort_values("除权除息日", ascending=False)
            for _, row in recent.iterrows():
                dps = _safe_float(row.get("派息", 0))
                if dps > 100:  # 兜底: 部分新浪数据按每10股展示
                    dps /= 10.0
                if dps > 0:
                    return dps
    except Exception as e:
        logging.getLogger(__name__).warning(f"[financial_data] 新浪分红接口失败: {e}")
    return 0.0


def _read_cached_indicators(symbol: str, max_age_days: int) -> dict[str, float] | None:
    """Read the latest report period without requiring SQLite write access."""
    if not DB_PATH.exists():
        return None
    conn = None
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        row = conn.execute(
            "SELECT MAX(date) FROM financial_indicators WHERE symbol = ?",
            (symbol,),
        ).fetchone()
        if not row or not row[0]:
            return None
        last_date = datetime.strptime(str(row[0]), "%Y%m%d")
        if (datetime.now() - last_date).days >= max_age_days:
            return None
        latest = conn.execute(
            """SELECT indicator, value FROM financial_indicators
               WHERE symbol = ? AND date = ? ORDER BY indicator""",
            (symbol, row[0]),
        ).fetchall()
        return {name: value for name, value in latest if value is not None} or None
    except (sqlite3.Error, ValueError):
        return None
    finally:
        if conn is not None:
            conn.close()


def fetch_financial_indicators(
    symbol: str,
    force: bool = False,
    max_age_days: int = 30,
) -> dict[str, float]:
    """
    获取单只股票的全部80项财务指标×多期历史。

    V5.1 expansion: 从 financial_abstract 提取全部指标，存储所有可用期次。

    D5收敛登记: 独立能力保留 —— 与 fundamental.fetch_financials 异名异实现：
    本函数取 akshare financial_abstract 80指标×多期 + SQLite 缓存(max_age_days)，
    彼取 THS 摘要字段 + 东财 push2 + 新浪兜底(内存600s TTL)。
    D5收敛: 同名异口径保留 —— 派生 pe_ttm/pb/ps_ttm/pcf_ttm = 最新收盘价/每股
    (TTM口径)，与 fundamental.py 东财行情字段 PE/PB、fundamental_analysis.py
    市值/年报口径 PE/PB/PS 不同；roe/gross_margin/net_margin 等为百分数(如15=15%)，
    与 fundamental_analysis 评分输入的小数口径(0.15=15%)不同。

    Returns:
        {factor_name: latest_value, ...} — 为保持向后兼容返回最新期次
    """
    # Read-first hot path: serving cached quarterly reports must not require WAL or
    # CREATE TABLE permissions. This also keeps the endpoint available when a
    # cache volume is mounted read-only.
    if not force:
        cached = _read_cached_indicators(symbol, max_age_days)
        if cached:
            return cached

    try:
        db = _db()
        import akshare as ak

        result: dict[str, float] = {}
        close_price = 0.0
        try:
            from quant_system.data_store import get_store
            ds = get_store()
            cdf = ds.get(symbol, days=5)
            if cdf is not None and not cdf.empty and "close" in cdf.columns:
                close_price = float(cdf["close"].iloc[-1])
        except Exception as e:
            logging.getLogger(__name__).error(f"[financial_data] 操作失败: {e}", exc_info=True)

        date_cols: list[str] = []
        # 从 financial_abstract 获取全部80项指标 × 多期
        fin = ak.stock_financial_abstract(symbol=symbol)
        if fin is not None and not fin.empty:
            date_cols = [
                c for c in fin.columns
                if c not in ("选项", "指标") and str(c).replace(".", "", 1).lstrip("-").isdigit()
            ]
            date_cols.sort(reverse=True)  # latest first

            keys_map = _build_abstract_keys()
            indicator_names = list(fin["指标"].values)

            # 为每个期次 + 每个指标存入数据库
            for period_date in date_cols:
                period_idx = dict(zip(indicator_names, fin[period_date].values))
                for indicator_name, raw_val in period_idx.items():
                    # 找内部名
                    internal_name = keys_map.get(indicator_name)
                    if not internal_name:
                        # 找不到映射的就用中文名直接存
                        internal_name = indicator_name
                    # P2-Q3-fix(M377): 缺失/非数值 → None 入库为 NULL，不再写 0.0
                    try:
                        val = float(raw_val) if raw_val is not None and raw_val != "" else None
                    except (TypeError, ValueError, TypeError):
                        val = None
                    if val is not None and not np.isfinite(val):
                        val = None
                    _write_indicator(db, symbol, period_date, internal_name, val)
                    # 只取最新期次的值放入 result（缺失值不入 result，保持调用方 .get(.., 0) 兼容）
                    if period_date == date_cols[0] and val is not None:
                        result[internal_name] = val

            # 估值因子用最新收盘价 + 财务数据计算
            lat = date_cols[0]
            latest_idx = dict(zip(indicator_names, fin[lat].values))
            bvps = _safe_float(latest_idx.get("每股净资产", 0))
            eps = _safe_float(latest_idx.get("基本每股收益", 0))
            sales_ps = _safe_float(latest_idx.get("每股营业收入", 0))
            ocf_ps = _safe_float(latest_idx.get("每股经营现金流", 0))

            if close_price > 0:
                if bvps > 0:
                    result["pb"] = round(close_price / bvps, 4)
                    result["bp"] = round(bvps / close_price, 4)
                if eps > 0:
                    result["pe_ttm"] = round(close_price / eps, 4)
                    result["ep"] = round(eps / close_price, 4)
                if sales_ps > 0:
                    result["ps_ttm"] = round(close_price / sales_ps, 4)
                    result["sp"] = round(sales_ps / close_price, 4)
                if ocf_ps > 0:
                    result["pcf_ttm"] = round(close_price / ocf_ps, 4)
                    result["cp"] = round(ocf_ps / close_price, 4)

        # 股息率 — 额外接口
        if close_price > 0:
            dps = _recent_dividend_per_share(symbol)
            if dps > 0:
                result["div_yield"] = round(dps / close_price * 100, 2)

        # 多期变化因子：A股利润表为年初累计数，只比较同一报告期（如 0331 vs 上年 0331）。
        if len(date_cols) >= 2:
            lat = date_cols[0]
            prev = _same_report_period_previous(lat, date_cols)
        if len(date_cols) >= 2 and prev:
            latest_idx = dict(zip(indicator_names, fin[lat].values))
            prev_idx = dict(zip(indicator_names, fin[prev].values))

            # ROE变动
            # D5收敛: 百分数口径（如 15 = 15%），与 fundamental_analysis 小数口径不同。
            cur_roe = _safe_float(latest_idx.get("净资产收益率(ROE)", 0))
            prev_roe = _safe_float(prev_idx.get("净资产收益率(ROE)", 0))
            if cur_roe != 0 and prev_roe != 0:
                result["roe_change"] = cur_roe - prev_roe

            # 毛利率变动
            # D5收敛: 百分数口径，与 fundamental_analysis.score_profitability 小数口径不同。
            cur_gm = _safe_float(latest_idx.get("毛利率", 0))
            prev_gm = _safe_float(prev_idx.get("毛利率", 0))
            if cur_gm != 0 and prev_gm != 0:
                result["margin_change"] = cur_gm - prev_gm

            # 净利润同比增速（同一报告期累计值对比），避免 Q1 累计误比上年年报。
            # D5收敛: 百分数口径，与 fundamental_analysis._normalize_growth 的归一化小数不同。
            cur_profit = _profit_value(latest_idx)
            prev_profit = _profit_value(prev_idx)
            if prev_profit != 0 and cur_profit != 0:
                result["earnings_growth_qoq"] = (cur_profit / prev_profit - 1) * 100

        if date_cols:
            for indicator in (
                "pe_ttm", "ep", "pb", "bp", "ps_ttm", "sp", "pcf_ttm", "cp",
                "div_yield", "roe_change", "margin_change", "earnings_growth_qoq",
            ):
                _write_indicator(db, symbol, date_cols[0], indicator, result.get(indicator))

        # 保存到数据库
        db.commit()
        return result

    except Exception as e:
        return {"_error": str(e)}


def _safe_float(val: Any, default: float = 0.0) -> float:
    """安全类型转换（D1 收敛: 转发 quant_system.utils.safe_float）。"""
    return _safe_float_impl(
        val, default=default, allow_bool=True,
        clean_commas=False, clean_percent=False,
    )


def batch_fetch(symbols: list[str], max_workers: int = 3) -> dict[str, dict[str, float]]:
    """批量获取财务数据。

    P2-Q3-fix(L380): max_workers 参数此前从未使用（纯串行）。现用
    ThreadPoolExecutor 实现并发；每个 worker 线程通过线程本地连接复用 DB 连接。

    D5收敛登记: 独立能力保留（SQLite 批量缓存层，无等价实现）。
    """
    results: dict[str, dict[str, float]] = {}
    if not symbols:
        return results
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_sym = {pool.submit(fetch_financial_indicators, sym): sym for sym in symbols}
        for i, future in enumerate(as_completed(future_to_sym), 1):
            sym = future_to_sym[future]
            try:
                data = future.result()
            except Exception as e:
                data = {"_error": str(e)}
            results[sym] = data
            vals = {k: v for k, v in data.items() if v != 0 and not k.startswith("_")}
            print(f"  [{i}/{len(symbols)}] {sym}: {len(vals)} indicators")
    return results


def get_latest_value(symbol: str, factor_name: str) -> float | None:
    """获取缓存中最新的某个财务因子值。

    D5收敛登记: 独立能力保留（因子截面读取入口，factor_model 等消费方）。
    W2.5 修复: 缺失返回 None（与"真实 0"区分，如 ROE=0 是有效值）。
    """
    db = _db()
    row = db.execute(
        """SELECT value FROM financial_indicators
           WHERE symbol = ? AND indicator = ?
           ORDER BY date DESC LIMIT 1""",
        (symbol, factor_name),
    ).fetchone()
    if row and row[0] is not None:
        return float(row[0])
    return None


def get_financial_summary(symbol: str, use_cache: bool = True) -> dict[str, float]:
    """
    V5.1 expansion: 返回所有可用财务指标，不再仅限5字段。

    D5收敛登记: 独立能力保留（opportunity 等消费方入口）。

    Returns:
        {indicator_name: value, ...} — 全部已缓存的指标
    """
    try:
        ind = fetch_financial_indicators(symbol, force=not use_cache, max_age_days=30)
        if not ind or "_error" in ind:
            return {}
        return ind
    except Exception:
        return {}


def get_all_for_date(symbols: list[str]) -> pd.DataFrame:
    """获取所有股票的最新财务因子矩阵: rows=symbols, cols=factor_names. (V5.1: 不限量)

    P2-Q3-fix(M378):
      ① symbols=[] 短路返回空 DataFrame，不再生成 ``IN ()`` 非法 SQL；
      ② 派生估值因子（pe_ttm/pb 等）由 fetch_financial_indicators 在抓取时入库，
         此处直接读取库中数据（缺失值在库中为 NULL，读取时过滤，代表「未知」而非 0）；
      ③ 数据新鲜度检查：缓存为空/整体过期时打印告警（不静默返回空矩阵）。

    D5收敛登记: 独立能力保留（截面因子矩阵读取，无等价实现）。
    """
    if not symbols:
        return pd.DataFrame(index=symbols)

    db = _db()
    # 数据新鲜度检查（M378-③）
    fresh_row = db.execute("SELECT MAX(updated) FROM financial_indicators").fetchone()
    if not fresh_row or not fresh_row[0]:
        print("⚠️ financial_data: 缓存为空，请先执行 --cache 缓存财务数据", file=sys.stderr)
    else:
        try:
            last_updated = datetime.fromisoformat(fresh_row[0])
            if (datetime.now() - last_updated).days > 30:
                print(
                    f"⚠️ financial_data: 缓存数据已过期({last_updated:%Y-%m-%d})，建议重新缓存",
                    file=sys.stderr,
                )
        except ValueError:
            pass

    rows = db.execute(
        """SELECT symbol, indicator, value
           FROM financial_indicators
           WHERE symbol IN ({})
           ORDER BY symbol, date DESC""".format(",".join("?" * len(symbols))),
        symbols,
    ).fetchall()
    if not rows:
        return pd.DataFrame(index=symbols)

    # 取每个symbol × indicator的最新值
    grouped: dict[str, dict[str, float]] = {}
    seen: set[tuple[str, str]] = set()
    for sym, ind, val in rows:
        key = (sym, ind)
        if key not in seen and val is not None:
            seen.add(key)
            if sym not in grouped:
                grouped[sym] = {}
            grouped[sym][ind] = float(val)

    df = pd.DataFrame.from_dict(grouped, orient="index")
    return df


def cache_all_indicators(symbols: list[str]) -> dict[str, int]:
    """缓存全部股票的全部指标。

    D5收敛登记: 独立能力保留（批量缓存入口，无等价实现）。
    """
    results = batch_fetch(symbols)
    return {sym: len(vals) for sym, vals in results.items()}


# ── CLI ──

def _cli_show(symbol: str) -> None:
    """Display cached financial data for a symbol."""
    db = _db()
    # Get latest 5 periods
    periods = db.execute(
        """SELECT DISTINCT date FROM financial_indicators
           WHERE symbol = ? ORDER BY date DESC LIMIT 5""",
        (symbol,),
    ).fetchall()
    if not periods:
        print(f"{symbol}: 无缓存数据，尝试抓取...")
        result = fetch_financial_indicators(symbol, force=True)
        for k, v in sorted(result.items()):
            if not k.startswith("_"):
                print(f"  {k:>24}: {v}")
        return

    print(f"{symbol}: {len(periods)} 个期次")
    for (pdate,) in periods:
        indicators = db.execute(
            """SELECT indicator, value FROM financial_indicators
               WHERE symbol = ? AND date = ?
               ORDER BY indicator""",
            (symbol, pdate),
        ).fetchall()
        vals = {r[0]: r[1] for r in indicators if r[1] is not None}
        print(f"\n  期次 {pdate}: {len(vals)} 个指标")
        for k, v in sorted(vals.items()):
            print(f"    {k:>24}: {v}")


if __name__ == "__main__":
    if "--cache" in sys.argv:
        # P2-Q3-fix(L381): 文档声称「缓存全部」，此前只缓存前5只。现默认缓存全部
        # 自选股股票池（watchlist 409只），并提供 --limit N 便于快速抽样验证。
        from quant_system.watchlist import get_watchlist
        stocks = get_watchlist()
        syms = [s["symbol"] for s in stocks if s.get("symbol")]
        if "--limit" in sys.argv:
            idx = sys.argv.index("--limit") + 1
            if idx < len(sys.argv):
                syms = syms[: int(sys.argv[idx])]
        print(f"缓存 {len(syms)} 只股票财务指标...")
        batch_fetch(syms, max_workers=3)
        print("完成")
    elif "--show" in sys.argv:
        idx = sys.argv.index("--show") + 1
        if idx < len(sys.argv):
            _cli_show(sys.argv[idx])
    else:
        print("用法: python3 -m quant_system.financial_data --cache | --show <symbol>")
