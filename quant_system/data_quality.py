"""
量化交易系统 — 数据质量管理 (Data Quality)

提供退市股票查询、幸存者偏差检查、复权价格验证、
价格异常检测、缺失日期填充等功能。

用法:
  from quant_system.data_quality import (get_delisted_stocks,
      check_survivorship_bias, validate_adjusted_prices,
      detect_price_anomalies, fill_missing_dates)

D1 数据域收敛登记 (2026-08-11):
  - DataFrame 级质量校验（空帧/缺列/NaN/价格过滤/fill_gaps）唯一真源为
    data_store._validate / data_store._fill_missing_dates（DataStore.get 内部调用）。
  - 本模块保留为独立的质量工具（SQLite quality.db 退市/幸存者偏差/复权校验、
    列表式 bars 的价格异常检测与缺失日期填充），输入形态（list[list]）与
    data_store（DataFrame）不同，不重复转发；公共函数名与签名保持不变。
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

# V4.1 fix: trading calendar cache for holiday filtering
# 2026-08-01: 交易日历统一收归 market_clock.get_trade_calendar（全系统唯一源）
try:
    from quant_system.market_clock import get_trade_calendar as _get_trade_calendar
except Exception:
    from market_clock import get_trade_calendar as _get_trade_calendar  # type: ignore[no-redef]

from quant_system.utils import to_float as _to_float

_TRADE_CALENDAR_CACHE: set | None = None

# P2-Q1-fix (Q1-M007): 板块涨跌幅阈值源（主板±10%/双创±20%/北交所±30%/ST±5%）
try:
    from quant_system.market_rules import get_price_limit_pct
except Exception:
    try:
        from market_rules import get_price_limit_pct  # type: ignore[no-redef]
    except Exception:
        get_price_limit_pct = None

# P2-Q1-fix (Q1-L017): 交易日历降级只告警一次，避免逐日重复刷屏
_CAL_FALLBACK_WARNED = False


def get_trade_calendar() -> set:
    """获取A股交易日历，缓存模块级变量（委托 market_clock 唯一源）。"""
    global _TRADE_CALENDAR_CACHE
    if _TRADE_CALENDAR_CACHE is not None:
        return _TRADE_CALENDAR_CACHE
    _TRADE_CALENDAR_CACHE = _get_trade_calendar()
    return _TRADE_CALENDAR_CACHE

ROOT = Path(__file__).resolve().parent
CST = timezone(timedelta(hours=8))
DB_PATH = ROOT / "quality.db"

EASTMONEY_H = {"User-Agent": "Mozilla/5.0", "Referer": "https://emweb.securities.eastmoney.com/"}


# ════════════════════════════════════════════════════════════════
#  内部: SQLite 数据库连接
# ════════════════════════════════════════════════════════════════

import logging

logger = logging.getLogger("quant_data_quality")

# 模块级 SQLite 连接复用
_QUALITY_DB_CONN: sqlite3.Connection | None = None
_QUALITY_DB_LOCK = threading.Lock()


def _get_db() -> sqlite3.Connection:
    """连接 quality.db, 模块级单例复用."""
    global _QUALITY_DB_CONN
    if _QUALITY_DB_CONN is not None:
        return _QUALITY_DB_CONN
    with _QUALITY_DB_LOCK:
        if _QUALITY_DB_CONN is not None:
            return _QUALITY_DB_CONN
        _QUALITY_DB_CONN = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=30)
        _QUALITY_DB_CONN.execute("PRAGMA journal_mode=WAL")
        _QUALITY_DB_CONN.execute("PRAGMA busy_timeout=30000")
        _QUALITY_DB_CONN.execute("""
            CREATE TABLE IF NOT EXISTS delisted_stocks (
                symbol TEXT PRIMARY KEY,
                name TEXT,
                delist_date TEXT,
                last_price REAL,
                reason TEXT,
                updated_at TEXT
            )
        """)
        _QUALITY_DB_CONN.execute("""
            CREATE TABLE IF NOT EXISTS quality_cache (
                key TEXT PRIMARY KEY,
                data TEXT,
                updated_at TEXT
            )
        """)
        _QUALITY_DB_CONN.commit()
    return _QUALITY_DB_CONN


# ════════════════════════════════════════════════════════════════
#  1. 退市股票获取
# ════════════════════════════════════════════════════════════════

def get_delisted_stocks() -> list[dict[str, Any]]:
    """从 AkShare 获取退市股票列表.

    缓存到 SQLite: quality.db, 每年更新一次.

    Returns:
        [{symbol, name, delist_date, last_price, reason}, ...]
    """
    # 检查缓存是否在一年内
    conn = _get_db()
    row = conn.execute(
        "SELECT data, updated_at FROM quality_cache WHERE key = 'delisted_stocks'"
    ).fetchone()
    if row:
        try:
            updated = datetime.fromisoformat(row[1])
            if datetime.now(CST) - updated < timedelta(days=365):
                return json.loads(row[0])
        except (ValueError, TypeError):
            pass

    result: list[dict[str, Any]] = []
    source_ok = False

    # 尝试从 AKShare 获取
    try:
        import akshare as ak
        df = ak.stock_delist_em()
        if df is not None and not df.empty:
            source_ok = True
            # 取退市信息
            for _, row_data in df.iterrows():
                sec_code = str(row_data.get("股票代码", "") or row_data.get("代码", "")).strip().zfill(6)
                sec_name = str(row_data.get("股票简称", "") or row_data.get("名称", "")).strip()
                delist_date = str(row_data.get("实施日期", "") or row_data.get("退市日期", "") or "")[:10]
                reason = str(row_data.get("主要原因", "") or row_data.get("原因", "") or "摘牌")
                result.append({
                    "symbol": sec_code,
                    "name": sec_name,
                    "delist_date": delist_date,
                    "last_price": 0.0,
                    "reason": reason,
                })
    except Exception as e:
        logger.error(f"[data_quality] 操作失败: {e}", exc_info=True)

    if not result:
        # 从东方财富获取退市整理期股票
        try:
            url = ("https://push2delay.eastmoney.com/api/qt/clist/get?"
                   "pn=1&pz=500&po=1&np=1&fields=f12,f14,f100,f71,f170"
                   "&fs=m:0+t:6+f:!2,m:0+t:80+f:!2")
            r = requests.get(url, headers=EASTMONEY_H, timeout=10)
            if r.status_code == 200:
                data = r.json().get("data", {}).get("diff", [])
                if data:
                    source_ok = True
                for item in data:
                    result.append({
                        "symbol": str(item.get("f12", "")).zfill(6),
                        "name": str(item.get("f14", "")),
                        "delist_date": "",
                        "last_price": _to_float(item.get("f100")),
                        "reason": str(item.get("f71", "")),
                    })
        except Exception as e:
            logger.error(f"[data_quality] 操作失败: {e}", exc_info=True)

    # 审计 2026-08-16：只有真实拉到非空数据才写成功缓存；
    # 全源失败时空列表不得覆盖已有缓存，避免把“未知”伪装成“无退市股”
    if source_ok and result:
        try:
            conn.execute(
                "INSERT OR REPLACE INTO quality_cache (key, data, updated_at) VALUES (?, ?, ?)",
                ("delisted_stocks", json.dumps(result, ensure_ascii=False),
                 datetime.now(CST).isoformat()),
            )
            conn.commit()
        except Exception as e:
            logger.error(f"[data_quality] 操作失败: {e}", exc_info=True)

    return result


# ════════════════════════════════════════════════════════════════
#  2. 幸存者偏差检查
# ════════════════════════════════════════════════════════════════

def check_survivorship_bias(
    strategy_symbols: list[str],
    backtest_start: str,
) -> list[dict[str, Any]]:
    """检查策略回测中是否存在幸存者偏差.

    Args:
        strategy_symbols: 策略用到的股票代码列表.
        backtest_start: 回测开始日期 (YYYY-MM-DD).

    Returns:
        已退市但仍出现在策略中的股票列表:
        [{symbol, name, delist_date, note}, ...]
    """
    delisted = get_delisted_stocks()
    # P2-Q1-fix (Q1-L016): strategy_symbols 可能含 int，统一 str() 再补零
    strategy_set = {str(s).strip().zfill(6) for s in strategy_symbols}

    # P2-Q1-fix (Q1-M010): 日期比较统一走 pd.to_datetime——
    #   纯字符串比较在 backtest_start 为 "20200101"（YYYYMMDD）而 delist_date 为
    #   "2023-05-10"（YYYY-MM-DD）时格式不一致恒 True，全部退市股被误标。
    bt_start_ts = pd.to_datetime(backtest_start, errors="coerce")
    if pd.isna(bt_start_ts):
        # 调用方传入无法解析的回测起始日 → 可见告警并回退字符串比较（尽力兼容）
        logger.warning(
            "check_survivorship_bias: 无法解析 backtest_start=%r，回退字符串比较",
            backtest_start,
        )

    result: list[dict[str, Any]] = []
    for d in delisted:
        sym = str(d.get("symbol", "")).strip().zfill(6)
        if sym not in strategy_set:
            continue
        delist_date = str(d.get("delist_date", "") or "").strip()
        delist_ts = pd.to_datetime(delist_date, errors="coerce") if delist_date else pd.NaT
        if pd.isna(delist_ts):
            # 无退市日期（如 Eastmoney 兜底源）→ 单独标记，不再被静默跳过
            result.append({
                "symbol": sym,
                "name": d.get("name", ""),
                "delist_date": "",
                "last_price": d.get("last_price", 0.0),
                "reason": d.get("reason", ""),
                "note": f"{sym} 在退市列表但缺少退市日期，无法核对回测区间，请人工确认",
            })
        elif not pd.isna(bt_start_ts):
            if delist_ts >= bt_start_ts:
                result.append({
                    "symbol": sym,
                    "name": d.get("name", ""),
                    "delist_date": delist_date,
                    "last_price": d.get("last_price", 0.0),
                    "reason": d.get("reason", ""),
                    "note": f"策略在 {backtest_start} 后仍使用了已于 {delist_date} 退市的 {sym}",
                })
        else:
            # backtest_start 无法解析：回退字符串比较（与 V5.4 行为兼容）
            if delist_date >= str(backtest_start):
                result.append({
                    "symbol": sym,
                    "name": d.get("name", ""),
                    "delist_date": delist_date,
                    "last_price": d.get("last_price", 0.0),
                    "reason": d.get("reason", ""),
                    "note": f"策略在 {backtest_start} 后仍使用了已于 {delist_date} 退市的 {sym}",
                })

    return result


# ════════════════════════════════════════════════════════════════
#  3. 复权价格验证
# ════════════════════════════════════════════════════════════════

def validate_adjusted_prices(symbol: str) -> dict[str, Any]:
    """验证复权价格的合理性.

    检查:
      - 复权因子单调递增 (不应下降)
      - 日收益率是否异常 (±20% 以上标记)

    Args:
        symbol: 6位股票代码.

    Returns:
        {ok: bool, anomalies: [{date, pct_chg, note}], adjust_factor_ok: bool}
    """
    result: dict[str, Any] = {
        "ok": True,
        "anomalies": [],
        "adjust_factor_ok": True,
        "symbol": symbol,
    }

    try:
        import akshare as ak
        df = ak.stock_zh_a_hist(symbol=symbol, adjust="qfq")
        if df is None or df.empty:
            # Try without adjustment
            df = ak.stock_zh_a_hist(symbol=symbol, adjust="")
            result["ok"] = False
            return result
    except Exception:
        result["ok"] = False
        result["note"] = "无法获取数据"
        return result

    # 列名标准化
    df.columns = [str(c).strip() for c in df.columns]
    # P2-Q1-fix (Q1-L015): 兜底列索引修正。akshare 中文列序为
    #   [日期,股票代码,开盘,收盘,最高,最低,...] → 收盘=3、最高=4、最低=5。
    #   V5.4 版 fallback 用 columns[2]=开盘 / columns[1]=股票代码(非数值)，
    #   兜底路径必然出错被 except 吞掉。
    date_col = "日期" if "日期" in df.columns else df.columns[0]
    close_col = "收盘" if "收盘" in df.columns else None
    high_col = "最高" if "最高" in df.columns else None
    low_col = "最低" if "最低" in df.columns else None
    # 兜底英文列（akshare 中文列序为 [日期,股票代码,开盘,收盘,最高,最低,...]）
    if close_col is None or high_col is None or low_col is None:
        try:
            close_col = close_col if close_col is not None else df.columns[3]
            high_col = high_col if high_col is not None else df.columns[4]
            low_col = low_col if low_col is not None else df.columns[5]
        except IndexError:
            result["ok"] = False
            result["note"] = f"K线列数不足，无法定位价格列: {list(df.columns)}"
            return result
    if close_col not in df.columns or high_col not in df.columns or low_col not in df.columns:
        result["ok"] = False
        result["note"] = f"K线缺少价格列: {list(df.columns)}"
        return result

    closes = df[close_col].values.astype(float)
    dates = df[date_col].astype(str).values

    # P2-Q1-fix (Q1-M007): 涨跌幅异常阈值按板块取（主板±10%/双创±20%/北交所±30%，
    #   ST±5% 需证券简称，本函数无 name 参数 → 主板 ST 暂按 10% 判定）。
    #   V5.4 版恒为 ±20% → 创业板(300/301)/科创板(688) 触发涨跌停即被误报。
    limit_pct = 20.0
    if get_price_limit_pct is not None:
        try:
            _lp = get_price_limit_pct(symbol)
            if _lp and _lp > 0:
                limit_pct = float(_lp)
        except Exception as e:
            logger.error(f"[data_quality] 操作失败: {e}", exc_info=True)

    # 检查日收益率异常
    for i in range(1, len(closes)):
        if closes[i - 1] == 0:
            continue
        pct = (closes[i] / closes[i - 1] - 1) * 100
        # P2-Q1-fix (Q1-M007): 仅当收益率「超出」板块涨跌幅限制才标记（+0.02% 容忍四舍五入），
        #   涨停/跌停价恰好触及限制不被误报。
        if abs(pct) > limit_pct + 0.02:
            result["anomalies"].append({
                "date": str(dates[i])[:10],
                "pct_chg": round(pct, 2),
                "note": f"日收益率 {pct:+.2f}% 异常 ({'除权除息' if abs(pct) > 20 else '极端事件'})",
            })
            result["ok"] = False

    # 检查复权因子单调性 (通过对比不复权价格)
    try:
        df_raw = ak.stock_zh_a_hist(symbol=symbol, adjust="")
        if df_raw is not None and not df_raw.empty:
            raw_closes = df_raw[close_col].values.astype(float)
            # 简单复权因子: qfq_close / raw_close
            factors = [c / r if r != 0 else 1.0 for c, r in zip(closes, raw_closes)]
            for i in range(1, len(factors)):
                if factors[i] < factors[i - 1] - 0.001:  # 允许微小误差
                    result["adjust_factor_ok"] = False
                    result["anomalies"].append({
                        "date": str(dates[i])[:10],
                        "pct_chg": 0.0,
                        "note": f"复权因子下降: {factors[i-1]:.4f} → {factors[i]:.4f}",
                    })
                    result["ok"] = False
                    break
    except Exception as e:
        logger.error(f"[data_quality] 操作失败: {e}", exc_info=True)

    # 检查高开低收一致性
    for i in range(len(closes)):
        try:
            h = float(df[high_col].iloc[i])
            l_ = float(df[low_col].iloc[i])
            c = float(df[close_col].iloc[i])
            if h < l_ or h < c or l_ > c:
                result["anomalies"].append({
                    "date": str(dates[i])[:10],
                    "pct_chg": 0.0,
                    "note": f"价格不一致: 高{h} 低{l_} 收{c}",
                })
                result["ok"] = False
        except Exception as e:
            logger.error(f"[data_quality] 操作失败: {e}", exc_info=True)

    return result


# ════════════════════════════════════════════════════════════════
#  4. 价格异常点检测
# ════════════════════════════════════════════════════════════════

def detect_price_anomalies(
    symbol: str,
    bars: list[list] | None = None,
    z_threshold: float = 4.0,
) -> list[dict[str, Any]]:
    """检测价格异常点 (数据错误或极端事件).

    方法: 计算日收益率的 z-score, 标记 |z| > threshold 的点.

    Args:
        symbol: 股票代码 (仅用于标记).
        bars: K线列表, 每项为 [date, open, high, low, close, volume].
              None 时自动从 akshare 获取.
        z_threshold: z-score 阈值, 默认 4.0.

    Returns:
        [{date, pct_chg, z_score, likely_error: bool}]
    """
    anomalies: list[dict[str, Any]] = []

    if bars is None:
        try:
            import akshare as ak
            df = ak.stock_zh_a_hist(symbol=symbol, adjust="qfq")
            if df is None or df.empty:
                return anomalies
            df.columns = [str(c).strip() for c in df.columns]
            close_col = "收盘" if "收盘" in df.columns else df.columns[2]
            date_col = "日期" if "日期" in df.columns else df.columns[0]
            closes = df[close_col].values.astype(float)
            dates = df[date_col].astype(str).values
        except Exception:
            return anomalies
    else:
        closes = []
        dates = []
        for bar in bars:
            if len(bar) < 5:
                continue
            # P2-Q1-fix (Q1-M008): bars 分支 float(bar[4]) 无保护——
            #   非数值/NaN 时原逻辑直接抛 ValueError 中断整个检测。
            #   （akshare 分支有 astype 兜底，此分支没有）现在跳过非法值。
            try:
                c = float(bar[4])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(c):
                continue
            dates.append(str(bar[0]))
            closes.append(c)

    if len(closes) < 10:
        return anomalies

    # 计算日收益率
    returns = []
    for i in range(1, len(closes)):
        if closes[i - 1] == 0:
            returns.append(0.0)
        else:
            returns.append((closes[i] / closes[i - 1] - 1) * 100)

    if not returns:
        return anomalies

    # 计算 z-score
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)
    std = math.sqrt(variance) if variance > 0 else 1.0

    for i, r in enumerate(returns):
        z = (r - mean) / std if std > 0 else 0.0
        if abs(z) > z_threshold:
            likely_error = abs(r) > 15.0  # >15% 单日很可能是数据错误
            anomalies.append({
                "date": str(dates[i + 1])[:10],
                "pct_chg": round(r, 2),
                "z_score": round(z, 2),
                "likely_error": likely_error,
            })

    return anomalies


# ════════════════════════════════════════════════════════════════
#  5. 缺失日期填充
# ════════════════════════════════════════════════════════════════

def fill_missing_dates(bars: list[list]) -> list[list]:
    """填充缺失交易日.

    规则:
      - 跳过周末和法定节假日
      - 缺失交易日用前一日收盘价填充
      - 确保日期序列连续 (仅限交易日)

    Args:
        bars: K线列表, 每项为 [date, open, high, low, close, volume, ...].
              date 应为 YYYY-MM-DD 格式.

    Returns:
        填充后的 K线列表.
    """
    if not bars:
        return []

    # 排序
    sorted_bars = sorted(bars, key=lambda x: str(x[0]))
    result: list[list] = []

    for i, bar in enumerate(sorted_bars):
        result.append(list(bar))
        if i < len(sorted_bars) - 1:
            cur_date = _parse_date(str(bar[0]))
            next_date = _parse_date(str(sorted_bars[i + 1][0]))
            if cur_date and next_date:
                missing = _get_missing_trading_days(cur_date, next_date)
                for md in missing:
                    fill = list(bar)  # 复制前一日
                    fill[0] = md
                    # 保持开盘=前收, 高低=前收, 收盘=前收, 成交量为0
                    prev_close = bar[4] if len(bar) > 4 else 0.0
                    for j in range(1, min(5, len(fill))):
                        fill[j] = prev_close
                    # P2-Q1-fix (Q1-M009): 停牌合成 bar 的成交相关字段全部置 0。
                    #   V5.4 版 volume 置 0 但 amount(index6)/pct_chg(index7+)
                    #   保留前日值 → 出现「量0额非0、涨跌幅非0」的自相矛盾 bar。
                    for j in range(5, len(fill)):
                        fill[j] = 0  # volume/amount/pct_chg/换手率 等
                    result.append(fill)

    # 重新排序
    result.sort(key=lambda x: str(x[0]))
    return result


def _parse_date(s: str) -> datetime | None:
    """解析日期字符串."""
    raw = s.strip()
    compact = raw.replace("-", "")
    if len(compact) == 8:
        try:
            return datetime.strptime(compact, "%Y%m%d")
        except ValueError:
            pass
    # V11 审计修复（Medium）: 原实现先 replace("-","") 再判断 "-" in s，死分支恒 False，
    # 非补零格式（2024-1-5）永远返回 None。修正为对原始字符串判断。
    if "-" in raw:
        for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d"):
            try:
                return datetime.strptime(raw[:19].strip(), fmt)
            except ValueError:
                continue
    return None


def _is_weekend(d: datetime) -> bool:
    return d.weekday() >= 5


def _get_missing_trading_days(start: datetime, end: datetime) -> list[str]:
    """获取 start 到 end 之间缺失的交易日 (不包含 start 和 end 本身)."""
    missing: list[str] = []
    cur = start + timedelta(days=1)
    cal = _get_trade_calendar()  # V4.1 fix: use A-share trading calendar
    # P2-Q1-fix (Q1-L017): 交易日历获取失败（空 set）时，所有非周末日都会被当作
    #   交易日 → 节假日也会被填成合成 bar。降级语义可见化：告警一次。
    if not cal:
        global _CAL_FALLBACK_WARNED
        if not _CAL_FALLBACK_WARNED:
            logger.warning(
                "交易日历为空，_get_missing_trading_days 降级为「工作日粗筛」——"
                "法定节假日将被当作交易日填充，可能产生合成K线"
            )
            _CAL_FALLBACK_WARNED = True
    while cur < end:
        if not _is_weekend(cur):
            date_str = cur.strftime("%Y-%m-%d")
            if not cal or date_str in cal:  # V4.1 fix: check trading calendar (skip holidays)
                missing.append(date_str)
        cur += timedelta(days=1)
    return missing


# ════════════════════════════════════════════════════════════════
#  内部工具
# ════════════════════════════════════════════════════════════════
