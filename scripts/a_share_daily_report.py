#!/usr/bin/env python3
"""Generate A-share quantitative daily report with fixed data sources.

Data source policy:
1. baostock: index daily/weekly/monthly K lines and technical indicators. Stable, no token.
2. AkShare Sina/Tencent spot: full A-share breadth, equal-weight return, turnover.
3. AkShare exchange summary: SSE/SZSE exchange aggregate cross-check.
4. AkShare THS industry summary: sector turnover and sector return tables.
5. Hot stock sentiment: external hot ranking first, turnover attention pool fallback.
6. Cross-asset/oil: data_sources fallback registry + Nasdaq Data Link where subscribed.
7. Optional Eastmoney direct APIs are deliberately excluded from the main chain because
   this host has shown frequent RemoteDisconnected/long hangs.
"""
from __future__ import annotations
import logging

import argparse
import json
import math
import os
import re
import signal
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from report_delivery import finalize_report
except Exception:  # pragma: no cover - report still must generate if delivery helper is unavailable
    finalize_report = None

try:
    from report_paths import report_path
except Exception:  # pragma: no cover
    report_path = None

try:
    from report_data_base import compact_inputs_for_report
except Exception:  # pragma: no cover
    compact_inputs_for_report = None

try:
    from data_sources import (
        fetch_nasdaq_data_link_dataset,
        get_cross_asset_fallback_sources,
        get_international_oil_sources,
        get_nasdaq_api,
    )
except Exception:  # pragma: no cover
    fetch_nasdaq_data_link_dataset = None
    get_cross_asset_fallback_sources = None
    get_international_oil_sources = None
    get_nasdaq_api = None

INDEX_CODES = {
    "上证指数": "sh.000001",
    "深证成指": "sz.399001",
    "创业板指": "sz.399006",
    "沪深300": "sh.000300",
    "中证500": "sh.000905",
    "中证1000": "sh.000852",
    "国证2000": "sz.399303",
    "红利指数": "sh.000015",
    "万得微盘股": "w.8841431",  # V4.1 arch fix: Wind微盘股指数，通过akshare fallback获取

}

TMT_INDUSTRY_PREFIXES = ("C39", "I63", "I64", "I65", "R86", "R87")
TMT_INDUSTRY_NOTE = "证监会行业分类：C39/I63/I64/I65/R86/R87"

# V4.1 arch fix: 微盘股指映射从国证2000改为万得微盘股指数8841431
# 国证2000实际对应小盘风格，万得微盘股指数8841431才是真正的微盘股
# baostock不直接支持Wind指数代码，通过fetch_wind_micro_cap_index fallback获取
STYLE_INDEX_NAMES = {
    "大盘股": "沪深300",
    "小盘股": "中证1000",
    "微盘股": "万得微盘股",
}

# ===== D7 真源登记（用户硬规则口径，唯一真源，禁止改动数值/口径逻辑）=====
# 1) 去ST: filtered_a_share（过滤 ST/*ST + 北交所）
# 2) 全A等权: breadth_metrics（全A等权涨跌幅%/涨跌幅中位数/涨停占比，量能口径=成交量合计/成交额(亿)）
# 3) 热股等权: fetch_hot_rank_equally_weighted + fallback_hot_attention_pool（东财热股榜前50/成交额前N）
# 4) 融资TMT: fetch_margin_metrics（TMT_INDUSTRY_PREFIXES=C39/I63/I64/I65/R86/R87，全A融资/TMT融资占比）
# 5) 风格: STYLE_INDEX_NAMES + _segment_section（主板/双创/北交所+风格分层）+ build_technical_model/score_technical_row
# 登记日期: 2026-08-11（D7）。任何收敛到公共模块的动作只允许转发 import，不得复制或改写下列函数。
# ====================================================================

ROOT = Path(os.environ.get("QUANT_ROOT", str(Path(__file__).resolve().parent.parent)))
DATA_DIR = ROOT / "generated" / "a_share_data"
REPORT_DATA_BASE_LATEST = ROOT / "generated" / "report_data_base" / "latest.json"
REPORT_DIR = Path(os.environ.get("QUANT_DESKTOP", str(Path.home() / "Desktop")) + "/每日任务/A股量化日报")

# 请求级缓存：同一 akshare 接口+参数在 TTL 内重复调用直接命中缓存，减少网络重复请求。
# 单栈化(V3)：改走 quant_platform.legacy 薄转发，不再直接依赖 quant_v6。
# 缓存不可用（导入失败）时自动降级为直接调用，输出结构/内容不变。
try:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from quant_platform.legacy import cached_fetch
except Exception:
    cached_fetch = None

try:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from quant_system.analysis_core.data_contract import format_pct_value
except Exception:  # pragma: no cover - standalone script may run outside quant_system checkout
    format_pct_value = None

# 默认当日有效（6 小时）：历史行情/日频数据可复用；实时快照如需更新可调小该值。
_AK_CACHE_TTL_SECONDS = 6 * 3600


def _ak_fetch(api_name: str, *args, ttl_seconds: int = _AK_CACHE_TTL_SECONDS, **kwargs) -> Any:
    """akshare 调用 + 请求级缓存：同接口+参数在 TTL 内不重复发起网络请求。"""
    import akshare as ak

    fn = getattr(ak, api_name, None)
    if fn is None:
        raise AttributeError(f"akshare 无此接口: {api_name}")
    if cached_fetch is None:
        return fn(*args, **kwargs)
    return cached_fetch(fn, *args, ttl_seconds=ttl_seconds, **kwargs)


class Timeout(Exception):
    pass


class time_limit:
    def __init__(self, seconds: int):
        self.seconds = seconds
        self.prev = None
        self._supported = hasattr(signal, "SIGALRM")

    def __enter__(self):
        if self._supported:
            self.prev = signal.signal(signal.SIGALRM, self._handler)
            signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc, tb):
        if self._supported:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self.prev)
        return False

    def _handler(self, signum, frame):
        raise Timeout(f"operation timed out after {self.seconds}s")


@dataclass
class SourceStatus:
    name: str
    ok: bool
    ms: float
    detail: str


def safe_float(v: Any) -> float | None:
    try:
        if v is None or v == "" or (isinstance(v, float) and math.isnan(v)):
            return None
        return float(str(v).replace(",", ""))
    except Exception:
        return None


def limit_threshold_for_code(code_series: pd.Series) -> pd.Series:
    """Determine the appropriate limit-up/-down threshold for each stock by code prefix.

    - 科创板 (688xxx): 20%
    - 创业板 (300xxx, 301xxx): 20%
    - 北交所 (8xxxxx, 4xxxxx, 920xxx): 30%, normally filtered out already
    - 主板 (60xxxx, 00xxxx, 001xxx, 002xxx, 003xxx): 10%

    Returns a Series with the absolute threshold value (e.g. 9.8 for 10%% stocks, 19.6 for 20%% stocks).
    """
    codes = code_series.astype(str).str.strip().str.lower()
    codes = codes.str.replace(r"^(sh|sz|bj)", "", regex=True)
    is_high_limit = codes.str.match(r"^(688|300|301)\d{3}$")
    result = pd.Series(9.8, index=code_series.index)
    result[is_high_limit] = 19.6
    return result


def compute_technical_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add common technical indicators using only pandas.

    Indicators are computed from OHLCV and are safe for daily/weekly/monthly K lines.
    Missing columns simply produce missing outputs rather than failing the report.
    """
    out = df.copy()
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    close = out.get("close")
    high = out.get("high")
    low = out.get("low")
    if close is None:
        return out

    for n in [5, 10, 20, 30, 60, 120, 144, 200, 250, 300]:
        out[f"ma{n}"] = close.rolling(n).mean()
        out[f"bias{n}"] = (close / out[f"ma{n}"] - 1) * 100

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd_dif"] = ema12 - ema26
    out["macd_dea"] = out["macd_dif"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = (out["macd_dif"] - out["macd_dea"]) * 2

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    for n in [6, 12, 14, 24]:
        avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
        avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
        rs = avg_gain / avg_loss.replace(0, pd.NA)
        out[f"rsi{n}"] = 100 - 100 / (1 + rs)

    if high is not None and low is not None:
        low9 = low.rolling(9).min()
        high9 = high.rolling(9).max()
        rsv = (close - low9) / (high9 - low9).replace(0, pd.NA) * 100
        out["kdj_k"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
        out["kdj_d"] = out["kdj_k"].ewm(alpha=1 / 3, adjust=False).mean()
        out["kdj_j"] = 3 * out["kdj_k"] - 2 * out["kdj_d"]

        typical = (high + low + close) / 3
        ma_typical = typical.rolling(14).mean()
        md = (typical - ma_typical).abs().rolling(14).mean()
        out["cci14"] = (typical - ma_typical) / (0.015 * md.replace(0, pd.NA))
        out["atr14"] = pd.concat([
            (high - low),
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1).rolling(14).mean()

    if "amount" in out.columns:
        out["amount_ma5"] = out["amount"].rolling(5).mean()
        out["amount_ma20"] = out["amount"].rolling(20).mean()
    return out


def technical_signal(row: pd.Series) -> str:
    parts: list[str] = []
    macd_hist = safe_float(row.get("macd_hist"))
    macd_dif = safe_float(row.get("macd_dif"))
    macd_dea = safe_float(row.get("macd_dea"))
    rsi14 = safe_float(row.get("rsi14"))
    k = safe_float(row.get("kdj_k"))
    d = safe_float(row.get("kdj_d"))
    cci = safe_float(row.get("cci14"))
    bias144 = safe_float(row.get("bias144"))
    bias300 = safe_float(row.get("bias300"))
    if macd_hist is not None and macd_dif is not None and macd_dea is not None:
        parts.append("MACD多头" if macd_dif > macd_dea and macd_hist > 0 else ("MACD空头" if macd_dif < macd_dea and macd_hist < 0 else "MACD钝化/收敛"))
    if rsi14 is not None:
        parts.append("RSI过热" if rsi14 >= 70 else ("RSI弱势" if rsi14 <= 35 else "RSI中性"))
    if k is not None and d is not None:
        parts.append("KDJ偏强" if k > d and k < 85 else ("KDJ超买" if k >= 85 else ("KDJ偏弱" if k < d else "KDJ中性")))
    if cci is not None:
        parts.append("CCI强趋势" if cci >= 100 else ("CCI弱趋势" if cci <= -100 else "CCI震荡"))
    if bias144 is not None and bias300 is not None:
        parts.append(
            f"距144/300MA {format_pct_value(bias144, digits=1, unit='pct')}/"
            f"{format_pct_value(bias300, digits=1, unit='pct')}"
        )
    return "；".join(parts) if parts else "缺失"


def _round_last(row: pd.Series, key: str, ndigits: int = 2) -> Any:
    v = safe_float(row.get(key))
    return round(v, ndigits) if v is not None else None

def _fetch_index_k_local(end_date: str, start_date: str = "2024-01-01") -> list[dict]:
    """从本地仓库读指数K线（kline/{code}.parquet），本地最新到 end_date 才返回。

    指数代码映射（baostock → 本地文件名）：
      sh.000001→000001, sz.399001→399001, sh.000300→000300 ...
    本地文件无 amount（腾讯源指数仅 date/open/high/low/close/volume）时，
    成交额/量比置 None，其余指标照常计算。
    """
    import re as _re
    from pathlib import Path as _Path
    rows: list[dict] = []
    kline_dir = ROOT / "data_warehouse" / "kline"
    target = pd.Timestamp(end_date)
    # 本地指数白名单：仅 000300 为真实指数文件（服务器导出的 index_000300.parquet）
    # ⚠️ 000001/000016/000905/000852/000688 等本地文件均为个股数据（代码与个股冲突），
    #    误用会污染报告（如 000905=厦门港务 8.54 元，中证500 应为 ~6000 点）
    LOCAL_INDEX_SAFE = {"000300"}
    for name, code in INDEX_CODES.items():
        if code.startswith("w."):
            continue  # 万得微盘股无本地文件
        m = _re.search(r"(\d{6})", code)
        if not m:
            continue
        local_code = m.group(1)
        if local_code not in LOCAL_INDEX_SAFE:
            continue  # 与个股代码冲突的指数不用本地，走 baostock
        p = kline_dir / f"{local_code}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if df.empty or "date" not in df.columns:
            continue
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        if df["date"].max() < target:
            continue  # 本地滞后于目标日，交给 baostock 补充
        df = df.sort_values("date")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["pct_calc"] = df["close"].pct_change() * 100
        df = compute_technical_frame(df)
        last = df.iloc[-1]
        rows.append({
            "指数": name, "代码": code, "数据源": "warehouse.kline.local",
            "收盘": round(float(last["close"]), 2),
            "涨跌幅%": round(float(last["pct_calc"]), 2) if pd.notna(last["pct_calc"]) else None,
            "成交额(亿)": round(float(last["amount"]) / 1e8, 1) if "amount" in df.columns and pd.notna(last["amount"]) else None,
            "量比20日": None,
            "MACD_DIF": _round_last(last, "macd_dif"),
            "MACD_DEA": _round_last(last, "macd_dea"),
            "MACD柱": _round_last(last, "macd_hist"),
            "RSI6": _round_last(last, "rsi6"),
            "RSI14": _round_last(last, "rsi14"),
            "KDJ_K": _round_last(last, "kdj_k"),
            "KDJ_D": _round_last(last, "kdj_d"),
            "KDJ_J": _round_last(last, "kdj_j"),
            "CCI14": _round_last(last, "cci14"),
            "ATR14": _round_last(last, "atr14"),
            "技术信号": technical_signal(last),
            "MA5": round(float(last["ma5"]), 2) if pd.notna(last.get("ma5")) else None,
            "MA20": round(float(last["ma20"]), 2) if pd.notna(last.get("ma20")) else None,
            "MA60": round(float(last["ma60"]), 2) if pd.notna(last.get("ma60")) else None,
            "MA144": round(float(last["ma144"]), 2) if pd.notna(last.get("ma144")) else None,
            "MA300": round(float(last["ma300"]), 2) if pd.notna(last.get("ma300")) else None,
            "距MA20%": round((float(last["close"]) / float(last["ma20"]) - 1) * 100, 2) if pd.notna(last.get("ma20")) and last.get("ma20") else None,
            "距MA60%": round((float(last["close"]) / float(last["ma60"]) - 1) * 100, 2) if pd.notna(last.get("ma60")) and last.get("ma60") else None,
            "距MA144%": round((float(last["close"]) / float(last["ma144"]) - 1) * 100, 2) if pd.notna(last.get("ma144")) and last.get("ma144") else None,
            "距MA300%": round((float(last["close"]) / float(last["ma300"]) - 1) * 100, 2) if pd.notna(last.get("ma300")) and last.get("ma300") else None,
            "年内高点回撤%": round((float(last["close"]) / float(df["close"].max()) - 1) * 100, 2),
            "年内低点反弹%": round((float(last["close"]) / float(df["close"].min()) - 1) * 100, 2),
        })
    return rows


def fetch_index_k(end_date: str, start_date: str = "2024-01-01") -> tuple[pd.DataFrame, SourceStatus]:
    t0 = time.time()

    # ── V3 (2026-08-07): 本地仓库优先（数据基座榨干）──
    #   kline/000300.parquet 为真实指数文件（服务器导出）；本地命中先占位，
    #   其余指数仍走 baostock 补充，避免本地白名单外指数缺失。
    local_rows: list[dict] = []
    try:
        local_rows = _fetch_index_k_local(end_date, start_date)
    except Exception as _le:  # noqa: BLE001
        sys.stderr.write(f"本地指数K线读取失败: {_le}\n")
    local_codes = {r["代码"] for r in local_rows}

    import baostock as bs

    rows = list(local_rows)  # 本地行先入
    bs_login_ok = False
    lg = bs.login()
    try:
        if lg.error_code == "0":
            bs_login_ok = True
        else:
            sys.stderr.write(f"baostock.login 失败({lg.error_msg})，指数走 akshare 兜底\n")
        for name, code in INDEX_CODES.items():
            if not bs_login_ok:
                break
            if code in local_codes:
                continue  # 本地已命中，跳过 baostock
            rs = bs.query_history_k_data_plus(
                code,
                "date,code,open,high,low,close,volume,amount,pctChg",
                start_date=start_date,
                end_date=f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}",
                frequency="d",
            )
            data = []
            while rs.next():
                r = rs.get_row_data()
                data.append({
                    "date": r[0], "code": r[1], "open": safe_float(r[2]),
                    "high": safe_float(r[3]), "low": safe_float(r[4]), "close": safe_float(r[5]),
                    "volume": safe_float(r[6]), "amount": safe_float(r[7]), "pct": safe_float(r[8]),
                })
            df = pd.DataFrame(data)
            if df.empty:
                rows.append({"指数": name, "代码": code, "错误": "empty"})
                continue
            for col in ["close", "amount", "pct"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            # baostock 的 pctChg 在本环境出现过日期/符号错位，指数涨跌幅统一按前一
            # 有效交易日收盘价本地重算，避免把收涨日写成收跌。
            df["pct_calc"] = df["close"].pct_change() * 100
            df = compute_technical_frame(df)
            last = df.iloc[-1]
            rows.append({
                "指数": name,
                "代码": code,
                "数据源": "baostock.index_k",
                "收盘": round(float(last["close"]), 2),
                "涨跌幅%": round(float(last["pct_calc"]), 2) if pd.notna(last["pct_calc"]) else None,
                "成交额(亿)": round(float(last["amount"]) / 1e8, 1) if pd.notna(last["amount"]) else None,
                "量比20日": round(float(last["amount"] / last["amount_ma20"]), 2) if pd.notna(last["amount_ma20"]) and last["amount_ma20"] else None,
                "MACD_DIF": _round_last(last, "macd_dif"),
                "MACD_DEA": _round_last(last, "macd_dea"),
                "MACD柱": _round_last(last, "macd_hist"),
                "RSI6": _round_last(last, "rsi6"),
                "RSI14": _round_last(last, "rsi14"),
                "KDJ_K": _round_last(last, "kdj_k"),
                "KDJ_D": _round_last(last, "kdj_d"),
                "KDJ_J": _round_last(last, "kdj_j"),
                "CCI14": _round_last(last, "cci14"),
                "ATR14": _round_last(last, "atr14"),
                "技术信号": technical_signal(last),
                "MA5": round(float(last["ma5"]), 2) if pd.notna(last["ma5"]) else None,
                "MA20": round(float(last["ma20"]), 2) if pd.notna(last["ma20"]) else None,
                "MA60": round(float(last["ma60"]), 2) if pd.notna(last["ma60"]) else None,
                "MA144": round(float(last["ma144"]), 2) if pd.notna(last["ma144"]) else None,
                "MA300": round(float(last["ma300"]), 2) if pd.notna(last["ma300"]) else None,
                "距MA20%": round((float(last["close"]) / float(last["ma20"]) - 1) * 100, 2) if pd.notna(last["ma20"]) and last["ma20"] else None,
                "距MA60%": round((float(last["close"]) / float(last["ma60"]) - 1) * 100, 2) if pd.notna(last["ma60"]) and last["ma60"] else None,
                "距MA144%": round((float(last["close"]) / float(last["ma144"]) - 1) * 100, 2) if pd.notna(last["ma144"]) and last["ma144"] else None,
                "距MA300%": round((float(last["close"]) / float(last["ma300"]) - 1) * 100, 2) if pd.notna(last["ma300"]) and last["ma300"] else None,
                "年内高点回撤%": round((float(last["close"]) / float(df["close"].max()) - 1) * 100, 2),
                "年内低点反弹%": round((float(last["close"]) / float(df["close"].min()) - 1) * 100, 2),
            })
    finally:
        try:
            bs.logout()
        except Exception as e:
            logging.getLogger(__name__).error(f"[a_share_daily_report] 操作失败: {e}", exc_info=True)

    # V4.2 fix: baostock 晚间不稳定（网络接收错误）时，用 akshare 补齐缺失指数，
    # 保证日报在主源挂掉时仍能生成完整指数章节。
    got_codes = {r.get("代码") for r in rows}
    for name, code in INDEX_CODES.items():
        if code in got_codes:
            continue
        if code.startswith("w."):
            continue  # 万得微盘股走 fetch_wind_micro_cap_index 专用兜底
        m = re.search(r"(\d{6})", code)
        if not m:
            continue
        sym = ("sh" if code.startswith("sh") else "sz") + m.group(1)
        try:
            with time_limit(20):
                adf = _ak_fetch("stock_zh_index_daily", symbol=sym)
            if adf is None or adf.empty:
                continue
            adf = adf.copy()
            adf["date"] = pd.to_datetime(adf["date"]).dt.strftime("%Y%m%d")
            adf = adf[adf["date"] <= end_date]
            if adf.empty or str(adf["date"].iloc[-1]) != end_date:
                continue
            for col in ["open", "high", "low", "close", "volume"]:
                adf[col] = pd.to_numeric(adf[col], errors="coerce")
            adf["pct"] = adf["close"].pct_change() * 100
            adf = compute_technical_frame(adf)
            last = adf.iloc[-1]
            rows.append({
                "指数": name, "代码": code, "数据源": f"akshare.stock_zh_index_daily({sym})",
                "收盘": round(float(last["close"]), 2),
                "涨跌幅%": round(float(last["pct"]), 2) if pd.notna(last["pct"]) else None,
                "成交额(亿)": None, "量比20日": None,
                "MACD_DIF": _round_last(last, "macd_dif"),
                "MACD_DEA": _round_last(last, "macd_dea"),
                "MACD柱": _round_last(last, "macd_hist"),
                "RSI6": _round_last(last, "rsi6"),
                "RSI14": _round_last(last, "rsi14"),
                "KDJ_K": _round_last(last, "kdj_k"),
                "KDJ_D": _round_last(last, "kdj_d"),
                "KDJ_J": _round_last(last, "kdj_j"),
                "CCI14": _round_last(last, "cci14"),
                "ATR14": _round_last(last, "atr14"),
                "技术信号": technical_signal(last),
                "MA5": round(float(last["ma5"]), 2) if pd.notna(last.get("ma5")) else None,
                "MA20": round(float(last["ma20"]), 2) if pd.notna(last.get("ma20")) else None,
                "MA60": round(float(last["ma60"]), 2) if pd.notna(last.get("ma60")) else None,
                "MA144": round(float(last["ma144"]), 2) if pd.notna(last.get("ma144")) else None,
                "MA300": round(float(last["ma300"]), 2) if pd.notna(last.get("ma300")) else None,
                "距MA20%": round((float(last["close"]) / float(last["ma20"]) - 1) * 100, 2) if pd.notna(last.get("ma20")) and last.get("ma20") else None,
                "距MA60%": round((float(last["close"]) / float(last["ma60"]) - 1) * 100, 2) if pd.notna(last.get("ma60")) and last.get("ma60") else None,
                "距MA144%": round((float(last["close"]) / float(last["ma144"]) - 1) * 100, 2) if pd.notna(last.get("ma144")) and last.get("ma144") else None,
                "距MA300%": round((float(last["close"]) / float(last["ma300"]) - 1) * 100, 2) if pd.notna(last.get("ma300")) and last.get("ma300") else None,
                "年内高点回撤%": round((float(last["close"]) / float(adf["close"].max()) - 1) * 100, 2),
                "年内低点反弹%": round((float(last["close"]) / float(adf["close"].min()) - 1) * 100, 2),
            })
        except Exception as _afe:  # noqa: BLE001
            sys.stderr.write(f"akshare 指数兜底失败 {name} {sym}: {_afe}\n")

    out = pd.DataFrame(rows)
    src = "baostock.index_k+akshare.fallback" if not bs_login_ok or any(r.get("数据源", "").startswith("akshare") for r in rows) else "baostock.index_k"
    return out, SourceStatus(src, not out.empty, (time.time() - t0) * 1000, f"rows={len(out)}")


def fetch_wind_micro_cap_index(end_date: str) -> tuple[dict[str, Any] | None, SourceStatus]:
    """Fetch Wind micro-cap index (8841431) via akshare.

    V4.1 arch fix: 万得微盘股指数不在baostock覆盖范围内，
    通过akshare的指数行情接口获取。
    """
    t0 = time.time()
    try:
        with time_limit(15):
            # Try multiple code formats as Wind indices may use different conventions
            df = None
            for sym in ("sz8841431", "sh8841431"):
                try:
                    df = _ak_fetch("stock_zh_index_daily", symbol=sym)
                    if df is not None and not df.empty:
                        break
                except Exception as e:
                    logging.getLogger(__name__).error(f"[a_share_daily_report] 操作失败: {e}", exc_info=True)
                    continue
            if df is None or df.empty:
                return None, SourceStatus("akshare.wind_micro_cap", False, (time.time() - t0) * 1000, "empty from all code formats")
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
        df = df[df["date"] <= end_date]
        if df.empty:
            return None, SourceStatus("akshare.wind_micro_cap", False, (time.time() - t0) * 1000, f"no rows <= {end_date}")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["pct"] = df["close"].pct_change() * 100
        df["ma5"] = df["close"].rolling(5).mean()
        df["ma20"] = df["close"].rolling(20).mean()
        df["ma60"] = df["close"].rolling(60).mean()
        df["ma144"] = df["close"].rolling(144).mean()
        df["ma300"] = df["close"].rolling(300).mean()
        last = df.iloc[-1]
        row = {
            "指数": "万得微盘股",
            "代码": "w.8841431",
            "收盘": round(float(last["close"]), 2),
            "涨跌幅%": round(float(last["pct"]), 2) if pd.notna(last["pct"]) else None,
            "成交额(亿)": "缺失",
            "量比20日": "缺失",
            "MA5": round(float(last["ma5"]), 2) if pd.notna(last["ma5"]) else None,
            "MA20": round(float(last["ma20"]), 2) if pd.notna(last["ma20"]) else None,
            "MA60": round(float(last["ma60"]), 2) if pd.notna(last["ma60"]) else None,
            "MA144": round(float(last["ma144"]), 2) if pd.notna(last["ma144"]) else None,
            "MA300": round(float(last["ma300"]), 2) if pd.notna(last["ma300"]) else None,
            "距MA20%": round((float(last["close"]) / float(last["ma20"]) - 1) * 100, 2) if pd.notna(last["ma20"]) and last["ma20"] else None,
            "距MA60%": round((float(last["close"]) / float(last["ma60"]) - 1) * 100, 2) if pd.notna(last["ma60"]) and last["ma60"] else None,
            "距MA144%": round((float(last["close"]) / float(last["ma144"]) - 1) * 100, 2) if pd.notna(last["ma144"]) and last["ma144"] else None,
            "距MA300%": round((float(last["close"]) / float(last["ma300"]) - 1) * 100, 2) if pd.notna(last["ma300"]) and last["ma300"] else None,
            "年内高点回撤%": round((float(last["close"]) / float(df["close"].max()) - 1) * 100, 2),
            "年内低点反弹%": round((float(last["close"]) / float(df["close"].min()) - 1) * 100, 2),
            "数据源": "akshare.stock_zh_index_daily(w.8841431)",
        }
        return row, SourceStatus("akshare.wind_micro_cap", True, (time.time() - t0) * 1000, f"rows={len(df)}")
    except Exception as e:
        return None, SourceStatus("akshare.wind_micro_cap", False, (time.time() - t0) * 1000, repr(e)[:160])


def fetch_index_fallback_sina(index_name: str, symbol: str, end_date: str) -> tuple[dict[str, Any] | None, SourceStatus]:
    """Fallback for index series missing in baostock, currently used for 科创50."""
    t0 = time.time()
    try:
        with time_limit(15):
            df = _ak_fetch("stock_zh_index_daily", symbol=symbol)
        if df.empty:
            return None, SourceStatus(f"akshare.stock_zh_index_daily.{symbol}", False, (time.time() - t0) * 1000, "empty")
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
        df = df[df["date"] <= end_date]
        if df.empty:
            return None, SourceStatus(f"akshare.stock_zh_index_daily.{symbol}", False, (time.time() - t0) * 1000, f"no rows <= {end_date}")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["pct"] = df["close"].pct_change() * 100
        df["ma5"] = df["close"].rolling(5).mean()
        df["ma20"] = df["close"].rolling(20).mean()
        df["ma60"] = df["close"].rolling(60).mean()
        df["ma144"] = df["close"].rolling(144).mean()
        df["ma300"] = df["close"].rolling(300).mean()
        last = df.iloc[-1]
        if last["date"] != end_date:
            return None, SourceStatus(f"akshare.stock_zh_index_daily.{symbol}", False, (time.time() - t0) * 1000, f"latest={last['date']}, want={end_date}")
        row = {
            "指数": index_name,
            "代码": "sh.000688",
            "收盘": round(float(last["close"]), 2),
            "涨跌幅%": round(float(last["pct"]), 2) if pd.notna(last["pct"]) else None,
                "成交额(亿)": "缺失",
                "量比20日": "缺失",
            "MA5": round(float(last["ma5"]), 2) if pd.notna(last["ma5"]) else None,
            "MA20": round(float(last["ma20"]), 2) if pd.notna(last["ma20"]) else None,
            "MA60": round(float(last["ma60"]), 2) if pd.notna(last["ma60"]) else None,
            "MA144": round(float(last["ma144"]), 2) if pd.notna(last["ma144"]) else None,
            "MA300": round(float(last["ma300"]), 2) if pd.notna(last["ma300"]) else None,
            "距MA20%": round((float(last["close"]) / float(last["ma20"]) - 1) * 100, 2) if pd.notna(last["ma20"]) and last["ma20"] else None,
            "距MA60%": round((float(last["close"]) / float(last["ma60"]) - 1) * 100, 2) if pd.notna(last["ma60"]) and last["ma60"] else None,
            "距MA144%": round((float(last["close"]) / float(last["ma144"]) - 1) * 100, 2) if pd.notna(last["ma144"]) and last["ma144"] else None,
            "距MA300%": round((float(last["close"]) / float(last["ma300"]) - 1) * 100, 2) if pd.notna(last["ma300"]) and last["ma300"] else None,
            "年内高点回撤%": round((float(last["close"]) / float(df["close"].max()) - 1) * 100, 2),
            "年内低点反弹%": round((float(last["close"]) / float(df["close"].min()) - 1) * 100, 2),
            "数据源": f"akshare.stock_zh_index_daily({symbol})",
        }
        return row, SourceStatus(f"akshare.stock_zh_index_daily.{symbol}", True, (time.time() - t0) * 1000, f"rows={len(df)}")
    except Exception as e:
        return None, SourceStatus(f"akshare.stock_zh_index_daily.{symbol}", False, (time.time() - t0) * 1000, repr(e)[:160])


def format_metric(v: Any, suffix: str = "") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v):
        return "缺失"
    if suffix == "%" and format_pct_value is not None:
        return format_pct_value(v, digits=2, unit="pct")
    return f"{v}{suffix}"



def fetch_index_multi_period_technicals(end_date: str, start_date: str = "2018-01-01") -> tuple[pd.DataFrame, SourceStatus]:
    """Fetch daily/weekly/monthly index technicals from baostock.

    This is intentionally index-level: broad market timing needs stable K lines more
    than noisy per-stock indicators. If baostock fails, the report continues with
    daily index table and records the failure in fallback status.
    """
    t0 = time.time()

    # ── V3 (2026-08-07): 本地优先（数据基座榨干）+ 总超时保护 ──
    #   baostock 27 次查询（3频率×9指数）晚上易 hang；先用本地日线合成周/月线，
    #   全部本地命中则零网络；否则 baostock 补充并加 90s 硬超时。
    try:
        local_rows = _fetch_index_multi_period_local(end_date, start_date)
        # 本地命中全部 3 频率 × 主要指数才算成功
        if local_rows and len(local_rows) >= 9:
            return pd.DataFrame(local_rows), SourceStatus(
                "warehouse.kline.index_local_multi", True, (time.time() - t0) * 1000,
                f"rows={len(local_rows)} (本地指数日/周/月线合成)")
    except Exception as _le:  # noqa: BLE001
        sys.stderr.write(f"本地多周期指数技术指标失败: {_le}\n")

    import baostock as bs
    import signal as _sig

    def _alarm_handler(signum, frame):
        raise TimeoutError("baostock multi-period technicals timeout")

    old_handler = None
    if hasattr(_sig, "SIGALRM"):
        old_handler = _sig.signal(_sig.SIGALRM, _alarm_handler)
        _sig.alarm(90)  # 90s 硬超时
    rows: list[dict[str, Any]] = []
    freq_map = {"日线": "d", "周线": "w", "月线": "m"}
    lg = bs.login()
    try:
        if lg.error_code != "0":
            return pd.DataFrame(), SourceStatus("baostock.index_multi_period_technical", False, 0, lg.error_msg)
        end = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
        for period_name, freq in freq_map.items():
            for name, code in INDEX_CODES.items():
                rs = bs.query_history_k_data_plus(
                    code,
                    "date,code,open,high,low,close,volume,amount",
                    start_date=start_date,
                    end_date=end,
                    frequency=freq,
                )
                data = []
                while rs.next():
                    r = rs.get_row_data()
                    data.append({
                        "date": r[0], "code": r[1], "open": safe_float(r[2]),
                        "high": safe_float(r[3]), "low": safe_float(r[4]), "close": safe_float(r[5]),
                        "volume": safe_float(r[6]), "amount": safe_float(r[7]),
                    })
                df = pd.DataFrame(data)
                if df.empty or len(df) < 30:
                    rows.append({"周期": period_name, "指数": name, "代码": code, "技术信号": "K线不足/缺失"})
                    continue
                df = compute_technical_frame(df)
                df["pct_calc"] = df["close"].pct_change() * 100
                last = df.iloc[-1]
                rows.append({
                    "周期": period_name,
                    "指数": name,
                    "代码": code,
                    "日期": last.get("date"),
                    "收盘": _round_last(last, "close"),
                    "涨跌幅%": _round_last(last, "pct_calc"),
                    "MA20": _round_last(last, "ma20"),
                    "MA60": _round_last(last, "ma60"),
                    "MA144": _round_last(last, "ma144"),
                    "MA300": _round_last(last, "ma300"),
                    "距MA20%": _round_last(last, "bias20"),
                    "距MA60%": _round_last(last, "bias60"),
                    "距MA144%": _round_last(last, "bias144"),
                    "距MA300%": _round_last(last, "bias300"),
                    "MACD_DIF": _round_last(last, "macd_dif"),
                    "MACD_DEA": _round_last(last, "macd_dea"),
                    "MACD柱": _round_last(last, "macd_hist"),
                    "RSI6": _round_last(last, "rsi6"),
                    "RSI14": _round_last(last, "rsi14"),
                    "KDJ_K": _round_last(last, "kdj_k"),
                    "KDJ_D": _round_last(last, "kdj_d"),
                    "KDJ_J": _round_last(last, "kdj_j"),
                    "CCI14": _round_last(last, "cci14"),
                    "ATR14": _round_last(last, "atr14"),
                    "技术信号": technical_signal(last),
                })
    finally:
        if hasattr(_sig, "SIGALRM"):
            _sig.alarm(0)
            try:
                _sig.signal(_sig.SIGALRM, old_handler)
            except Exception as e:
                logging.getLogger(__name__).error(f"[a_share_daily_report] 操作失败: {e}", exc_info=True)
        try:
            bs.logout()
        except Exception as e:
            logging.getLogger(__name__).error(f"[a_share_daily_report] 操作失败: {e}", exc_info=True)
    out = pd.DataFrame(rows)
    return out, SourceStatus("baostock.index_multi_period_technical", not out.empty, (time.time() - t0) * 1000, f"rows={len(out)}")


def _fetch_index_multi_period_local(end_date: str, start_date: str = "2018-01-01") -> list[dict]:
    """从本地仓库指数日线合成日/周/月技术指标（零网络）。

    本地日线（kline/{code}.parquet）足够长时，用 resample 合成周/月线，
    避免 baostock 27 次网络查询（晚上易 hang）。
    """
    import re as _re
    rows: list[dict] = []
    kline_dir = ROOT / "data_warehouse" / "kline"
    target = pd.Timestamp(end_date)
    freq_map = {"日线": "D", "周线": "W-FRI", "月线": "ME"}
    # 本地指数白名单：仅 000300 为真实指数（000001/000905 等与个股代码冲突，误用会污染报告）
    LOCAL_INDEX_SAFE = {"000300"}
    for name, code in INDEX_CODES.items():
        if code.startswith("w."):
            continue
        m = _re.search(r"(\d{6})", code)
        if not m:
            continue
        if m.group(1) not in LOCAL_INDEX_SAFE:
            continue
        p = kline_dir / f"{m.group(1)}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if df.empty or "date" not in df.columns:
            continue
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        if df["date"].max() < target:
            continue
        df = df.sort_values("date").set_index("date")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        for period_name, rule in freq_map.items():
            try:
                if rule == "D":
                    rdf = df.reset_index()
                else:
                    rdf = df.resample(rule).agg({"open": "first", "high": "max",
                                                 "low": "min", "close": "last",
                                                 "volume": "sum"}).dropna(subset=["close"]).reset_index()
                if len(rdf) < 30:
                    rows.append({"周期": period_name, "指数": name, "代码": code, "技术信号": "K线不足/缺失"})
                    continue
                tdf = compute_technical_frame(rdf)
                tdf["pct_calc"] = tdf["close"].pct_change() * 100
                last = tdf.iloc[-1]
                rows.append({
                    "周期": period_name, "指数": name, "代码": code, "数据源": "warehouse.kline.local",
                    "日期": str(last.get("date"))[:10],
                    "收盘": _round_last(last, "close"),
                    "涨跌幅%": _round_last(last, "pct_calc"),
                    "MA20": _round_last(last, "ma20"),
                    "MA60": _round_last(last, "ma60"),
                    "MA144": _round_last(last, "ma144"),
                    "MA300": _round_last(last, "ma300"),
                    "距MA20%": _round_last(last, "bias20"),
                    "距MA60%": _round_last(last, "bias60"),
                    "距MA144%": _round_last(last, "bias144"),
                    "距MA300%": _round_last(last, "bias300"),
                    "MACD_DIF": _round_last(last, "macd_dif"),
                    "MACD_DEA": _round_last(last, "macd_dea"),
                    "MACD柱": _round_last(last, "macd_hist"),
                    "RSI6": _round_last(last, "rsi6"),
                    "RSI14": _round_last(last, "rsi14"),
                    "KDJ_K": _round_last(last, "kdj_k"),
                    "KDJ_D": _round_last(last, "kdj_d"),
                    "KDJ_J": _round_last(last, "kdj_j"),
                    "CCI14": _round_last(last, "cci14"),
                    "ATR14": _round_last(last, "atr14"),
                    "技术信号": technical_signal(last),
                })
            except Exception as e:
                logging.getLogger(__name__).error(f"[a_share_daily_report] 操作失败: {e}", exc_info=True)
                continue
    return rows

def fetch_spot_tencent(trade_date: str, batch_size: int = 80) -> tuple[pd.DataFrame, SourceStatus]:
    """Fetch broad A-share spot data from Tencent's lightweight quote endpoint.

    AkShare full-market endpoints can fail with RemoteDisconnected on this host.
    Tencent's s_ quote returns name, code, latest price, pct change, volume and
    amount in compact batches; it is sufficient for breadth, equal-weight return,
    turnover and the hot-attention turnover fallback.
    """
    t0 = time.time()
    import baostock as bs

    lg = bs.login()
    rows: list[dict[str, Any]] = []
    try:
        if lg.error_code != "0":
            return pd.DataFrame(), SourceStatus("tencent.qt.gtimg+baostock.query_all_stock", False, 0, lg.error_msg)
        rs = bs.query_all_stock(day=f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}")
        codes: list[str] = []
        while rs.next():
            rec = dict(zip(rs.fields, rs.get_row_data()))
            code = str(rec.get("code") or "")
            name = str(rec.get("code_name") or "")
            if not (code.startswith("sh.6") or code.startswith("sz.00") or code.startswith("sz.30") or code.startswith("bj.")):
                continue
            prefix, num = code.split(".", 1)
            codes.append(prefix + num)
    finally:
        try:
            bs.logout()
        except Exception as e:
            logging.getLogger(__name__).error(f"[a_share_daily_report] 操作失败: {e}", exc_info=True)

    if not codes:
        return pd.DataFrame(), SourceStatus("tencent.qt.gtimg+baostock.query_all_stock", False, (time.time() - t0) * 1000, "empty stock list")

    errors: list[str] = []
    for i in range(0, len(codes), batch_size):
        batch = codes[i : i + batch_size]
        url = "https://qt.gtimg.cn/q=" + ",".join("s_" + c for c in batch)
        text = ""
        for attempt in range(2):
            try:
                with urllib.request.urlopen(url, timeout=60) as resp:
                    text = resp.read().decode("gbk", "ignore")
                break
            except Exception as e:
                errors.append(repr(e)[:120])
                if attempt == 0:
                    time.sleep(2)
        if not text:
            continue
        for item in text.split(";\n"):
            item = item.strip().rstrip(";")
            if '="' not in item:
                continue
            quote = item.split('="', 1)[1].rstrip('"')
            parts = quote.split("~")
            if len(parts) < 10:
                continue
            market = item.split("=", 1)[0].split("s_", 1)[-1][:2].lower()
            code_num = parts[2]
            pct = safe_float(parts[5])
            if pct is None:
                continue
            vol_lot = safe_float(parts[6]) or 0.0
            amount_wan = safe_float(parts[7]) or 0.0
            code = f"{market}{code_num}"
            rows.append({
                "代码": code,
                "名称": parts[1],
                "最新价": safe_float(parts[3]),
                "涨跌幅": pct,
                "成交量": vol_lot * 100,
                "成交额": amount_wan * 10000,
            })
    df = pd.DataFrame(rows)
    ok = not df.empty
    detail = f"rows={len(df)}, batches={(len(codes) + batch_size - 1) // batch_size}"
    if errors:
        detail += f", batch_errors={len(errors)} first={errors[0]}"
    return df, SourceStatus("tencent.qt.gtimg+baostock.query_all_stock", ok, (time.time() - t0) * 1000, detail if ok else (errors[0] if errors else "empty quotes"))


def fetch_spot(timeout_s: int = 90, trade_date: str | None = None) -> tuple[pd.DataFrame, dict[str, Any], SourceStatus]:
    t0 = time.time()

    # ── V2 (2026-08-07): 仓库快照优先（数据资产化 V2-1）──
    #   realtime_snapshot/{YYYYMMDD}/*.parquet 收盘快照，命中即免现场抓数。
    if trade_date is None:
        trade_date = datetime.now().strftime("%Y%m%d")
    try:
        snap_dir = ROOT / "data_warehouse" / "realtime_snapshot" / trade_date.replace("-", "")
        snap_files = sorted(snap_dir.glob("*.parquet")) if snap_dir.exists() else []
        if snap_files:
            df = pd.concat([pd.read_parquet(f) for f in snap_files], ignore_index=True)
            if len(df):
                # 快照列 -> 日报标准列
                out = pd.DataFrame({
                    "代码": df["code"].astype(str).str.extract(r"(\d{6})")[0],
                    "名称": df.get("name", pd.Series(index=df.index)),
                    "最新价": pd.to_numeric(df.get("price"), errors="coerce"),
                    "涨跌幅": pd.to_numeric(df.get("pct_chg"), errors="coerce"),
                    "成交额": pd.to_numeric(df.get("amount_wan"), errors="coerce") * 1e4,
                    "成交量": pd.to_numeric(df.get("volume_lot"), errors="coerce") * 100,
                    "昨收": pd.to_numeric(df.get("pre_close"), errors="coerce"),
                    "今开": pd.to_numeric(df.get("open"), errors="coerce"),
                    "最高": pd.to_numeric(df.get("high"), errors="coerce"),
                    "最低": pd.to_numeric(df.get("low"), errors="coerce"),
                    "换手率": pd.to_numeric(df.get("turnover"), errors="coerce"),
                })
                valid = out.dropna(subset=["涨跌幅"])
                if not valid.empty:
                    breadth = {
                        "股票数": int(len(valid)),
                        "上涨": int((valid["涨跌幅"] > 0).sum()),
                        "下跌": int((valid["涨跌幅"] < 0).sum()),
                        "平盘": int((valid["涨跌幅"] == 0).sum()),
                        "上涨占比%": round(float((valid["涨跌幅"] > 0).mean() * 100), 1),
                        "涨超5%": int((valid["涨跌幅"] >= 5).sum()),
                        "跌超5%": int((valid["涨跌幅"] <= -5).sum()),
                        "近似涨停": int((valid["涨跌幅"] >= limit_threshold_for_code(valid["代码"])).sum()),
                        "近似跌停": int((valid["涨跌幅"] <= -limit_threshold_for_code(valid["代码"])).sum()),
                        "涨跌幅中位数%": round(float(valid["涨跌幅"].median()), 2),
                        "平均涨跌幅%": round(float(valid["涨跌幅"].mean()), 2),
                        "全A成交额(亿)": round(float(valid["成交额"].sum()) / 1e8, 1) if "成交额" in valid else None,
                        "数据说明": f"仓库快照 realtime_snapshot/{trade_date} (文件数={len(snap_files)})",
                    }
                    return out, breadth, SourceStatus(
                        f"warehouse.realtime_snapshot/{trade_date}", True,
                        (time.time() - t0) * 1000, f"rows={len(valid)}",
                    )
    except Exception as _snap_exc:  # noqa: BLE001
        sys.stderr.write(f"仓库快照读取失败 {trade_date}: {_snap_exc}\n")

    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {
            "代码": "代码", "名称": "名称", "最新价": "最新价", "涨跌幅": "涨跌幅",
            "成交额": "成交额", "成交量": "成交量", "昨收": "昨收", "今开": "今开",
            "最高": "最高", "最低": "最低", "今开盘价": "今开", "今收盘价": "最新价",
            "涨跌额": "涨跌额", "涨跌百分比": "涨跌幅", "换手率": "换手率",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        for col in ["最新价", "涨跌幅", "成交额", "成交量", "昨收", "今开", "最高", "最低"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    attempts: list[tuple[str, Any]] = [
        ("akshare.stock_zh_a_spot_sina", lambda: _ak_fetch("stock_zh_a_spot")),
        ("akshare.stock_zh_a_spot_em", lambda: _ak_fetch("stock_zh_a_spot_em")),
    ]
    last_err = None
    for source_name, getter in attempts:
        try:
            with time_limit(25 if source_name.endswith("sina") else min(timeout_s, 20)):
                df = getter()
            if df is None or df.empty:
                last_err = "empty"
                continue
            df = normalize(df)
            valid_cols = [c for c in ["涨跌幅", "成交额", "成交量"] if c in df.columns]
            valid = df.dropna(subset=["涨跌幅"]) if "涨跌幅" in df.columns else pd.DataFrame()
            if valid.empty:
                last_err = f"missing valid rows; cols={list(df.columns)[:20]}"
                continue
            breadth = {
                "股票数": int(len(valid)),
                "上涨": int((valid["涨跌幅"] > 0).sum()),
                "下跌": int((valid["涨跌幅"] < 0).sum()),
                "平盘": int((valid["涨跌幅"] == 0).sum()),
                "上涨占比%": round(float((valid["涨跌幅"] > 0).mean() * 100), 1) if len(valid) else None,
                "涨超5%": int((valid["涨跌幅"] >= 5).sum()),
                "跌超5%": int((valid["涨跌幅"] <= -5).sum()),
                "近似涨停": int((valid["涨跌幅"] >= 9.8).sum()) if "代码" not in valid.columns else int((valid["涨跌幅"] >= limit_threshold_for_code(valid["代码"])).sum()),
                "近似跌停": int((valid["涨跌幅"] <= -9.8).sum()) if "代码" not in valid.columns else int((valid["涨跌幅"] <= -limit_threshold_for_code(valid["代码"])).sum()),
                "涨跌幅中位数%": round(float(valid["涨跌幅"].median()), 2) if len(valid) else None,
                "平均涨跌幅%": round(float(valid["涨跌幅"].mean()), 2) if len(valid) else None,
                "全A成交额(亿)": round(float(valid["成交额"].sum()) / 1e8, 1) if "成交额" in valid else None,
            }
            return df, breadth, SourceStatus(source_name, True, (time.time() - t0) * 1000, f"rows={len(valid)}")
        except Exception as e:
            last_err = repr(e)[:160]
    df, st = fetch_spot_tencent(trade_date or datetime.now().strftime("%Y%m%d"))
    if not df.empty:
        valid = df.dropna(subset=["涨跌幅"])
        # Tencent data has code column with market prefix (e.g. "sh600519")
        limit_up_mask = valid["涨跌幅"] >= limit_threshold_for_code(valid["代码"]) if "代码" in valid.columns else valid["涨跌幅"] >= 9.8
        limit_down_mask = valid["涨跌幅"] <= -limit_threshold_for_code(valid["代码"]) if "代码" in valid.columns else valid["涨跌幅"] <= -9.8
        breadth = {
            "股票数": int(len(valid)),
            "上涨": int((valid["涨跌幅"] > 0).sum()),
            "下跌": int((valid["涨跌幅"] < 0).sum()),
            "平盘": int((valid["涨跌幅"] == 0).sum()),
            "上涨占比%": round(float((valid["涨跌幅"] > 0).mean() * 100), 1) if len(valid) else None,
            "涨超5%": int((valid["涨跌幅"] >= 5).sum()),
            "跌超5%": int((valid["涨跌幅"] <= -5).sum()),
            "近似涨停": int(limit_up_mask.sum()),
            "近似跌停": int(limit_down_mask.sum()),
            "涨跌幅中位数%": round(float(valid["涨跌幅"].median()), 2) if len(valid) else None,
            "平均涨跌幅%": round(float(valid["涨跌幅"].mean()), 2) if len(valid) else None,
            "全A成交额(亿)": round(float(valid["成交额"].sum()) / 1e8, 1) if "成交额" in valid else None,
            "数据说明": f"AkShare全市场接口失败，使用腾讯轻量行情fallback；AkShare错误: {last_err or 'unknown'}",
        }
        return df, breadth, st
    # All live sources failed — fall back to cached breadth snapshot
    import glob
    cached = sorted(glob.glob("generated/a_share_data/20*.json"))
    if cached:
        for fp in reversed(cached):
            try:
                with open(fp) as f:
                    cached_meta = json.load(f)
                cb = cached_meta.get("breadth", {})
                if isinstance(cb, dict) and "上涨占比%" in cb and cb["上涨占比%"] is not None:
                    cb_fresh = dict(cb)
                    cb_fresh["数据说明"] = f"缓存:{os.path.basename(fp)} (实时行情失败，使用上一个交易日缓存)"
                    return pd.DataFrame(), cb_fresh, SourceStatus("cached_breadth_fallback", True, (time.time() - t0) * 1000, f"from {os.path.basename(fp)}")
            except Exception as e:
                logging.getLogger(__name__).error(f"[a_share_daily_report] 操作失败: {e}", exc_info=True)
                continue
    return pd.DataFrame(), {"数据说明": f"全A实时行情失败: {last_err or 'unknown'}; Tencent fallback: {st.detail}"}, SourceStatus("akshare.stock_zh_a_spot+tencent.qt.gtimg", False, (time.time() - t0) * 1000, f"akshare={last_err or 'unknown'}; tencent={st.detail}")


def load_cached_spot_breadth() -> dict[str, Any]:
    """Return the latest cached breadth from generated report metadata."""
    if not DATA_DIR.exists():
        return {}
    for path in sorted(DATA_DIR.glob("*-meta.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logging.getLogger(__name__).error(f"[a_share_daily_report] 操作失败: {e}", exc_info=True)
            continue
        breadth = data.get("breadth")
        if isinstance(breadth, dict) and any(safe_float(breadth.get(k)) is not None for k in ("股票数", "上涨", "下跌", "上涨占比%")):
            cached = dict(breadth)
            cached["数据说明"] = f"实时宽度暂缺，使用缓存宽度: {data.get('trade_date') or path.stem.replace('-meta', '')}"
            return cached
    return {}


# D7真源登记: 去ST硬规则（过滤 ST/*ST 与北交所），全仓唯一口径，禁止改动
# 注: data_sources.fetch_a_share_market_breadth 为另一口径（不去ST），不可替代本函数。
def filtered_a_share(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    code = out.get("代码", pd.Series(dtype=str)).astype(str).str.lower()
    name = out.get("名称", pd.Series(dtype=str)).astype(str).str.upper()
    is_bj = code.str.startswith("bj") | code.str.startswith("8") | code.str.startswith("4") | code.str.startswith("920")
    is_st = name.str.contains("ST", na=False)
    return out[~is_bj & ~is_st].copy()


# D7真源登记: 全A等权涨跌幅/量能（成交量合计、成交额(亿)）硬规则，禁止改动数值口径
def breadth_metrics(df: pd.DataFrame, label: str) -> dict[str, Any]:
    valid = df.dropna(subset=["涨跌幅"]) if not df.empty and "涨跌幅" in df.columns else pd.DataFrame()
    if valid.empty:
        return {"口径": label, "数据说明": "empty"}
    # Determine per-stock limit threshold based on code prefix
    if "代码" in valid.columns:
        limit_thresh = limit_threshold_for_code(valid["代码"])
        is_limit_up = valid["涨跌幅"] >= limit_thresh
        is_limit_down = valid["涨跌幅"] <= -limit_thresh
    else:
        is_limit_up = valid["涨跌幅"] >= 9.8
        is_limit_down = valid["涨跌幅"] <= -9.8
    return {
        "口径": label,
        "股票数": int(len(valid)),
        "上涨": int((valid["涨跌幅"] > 0).sum()),
        "下跌": int((valid["涨跌幅"] < 0).sum()),
        "平盘": int((valid["涨跌幅"] == 0).sum()),
        "上涨占比%": round(float((valid["涨跌幅"] > 0).mean() * 100), 1),
        "全A等权涨跌幅%": round(float(valid["涨跌幅"].mean()), 2),
        "涨跌幅中位数%": round(float(valid["涨跌幅"].median()), 2),
        "涨超5%": int((valid["涨跌幅"] >= 5).sum()),
        "跌超5%": int((valid["涨跌幅"] <= -5).sum()),
        "近似涨停": int(is_limit_up.sum()),
        "近似跌停": int(is_limit_down.sum()),
        "涨停占比%": round(float(is_limit_up.mean() * 100), 2),
        "跌停占比%": round(float(is_limit_down.mean() * 100), 2),
        "成交量合计": round(float(valid["成交量"].sum()), 1) if "成交量" in valid else None,
        "成交额(亿)": round(float(valid["成交额"].sum()) / 1e8, 1) if "成交额" in valid else None,
    }


# D7真源登记: 热股等权 fallback（成交额前N等权），硬规则口径之一，禁止改动
def fallback_hot_attention_pool(spot: pd.DataFrame, reason: str, n: int = 100) -> tuple[dict[str, Any], SourceStatus]:
    """Build a stable proxy for dynamic hot-stock sentiment from the spot snapshot."""
    t0 = time.time()
    valid = spot.dropna(subset=["涨跌幅"]).copy() if not spot.empty and "涨跌幅" in spot.columns else pd.DataFrame()
    if valid.empty:
        return {"数据说明": f"热股接口失败且成交额关注池不可用: {reason}"}, SourceStatus("hot_attention_pool.turnover_top", False, (time.time() - t0) * 1000, "empty spot")
    if "成交额" not in valid.columns:
        return {"数据说明": f"热股接口失败且缺少成交额字段: {reason}"}, SourceStatus("hot_attention_pool.turnover_top", False, (time.time() - t0) * 1000, "missing amount")
    pool = valid.dropna(subset=["成交额"]).sort_values("成交额", ascending=False).head(n)
    if pool.empty:
        return {"数据说明": f"热股接口失败且成交额关注池为空: {reason}"}, SourceStatus("hot_attention_pool.turnover_top", False, (time.time() - t0) * 1000, "empty pool")
    return {
        "热股样本数": int(len(pool)),
        "热股等权涨跌幅%": round(float(pool["涨跌幅"].mean()), 2),
        "热股上涨占比%": round(float((pool["涨跌幅"] > 0).mean() * 100), 1),
        "数据说明": f"外部热股榜失败，改用过滤后成交额前{len(pool)}只构造动态关注池；原错误: {reason}",
        "热股口径": f"过滤后成交额前{len(pool)}只",
    }, SourceStatus("hot_attention_pool.turnover_top", True, (time.time() - t0) * 1000, f"rows={len(pool)}, fallback_from={reason[:80]}")


# D7真源登记: 热股等权（东财热股榜前50只等权涨跌幅），硬规则口径之一，禁止改动
def fetch_hot_rank_equally_weighted(spot: pd.DataFrame, timeout_s: int = 12) -> tuple[dict[str, Any], list[SourceStatus]]:
    t0 = time.time()
    try:
        with time_limit(timeout_s):
            hot = _ak_fetch("stock_hot_rank_em")
        if hot.empty:
            fallback, fallback_status = fallback_hot_attention_pool(spot, "stock_hot_rank_em empty")
            return fallback, [SourceStatus("akshare.stock_hot_rank_em", False, (time.time() - t0) * 1000, "empty"), fallback_status]
        code_col = next((c for c in hot.columns if "代码" in c), None)
        if not code_col:
            detail = f"missing code column: {list(hot.columns)}"
            fallback, fallback_status = fallback_hot_attention_pool(spot, detail)
            return fallback, [SourceStatus("akshare.stock_hot_rank_em", False, (time.time() - t0) * 1000, detail), fallback_status]
        hot_codes = hot[code_col].astype(str).str.extract(r"(\d{6})")[0].dropna().head(50).tolist()
        spot_codes = spot.get("代码", pd.Series(dtype=str)).astype(str).str.extract(r"(\d{6})")[0]
        matched = spot[spot_codes.isin(hot_codes)].dropna(subset=["涨跌幅"])
        if matched.empty:
            fallback, fallback_status = fallback_hot_attention_pool(spot, "stock_hot_rank_em matched=0")
            return fallback, [SourceStatus("akshare.stock_hot_rank_em", False, (time.time() - t0) * 1000, f"hot={len(hot)}, matched=0"), fallback_status]
        return {
            "热股样本数": int(len(matched)),
            "热股等权涨跌幅%": round(float(matched["涨跌幅"].mean()), 2) if len(matched) else None,
            "热股上涨占比%": round(float((matched["涨跌幅"] > 0).mean() * 100), 1) if len(matched) else None,
            "数据说明": "东方财富热股榜匹配全A实时行情",
            "热股口径": "东方财富热股榜前50只",
        }, [SourceStatus("akshare.stock_hot_rank_em", True, (time.time() - t0) * 1000, f"hot={len(hot)}, matched={len(matched)}")]
    except Exception as e:
        detail = repr(e)[:160]
        fallback, fallback_status = fallback_hot_attention_pool(spot, detail)
        return fallback, [SourceStatus("akshare.stock_hot_rank_em", False, (time.time() - t0) * 1000, detail), fallback_status]


def fetch_industry_summary(timeout_s: int = 20) -> tuple[pd.DataFrame, SourceStatus]:
    t0 = time.time()
    try:
        with time_limit(timeout_s):
            df = _ak_fetch("stock_board_industry_summary_ths")
        for col in ["涨跌幅", "总成交量", "总成交额", "上涨家数", "下跌家数"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df, SourceStatus("akshare.stock_board_industry_summary_ths", not df.empty, (time.time() - t0) * 1000, f"rows={len(df)}")
    except Exception as e:
        return pd.DataFrame(), SourceStatus("akshare.stock_board_industry_summary_ths", False, (time.time() - t0) * 1000, repr(e)[:160])


def fetch_concept_summary(timeout_s: int = 20) -> tuple[pd.DataFrame, SourceStatus]:
    """概念板块汇总：V3 融合后**本地仓库优先**（data_warehouse/classification/concept_board.parquet），
    东财接口作为 fallback。彻底解决东财限流时概念数据缺失。

    返回列统一为: 板块/涨跌幅/换手率/上涨家数/下跌家数/总市值/领涨股。
    """
    t0 = time.time()
    # 1) 本地仓库优先（V3：分类数据已落地，504 板块全量）
    try:
        from quant_system.data_store import DataStore
        _ds = DataStore()
        local = _ds.get_dataset("concept_board")
        if local is not None and not local.empty:
            rename = {
                "board_name": "板块", "pct_chg": "涨跌幅", "turnover": "换手率",
                "up_count": "上涨家数", "down_count": "下跌家数",
                "total_mv": "总市值", "leader_name": "领涨股",
                "leader_pct": "领涨股涨跌幅",
            }
            local = local.rename(columns={k: v for k, v in rename.items() if k in local.columns})
            keep = [c for c in ["板块", "涨跌幅", "换手率", "上涨家数", "下跌家数", "总市值", "领涨股", "领涨股涨跌幅"]
                    if c in local.columns]
            local = local[keep]
            for col in ["涨跌幅", "换手率", "上涨家数", "下跌家数", "总市值", "领涨股涨跌幅"]:
                if col in local.columns:
                    local[col] = pd.to_numeric(local[col], errors="coerce")
            src = f"warehouse.classification.concept_board rows={len(local)}"
            return local, SourceStatus("concept_board.parquet", True, (time.time() - t0) * 1000, src)
    except Exception as e:
        logging.getLogger(__name__).error(f"[a_share_daily_report] 操作失败: {e}", exc_info=True)  # 本地不可用则回退网络
    # 2) 网络 fallback（东财实时）
    try:
        with time_limit(timeout_s):
            df = _ak_fetch("stock_board_concept_name_em")
        rename = {
            "板块名称": "板块",
            "涨跌幅": "涨跌幅",
            "总市值": "总市值",
            "换手率": "换手率",
            "上涨家数": "上涨家数",
            "下跌家数": "下跌家数",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        for col in ["涨跌幅", "总市值", "换手率", "上涨家数", "下跌家数"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df, SourceStatus("akshare.stock_board_concept_name_em", not df.empty, (time.time() - t0) * 1000, f"rows={len(df)}")
    except Exception as e:
        return pd.DataFrame(), SourceStatus("akshare.stock_board_concept_name_em", False, (time.time() - t0) * 1000, repr(e)[:160])


def fetch_lhb_daily(trade_date: str) -> tuple[dict[str, Any], SourceStatus]:
    """龙虎榜当日数据：优先读本地 data_warehouse/market/lhb_*.parquet，
    其次 akshare 网络拉取当日榜（服务器/国内 IP 可用）。

    返回: {
      "date": trade_date,
      "total": 上榜家数,
      "top_buy": [当日净买额 Top8],
      "top_sell": [净卖额 Top8],
      "max_net": 最大净买额, "max_net_name": 个股,
      "hot_board": 上榜原因分布 Top5,
      "note": 数据说明,
    }
    """
    t0 = time.time()
    # 1) 本地 parquet 优先（离线可用，含历史全量）
    try:
        lhb_dir = ROOT / "data_warehouse" / "market"
        files = sorted(lhb_dir.glob("lhb_*.parquet"))
        if files:
            # trade_date 是 YYYYMMDD，季度文件命名 lhb_YYYYMMDD_YYYYMMDD
            want = trade_date.replace("-", "")[:8]
            for f in reversed(files):
                s, e = f.stem.replace("lhb_", "").split("_")
                if s <= want <= e:
                    df = pd.read_parquet(f)
                    if "上榜日" in df.columns:
                        df = df.copy()
                        df["上榜日"] = pd.to_datetime(df["上榜日"], errors="coerce")
                        day = df[df["上榜日"].dt.strftime("%Y%m%d") == want]
                        if len(day):
                            return _build_lhb_result(day, trade_date, f"本地parquet({f.name})"), \
                                SourceStatus(f"lhb.parquet({f.name})", True, (time.time() - t0) * 1000, f"rows={len(day)}")
    except Exception as e:
        logging.getLogger(__name__).error(f"[a_share_daily_report] 操作失败: {e}", exc_info=True)
    # 2) akshare 网络（补充当日最新榜，东财接口需国内 IP）
    try:
        import akshare as ak
        d = trade_date.replace("-", "")[:8]
        df = ak.stock_lhb_detail_em(start_date=d, end_date=d)
        if df is not None and len(df):
            return _build_lhb_result(df, trade_date, "akshare.stock_lhb_detail_em"), \
                SourceStatus("akshare.stock_lhb_detail_em", True, (time.time() - t0) * 1000, f"rows={len(df)}")
    except Exception as e:
        return {"date": trade_date, "total": 0, "note": f"龙虎榜数据获取失败: {str(e)[:120]}"}, \
            SourceStatus("lhb", False, (time.time() - t0) * 1000, repr(e)[:160])
    return {"date": trade_date, "total": 0, "note": "龙虎榜无当日数据"}, \
        SourceStatus("lhb", False, (time.time() - t0) * 1000, "当日无上榜")


def _build_lhb_result(df: pd.DataFrame, trade_date: str, note: str) -> dict[str, Any]:
    """龙虎榜 DataFrame → 日报展示结构（净买额 Top / 卖出 Top / 上榜原因分布）。"""
    d = df.copy()
    # 列归一
    code_col = "代码" if "代码" in d.columns else ("证券代码" if "证券代码" in d.columns else None)
    name_col = "名称" if "名称" in d.columns else ("证券简称" if "证券简称" in d.columns else None)
    net_col = "龙虎榜净买额" if "龙虎榜净买额" in d.columns else ("龙虎榜买入额" if "龙虎榜买入额" in d.columns else None)
    if code_col:
        d["_code"] = d[code_col].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    if name_col is None or net_col is None:
        return {"date": trade_date, "total": len(d), "note": note + "（列结构不匹配）"}
    d["_net"] = pd.to_numeric(d[net_col], errors="coerce").fillna(0.0)
    # 净买额 Top
    top_buy = d.nlargest(8, "_net")[[name_col, "_net", "_code"]].reset_index(drop=True) if len(d) else None
    top_sell = d.nsmallest(8, "_net")[[name_col, "_net", "_code"]].reset_index(drop=True) if len(d) else None
    # 上榜原因分布
    reason_col = None
    for c in d.columns:
        if "原因" in c or "上榜原因" in c:
            reason_col = c
            break
    hot_board = {}
    if reason_col:
        hot_board = d[reason_col].astype(str).value_counts().head(5).to_dict()
    max_net = float(d["_net"].max()) if len(d) else 0.0
    max_name = ""
    if len(d):
        mi = d["_net"].idxmax()
        max_name = str(d.loc[mi, name_col])
    def _fmt(x: float) -> str:
        return f"{x/1e8:.2f}亿" if abs(x) >= 1e8 else f"{x/1e4:.0f}万"
    return {
        "date": trade_date,
        "total": len(d),
        "top_buy": [{"name": r[name_col], "code": r["_code"], "net": _fmt(r["_net"])} for _, r in top_buy.iterrows()] if top_buy is not None else [],
        "top_sell": [{"name": r[name_col], "code": r["_code"], "net": _fmt(r["_net"])} for _, r in top_sell.iterrows()] if top_sell is not None else [],
        "max_net": _fmt(max_net), "max_net_name": max_name,
        "hot_board": hot_board,
        "note": note,
    }


def _lhb_section(lhb: dict[str, Any]) -> str:
    """龙虎榜 metrics → markdown 章节。"""
    if not lhb or lhb.get("total", 0) == 0:
        return f"- 龙虎榜：{lhb.get('note', '无当日数据')}"
    lines = [f"- **当日上榜**：{lhb.get('total')} 只（来源：{lhb.get('note', '')}）"]
    if lhb.get("max_net_name"):
        lines.append(f"- **净买额榜首**：{lhb.get('max_net_name')}（{lhb.get('max_net')}）")
    tb = lhb.get("top_buy", [])
    if tb:
        lines.append("\n### 净买入 Top8")
        lines.append("| 名称 | 代码 | 净买额 |")
        lines.append("|---|---|---|")
        for r in tb:
            lines.append(f"| {r['name']} | {r['code']} | {r['net']} |")
    ts = lhb.get("top_sell", [])
    if ts:
        lines.append("\n### 净卖出 Top8")
        lines.append("| 名称 | 代码 | 净卖额 |")
        lines.append("|---|---|---|")
        for r in ts:
            lines.append(f"| {r['name']} | {r['code']} | {r['net']} |")
    hb = lhb.get("hot_board", {})
    if hb:
        lines.append("\n### 上榜原因分布")
        for k, v in list(hb.items())[:5]:
            lines.append(f"- {k}: {v} 家")
    lines.append("\n- **用法**：龙虎榜净买额集中在少数个股时，短线情绪向这些方向聚焦；净卖出为主说明高位资金兑现。结合涨停/连板高度看情绪周期位置，不单独作为买卖依据。")
    return "\n".join(lines)


# D7真源登记: 风格分层章节（主板/双创/北交所 + 风格等权口径），硬规则展示层，禁止改动
def _segment_section(seg: dict[str, Any]) -> str:
    """市场口径分层章节（V3: 主板/双创/北交所 + 风格分层）。"""
    """市场口径分层 metrics → markdown 章节（用户 #11）。

    展示主板/创业板/科创板/双创/北交所各自的涨跌家数、等权涨跌幅、涨跌停。
    """
    if not seg or "meta" not in seg or seg.get("meta", {}).get("error"):
        return "- 市场口径分层数据缺失（spot 为空）"
    segs = [k for k in seg if k != "meta"]
    if not segs:
        return "- 市场口径分层无数据"
    lines = ["| 口径 | 股票数 | 等权涨跌幅% | 上涨占比% | 涨超5% | 跌超5% | 近似涨停 | 近似跌停 |"]
    lines.append("|---|---|---|---|---|---|---|---|")
    for k in ("全A", "主板", "创业板", "科创板", "双创", "北交所", "剔除ST北交"):
        if k not in seg:
            continue
        s = seg[k]
        if s.get("等权涨跌幅%") is None:
            lines.append(f"| {k} | {s.get('股票数', 0)} | 缺失 | - | - | - | - | - |")
            continue
        lines.append(
            f"| {k} | {s.get('股票数', 0)} | {s.get('等权涨跌幅%')} | "
            f"{s.get('上涨占比%')} | {s.get('涨超5%')} | {s.get('跌超5%')} | "
            f"{s.get('近似涨停')} | {s.get('近似跌停')} |"
        )
    lines.append("\n- **口径说明**：主板=60/00剔除ST；双创=创业板(30/301)+科创板(688)剔除ST；北交所单独；涨跌停阈值按板块（主板9.8%/双创19.8%/北交29.8%）。")
    lines.append("- **用法**：对比主板与双创的等权涨跌幅和涨停分布，判断风格扩散/退潮；北交所波动大，单独口径避免污染主板信号。")
    # 风格分层（成交额分位近似大小票）
    style = seg.get("_style", {})
    if style and style.get("meta", {}).get("error") is None:
        style_keys = [k for k in style if k != "meta"]
        if style_keys:
            lines.append("\n### 风格分层（成交额分位近似）")
            lines.append("| 风格 | 股票数 | 等权涨跌幅% | 上涨占比% |")
            lines.append("|---|---|---|---|")
            for k in style_keys:
                s = style[k]
                lines.append(f"| {k} | {s.get('股票数', 0)} | {s.get('等权涨跌幅%')} | {s.get('上涨占比%')} |")
    # V3 融合：短/中/长线（含涨停概念热点题材）
    sml = seg.get("_sml", {}) or {}
    hot = sml.get("短线_热点题材(涨停概念)")
    if hot and hot.get("Top概念"):
        lines.append("\n### 短线热点题材（涨停股概念聚合）")
        lines.append(f"- 涨停股 {hot.get('涨停股数')} 只，涨停最集中的概念：")
        for r in hot.get("Top概念", [])[:6]:
            lines.append(f"  - **{r.get('concept')}**：{r.get('涨停家数')} 只")
    return "\n".join(lines)


def _concept_heat_section(heat: dict) -> str:
    """概念热度全景章节（V3 融合：本地 concept_board 全量 504 板块）。

    heat 来自 quant_platform.openclaw_api.concept_heatmap()：
    涨幅榜/跌幅榜/换手活跃榜/全线飘红榜。
    """
    if not heat or heat.get("error"):
        return "- 概念热度数据缺失（classification/concept_board.parquet 未落地）"
    total = heat.get("total_concepts", 0)
    lines = [f"- **概念总数**：{total} 个（本地东财概念全量）"]

    def _fmt(rows, cols):
        if not rows:
            return "-"
        parts = []
        for r in rows:
            name = r.get("概念", r.get("board_name", "?"))
            pct = r.get("涨跌幅%", r.get("pct_chg"))
            leader = r.get("领涨股", "")
            s = f"{name} {pct:+.2f}%" if isinstance(pct, (int, float)) else f"{name} {pct}"
            if leader:
                s += f"(领涨:{leader})"
            parts.append(s)
        return "、".join(parts)

    up = heat.get("涨幅榜", [])
    down = heat.get("跌幅榜", [])
    dense = heat.get("全线飘红榜", [])
    if up:
        lines.append(f"- **涨幅榜**：{_fmt(up, None)}")
    if down:
        lines.append(f"- **跌幅榜**：{_fmt(down, None)}")
    if dense:
        lines.append(f"- **全线飘红（无下跌家数）**：{_fmt(dense, None)}")
    lines.append("- **用法**：概念涨幅榜与行业主线交叉验证，全线飘红板块为情绪极端聚焦方向；概念热度只作情绪温度，买卖依据仍看行业成交额与个股趋势。")
    return "\n".join(lines)


def fetch_exchange_summary(trade_date: str) -> tuple[dict[str, Any], list[SourceStatus]]:
    statuses = []
    summary: dict[str, Any] = {}
    t0 = time.time()
    try:
        sse = _ak_fetch("stock_sse_summary")
        summary["sse"] = sse.to_dict(orient="records")
        statuses.append(SourceStatus("akshare.stock_sse_summary", True, (time.time() - t0) * 1000, f"rows={len(sse)}"))
    except Exception as e:
        statuses.append(SourceStatus("akshare.stock_sse_summary", False, (time.time() - t0) * 1000, repr(e)[:160]))
    t1 = time.time()
    try:
        szse = _ak_fetch("stock_szse_summary", date=trade_date)
        summary["szse"] = szse.to_dict(orient="records")
        statuses.append(SourceStatus("akshare.stock_szse_summary", True, (time.time() - t1) * 1000, f"rows={len(szse)}"))
    except Exception as e:
        statuses.append(SourceStatus("akshare.stock_szse_summary", False, (time.time() - t1) * 1000, repr(e)[:160]))
    return summary, statuses


# D7真源登记: 融资TMT硬规则（全A融资/TMT融资，TMT_INDUSTRY_PREFIXES 口径），禁止改动
def fetch_margin_metrics(trade_date: str, timeout_s: int = 120) -> tuple[dict[str, Any], SourceStatus]:
    t0 = time.time()
    import baostock as bs

    try:
        with time_limit(timeout_s):
            sh = _ak_fetch("macro_china_market_margin_sh")
            sz = _ak_fetch("macro_china_market_margin_sz")
            for df in [sh, sz]:
                df["日期"] = pd.to_datetime(df["日期"]).dt.strftime("%Y%m%d")
                df["融资余额"] = pd.to_numeric(df["融资余额"], errors="coerce")
            common_dates = sorted(set(sh[sh["日期"] <= trade_date]["日期"]) & set(sz[sz["日期"] <= trade_date]["日期"]))
            if len(common_dates) < 2:
                return {"数据说明": "沪深两融汇总不足两个交易日"}, SourceStatus("akshare.margin_summary", False, (time.time() - t0) * 1000, "not enough dates")
            prev_date, latest_date = common_dates[-2], common_dates[-1]
            sh_latest = float(sh.loc[sh["日期"] == latest_date, "融资余额"].iloc[-1])
            sz_latest = float(sz.loc[sz["日期"] == latest_date, "融资余额"].iloc[-1])
            sh_prev = float(sh.loc[sh["日期"] == prev_date, "融资余额"].iloc[-1])
            sz_prev = float(sz.loc[sz["日期"] == prev_date, "融资余额"].iloc[-1])
            all_latest = sh_latest + sz_latest
            all_prev = sh_prev + sz_prev

            def margin_detail(date: str) -> pd.DataFrame:
                # ── V2 (2026-08-07): 仓库 margin_detail 优先（两融明细落盘）──
                try:
                    parts = []
                    for mkt in ("sh", "sz"):
                        p = ROOT / "data_warehouse" / "market" / f"margin_detail_{mkt}" / f"{date}.parquet"
                        if p.exists():
                            d = pd.read_parquet(p)
                            # 仓库列: 信用交易日期/标的证券代码/标的证券简称/融资余额...
                            if "标的证券代码" in d.columns:
                                d = d.rename(columns={"标的证券代码": "证券代码", "标的证券简称": "证券简称"})
                            parts.append(d)
                    if len(parts) == 2:
                        out = pd.concat(parts, ignore_index=True)
                        out["code6"] = out["证券代码"].astype(str).str.extract(r"(\d{6})")[0]
                        out["融资余额"] = pd.to_numeric(out["融资余额"], errors="coerce")
                        return out.dropna(subset=["code6"])
                except Exception as e:
                    logging.getLogger(__name__).error(f"[a_share_daily_report] 操作失败: {e}", exc_info=True)
                # 仓库缺失/异常 -> akshare 网络拉取
                sse = _ak_fetch("stock_margin_detail_sse", date=date).rename(columns={"标的证券代码": "证券代码", "标的证券简称": "证券简称"})
                szse = _ak_fetch("stock_margin_detail_szse", date=date)
                out = pd.concat([sse, szse], ignore_index=True)
                out["code6"] = out["证券代码"].astype(str).str.extract(r"(\d{6})")[0]
                out["融资余额"] = pd.to_numeric(out["融资余额"], errors="coerce")
                return out.dropna(subset=["code6"])

            cur_detail = margin_detail(latest_date)
            prev_detail = margin_detail(prev_date)

            # ── V4.2 fix: baostock 晚间不稳定，行业映射失败时：
            #   1) 优先读本地行业映射缓存（baostock 成功时落盘）
            #   2) 否则尝试 tushare stock_basic 行业兜底
            #   3) 全A融资余额始终返回，TMT 拆不出时标记缺失，不再整体失败
            ind = None
            ind_src = ""
            import os as _os
            ind_cache = ROOT / "data_warehouse" / "market" / "industry_baostock_map.parquet"
            if ind_cache.exists():
                try:
                    ind = pd.read_parquet(ind_cache)
                    ind_src = "cache"
                except Exception:
                    ind = None
            if ind is None:
                lg = bs.login()
                try:
                    if lg.error_code == "0":
                        rows = []
                        rs = bs.query_stock_industry(date=f"{latest_date[:4]}-{latest_date[4:6]}-{latest_date[6:]}")
                        while rs.next():
                            rows.append(rs.get_row_data())
                        if rows:
                            ind = pd.DataFrame(rows, columns=rs.fields)
                            ind_src = "baostock"
                            try:  # 成功即落盘缓存
                                _os.makedirs(ind_cache.parent, exist_ok=True)
                                ind.to_parquet(ind_cache)
                            except Exception as e:
                                logging.getLogger(__name__).error(f"[a_share_daily_report] 操作失败: {e}", exc_info=True)
                finally:
                    try:
                        bs.logout()
                    except Exception as e:
                        logging.getLogger(__name__).error(f"[a_share_daily_report] 操作失败: {e}", exc_info=True)
            if ind is None:
                try:  # tushare 行业兜底（名称匹配 TMT 关键词）
                    import tushare as ts_mod
                    ts_mod.set_token(ts_mod.get_token())
                    pro = ts_mod.pro_api()
                    tb = pro.stock_basic(exchange="", list_status="L", fields="ts_code,symbol,name,industry")
                    tb["code6"] = tb["symbol"].astype(str).str.extract(r"(\d{6})")[0]
                    ind = tb.rename(columns={"industry": "industry"})
                    ind_src = "tushare"
                except Exception:
                    ind = None

            if ind is None or ind.empty:
                return {
                    "日期": latest_date,
                    "前一交易日": prev_date,
                    "全A融资余额(亿)": round(all_latest / 1e8, 1),
                    "全A较前日变化(亿)": round((all_latest - all_prev) / 1e8, 1),
                    "全A较前日变化%": round((all_latest / all_prev - 1) * 100, 2) if all_prev else None,
                    "TMT融资余额(亿)": None,
                    "TMT较前日变化(亿)": None,
                    "TMT较前日变化%": None,
                    "TMT占全A融资余额%": None,
                    "TMT口径": TMT_INDUSTRY_NOTE,
                    "TMT行业分布": [],
                    "数据说明": f"行业映射不可用({ind_src or 'none'})，仅返回全A融资；融资融券数据通常滞后披露，本次使用最近披露日 {latest_date}",
                }, SourceStatus("akshare.margin_summary+detail(+industry缺)", True, (time.time() - t0) * 1000, f"date={latest_date}, prev={prev_date}, detail_rows={len(cur_detail)}")

            ind = ind.copy()
            if "code" in ind.columns and "code6" not in ind.columns:
                ind["code6"] = ind["code"].astype(str).str.extract(r"(\d{6})")[0]
            ind["is_tmt"] = ind["industry"].astype(str).str.startswith(TMT_INDUSTRY_PREFIXES)
            if ind_src == "tushare":
                # tushare 行业为名称（非证监会代码），用 TMT 关键词匹配
                _tmt_kw = ("计算机", "通信", "软件", "互联网", "电信", "广播", "电视", "电影", "录音", "新闻", "出版")
                ind["is_tmt"] = ind["industry"].astype(str).apply(lambda s: any(k in s for k in _tmt_kw))
            mapper = ind[["code6", "industry", "is_tmt"]].dropna(subset=["code6"])
            cur = cur_detail.merge(mapper, on="code6", how="left")
            pre = prev_detail.merge(mapper, on="code6", how="left")
            tmt_latest = float(cur.loc[cur["is_tmt"] == True, "融资余额"].sum())
            tmt_prev = float(pre.loc[pre["is_tmt"] == True, "融资余额"].sum())
            tmt_industry = (
                cur.loc[cur["is_tmt"] == True]
                .groupby("industry", dropna=True)["融资余额"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )
            if not tmt_industry.empty:
                tmt_industry["融资余额(亿)"] = (tmt_industry["融资余额"] / 1e8).round(1)
                tmt_industry = tmt_industry.drop(columns=["融资余额"])
            return {
                "日期": latest_date,
                "前一交易日": prev_date,
                "全A融资余额(亿)": round(all_latest / 1e8, 1),
                "全A较前日变化(亿)": round((all_latest - all_prev) / 1e8, 1),
                "全A较前日变化%": round((all_latest / all_prev - 1) * 100, 2) if all_prev else None,
                "TMT融资余额(亿)": round(tmt_latest / 1e8, 1),
                "TMT较前日变化(亿)": round((tmt_latest - tmt_prev) / 1e8, 1),
                "TMT较前日变化%": round((tmt_latest / tmt_prev - 1) * 100, 2) if tmt_prev else None,
                "TMT占全A融资余额%": round(tmt_latest / all_latest * 100, 2) if all_latest else None,
                "TMT口径": TMT_INDUSTRY_NOTE,
                "TMT行业分布": tmt_industry.to_dict(orient="records"),
                "数据说明": f"行业映射源={ind_src}；融资融券数据通常滞后披露；本次使用最近披露日 {latest_date} 对比 {prev_date}",
            }, SourceStatus("akshare.margin_summary+detail+industry", True, (time.time() - t0) * 1000, f"date={latest_date}, prev={prev_date}, detail_rows={len(cur_detail)}, ind_src={ind_src}")
    except Exception as e:
        return {"数据说明": f"融资余额接口失败: {repr(e)[:160]}"}, SourceStatus("akshare.margin_summary+detail+baostock.industry", False, (time.time() - t0) * 1000, repr(e)[:180])


def table_or_empty(df: pd.DataFrame, columns: list[str], n: int, sort_col: str, asc: bool) -> pd.DataFrame:
    if df.empty or sort_col not in df.columns:
        return pd.DataFrame(columns=columns)
    cols = [c for c in columns if c in df.columns]
    return df.sort_values(sort_col, ascending=asc).head(n)[cols]


def top_names(df: pd.DataFrame, n: int = 5) -> str:
    if df.empty or "板块" not in df.columns:
        return "缺失"
    return "、".join(df["板块"].astype(str).head(n).tolist())


def describe_volume_price(core_breadth: dict[str, Any], idx: pd.DataFrame) -> str:
    sh = idx[idx["指数"] == "上证指数"].iloc[0] if not idx.empty and (idx["指数"] == "上证指数").any() else None
    amount = safe_float(core_breadth.get("成交额(亿)"))
    equal_ret = safe_float(core_breadth.get("全A等权涨跌幅%"))
    sh_ret = safe_float(sh.get("涨跌幅%")) if sh is not None else None
    sh_vol_ratio = safe_float(sh.get("量比20日")) if sh is not None else None
    amount_desc = "成交额缺失"
    if amount is not None:
        amount_desc = f"过滤后成交额约 {amount:.1f} 亿元"
    vol_desc = "量比缺失"
    if sh_vol_ratio is not None:
        vol_desc = "指数成交低于20日均量" if sh_vol_ratio < 0.95 else ("指数成交接近20日均量" if sh_vol_ratio < 1.05 else "指数成交高于20日均量")
    if equal_ret is not None and equal_ret > 0 and sh_ret is not None and sh_ret < 0:
        return f"{amount_desc}，{vol_desc}；全A等权上涨但上证下跌，属于宽度修复与指数权重承压并存，优先看上涨行业能否延续放量扩散。"
    if equal_ret is not None and equal_ret > 0:
        return f"{amount_desc}，{vol_desc}；赚钱效应修复，若后续成交继续放大且跌停维持低位，量价结构可视为修复确认。"
    if equal_ret is not None and equal_ret < 0 and sh_vol_ratio is not None and sh_vol_ratio > 1:
        return f"{amount_desc}，{vol_desc}；下跌伴随放量，按量价分析属于真实抛压释放，短线不宜抢先确认反转。"
    return f"{amount_desc}，{vol_desc}；当前量价信号不够单边，继续观察成交额和宽度是否同向改善。"


def fetch_hyyyb_daily(trade_date: str) -> tuple[dict[str, Any], SourceStatus]:
    """活跃营业部当日数据：本地 data_warehouse/market/lhb_hyyyb_em.parquet（48万行，2020-01→今）。

    返回: {
      "date": trade_date,
      "total": 当日上榜营业部数,
      "top_buy": [{营业部, 净买额, 买入个股} Top8],
      "top_sell": [净卖额 Top8],
      "hot_stock": 被最多营业部买入的个股 Top5,
      "note": 数据说明,
    }
    """
    t0 = time.time()
    try:
        p = ROOT / "data_warehouse" / "market" / "lhb_hyyyb_em.parquet"
        if not p.exists():
            return {"date": trade_date, "total": 0, "note": "lhb_hyyyb_em.parquet 不存在（服务器补抓后同步）"}, \
                SourceStatus("lhb_hyyyb", False, (time.time() - t0) * 1000, "file missing")
        df = pd.read_parquet(p)
        if df.empty or "上榜日" not in df.columns:
            return {"date": trade_date, "total": 0, "note": "营业部数据为空"}, \
                SourceStatus("lhb_hyyyb", False, (time.time() - t0) * 1000, "empty")
        df = df.copy()
        df["上榜日"] = pd.to_datetime(df["上榜日"], errors="coerce")
        want = trade_date.replace("-", "")[:8]
        day = df[df["上榜日"].dt.strftime("%Y%m%d") == want]
        if day.empty:
            # 找最近一个交易日
            last = df[df["上榜日"] <= pd.Timestamp(want[:4] + "-" + want[4:6] + "-" + want[6:])]
            if not last.empty:
                d_last = last["上榜日"].max()
                day = df[df["上榜日"] == d_last]
                note = f"本地parquet(lhb_hyyyb_em) 无{want}数据，用最近交易日{d_last.strftime('%Y-%m-%d')}"
            else:
                return {"date": trade_date, "total": 0, "note": "营业部数据早于数据起点"}, \
                    SourceStatus("lhb_hyyyb", False, (time.time() - t0) * 1000, "no day")
        else:
            note = f"本地parquet(lhb_hyyyb_em) rows={len(day)}"
        buy_col = "买入总金额" if "买入总金额" in day.columns else None
        sell_col = "卖出总金额" if "卖出总金额" in day.columns else None
        net_col = "总买卖净额" if "总买卖净额" in day.columns else None
        stk_col = "买入股票" if "买入股票" in day.columns else None
        name_col = "营业部名称" if "营业部名称" in day.columns else None
        top_buy, top_sell, hot = [], [], []
        if name_col and net_col:
            d2 = day.dropna(subset=[net_col])
            if not d2.empty:
                d2 = d2.sort_values(net_col, ascending=False)
                for _, r in d2.head(8).iterrows():
                    top_buy.append({"name": str(r[name_col]), "net": float(r[net_col]),
                                    "stocks": str(r[stk_col])[:60] if stk_col and pd.notna(r[stk_col]) else ""})
                for _, r in d2.tail(8).iterrows():
                    top_sell.append({"name": str(r[name_col]), "net": float(r[net_col]),
                                     "stocks": str(r[stk_col])[:60] if stk_col and pd.notna(r[stk_col]) else ""})
        # 热门个股：按"买入股票"字段统计提及次数
        if stk_col:
            from collections import Counter
            cnt = Counter()
            for v in day[stk_col].dropna():
                for s in str(v).replace("，", ",").split(","):
                    s = s.strip()
                    if len(s) >= 2:
                        cnt[s] += 1
            hot = [{"stock": k, "count": v} for k, v in cnt.most_common(5)]
        return {"date": trade_date, "total": len(day), "top_buy": top_buy, "top_sell": top_sell,
                "hot_stock": hot, "note": note}, \
            SourceStatus("lhb_hyyyb", True, (time.time() - t0) * 1000, f"rows={len(day)}")
    except Exception as e:
        return {"date": trade_date, "total": 0, "note": f"营业部数据获取失败: {str(e)[:100]}"}, \
            SourceStatus("lhb_hyyyb", False, (time.time() - t0) * 1000, repr(e)[:160])


def fetch_jgmmtj_daily(trade_date: str) -> tuple[dict[str, Any], SourceStatus]:
    """机构龙虎榜当日数据：本地 data_warehouse/market/lhb_jgmmtj_em.parquet（5.8万行）。

    返回: {"date", "total": 机构上榜家数, "top_net": 机构净买 Top8, "top_out": 机构净卖 Top8, "note"}
    """
    t0 = time.time()
    try:
        p = ROOT / "data_warehouse" / "market" / "lhb_jgmmtj_em.parquet"
        if not p.exists():
            return {"date": trade_date, "total": 0, "note": "lhb_jgmmtj_em.parquet 不存在"}, \
                SourceStatus("lhb_jgmmtj", False, (time.time() - t0) * 1000, "file missing")
        df = pd.read_parquet(p)
        if df.empty or "上榜日期" not in df.columns:
            return {"date": trade_date, "total": 0, "note": "机构龙虎榜数据为空"}, \
                SourceStatus("lhb_jgmmtj", False, (time.time() - t0) * 1000, "empty")
        df = df.copy()
        df["上榜日期"] = pd.to_datetime(df["上榜日期"], errors="coerce")
        want = trade_date.replace("-", "")[:8]
        day = df[df["上榜日期"].dt.strftime("%Y%m%d") == want]
        if day.empty:
            return {"date": trade_date, "total": 0, "note": f"机构龙虎榜当日无数据（{want}）"}, \
                SourceStatus("lhb_jgmmtj", False, (time.time() - t0) * 1000, "no day")
        name_col = "名称" if "名称" in day.columns else "代码"
        net_col = "机构买入净额" if "机构买入净额" in day.columns else None
        top_net, top_out = [], []
        if net_col:
            d2 = day.dropna(subset=[net_col])
            if not d2.empty:
                d2 = d2.sort_values(net_col, ascending=False)
                for _, r in d2.head(8).iterrows():
                    top_net.append({"name": str(r[name_col]), "code": str(r.get("代码", "")),
                                    "net": float(r[net_col]), "buy_org": int(r.get("买方机构数", 0) or 0),
                                    "sell_org": int(r.get("卖方机构数", 0) or 0)})
                for _, r in d2.tail(8).iterrows():
                    top_out.append({"name": str(r[name_col]), "code": str(r.get("代码", "")),
                                    "net": float(r[net_col])})
        return {"date": trade_date, "total": len(day), "top_net": top_net, "top_out": top_out,
                "note": f"本地parquet(lhb_jgmmtj_em) rows={len(day)}"}, \
            SourceStatus("lhb_jgmmtj", True, (time.time() - t0) * 1000, f"rows={len(day)}")
    except Exception as e:
        return {"date": trade_date, "total": 0, "note": f"机构龙虎榜获取失败: {str(e)[:100]}"}, \
            SourceStatus("lhb_jgmmtj", False, (time.time() - t0) * 1000, repr(e)[:160])


def fetch_gdhs_snapshot(trade_date: str) -> tuple[dict[str, Any], SourceStatus]:
    """股东户数快照：本地 data_warehouse/market/gdhs_all.parquet（全市场，最近一期）。

    返回: {"date", "total": 披露家数, "reduce_top": 户数减少(筹码集中)Top8, "add_top": 户数增加(筹码分散)Top8, "note"}
    """
    t0 = time.time()
    try:
        p = ROOT / "data_warehouse" / "market" / "gdhs_all.parquet"
        if not p.exists():
            return {"date": trade_date, "total": 0, "note": "gdhs_all.parquet 不存在"}, \
                SourceStatus("gdhs", False, (time.time() - t0) * 1000, "file missing")
        df = pd.read_parquet(p)
        if df.empty:
            return {"date": trade_date, "total": 0, "note": "股东户数数据为空"}, \
                SourceStatus("gdhs", False, (time.time() - t0) * 1000, "empty")
        name_col = "名称" if "名称" in df.columns else None
        code_col = "代码" if "代码" in df.columns else None
        chg_col = "股东户数-增减比例" if "股东户数-增减比例" in df.columns else None
        reduce_top, add_top = [], []
        if chg_col and name_col:
            d2 = df.dropna(subset=[chg_col])
            if not d2.empty:
                d2 = d2.sort_values(chg_col)  # 负=减少=集中
                for _, r in d2.head(8).iterrows():
                    reduce_top.append({"name": str(r[name_col]),
                                       "code": str(r[code_col]) if code_col else "",
                                       "chg": float(r[chg_col])})
                for _, r in d2.tail(8).iterrows():
                    add_top.append({"name": str(r[name_col]),
                                    "code": str(r[code_col]) if code_col else "",
                                    "chg": float(r[chg_col])})
        cutoff = df["股东户数统计截止日-本次"].max() if "股东户数统计截止日-本次" in df.columns else None
        return {"date": str(cutoff) if cutoff is not None else trade_date, "total": len(df),
                "reduce_top": reduce_top, "add_top": add_top,
                "note": f"本地parquet(gdhs_all) 全市场{len(df)}家，统计截止{cutoff}"}, \
            SourceStatus("gdhs", True, (time.time() - t0) * 1000, f"rows={len(df)}")
    except Exception as e:
        return {"date": trade_date, "total": 0, "note": f"股东户数获取失败: {str(e)[:100]}"}, \
            SourceStatus("gdhs", False, (time.time() - t0) * 1000, repr(e)[:160])


def _hyyyb_section(hy: dict[str, Any]) -> str:
    """营业部 metrics → markdown 章节。"""
    if not hy.get("top_buy") and not hy.get("top_sell"):
        return f"- 活跃营业部：{hy.get('note', '无数据')}"
    lines = [f"- 活跃营业部：{hy.get('note', '')}。当日上榜 {hy.get('total', 0)} 家营业部。"]
    if hy.get("top_buy"):
        lines.append("\n### 净买入 Top8 营业部")
        lines.append("| 营业部 | 净买额(亿) | 买入个股 |")
        lines.append("|---|---|---|")
        for r in hy["top_buy"]:
            lines.append(f"| {r['name']} | {r['net']/1e8:.2f} | {r['stocks']} |")
    if hy.get("top_sell"):
        lines.append("\n### 净卖出 Top8 营业部")
        lines.append("| 营业部 | 净买额(亿) | 买入个股 |")
        lines.append("|---|---|---|")
        for r in hy["top_sell"]:
            lines.append(f"| {r['name']} | {r['net']/1e8:.2f} | {r['stocks']} |")
    if hy.get("hot_stock"):
        lines.append("\n### 营业部聚焦个股")
        lines.append("| 个股 | 营业部提及 |")
        lines.append("|---|---|")
        for r in hy["hot_stock"]:
            lines.append(f"| {r['stock']} | {r['count']} |")
    return "\n".join(lines)


def _jgmmtj_section(jg: dict[str, Any]) -> str:
    """机构龙虎榜 metrics → markdown 章节。"""
    if not jg.get("top_net") and not jg.get("top_out"):
        return f"- 机构龙虎榜：{jg.get('note', '无数据')}"
    lines = [f"- 机构龙虎榜：{jg.get('note', '')}。当日 {jg.get('total', 0)} 家机构上榜。"]
    if jg.get("top_net"):
        lines.append("\n### 机构净买入 Top8")
        lines.append("| 名称 | 代码 | 机构净买额(亿) | 买方机构 | 卖方机构 |")
        lines.append("|---|---|---|---|---|")
        for r in jg["top_net"]:
            lines.append(f"| {r['name']} | {r['code']} | {r['net']/1e8:.2f} | {r['buy_org']} | {r['sell_org']} |")
    if jg.get("top_out"):
        lines.append("\n### 机构净卖出 Top8")
        lines.append("| 名称 | 代码 | 机构净买额(亿) |")
        lines.append("|---|---|---|")
        for r in jg["top_out"]:
            lines.append(f"| {r['name']} | {r['code']} | {r['net']/1e8:.2f} |")
    return "\n".join(lines)


def _gdhs_section(gd: dict[str, Any]) -> str:
    """股东户数 metrics → markdown 章节。"""
    if not gd.get("reduce_top") and not gd.get("add_top"):
        return f"- 股东户数：{gd.get('note', '无数据')}"
    lines = [f"- 股东户数：{gd.get('note', '')}"]
    if gd.get("reduce_top"):
        lines.append("\n### 筹码集中（户数减少）Top8")
        lines.append("| 名称 | 代码 | 户数增减% |")
        lines.append("|---|---|---|")
        for r in gd["reduce_top"]:
            lines.append(f"| {r['name']} | {r['code']} | {r['chg']:.1f}% |")
    if gd.get("add_top"):
        lines.append("\n### 筹码分散（户数增加）Top8")
        lines.append("| 名称 | 代码 | 户数增减% |")
        lines.append("|---|---|---|")
        for r in gd["add_top"]:
            lines.append(f"| {r['name']} | {r['code']} | {r['chg']:.1f}% |")
    return "\n".join(lines)


def build_oil_impact(trade_date: str) -> tuple[list[str], SourceStatus, dict[str, Any]]:
    """Build A-share-facing oil conclusions from the shared oil source layer.

    The A-share daily only needs tradable implications, not a full oil report.
    When a recent oil archive exists, keep a short provenance excerpt; otherwise
    fall back to the fixed cross-asset signal map in data_sources.py.
    """
    t0 = time.time()
    if get_international_oil_sources is None:
        status = SourceStatus("data_sources.get_international_oil_sources", False, 0, "import failed")
        return [
            "- 原油来源缺失，不能确认最新油价、库存、美元和实际利率组合；A股成本端与风险偏好判断降权。",
            "- 行业映射仍按固定传导：油价上行利多上游资源、油服、油运和煤化工相对表现，压制航空、交运、轮胎、包装材料和部分化工下游利润；油价下行则相反。",
        ], status, {"ok": False, "note": "import failed"}
    try:
        oil = get_international_oil_sources()
        data = oil.data or {}
        archive = data.get("archive") or {}
        content = str(archive.get("content") or "")
        archive_path = str(archive.get("archive_path") or "")
        excerpt_lines: list[str] = []
        for raw in content.splitlines():
            line = raw.strip()
            if not line or line.startswith("|") or set(line) <= {"-", "|", " "}:
                continue
            if any(k in line for k in ["油", "WTI", "Brent", "库存", "美元", "实际利率", "通胀", "风险偏好", "人民币", "航空", "化工"]):
                excerpt_lines.append(line.lstrip("-#* "))
            if len(excerpt_lines) >= 4:
                break
        archive_note = "；".join(excerpt_lines[:3]) if excerpt_lines else "最新原油报告未提取到可用摘要，使用固定跨资产映射。"
        lines = [
            f"- **来源**：{archive_path or '未找到桌面原油报告'}；{oil.note}。",
            f"- **油价/库存/美元/实际利率**：{archive_note} 对A股只看风险偏好和成本利润再分配，不写空泛联动。若油价与美元/实际利率共振上行，权益风险偏好承压；若油价回落且美元、实际利率不再上冲，成长和消费成本端压力缓和。",
            "- **受益方向**：油价或成品油裂解价差走强时，上游资源、油服、油运、煤化工相对占优；若库存累积压制油价，上述方向弹性下降，交易上更看个股兑现和供给约束。",
            "- **承压/修复方向**：油价上行会挤压航空、交运、轮胎、包装材料、化工下游利润；油价下行则改善成本端，但需要成交额放大确认，不用单日油价替代行业趋势。",
            "- **宏观传导**：油价上行推升输入型通胀和PPI预期，可能压缩宽松想象，并通过人民币、北向/外资风险偏好影响大盘成长估值；油价下行则缓和通胀约束，但若来自需求走弱，也会压制周期和出口链预期。",
        ]
        status = SourceStatus("data_sources.get_international_oil_sources", bool(oil.ok), (time.time() - t0) * 1000, oil.note)
        return lines, status, {"ok": oil.ok, "archive_path": archive_path, "note": oil.note, "excerpt": excerpt_lines[:4], "cross_asset_signals": data.get("cross_asset_signals")}
    except Exception as e:
        status = SourceStatus("data_sources.get_international_oil_sources", False, (time.time() - t0) * 1000, repr(e)[:180])
        return [
            f"- 原油来源读取失败：{repr(e)[:120]}。本段只保留固定行业传导，不能确认最新油价/库存数据。",
            "- 油价上行偏利多上游资源、油服、油运、煤化工，偏压制航空、交运、轮胎、包装材料和化工下游；油价下行则方向相反。",
            "- 对通胀预期、人民币、北向/外资风险偏好的判断降权，等待下一份原油报告或官方 EIA/OPEC 数据修复。",
        ], status, {"ok": False, "error": repr(e)[:180]}


def build_cross_asset_impact(trade_date: str) -> tuple[list[str], list[SourceStatus], dict[str, Any]]:
    """Build the required oil/cross-asset section and record every fallback used."""
    lines, oil_status, oil_meta = build_oil_impact(trade_date)
    statuses = [oil_status]
    meta: dict[str, Any] = {"oil": oil_meta}

    t0 = time.time()
    if get_cross_asset_fallback_sources is None:
        statuses.append(SourceStatus("data_sources.get_cross_asset_fallback_sources", False, 0, "import failed"))
        fallback_data: dict[str, Any] = {}
    else:
        try:
            fallback = get_cross_asset_fallback_sources()
            fallback_data = fallback.data or {}
            meta["cross_asset_fallback_sources"] = fallback.to_dict()
            statuses.append(SourceStatus("data_sources.get_cross_asset_fallback_sources", bool(fallback.ok), (time.time() - t0) * 1000, fallback.note))
        except Exception as e:
            fallback_data = {}
            meta["cross_asset_fallback_error"] = repr(e)[:180]
            statuses.append(SourceStatus("data_sources.get_cross_asset_fallback_sources", False, (time.time() - t0) * 1000, repr(e)[:180]))

    t0 = time.time()
    if get_nasdaq_api is None:
        nasdaq_note = "import failed"
        nasdaq_ok = False
        meta["nasdaq_api"] = {"ok": False, "note": nasdaq_note}
    else:
        try:
            nasdaq = get_nasdaq_api()
            nasdaq_ok = bool(nasdaq.ok)
            nasdaq_note = nasdaq.note
            meta["nasdaq_api"] = nasdaq.to_dict()
        except Exception as e:
            nasdaq_ok = False
            nasdaq_note = repr(e)[:180]
            meta["nasdaq_api"] = {"ok": False, "note": nasdaq_note}
    statuses.append(SourceStatus("data_sources.get_nasdaq_api", nasdaq_ok, (time.time() - t0) * 1000, nasdaq_note))

    nasdaq_attempts: list[dict[str, Any]] = []
    if fetch_nasdaq_data_link_dataset is None:
        statuses.append(SourceStatus("data_sources.fetch_nasdaq_data_link_dataset", False, 0, "import failed"))
    else:
        for dataset in ["CHRIS/CME_CL1", "CHRIS/ICE_B1"]:
            t0 = time.time()
            try:
                result = fetch_nasdaq_data_link_dataset(dataset, params={"rows": 5})
                nasdaq_attempts.append(result.to_dict())
                statuses.append(SourceStatus(f"nasdaq_data_link.{dataset}", bool(result.ok), (time.time() - t0) * 1000, result.note))
            except Exception as e:
                nasdaq_attempts.append({"dataset": dataset, "ok": False, "note": repr(e)[:180]})
                statuses.append(SourceStatus(f"nasdaq_data_link.{dataset}", False, (time.time() - t0) * 1000, repr(e)[:180]))
    meta["nasdaq_attempts"] = nasdaq_attempts

    rate_series = (fallback_data.get("rates_and_dollar") or {}).get("series") or {}
    lines.extend([
        "- **美元/利率/人民币**：跨资产 fallback 已登记 FRED 美元、名义利率、实际利率和通胀预期序列；若美元与实际利率上行，A股估值端先压制成长和小微盘，高股息/资源相对抗压；若美元和实际利率回落，人民币压力缓和，成长修复的胜率提高。",
        f"- **可用序列**：油价使用 WTI/Brent 与本地原油归档；利率/美元 fallback 序列包括 {', '.join(str(v) for v in list(rate_series.values())[:8]) or '缺失'}。这些序列用于方向验证，日内交易仍以A股成交额和宽度为准。",
    ])
    if nasdaq_attempts:
        ok_attempts = [a for a in nasdaq_attempts if a.get("ok")]
        if ok_attempts:
            ds = ok_attempts[0].get("data", {}).get("dataset") or ok_attempts[0].get("name") or "Nasdaq Data Link"
            newest = ok_attempts[0].get("data", {}).get("newest_available_date")
            lines.append(f"- **Nasdaq Data Link**：已用保存的 nasdaq_api 尝试商品连续合约数据，成功数据集 {ds}，最新可用日期 {newest or '未披露'}；用于交叉验证油价方向。")
        else:
            reasons = "；".join(f"{a.get('data', {}).get('dataset') or a.get('dataset')}: {a.get('note')}" for a in nasdaq_attempts[:2])
            lines.append(f"- **Nasdaq Data Link**：已读取 nasdaq_api 配置并尝试 CHRIS 原油/Brent 连续合约，当前未取得订阅数据：{reasons}。本段结论改用原油归档、OPEC/EIA/FRED 官方源登记和固定行业映射，跨资产精度降权。")
    else:
        lines.append("- **Nasdaq Data Link**：底层函数不可用，不能尝试全球/商品/利率订阅数据；跨资产结论降权。")
    lines.append("- **行业映射**：上游资源、油服、煤化工看油价和库存方向；化工下游、航空交运、轮胎包装看成本挤压或修复；人民币与外资风险偏好决定大盘成长估值弹性，小微盘仍优先服从A股宽度和跌停约束。")
    return lines, statuses, meta


def resolve_default_trade_date(now: datetime | None = None) -> tuple[str, SourceStatus]:
    """Return the latest completed A-share trading date for unattended cron runs."""
    t0 = time.time()
    now = now or datetime.now()
    today = now.strftime("%Y-%m-%d")
    start = (now - timedelta(days=14)).strftime("%Y-%m-%d")
    latest_allowed = today if (now.hour, now.minute) >= (15, 30) else (now - timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        import baostock as bs

        lg = bs.login()
        rows: list[dict[str, str]] = []
        try:
            if lg.error_code != "0":
                raise RuntimeError(lg.error_msg)
            rs = bs.query_trade_dates(start_date=start, end_date=today)
            while rs.next():
                rows.append(dict(zip(rs.fields, rs.get_row_data())))
        finally:
            try:
                bs.logout()
            except Exception as e:
                logging.getLogger(__name__).error(f"[a_share_daily_report] 操作失败: {e}", exc_info=True)
        open_days = sorted(
            r["calendar_date"]
            for r in rows
            if str(r.get("is_trading_day")) == "1" and r.get("calendar_date") <= latest_allowed
        )
        if open_days:
            trade_date = open_days[-1].replace("-", "")
            detail = f"now={now.strftime('%Y-%m-%d %H:%M')}, latest_completed={trade_date}"
            return trade_date, SourceStatus("baostock.query_trade_dates", True, (time.time() - t0) * 1000, detail)
    except Exception as e:
        fallback = latest_allowed.replace("-", "")
        return fallback, SourceStatus("baostock.query_trade_dates", False, (time.time() - t0) * 1000, f"fallback={fallback}; {repr(e)[:140]}")
    fallback = latest_allowed.replace("-", "")
    return fallback, SourceStatus("baostock.query_trade_dates", False, (time.time() - t0) * 1000, f"no open day found; fallback={fallback}")


def build_skill_analysis(
    idx: pd.DataFrame,
    core_breadth: dict[str, Any],
    hot_metrics: dict[str, Any],
    sector_up: pd.DataFrame,
    sector_down: pd.DataFrame,
    sector_amount: pd.DataFrame,
    concept_up: pd.DataFrame,
    concept_down: pd.DataFrame,
    margin_metrics: dict[str, Any],
) -> list[str]:
    sh = idx[idx["指数"] == "上证指数"].iloc[0] if not idx.empty and (idx["指数"] == "上证指数").any() else None
    cy = idx[idx["指数"] == "创业板指"].iloc[0] if not idx.empty and (idx["指数"] == "创业板指").any() else None
    hs300 = idx[idx["指数"] == "沪深300"].iloc[0] if not idx.empty and (idx["指数"] == "沪深300").any() else None
    zz1000 = idx[idx["指数"] == "中证1000"].iloc[0] if not idx.empty and (idx["指数"] == "中证1000").any() else None

    width = safe_float(core_breadth.get("上涨占比%"))
    equal_ret = safe_float(core_breadth.get("全A等权涨跌幅%"))
    limit_down = safe_float(core_breadth.get("近似跌停"))
    hot_ret = safe_float(hot_metrics.get("热股等权涨跌幅%"))
    tmt_margin_delta = safe_float(margin_metrics.get("TMT较前日变化(亿)"))

    sh_ma144 = safe_float(sh.get("距MA144%")) if sh is not None else None
    sh_ma300 = safe_float(sh.get("距MA300%")) if sh is not None else None
    cy_ma144 = safe_float(cy.get("距MA144%")) if cy is not None else None
    hs300_ret = safe_float(hs300.get("涨跌幅%")) if hs300 is not None else None
    zz1000_ret = safe_float(zz1000.get("涨跌幅%")) if zz1000 is not None else None

    market_pulse = "市场脉冲："
    if width is not None and width >= 60 and equal_ret is not None and equal_ret > 0:
        market_pulse += "宽度显著修复，赚钱效应从个别权重扩散到多数股票；但仍需结合指数趋势确认。"
    elif width is not None and width < 35:
        market_pulse += "宽度不足，市场仍是弱修复或退潮，仓位应服从风险预算。"
    else:
        market_pulse += "宽度处于中间区间，偏结构行情，不能只看指数涨跌下结论。"

    trend = "Murphy趋势："
    if sh_ma144 is not None and sh_ma144 < 0 and sh_ma300 is not None and sh_ma300 > 0:
        trend += "上证跌破144MA但仍在300MA上方，属于牛市中后期的中级调整/震荡带，不宜把短线修复直接当新主升。"
    elif sh_ma144 is not None and sh_ma144 > 0 and sh_ma300 is not None and sh_ma300 > 0:
        trend += "上证同时站上144MA与300MA，中长期趋势仍保持多头框架。"
    else:
        trend += "指数趋势信号混杂，144MA/300MA仍是主判据，20/60MA仅作短线节奏。"
    if cy_ma144 is not None:
        trend += f" 创业板距144MA {cy_ma144}%，用于判断成长风格是否重新回到趋势线上方。"

    breadth = "Appel宽度："
    if limit_down is not None and limit_down < 50:
        breadth += "跌停数量低于50，恐慌扩散暂缓；"
    elif limit_down is not None and limit_down >= 80:
        breadth += "跌停数量超过80，仍是风险优先；"
    else:
        breadth += "跌停数量处于观察区；"
    breadth += f"涨幅行业集中在 {top_names(sector_up, 5)}，跌幅行业集中在 {top_names(sector_down, 5)}。"

    volume_price = "量价分析：" + describe_volume_price(core_breadth, idx)

    reversal = "多空转折："
    if hot_ret is not None and equal_ret is not None and hot_ret > equal_ret:
        reversal += "热股/高关注池强于全A等权，说明资金仍愿意追逐高辨识度方向；后续要看强势板块回踩是否缩量，避免假突破。"
    else:
        reversal += "热股未明显强于全A时，不把单日反抽视作破底翻；需要放量突破、回踩不破和跌停收敛三者配合。"

    wu = "伍朝辉起涨点："
    if not sector_amount.empty:
        leading_amount = top_names(sector_amount, 5)
        wu += f"成交额前列方向为 {leading_amount}。若这些方向同时位居涨幅前列，可视为量峰推动；若成交额大但涨幅弱或为负，则更像分歧换手。"
    else:
        wu += "成交额板块缺失，不能判断量峰推动。"

    wyckoff = "Wyckoff结构："
    if width is not None and width >= 60 and hs300_ret is not None and zz1000_ret is not None and zz1000_ret < hs300_ret:
        wyckoff += "宽度修复但小盘弱于大盘，可能是局部吸筹/修复与高位派发并存，不能简单按全面拉抬处理。"
    elif width is not None and width >= 60:
        wyckoff += "宽度扩散时更接近拉抬阶段，但仍需连续性验证。"
    else:
        wyckoff += "结构未扩散前，更可能是震荡吸筹或弱反弹。"

    valuation = "估值/风险管理："
    if tmt_margin_delta is not None and tmt_margin_delta < 0:
        valuation += f"TMT融资余额较前日减少 {tmt_margin_delta} 亿元，成长高弹性方向的杠杆资金仍在收缩；即使板块反弹，也应降低追高仓位。"
    else:
        valuation += "融资余额未显示明显降杠杆时，可结合板块估值分位和盈利预期做二次筛选，但仍不以单日涨幅替代内在价值判断。"

    concepts = "概念板块："
    concepts += f"涨幅概念集中在 {top_names(concept_up, 8)}；跌幅概念集中在 {top_names(concept_down, 8)}。概念只作情绪与主题温度，不替代行业成交额主线。"

    return [market_pulse, trend, breadth, volume_price, reversal, wu, wyckoff, valuation, concepts]



def score_technical_row(row: pd.Series) -> tuple[int, list[str]]:
    """Return a compact trend score and reasons for one period/index row."""
    score = 0
    reasons: list[str] = []
    bias144 = safe_float(row.get("距MA144%") if "距MA144%" in row else row.get("bias144"))
    bias300 = safe_float(row.get("距MA300%") if "距MA300%" in row else row.get("bias300"))
    macd_hist = safe_float(row.get("MACD柱") if "MACD柱" in row else row.get("macd_hist"))
    macd_dif = safe_float(row.get("MACD_DIF") if "MACD_DIF" in row else row.get("macd_dif"))
    macd_dea = safe_float(row.get("MACD_DEA") if "MACD_DEA" in row else row.get("macd_dea"))
    rsi14 = safe_float(row.get("RSI14") if "RSI14" in row else row.get("rsi14"))
    k = safe_float(row.get("KDJ_K") if "KDJ_K" in row else row.get("kdj_k"))
    d = safe_float(row.get("KDJ_D") if "KDJ_D" in row else row.get("kdj_d"))
    cci = safe_float(row.get("CCI14") if "CCI14" in row else row.get("cci14"))

    if bias144 is not None:
        score += 2 if bias144 > 0 else -2
        reasons.append("144MA上方" if bias144 > 0 else "144MA下方")
    if bias300 is not None:
        score += 2 if bias300 > 0 else -2
        reasons.append("300MA上方" if bias300 > 0 else "300MA下方")
    if macd_hist is not None and macd_dif is not None and macd_dea is not None:
        if macd_dif > macd_dea and macd_hist > 0:
            score += 2; reasons.append("MACD多头")
        elif macd_dif < macd_dea and macd_hist < 0:
            score -= 2; reasons.append("MACD空头")
    if rsi14 is not None:
        if rsi14 >= 70:
            score += 1; reasons.append("RSI强但过热")
        elif rsi14 <= 35:
            score -= 1; reasons.append("RSI弱")
        else:
            reasons.append("RSI中性")
    if k is not None and d is not None:
        if k > d and k < 85:
            score += 1; reasons.append("KDJ改善")
        elif k < d:
            score -= 1; reasons.append("KDJ转弱")
        elif k >= 85:
            reasons.append("KDJ高位")
    if cci is not None:
        if cci >= 100:
            score += 1; reasons.append("CCI强趋势")
        elif cci <= -100:
            score -= 1; reasons.append("CCI弱趋势")
    return score, reasons


def build_technical_model(multi_tech: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if multi_tech.empty:
        return pd.DataFrame(), "技术模型缺失：日/周/月K线未成功返回。"
    rows: list[dict[str, Any]] = []
    weights = {"日线": 1.0, "周线": 1.5, "月线": 2.0}
    for index_name, g in multi_tech.groupby("指数"):
        scores: dict[str, int] = {}
        reasons_by_period: dict[str, str] = {}
        weighted = 0.0
        weight_sum = 0.0
        for _, r in g.iterrows():
            period = str(r.get("周期"))
            sc, rs = score_technical_row(r)
            scores[period] = sc
            reasons_by_period[period] = "、".join(rs[:5])
            w = weights.get(period, 1.0)
            weighted += sc * w
            weight_sum += w
        total = round(weighted / weight_sum, 2) if weight_sum else None
        if total is None:
            level = "缺失"
        elif total >= 3:
            level = "趋势多头"
        elif total >= 1:
            level = "偏强/修复"
        elif total > -1:
            level = "震荡"
        elif total > -3:
            level = "偏弱/反弹观察"
        else:
            level = "趋势空头"
        rows.append({
            "指数": index_name,
            "日线分": scores.get("日线"),
            "周线分": scores.get("周线"),
            "月线分": scores.get("月线"),
            "加权分": total,
            "模型结论": level,
            "日线要点": reasons_by_period.get("日线"),
            "周线要点": reasons_by_period.get("周线"),
            "月线要点": reasons_by_period.get("月线"),
        })
    out = pd.DataFrame(rows).sort_values("加权分", ascending=False, na_position="last")
    weak = out[out["加权分"].fillna(0) <= -1]
    strong = out[out["加权分"].fillna(0) >= 1]
    if len(strong) and len(weak):
        summary = f"技术结构分化：偏强 {top_names(strong.rename(columns={'指数':'板块'}), 5)}；偏弱 {top_names(weak.rename(columns={'指数':'板块'}), 5)}。仓位看强弱扩散，不看单一指数。"
    elif len(strong):
        summary = f"技术面偏修复：{top_names(strong.rename(columns={'指数':'板块'}), 6)} 评分为正；仍需宽度和成交额确认。"
    elif len(weak):
        summary = f"技术面偏弱：{top_names(weak.rename(columns={'指数':'板块'}), 6)} 评分为负；日线反弹未必等于周/月趋势修复。"
    else:
        summary = "技术面整体震荡：指标无明确共振，继续以宽度、成交额、144/300MA为主判据。"
    return out, summary

def build_portfolio_section() -> list[str]:
    """Build portfolio/inventory section for the daily report."""
    lines = []
    try:
        import sqlite3
        db_path = Path.home() / ".quant_system" / "trade_log.db"
        if not db_path.exists():
            lines.append("- 📦 **持仓**：未初始化（~/.quant_system/trade_log.db 不存在）\n")
            return lines
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM positions WHERE shares > 0").fetchall()
        conn.close()
        if not rows:
            lines.append("- 📦 **持仓**：当前空仓\n")
            return lines
        positions = [dict(r) for r in rows]
        total_cost = sum(p["total_cost"] for p in positions)
        total_val = sum(p.get("total_value") or p["total_cost"] for p in positions)
        pnl = total_val - total_cost
        pnl_pct = round((total_val / total_cost - 1) * 100, 2) if total_cost > 0 else 0
        pnl_emoji = "🟢" if pnl >= 0 else "🔴"
        lines.append(f"- 📦 **持仓**：{len(positions)}只 成本{total_cost:,.0f} 市值{total_val:,.0f} "
                     f"{pnl_emoji}{pnl:+.0f} ({pnl_pct:+.2f}%)\n")
        for p in positions[:5]:
            cost = p["cost_price"]
            curr = p.get("current_price")
            if curr:
                pnl_str = f"{'🟢+' if p['pnl']>=0 else '🔴'}{p['pnl_pct']:.2f}%"
                curr_str = f"@{curr:.2f}"
            else:
                pnl_str = "待刷新"
                curr_str = f"@{cost:.2f}*"
            sig = f" [{p['signal_type']}]" if p.get("signal_type") else ""
            lines.append(f"  - {p['symbol']} {p['name'][:8]} {p['shares']}股 {curr_str} "
                         f"({pnl_str}){sig}")
        if len(positions) > 5:
            lines.append(f"  ... 共{len(positions)}只，剩余{len(positions)-5}只略\n")
        lines.append("")
    except Exception as e:
        lines.append(f"- 📦 **持仓**：读取异常 ({str(e)[:80]})\n")
    return lines


def make_report(trade_date: str) -> tuple[str, dict[str, Any]]:
    statuses: list[SourceStatus] = []
    idx, st = fetch_index_k(trade_date)
    statuses.append(st)
    multi_tech, multi_tech_status = fetch_index_multi_period_technicals(trade_date)
    statuses.append(multi_tech_status)

    if not idx.empty and "错误" in idx.columns:
        missing_kc50 = idx[(idx["指数"] == "科创50") & (idx["错误"].notna())]
        if not missing_kc50.empty:
            kc50_row, kc50_status = fetch_index_fallback_sina("科创50", "sh000688", trade_date)
            statuses.append(kc50_status)
            if kc50_row:
                idx = pd.concat([idx[~((idx["指数"] == "科创50") & (idx["错误"].notna()))], pd.DataFrame([kc50_row])], ignore_index=True)
        # V4.1 arch fix: Wind微盘股指数8841431 fallback
        missing_wind = idx[(idx["指数"] == "万得微盘股") & (idx["错误"].notna())]
        if not missing_wind.empty:
            wind_row, wind_status = fetch_wind_micro_cap_index(trade_date)
            statuses.append(wind_status)
            if wind_row:
                idx = pd.concat([idx[~((idx["指数"] == "万得微盘股") & (idx["错误"].notna()))], pd.DataFrame([wind_row])], ignore_index=True)

    missing_idx = idx[idx.get("错误").notna()] if "错误" in idx.columns else pd.DataFrame()
    idx_display = idx.drop(columns=["错误"], errors="ignore")
    if not idx_display.empty and "收盘" in idx_display.columns:
        idx_display = idx_display[pd.to_numeric(idx_display["收盘"], errors="coerce").notna()]
    if not idx_display.empty:
        idx_display = idx_display.where(pd.notna(idx_display), "缺失")

    spot = pd.DataFrame()
    breadth: dict[str, Any] = {}
    try:
        spot, breadth, st = fetch_spot(trade_date=trade_date)
        statuses.append(st)
    except Exception as e:
        statuses.append(SourceStatus("akshare.stock_zh_a_spot_sina", False, 0, repr(e)[:180]))
        breadth = {"数据说明": "全A实时行情失败，不能计算完整市场宽度"}

    if spot.empty:
        cached_breadth = load_cached_spot_breadth()
        if cached_breadth:
            breadth = cached_breadth
            statuses.append(SourceStatus("generated.a_share_data.cached_breadth", True, 0, cached_breadth.get("数据说明", "cached breadth")))

    summary, summary_statuses = fetch_exchange_summary(trade_date)
    statuses.extend(summary_statuses)

    filtered = filtered_a_share(spot)
    raw_breadth = breadth_metrics(spot, "原始全市场")
    core_breadth = breadth_metrics(filtered, "剔除ST/*ST/北交所")
    if spot.empty and breadth:
        raw_breadth = dict(breadth)
        core_breadth = dict(breadth)
    # V3 (2026-08-07): 市场口径分层（主板/创业板/科创板/双创/北交所）——用户 #11
    try:
        from quant_platform.market_segments import segment_market, style_segments, short_medium_long
        seg_metrics = segment_market(spot) if not spot.empty else {"meta": {"error": "spot empty"}}
        style_metrics = style_segments(spot) if not spot.empty else {"meta": {"error": "spot empty"}}
        seg_metrics["_style"] = style_metrics
        # V3 融合：短/中/长线（含涨停概念热点题材——用户 #13 短线侧重）
        sml = short_medium_long(spot) if not spot.empty else {}
        if sml:
            seg_metrics["_sml"] = sml
    except Exception:
        seg_metrics = {"meta": {"error": "segment_market failed"}}
    hot_metrics, hot_statuses = fetch_hot_rank_equally_weighted(filtered)
    statuses.extend(hot_statuses)
    industry, industry_status = fetch_industry_summary()
    statuses.append(industry_status)
    concept, concept_status = fetch_concept_summary()
    statuses.append(concept_status)
    # V3 融合：概念热度全景（本地 concept_board 全量，供 6 章热点识别）
    concept_heat = {}
    try:
        from quant_platform.openclaw_api import concept_heatmap
        concept_heat = concept_heatmap(8) or {}
    except Exception:
        concept_heat = {}
    margin_metrics, margin_status = fetch_margin_metrics(trade_date)
    statuses.append(margin_status)
    # V10.2: 龙虎榜（短线必看，本地 parquet 优先）
    lhb_metrics, lhb_status = fetch_lhb_daily(trade_date)
    statuses.append(lhb_status)
    # V10.3: 短线资金面扩展——活跃营业部/机构龙虎榜/股东户数（本地 parquet）
    hyyyb_metrics, hyyyb_status = fetch_hyyyb_daily(trade_date)
    statuses.append(hyyyb_status)
    jgmmtj_metrics, jgmmtj_status = fetch_jgmmtj_daily(trade_date)
    statuses.append(jgmmtj_status)
    gdhs_metrics, gdhs_status = fetch_gdhs_snapshot(trade_date)
    statuses.append(gdhs_status)

    sector_up = table_or_empty(industry, ["板块", "涨跌幅", "总成交量", "总成交额", "上涨家数", "下跌家数"], 15, "涨跌幅", False)
    sector_down = table_or_empty(industry, ["板块", "涨跌幅", "总成交量", "总成交额", "上涨家数", "下跌家数"], 15, "涨跌幅", True)
    sector_amount = table_or_empty(industry, ["板块", "涨跌幅", "总成交量", "总成交额", "上涨家数", "下跌家数"], 15, "总成交额", False)
    concept_up = table_or_empty(concept, ["板块", "涨跌幅", "换手率", "上涨家数", "下跌家数"], 20, "涨跌幅", False)
    concept_down = table_or_empty(concept, ["板块", "涨跌幅", "换手率", "上涨家数", "下跌家数"], 20, "涨跌幅", True)
    oil_lines, cross_asset_statuses, oil_meta = build_cross_asset_impact(trade_date)
    statuses.extend(cross_asset_statuses)

    risk = 0
    reasons = []
    def idx_row(name: str) -> pd.Series | None:
        hit = idx[idx["指数"] == name]
        return hit.iloc[0] if not hit.empty else None
    sh = idx_row("上证指数")
    cy = idx_row("创业板指")
    if sh is not None and safe_float(sh.get("涨跌幅%")) is not None and sh.get("涨跌幅%") <= -2:
        risk += 1; reasons.append("沪指跌幅超过2%")
    if cy is not None and safe_float(cy.get("涨跌幅%")) is not None and cy.get("涨跌幅%") <= -3:
        risk += 1; reasons.append("创业板跌幅超过3%")
    if sh is not None and safe_float(sh.get("收盘")) and safe_float(sh.get("MA144")) and sh.get("收盘") < sh.get("MA144"):
        risk += 1; reasons.append("沪指跌破144MA")
    if cy is not None and safe_float(cy.get("收盘")) and safe_float(cy.get("MA144")) and cy.get("收盘") < cy.get("MA144"):
        risk += 1; reasons.append("创业板跌破144MA")
    if safe_float(core_breadth.get("上涨占比%")) is not None and core_breadth["上涨占比%"] < 25:
        risk += 2; reasons.append("过滤后上涨占比低于25%，市场宽度极弱")
    if safe_float(core_breadth.get("近似跌停")) is not None and core_breadth["近似跌停"] >= 80:
        risk += 2; reasons.append("近似跌停数量超过80")
    risk_level = "高风险/防守" if risk >= 6 else ("中高风险/等待修复" if risk >= 4 else "中性震荡")

    emotion = pd.DataFrame([
        {"指标": "过滤后全A等权涨跌幅", "数值": format_metric(core_breadth.get("全A等权涨跌幅%"), "%")},
        {"指标": "过滤后上涨占比", "数值": format_metric(core_breadth.get("上涨占比%"), "%")},
        {"指标": "过滤后跌停占比", "数值": format_metric(core_breadth.get("跌停占比%"), "%")},
        {"指标": "热股等权涨跌幅", "数值": format_metric(hot_metrics.get("热股等权涨跌幅%"), "%")},
        {"指标": "热股上涨占比", "数值": format_metric(hot_metrics.get("热股上涨占比%"), "%")},
        {"指标": "成交额", "数值": format_metric(core_breadth.get("成交额(亿)"), "亿元")},
    ])
    style_rows = []
    for style, index_name in STYLE_INDEX_NAMES.items():
        row = idx_row(index_name)
        if row is not None:
            style_rows.append({"风格": style, "对应指数": index_name, "涨跌幅%": row.get("涨跌幅%"), "距144MA%": row.get("距MA144%"), "距300MA%": row.get("距MA300%")})
    style_df = pd.DataFrame(style_rows)
    margin_table = pd.DataFrame([
        {"指标": "全A融资余额", "数值": format_metric(margin_metrics.get("全A融资余额(亿)"), "亿元")},
        {"指标": "全A较前日变化", "数值": format_metric(margin_metrics.get("全A较前日变化(亿)"), "亿元")},
        {"指标": "全A较前日变化%", "数值": format_metric(margin_metrics.get("全A较前日变化%"), "%")},
        {"指标": "TMT融资余额", "数值": format_metric(margin_metrics.get("TMT融资余额(亿)"), "亿元")},
        {"指标": "TMT较前日变化", "数值": format_metric(margin_metrics.get("TMT较前日变化(亿)"), "亿元")},
        {"指标": "TMT较前日变化%", "数值": format_metric(margin_metrics.get("TMT较前日变化%"), "%")},
        {"指标": "TMT占全A融资余额", "数值": format_metric(margin_metrics.get("TMT占全A融资余额%"), "%")},
    ])
    tmt_industry_df = pd.DataFrame(margin_metrics.get("TMT行业分布") or [])
    skill_analysis = build_skill_analysis(
        idx_display,
        core_breadth,
        hot_metrics,
        sector_up,
        sector_down,
        sector_amount,
        concept_up,
        concept_down,
        margin_metrics,
    )

    red = idx_row("红利指数")
    date_fmt = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
    volume_price_note = describe_volume_price(core_breadth, idx_display)
    trend_cols = [c for c in ["指数", "收盘", "涨跌幅%", "距MA20%", "距MA60%", "距MA144%", "距MA300%", "年内高点回撤%", "技术信号"] if c in idx_display.columns]
    tech_cols = [c for c in ["周期", "指数", "日期", "收盘", "涨跌幅%", "距MA20%", "距MA60%", "距MA144%", "距MA300%", "MACD_DIF", "MACD_DEA", "MACD柱", "RSI6", "RSI14", "KDJ_K", "KDJ_D", "KDJ_J", "CCI14", "ATR14", "技术信号"] if not multi_tech.empty and c in multi_tech.columns]
    tech_summary = "缺失" if multi_tech.empty or not tech_cols else multi_tech[tech_cols].to_markdown(index=False)
    tech_model, tech_model_summary = build_technical_model(multi_tech)
    lines = [f"# A股量化日报 - {date_fmt}\n"]
    lines += [
        "## 1. 核心结论与仓位/风险提示\n",
        f"- **市场状态：{risk_level}**。量化风险分 {risk}/8。触发项：{'；'.join(reasons) if reasons else '暂无极端风险触发'}。",
        f"- **赚钱效应**：热股等权 {format_metric(hot_metrics.get('热股等权涨跌幅%'), '%')}，全A等权 {format_metric(core_breadth.get('全A等权涨跌幅%'), '%')}；过滤后上涨占比 {format_metric(core_breadth.get('上涨占比%'), '%')}。",
        f"- **宽度风险**：过滤后上涨 {core_breadth.get('上涨')}、下跌 {core_breadth.get('下跌')}；近似涨停 {core_breadth.get('近似涨停')}、近似跌停 {core_breadth.get('近似跌停')}。",
        f"- **量能与融资**：过滤后成交额约 {core_breadth.get('成交额(亿)')} 亿元，成交量 {core_breadth.get('成交量合计')}；全A融资 {format_metric(margin_metrics.get('全A融资余额(亿)'), '亿元')}，TMT融资 {format_metric(margin_metrics.get('TMT融资余额(亿)'), '亿元')}。",
        "- **仓位结论**：牛市中后期仍以 144MA/300MA 控制风险预算；宽度、跌停占比、成交额没有同向修复前，不用单日指数反弹放大仓位。\n",
        "## 2. 指数与风格\n",
        idx_display.to_markdown(index=False),
        ("\n### 缺失指数数据\n" + missing_idx[["指数", "代码", "错误"]].to_markdown(index=False) if not missing_idx.empty else ""),
        "\n### 大盘股、小盘股、微盘股表现\n",
        style_df.to_markdown(index=False),
        "\n## 3. 赚钱效应\n",
        pd.DataFrame([core_breadth, raw_breadth]).to_markdown(index=False),
        f"\n- **热股等权**：{hot_metrics.get('数据说明')}；热股等权 {format_metric(hot_metrics.get('热股等权涨跌幅%'), '%')}，热股上涨占比 {format_metric(hot_metrics.get('热股上涨占比%'), '%')}。",
        f"- **涨跌停比例**：过滤后涨停占比 {format_metric(core_breadth.get('涨停占比%'), '%')}；跌停占比 {format_metric(core_breadth.get('跌停占比%'), '%')}。",
        "- **炸板/封板质量**：当前主链路没有稳定炸板率和封板率，暂用涨跌停数量、跌停占比和热股等权替代；涨停占比低且跌停占比升高时，封板质量按弱处理。\n",
        "## 3.5 市场口径分层（主板/双创/北交所）\n",
        _segment_section(seg_metrics),
        "## 4. 量能\n",
        f"- **成交量**：过滤后全A成交量合计 {core_breadth.get('成交量合计')}。",
        f"- **成交额**：过滤后全A成交额约 {core_breadth.get('成交额(亿)')} 亿元；原始全市场成交额约 {raw_breadth.get('成交额(亿)')} 亿元。",
        f"- **放量/缩量判断**：{volume_price_note}\n",
        "## 5. 融资\n",
        f"- **数据日期**：{margin_metrics.get('日期', '缺失')}；**前一交易日**：{margin_metrics.get('前一交易日', '缺失')}。",
        f"- **TMT口径**：{margin_metrics.get('TMT口径', '缺失')}。",
        margin_table.to_markdown(index=False),
        "\n### TMT 融资余额行业分布\n",
        (tmt_industry_df.to_markdown(index=False) if not tmt_industry_df.empty else "缺失"),
        "\n## 6. 板块/行业/概念\n",
        "### 涨幅靠前行业\n",
        sector_up.to_markdown(index=False),
        "\n### 跌幅靠前行业\n",
        sector_down.to_markdown(index=False),
        "\n### 成交额靠前行业\n",
        sector_amount.to_markdown(index=False),
        "\n### 涨幅靠前概念\n",
        concept_up.to_markdown(index=False),
        "\n### 跌幅靠前概念\n",
        concept_down.to_markdown(index=False),
        "\n### 板块强弱与量价归因\n",
        f"- **涨幅行业主线**：{top_names(sector_up, 8)}。",
        f"- **跌幅行业主线**：{top_names(sector_down, 8)}。",
        f"- **成交额主线**：{top_names(sector_amount, 8)}。",
        f"- **量价归因**：{volume_price_note}",
        "\n### 概念热度全景（本地 504 板块）\n",
        _concept_heat_section(concept_heat),
        "\n## 6.5 龙虎榜（短线资金动向）\n",
        _lhb_section(lhb_metrics),
        "\n## 6.6 活跃营业部（游资席位）\n",
        _hyyyb_section(hyyyb_metrics),
        "\n## 6.7 机构龙虎榜\n",
        _jgmmtj_section(jgmmtj_metrics),
        "\n## 6.8 股东户数（筹码集中度）\n",
        _gdhs_section(gdhs_metrics),
        "\n## 7. 趋势与技术指标\n",
        "- **核心均线**：300MA/144MA 是主判据；20MA/60MA 只作短线辅助。技术指标用于确认节奏，不替代宽度、成交额和跌停约束。",
        (idx_display[trend_cols].to_markdown(index=False) if trend_cols else "缺失"),
        "\n### 日线/周线/月线 MACD、均线、RSI、KDJ、CCI\n",
        tech_summary,
        "\n### 日/周/月技术模型评分\n",
        (tech_model.to_markdown(index=False) if not tech_model.empty else "缺失"),
        f"\n- **模型结论**：{tech_model_summary}",
        "\n### 技术信号用法\n",
        "- **MACD**：DIF>DEA且柱体为正才视作多头确认；若价格在144/300MA下方，MACD金叉只当反弹。",
        "- **RSI/KDJ/CCI**：RSI>70、KDJ高位、CCI>100 表示强趋势但追高风险上升；RSI<35或CCI<-100 优先看止跌，不直接抄底。",
        "- **日/周/月共振**：日线修复、周线未破、月线在144/300MA上方才提高趋势仓位；若周/月线转弱，日线指标只用于减仓节奏。",
        "\n### Skills 综合研判\n",
        "\n".join(f"- **{item.split('：', 1)[0]}**：{item.split('：', 1)[1] if '：' in item else item}" for item in skill_analysis),
        "\n## 8. 原油/跨资产影响\n",
        "\n".join(oil_lines),
        "\n## 9. 情绪指标汇总\n",
        emotion.to_markdown(index=False),
        "\n### ML 市场风险评分（集成分析）\n",
    ]
    try:
        from quant_system.portfolio_risk import compute_ml_risk_score
        ml_risk = compute_ml_risk_score()
        ml_lines = [
            f"- **ML综合风险分**：{ml_risk.get('ml_risk_score', 'N/A')}/100 — {ml_risk.get('ml_risk_level', '未知')}（置信度 {float(ml_risk.get('ml_confidence',0))*100:.0f}%）",
            f"- **市场温度**：{ml_risk.get('market_temperature', 'N/A')} | **广度比**：{ml_risk.get('breadth_ratio', 'N/A')} | **144线上占比**：{ml_risk.get('pct_above_ma144', 'N/A')}% | **300线上占比**：{ml_risk.get('pct_above_ma300', 'N/A')}%",
            f"- **市场阶段**：{ml_risk.get('market_regime', '未知')}",
            f"- **触发信号**：{'、'.join(ml_risk.get('signals',[])) if ml_risk.get('signals') else '无特别信号'}",
            f"- **ML解读**：{'⚠️ 市场当前处于风险预警状态，建议控制仓位、降低杠杆、优先防守' if (ml_risk.get('ml_risk_score',50) or 50) >= 55 else '市场环境中性偏安全，可维持正常仓位但需关注宽度和量能变化' if (ml_risk.get('ml_risk_score',50) or 50) >= 40 else '✅ 市场环境相对安全，可适度积极但注意节奏'}。",
        ]
    except Exception as e:
        ml_lines = [f"- ML市场风险评分暂不可用：{e}"]
    lines += ml_lines
    lines += ["\n### 量化信号解读\n",
        "- **趋势**：牛市中后期核心观察 144MA/300MA；MA20/MA60 只作短线辅助。",
        "- **宽度**：主口径剔除 ST、*ST、北交所。上涨占比低于 25% 是明显弱势，低于 15% 是极弱；修复第一阈值看 35%，强修复看 50%。",
        "- **跌停**：近似跌停超过 80 家时，优先控制风险；降到 50 家以下才算恐慌缓和。",
        "- **量能**：关注成交量/成交额是否与方向一致；高成交下跌说明抛压真实，缩量止跌才有观察价值。\n",
        "## 10. 数据源、失败项、fallback\n",
        pd.DataFrame([s.__dict__ for s in statuses]).to_markdown(index=False),
        "\n## 数据源策略\n",
        "- 主源：baostock 指数日/周/月 K 线与技术指标；AkShare 新浪全 A 实时行情；AkShare 交易所摘要交叉验证；同花顺行业摘要做板块涨跌和成交额。",
        "- 技术指标：底层用 pandas 从 OHLCV 计算 MA5/10/20/30/60/120/144/200/250/300、MACD、RSI6/12/14/24、KDJ、CCI14、ATR14；不依赖 ta-lib，避免环境依赖失效。",
        "- 热股口径：外部热股榜优先；若动态热榜接口失败，使用过滤后全 A 成交额前100只作为市场关注池，计算等权涨跌幅和上涨占比，并在数据说明中标注。",
        "- 融资余额：沪深交易所两融汇总计算全A融资余额；沪深两融个股明细叠加 baostock 证监会行业分类计算 TMT 融资余额。两融数据按最近披露信用交易日与前一交易日比较。",
        "- 原油/跨资产影响：优先读取最近国际原油报告、get_cross_asset_fallback_sources()、get_nasdaq_api()；Nasdaq Data Link 订阅数据失败时，列明失败原因并回到 OPEC/EIA/FRED 与本地归档。",
        "- 排除主链路：东方财富全量分页接口在本机多次断连/卡死，仅保留为人工排查或短超时备选。",
        "- 若任一接口失败，报告保留成功部分并标注失败，不编造缺失数据。\n",
        "## 11. 备案与投递状态\n",
        "- 本节由 report_delivery.finalize_report(...) 统一生成；桌面路径、有道云、飞书、微信、QQ、webchat 状态以后续页脚为准。\n",
        "## 附录：持仓相关观察\n",
    ]
    lines += build_portfolio_section()
    if red is not None:
        lines.append(f"- **红利指数**：收盘 {red.get('收盘')}，涨跌幅 {red.get('涨跌幅%')}%，距 MA20 {red.get('距MA20%')}%，距 MA60 {red.get('距MA60%')}%。")
    lines += [
        "- **牧原股份/生猪链**：若市场系统性杀跌但猪价/产能逻辑未破，继续按估值预警区间跟踪，不因单日情绪放大仓位。",
        "- **紫金矿业/有色**：拆分商品价格趋势与 A 股风险偏好。铜金价格未破趋势时，股价回撤更偏估值/情绪冲击。",
        "- **黄金宏观**：黄金仍看美元、美债实际利率、避险需求，不直接由 A 股涨跌推导。",
        "- **159792 中概互联**：按 0.428-0.55 区间管理定投节奏，不因单日情绪冲动加仓。\n",
        "## 附录：明日观察清单\n",
        "| 观察项 | 修复信号 | 风险延续信号 |\n|---|---|---|\n| 过滤后上涨占比 | >35%，强修复 >50% | <25% |\n| 过滤后近似跌停 | <50 | >=80 |\n| 指数位置 | 收回144MA，强修复看300MA | 继续远离144MA/300MA |\n| 板块结构 | 放量上涨板块扩散 | 高成交板块继续杀跌 |\n| 热股等权 | 转正且上涨占比>50% | 继续弱于全A等权 |\n",
    ]
    meta = {
        "trade_date": trade_date,
        "risk": risk,
        "risk_level": risk_level,
        "breadth": core_breadth,
        "raw_breadth": raw_breadth,
        "hot_metrics": hot_metrics,
        "margin_metrics": margin_metrics,
        "lhb_metrics": lhb_metrics,
        "hyyyb_metrics": hyyyb_metrics,
        "jgmmtj_metrics": jgmmtj_metrics,
        "gdhs_metrics": gdhs_metrics,
        "skill_analysis": skill_analysis,
        "sector_up": sector_up.to_dict(orient="records"),
        "sector_down": sector_down.to_dict(orient="records"),
        "sector_amount": sector_amount.to_dict(orient="records"),
        "technical_multi_period": multi_tech.to_dict(orient="records"),
        "technical_model": tech_model.to_dict(orient="records"),
        "technical_model_summary": tech_model_summary,
        "concept_up": concept_up.to_dict(orient="records"),
        "concept_down": concept_down.to_dict(orient="records"),
        "statuses": [s.__dict__ for s in statuses],
        "exchange_summary": summary,
        "missing_indices": missing_idx.to_dict(orient="records") if not missing_idx.empty else [],
        "oil_impact": oil_meta,
    }
    return "\n".join(lines), meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="trade date YYYYMMDD; default is latest completed A-share trading day")
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-finalize", action="store_true", help="write report/meta only; skip report_delivery finalization for tests")
    parser.add_argument("--deliver", action="store_true", help="实际投递飞书/微信/QQ（默认仅生成与备案，不发送外部消息）")
    parser.add_argument("--channels", default="feishu", help="逗号分隔渠道，如 feishu,qq；仅 --deliver 时生效")
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    date_status = None
    if args.date:
        trade_date = args.date
    else:
        trade_date, date_status = resolve_default_trade_date()
    report, meta = make_report(trade_date)
    if date_status is not None:
        meta["statuses"].insert(0, date_status.__dict__)
        marker = "## 10. 数据源、失败项、fallback\n"
        if marker in report:
            status_table = pd.DataFrame(meta["statuses"]).to_markdown(index=False)
            before, after = report.split(marker, 1)
            rest = after.split("\n\n## 数据源策略\n", 1)
            if len(rest) == 2:
                report = before + marker + status_table + "\n\n## 数据源策略\n" + rest[1]
    date_fmt = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
    out = Path(args.out) if args.out else (report_path("a_share_daily", d=date_fmt) if report_path else REPORT_DIR / f"{date_fmt}-A股量化日报.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    # Keep the long-form legacy Markdown, while writing the machine-readable
    # research contract beside it for the unified report viewer.
    try:
        from report_contract import make_report_contract, validate_and_write_contract
        source_items = []
        evidence_items = []
        for index, item in enumerate(meta.get("statuses") or [], 1):
            source_id = f"src-daily-{index}"
            source_items.append({
                "id": source_id,
                "name": item.get("name") or f"日报来源 {index}",
                "kind": "market",
                "status": "ok" if item.get("ok") else "failed",
                "observed_at": trade_date,
                "locator": item.get("name"),
                "note": item.get("detail") or item.get("error") or "",
            })
        primary_source = source_items[0]["id"] if source_items else "src-daily-undisclosed"
        if not source_items:
            source_items = [{"id": primary_source, "name": "日报来源未声明", "kind": "unknown", "status": "failed", "observed_at": trade_date, "note": "没有可登记来源"}]
        for claim, value, unit in (
            ("过滤后上涨占比", core_breadth.get("上涨占比%"), "pct"),
            ("过滤后成交额", core_breadth.get("成交额(亿)"), "CNY_100m"),
            ("风险分", meta.get("risk"), "score"),
            ("融资余额", margin_metrics.get("全A融资余额(亿)"), "CNY_100m"),
        ):
            if value is not None:
                evidence_items.append({"claim": claim, "value": value, "unit": unit, "observed_at": trade_date, "source_ref": primary_source, "quality": "primary"})
        refs = [f"ev-{i + 1}" for i in range(len(evidence_items))]
        research_contract = make_report_contract(
            "daily_market", f"A股量化日报 {date_fmt}", date_fmt,
            subject={"id": "ashare-market", "name": "A股市场", "kind": "market"},
            report_id=f"daily_market:ashare-market:{date_fmt}",
            summary={"stance": "defensive" if meta.get("risk", 0) >= 4 else "observe", "summary": f"市场状态：{meta.get('risk_level', '未知')}；风险分 {meta.get('risk', '未知')}/8。", "horizon": "intraday", "confidence": None},
            evidence=evidence_items,
            transmission=[{"from": "宽度/量能/融资", "to": "风险预算", "mechanism": "宽度、成交额与融资变化共同校验风险偏好", "direction": "mixed", "evidence_refs": refs, "confidence": None}],
            risks=[{"description": reason, "trigger": reason, "impact": "控制仓位并等待修复", "severity": "high" if meta.get("risk", 0) >= 6 else "medium", "evidence_refs": refs[:3]} for reason in (meta.get("skill_analysis") or meta.get("statuses") or [])[:4] if isinstance(reason, str)],
            actions=[{"action": "validate", "target": "次日市场宽度与量能", "condition": "上涨占比、成交额和指数趋势至少两类同向修复", "invalidated_by": "来源失败、数据日期错位或跌停扩散", "horizon": "next_session", "evidence_refs": refs}],
            data_gaps=[{"field": str(row.get("指数") or "index"), "reason": "source_failed", "impact": str(row.get("错误") or "指数证据缺失"), "fallback": "不补零，不形成强结论", "as_of": date_fmt} for row in (meta.get("missing_indices") or [])],
            sources=source_items,
            chapters=[{"title": "市场状态", "conclusion": f"{meta.get('risk_level', '未知')}，风险分 {meta.get('risk', '未知')}/8", "evidence": refs, "implication": "先确认宽度、量能和融资，再调整风险预算", "next_check": "次日复核修复信号"}, {"title": "行业与技术", "conclusion": "行业、概念和技术指标作为辅助证据", "evidence": refs, "implication": "不以单一热点或技术金叉代替市场门控", "next_check": "检查主线扩散和成交额"}],
            metadata={"legacy_meta_path": str(DATA_DIR / f"{trade_date}-meta.json"), "legacy_fields": ["breadth", "statuses", "oil_impact"]},
        )
        validate_and_write_contract(out, research_contract)
        meta["research_contract"] = research_contract
    except Exception as exc:
        meta["research_contract_error"] = str(exc)[:240]
    delivery_status = None
    if finalize_report is not None and not args.no_finalize:
        try:
            channels = [c.strip() for c in str(args.channels).split(",") if c.strip()] if args.deliver else None
            delivery_status = finalize_report(
                out,
                probe_youdao=False,
                deliver=args.deliver,
                channels=channels,
            )
        except Exception as e:
            delivery_status = {"error": repr(e)[:240]}
    (DATA_DIR / f"{trade_date}-meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)
    print(json.dumps({"risk_level": meta["risk_level"], "risk": meta["risk"], "breadth": meta["breadth"], "statuses": meta["statuses"], "delivery_status": delivery_status}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
