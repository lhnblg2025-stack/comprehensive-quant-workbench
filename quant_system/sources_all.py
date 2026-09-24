"""
A 股日 K 线多源统一接入层。

按优先级: baostock → Tencent原生 → Tushare Pro → akShare(TX fallback)
各源输出统一格式：date, open, close, high, low, volume(股), amount(元)
"""

from __future__ import annotations

from pathlib import Path
import time

import pandas as pd


def fetch_daily_unified(
    symbol: str,
    start: str = "20000101",
    end: str = "",
    adjust: str = "qfq",
    timeout: int = 20,
) -> pd.DataFrame:
    """多源并联获取 A 股日 K 线，统一输出格式。

    Parameters
    ----------
    symbol : str — 6 位股票代码
    start : str — YYYYMMDD
    end : str — YYYYMMDD
    adjust : str — "qfq"(前复权) / "hfq"(后复权) / ""(不复权)
    timeout : int

    Returns
    -------
    pd.DataFrame with columns: date, open, close, high, low, volume, amount
    """
    end = end or time.strftime("%Y%m%d")

    errors: list[str] = []

    # 1. baostock (纯 Python，零依赖，海外可用)
    try:
        df = _fetch_baostock(symbol, start, end, adjust=adjust)
        if df is not None and not df.empty:
            return _normalize(df, "baostock")
    except Exception as e:
        errors.append(f"baostock={e!r}")

    # 2. Tencent 原生（海外可用，绕过 akshare bug）
    try:
        from .sources_tencent import fetch_tencent_daily

        df = fetch_tencent_daily(symbol, start=start, end=end, adjust=adjust, timeout=timeout)
        if df is not None and not df.empty:
            return _normalize(df, "tencent")
    except Exception as e:
        errors.append(f"tencent={e!r}")

    # 3. Tushare Pro（已有 token）
    try:
        df = _fetch_tushare(symbol, start.replace("-", ""), end.replace("-", ""), adjust=adjust)
        if df is not None and not df.empty:
            return _normalize(df, "tushare")
    except Exception as e:
        errors.append(f"tushare={e!r}")

    # 4. akShare Tencent fallback (original)
    try:
        import akshare as ak

        sym = _market_symbol(symbol)
        df = ak.stock_zh_a_hist_tx(
            symbol=sym, start_date=start, end_date=end,
            adjust=adjust, timeout=timeout,
        )
        if df is not None and not df.empty:
            return _normalize(df, "akshare")
    except Exception as e:
        errors.append(f"akshare_tx={e!r}")

    raise RuntimeError(
        f"所有数据源均失败 [{symbol} {start}~{end}]: {'; '.join(errors)}"
    )


# ── 各源实现 ──────────────────────────────────────────────────────────


def _fetch_baostock(symbol: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame | None:
    """baostock 直连。"""
    import baostock as bs

    bs.login()
    try:
        start_fmt = f"{start[:4]}-{start[4:6]}-{start[6:8]}"
        end_fmt = f"{end[:4]}-{end[4:6]}-{end[6:8]}"
        adjustflag = {"qfq": "2", "hfq": "1", "": "3", None: "3"}.get(adjust, "2")
        rs = bs.query_history_k_data_plus(
            # V11 审计修复（Medium）: 北交所（4/8/920）原被拼 sz → 取数失败，
            # 复用 market_prefix（已正确支持北交所 bj）。
            f"{market_prefix(symbol)}.{symbol}",
            "date,open,high,low,close,volume,amount",
            start_date=start_fmt, end_date=end_fmt,
            frequency="d", adjustflag=adjustflag,  # 2=qfq, 1=hfq, 3=unadjusted
        )
        rows = []
        while rs.next():
            row = rs.get_row_data()
            rows.append({
                "date": row[0],
                "open": row[1],
                "high": row[2],
                "low": row[3],
                "close": row[4],
                "volume": row[5],
                "amount": row[6],
            })
        return pd.DataFrame(rows) if rows else None
    finally:
        bs.logout()


def _fetch_tushare(symbol: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame | None:
    """Tushare Pro（需要 token）。"""
    import json, tushare as ts

    # P2-Q2-fix: M190 用绝对路径读取 config, 避免依赖 CWD
    # V12.3 密钥治理: env:NAME 占位符经 secret_loader 解析(环境变量/.env.secrets)
    try:
        from scripts.secret_loader import get_secret  # noqa: PLC0415
        token = get_secret("TUSHARE_API_KEY")
    except Exception:  # noqa: BLE001
        token = None
    if not token:
        config_path = Path(__file__).resolve().parent.parent / "config" / "private_data_sources.json"
        with open(config_path, encoding="utf-8") as f:
            token = json.load(f)["tushare_pro"]["api_key"]
    pro = ts.pro_api(token)

    # V11 审计修复（Medium）: 复用 market_prefix，北交所不再错归 SZ
    _pfx = market_prefix(symbol)
    exchange = {"sh": "SH", "sz": "SZ", "bj": "BJ"}.get(_pfx, "SZ")
    ts_code = f"{symbol}.{exchange}"
    if adjust in ("qfq", "hfq"):
        df = ts.pro_bar(
            ts_code=ts_code,
            start_date=start, end_date=end,
            adj=adjust,
            api=pro,
        )
    else:
        df = pro.daily(
            ts_code=ts_code,
            start_date=start, end_date=end,
        )
    if df is None or df.empty:
        return None
    df = df.rename(columns={
        "trade_date": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "vol": "volume",
        "amount": "amount",
    })
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# ── 归一化 ────────────────────────────────────────────────────────────

def _normalize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """统一字段名和数量单位。

    source     volume单位    amount单位
    ───────    ─────────    ──────────
    baostock   股(share)    元(yuan)      → 不变
    tencent    手(lot)      万元          → ×100, ×10000
    tushare    手(lot)      千元          → ×100, ×1000
    akshare    stock_zh_a_hist_tx 的英文6列中 amount 实为成交量(手),
               amount列转 volume(×100→股) 后删除 amount
    """
    out = df.copy()

    # Ensure required columns
    for col in ["date", "open", "close", "high", "low"]:
        if col not in out.columns:
            return pd.DataFrame()  # missing essential columns

    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    if "amount" in out.columns:
        out["amount"] = pd.to_numeric(out["amount"], errors="coerce")

    if source == "tencent":
        # Tencent: volume in 手 → 股, amount in 万元 → 元
        if "volume" in out.columns:
            out["volume"] = out["volume"] * 100
        if "amount" in out.columns:
            out["amount"] = out["amount"] * 10000

    elif source == "tushare":
        # Tushare: volume in 手 → 股, amount in 千元 → 元
        if "volume" in out.columns:
            out["volume"] = out["volume"] * 100
        if "amount" in out.columns:
            out["amount"] = out["amount"] * 1000

    elif source == "akshare":
        # stock_zh_a_hist_tx 的英文6列中 amount 实为成交量(手)。若无 volume，转为股数，避免把手数写成成交额。
        if "volume" not in out.columns and "amount" in out.columns:
            out["volume"] = out["amount"] * 100
            out = out.drop(columns=["amount"])

    # Standard columns
    keep = [c for c in ["date", "open", "high", "low", "close", "volume", "amount"]
            if c in out.columns]
    numeric_cols = [c for c in keep if c != "date" and out[c].dtype == object]
    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["date"] = pd.to_datetime(out["date"])
    return out[keep].dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)


def market_prefix(symbol: str) -> str:
    """统一 A 股市场前缀: bj(北交所) / sh(沪) / sz(深).

    规则 (P2-Q2-fix: M191): 4/8 开头或 920xxx → bj;
    5/6 或 900xxx → sh; 0/1/2/3 → sz.
    """
    s = str(symbol).strip().zfill(6)
    if s.startswith(("4", "8")) or s.startswith("920"):
        return "bj"
    if s.startswith(("5", "6", "9")):
        return "sh"
    return "sz"


def _market_symbol(symbol: str) -> str:
    """600519 → sh600519, 832000 → bj832000 (P2-Q2-fix: M191 北交所前缀)."""
    s = str(symbol).strip().zfill(6)
    return f"{market_prefix(s)}{s}"


__all__ = ["fetch_daily_unified"]
