"""
腾讯原生 API — A 股日 K 线直连，绕过 akshare 的 buggy wrapper。

端点: proxy.finance.qq.com  /  web.ifzq.gtimg.cn
海外可用性: ✅（和 qt.gtimg.cn 同源，英国已验证）
"""

from __future__ import annotations
import logging

import re
import time

import pandas as pd
import requests as _req

# demjson 在 akshare 内捆绑，优先用它，否则回退标准 json
try:
    from akshare.utils.demjson import demjson as _demjson
except ImportError:
    try:
        import demjson as _demjson
    except ImportError:
        _demjson = None


def fetch_tencent_daily(
    symbol: str,
    start: str = "20000101",
    end: str = "",
    adjust: str = "qfq",
    timeout: int = 20,
) -> pd.DataFrame:
    """从腾讯原生接口获取 A 股日 K 线（一次性跨年）。

    Parameters
    ----------
    symbol : str — 6 位股票代码，如 "600519"
    start : str — 开始日期 YYYYMMDD
    end : str — 结束日期 YYYYMMDD，空则到今天
    adjust : str — "qfq"(前复权) / "hfq"(后复权) / ""(不复权)
    timeout : int — 单次请求超时

    Returns
    -------
    pd.DataFrame with columns: date, open, close, high, low, volume, amount
    """
    code = _to_tencent_code(symbol)
    end = end or time.strftime("%Y%m%d")

    start_year = int(start[:4]) if len(start) >= 4 else 2000
    end_year = int(end[:4]) if len(end) >= 4 else int(time.strftime("%Y"))

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://finance.qq.com",
    }
    url = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"

    all_rows: list[dict] = []
    # 接口的 640 条上限足以覆盖两个自然年。按两年批次请求，避免原来
    # 每只股票逐年重复拉取重叠区间；仍逐批记录失败年份，禁止静默吞错。
    failed_years: list[str] = []
    for year in range(start_year, end_year + 1, 2):
        batch_end_year = min(year + 1, end_year)
        adj_key = adjust if adjust else ""
        params = {
            "_var": f"kline_day{adj_key}{year}",
            "param": f"{code},day,{year}-01-01,{batch_end_year + 1}-12-31,640,{adjust}",
            "r": str(time.time()),
        }
        try:
            r = _req.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 200:
                rows = _parse_proxy_kline(r.text, code, adjust)
                if rows is not None and not rows.empty:
                    all_rows.append(rows)
                else:
                    failed_years.extend(str(y) for y in range(year, batch_end_year + 1))
            else:
                failed_years.extend(str(y) for y in range(year, batch_end_year + 1))
        except Exception as e:
            failed_years.extend(str(y) for y in range(year, batch_end_year + 1))
            logging.getLogger(__name__).error(f"[sources_tencent] {year}-{batch_end_year} 拉取失败: {e}", exc_info=True)
        time.sleep(0.2)  # rate limit

    if failed_years:
        logging.getLogger(__name__).warning(
            f"[sources_tencent] {symbol} 请求区间有 {len(failed_years)} 年缺失: {','.join(failed_years)}"
        )

    if all_rows:
        df = pd.concat(all_rows, ignore_index=True)
        df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
        # Filter by actual start/end
        mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
        out = df[mask].reset_index(drop=True)
        # 标记缺失年份（供上层感知降级）
        out.attrs["source_missing_years"] = failed_years
        return out

    return pd.DataFrame()


def _to_tencent_code(symbol: str) -> str:
    """600519 → sh600519, 002714 → sz002714, 832000 → bj832000.

    P2-Q2-fix: M191 统一市场前缀 (4/8/920xxx→bj), 修正北交所被归为 sh 的问题.
    """
    from .sources_all import market_prefix

    s = str(symbol).strip().zfill(6)
    return market_prefix(s) + s


def _decode_jsonp(text: str) -> dict | None:
    """从 JSONP 响应中提取 JSON 对象，支持多种解码方式。"""
    if not text or not text.strip():
        return None

    # Try demjson (bundled in akshare, handles JS-style JSON)
    if _demjson is not None:
        try:
            if "=" in text:
                data_text = text[text.find("={") + 1:]
                if data_text.strip():
                    return _demjson.decode(data_text)
        except Exception as e:
            logging.getLogger(__name__).error(f"[sources_tencent] 操作失败: {e}", exc_info=True)

    # Fallback: extract JSON with regex (for standard JSONP)
    try:
        match = re.search(r"=\s*(\{.*\})\s*$", text, re.DOTALL)
        if match:
            import json
            return json.loads(match.group(1))
    except Exception as e:
        logging.getLogger(__name__).error(f"[sources_tencent] 操作失败: {e}", exc_info=True)

    # Last resort: strip var= prefix and try json.loads
    try:
        if "=" in text:
            data_text = text[text.index("=") + 1:].strip().rstrip(";")
            import json
            return json.loads(data_text)
    except Exception as e:
        logging.getLogger(__name__).error(f"[sources_tencent] 操作失败: {e}", exc_info=True)

    return None


def _parse_proxy_kline(text: str, code: str, adjust: str = "") -> pd.DataFrame | None:
    """解析腾讯 proxy 端点的日 K 线响应。

    P2-Q2-fix: L202 依据请求的 adjust 显式选择对应键 (qfq→qfqday, hfq→hfqday,
    不复权→day), 避免服务端同时返回 day 与 qfqday 时静默取到不复权数据.
    """
    try:
        decoded = _decode_jsonp(text)
        if decoded is None:
            return None

        data = decoded.get("data", {})
        if not isinstance(data, dict):
            return None

        pref_key = {"qfq": "qfqday", "hfq": "hfqday"}.get(adjust or "", "day")
        # code might be under the full symbol or a different key
        kline_data = None
        used_adjust = pref_key
        for key in [code, code.upper(), code.lower()]:
            entry = data.get(key, {})
            if isinstance(entry, dict):
                kline_data = entry.get(pref_key)
                if kline_data is None:
                    # 审计 2026-08-16：显式键缺失时回退必须显式告警并标记所用复权口径，
                    # 禁止静默拿不复权/不同复权数据冒充请求口径
                    for alt in ("qfqday", "hfqday", "day"):
                        kline_data = entry.get(alt)
                        if kline_data:
                            used_adjust = alt
                            logging.getLogger(__name__).warning(
                                f"[sources_tencent] 请求调整口径 {adjust or 'day'} 缺失，"
                                f"已降级使用键 {alt}（{code}）——复权口径与请求不一致"
                            )
                            break
                if kline_data:
                    break
        if not kline_data or not isinstance(kline_data, list):
            return None

        rows = []
        for d in kline_data:
            if not isinstance(d, list) or len(d) < 6:
                continue
            try:
                amount = 0.0
                # P2-Q2-fix: L201 空 amount (通常是最新未完成 bar) 置 0.0 保留行,
                # 不再因 float('') 抛 ValueError 而整根 K 线被静默剔除
                if len(d) > 8 and d[8] not in (None, ""):
                    try:
                        amount = float(d[8])
                    except (ValueError, TypeError):
                        amount = 0.0
                rows.append({
                    "date": str(d[0]),
                    "open": float(d[1]),
                    "close": float(d[2]),
                    "high": float(d[3]),
                    "low": float(d[4]),
                    "volume": float(d[5]),
                    "amount": amount,
                })
            except (ValueError, IndexError, TypeError):
                continue

        if not rows:
            return None

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df["symbol"] = code
        df["tencent_price_key"] = used_adjust
        return df
    except Exception:
        return None


def _parse_trend_kline(text: str) -> pd.DataFrame | None:
    """备用解析（web.ifzq 端点返回的周趋势，只有日期和价格）。"""
    try:
        decoded = _decode_jsonp(text)
        if decoded is None:
            return None
        data = decoded.get("data", [])
        if not data or not isinstance(data, list):
            return None

        rows = []
        for d in data:
            if not isinstance(d, list) or len(d) < 2:
                continue
            try:
                rows.append({"date": str(d[0]), "close": float(d[1])})
            except (ValueError, IndexError):
                continue

        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception:
        return None


__all__ = ["fetch_tencent_daily"]
