"""统一行情适配层：云端东方财富、本机 AkShare、Yahoo Chart。

数据源分层：东方财富及其 AkShare 封装只允许在国内云端采集并回传；本机补源顺序为 yahoo_chart -> sina_daily -> tencent_daily。
免费源仅用于缺口补齐和交叉校验，不用于声称提供 A 股完整历史。
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests


@dataclass(frozen=True)
class ProviderConfig:
    timeout: float = float(os.getenv("MARKET_DATA_TIMEOUT", "30"))
    retries: int = int(os.getenv("MARKET_DATA_RETRIES", "4"))
    min_interval: float = float(os.getenv("MARKET_DATA_MIN_INTERVAL", "0.35"))


class MarketDataError(RuntimeError):
    pass


class MarketDataProvider:
    name = "base"

    def __init__(self, config: ProviderConfig | None = None) -> None:
        self.config = config or ProviderConfig()
        self._last_request = 0.0

    def _throttle(self) -> None:
        wait = self.config.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _get(self, url: str, **kwargs: Any) -> requests.Response:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; QuantWorkbench/1.0)"}
        headers.update(kwargs.pop("headers", {}))
        last: Exception | None = None
        for attempt in range(self.config.retries):
            try:
                self._throttle()
                res = requests.get(url, timeout=self.config.timeout, headers=headers, **kwargs)
                res.raise_for_status()
                return res
            except (requests.RequestException, ValueError) as exc:
                last = exc
                if attempt + 1 < self.config.retries:
                    time.sleep(min(8.0, 0.8 * (attempt + 1)))
        raise MarketDataError(f"{self.name} request failed: {last}")


class EastmoneyProvider(MarketDataProvider):
    name = "cloud_eastmoney"
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"

    def fetch(self, code: str, start: str, end: str, adjust: str = "raw") -> pd.DataFrame:
        code = str(code).zfill(6)
        fqt = {"raw": 0, "qfq": 1, "hfq": 2}[adjust]
        market = 1 if code.startswith(("6", "9")) else 0
        params = {
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116",
            "ut": "7eea3edcaed734bea9cbfc24409ed989",
            "klt": "101",
            "fqt": fqt,
            "secid": f"{market}.{code}",
            "beg": pd.Timestamp(start).strftime("%Y%m%d"),
            "end": pd.Timestamp(end).strftime("%Y%m%d"),
            "_": time.time_ns(),
        }
        res = self._get(self.url, params=params, headers={"Referer": "https://quote.eastmoney.com/"})
        payload = res.json()
        rows = ((payload.get("data") or {}).get("klines") or [])
        out: list[dict[str, Any]] = []
        for row in rows:
            fields = row.split(",")
            if len(fields) < 7:
                continue
            out.append(
                {
                    "date": fields[0],
                    "open": float(fields[1]),
                    "close": float(fields[2]),
                    "high": float(fields[3]),
                    "low": float(fields[4]),
                    "volume": float(fields[5]),
                    "amount": float(fields[6]),
                    "pct_chg": float(fields[8]) if len(fields) > 8 and fields[8] else None,
                    "turnover": float(fields[10]) if len(fields) > 10 and fields[10] else None,
                }
            )
        return _normalise(out, code, self.name, adjust)


class AkshareProvider(MarketDataProvider):
    name = "local_akshare"

    def fetch(self, code: str, start: str, end: str, adjust: str = "raw") -> pd.DataFrame:
        try:
            import akshare as ak
        except ImportError as exc:
            raise MarketDataError("AkShare 未安装，请安装 akshare") from exc
        adjust_arg = {"raw": "", "qfq": "qfq", "hfq": "hfq"}[adjust]
        last: Exception | None = None
        for attempt in range(self.config.retries):
            try:
                self._throttle()
                frame = ak.stock_zh_a_hist(
                    symbol=str(code).zfill(6),
                    period="daily",
                    start_date=pd.Timestamp(start).strftime("%Y%m%d"),
                    end_date=pd.Timestamp(end).strftime("%Y%m%d"),
                    adjust=adjust_arg,
                )
                return _normalise(frame, str(code).zfill(6), self.name, adjust)
            except Exception as exc:  # AkShare wraps several HTTP clients.
                last = exc
                if attempt + 1 < self.config.retries:
                    time.sleep(min(8.0, 0.8 * (attempt + 1)))
        raise MarketDataError(f"{self.name} request failed: {last}")


class YahooChartProvider(MarketDataProvider):
    name = "yahoo_chart"

    def fetch(self, symbol: str, start: str, end: str, adjust: str = "raw") -> pd.DataFrame:
        period1 = int(pd.Timestamp(start, tz="UTC").timestamp())
        period2 = int((pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)).timestamp())
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {"period1": period1, "period2": period2, "interval": "1d", "events": "history", "includeAdjustedClose": "true"}
        payload = self._get(url, params=params).json()
        result = ((payload.get("chart") or {}).get("result") or [])
        if not result:
            raise MarketDataError(f"{self.name}: no data for {symbol}")
        result = result[0]
        q = result.get("indicators", {}).get("quote", [{}])[0]
        adj = result.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", [])
        rows = []
        for i, stamp in enumerate(result.get("timestamp", [])):
            close = q.get("close", [None])[i]
            if close is None:
                continue
            rows.append({"date": datetime.fromtimestamp(stamp, timezone.utc).date(), "open": q["open"][i], "high": q["high"][i], "low": q["low"][i], "close": close, "volume": q.get("volume", [None])[i], "amount": None, "adj_close": adj[i] if i < len(adj) else None})
        return _normalise(rows, symbol, self.name, adjust)


class SinaProvider(MarketDataProvider):
    name = "sina_daily"

    def fetch(self, code: str, start: str, end: str, adjust: str = "raw") -> pd.DataFrame:
        code = str(code).zfill(6)
        prefix = "sh" if code.startswith(("5", "6", "9")) else "sz"
        url = "https://stock.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
        payload = self._get(url, params={"symbol": prefix + code, "scale": 240, "ma": "no", "datalen": 5000}).json()
        rows = [{"date": item.get("day"), "open": item.get("open"), "high": item.get("high"), "low": item.get("low"), "close": item.get("close"), "volume": item.get("volume"), "amount": item.get("amount")} for item in (payload or [])]
        frame = _normalise(rows, code, self.name, adjust)
        return frame[(frame["date"] >= pd.Timestamp(start)) & (frame["date"] <= pd.Timestamp(end))].reset_index(drop=True)


class TencentProvider(MarketDataProvider):
    name = "tencent_daily"

    def fetch(self, code: str, start: str, end: str, adjust: str = "raw") -> pd.DataFrame:
        code = str(code).zfill(6)
        prefix = "sh" if code.startswith(("5", "6", "9")) else "sz"
        url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        payload = self._get(url, params={"param": f"{prefix}{code},day,{start},{end},5000,{adjust}"}).json()
        data = ((payload.get("data") or {}).get(prefix + code) or {})
        rows = data.get("day") or data.get("qfqday") or data.get("hfqday") or []
        frame = _normalise([{"date": r[0], "open": r[1], "close": r[2], "high": r[3], "low": r[4], "volume": r[5], "amount": r[6] if len(r) > 6 else None} for r in rows], code, self.name, adjust)
        return frame[(frame["date"] >= pd.Timestamp(start)) & (frame["date"] <= pd.Timestamp(end))].reset_index(drop=True)


def _normalise(frame: Any, code: str, source: str, adjust: str) -> pd.DataFrame:
    if isinstance(frame, list):
        frame = pd.DataFrame(frame)
    if frame is None or len(frame) == 0:
        return pd.DataFrame(columns=["date", "code", "open", "high", "low", "close", "volume", "amount", "source", "adjust"])
    frame = frame.copy()
    aliases = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount"}
    frame.rename(columns=aliases, inplace=True)
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
    for col in ("open", "high", "low", "close", "volume", "amount", "adj_close"):
        if col not in frame:
            frame[col] = pd.NA
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if adjust == "hfq" and frame["adj_close"].notna().any():
        frame["close"] = frame["adj_close"]
    frame["code"] = code
    frame["source"] = source
    frame["adjust"] = adjust
    frame["retrieved_at"] = datetime.now(timezone.utc).isoformat()
    return frame.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def fetch_with_fallback(code: str, start: str, end: str, adjust: str = "raw", config: ProviderConfig | None = None) -> pd.DataFrame:
    """本机只调用 Yahoo/新浪/腾讯；AkShare/东方财富由云端单独采集。"""
    code_text = str(code).zfill(6)
    yahoo_symbol = code_text
    if not code_text.endswith((".SS", ".SZ", ".HK", ".US")):
        yahoo_symbol = f"{code_text}.SS" if code_text.startswith(("5", "6", "688", "9")) else f"{code_text}.SZ"
    providers: list[tuple[MarketDataProvider, str]] = [
        (YahooChartProvider(config), yahoo_symbol),
        (SinaProvider(config), code),
        (TencentProvider(config), code),
    ]
    errors = []
    for provider, symbol in providers:
        try:
            frame = provider.fetch(symbol, start, end, adjust)
            if not frame.empty:
                return frame
            errors.append(f"{provider.name}: empty")
        except Exception as exc:
            errors.append(f"{provider.name}: {exc}")
    raise MarketDataError(f"{code} all providers failed; " + " | ".join(errors))
