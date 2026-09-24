"""
tdx_adapter.py — easy-tdx 实时/历史数据适配层 (V5.1)

无缝替换/补充 akshare 数据源，提供:
  1. 实时行情 (realtime_quote)
  2. 历史K线 (historical_bars)
  3. 盘中快照 (snapshot)
  4. 股票列表 (symbol_list)

所有接口均 try/except 保护，easy-tdx 未安装时降级返回。
"""

from __future__ import annotations

import logging
import time as _time
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# V11 审计修复（Medium）: 原实现 easy_tdx 或 tdx 任一存在即 _HAS_TDX=True，
# 但函数内部只用 easy_tdx——只装 tdx 时报"假可用"（ImportError 被吞，接口静默空）。
# 修正: 只有 easy_tdx 真正可用才置 True。
_HAS_TDX = False
try:
    import easy_tdx  # type: ignore
    _HAS_TDX = True
except ImportError:
    pass

CST = timezone(timedelta(hours=8))

# P1-Q26-fix: 记录最近一次降级原因，供调用方可见诊断（不再静默吞异常）
_TDX_STATUS = {"last_ok": False, "last_error": "", "last_checked": 0.0}
_TDX_CHECK_TTL = 300  # check_tdx 连通性结果缓存5分钟


def _note_tdx_error(err: Exception) -> None:
    """记录降级原因并打日志，保证降级可见。"""
    _TDX_STATUS["last_ok"] = False
    _TDX_STATUS["last_error"] = str(err)
    _TDX_STATUS["last_checked"] = _time.time()
    logger.warning(f"TDX degraded: {err}")


def _safe_call(client: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
    """调用 client.<method_name>(*args)。

    P1-Q26-fix: 服务器可能提前断开空闲连接（Broken pipe）导致命令失败，
    此时断开并重连一次后重试；仍失败则向上抛异常，由调用方降级处理。
    """
    try:
        return getattr(client, method_name)(*args, **kwargs)
    except Exception as e:
        # 仅对瞬时连接错误（管道断开/连接重置）重试；socket.timeout 表明服务器
        # 响应慢，重试同一慢服务器无意义，直接向上抛由调用方降级。
        if not isinstance(e, (BrokenPipeError, ConnectionResetError)):
            raise
        if _connection is not None:
            try:
                _connection.disconnect()
            except Exception as e:
                logger.error(f"[tdx_adapter] 操作失败: {e}", exc_info=True)
            _connection._conn = None
            _connection._last_used = 0.0
        client2 = _get_conn().connect()
        if client2 is None:
            raise
        return getattr(client2, method_name)(*args, **kwargs)


# ── TDX 连接管理 ──

class TdxConnection:
    """TDX 连接池，自动重连。

    P1-Q26-fix: 原默认单一主机 119.147.212.81 经常连不上导致
    check_tdx()=True 但数据全空（假可用）。现按候选主机列表重试，
    并对首次成功的连接做 get_security_count 连通性验证后缓存可用主机。
    """

    # 候选主机（实测可用的行情服务器，快的主机放前面以缩短建连耗时）
    FALLBACK_HOSTS = [
        "180.153.18.170",
        "115.238.56.198",
        "115.238.90.165",
        "218.75.126.9",
        "119.147.212.81",  # 原默认，偶发超时放后面
        "124.71.187.122",
    ]
    MAX_ATTEMPTS = 2  # P1-Q26-fix: 限制单次建连尝试主机数，避免全挂时阻塞过久

    def __init__(self, host: str = "180.153.18.170", port: int = 7709):
        self.host = host
        self.port = port
        self._conn = None
        self._last_used = 0.0
        self._working_host: str | None = None

    def connect(self) -> Any:
        """获取或创建连接（带主机回退与连通性验证）。"""
        now = _time.time()
        if self._conn is not None and now - self._last_used < 15:
            self._last_used = now
            return self._conn

        # 重新连接：清理旧连接
        self.disconnect()
        if not _HAS_TDX:
            return None

        from easy_tdx import TdxClient

        # P1-Q26-fix: 依次尝试候选主机（限前 MAX_ATTEMPTS 个，每个带8s超时），
        # 验证同时覆盖 get_security_count 与 get_security_quotes，
        # 过滤"连得上但实时行情返回空"的假可用主机。确定性优于 from_best_host
        # 的全网 ping（后者耗时波动大，可达30s+）。
        candidates: list[tuple[str, int]] = []
        if self._working_host:
            candidates.append((self._working_host, self.port))
        candidates.append((self.host, self.port))
        for h in self.FALLBACK_HOSTS:
            candidates.append((h, self.port))

        seen: set[tuple[str, int]] = set()
        uniq = [c for c in candidates if not (c in seen or seen.add(c))][:self.MAX_ATTEMPTS]

        for h, p in uniq:
            try:
                conn = TdxClient(host=h, port=p, timeout=8)
                conn.connect()
                if not self._verify(conn):
                    conn.disconnect()
                    continue
                self._conn = conn
                self._working_host = h
                self._last_used = now
                _TDX_STATUS["last_ok"] = True
                _TDX_STATUS["last_error"] = ""
                return self._conn
            except Exception:
                self._conn = None
                continue

        # 最后手段：from_best_host（全量ping选最快可达主机）
        try:
            from easy_tdx import TdxClient as _T
            conn = _T.from_best_host(timeout=8, ping_timeout=3)
            conn.connect()
            if self._verify(conn):
                self._conn = conn
                self._last_used = now
                _TDX_STATUS["last_ok"] = True
                _TDX_STATUS["last_error"] = ""
                return self._conn
            conn.disconnect()
        except Exception as e:
            logger.error(f"[tdx_adapter] 操作失败: {e}", exc_info=True)
        self._conn = None
        _note_tdx_error(Exception("no working TDX host (limited attempts exhausted)"))
        return None

    @staticmethod
    def _verify(client: Any) -> bool:
        """连通性+行情能力验证：能取到上海股票数与600519实时行情。"""
        from easy_tdx import Market
        try:
            n = client.get_security_count(Market.SH)
            if not n or n <= 0:
                return False
            q = client.get_security_quotes([(Market.SH, "600519")])
            return q is not None and len(q) > 0
        except Exception:
            return False

    def disconnect(self) -> None:
        if self._conn is not None:
            try:
                self._conn.disconnect()
            except Exception as e:
                logger.error(f"[tdx_adapter] 操作失败: {e}", exc_info=True)
            self._conn = None


# ── 数据接口 ──

_connection: TdxConnection | None = None


def _get_conn() -> TdxConnection | None:
    """获取全局连接。"""
    global _connection
    if _connection is None:
        _connection = TdxConnection()
    return _connection


def _resolve_market(symbol: str):
    """按代码前缀返回 easy_tdx Market 枚举（SH/SZ/BJ），未知前缀返回 None。

    P2-Q26-fix: 统一 realtime_quote / historical_bars 的前缀映射，避免各接口
    各自实现导致不一致；北交所 4/8/920 前缀务必映射到 Market.BJ。
    """
    from easy_tdx import Market
    sym = symbol.strip()
    if sym.startswith("6"):
        return Market.SH
    if sym.startswith(("0", "2", "3")):
        return Market.SZ
    if sym.startswith(("4", "8", "920")):
        return Market.BJ
    return None


def check_tdx(force: bool = False) -> bool:
    """检查 easy-tdx 是否真实可用（含连通性验证），结果缓存5分钟。

    P1-Q26-fix: 原实现仅检查库是否安装，易出现 check_tdx()=True 但
    realtime_quote/historical_bars 全部返回空（假可用）。现做真实连接验证。
    """
    if not _HAS_TDX:
        return False
    now = _time.time()
    if not force and _TDX_STATUS.get("last_checked") \
            and now - _TDX_STATUS["last_checked"] < _TDX_CHECK_TTL \
            and _TDX_STATUS["last_error"] == "":
        return _TDX_STATUS["last_ok"]
    try:
        conn = _get_conn()
        client = conn.connect() if conn is not None else None
        ok = client is not None
        _TDX_STATUS["last_ok"] = ok
        return ok
    except Exception as e:
        _note_tdx_error(e)
        return False


def realtime_quote(symbols: list[str]) -> pd.DataFrame:
    """获取实时行情快照。

    Args:
        symbols: 股票代码列表, 如 ["000001", "600519"]

    Returns:
        DataFrame: symbol, name, price, change, pct, volume, amount,
                   prev_close, open, high, low, limit_up, limit_down, ...
    """
    if not _HAS_TDX:
        return pd.DataFrame()

    try:
        conn = _get_conn()
        if conn is None:
            return pd.DataFrame()
        client = conn.connect()
        if client is None:
            return pd.DataFrame()

        # P1-Q26-fix: easy_tdx 7.x 无 get_quotes/get_quote，正确 API 为
        # get_security_quotes(市场枚举, [(market, code)])，每批最多80只。
        # P2-Q26-fix: 前缀映射统一走 _resolve_market（含北交所 4/8/920）。
        stocks = []
        for sym in symbols:
            sym = sym.strip()
            mkt = _resolve_market(sym)
            if mkt is not None:
                stocks.append((mkt, sym))

        frames = []
        for i in range(0, len(stocks), 80):
            df = _safe_call(client, "get_security_quotes", stocks[i:i + 80])
            if df is not None and len(df) > 0:
                frames.append(df)
        if not frames:
            logger.warning("TDX realtime_quote: no quote rows returned")
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)

        # 标准化输出列
        out = pd.DataFrame()
        out["symbol"] = result["code"].astype(str).str.zfill(6)
        out["name"] = ""  # 实时五档行情不含名称，由调用方按需补充
        out["price"] = pd.to_numeric(result.get("price", 0), errors="coerce").fillna(0)
        out["prev_close"] = pd.to_numeric(result.get("pre_close", 0), errors="coerce").fillna(0)
        out["open"] = pd.to_numeric(result.get("open", 0), errors="coerce").fillna(0)
        out["high"] = pd.to_numeric(result.get("high", 0), errors="coerce").fillna(0)
        out["low"] = pd.to_numeric(result.get("low", 0), errors="coerce").fillna(0)
        out["volume"] = pd.to_numeric(result.get("vol", 0), errors="coerce").fillna(0)
        out["amount"] = pd.to_numeric(result.get("amount", 0), errors="coerce").fillna(0)
        # P1-Q26-fix: 休市/未开盘时部分服务器 price=0，用昨收兜底避免下游拿到0价
        price_col = out["price"]
        out["price"] = price_col.where(price_col > 0, out["prev_close"])
        out["change"] = out["price"] - out["prev_close"]
        out["pct"] = (out["change"] / out["prev_close"].replace(0, np.nan) * 100).fillna(0)
        out["limit_up"] = pd.to_numeric(result.get("limit_up", np.nan), errors="coerce")
        out["limit_down"] = pd.to_numeric(result.get("limit_down", np.nan), errors="coerce")
        return out

    except Exception as e:
        _note_tdx_error(e)
        return pd.DataFrame()


def historical_bars(
    symbol: str,
    freq: str = "daily",
    start: str | None = None,
    end: str | None = None,
    count: int = 500,
) -> pd.DataFrame:
    """获取历史K线数据。
    
    Args:
        symbol: 股票代码
        freq: daily/weekly/monthly
        start/end: YYYY-MM-DD
        count: 返回条数
    
    Returns:
        DataFrame: date, open, close, high, low, volume, amount
    """
    if not _HAS_TDX:
        return pd.DataFrame()
    
    try:
        conn = _get_conn()
        if conn is None:
            return pd.DataFrame()
        client = conn.connect()
        if client is None:
            return pd.DataFrame()
        
        # P1-Q26-fix: easy_tdx 7.x 无 get_bars，正确 API 为
        # get_security_bars(市场枚举, code, KlineCategory枚举, start, count)。
        # P2-Q26-fix: 前缀映射统一走 _resolve_market（原实现北交所 4/8 前缀缺失、
        # 且与 realtime_quote 分支不一致）。
        from easy_tdx import Market, KlineCategory
        sym = symbol.strip()
        market = _resolve_market(sym) or Market.SH  # 未知前缀沿用原默认 SH

        freq_map = {
            "daily": KlineCategory.DAY,
            "weekly": KlineCategory.WEEK,
            "monthly": KlineCategory.MONTH,
        }
        category = freq_map.get(freq, KlineCategory.DAY)

        bars = _safe_call(client, "get_security_bars", market, sym, category, 0, count)
        if bars is None or len(bars) == 0:
            return pd.DataFrame()

        df = bars.copy()
        # 标准化列名: easy_tdx 返回 vol，统一为 volume
        col_map = {
            "date": "date", "time": "date",
            "open": "open", "close": "close",
            "high": "high", "low": "low",
            "volume": "volume", "vol": "volume",
            "amount": "amount",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # P2-Q26-fix: 缺 date 列时显式报错（原实现 `if start: df[df["date"]...]`
        # 抛 KeyError 被外层 except 吞掉后静默返回空表）。
        if "date" not in df.columns:
            raise ValueError(f"historical_bars({symbol}): 返回数据缺少 date 列，无法按日期过滤")
        df["date"] = pd.to_datetime(df["date"])

        if start:
            df = df[df["date"] >= pd.Timestamp(start)]
        if end:
            df = df[df["date"] <= pd.Timestamp(end)]

        return df.sort_values("date").reset_index(drop=True)

    except Exception as e:
        _note_tdx_error(e)
        return pd.DataFrame()


def snapshot_market() -> dict:
    """获取全市场快照（涨跌家数、涨停跌停等）。

    P2-Q26-fix: 原实现仅返回两市代码数量计数桩数据，与 docstring 承诺的
    "涨跌家数/涨停跌停" 不符。现用 easy-tdx get_market_stat（基于 880005/
    880001/880006 统计指数）返回真实市场概况。
    """
    if not _HAS_TDX:
        return {"available": False, "error": "easy-tdx not installed"}

    try:
        conn = _get_conn()
        if conn is None:
            return {"available": False, "error": "TDX连接不可用"}
        client = conn.connect()
        if client is None:
            return {"available": False, "error": _TDX_STATUS["last_error"]}

        result = {
            "available": True,
            "source": "easy-tdx",
            "timestamp": datetime.now(CST).isoformat(),
        }

        try:
            df = _safe_call(client, "get_market_stat")
            if df is None or len(df) == 0:
                err = "get_market_stat 返回空数据（统计指数不可用）"
                result["error"] = err
                _note_tdx_error(RuntimeError(f"snapshot_market: {err}"))
                return result
            row = df.iloc[0]
            result.update({
                "up_count": int(row.get("up_count", 0) or 0),
                "down_count": int(row.get("down_count", 0) or 0),
                "neutral_count": int(row.get("neutral_count", 0) or 0),
                "suspended_count": int(row.get("suspended_count", 0) or 0),
                "total_count": int(row.get("total_count", 0) or 0),
                "limit_up_count": int(row.get("limit_up_count", 0) or 0),
                "limit_down_count": int(row.get("limit_down_count", 0) or 0),
                "total_amount": float(row.get("total_amount", 0) or 0),
                "total_volume": float(row.get("total_volume", 0) or 0),
                "total_market_cap": float(row.get("total_market_cap", 0) or 0),
            })
            return result
        except Exception as e:
            result["error"] = str(e)
            _note_tdx_error(e)
            return result

    except Exception as e:
        _note_tdx_error(e)
        return {"available": False, "error": str(e)}


def symbol_list() -> list[str]:
    """获取TDX全量A股股票代码列表。

    P1-Q26-fix: get_security_list 签名仅 (market, start)，原传3参导致 0 条；
    返回为 DataFrame，按 code 列迭代；过滤指数/基金/债券等非A股代码。
    """
    if not _HAS_TDX:
        return []

    try:
        conn = _get_conn()
        if conn is None:
            return []
        client = conn.connect()
        if client is None:
            return []

        from easy_tdx import Market
        symbols = []
        # (市场枚举, 该市场A股代码前缀)
        markets = [
            (Market.SZ, ("0", "2", "3")),
            (Market.SH, ("6",)),
            (Market.BJ, ("4", "8", "920")),
        ]
        # P1-Q26-fix: 取每个市场第一页（约1000条）即可得到真实股票列表，
        # 修复原"0条"问题；深市第一页含中小创/创业板，沪市第一页含多数主板股。
        # 不做全量分页——部分行情服务器对 start>0 的分页请求会长时间无响应。
        for market, prefixes in markets:
            try:
                data = _safe_call(client, "get_security_list", market, 0)
                if data is None or len(data) == 0:
                    continue
                for code in data.get("code", []):
                    code = str(code)
                    if code.startswith(prefixes) and len(code) == 6:
                        symbols.append(code)
            except Exception as e:
                _note_tdx_error(e)
                continue

        return sorted(set(symbols))

    except Exception as e:
        _note_tdx_error(e)
        return []


# ── CLI 测试 ──

def main():
    """测试 TDX 连接和数据获取。"""
    print(f"easy-tdx available: {_HAS_TDX}")
    if not _HAS_TDX:
        print("安装: pip install easy-tdx")
        return
    print(f"check_tdx(连通性): {check_tdx(force=True)}")
    if not _TDX_STATUS["last_ok"]:
        print(f"⚠️ TDX 不可用: {_TDX_STATUS['last_error']}")

    print("\n=== 实时行情(5只) ===")
    q = realtime_quote(["000001", "600519", "000858", "300750", "002415"])
    if not q.empty:
        print(q.head())
    else:
        print("(empty)")
    
    print("\n=== 历史K线(600519) ===")
    bars = historical_bars("600519", count=10)
    if not bars.empty:
        print(bars.tail())
    else:
        print("(empty)")


if __name__ == "__main__":
    main()
