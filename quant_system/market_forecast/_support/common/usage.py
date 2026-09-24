"""
usage.py — V7.0 数据源用量记账与熔断器（长期稳定方案核心）
============================================================
- 每日调用预算（SQLite 记账）：source | date | count | limit
- 熔断器：连续失败 N 次 → 熔断 T 秒 → 自动降级
- 超预算自动跳过（返回 None 由调用方走缓存/备源）
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from quant_system.market_forecast._support.common.logger import get_logger

log = get_logger("qv6.usage")

DEFAULT_DB = Path(__file__).resolve().parent.parent.parent / "data_cache" / "usage.sqlite3"

# 每日预算（免费额度长期稳定值）
DEFAULT_BUDGETS = {
    "yfinance": 30,        # 无硬限额，自控预算（日频实际 ~10 次）
    "fred": 50,            # 免费 key 120次/分，自控预算
    "alpha_vantage": 5,    # 免费 25 次/天 → 只用 5 次做交叉验证
    "gnews": 0,            # 弃用
    "mediastack": 0,       # 弃用
    "akshare": 100000,     # 无限量，仅记账（实际控速防封 IP）
    "baostock": 100000,
    "cninfo": 100000,
    # 独立熔断 source（防单接口失败拖垮 akshare 主通道）
    "kline_em": 20000,     # 东财日K
    "kline_sina": 20000,   # 新浪日K（回退）
    "akshare_cb": 5000,    # 转债（东财，偶发不稳定）
    "akshare_repo": 5000,  # 回购利率（东财，偶发不稳定）
    "akshare_movie": 500,  # 票房（猫眼/东财，不稳定）
    # 全市场仓库（market_warehouse）
    "wh_kline_em": 60000,      # 仓库东财日K（全市场抓取）
    "wh_kline_sina": 60000,    # 仓库新浪日K（回退）
    "wh_financial": 60000,     # 仓库财务（新浪）
    "wh_valuation": 60000,     # 仓库估值（baostock 回退）
    "wh_valuation_em": 60000,  # 仓库估值（东财 stock_value_em 主源）
    "wh_event": 20000,         # 仓库事件/资金域（lhb/north/margin 等）
}


@dataclass
class CircuitBreaker:
    """熔断器：连续失败 threshold 次 → 熔断 cooldown 秒。"""

    threshold: int = 3
    cooldown: float = 600.0
    _fails: int = 0
    _open_until: float = 0.0

    def record_success(self) -> None:
        self._fails = 0
        self._open_until = 0.0

    def record_failure(self) -> None:
        self._fails += 1
        if self._fails >= self.threshold:
            self._open_until = time.time() + self.cooldown
            log.warning(f"熔断器打开 {self.cooldown:.0f}s（连续失败 {self._fails}）")

    @property
    def is_open(self) -> bool:
        if self._open_until > time.time():
            return True
        if self._open_until and self._open_until <= time.time():
            self._open_until = 0.0
            self._fails = 0
        return False


class UsageTracker:
    """用量记账 + 预算控制。"""

    def __init__(self, db_path: Path | str | None = None,
                 budgets: dict | None = None):
        self.db = Path(db_path or DEFAULT_DB)
        self.db.parent.mkdir(parents=True, exist_ok=True)
        self.budgets = {**DEFAULT_BUDGETS, **(budgets or {})}
        self.breakers: dict[str, CircuitBreaker] = {}
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS usage_log (
                    source TEXT NOT NULL,
                    date TEXT NOT NULL,
                    count INTEGER NOT NULL,
                    PRIMARY KEY (source, date)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS failure_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    source TEXT NOT NULL,
                    error TEXT DEFAULT ''
                )
            """)

    def _breaker(self, source: str) -> CircuitBreaker:
        if source not in self.breakers:
            self.breakers[source] = CircuitBreaker()
        return self.breakers[source]

    # ── 查询 ──────────────────────────────────────────────

    def today_count(self, source: str, date: str | None = None) -> int:
        d = date or time.strftime("%Y-%m-%d")
        with sqlite3.connect(self.db) as conn:
            row = conn.execute(
                "SELECT count FROM usage_log WHERE source=? AND date=?",
                (source, d)).fetchone()
        return row[0] if row else 0

    def remaining(self, source: str, date: str | None = None) -> int:
        limit = self.budgets.get(source, 0)
        if limit <= 0:
            return 0
        return max(0, limit - self.today_count(source, date))

    def over_budget(self, source: str, date: str | None = None) -> bool:
        return self.remaining(source, date) <= 0

    def is_allowed(self, source: str) -> bool:
        """熔断 + 预算双检查。"""
        if self._breaker(source).is_open:
            return False
        if self.over_budget(source):
            return False
        return True

    # ── 记录 ──────────────────────────────────────────────

    def record(self, source: str, n: int = 1, date: str | None = None) -> None:
        d = date or time.strftime("%Y-%m-%d")
        with sqlite3.connect(self.db) as conn:
            conn.execute(
                "INSERT INTO usage_log (source, date, count) VALUES (?,?,?) "
                "ON CONFLICT(source, date) DO UPDATE SET count = count + excluded.count",
                (source, d, n))

    def record_success(self, source: str) -> None:
        self._breaker(source).record_success()

    def record_failure(self, source: str, error: str = "") -> None:
        self._breaker(source).record_failure()
        with sqlite3.connect(self.db) as conn:
            conn.execute("INSERT INTO failure_log (ts, source, error) VALUES (?,?,?)",
                         (time.time(), source, str(error)[:300]))

    # ── 汇总 ──────────────────────────────────────────────

    def daily_report(self, date: str | None = None) -> list[dict]:
        d = date or time.strftime("%Y-%m-%d")
        rows = []
        for src in sorted(self.budgets):
            cnt = self.today_count(src, d)
            limit = self.budgets[src]
            rows.append({
                "source": src, "date": d, "count": cnt,
                "limit": limit, "pct": round(cnt / limit * 100, 1) if limit else 100.0,
                "open": self._breaker(src).is_open,
            })
        return rows

    def failures_recent(self, minutes: int = 1440) -> int:
        since = time.time() - minutes * 60
        with sqlite3.connect(self.db) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM failure_log WHERE ts > ?", (since,)).fetchone()
        return row[0]


_default: UsageTracker | None = None


def get_tracker() -> UsageTracker:
    global _default
    if _default is None:
        _default = UsageTracker()
    return _default
