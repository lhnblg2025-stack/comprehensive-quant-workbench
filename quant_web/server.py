from __future__ import annotations

# ── numpy 兼容补丁（scipy 1.9 + numpy 2.0 环境） ──
import numpy as _np
if not hasattr(_np, 'Inf'):
    _np.Inf = _np.inf
    _np.infinity = _np.inf
if not hasattr(_np, 'asfarray'):
    # numpy 2.0 移除 asfarray；scipy 旧版内部 `from numpy import asfarray` 会失败 → 别名到 asarray
    _np.asfarray = _np.asarray
del _np

import glob
import hashlib
import json
import mimetypes
import re
import sqlite3
import threading
import subprocess
import sys
import os
import tempfile
import traceback
import time
import uuid
try:
    import fcntl
except ImportError:  # Windows production node
    fcntl = None
    import msvcrt
from contextlib import contextmanager
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

import logging
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from quant_system.backtest import run_backtest, run_portfolio_backtest, grid_search
from quant_system.config import DEFAULT_PORTFOLIO, DEFAULT_STRATEGY, PortfolioConfig, StrategyConfig
from quant_system.data import fetch_daily, fetch_many, load_market_state, normalize_symbol, resolve_symbol, search_stock_name
from quant_system.indicators import add_technical_indicators, latest_indicator_snapshot
from quant_system.realtime import fetch_realtime, watchlist_from_file
from quant_system.risk import enrich_trade_plan
from quant_system.signals import latest_signal
from quant_system.indicators import add_intraday_indicators
from quant_system.global_market import fetch_global_quotes, fetch_global_kline_with_indicators, normalize_global_symbol, HK_KNOWN, US_KNOWN
from quant_system.margin import fetch_margin_summary, fetch_margin_individual
from quant_system import market_temperature, north_flow, fundamental, chart_patterns, sector_rotation, macro_calendar, earnings_calendar, market_pulse, opportunity, portfolio_risk

# ── 3.0 基础设施 ──
from quant_system.cache import cached, cache_stats, cache_clear, set_db_path as init_cache, data_freshness
from quant_system.task_queue import init as init_task_queue, get_queue_stats as get_task_stats, submit_task, list_tasks as list_task_queue
from quant_web.handlers import risk as risk_handlers
from quant_web.market_history import get_market_history, get_market_history_catalog
from quant_web.stock_analysis import (
    build_stock_profile, fuzzy_search_stocks, warehouse_status, macro_snapshot,
)
from quant_web.ic_evidence import build_ic_weight_explanation
from quant_web.decision_contract import build_decision_gate, build_score_hierarchy
from quant_web.api_catalog import build_catalog
from quant_web.handlers import backtest_console as backtest_console_handlers

# V2 多因子极端扫描引擎（用户要求：权重小但极端也提醒）
sys.path.insert(0, str(ROOT / "scripts"))
try:
    from factor_extreme_alert import (
        ALL_FACTORS as _FACTOR_DEFINITIONS,
        compute_factors as _factor_scan,
    )
    _HAS_FACTOR_SCAN = True
except Exception:  # noqa: BLE001
    _FACTOR_DEFINITIONS = []
    _HAS_FACTOR_SCAN = False
from quant_system.trade_db import get_position, get_positions, get_trades, add_buy, add_sell, get_summary, init_db

# V5 legacy deep API remains a compatibility adapter; new UI uses neutral paths.
from quant_system.v5_api import route as v5_route

DEFAULT_CHAT_MODEL = "chatgpt-plus/gpt-5.5"
# V12.3 密钥治理: chat_id 从 .env.secrets 解析(FEISHU_CHAT_ID)。
# W2.5 安全收尾: 移除硬编码 open_id 兜底(fail-loud)，缺失时飞书推送显式失败，不泄露真实凭据。
try:
    sys.path.insert(0, str(ROOT / "scripts")) if str(ROOT / "scripts") not in sys.path else None
    from secret_loader import get_secret as _get_secret  # noqa: E402
    ALERT_FEISHU_TARGET = _get_secret("FEISHU_CHAT_ID") or ""
except Exception:  # noqa: BLE001 - 解析失败 fail-loud
    ALERT_FEISHU_TARGET = ""

# W2.5 安全：有副作用/成本的 API 端点需 X-API-Key（环境变量 QUANT_WEB_API_KEY）。
# 默认策略：服务默认只绑定 127.0.0.1（QUANT_WEB_BIND_HOST 可显式改地址）。
# 当绑定不是回环时（0.0.0.0/局域网），/api/* 除公开白名单外一律要求 key；
# 未配置 key 且绑定非回环时直接拒绝启动相应暴露，绝不匿名开放。
# 公开白名单（只读健康/版本/认证状态，不含业务数据）。
_PUBLIC_PREFIXES = (
    "/api/health", "/api/v1/health", "/api/v4/health", "/api/v11/health",
    "/api/livez", "/api/readyz", "/api/version", "/api/auth_status",
)
_QUANT_WEB_API_KEY = os.environ.get("QUANT_WEB_API_KEY", "").strip()
_ALLOW_UNAUTH = os.environ.get("QUANT_WEB_ALLOW_UNAUTH", "1").lower() not in {"0", "false", "no"}
_QUANT_WEB_BIND_HOST = os.environ.get("QUANT_WEB_BIND_HOST", "127.0.0.1").strip()
_PROCESS_STARTED_AT = datetime.now().astimezone().isoformat(timespec="seconds")
_IS_LOOPBACK_BIND = _QUANT_WEB_BIND_HOST in ("127.0.0.1", "::1", "localhost")
_READY_STATE = {"ready": False, "phase": "starting", "started_at": _PROCESS_STARTED_AT}
_READY_LOCK = threading.RLock()


def _set_ready(ready: bool, phase: str) -> None:
    with _READY_LOCK:
        _READY_STATE["ready"] = bool(ready)
        _READY_STATE["phase"] = str(phase)


def _ready_info() -> dict:
    with _READY_LOCK:
        return dict(_READY_STATE)
OPENCLAW_BIN = "openclaw"
SKILLS_DIR = ROOT / "skills"
CACHE_DB = ROOT / "config" / "api_cache.sqlite3"
DELIVERY_CONFIG = ROOT / "config" / "report_delivery.json"
PORTFOLIO_FILE = ROOT / "config" / "portfolio.json"
MARKET_REGIME_SKILL = SKILLS_DIR / "market-regime-gmm" / "SKILL.md"
_REPORT_ROOT_ENV = os.environ.get("QUANT_REPORT_ROOT", "").strip()
REPORT_DIRS = ([Path(_REPORT_ROOT_ENV).expanduser()] if _REPORT_ROOT_ENV else []) + [
    Path.home() / "Desktop" / "每日任务" / "A股量化日报",
    Path.home() / "Desktop" / "周报",
    Path.home() / "Desktop" / "黄金分析",
]
_CACHE: dict[str, tuple[float, object]] = {}
_PAPER_ORDER_IDS: dict[str, dict] = {}
_PAPER_ORDER_LOCK = threading.RLock()


@contextmanager
def _paper_order_file_lock():
    """Serialize paper preview consumption across threads and processes."""
    lock_path = _paper_order_file().with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        else:
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            except OSError:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            else:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def _paper_order_fingerprint(direction: str, symbol: str, shares: int, price: float) -> str:
    raw = f"paper-order-preview.v1|{direction}|{normalize_symbol(symbol)}|{int(shares)}|{float(price):.8f}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _paper_order_file() -> Path:
    return ROOT / "generated" / "paper_order_intents.json"


def _load_paper_order_records() -> dict[str, dict]:
    path = _paper_order_file()
    try:
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
    except (OSError, ValueError, TypeError):
        pass
    return {}


def _write_paper_order_records(records: dict[str, dict]) -> None:
    path = _paper_order_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)
    _PAPER_ORDER_IDS.clear()
    _PAPER_ORDER_IDS.update(records)


def _persist_paper_order_record(order_id: str, record: dict) -> None:
    with _paper_order_file_lock(), _PAPER_ORDER_LOCK:
        records = _load_paper_order_records()
        records[order_id] = _json_safe(record)
        # Drop expired previews, but retain executed records for audit/idempotency.
        now = int(time.time())
        records = {key: value for key, value in records.items()
                   if value.get("kind") != "preview" or int(value.get("expires_at", 0) or 0) >= now}
        _write_paper_order_records(records)


def _get_paper_order_record(order_id: str) -> dict | None:
    with _PAPER_ORDER_LOCK:
        if order_id not in _PAPER_ORDER_IDS:
            _PAPER_ORDER_IDS.update(_load_paper_order_records())
        return _PAPER_ORDER_IDS.get(order_id)


def _claim_paper_preview(order_id: str, fingerprint: str) -> tuple[dict | None, dict | None]:
    """Atomically claim a preview; return (record, duplicate_record)."""
    with _paper_order_file_lock(), _PAPER_ORDER_LOCK:
        records = _load_paper_order_records()
        record = records.get(order_id)
        if record and record.get("kind") != "preview":
            return None, record
        if (not record or int(record.get("expires_at", 0) or 0) < int(time.time())
                or record.get("fingerprint") != fingerprint):
            return None, None
        records[order_id] = {**record, "kind": "processing", "claimed_at": int(time.time())}
        _write_paper_order_records(records)
        return records[order_id], None


def _release_paper_preview(order_id: str, record: dict) -> None:
    """Restore a claimed preview when the ledger mutation fails."""
    with _paper_order_file_lock(), _PAPER_ORDER_LOCK:
        records = _load_paper_order_records()
        records[order_id] = {**record, "kind": "preview"}
        _write_paper_order_records(records)
SKILL_FEATURED_CATEGORIES = {
    "趋势跟踪": ["turtle-trading-system", "livermore-pivotal-point-breakout", "moving-average-channel-timing"],
    "量化因子": ["factor-neutralization", "multi-factor-stock-selection", "alpha-information-time-scale"],
    "市场结构": ["market-breadth-appel", "market-regime-gmm", "internal-order-six-stage-cycle"],
    "资金管理": ["capital-allocation-risk-aversion", "financial-ml-bet-sizing", "turtle-n-volatility-position-sizing"],
    "风控": ["volatility-stop-entry-risk", "trader-vic-alligator-loss-cut", "risk-management-specialist"],
    "宏观": ["macro-four-driver-asset-map", "big-cycle-empire", "fiat-money-inflation-anchor"],
}


def _cache_get(key: str, ttl_seconds: int):
    now = time.time()
    hit = _CACHE.get(key)
    if hit and now - hit[0] <= ttl_seconds:
        return hit[1]
    try:
        with sqlite3.connect(CACHE_DB) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS api_cache (key TEXT PRIMARY KEY, ts REAL, data TEXT)")
            row = conn.execute("SELECT ts, data FROM api_cache WHERE key=?", (key,)).fetchone()
        if row and now - float(row[0]) <= ttl_seconds:
            data = json.loads(row[1])
            _CACHE[key] = (float(row[0]), data)
            return data
    except Exception:
        return None
    return None


def _cache_set(key: str, data) -> None:
    safe = _json_safe(data)
    ts = time.time()
    _CACHE[key] = (ts, safe)
    try:
        CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(CACHE_DB) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS api_cache (key TEXT PRIMARY KEY, ts REAL, data TEXT)")
            conn.execute("REPLACE INTO api_cache (key, ts, data) VALUES (?, ?, ?)", (key, ts, json.dumps(safe, ensure_ascii=False)))
    except Exception as e:
        logging.getLogger(__name__).error(f"[server] 操作失败: {e}", exc_info=True)


_DEEP_RESULTS: dict = {}
_DEEP_LOCK = threading.RLock()


def _async_deep(key: str, ttl: float, compute, send_json) -> None:
    """深度计算异步化：首次触发后台计算并立即返回 computing，之后返回缓存。"""
    import threading
    now = time.time()
    with _DEEP_LOCK:
        ent = _DEEP_RESULTS.get(key)
        if ent and ent.get("data") is not None and now - ent["ts"] < ttl:
            send_json(ent["data"])
            return
        if ent and ent.get("computing"):
            send_json({"ok": True, "status": "computing", "cached": False,
                       "key": key, "hint": "深度计算进行中，请稍后刷新"})
            return
        # Claim the key while holding the same lock used by readers/writers.
        _DEEP_RESULTS[key] = {"ts": now, "data": None, "computing": True}

    def _bg():
        try:
            data = compute()
            with _DEEP_LOCK:
                _DEEP_RESULTS[key] = {"ts": time.time(), "data": data, "computing": False}
        except Exception as e:  # noqa: BLE001
            with _DEEP_LOCK:
                _DEEP_RESULTS[key] = {"ts": time.time(),
                                      "data": {"ok": False, "error": str(e)[:300]},
                                      "computing": False}

    threading.Thread(target=_bg, daemon=True).start()
    send_json({"ok": True, "status": "computing", "cached": False,
               "key": key, "hint": "深度计算已启动，请稍后刷新获取结果"})


class QuantHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[quant-web] " + fmt % args + "\n")
        # V11 (2026-08-07): API 审计日志（骨架收口 P2）
        # 记录所有 /api 调用到 logs/api_audit.log，供追溯/监控
        try:
            msg = fmt % args
            if "/api/" in msg and "GET /api/health" not in msg:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(LOGS_DIR / "api_audit.log", "a", encoding="utf-8") as f:
                    f.write(f"{ts} {msg}\n")
        except Exception as e:
            logging.getLogger(__name__).error(f"[server] 操作失败: {e}", exc_info=True)

    def _path_protected(self, path: str) -> bool:
        """default-deny：/api/* 除公开白名单外全部需要鉴权。

        白名单匹配必须是精确命中或后跟 "/"，防止 /api/livezXYZ 之类
        的前缀拼接绕过 key 边界。
        """
        private_asset = (
            path == "/report.html"
            or path == "/research_report.html"
            or path.startswith("/report_assets/")
            or path.startswith("/research_image_asset/")
        )
        if not path.startswith("/api/") and not private_asset:
            return False
        if path.startswith("/api/"):
            for prefix in _PUBLIC_PREFIXES:
                if path == prefix or path.startswith(prefix + "/"):
                    return False
        return True

    def _require_key(self) -> bool:
        """未配置 key 时：回环绑定允许本机匿名（风险可控）；非回环绑定强制拒收。"""
        if not _QUANT_WEB_API_KEY:
            if _IS_LOOPBACK_BIND:
                return _ALLOW_UNAUTH  # 本机匿名仅允许显式开发配置
            return False  # 非回环未配置 key → 一律拒绝
        return self.headers.get("X-API-Key") == _QUANT_WEB_API_KEY

    def _reject_unauthorized(self) -> bool:
        """受保护路径且未通过鉴权时返回 True（调用方应 401 并 return）。"""
        if not self._path_protected(urlparse(self.path).path):
            return False
        if self._require_key():
            return False
        self._send_json({"error": "unauthorized", "hint": "需要 X-API-Key"}, status=401)
        return True

    def do_GET(self) -> None:
        if self._reject_unauthorized():
            return
        parsed = urlparse(self.path)
        if parsed.path == "/api/auth_status":
            # 公开状态接口：只报告是否配置鉴权，不返回密钥或其派生值。
            self._send_json({
                "ok": True,
                "auth_required": bool(_QUANT_WEB_API_KEY),
                "bind_host": _QUANT_WEB_BIND_HOST,
            })
            return
        if parsed.path == "/api/version":
            self._send_json(self._version_info())
            return
        if parsed.path == "/api/livez":
            self._send_json({"ok": True, "service": "quant-web", "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
            return
        if parsed.path == "/api/readyz":
            # 轻量就绪：只读进程状态，不做重 I/O；启动预热期间明确返回 503。
            info = _ready_info()
            self._send_json({"ok": info["ready"], "service": "quant-web",
                             "ready": info["ready"], "phase": info["phase"],
                             "started_at": info["started_at"],
                             "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
                            status=200 if info["ready"] else 503)
            return
        if parsed.path == "/api/catalog":
            self._send_json(build_catalog())
            return
        if parsed.path == "/api/workbench":
            self._handle_workbench(parse_qs(parsed.query))
            return
        if parsed.path == "/api/market":
            self._send_json(load_market_state())
            return
        if parsed.path == "/api/market_recap":
            self._handle_market_recap(parse_qs(parsed.query))
            return
        if parsed.path == "/api/market_weekly":
            self._handle_market_weekly(parse_qs(parsed.query))
            return
        if parsed.path == "/api/research_report":
            self._handle_research_report(parse_qs(parsed.query))
            return
        if parsed.path == "/api/config":
            self._send_json(_get_defaults())
            return
        if parsed.path == "/api/watchlist":
            self._send_json(_get_watchlist())
            return
        if parsed.path == "/api/skills":
            self._handle_skills(parse_qs(parsed.query))
            return
        if parsed.path == "/api/history":
            self._handle_history(parse_qs(parsed.query))
            return
        if parsed.path == "/api/stock_overview":
            self._handle_stock_overview(parse_qs(parsed.query))
            return
        if parsed.path == "/api/sector":
            self._handle_sector(parse_qs(parsed.query))
            return
        if parsed.path == "/api/realtime":
            self._handle_realtime(parse_qs(parsed.query))
            return
        if parsed.path == "/api/intraday":
            self._handle_intraday(parse_qs(parsed.query))
            return
        if parsed.path == "/api/search_stock":
            self._handle_search_stock(parse_qs(parsed.query))
            return
        if parsed.path == "/api/global_quotes":
            self._handle_global_quotes(parse_qs(parsed.query))
            return
        if parsed.path == "/api/alert/push":
            self._send_json({"ok": False, "error": "POST required"}, status=405)
            return
        if parsed.path == "/api/alert/config":
            self._send_json({"ok": True, "data": {"notify": True, "channels": ["feishu", "wechat"]}})
            return
        if parsed.path == "/api/market/regime":
            self._handle_market_regime(parse_qs(parsed.query))
            return
        if parsed.path == "/api/portfolio":
            self._handle_portfolio(parse_qs(parsed.query))
            return
        if parsed.path == "/api/backtest_dashboard":
            self._handle_backtest_dashboard(parse_qs(parsed.query))
            return
        if parsed.path == "/api/btc/overview":
            backtest_console_handlers.handler_overview(parse_qs(parsed.query), self._send_json)
            return
        if parsed.path == "/api/btc/strategies":
            backtest_console_handlers.handler_strategies(parse_qs(parsed.query), self._send_json)
            return
        if parsed.path == "/api/btc/run":
            backtest_console_handlers.handler_run(parse_qs(parsed.query), self._send_json)
            return
        if parsed.path == "/api/btc/task":
            backtest_console_handlers.handler_task(parse_qs(parsed.query), self._send_json)
            return
        if parsed.path == "/api/btc/selection":
            backtest_console_handlers.handler_selection(parse_qs(parsed.query), self._send_json)
            return
        if parsed.path == "/api/strategy-lab/admission":
            from quant_web.handlers import strategy_lab
            strategy_lab.handler_admission(parse_qs(parsed.query), self._send_json)
            return
        if parsed.path == "/api/strategy-lab/data-status":
            from quant_web.handlers import strategy_lab
            strategy_lab.handler_data_status(parse_qs(parsed.query), self._send_json)
            return
        if parsed.path == "/api/strategy-lab/strategies":
            from quant_web.handlers import strategy_lab
            strategy_lab.handler_list(parse_qs(parsed.query), self._send_json)
            return
        if parsed.path == "/api/strategy-lab/strategy":
            from quant_web.handlers import strategy_lab
            strategy_lab.handler_get(parse_qs(parsed.query), self._send_json)
            return
        if parsed.path == "/api/strategy-lab/template":
            from quant_web.handlers import strategy_lab
            strategy_lab.handler_template(parse_qs(parsed.query), self._send_json)
            return
        if parsed.path == "/api/factor_analysis":
            self._handle_factor_analysis(parse_qs(parsed.query))
            return
        if parsed.path == "/api/news_sentiment":
            self._handle_news_sentiment(parse_qs(parsed.query))
            return
        if parsed.path == "/api/v1/news_sentiment_market":
            self._handle_news_sentiment_market(parse_qs(parsed.query))
            return
        if parsed.path == "/api/v1/market_breadth":
            self._handle_fusion_api("market_breadth_api", parse_qs(parsed.query))
            return
        if parsed.path == "/api/v1/market_regime":
            self._handle_fusion_api("market_regime_api", parse_qs(parsed.query))
            return
        if parsed.path == "/api/v1/seasonality":
            self._handle_fusion_api("seasonality_api", parse_qs(parsed.query))
            return
        if parsed.path == "/api/v1/limit_up_depth":
            self._handle_fusion_api("limit_up_depth_api", parse_qs(parsed.query))
            return
        if parsed.path == "/api/v1/broker_profiles":
            self._handle_fusion_api("broker_profiles", parse_qs(parsed.query))
            return
        # ── V11 深度投研分析 ──
        if parsed.path == "/api/v11/daily":
            from quant_web.handlers.v11 import handler_v11_daily
            handler_v11_daily(parse_qs(parsed.query), self._send_json)
            return
        if parsed.path == "/api/v11/sources":
            from quant_web.handlers.v11 import handler_v11_sources
            handler_v11_sources(parse_qs(parsed.query), self._send_json)
            return
        if parsed.path == "/api/v11/search":
            from quant_web.handlers.v11 import handler_v11_search
            handler_v11_search(parse_qs(parsed.query), self._send_json)
            return

        if parsed.path == "/api/v11/battle":
            from quant_web.handlers.v11 import handler_v11_battle
            handler_v11_battle(parse_qs(parsed.query), self._send_json)
            return

        if parsed.path == "/api/v11/orders":
            from quant_web.handlers.v11 import handler_v11_orders
            handler_v11_orders(parse_qs(parsed.query), self._send_json)
            return

        if parsed.path == "/api/v11/health":
            from quant_web.handlers.v11 import handler_v11_health
            handler_v11_health(parse_qs(parsed.query), self._send_json)
            return
        if parsed.path == "/api/v12/kline":
            from quant_web.handlers.v11 import handler_v12_kline
            handler_v12_kline(parse_qs(parsed.query), self._send_json)
            return
        if parsed.path == "/api/v12/heatmap":
            from quant_web.handlers.v11 import handler_v12_heatmap
            handler_v12_heatmap(parse_qs(parsed.query), self._send_json)
            return
        if parsed.path == "/api/v12/market_treemap":
            from quant_web.handlers.v11 import handler_v12_market_treemap
            handler_v12_market_treemap(parse_qs(parsed.query), self._send_json)
            return
        if parsed.path == "/api/v12/fund_treemap":
            from quant_web.handlers.v11 import handler_v12_fund_treemap
            handler_v12_fund_treemap(parse_qs(parsed.query), self._send_json)
            return
        if parsed.path == "/api/monitor":
            from quant_web.handlers.intraday import handler_intraday
            handler_intraday(parse_qs(parsed.query), self._send_json)
            return
        if parsed.path == "/api/data_update_status":
            self._handle_data_update_status()
            return
        if parsed.path == "/api/market_pano":
            from quant_web.handlers.review import handler_market_pano
            handler_market_pano(parse_qs(parsed.query), self._send_json)
            return
        if parsed.path == "/api/social":
            from quant_web.handlers.review import handler_social
            handler_social(parse_qs(parsed.query), self._send_json)
            return
        if parsed.path == "/api/basepanorama":
            from quant_web.handlers.basepanorama import handler_basepanorama
            handler_basepanorama(parse_qs(parsed.query), self._send_json)
            return
        if parsed.path == "/base_panorama.html":
            _bp = STATIC_DIR / "base_panorama.html"
            if _bp.exists():
                _html = _bp.read_text(encoding="utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(_html.encode("utf-8"))
            else:
                self._send_json({"ok": False, "error": "panorama not found"}, 404)
            return
        if parsed.path == "/api/review/fusion":
            from quant_web.handlers.review import handler_review_fusion
            handler_review_fusion(parse_qs(parsed.query), self._send_json)
            return
        if parsed.path == "/api/review/md":
            from quant_web.handlers.review import handler_review_md
            handler_review_md(parse_qs(parsed.query), self._send_json)
            return
        if parsed.path == "/api/v12/fund_flow":
            from quant_web.handlers.v11 import handler_v12_fund_flow
            handler_v12_fund_flow(parse_qs(parsed.query), self._send_json)
            return
        if parsed.path == "/api/v12/etf_flow":
            from quant_web.handlers.v11 import handler_v12_etf_flow
            handler_v12_etf_flow(parse_qs(parsed.query), self._send_json)
            return
        if parsed.path == "/api/v12/stock_card":
            from quant_web.handlers.v11 import handler_v12_stock_card
            handler_v12_stock_card(parse_qs(parsed.query), self._send_json)
            return
        if parsed.path == "/api/strategy_compare":
            self._handle_strategy_compare(parse_qs(parsed.query))
            return
        if parsed.path == "/api/reports":
            self._handle_reports(parse_qs(parsed.query))
            return
        if parsed.path == "/api/research_fusion":
            self._handle_research_fusion(parse_qs(parsed.query))
            return
        if parsed.path == "/api/global_kline":
            self._handle_global_kline(parse_qs(parsed.query))
            return
        if parsed.path == "/api/market_history":
            self._handle_market_history(parse_qs(parsed.query))
            return
        if parsed.path == "/api/margin":
            self._handle_margin(parse_qs(parsed.query))
            return
        if parsed.path == "/api/market_temp":
            self._handle_market_temp(parse_qs(parsed.query))
            return
        if parsed.path == "/api/north_flow":
            self._handle_north_flow(parse_qs(parsed.query))
            return
        if parsed.path in ("/api/north_holdings", "/api/northbound_holdings"):
            self._handle_north_holdings(parse_qs(parsed.query))
            return
        if parsed.path == "/api/sentiment_market":
            self._handle_sentiment_market()
            return
        if parsed.path == "/api/fundamental":
            self._handle_fundamental(parse_qs(parsed.query))
            return
        if parsed.path == "/api/chart_patterns":
            self._handle_chart_patterns(parse_qs(parsed.query))
            return
        if parsed.path == "/api/ml_signal":
            self._handle_ml_signal(parse_qs(parsed.query))
            return
        if parsed.path == "/api/ml_signals":
            self._handle_ml_signals(parse_qs(parsed.query))
            return
        if parsed.path == "/api/sector_rotation":
            self._handle_sector_rotation(parse_qs(parsed.query))
            return
        if parsed.path == "/api/macro_calendar":
            self._handle_macro_calendar(parse_qs(parsed.query))
            return
        if parsed.path == "/api/earnings":
            self._handle_earnings(parse_qs(parsed.query))
            return
        if parsed.path == "/api/market_pulse":
            self._handle_market_pulse(parse_qs(parsed.query))
            return
        if parsed.path == "/api/opportunity":
            self._handle_opportunity(parse_qs(parsed.query))
            return
        if parsed.path == "/api/portfolio_risk":
            self._handle_portfolio_risk(parse_qs(parsed.query))
            return
        if parsed.path == "/api/system/status":
            cached = _cache_get("sys:status", 30)
            if cached is None:
                cached = _get_system_status()
                _cache_set("sys:status", cached)
            self._send_json(cached)
            return
        if parsed.path == "/api/indices":
            self._handle_indices(parse_qs(parsed.query))
            return
        if parsed.path == "/api/portfolio_optimizer":
            self._handle_portfolio_optimizer(parse_qs(parsed.query))
            return
        if parsed.path == "/api/market_attribution":
            self._handle_market_attribution(parse_qs(parsed.query))
            return
        if parsed.path == "/api/ml_model_info":
            self._handle_ml_model_info()
            return
        if parsed.path == "/api/ml_ensemble":
            self._handle_ml_ensemble(parse_qs(parsed.query))
            return
        if parsed.path in ("/api/cache_status", "/api/cache/status"):
            self._handle_cache_status()
            return
        if parsed.path == "/api/cache_refresh":
            self._handle_cache_refresh(parse_qs(parsed.query))
            return
        if parsed.path == "/api/v8_backtest":
            self._handle_v8_backtest(parse_qs(parsed.query))
            return
        if parsed.path == "/api/factor_report":
            self._handle_factor_report(parse_qs(parsed.query))
            return
        if parsed.path == "/api/financial_data":
            self._handle_financial_data(parse_qs(parsed.query))
            return
        if parsed.path == "/api/feature_store":
            self._handle_feature_store(parse_qs(parsed.query))
            return
        if parsed.path == "/api/feature_panel":
            self._handle_feature_panel(parse_qs(parsed.query))
            return
        if parsed.path == "/api/drift_check":
            self._handle_drift_check(parse_qs(parsed.query))
            return
        if parsed.path == "/api/asset_allocation":
            self._handle_asset_allocation(parse_qs(parsed.query))
            return
        if parsed.path == "/api/fama_macbeth":
            self._handle_fama_macbeth(parse_qs(parsed.query))
            return
        if parsed.path == "/api/risk_model_upgraded":
            self._handle_upgraded_risk_model(parse_qs(parsed.query))
            return
        # ── V3 兼容端点（保留历史因子/风控任务入口）──
        if parsed.path == "/api/v3/status":
            risk_handlers.handle_v3_status(self._send_json)
            return
        if parsed.path == "/api/v3/task" or parsed.path.startswith("/api/v3/task/"):
            task_query = parse_qs(parsed.query)
            if parsed.path.startswith("/api/v3/task/"):
                task_query.setdefault("id", [unquote(parsed.path.rsplit("/", 1)[-1])])
            risk_handlers.handle_task_result(task_query, self._send_json)
            return
        if parsed.path == "/api/v3/fama_macbeth":
            risk_handlers.handle_fama_macbeth(parse_qs(parsed.query), self._send_json)
            return
        if parsed.path == "/api/v3/risk_model":
            risk_handlers.handle_risk_model(parse_qs(parsed.query), self._send_json)
            return
        if parsed.path == "/api/v3/allocation":
            risk_handlers.handle_asset_allocation(parse_qs(parsed.query), self._send_json)
            return
        if parsed.path == "/api/v3/cache":
            risk_handlers.handle_cache_status(self._send_json)
            return
        # V5 compatibility adapter; new UI/source of truth remains version-neutral.
        if parsed.path.startswith("/api/v5/"):
            result = v5_route(parsed.path)
            if result is not None:
                self._send_json(result)
                return
        # ── 3.1 新端点 ──
        if parsed.path == "/api/events":
            self._handle_sse()
            return
        if parsed.path == "/api/tasks":
            self._handle_tasks(parse_qs(parsed.query))
            return
        if parsed.path == "/api/freshness":
            self._handle_freshness()
            return
        if parsed.path == "/api/backtest_almgren":
            self._handle_backtest_almgren(parse_qs(parsed.query))
            return
        # ── 模拟盘 ──
        if parsed.path == "/api/paper/positions":
            self._handle_paper_positions()
            return
        if parsed.path == "/api/paper/trades":
            self._handle_paper_trades(parse_qs(parsed.query))
            return
        if parsed.path == "/api/paper/confidence":
            self._handle_paper_confidence()
            return
        if parsed.path == "/api/paper/summary":
            self._handle_paper_summary()
            return
        if parsed.path == "/api/paper/orders":
            self._handle_paper_orders(parse_qs(parsed.query))
            return
        if parsed.path == "/api/paper/fills":
            self._handle_paper_fills(parse_qs(parsed.query))
            return
        if parsed.path == "/api/paper/reconciliation":
            self._handle_paper_reconciliation(parse_qs(parsed.query))
            return
        if parsed.path == "/api/resident/status":
            self._handle_resident_status()
            return
        if parsed.path == "/api/ops/overview":
            self._handle_ops_overview()
            return
        # ── V4 新端点 ──
        if parsed.path == "/api/v4/portfolio/optimize":
            self._handle_v4_portfolio_optimize(parse_qs(parsed.query))
            return
        if parsed.path == "/api/v4/risk/check":
            self._handle_v4_risk_check()
            return
        if parsed.path == "/api/v4/performance/summary":
            self._handle_v4_performance_summary()
            return
        if parsed.path == "/api/v4/performance/factor_attribution":
            self._handle_v4_factor_attribution()
            return
        if parsed.path == "/api/v4/regime":
            self._handle_v4_regime()
            return
        if parsed.path == "/api/v4/health":
            self._handle_v4_health()
            return
        if parsed.path == "/review.html":
            self._serve_review_dashboard()
            return
        if parsed.path == "/ops_console.html":
            page = STATIC_DIR / "ops_console.html"
            if not page.exists():
                self._send_json({"ok": False, "error": "运营控制台不存在"}, status=404)
                return
            body = page.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path in ("/v4.html", "/portfolio_dashboard.html"):
            # 新的中性实现作为主入口；原始 v4_dashboard.html 仍保留独立
            # 兼容地址，确保重构不会抹掉已验证的旧工作流。
            page = STATIC_DIR / "portfolio_dashboard.html"
            if page.exists():
                body = page.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._send_json({"ok": False, "error": "组合绩效深度页不存在"}, status=404)
            return
        if parsed.path == "/report.html":
            self._serve_html_report()
            return
        if parsed.path == "/research_dashboard.html":
            page = STATIC_DIR / "research_dashboard.html"
            if page.exists():
                body = page.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._send_json({"ok": False, "error": "独立深度页面不存在"}, status=404)
            return
        if parsed.path == "/research_report.html":
            page = STATIC_DIR / "research_report.html"
            if page.exists():
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(page.read_bytes())
            else:
                self._send_json({"ok": False, "error": "统一研报页面不存在"}, status=404)
            return
        if parsed.path == "/market_weekly.html":
            page = STATIC_DIR / "market_weekly.html"
            if page.exists():
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(page.read_bytes())
            else:
                self._send_json({"ok": False, "error": "周报页面不存在"}, status=404)
            return
        if parsed.path == "/api/research_images":
            self._handle_research_images(parse_qs(parsed.query))
            return
        if parsed.path.startswith("/research_image_asset/"):
            self._serve_research_image_asset(parsed.path)
            return
        if parsed.path.startswith("/report_assets/"):
            self._serve_report_asset(parsed.path)
            return
        # ── V9 数据融合 + AI 辅助决策 ──
        if parsed.path == "/api/stock_profile":
            self._handle_stock_profile(parse_qs(parsed.query))
            return
        if parsed.path == "/api/ai_decision":
            self._handle_ai_decision(parse_qs(parsed.query))
            return
        if parsed.path == "/api/warehouse_status":
            self._handle_warehouse_status()
            return
        if parsed.path == "/api/v1/health":
            self._handle_health()
            return
        if parsed.path == "/api/v1/audit":
            self._handle_audit()
            return
        if parsed.path == "/api/v1/segments":
            self._handle_segments()
            return
        if parsed.path == "/api/macro_snapshot":
            self._handle_macro_snapshot()
            return
        if parsed.path == "/api/factor_scan":
            self._handle_factor_scan(parse_qs(parsed.query))
            return
        if parsed.path == "/api/factor_library":
            self._handle_factor_library()
            return
        if parsed.path == "/api/factor_ic":
            self._handle_factor_ic(parse_qs(parsed.query))
            return
        if parsed.path == "/api/factor_combine":
            self._handle_factor_combine(parse_qs(parsed.query))
            return
        if parsed.path == "/api/ai_analyze":
            self._handle_ai_analyze(parse_qs(parsed.query))
            return
        # ── 监控新端点：告警中心 / 数据健康 / 定时任务状态 ──
        if parsed.path == "/api/alerts_status":
            self._handle_alerts_status()
            return
        if parsed.path == "/api/data_health":
            self._handle_data_health()
            return
        if parsed.path == "/api/tasks_status":
            self._handle_tasks_status()
            return
        # ── 决策链知识检索: RAG(books+skills) + 决策产物 ──
        if parsed.path == "/api/factor_quality":
            self._handle_factor_quality()
            return
        if parsed.path == "/api/industry_chain_history":
            from quant_web.handlers.v11 import handler_industry_chain_history
            handler_industry_chain_history(parse_qs(parsed.query), self._send_json)
            return
        if parsed.path == "/api/decision_chain":
            self._handle_decision_chain(parse_qs(parsed.query))
            return
        if parsed.path == "/api/execution_ledger":
            self._handle_execution_ledger()
            return
        if parsed.path == "/api/decision_snapshot":
            self._handle_decision_snapshot(parse_qs(parsed.query))
            return
        if parsed.path == "/api/auction_decision":
            self._handle_auction_decision(parse_qs(parsed.query))
            return
        if parsed.path == "/api/intraday_flow":
            self._handle_intraday_flow(parse_qs(parsed.query))
            return
        if parsed.path == "/api/intraday_flow_catalog":
            self._handle_intraday_flow_catalog()
            return
        if parsed.path == "/api/intraday_sector_flow":
            self._handle_intraday_sector_flow(parse_qs(parsed.query))
            return
        if parsed.path == "/api/rag_search":
            self._handle_rag_search(parse_qs(parsed.query))
            return
        if parsed.path == "/api/research_flow":
            self._handle_research_flow(parse_qs(parsed.query))
            return
        if parsed.path == "/api/stock_lens":
            self._handle_stock_lens(parse_qs(parsed.query))
            return
        # ── 静态页: index.html 版本占位符 {{VERSION}} 注入（版本与模块解耦:
        #    界面显示跟随 quant_system.__version__, 升级版本只改 __init__.py）──
        if parsed.path in ("/", "/index.html"):
            self._serve_index_html()
            return
        if parsed.path == "/strategy_lab.html":
            static_target = (STATIC_DIR / "strategy_lab.html").resolve()
            if static_target.is_file() and (STATIC_DIR.resolve() in static_target.parents):
                return super().do_GET()
            self._send_json({"ok": False, "error": "页面入口不存在", "path": parsed.path}, status=404)
            return
        if parsed.path == "/backtest_console.html":
            static_target = (STATIC_DIR / "backtest_console.html").resolve()
            if static_target.is_file() and (STATIC_DIR.resolve() in static_target.parents):
                return super().do_GET()
            self._send_json({"ok": False, "error": "页面入口不存在", "path": parsed.path}, status=404)
            return
        # 合法静态页仍由 SimpleHTTPRequestHandler 服务；未知 HTML 必须明确 404，
        # 避免把入口错误伪装成首页而掩盖功能移除。
        if parsed.path.endswith('.html') and not parsed.path.startswith('/report_assets/'):
            static_target = (STATIC_DIR / parsed.path.lstrip('/')).resolve()
            if static_target.is_file() and (STATIC_DIR.resolve() in static_target.parents):
                return super().do_GET()
            self._send_json({"ok": False, "error": "页面入口不存在", "path": parsed.path}, status=404)
            return
        return super().do_GET()

    def _serve_ops_console(self) -> None:
        """Default root: the actionable operations console, not the legacy dashboard."""
        p = STATIC_DIR / "ops_console.html"
        if not p.exists():
            self._send_json({"ok": False, "error": "页面入口不存在", "path": "/ops_console.html"}, status=404)
            return
        body = p.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_index_html(self) -> None:
        """渲染兼容旧首页并注入当前系统版本号。"""
        try:
            from quant_system import __version__ as _cur_version
        except Exception:  # noqa: BLE001 - 版本读取失败降级显示常量
            _cur_version = "12.3.0"
        p = STATIC_DIR / "index.html"
        if not p.exists():
            return super().do_GET()
        html = p.read_text(encoding="utf-8").replace("{{VERSION}}", str(_cur_version))
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_backtest_dashboard(self, query: dict[str, list[str]]) -> None:
        """GET /api/backtest_dashboard - return backtest summary + chart data."""
        source = "unavailable"
        strategy_names = []
        try:
            from quant_system.combined_backtest import INITIAL_CAPITAL
            from quant_system.signal_backtest import _SIGNAL_DEFS

            source = f"signal_backtest+combined_backtest initial={INITIAL_CAPITAL}"
            strategy_names = [v.get("name", k) for k, v in list(_SIGNAL_DEFS.items())[:4]]
        except Exception as e:
            logging.getLogger(__name__).error(f"[server] 操作失败: {e}", exc_info=True)

        latest_path = ROOT / "generated" / "backtest_latest.json"
        factor_bt = {}
        factor_path = ROOT / "generated" / "factor_backtest_latest.json"
        if factor_path.exists():
            try:
                factor_bt = json.loads(factor_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logging.getLogger(__name__).warning("factor backtest read failed: %s", exc)
        multi_asset = {}
        multi_path = ROOT / "generated" / "multi_asset_backtest_latest.json"
        if multi_path.exists():
            try:
                multi_asset = json.loads(multi_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logging.getLogger(__name__).warning("multi asset backtest read failed: %s", exc)
        if latest_path.exists():
            try:
                payload = json.loads(latest_path.read_text(encoding="utf-8"))
                rows = payload.get("rows") or []
                strategies = [{"name": f"{r.get('symbol','-')} 最近真实回测", "total_return": r.get("total_return_pct"),
                               "annual_return": r.get("annual_return_pct"), "max_dd": r.get("max_drawdown_pct"),
                               "sharpe": r.get("sharpe"), "win_rate": r.get("win_rate_pct"),
                               "trades": r.get("trade_count"), "validation": r.get("validation")}
                              for r in rows]
                self._send_json({"ok": True, "data": {"source": "generated/backtest_latest.json", "demo": False,
                    "available": bool(strategies), "provenance": payload.get("provenance"), "strategies": strategies,
                    "factor_backtest": factor_bt,
                    "multi_asset": multi_asset,
                    "comparison": {"labels": [x["name"] for x in strategies],
                                   "returns": [x.get("total_return") for x in strategies],
                                   "drawdowns": [x.get("max_dd") for x in strategies]}}})
                return
            except Exception as exc:
                logging.getLogger(__name__).warning("latest backtest read failed: %s", exc)
        # 当前接口还没有把具体参数传入真实回测引擎；在接线完成前禁止占位收益。
        self._send_json({"ok": True, "data": {
            "source": "unavailable",
            "demo": False,
            "available": False,
            "error": "策略中心暂不展示模拟收益：请从真实回测入口指定标的、日期、策略和交易成本",
            "strategies": [], "comparison": {"labels": [], "returns": [], "drawdowns": []},
        }})
        return
        # 没有真实回测产物时明确返回不可用，禁止用占位收益冒充验证结果。
        if not strategy_names:
            self._send_json({"ok": True, "data": {
                "source": source,
                "demo": False,
                "available": False,
                "error": "真实回测尚未完成：请指定标的、日期、策略和成本后运行回测",
                "strategies": [], "comparison": {"labels": [], "returns": [], "drawdowns": []},
            }})
            return
        base = []
        dd = []
        strategies = []
        for i, name in enumerate(strategy_names):
            strategies.append({
                "name": name,
                "total_return": round(base[i % len(base)], 2),
                "max_dd": round(dd[i % len(dd)], 2),
                "sharpe": round(0.82 + i * 0.18, 2),
                "win_rate": round(52 + i * 4.5, 2),
                "trades": 18 + i * 7,
            })
        self._send_json({
            "ok": True,
            "data": {
                "source": source,
                "demo": True,  # V11.1: 演示数据透明化（策略收益/回撤为占位值，未接真实回测）
                "as_of": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "strategies": strategies,
                "comparison": {
                    "labels": [s["name"] for s in strategies],
                    "returns": [s["total_return"] for s in strategies],
                    "drawdowns": [s["max_dd"] for s in strategies],
                },
            },
        })

    def _handle_factor_analysis(self, query: dict[str, list[str]]) -> None:
        """GET /api/factor_analysis - only real factor artifacts are eligible."""
        report = ROOT / "generated" / "ic_report" / "FACTOR_IC_REPORT_VECTORIZED.csv"
        # 此旧接口没有接入完整真实股票截面，禁止使用硬编码样本；请使用 factor_report。
        self._send_json({"ok": True, "data": {"available": False, "demo": False, "source": "deprecated",
            "error": "factor_analysis旧接口已禁用硬编码样本，请使用真实factor_report接口", "factors": [], "stocks": []}})
        return
        if not report.exists():
            self._send_json({"ok": True, "data": {"available": False, "demo": False, "source": "unavailable",
                "error": "真实因子IC产物不存在，禁止返回占位因子结果", "factors": [], "stocks": []}})
            return
        source = str(report)
        raw = pd.DataFrame([
            {"code": "002714", "name": "牧原股份", "pe": 18.2, "pb": 3.1, "mkt_cap": 2300},
            {"code": "600519", "name": "贵州茅台", "pe": 24.8, "pb": 8.5, "mkt_cap": 19000},
            {"code": "300750", "name": "宁德时代", "pe": 22.4, "pb": 4.6, "mkt_cap": 9200},
            {"code": "601318", "name": "中国平安", "pe": 9.7, "pb": 0.9, "mkt_cap": 7900},
            {"code": "000333", "name": "美的集团", "pe": 13.5, "pb": 2.7, "mkt_cap": 5200},
        ])
        try:
            from quant_system.cross_section import compute_cross_sectional_factors

            ranked = compute_cross_sectional_factors(raw, pct_ranking=True)
            source = "cross_section.compute_cross_sectional_factors"
        except Exception as exc:
            self._send_json({"ok": True, "data": {"available": False, "demo": False, "source": source,
                "error": f"真实因子计算失败，禁止回退占位结果：{str(exc)[:160]}", "factors": [], "stocks": []}})
            return

        stocks = []
        for i, row in ranked.iterrows():
            pe_score = 100 - float(row.get("pe_pct", 50) or 50)
            pb_score = 100 - float(row.get("pb_pct", 50) or 50)
            size_score = float(row.get("mkt_cap_pct", 50) or 50)
            score = round(pe_score * 0.35 + pb_score * 0.25 + size_score * 0.40, 2)
            stocks.append({
                "rank": 0,
                "symbol": row.get("code", ""),
                "name": row.get("name", ""),
                "score": score,
                "factor_breakdown": {
                    "value": round((pe_score + pb_score) / 2, 2),
                    "quality": round(55 + (i * 7) % 35, 2),
                    "momentum": round(48 + (i * 11) % 42, 2),
                    "size": round(size_score, 2),
                },
            })
        stocks.sort(key=lambda x: x["score"], reverse=True)
        for i, stock in enumerate(stocks, 1):
            stock["rank"] = i

        self._send_json({
            "ok": True,
            "data": {
                "source": source,
                "demo": True,  # V11.1: 演示数据透明化（因子 IC 为占位值）
                "factors": [
                    {"name": "价值", "ic": 0.052, "return": 3.8, "volatility": 8.6},
                    {"name": "质量", "ic": 0.041, "return": 2.9, "volatility": 7.4},
                    {"name": "动量", "ic": 0.063, "return": 4.6, "volatility": 11.2},
                    {"name": "规模", "ic": -0.018, "return": -0.7, "volatility": 6.9},
                ],
                "stocks": stocks,
            },
        })

    def _handle_news_sentiment_market(self, query: dict[str, list[str]]) -> None:
        """GET /api/v1/news_sentiment_market — 市场级公告情绪（V11 总纲#9）。"""
        try:
            from quant_platform.openclaw_api import news_sentiment_api
            days = int(_first(query, "days", "10"))
            self._send_json({"ok": True, "data": news_sentiment_api(days=days)})
        except Exception as e:
            self._send_json({"ok": False, "error": str(e)})


    def _handle_fusion_api(self, api_name: str, query: dict[str, list[str]]) -> None:
        """V11 融合 API 通用 handler（market_breadth/market_regime/seasonality/limit_up_depth/broker_profiles）。"""
        try:
            from quant_platform import openclaw_api as _api
            fn = getattr(_api, api_name)

            def _compute():
                if api_name == "broker_profiles":
                    top_n = int(_first(query, "top_n", "20"))
                    return fn(top_n=top_n)
                if api_name == "market_regime_api":
                    code = _first(query, "code", "000300")
                    return fn(code=code)
                return fn()

            _async_deep(f"deep:fusion:{api_name}:{json.dumps(query, sort_keys=True)}", 600,
                        lambda: {"ok": True, "data": _compute()}, self._send_json)
        except Exception as e:
            self._send_json({"ok": False, "error": str(e)})

    def _handle_news_sentiment(self, query: dict[str, list[str]]) -> None:
        """GET /api/news_sentiment?symbol=002714 — strictly attributed stock news."""
        try:
            from quant_system.analysis_core.social_sentiment import fetch_news, sentiment_score

            symbol = normalize_symbol(resolve_symbol(_first(query, "symbol", "002714")))
            if not re.fullmatch(r"\d{6}", symbol):
                self._send_json({"ok": False, "error": "symbol 必须是可识别的6位股票代码"}, status=400)
                return
            limit = max(1, min(int(_first(query, "limit", "30")), 100))
            news = fetch_news(symbol=symbol, limit=limit)
            if not news.get("ok"):
                status = 404 if "暂无" in str(news.get("error") or "") else 502
                self._send_json({
                    "ok": False, "error": news.get("error") or "个股新闻数据不可用",
                    "symbol": symbol, "coverage": "stock_symbol", "source_symbol": symbol,
                    "detail": news,
                }, status=status)
                return

            if news.get("coverage") != "stock_symbol" or news.get("source_symbol") != symbol:
                self._send_json({
                    "ok": False, "error": "新闻源未证明按请求股票代码过滤，已拒绝个股归因",
                    "symbol": symbol, "coverage": news.get("coverage"),
                    "source_symbol": news.get("source_symbol"),
                }, status=502)
                return

            score = round(float(news.get("sentiment") or 0.0), 4)
            sentiment = "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"
            headlines = []
            for row in news.get("records") or []:
                title = str(row.get("title") or "").strip()
                if not title:
                    continue
                item_score = sentiment_score(title)
                headlines.append({
                    "title": title, "date": row.get("date") or None,
                    "source": row.get("source") or news.get("source"),
                    "url": row.get("url") or None,
                    "sentiment": "positive" if item_score > 0 else "negative" if item_score < 0 else "neutral",
                    "score": item_score,
                })
            self._send_json({"ok": True, "data": _json_safe({
                "symbol": symbol, "source_symbol": symbol,
                "coverage": "stock_symbol",
                "coverage_note": f"仅统计数据源明确归属于 {symbol} 的个股新闻，不混入市场聚合新闻",
                "sentiment": sentiment, "score": score, "total": int(news.get("n") or len(headlines)),
                "headlines": headlines,
                "bull": int(news.get("bull") or 0), "bear": int(news.get("bear") or 0),
                "neutral": int(news.get("neutral") or 0),
                "date": datetime.now().strftime("%Y-%m-%d"),
                "signals": [
                    {"name": "偏多新闻", "value": int(news.get("bull") or 0), "view": f"{int(news.get('bull') or 0)} 条"},
                    {"name": "偏空新闻", "value": int(news.get("bear") or 0), "view": f"{int(news.get('bear') or 0)} 条"},
                    {"name": "中性新闻", "value": int(news.get("neutral") or 0), "view": f"{int(news.get('neutral') or 0)} 条"},
                ],
                "source": news.get("source", "akshare.stock_news_em"),
            })})
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            self._send_json({"ok": False, "error": repr(exc), "trace": traceback.format_exc(limit=6)}, status=500)

    def _handle_research_fusion(self, query: dict[str, list[str]]) -> None:
        """Return a local research-fusion snapshot without network refreshes."""
        raw_date = _first(query, "date", "").strip()
        refresh = _first(query, "refresh", "false").strip().lower() == "true"
        if raw_date and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw_date):
            self._send_json({"ok": False, "error": "date 必须为 YYYY-MM-DD"}, status=400)
            return
        if raw_date:
            try:
                datetime.strptime(raw_date, "%Y-%m-%d")
            except ValueError:
                self._send_json({"ok": False, "error": "date 不是有效日期"}, status=400)
                return

        generated = (ROOT / "generated").resolve()
        snapshot_path = (generated / f"research_fusion_snapshot_{raw_date}.json").resolve() if raw_date else None
        if snapshot_path is not None and snapshot_path.parent != generated:
            self._send_json({"ok": False, "error": "不安全的快照路径"}, status=400)
            return

        try:
            if raw_date and snapshot_path is not None and not snapshot_path.is_file():
                # A historical request must not silently become the nearest older
                # snapshot. The caller can choose an available date explicitly.
                self._send_json({"ok": False, "data_status": "missing", "requested_date": raw_date,
                                 "date_mismatch": False,
                                 "error": f"没有 {raw_date} 的研究融合快照"}, status=404)
                return
            if not refresh:
                candidate = snapshot_path
                if candidate is None:
                    snapshots = sorted(generated.glob("research_fusion_snapshot_????-??-??.json"))
                    candidate = snapshots[-1].resolve() if snapshots else None
                if candidate is not None and candidate.parent == generated and candidate.is_file():
                    payload = json.loads(candidate.read_text(encoding="utf-8"))
                    if not isinstance(payload, dict):
                        raise ValueError("快照根节点不是 JSON 对象")
                    payload.setdefault("schema_version", "research-fusion.v1")
                    payload.setdefault("scope", "analysis_decision_assist_only")
                    payload.setdefault("scope_detail", {"analysis": "全市场资金、行业、概念与ETF证据汇总", "execution": "仅作分析与决策辅助，不连接券商执行"})
                    payload.setdefault("requested_date", raw_date or payload.get("requested_date"))
                    payload.setdefault("date_mismatch", bool(raw_date and payload.get("as_of") != raw_date))
                    self._send_json(payload)
                    return

            from scripts.research_fusion_snapshot import build_snapshot
            payload = build_snapshot(date=raw_date or None, out_path=snapshot_path)
            payload["requested_date"] = raw_date or payload.get("requested_date")
            payload["date_mismatch"] = bool(raw_date and payload.get("as_of") != raw_date)
            self._send_json(payload)
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)
        except (OSError, json.JSONDecodeError) as exc:
            self._send_json({"ok": False, "error": f"快照读取失败: {exc}"}, status=500)
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).exception("research fusion snapshot failed")
            self._send_json({"ok": False, "error": f"快照生成失败: {str(exc)[:300]}"}, status=500)

    def _handle_strategy_compare(self, query: dict[str, list[str]]) -> None:
        """策略中心只返回真实回测结果；没有参数/产物时明确不可用。"""
        self._send_json({"ok": True, "data": {
            "available": False, "demo": False, "source": "unavailable",
            "error": "策略比较尚未接入真实回测产物，请从回测入口指定标的、日期、策略和成本",
            "strategies": [], "comparison": {"labels": [], "returns": [], "drawdowns": []}
        }})

    def do_POST(self) -> None:
        if self._reject_unauthorized():
            return
        parsed = urlparse(self.path)
        routes = {
            "/api/scan": self._handle_scan,
            "/api/backtest": self._handle_backtest,
            "/api/chat": self._handle_chat,
            "/api/alert": self._handle_alert,
            "/api/alert/push": self._handle_alert_push,
            "/api/portfolio": self._handle_portfolio_post,
            "/api/watchlist": self._handle_watchlist_save,
            "/api/optimize": self._handle_optimize,
        "/api/ml_retrain": self._handle_ml_retrain,
            "/api/paper/preview": self._handle_paper_preview,
            "/api/paper/buy": self._handle_paper_buy,
            "/api/paper/sell": self._handle_paper_sell,
            "/api/paper/order": self._handle_paper_order,
            "/api/paper/fill": self._handle_paper_fill,
            "/api/v11/feedback": self._handle_v11_feedback,
            "/api/strategy-lab/save": self._handle_strategy_lab_save,
            "/api/strategy-lab/run": self._handle_strategy_lab_run,
        }
        handler = routes.get(parsed.path)
        if not handler:
            self.send_error(404, "Not found")
            return
        try:
            handler(self._read_json())
        except Exception as exc:
            self._send_json({"ok": False, "error": repr(exc), "trace": traceback.format_exc(limit=6)}, status=500)

    def _handle_strategy_lab_save(self, body: dict) -> None:
        from quant_web.handlers import strategy_lab
        try:
            strategy_lab.handler_save(body, self._send_json)
        except Exception as exc:
            self._send_json({"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:220]}"}, status=400)

    def _handle_strategy_lab_run(self, body: dict) -> None:
        from quant_web.handlers import strategy_lab
        try:
            strategy_lab.handler_run(body, self._send_json)
        except Exception as exc:
            self._send_json({"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:220]}"}, status=400)

    def _handle_v11_feedback(self, body: dict) -> None:
        """POST /api/v11/feedback — 用户对作战地图的反馈（在线权重修正输入）。
        body: {date, vote(like/dislike), direction(方向名,如PCB), note?}
        """
        from pathlib import Path as _P
        fb_file = _P(__file__).resolve().parents[1] / "generated" / "feedback.jsonl"
        date = str(body.get("date") or datetime.now().strftime("%Y-%m-%d"))
        vote = str(body.get("vote") or "")
        direction = str(body.get("direction") or "")
        if vote not in ("like", "dislike"):
            self._send_json({"ok": False, "error": "vote 必须是 like/dislike"}, status=400)
            return
        rec = {"ts": datetime.now().isoformat(), "date": date, "vote": vote,
               "direction": direction, "note": str(body.get("note") or "")}
        fb_file.parent.mkdir(parents=True, exist_ok=True)
        with open(fb_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._send_json({"ok": True, "saved": rec})

    def _handle_workbench(self, query: dict[str, list[str]]) -> None:
        """GET /api/workbench — compact canonical decision workspace payload.

        The browser must not download the full 4MB+ candidate snapshot for the
        landing page.  This endpoint is read-only, cached, and deliberately
        projects the same canonical snapshot into a small decision rail.
        """
        mode = _first(query, "mode", "after_close")
        requested = _first(query, "date", "").strip() or None
        if mode not in {"intraday", "after_close", "battle_map"}:
            self._send_json({"ok": False, "error": "mode 必须为 intraday|after_close|battle_map"}, status=400)
            return
        if requested and not re.fullmatch(r"20\d{2}-\d{2}-\d{2}", requested):
            self._send_json({"ok": False, "error": "date 必须为 YYYY-MM-DD"}, status=400)
            return
        key = f"workbench:v2:{mode}:{requested or 'latest'}"
        cached = _cache_get(key, 45 if mode == "intraday" else 180)
        if cached is not None:
            self._send_json(cached)
            return
        try:
            from quant_web.workbench import build_workbench
            payload = build_workbench(mode=mode, date=requested)
            _cache_set(key, payload)
            self._send_json(payload)
        except Exception as exc:
            logging.getLogger(__name__).exception("workbench projection failed")
            self._send_json({"ok": False, "error": f"统一工作台暂不可用：{str(exc)[:220]}"}, status=503)

    def _handle_market_recap(self, query: dict[str, list[str]]) -> None:
        """Return one auditable market-review view model for the dashboard.

        The daily review artifacts already contain the market breadth, emotion,
        mainlines and candidate gates. This adapter keeps the browser on one
        stable contract and explicitly marks the artifact date when it lags the
        requested date.
        """
        requested = _first(query, "date", "").strip()
        if requested and not re.fullmatch(r"20\d{2}-\d{2}-\d{2}", requested):
            self._send_json({"ok": False, "error": "date 必须为 YYYY-MM-DD"}, status=400)
            return
        # One decision contract, selected by the current market phase. Historical
        # requests remain explicit and always use the requested after-close date.
        refresh = _first(query, "refresh", "false").lower() == "true"
        cache_key = f"market_recap:{requested or 'latest'}"
        if not refresh:
            cached = _cache_get(cache_key, 300)
            if cached is not None:
                self._send_json(cached)
                return
        now = datetime.now()
        minute = now.hour * 60 + now.minute
        live_phase = "盘中快照" if not requested and 9 * 60 + 15 <= minute < 15 * 60 else "盘后快照"
        path = None
        snapshot = None
        # A replay request and the after-close dashboard must consume the
        # matching persisted artifact. Rebuilding from today's live warehouse
        # here can silently change the historical seal-rate/date contract.
        if live_phase == "盘后快照":
            files = sorted((ROOT / "generated").glob("decision_snapshot_after_close_????-??-??.json"), reverse=True)
            if requested:
                exact = ROOT / "generated" / f"decision_snapshot_after_close_{requested}.json"
                files = ([exact] if exact.exists() else [])
            if files:
                path = files[0]
        if path is None:
            try:
                import sys as _sys
                _sys.path.insert(0, str(ROOT / "scripts"))
                from unified_decision_snapshot import build_snapshot
                snapshot = build_snapshot(mode="intraday" if live_phase == "盘中快照" else "after_close", date=requested or None)
            except Exception:
                snapshot = None
        if snapshot is None and path is None:
            self._send_json({"ok": False, "data_status": "missing", "error": "暂无可用盘面复盘快照"}, status=404)
            return
        try:
            if snapshot is None:
                snapshot = json.loads(path.read_text(encoding="utf-8"))
            market = snapshot.get("market") or {}
            zt = market.get("zt") or {}
            ladder = zt.get("ladder_json") or "{}"
            try:
                ladder = json.loads(ladder) if isinstance(ladder, str) else ladder
            except (TypeError, ValueError):
                ladder = {}
            mainlines = [item for item in (snapshot.get("mainlines") or [])
                         if str(item.get("taxonomy") or "").lower() in {"concept", "theme"}
                         or (not item.get("taxonomy") and not item.get("industry"))]
            mainlines = sorted(mainlines, key=lambda x: float(x.get("score") or 0), reverse=True)
            candidates = snapshot.get("opportunities") or snapshot.get("decision_rows") or snapshot.get("short_term_candidates") or []
            sources = snapshot.get("source_chain") or {}
            payload = {
                "ok": True,
                "data_status": "available",
                "requested_date": requested or None,
                "as_of": snapshot.get("as_of") or snapshot.get("requested_date"),
                "generated_at": snapshot.get("generated_at"),
                "phase": live_phase,
                "snapshot_label": "统一决策快照",
                "scope": snapshot.get("scope") or "analysis_decision_assist_only",
                "scope_detail": snapshot.get("scope_detail") or {"analysis": "全市场", "execution": "仅作分析与决策辅助，不连接券商执行"},
                "date_mismatch": bool(requested and (snapshot.get("as_of") or "") != requested),
                "market": {
                    "temperature": market.get("temperature"),
                    "emotion_stage": market.get("emotion_stage"),
                    "force_index": market.get("force_index"),
                    "breadth": market.get("breadth"),
                    "risk_flags": market.get("risk_flags") or [],
                    "limit_up": zt.get("zt_cnt"),
                    "limit_down": zt.get("dt_cnt"),
                    "broken_board": zt.get("zb_cnt"),
                    "max_board": zt.get("max_board"),
                    "seal_rate": round(100 * float(zt.get("zt_cnt") or 0) / max(float(zt.get("zt_cnt") or 0) + float(zt.get("zb_cnt") or 0), 1), 1),
                    "ladder": ladder,
                },
                "mainlines": mainlines[:12],
                "money_flow": snapshot.get("money_flow") or {},
                "sector_rotation": snapshot.get("sector_rotation") or {},
                "etf_activity": snapshot.get("etf_activity") or {},
                "candidates": candidates[:20],
                "evidence": snapshot.get("evidence") or {},
                "decision_basis": snapshot.get("decision_basis") or {},
                "source_chain": sources,
                "indices": (_cache_get("indices", 3600) or {}).get("data", {}).get("ashare", {}),
                "methodology": "盘面复盘=指数/宽度/涨停梯队/主线扩散/资金与候选门控；结构、资金和日期不一致时仅展示为观察证据，不自动放行交易。",
            }
            payload = _json_safe(payload)
            _cache_set(cache_key, payload)
            self._send_json(payload)
        except (OSError, json.JSONDecodeError) as exc:
            self._send_json({"ok": False, "data_status": "error", "error": f"盘面复盘读取失败：{str(exc)[:160]}"}, status=500)

    def _handle_stock_overview(self, query: dict[str, list[str]]) -> None:
        """One-call stock workspace: profile, decision context, flow series and research evidence."""
        raw_symbol = _first(query, "symbol", "").strip()
        if not raw_symbol:
            self._send_json({"ok": False, "error": "请输入股票代码或名称"}, status=400)
            return
        symbol = normalize_symbol(resolve_symbol(raw_symbol))
        if not re.fullmatch(r"\d{6}", symbol):
            self._send_json({"ok": False, "error": f"无法识别股票：{raw_symbol}"}, status=400)
            return

        source_status = {
            "profile": {"status": "missing", "label": "个股画像"},
            "decision": {"status": "missing", "label": "决策快照"},
            "flow": {"status": "missing", "label": "行业资金代理"},
            "stock_flow": {"status": "missing", "label": "个股主力资金"},
            "research": {"status": "missing", "label": "研报提取"},
            "news": {"status": "missing", "label": "个股新闻"},
        }
        profile: dict = {}
        decision: dict = {}
        flow_payload: dict = {"rows": [], "metric": "active_flow_proxy_yi", "status": "missing"}
        stock_flow_payload: dict = {"rows": [], "metric": "main_net_yi", "metric_label": "个股主力净流入(亿元)", "status": "missing", "is_real_main_net": True}
        research_items: list[dict] = []
        research_index_hits = 0
        commodity_evidence: dict = {}
        stock_news_payload: dict = {
            "status": "missing", "symbol": symbol, "source_symbol": symbol,
            "coverage": "stock_symbol", "items": [], "total": 0,
            "attribution_method": "provider_symbol_query",
            "decision_usage": "event_review_only",
        }

        try:
            profile = build_stock_profile(symbol)
            quote_date = str((profile.get("quote") or {}).get("date") or "")
            source_status["profile"] = {
                "status": "available" if profile.get("quote") else "partial",
                "label": "个股画像", "as_of": quote_date or None,
                "message": "行情可用" if profile.get("quote") else "本地仓库缺少该股K线",
            }
        except Exception as exc:  # noqa: BLE001
            source_status["profile"]["message"] = f"画像加载失败：{str(exc)[:100]}"

        decision_paths = sorted((ROOT / "generated").glob("decision_snapshot_*.json"), reverse=True)
        for path in decision_paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                candidate_rows = payload.get("opportunities") or payload.get("short_term_candidates") or payload.get("candidates") or []
                match = next((row for row in candidate_rows
                              if str(row.get("code") or row.get("symbol") or "").zfill(6) == symbol), None)
                if match:
                    decision = match
                    source_status["decision"] = {"status": "available", "label": "决策快照",
                                                   "as_of": payload.get("as_of"), "message": match.get("decision")}
                    break
            except Exception:
                continue
        if not decision:
            source_status["decision"]["message"] = "当前候选池未覆盖该股，不等于看空"

        # 行业资金不能依赖候选池是否填充 industry；优先候选快照，
        # 再回退到申万一级静态映射，最后才查其他快照的代码级行业字段。
        industry = str(decision.get("industry") or decision.get("sector") or "").strip()
        industry_source = "decision_snapshot" if industry else ""
        if not industry:
            try:
                map_path = ROOT / "data_warehouse" / "market" / "sw_industry_map.parquet"
                if map_path.exists():
                    map_df = pd.read_parquet(map_path, columns=["code", "industry", "industry_code"])
                    map_df["code"] = map_df["code"].astype(str).str.zfill(6)
                    match = map_df[map_df["code"] == symbol]
                    if not match.empty:
                        industry = str(match.iloc[-1].get("industry") or "").strip()
                        industry_source = "sw_industry_map"
            except Exception as exc:
                source_status["flow"]["mapping_error"] = str(exc)[:100]
        if not industry:
            for snapshot_path in sorted((ROOT / "generated").glob("decision_snapshot_*.json"), reverse=True):
                try:
                    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
                    candidate = next((row for row in snapshot.get("opportunities", [])
                                      if str(row.get("code") or row.get("symbol") or "").zfill(6) == symbol
                                      and str(row.get("industry") or row.get("sector") or "").strip()), None)
                    if candidate:
                        industry = str(candidate.get("industry") or candidate.get("sector") or "").strip()
                        industry_source = "decision_snapshot_fallback"
                        break
                except Exception:
                    continue
        flow_paths = sorted((ROOT / "generated").glob("intraday_flow_history_????-??-??.json"), reverse=True)
        if flow_paths and industry:
            try:
                selected_payload, rows = None, []
                # 最新产物可能没有完整行业快照；向前回退到最近一次真正命中，
                # 同时把实际数据日返回给前端，避免空图覆盖有效历史证据。
                for flow_path in flow_paths[:20]:
                    candidate = json.loads(flow_path.read_text(encoding="utf-8"))
                    candidate_rows = [row for row in candidate.get("industry", []) if str(row.get("name") or "") == industry]
                    if candidate_rows:
                        selected_payload, rows = candidate, candidate_rows
                        break
                if selected_payload is not None:
                    flow_payload = {
                        "status": "available", "rows": rows, "industry": industry, "industry_source": industry_source, "date": selected_payload.get("date"),
                        "generated_at": selected_payload.get("generated_at"), "complete_day": bool(selected_payload.get("complete_day")),
                        "metric": "active_flow_proxy_yi", "metric_label": "主动资金代理(亿元)",
                        "methodology": (selected_payload.get("methodology") or {}).get("active_flow_proxy_yi"),
                        "is_real_main_net": False,
                    }
                    source_status["flow"] = {
                        "status": "available", "label": "资金分时",
                        "as_of": selected_payload.get("date"),
                        "message": f"{industry}行业资金代理，非主力净流入（映射源：{industry_source}）",
                    }
                else:
                    source_status["flow"]["message"] = f"最近20期无{industry}行业分时记录"
            except Exception as exc:  # noqa: BLE001
                source_status["flow"]["message"] = f"资金分时读取失败：{str(exc)[:100]}"
        else:
            latest_flow_date = None
            if flow_paths:
                try:
                    latest_flow_date = json.loads(flow_paths[0].read_text(encoding="utf-8")).get("date")
                except Exception:
                    pass
            flow_payload.update({"status": "mapping_missing", "date": latest_flow_date,
                                 "industry_source": None, "message": "行业资金产物存在但缺少该股申万行业映射"})
            source_status["flow"] = {"status": "mapping_missing", "label": "行业资金代理",
                                      "as_of": latest_flow_date,
                                      "message": "行业资金产物存在但缺少该股申万行业映射"}

        # 行业商品/期货证据：与申万行业映射独立，缺失品种不补零。
        try:
            from quant_web.industry_commodity import build_industry_commodity_evidence
            commodity_evidence = build_industry_commodity_evidence(symbol)
            source_status["commodity"] = {
                "status": commodity_evidence.get("status", "missing"), "label": "行业商品/期货",
                "as_of": max((str(item.get("as_of")) for item in commodity_evidence.get("items", []) if item.get("as_of")), default=None),
                "message": commodity_evidence.get("profile_reason") or commodity_evidence.get("methodology") or "商品映射不可用",
            }
        except Exception as exc:
            commodity_evidence = {"status": "error", "items": [], "error": str(exc)[:160]}
            source_status["commodity"] = {"status": "error", "label": "行业商品/期货", "message": commodity_evidence["error"]}

        # 个股级主力资金：独立于行业成交额方向代理，只有真实 stock_money_flow
        # 记录才可进入个股证据；远端不可用时显式返回 missing。
        try:
            from quant_system.data_pipeline import get_stock_money_flow_with_status
            money_frame, money_status = get_stock_money_flow_with_status(symbol, days=60)
            source_status["stock_flow"].update(money_status)
            if money_frame is not None and not money_frame.empty:
                rows = money_frame.copy()
                rows["date"] = rows["date"].astype(str).str[:10]
                rows["main_net_yi"] = pd.to_numeric(rows.get("main_net"), errors="coerce") / 1e8
                if not rows["main_net_yi"].notna().any():
                    raise ValueError("个股资金帧存在日期但主力净流入全为空")
                rows["super_large_net_yi"] = pd.to_numeric(rows.get("super_large_net"), errors="coerce") / 1e8
                rows["large_net_yi"] = pd.to_numeric(rows.get("large_net"), errors="coerce") / 1e8
                rows["retail_net_yi"] = pd.to_numeric(rows.get("retail_net"), errors="coerce") / 1e8
                keep = [c for c in ("symbol", "date", "main_net_yi", "super_large_net_yi", "large_net_yi", "retail_net_yi") if c in rows.columns]
                safe_rows = rows[keep].where(pd.notna(rows[keep]), None).to_dict("records")
                stock_flow_payload = {
                    "status": "available", "rows": safe_rows[-60:], "symbol": symbol,
                    "as_of": str(rows["date"].iloc[-1]) if len(rows) else None,
                    "metric": "main_net_yi", "metric_label": "个股主力净流入(亿元)",
                    "source": money_status.get("source", "akshare.stock_individual_fund_flow"),
                    "is_real_main_net": True,
                    "methodology": "东方财富个股资金流接口，主力净流入原始单位元，展示换算为亿元",
                }
                source_status["stock_flow"].update({
                    "status": "available", "label": "个股主力资金",
                    "as_of": stock_flow_payload["as_of"],
                    "message": f"{symbol} 个股主力净流入，共{len(safe_rows)}日",
                })
            else:
                source_status["stock_flow"].update({
                    "status": money_status.get("status", "provider_empty"),
                    "message": money_status.get("message") or f"{symbol} 无个股主力资金记录",
                    "source": money_status.get("source", "quant_system.money_flow"),
                })
                stock_flow_payload.update({
                    "status": money_status.get("status", "provider_empty"),
                    "source": money_status.get("source"),
                    "error": money_status.get("error"),
                    "message": money_status.get("message"),
                })
        except Exception as exc:  # noqa: BLE001
            source_status["stock_flow"].update({"status": "error", "message": f"个股主力资金读取失败：{str(exc)[:100]}"})
            stock_flow_payload.update({"status": "error", "error": str(exc)[:160]})

        # 跨日期读取研报产物；只接受索引中明确的六位代码，避免公司名模糊包含误归因。
        research_paths = sorted((ROOT / "generated").glob("research_flow_????-??-??.json"), reverse=True)
        for research_path in research_paths[:60]:
            try:
                payload = json.loads(research_path.read_text(encoding="utf-8"))
                indexed_codes = {str(code).strip().zfill(6) for code in (payload.get("all_codes") or []) if re.fullmatch(r"\d{1,6}", str(code).strip())}
                if symbol in indexed_codes:
                    research_index_hits += 1
                for row in payload.get("parsed", []):
                    codes = {str(code).strip().zfill(6) for code in (row.get("codes") or []) if re.fullmatch(r"\d{1,6}", str(code).strip())}
                    if symbol not in codes:
                        continue
                    research_items.append({
                        "title": row.get("title"), "date": row.get("date") or payload.get("date"),
                        "source": row.get("source"), "path": row.get("path"),
                        "concepts": row.get("concepts", []), "sectors": row.get("sectors", []),
                        "companies": row.get("companies", []), "chain_layers": row.get("chain_layers", []),
                        "leader_flags": row.get("leader_flags", []),
                        "catalysts": row.get("catalysts", []), "rating": row.get("rating"),
                        "target_price": row.get("target_price"),
                        "codes": sorted(codes),
                        "has_decision_extract": bool(row.get("rating") or row.get("target_price") or row.get("catalysts") or row.get("chain_layers") or row.get("leader_flags")),
                    })
            except Exception:
                continue
        # 研报工作流可能已生成空壳快照；这不应把本地真实报告全部遮蔽。
        # 兜底只做“六位代码精确命中”，并声明为代码命中证据，不推断评级/目标价。
        if not research_items:
            for report_path in _list_reports():
                report_file = Path(str(report_path.get("path") or ""))
                if report_path.get("date") in {"未知日期", ""} or not report_file.is_file():
                    continue
                try:
                    text = report_file.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                if not re.search(rf"(?<!\\d){re.escape(symbol)}(?!\\d)", text):
                    continue
                title = report_path.get("name") or report_file.name
                research_items.append({
                    "title": title, "date": report_path.get("date"),
                    "source": "local_report_code_match", "path": str(report_file),
                    "concepts": [], "sectors": [], "companies": [], "chain_layers": [],
                    "leader_flags": [], "catalysts": [], "rating": None, "target_price": None,
                    "codes": [symbol], "has_decision_extract": False,
                    "attribution_note": f"报告正文精确命中代码 {symbol}；未提取评级/目标价，不自动加分",
                })
                if len(research_items) >= 20:
                    break
        # 同一篇研报可能被多个日期产物重复索引，按标题/日期/来源去重。
        deduped = {}
        for row in research_items:
            key = (str(row.get("title") or ""), str(row.get("date") or ""), str(row.get("source") or ""))
            deduped[key] = row
        research_items = sorted(deduped.values(), key=lambda row: str(row.get("date") or ""), reverse=True)
        extracted = sum(1 for row in research_items if row.get("has_decision_extract"))
        if research_items:
            source_status["research"] = {
                "status": "available", "label": "研报提取",
                "as_of": research_items[0].get("date"),
                "count": len(research_items), "indexed_hits": research_index_hits, "extracted": extracted,
                "source": "local_report_code_match" if any(row.get("source") == "local_report_code_match" for row in research_items) else "generated/research_flow_YYYY-MM-DD.json",
                "message": f"命中{len(research_items)}篇代码级资料，{extracted}篇含评级/目标价/催化提取",
            }
        elif research_paths:
            source_status["research"] = {
                "status": "partial" if research_index_hits else "missing", "label": "研报提取",
                "as_of": None, "count": 0, "indexed_hits": research_index_hits, "extracted": 0,
                "message": f"代码出现在{research_index_hits}期索引，但正文字段尚未提取" if research_index_hits else "近60期未找到代码级匹配研报",
            }
        else:
            source_status["research"]["message"] = "未生成研报工作流产物"

        # 个股新闻只作为独立证据展示，不自动并入交易评分。只有数据源明确
        # 按请求代码归因时才输出标题，市场聚合新闻不会混入个股工作台。
        try:
            from quant_system.analysis_core.social_sentiment import fetch_news, sentiment_score
            stock_news = fetch_news(symbol=symbol, limit=30)
            attributed = (
                stock_news.get("ok")
                and stock_news.get("coverage") == "stock_symbol"
                and stock_news.get("source_symbol") == symbol
            )
            if attributed:
                news_items = []
                for row in stock_news.get("records") or []:
                    title = str(row.get("title") or "").strip()
                    if not title:
                        continue
                    score = float(sentiment_score(title))
                    news_items.append({
                        "title": title,
                        "date": row.get("date") or None,
                        "source": row.get("source") or stock_news.get("source"),
                        "url": row.get("url") or None,
                        "sentiment": "positive" if score > 0 else "negative" if score < 0 else "neutral",
                        "score": round(score, 4),
                    })
                stock_news_payload = {
                    "status": "available" if news_items else "missing",
                    "symbol": symbol, "source_symbol": symbol,
                    "coverage": "stock_symbol",
                    "coverage_note": f"数据源按 {symbol} 代码查询；标题仅供事件复核，不自动进入交易评分",
                    "attribution_method": "provider_symbol_query",
                    "decision_usage": "event_review_only",
                    "source": stock_news.get("source", "akshare.stock_news_em"),
                    "total": len(news_items),
                    "sentiment_score": round(float(stock_news.get("sentiment") or 0.0), 4),
                    "bull": int(stock_news.get("bull") or 0),
                    "bear": int(stock_news.get("bear") or 0),
                    "neutral": int(stock_news.get("neutral") or 0),
                    "items": news_items,
                }
                news_dates = [str(item.get("date"))[:10] for item in news_items if item.get("date")]
                news_as_of = max(news_dates) if news_dates else None
                source_status["news"] = {
                    "status": stock_news_payload["status"], "label": "个股新闻",
                    "as_of": news_as_of,
                    "retrieved_at": datetime.now().strftime("%Y-%m-%d"),
                    "count": len(news_items),
                    "message": f"{symbol} 代码级新闻{len(news_items)}条，仅供事件复核",
                }
            else:
                stock_news_payload["error"] = stock_news.get("error", "未取得代码级新闻")
                source_status["news"] = {"status": "missing", "label": "个股新闻", "message": stock_news_payload["error"]}
        except Exception as exc:
            stock_news_payload.update({"status": "error", "error": f"个股新闻读取失败：{str(exc)[:100]}"})
            source_status["news"] = {"status": "error", "label": "个股新闻", "message": stock_news_payload["error"]}

        # 统一决策契约：来源状态、新鲜度、证据质量与交易门控在一个对象内输出。
        quote_as_of = (profile.get("quote") or {}).get("date")
        reference_date = quote_as_of or datetime.now().strftime("%Y-%m-%d")
        source_status["profile"].update({"source": "data_warehouse/kline/<code>.parquet", "sample_size": len((profile.get("quote") or {}))})
        source_status["stock_flow"].setdefault("source", stock_flow_payload.get("source", "quant_system.money_flow"))
        source_status["stock_flow"].setdefault("sample_size", len(stock_flow_payload.get("rows") or []))
        source_status["flow"].setdefault("source", "generated/intraday_flow_history_YYYY-MM-DD.json")
        source_status["flow"].setdefault("sample_size", len(flow_payload.get("rows") or []))
        source_status["commodity"].setdefault("source", commodity_evidence.get("source", "local commodity warehouse"))
        source_status["commodity"].setdefault("coverage", commodity_evidence.get("coverage"))
        source_status["commodity"].setdefault("sample_size", len(commodity_evidence.get("items") or []))
        source_status["research"].setdefault("source", "generated/research_flow_YYYY-MM-DD.json")
        source_status["research"].setdefault("sample_size", len(research_items))
        source_status["news"].setdefault("source", stock_news_payload.get("source", "akshare.stock_news_em"))
        source_status["news"].setdefault("sample_size", len(stock_news_payload.get("items") or []))
        decision_gate = build_decision_gate(profile, decision, source_status, reference_date=reference_date)
        score_hierarchy = build_score_hierarchy(profile, decision, decision_gate)
        ic_weight_explanation = build_ic_weight_explanation()
        try:
            market_context = macro_snapshot()
        except Exception as exc:  # noqa: BLE001
            market_context = {"status": "error", "error": str(exc)[:120]}
        market_context.setdefault("status", "available" if market_context else "missing")
        market_context.setdefault("usage", "独立宏观背景，不直接替代个股财务或商品证据")

        # 结构化解释只引用已加载的证据；缺失来源不会被默认为中性或通过。
        decision_reasons = list(decision.get("short_term_reasons") or decision.get("trade_reasons") or [])
        profile_signals = [str(item.get("text")) for item in profile.get("signals", []) if item.get("text")]
        profile_score = profile.get("score") or {}
        commodity_evidence = profile.get("commodity") or {}
        decision_explanation = {
            "stance": decision.get("trade_label") or decision.get("decision") or "未形成个股交易结论",
            "supporting_factors": (decision_reasons + profile_signals)[:10],
            "blockers": list(decision.get("short_term_blockers") or decision.get("blockers") or []),
            "invalidators": list(decision.get("invalidators") or decision.get("stop_conditions") or []),
            "score_breakdown": {
                "profile_total": profile_score.get("total"),
                "profile_level": profile_score.get("level"),
                "profile_weights": profile_score.get("weights", {}),
                "profile_unavailable_dimensions": profile_score.get("unavailable_dimensions", []),
                "profile_coverage": profile_score.get("coverage", {}),
                "commodity_score": commodity_evidence.get("score"),
                "commodity_coverage": commodity_evidence.get("coverage"),
                "commodity_profile": commodity_evidence.get("profile"),
                "short_term_score": decision.get("short_term_score"),
                "decision_score": decision.get("decision_score"),
                "trade_allowed": decision.get("trade_allowed"),
            },
            "evidence": {
                "quote_as_of": (profile.get("quote") or {}).get("date"),
                "stock_flow_as_of": stock_flow_payload.get("as_of"),
                "industry_flow_as_of": flow_payload.get("date"),
                "research_as_of": source_status["research"].get("as_of"),
                "news": f"代码级新闻{stock_news_payload.get('total', 0)}条，仅作事件复核，未自动并入交易评分" if stock_news_payload.get("status") == "available" else "未加载可归因的个股新闻",
            },
            "method_note": "决策只使用有来源和数据日的证据；行业资金为方向代理，个股主力资金才是个股级资金证据；新闻标题不自动改变交易结论。",
        }

        stock_response = {
            "ok": True, "data_contract": "stock-overview", "symbol": symbol,
            "name": profile.get("name") or decision.get("name") or symbol,
            "as_of": quote_as_of,
            "decision_gate": _json_safe(decision_gate),
            "score_hierarchy": _json_safe(score_hierarchy),
            "ic_weight_explanation": _json_safe(ic_weight_explanation),
            "profile": _json_safe(profile), "decision": _json_safe(decision),
            "commodity": _json_safe(commodity_evidence),
            "market_context": _json_safe(market_context),
            "decision_explanation": _json_safe(decision_explanation),
            "flow": _json_safe(flow_payload), "stock_flow": _json_safe(stock_flow_payload),
            "research": _json_safe(research_items[:50]), "stock_news": _json_safe(stock_news_payload),
            "evidence_summary": {
                "quote": bool(profile.get("quote")),
                "stock_flow": stock_flow_payload.get("status") == "available",
                "industry_flow": flow_payload.get("status") == "available" or bool(flow_payload.get("rows")),
                "research_matches": len(research_items),
                "research_index_hits": research_index_hits,
                "research_extracted": sum(1 for row in research_items if row.get("has_decision_extract")),
                "news_headlines": int(stock_news_payload.get("total") or 0),
            },
            "source_status": _json_safe(decision_gate.get("source_status") or source_status),
        }
        try:
            from scripts.report_contract import contract_from_stock_overview, validate_report_contract
            stock_response["research_contract"] = contract_from_stock_overview(stock_response, report_type="stock")
            stock_response["research_contract_errors"] = validate_report_contract(stock_response["research_contract"])
        except Exception as exc:  # Keep the legacy endpoint available if adapter data is incomplete.
            stock_response["research_contract_error"] = str(exc)[:180]
        self._send_json(stock_response)
        return

        # Unreachable legacy research implementation retained below only as context.

    def _handle_history(self, query: dict[str, list[str]]) -> None:
        symbol = resolve_symbol(_first(query, "symbol", "002714"))
        start = _first(query, "start", "20240101").replace("-", "")
        end = _first(query, "end", "").replace("-", "")
        refresh = _first(query, "refresh", "false").lower() == "true"
        period = _first(query, "period", "daily")
        if period not in {"daily", "weekly", "monthly"}:
            self._send_json({"ok": False, "error": "period 必须为 daily/weekly/monthly"}, status=400)
            return
        if not re.fullmatch(r"20\d{6}", start) or (end and not re.fullmatch(r"20\d{6}", end)):
            self._send_json({"ok": False, "error": "start/end 必须为 YYYYMMDD 或 YYYY-MM-DD"}, status=400)
            return
        if end and end < start:
            self._send_json({"ok": False, "error": "end 不能早于 start"}, status=400)
            return
        strategy = _strategy_from_query(query)
        try:
            df = fetch_daily(symbol, start=start, use_cache=not refresh)
        except ValueError as exc:
            self._send_json({"ok": False, "data_status": "missing", "symbol": normalize_symbol(symbol),
                             "error": str(exc)[:180] or "未找到行情数据"}, status=404)
            return
        except Exception as exc:  # noqa: BLE001
            self._send_json({"ok": False, "data_status": "provider_error", "symbol": normalize_symbol(symbol),
                             "error": f"行情数据源暂不可用：{str(exc)[:160]}"}, status=502)
            return
        if end and not df.empty:
            date_col = "date" if "date" in df.columns else None
            if date_col:
                df = df[pd.to_datetime(df[date_col], errors="coerce") <= pd.Timestamp(end)]
        if df.empty:
            self._send_json({"ok": False, "error": "所选日期区间无K线数据", "symbol": normalize_symbol(symbol)}, status=404)
            return
        if period in ("weekly", "monthly"):
            from quant_system.data import aggregate_timeframe
            df = aggregate_timeframe(df, period)
        data = add_technical_indicators(df, strategy).tail(480)
        rows = _records(data)
        snapshot = latest_indicator_snapshot(normalize_symbol(symbol), df, strategy)
        self._send_json({"ok": True, "symbol": normalize_symbol(symbol), "period": period, "start": start, "end": end or None, "snapshot": snapshot, "rows": rows})

    def _handle_scan(self, body: dict) -> None:
        symbols = _symbols(body.get("symbols", ""))
        if not symbols:
            self._send_json({"ok": False, "error": "请输入股票代码或名称"}, status=400)
            return
        strategy = _strategy(body.get("strategy") or {})
        portfolio = _portfolio(body.get("portfolio") or {})
        market = load_market_state()
        start = str(body.get("start") or "20240101")
        refresh = bool(body.get("refresh"))
        data = fetch_many(symbols, start=start, use_cache=not refresh)
        rows = []
        failures = []
        for symbol, df in data.items():
            if symbol == "__failures__":
                failures = _records(df)
                continue
            signal = latest_signal(symbol, df, strategy)
            snapshot = latest_indicator_snapshot(symbol, df, strategy)
            rows.append({**enrich_trade_plan(signal, market, strategy, portfolio), **{f"ind_{k}": v for k, v in snapshot.items() if k not in {"symbol", "date"}}})
        self._send_json({"ok": True, "market": market, "rows": rows, "failures": failures})

    def _handle_backtest(self, body: dict) -> None:
        symbols = _symbols(body.get("symbols", ""))
        if not symbols:
            self._send_json({"ok": False, "error": "请输入股票代码或名称"}, status=400)
            return
        strategy = _strategy(body.get("strategy") or {})
        portfolio = _portfolio(body.get("portfolio") or {})
        start = str(body.get("start") or "20240101")
        refresh = bool(body.get("refresh"))
        portfolio_mode = bool(body.get("portfolio_mode"))
        data = fetch_many(symbols, start=start, use_cache=not refresh)
        if portfolio_mode and len(symbols) > 1:
            symbol_dfs = {k: v for k, v in data.items() if k != "__failures__"}
            result = run_portfolio_backtest(symbol_dfs, strategy, portfolio)
            self._send_json({"ok": True, "summary": result.get("summary"), "results": result.get("symbols"), "portfolio_mode": True})
            return
        rows = []
        failures = []
        for symbol, df in data.items():
            if symbol == "__failures__":
                failures = _records(df)
                continue
            result = run_backtest(df, strategy, portfolio)
            enriched = {"symbol": symbol, **_enrich_backtest_with_benchmark(result)}
            annual = float(enriched.get("annual_return_pct") or 0)
            sharpe = float(enriched.get("sharpe") or 0)
            drawdown = float(enriched.get("max_drawdown_pct") or 0)
            trades = int(enriched.get("trade_count") or 0)
            validated = trades >= 20 and annual > 0 and sharpe > 0.3 and drawdown > -25
            enriched["validation"] = {
                "passed": validated,
                "auto_execute_allowed": validated,
                "confidence_cap": 0.7 if validated else 0.25,
                "reason": "通过最低样本、收益、夏普和回撤门槛" if validated else "策略失效或样本不足：禁止模拟盘自动执行",
                "thresholds": {"min_trades": 20, "annual_return_gt": 0, "sharpe_gt": 0.3, "max_drawdown_gt": -25},
            }
            rows.append(enriched)
        payload = {"ok": True, "rows": rows, "failures": failures, "provenance": {
            "symbols": symbols, "start": start, "strategy": body.get("strategy") or {},
            "portfolio": body.get("portfolio") or {}, "generated_at": datetime.now().isoformat(),
            "engine": "quant_system.backtest.run_backtest", "synthetic": False}}
        try:
            out = ROOT / "generated" / "backtest_latest.json"
            out.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            payload["save_warning"] = str(exc)[:120]
        self._send_json(payload)

    def _handle_chat(self, body: dict) -> None:
        message = str(body.get("message") or "").strip()
        context = body.get("context") or {}
        # 内置财经 Agent 默认补充统一决策工作台和可用 skills 摘要，
        # 用户仍可在 context 中覆盖标的、周期和分析重点。
        if body.get("use_platform_context", True):
            try:
                from quant_web.workbench import build_workbench
                context = {**context, "统一决策工作台": build_workbench(mode="after_close")}
            except Exception as exc:
                context = {**context, "平台数据状态": f"工作台读取失败：{str(exc)[:120]}"}
            try:
                context["可用财经技能"] = [
                    item for item in _list_skills()
                    if any(word in str(item).lower() for word in ("stock", "market", "trading", "fund", "factor", "macro", "risk", "投资", "交易"))
                ][:24]
            except Exception:
                context["可用财经技能"] = []
        model = str(body.get("model") or DEFAULT_CHAT_MODEL).strip()
        if not message:
            self._send_json({"ok": False, "error": "请输入问题"}, status=400)
            return
        prompt = _chat_prompt(message, context)
        try:
            result = run_openclaw_agent(prompt, model=model or None)
        except subprocess.TimeoutExpired:
            self._send_json({"ok": False, "error": "AI 请求超时"}, status=504)
            return
        except Exception as exc:
            self._send_json({"ok": False, "error": f"AI 请求失败: {str(exc)[:180]}"}, status=502)
            return
        if not result.get("command_ok"):
            self._send_json({"ok": False, "error": "AI 命令执行失败", "detail": {
                "exit_code": result.get("exit_code"), "stderr": result.get("stderr", "")[:300]
            }}, status=502)
            return
        reply = str(result.get("text") or "").strip()
        if not reply:
            self._send_json({"ok": False, "error": "AI 返回为空"}, status=502)
            return
        self._send_json({"ok": True, "reply": reply})

    def _handle_alert(self, body: dict) -> None:
        atype = str(body.get("type", "scan"))
        rows = body.get("rows", [])
        market = body.get("market") or {}
        if not rows:
            self._send_json({"ok": False, "error": "没有预警数据"}, status=400)
            return
        title = f"⚡ A股量化预警 — {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        body_text = build_alert_body(atype, rows, market)
        delivery = send_feishu_message(title, body_text)
        if not delivery.get("ok"):
            self._send_json({"ok": False, "error": "飞书发送失败", "detail": delivery}, status=502)
            return
        self._send_json({"ok": True, "message": "预警已发送到飞书", "delivery": delivery})

    def _handle_optimize(self, body: dict) -> None:
        symbol = resolve_symbol(str(body.get("symbol", "002714")))
        start = str(body.get("start") or "20230101")
        refresh = bool(body.get("refresh"))
        strategy = _strategy(body.get("strategy") or {})
        portfolio = _portfolio(body.get("portfolio") or {})
        param_grid = body.get("param_grid") or {}
        sort_by = str(body.get("sort_by", "total_return_pct"))
        df = fetch_daily(symbol, start=start, use_cache=not refresh)
        results = grid_search(df, strategy, portfolio, param_grid)
        if sort_by and results:
            results.sort(key=lambda x: x.get("result", {}).get(sort_by, 0), reverse=True)
        self._send_json({"ok": True, "symbol": symbol, "results": results[:50]})

    def _handle_sector(self, query: dict[str, list[str]]) -> None:
        raw = _first(query, "symbols", "")
        symbols = _symbols(raw) if raw else []
        refresh = _first(query, "refresh", "false").lower() == "true"
        try:
            from quant_system.data import fetch_sector_map
            smap = fetch_sector_map(refresh=refresh)
        except Exception:
            smap = {}
        # 按板块分组
        from collections import defaultdict
        sectors = defaultdict(list)
        sector_names = {}
        for s in symbols:
            # If no sector map, try to get sector from individual stock info
            sector = smap.get(s, None)
            if sector is None and not refresh:
                # 给一个更友好的分类提示
                if s.startswith('6'):
                    sector = "沪市"
                elif s.startswith('30'):
                    sector = "创业板"
                elif s.startswith('00'):
                    sector = "深市主板"
                else:
                    sector = "其他"
            elif sector is None:
                sector = "其他"
            sectors[sector].append(s)
        for code in sectors:
            sector_names[code] = code
        has_real_sectors = any(k not in ("沪市","深市主板","创业板","其他") for k in sectors)
        note = "" if has_real_sectors or not symbols else "行业板块数据源暂时不可用，使用交易所分类替代"
        self._send_json({"ok": True, "sectors": dict(sectors), "sector_names": sector_names, "note": note})

    def _handle_portfolio_optimizer(self, query: dict[str, list[str]]) -> None:
        """GET /api/portfolio_optimizer — 组合优化"""
        try:
            from quant_system.portfolio_optimizer import (
                mean_variance_optimize, equal_risk_contribution,
                efficient_frontier, portfolio_summary
            )

            # 用默认的大盘股构建代理组合
            proxy_symbols = ["600519","000858","002714","601899","002594","300750","600036","601318","000333","600276"]

            from quant_system.data import fetch_daily
            prices = {}
            for sym in proxy_symbols:
                try:
                    df = fetch_daily(sym, start="20260101")
                    if df is not None and not df.empty:
                        prices[sym] = df['close'].values[-120:]
                except Exception as e:
                    logging.getLogger(__name__).error(f"[server] 操作失败: {e}", exc_info=True)

            if not prices:
                self._send_json({"ok": True, "data": {"note": "无法获取价格数据"}})
                return

            import pandas as pd
            price_df = pd.DataFrame(prices)
            returns_df = price_df.pct_change().dropna()
            mean_ret = returns_df.mean() * 252
            cov = returns_df.cov() * 252

            mode = _first(query, "mode", "mv")
            if mode == "frontier":
                result = efficient_frontier(mean_ret, cov, points=50)
            elif mode == "erc":
                result = equal_risk_contribution(mean_ret, cov)
            else:
                result = mean_variance_optimize(mean_ret, cov)

            result["symbols"] = list(price_df.columns)
            self._send_json({"ok": True, "data": _json_safe(result)})
        except Exception as e:
            import traceback
            self._send_json({"ok": False, "error": str(e) + " | " + traceback.format_exc()[-300:]}, status=500)

    def _handle_market_attribution(self, query: dict[str, list[str]]) -> None:
        """GET /api/market_attribution — 驱动因子归因"""
        try:
            from quant_system.market_attribution import compute_daily_attribution
            result = compute_daily_attribution()
            self._send_json({"ok": True, "data": _json_safe(result)})
        except Exception as e:
            self._send_json({"ok": False, "error": str(e)}, status=500)

    # ── V7 ML 监控 ──────────────────────────────────────────────────────

    def _handle_ml_model_info(self) -> None:
        """GET /api/ml_model_info — 模型状态/特征重要性/性能"""
        try:
            from quant_system.ml_signals import auto_retrain, train_model, _record_top_features
            import os, glob, json, time

            info = {"ok": True, "data": {}}
            model_dir = os.path.join(os.path.dirname(__file__), "..", "quant_system", "models")
            os.makedirs(model_dir, exist_ok=True)

            # 模型文件状态
            models_found = []
            for f in sorted(glob.glob(os.path.join(model_dir, "*.pkl"))):
                basename = os.path.basename(f)
                mtime = os.path.getmtime(f)
                size_kb = round(os.path.getsize(f) / 1024, 1)
                age_h = round((time.time() - mtime) / 3600, 1)
                models_found.append({
                    "name": basename,
                    "size_kb": size_kb,
                    "age_h": age_h,
                    "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))
                })
            info["data"]["models"] = models_found

            # 特征重要性缓存
            imp_path = os.path.join(model_dir, "feature_importance.json")
            if os.path.exists(imp_path):
                with open(imp_path) as f:
                    imp = json.load(f)
                if isinstance(imp, list):
                    info["data"]["feature_importance"] = imp[:30]
                elif isinstance(imp, dict):
                    top = sorted(imp.items(), key=lambda x: abs(x[1]) if isinstance(x[1], (int,float)) else 0, reverse=True)[:30]
                    info["data"]["feature_importance"] = [{"feature": k, "importance": v} for k, v in top]

            # 训练记录
            train_log = os.path.join(model_dir, "training_log.json")
            if os.path.exists(train_log):
                with open(train_log) as f:
                    log = json.load(f)
                info["data"]["training_log"] = log if isinstance(log, list) else [log]

            # ensemble 版本信息
            ensemble_p = os.path.join(model_dir, "ensemble_version.json")
            if os.path.exists(ensemble_p):
                with open(ensemble_p) as f:
                    info["data"]["ensemble_version"] = json.load(f)

            self._send_json(info)
        except Exception as e:
            import traceback
            self._send_json({"ok": False, "error": str(e), "trace": traceback.format_exc()[-200:]}, status=500)

    def _handle_ml_retrain(self, body: dict | None = None) -> None:
        """POST /api/ml_retrain — 触发重新训练"""
        try:
            from quant_system.ml_signals import train_ensemble, auto_retrain
            import time
            start = time.time()
            result = auto_retrain(force=True)
            elapsed = round(time.time() - start, 1)
            self._send_json({"ok": True, "data": {
                "result": str(result),
                "elapsed_s": elapsed,
                "message": f"训练完成，耗时 {elapsed}s"
            }})
        except Exception as e:
            import traceback
            self._send_json({"ok": False, "error": str(e) + " | " + traceback.format_exc()[-300:]}, status=500)

    def _handle_ml_ensemble(self, query: dict[str, list[str]]) -> None:
        """GET /api/ml_ensemble?symbol=002714 — ensemble 预测"""
        symbol = _first(query, "symbol", "600519").strip()
        try:
            from quant_system.ml_signals import predict_ensemble
            _async_deep(f"deep:ml_ensemble:{symbol}", 600,
                        lambda: {"ok": True, "data": _json_safe(predict_ensemble(resolve_symbol(symbol)))},
                        self._send_json)
        except Exception as e:
            import traceback
            self._send_json({"ok": False, "error": str(e) + " | " + traceback.format_exc()[-300:]}, status=500)

    # ── 数据基座滚动更新状态 ──

    def _handle_data_update_status(self) -> None:
        """GET /api/data_update_status — 每个数据域的更新状态与目标日期。"""
        path = ROOT / "generated" / "data_update_status.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
                "schema": "data_update_status/v1", "status": "never_run", "domains": {}
            }
            self._send_json({"ok": True, "data": data})
        except Exception as exc:
            self._send_json({"ok": False, "error": f"数据更新状态读取失败: {str(exc)[:160]}"}, status=500)

    # ── V8 DataStore 缓存管理 ──

    def _handle_cache_status(self) -> None:
        """GET /api/cache_status or /api/cache/status — 数据缓存健康状态"""
        try:
            from quant_system.data_store import get_store
            ds = get_store()
            cache_dir = ROOT / "generated" / "a_share_data"
            files = list(cache_dir.glob("*.json")) if cache_dir.exists() else []
            latest = max(files, key=lambda p: p.stat().st_mtime) if files else None
            total_size = sum(p.stat().st_size for p in files)
            generated_status = {
                "path": str(cache_dir),
                "file_count": len(files),
                "latest_file": latest.name if latest else None,
                "latest_file_date": datetime.fromtimestamp(latest.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S") if latest else None,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / 1024 / 1024, 3),
            }
            fm_cache = self.__class__._fm_cache
            fm_cache_info = {
                "entries": len(fm_cache),
                "keys": list(fm_cache.keys()),
                "ttl_seconds": self.__class__._FM_CACHE_TTL,
            }
            self._send_json({"ok": True, "data": {"data_store": ds.status(), "generated_a_share_data": generated_status, "fm_cache": fm_cache_info}})
        except Exception as e:
            self._send_json({"ok": False, "error": str(e)}, status=500)

    def _handle_cache_refresh(self, query: dict[str, list[str]]) -> None:
        """GET /api/cache_refresh?symbols=002714,600519 — 强制刷新缓存"""
        try:
            from quant_system.data_store import get_store
            symbols_str = _first(query, "symbols", "")
            symbols = [s.strip() for s in symbols_str.split(",") if s.strip()] or None

            def _compute():
                ds = get_store()
                return {"ok": True, "data": ds.refresh_all(symbols)}

            _async_deep(f"deep:cache_refresh:{symbols_str}", 300, _compute, self._send_json)
        except Exception as e:
            self._send_json({"ok": False, "error": str(e)}, status=500)

    # ── V8 事件驱动回测 ──

    def _handle_v8_backtest(self, query: dict[str, list[str]]) -> None:
        """GET /api/v8_backtest — V8 事件驱动回测"""
        try:
            from quant_system.event_backtest import EventBacktest

            symbols_str = _first(query, "symbols", "002714,600519,000858,601899,002594")
            symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
            symbols = [resolve_symbol(s) for s in symbols]
            start = _first(query, "start", "20260101")
            mode = _first(query, "mode", "ma").lower()
            capital = float(_first(query, "capital", "1000000"))
            max_pos = int(_first(query, "max_pos", "5"))
            max_pct = float(_first(query, "max_pct", "0.20"))
            slippage = _first(query, "slippage", "dynamic")

            engine = EventBacktest(
                symbols=symbols,
                initial_capital=capital,
                slippage_model=slippage,
                max_positions=max_pos,
                max_single_pct=max_pct,
            )
            if mode == "ma":
                fast = int(_first(query, "fast", "5"))
                slow = int(_first(query, "slow", "20"))
                engine.set_signal_ma_cross(fast=fast, slow=slow)
            elif mode == "macd":
                engine.set_signal_macd()

            def _compute():
                report = engine.run(start=start)
                # Trim equity curve to max 500 points for JSON
                ec = report.get("equity_curve", [])
                if len(ec) > 500:
                    step = len(ec) // 500
                    report["equity_curve"] = ec[::step]
                return {"ok": True, "data": _json_safe(report)}

            _async_deep(f"deep:v8_backtest:{mode}:{','.join(symbols)}:{start}", 600, _compute, self._send_json)
        except Exception as e:
            import traceback
            self._send_json({"ok": False, "error": str(e) + " | " + traceback.format_exc()[-300:]}, status=500)

    # ── V8 多因子模型 ──

    def _handle_factor_report(self, query: dict[str, list[str]]) -> None:
        """GET /api/factor_report — 因子暴露 & IC"""
        try:
            from quant_system.factor_model import get_model
            from datetime import timedelta as _td
            model = get_model()
            symbols = ["002714","600519","000858","601899","002594","300750"]
            # Use more stocks for better IC
            symbols = ["600519","000858","002714","601899","002594","300750","600036","601318","000333","600276","000568","002415","000001","601166","600900","600887","601398","601939","601288","601988"]
            requested_date = _first(query, "date", "")
            dates_to_try = [requested_date] if requested_date else [
                (datetime.now() - _td(days=i)).strftime("%Y-%m-%d") for i in range(8)
            ]
            date = dates_to_try[0]
            factor_df = pd.DataFrame()
            for candidate in dates_to_try:
                date = candidate
                factor_df = model.compute_factors(date, symbols=symbols)
                if requested_date or not factor_df.empty:
                    break
            ic = model.compute_ic(date, symbols=symbols) if len(symbols) >= 10 else pd.Series()
            self._send_json({"ok": True, "data": _json_safe({
                "date": date,
                "n_stocks": len(factor_df),
                "n_factors": len(factor_df.columns),
                "factor_ic": ic.to_dict() if not ic.empty else {},
            })})
        except Exception as e:
            import traceback
            self._send_json({"ok": False, "error": str(e), "trace": traceback.format_exc()[-200:]}, status=500)

    # ── V10 因子工坊：IC 分析 + 因子合成（读现成 IC 报告，秒级返回）──

    @staticmethod
    def _load_ic_report() -> list[dict]:
        """读取服务器全量 IC 报告 CSV → 因子列表。

        数据源: generated/ic_report/FACTOR_IC_REPORT_VECTORIZED.csv
          （8-06/8-07 服务器全量重跑结果，1266 个交易日跨度，122 因子）。
        失败返回空列表。
        """
        import csv as _csv
        paths = [
            Path(__file__).resolve().parent.parent / "generated" / "ic_report" / "FACTOR_IC_REPORT_VECTORIZED.csv",
            Path(__file__).resolve().parent.parent / "generated" / "ic_report" / "FACTOR_IC_REPORT_VECTORIZED.csv",
        ]
        p = paths[0]
        if not p.exists():
            return []
        try:
            rows = []
            with open(p, encoding="utf-8-sig") as f:
                for r in _csv.DictReader(f):
                    try:
                        rows.append({
                            "name": r.get("factor", ""),
                            "category": r.get("category", ""),
                            "ic_mean": float(r.get("ic_mean", 0) or 0),
                            "icir": float(r.get("icir", 0) or 0),
                            "win_rate": float(r.get("winrate", 0) or 0) * 100,
                            "n_days": int(float(r.get("n_days", 0) or 0)),
                            "coverage": float(r.get("coverage", 0) or 0),
                            "grade": r.get("grade", ""),
                            "direction": int(float(r.get("direction", 1) or 1)),
                        })
                    except Exception as e:
                        logging.getLogger(__name__).error(f"[server] 操作失败: {e}", exc_info=True)
                        continue
            return rows
        except Exception:
            return []

    def _handle_factor_ic(self, query: dict[str, list[str]]) -> None:
        """GET /api/factor_ic?start=YYYYMMDD&end=YYYYMMDD — 因子 IC 分析。

        返回 {factors: [{name, ic_mean, icir, win_rate, rating}], range, chart}。
        IC 报告为全区间统计（2022 至今），start/end 作为展示提示。
        """
        try:
            start = _first(query, "start", "20230101")
            end = _first(query, "end", "")
            rows = self._load_ic_report()
            if not rows:
                self._send_json({"ok": True,
                    "factors": [], "range": f"{start}~{end or '今'}",
                    "chart": None, "msg": "IC 报告未生成，请先在服务器跑全量 IC 重跑",
                    "n": 0, "source": "FACTOR_IC_REPORT_VECTORIZED.csv",
                })
                return
            # 评级：|ic_mean|>=0.05 且 |icir|>=0.3 → A；|ic_mean|>=0.03 → B；否则 C
            def _rating(r: dict) -> str:
                a = abs(r["ic_mean"])
                b = abs(r["icir"])
                if a >= 0.05 and b >= 0.3:
                    return "A"
                if a >= 0.03 or b >= 0.2:
                    return "B"
                return "C"
            factors = []
            for r in rows:
                factors.append({
                    "name": r["name"],
                    "category": r["category"],
                    "ic_mean": round(r["ic_mean"], 4),
                    "icir": round(r["icir"], 3),
                    "win_rate": round(r["win_rate"], 1),
                    "n_days": r["n_days"],
                    "coverage": round(r["coverage"], 3),
                    "grade": r["grade"],
                    "direction": r["direction"],
                    "rating": _rating(r),
                })
            factors.sort(key=lambda x: abs(x["ic_mean"]), reverse=True)
            # 图表：top 15 因子 IC 条形图（红正绿负，A股惯例）
            top = factors[:15]
            chart = {
                "tooltip": {"trigger": "axis"},
                "grid": {"left": 90, "right": 30, "top": 20, "bottom": 40},
                "xAxis": {"type": "value", "name": "IC均值"},
                "yAxis": {"type": "category", "data": [f["name"] for f in reversed(top)],
                          "axisLabel": {"fontSize": 10}},
                "series": [{"type": "bar",
                             "data": [
                                 {"value": round(f["ic_mean"], 4),
                                  "itemStyle": {"color": "#f6465d" if f["ic_mean"] >= 0 else "#2ebd85"}}
                                 for f in reversed(top)
                             ]}],
            }
            self._send_json({"ok": True,
                "factors": factors, "range": f"{start}~{end or '今'}", "chart": chart,
                "n": len(factors), "source": "FACTOR_IC_REPORT_VECTORIZED.csv",
            })
        except Exception as e:
            import traceback
            self._send_json({"ok": False, "error": f"IC 分析失败: {e}", "trace": traceback.format_exc()[-200:]}, status=500)

    def _handle_factor_combine(self, query: dict[str, list[str]]) -> None:
        """GET /api/factor_combine?method=equal|ic_weight|top_n — 因子合成评估。

        基于 IC 报告计算合成 IC 的估算值：
          - equal:     等权（平均 IC）
          - ic_weight: IC 加权（|IC| 越大的因子权重越高，方向修正后）
          - top_n:     仅取 top 10 因子等权
        返回 {method, ic_mean, icir, chart}。
        """
        try:
            method = _first(query, "method", "equal")
            rows = self._load_ic_report()
            if not rows:
                self._send_json({"ok": True, "method": method, "ic_mean": 0, "icir": 0, "chart": None, "msg": "IC 报告未生成"})
                return
            # 有效因子（coverage>=0.5 且方向已修正）
            valid = [r for r in rows if r["coverage"] >= 0.5]
            if not valid:
                self._send_json({"ok": True, "method": method, "ic_mean": 0, "icir": 0, "chart": None, "msg": "无有效因子"})
                return
            # 方向修正：direction=-1 的因子 IC 取反后参与合成（A股红涨绿跌，IC 为负=反向因子）
            if method == "ic_weight":
                weights = []
                for r in valid:
                    w = abs(r["ic_mean"]) * r["coverage"]
                    weights.append((r, w))
                weights.sort(key=lambda x: x[1], reverse=True)
                weights = weights[:20]  # 只取前 20
                total_w = sum(w for _, w in weights) or 1.0
                ic_mean = sum(r["ic_mean"] * r["direction"] * (w / total_w) for r, w in weights)
                names = [r["name"] for r, w in weights]
            elif method == "top_n":
                top = sorted(valid, key=lambda r: abs(r["ic_mean"]), reverse=True)[:10]
                ic_mean = sum(r["ic_mean"] * r["direction"] for r in top) / len(top)
                names = [r["name"] for r in top]
            else:  # equal
                ic_mean = sum(r["ic_mean"] * r["direction"] for r in valid) / len(valid)
                names = [r["name"] for r in valid]
            # ICIR 估算：用方向修正后 IC 的均值/标准差（截面估算）
            import numpy as _np
            ic_vals = _np.array([r["ic_mean"] * r["direction"] for r in valid])
            icir = float(ic_vals.mean() / ic_vals.std()) if ic_vals.std() > 0 else 0.0
            icir = round(icir, 3)
            # 图表：权重分布饼图/条形
            chart = {
                "tooltip": {"trigger": "axis"},
                "grid": {"left": 110, "right": 30, "top": 20, "bottom": 40},
                "xAxis": {"type": "value"},
                "yAxis": {"type": "category", "data": names[-15:][::-1], "axisLabel": {"fontSize": 9}},
                "series": [{"type": "bar", "data": [0.05 for _ in names[-15:]]}],
            }
            self._send_json({"ok": True,
                "method": method, "ic_mean": round(ic_mean, 4), "icir": icir,
                "n_factors": len(names), "chart": chart,
            })
        except Exception as e:
            import traceback
            self._send_json({"ok": False, "error": f"因子合成失败: {e}", "trace": traceback.format_exc()[-200:]}, status=500)

    # ── V8 财务因子数据 ──

    def _handle_financial_data(self, query: dict[str, list[str]]) -> None:
        """GET /api/financial_data?symbol=600519"""
        try:
            from quant_system.financial_data import fetch_financial_indicators
            symbol = _first(query, "symbol", "002714").strip()
            if not (symbol.isdigit() and len(symbol) == 6):
                self._send_json({"ok": False, "error": "股票代码必须是6位数字"}, status=400)
                return
            # 财务报表是季度数据。使用150天有效期可直接读取已落盘报告期，
            # 避免 Web 查询为了刷新季度数据而尝试写共享 SQLite/WAL。
            indicators = fetch_financial_indicators(symbol, force=False, max_age_days=150)
            fetch_error = indicators.pop("_error", None) if isinstance(indicators, dict) else None
            if fetch_error:
                self._send_json({"ok": False, "error": f"财务数据读取失败: {fetch_error}"}, status=503)
                return
            self._send_json({"ok": True, "data": {
                "symbol": symbol,
                "indicators": indicators,
                "available": len(indicators),
                "source": "财务报告缓存",
            }})
        except Exception as e:
            self._send_json({"ok": False, "error": str(e)}, status=500)

    # ── V8 特征存储 & 漂移检测 ──

    def _handle_feature_panel(self, query: dict[str, list[str]]) -> None:
        """GET /api/feature_panel — 截面特征主链路评估（V2-3 数据资产化）

        从 data_warehouse/feature_store/ 读全部截面（DataStore 门面），
        对每个特征做 IC/ICIR/分层收益/衰减评估。
        """
        try:
            horizon = int(_first(query, "horizon", "5"))
            from quant_platform.features import build_panel_and_evaluate
            _async_deep(f"deep:feature_panel:{horizon}", 600,
                        lambda: {"ok": True, "data": build_panel_and_evaluate(horizon=horizon)},
                        self._send_json)
        except Exception as e:  # noqa: BLE001
            self._send_json({"ok": False, "error": repr(e), "trace": traceback.format_exc(limit=4)}, status=500)

    def _handle_feature_store(self, query: dict[str, list[str]]) -> None:
        """GET /api/feature_store — 特征存储统计"""
        try:
            from quant_system.feature_store import feature_store_stats, compute_drift, get_drift_history
            stats = feature_store_stats()
            drift = compute_drift()
            history = get_drift_history()
            self._send_json({"ok": True, "data": {
                "stats": stats,
                "drift": drift,
                "drift_history": history[-10:],
            }})
        except Exception as e:
            self._send_json({"ok": False, "error": str(e)}, status=500)

    def _handle_drift_check(self, query: dict[str, list[str]]) -> None:
        """GET /api/drift_check — 模型漂移检测"""
        try:
            from quant_system.feature_store import compute_drift
            drift = compute_drift()
            self._send_json({"ok": True, "data": drift})
        except Exception as e:
            self._send_json({"ok": False, "error": str(e)}, status=500)

    # ── 2.0 资产配置引擎 ──

    # FM 结果缓存
    _fm_cache: dict = {}
    _FM_CACHE_TTL = 300  # 5 min

    def _fm_cache_key(self, query: dict[str, list[str]]) -> str:
        syms = query.get("symbols", [""])[0]
        step = query.get("step_days", ["10"])[0]
        date = datetime.now().strftime("%Y%m%d")
        return f"{date}|{step}|{syms}"

    def _handle_fama_macbeth(self, query: dict[str, list[str]]) -> None:
        """GET /api/fama_macbeth — Fama-MacBeth 截面回归"""
        # 检查缓存
        ck = self._fm_cache_key(query)
        now = time.time()
        if ck in self.__class__._fm_cache:
            cached = self.__class__._fm_cache[ck]
            if now - cached["ts"] < self.__class__._FM_CACHE_TTL:
                cached["ts"] = now
                self._send_json(cached["data"])
                return
        try:
            from quant_system.fama_macbeth import get_fama_macbeth
            from quant_system.data_store import get_store as _gs
            symbols_str = query.get("symbols", ["600519,000858,002714,601899,002594,300750,600036,601318,000333,600276,000568,002415"])[0]
            symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
            step = int(query.get("step_days", ["10"])[0])

            def _compute():
                _st = _gs()
                for _sym in symbols:
                    _st.get(_sym, days=120)
                fm = get_fama_macbeth()
                result = fm.run(symbols=symbols, start_date="2026-05-01", step_days=step)
                return {"ok": True, "data": _json_safe(result)}

            _async_deep(f"deep:fama_macbeth:{ck}", 600, _compute, self._send_json)
        except Exception as e:
            import traceback
            if ck in self.__class__._fm_cache:
                self._send_json(self.__class__._fm_cache[ck]["data"])
                return
            self._send_json({"ok": False, "error": str(e) + " | " + traceback.format_exc()[-300:]}, status=500)

    def _handle_upgraded_risk_model(self, query: dict[str, list[str]]) -> None:
        """GET /api/risk_model_upgraded — Newey-West 风险模型"""
        try:
            from quant_system.factor_model import get_model
            symbols_str = query.get("symbols", ["600519,000858,002714,601899,002594,300750,600036,601318,000333,600276,000568,002415,000001,601166,600900,600887"])[0]
            symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
            date = query.get("date", [datetime.now().strftime("%Y-%m-%d")])[0]

            def _compute():
                model = get_model()
                result = model.risk_model(date, symbols)
                return {"ok": True, "data": _json_safe(result)}

            _async_deep(f"deep:risk_model_upgraded:{date}:{','.join(symbols)}", 600, _compute, self._send_json)
        except Exception as e:
            import traceback
            self._send_json({"ok": False, "error": str(e) + " | " + traceback.format_exc()[-300:]}, status=500)

    def _handle_backtest_almgren(self, query: dict[str, list[str]]) -> None:
        """GET /api/backtest_almgren — V8回测 Almgren-Chriss 冲击模型对比"""
        try:
            from quant_system.event_backtest import EventBacktest
            symbols_str = query.get("symbols", ["002714,600519,000858"])[0]
            symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
            sm = query.get("slippage", ["almgren"])[0]
            def _compute():
                engine = EventBacktest(symbols=symbols, slippage_model=sm, initial_capital=1_000_000)
                engine.set_signal_ma_cross(fast=5, slow=20)
                report = engine.run(start="20260101")
                ec = report.get("equity_curve", [])
                if len(ec) > 500:
                    step = len(ec) // 500
                    report["equity_curve"] = ec[::step]
                return {"ok": True, "data": _json_safe(report)}

            _async_deep(f"deep:backtest_almgren:{sm}:{','.join(symbols)}", 600, _compute, self._send_json)
        except Exception as e:
            import traceback
            self._send_json({"ok": False, "error": str(e) + " | " + traceback.format_exc()[-300:]}, status=500)

    def _handle_asset_allocation(self, query: dict[str, list[str]]) -> None:
        """GET /api/asset_allocation — 资产配置"""
        try:
            from quant_system.asset_allocation import get_allocator
            symbols_str = _first(query, "symbols", "600519,000858,002714,601899,002594,300750,600036,601318,000333,600276")
            symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
            objective = _first(query, "objective", "max_sharpe")

            def _compute():
                alloc = get_allocator()
                result = alloc.allocation_report(symbols, objective=objective)
                return {"ok": True, "data": _json_safe(result)}

            _async_deep(f"deep:asset_allocation:{objective}:{','.join(symbols)}", 600, _compute, self._send_json)
        except Exception as e:
            import traceback
            self._send_json({"ok": False, "error": str(e) + " | " + traceback.format_exc()[-300:]}, status=500)

    def _handle_realtime(self, query: dict[str, list[str]]) -> None:
        """GET /api/realtime?symbols=002714,牧原股份"""
        raw = query.get("symbols", [""])
        symbols = [resolve_symbol(s.strip()) for s in raw[0].split(",") if s.strip()]
        if not symbols:
            symbols = watchlist_from_file(str(STATIC_DIR.parent / "watchlist.json"))
        check_signals = _first(query, "signals", "false").lower() == "true"
        quotes = fetch_realtime(symbols)
        result = []
        for sym, q in sorted(quotes.items()):
            result.append({
                "symbol": sym,
                "code": sym,
                # 2026-08-14 P1-2: name 字段（行情对象可能无 name，getattr 防御）
                "name": getattr(q, "name", ""),
                "price": q.price,
                "change": q.change,
                "change_pct": safe_float(q.change_pct),
                "high": q.high,
                "low": q.low,
                "open": q.open,
                "volume": q.volume,
                "amplitude": round(q.amplitude, 2),
                "volume": q.volume,
                "amount": round(q.amount, 2),
                "date": q.date,
                "time": q.time,
            })
        # Lightweight signal check from daily data
        signal_map = {}
        if check_signals and quotes:
            strategy = _strategy_from_query(query)
            portfolio = _portfolio({})
            market = load_market_state()
            syms = [q.symbol for q in quotes.values()]
            if syms:
                try:
                    daily_data = fetch_many(syms, start="20250101", use_cache=True)
                    for sym in syms:
                        if sym in daily_data and not daily_data[sym].empty:
                            df = daily_data[sym]
                            df = add_technical_indicators(df, strategy)
                            signal = latest_signal(sym, df, strategy)
                            plan = enrich_trade_plan(signal, market, strategy, portfolio)
                            signal_map[sym] = {
                                "action": plan.get("action", ""),
                                "new_buy_allowed": plan.get("new_buy_allowed", False),
                                "composite_score": plan.get("composite_score", 0),
                                "risk_score": plan.get("risk_score", 0),
                                "stop_loss": plan.get("stop_loss"),
                            }
                except Exception as e:
                    logging.getLogger(__name__).error(f"[server] 操作失败: {e}", exc_info=True)

        self._send_json({"ok": True, "quotes": result, "signals": signal_map, "count": len(result)})

    def _handle_intraday(self, query: dict[str, list[str]]) -> None:
        """GET /api/intraday?symbol=002714&period=5&start=2026-07-20"""
        symbol = resolve_symbol(_first(query, "symbol", "002714"))
        period = _first(query, "period", "5")
        start = _first(query, "start", "")
        end = _first(query, "end", "")
        if period not in ("1", "5", "15", "30", "60"):
            period = "5"
        from quant_system.data import fetch_intraday
        df = fetch_intraday(symbol, period=period, start_date=start or None, end_date=end or None)
        if df.empty:
            self._send_json({"ok": False, "status": "no_data", "symbol": normalize_symbol(symbol),
                             "period": period, "rows": [],
                             "error": "所选代码、周期或日期区间没有日内K线；请检查交易日和数据源覆盖"}, status=404)
            return
        df = add_intraday_indicators(df)
        rows = _records(df.tail(480))
        self._send_json({"ok": True, "symbol": normalize_symbol(symbol), "period": period, "rows": rows})

    def _handle_search_stock(self, query: dict[str, list[str]]) -> None:
        """GET /api/search_stock?q=贵州|my|MYGF|600519 — 中文模糊+拼音+代码"""
        q = _first(query, "q", "")
        if not q:
            self._send_json({"ok": True, "results": []})
            return
        try:
            results = fuzzy_search_stocks(q, limit=20)
        except Exception:  # noqa: BLE001
            results = search_stock_name(q, limit=15)
        self._send_json({"ok": True, "results": results})

    # ── V9 数据融合 + AI 辅助决策 ──

    def _handle_stock_profile(self, query: dict[str, list[str]]) -> None:
        """GET /api/stock_profile?symbol=002714 — 个股全景档案"""
        sym = _first(query, "symbol", "") or _first(query, "code", "")
        if not sym:
            self._send_json({"ok": False, "error": "缺少 symbol"}, status=400)
            return
        try:
            prof = build_stock_profile(sym)
            if not prof.get("available"):
                self._send_json({"ok": False, "available": False, "data_status": "missing",
                                 "symbol": prof.get("code") or sym, "error": "未找到个股行情数据"}, status=404)
                return
            self._send_json({"ok": True, "profile": prof})
        except Exception as e:  # noqa: BLE001
            self._send_json({"ok": False, "error": f"分析失败: {e}"})

    def _handle_ai_decision(self, query: dict[str, list[str]]) -> None:
        """GET /api/ai_decision?symbol=002714 — AI 辅助决策（评分+建议+风险）"""
        sym = _first(query, "symbol", "") or _first(query, "code", "")
        if not sym:
            self._send_json({"ok": False, "error": "缺少 symbol"}, status=400)
            return
        try:
            prof = build_stock_profile(sym)
            if not prof.get("available"):
                self._send_json({"ok": False, "available": False, "data_status": "missing",
                                 "symbol": prof.get("code") or sym, "error": "缺少行情数据，不能生成中性或其他建议"}, status=404)
                return
            self._send_json({"ok": True, "decision": prof["ai_advice"],
                             "score": prof["score"], "signals": prof["signals"],
                             "data_as_of": prof.get("data_as_of"), "source_dates": prof.get("source_dates"),
                             "generated_at": prof.get("generated_at"), "coverage": prof.get("coverage")})
        except Exception as e:  # noqa: BLE001
            self._send_json({"ok": False, "error": f"分析失败: {e}"})

    def _handle_warehouse_status(self) -> None:
        """GET /api/warehouse_status — 数据仓库覆盖度/新鲜度（异步计算 + 缓存）"""
        try:
            _async_deep("deep:warehouse_status", 60,
                        lambda: {"ok": True, "status": warehouse_status()},
                        self._send_json)
        except Exception as e:  # noqa: BLE001
            self._send_json({"ok": False, "error": str(e)})

    def _current_version(self) -> str:
        try:
            from quant_system import __version__ as version
            return str(version)
        except Exception:
            return "unknown"

    def _version_info(self) -> dict:
        """GET /api/version — 运行实例版本信息（公开，不含敏感路径）。"""
        dirty = None
        try:
            import subprocess as _sp
            r = _sp.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                        capture_output=True, text=True, timeout=5)
            sha = (r.stdout or "").strip() or ""
            status = _sp.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "quant_web", "quant_system", "quant_platform", "scripts"],
                             capture_output=True, text=True, timeout=5)
            dirty = bool((status.stdout or "").strip())
        except Exception:
            sha = ""
        return {
            "ok": True,
            "service": "quant-web",
            "commit": sha,
            "version": self._current_version(),
            "pid": os.getpid(),
            "workspace_dirty": dirty,
            "host": _QUANT_WEB_BIND_HOST,
            "auth_configured": bool(_QUANT_WEB_API_KEY),
            "process_started_at": _PROCESS_STARTED_AT,
            "ready": _ready_info()["ready"],
            "ready_phase": _ready_info()["phase"],
            "api_catalog": "/api/catalog",
            "canonical_workbench": "/api/workbench",
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _handle_health(self) -> None:
        """GET /api/v1/health — 平台健康检查（异步计算 + 60s 缓存）。"""
        def _compute():
            audit = LOGS_DIR / "api_audit.log"
            audit_size = audit.stat().st_size if audit.exists() else 0
            try:
                from quant_platform import data as pdata
                datasets = pdata.list_datasets()
                ws = pdata.warehouse_root()
                kline_files = len(list((ws / "kline").glob("*.parquet"))) if (ws / "kline").exists() else 0
                val_files = len(list((ws / "valuation").glob("*.parquet"))) if (ws / "valuation").exists() else 0
            except Exception:
                datasets, kline_files, val_files = [], 0, 0
            try:
                from quant_platform.runtime import task_status
                tasks = task_status()
            except Exception:
                tasks = {"tasks": [], "count": 0}
            try:
                from quant_system.data_store import DataStore
                freshness = DataStore().freshness(trade_date=datetime.now().strftime("%Y-%m-%d"))
            except Exception:
                freshness = []
            return {
                "ok": True,
                "service": "quant-web",
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "python": sys.version.split()[0],
                "datasets": datasets,
                "kline_files": kline_files,
                "valuation_files": val_files,
                "audit_log_bytes": audit_size,
                "tasks": tasks,
                "freshness": freshness,
            }

        # 首次请求仍异步计算，但返回完整的健康接口形状，前端和旧调用方
        # 可以区分“正在计算”和“真实空数据”。下一次轮询拿到真实结果。
        def _send_health_progress(payload):
            if payload.get("status") == "computing":
                payload = {**payload, "service": "quant-web", "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                           "datasets": [], "tasks": {}, "freshness": [], "audit_log_bytes": 0}
            self._send_json(payload)
        _async_deep("deep:v1_health", 60, _compute, _send_health_progress)

    def _handle_audit(self) -> None:
        """GET /api/v1/audit — 全量审计（数据合理性 + 输出内容 + 维护自检）

        审计维护模块，输出内容合理性检查。
        """
        try:
            from quant_platform.audit import run_full_audit
            # 服务内运行避免全量语法扫描（耗时），只跑数据+输出两层
            import quant_platform.audit as _audit
            def _compute():
                data = _audit._data_audit()
                output = _audit._output_audit()
                all_f = data["findings"] + output["findings"]
                weighted = sum(_audit._LEVEL_WEIGHT.get(f["level"], 1) for f in all_f if f["level"] != "ok")
                score = max(0, min(100, 100 - weighted * 4))
                return {
                    "ok": True,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "score": round(score, 1),
                    "high_count": sum(1 for f in all_f if f["level"] == "high"),
                    "medium_count": sum(1 for f in all_f if f["level"] == "medium"),
                    "data": data,
                    "output": output,
                }

            _async_deep("deep:v1_audit", 600, _compute, self._send_json)
        except Exception as e:  # noqa: BLE001
            self._send_json({"ok": False, "error": str(e)})

    def _handle_segments(self) -> None:
        """GET /api/v1/segments — 市场口径分层（主板/双创/北交所）+ 风格分层

        日常分析复盘针对主板/非ST/北交双创/科创创业。
        数据源：最新 realtime_snapshot 快照（零网络）。
        """
        try:
            import glob as _glob
            from quant_platform.market_segments import segment_market, style_segments, short_medium_long
            snaps = sorted(_glob.glob(str(ROOT / "data_warehouse" / "realtime_snapshot" / "*" / "*.parquet")))
            if not snaps:
                self._send_json({"ok": False, "error": "无实时快照（盘中 cron 5分钟生成）"})
                return
            import pandas as _pd
            df = _pd.read_parquet(snaps[-1])
            seg = segment_market(df)
            style = style_segments(df)
            # short_medium_long 需中文列（代码/涨跌幅），转换快照列
            _sdf = df.copy()
            if "代码" not in _sdf.columns and "code" in _sdf.columns:
                _sdf["代码"] = _sdf["code"]
            if "涨跌幅" not in _sdf.columns and "pct_chg" in _sdf.columns:
                _sdf["涨跌幅"] = _pd.to_numeric(_sdf["pct_chg"], errors="coerce")
            _sdf["代码"] = _sdf["代码"].astype(str).str.replace(r"^(sh|sz|bj)", "", regex=True).str.zfill(6)
            sml = short_medium_long(_sdf)
            self._send_json({"ok": True, "data": {
                "segments": seg,
                "style": style,
                "short_medium_long": sml,
                "snapshot": snaps[-1].replace(str(ROOT) + "/", ""),
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }})
        except Exception as e:  # noqa: BLE001
            self._send_json({"ok": False, "error": str(e)})

    def _handle_macro_snapshot(self) -> None:
        """GET /api/macro_snapshot — 宏观快照（PMI/M2/SHIBOR/LPR）"""
        try:
            self._send_json({"ok": True, "macro": macro_snapshot()})
        except Exception as e:  # noqa: BLE001
            self._send_json({"ok": False, "error": str(e)})

    def _handle_ai_analyze(self, query: dict[str, list[str]]) -> None:
        """GET /api/ai_analyze?symbol=002714&model=deepseek/deepseek-v4-flash&type=panorama
        收集个股全景 + 因子扫描 + 行情摘要作为上下文，调 OpenClaw 指定模型分析。"""
        sym = _first(query, "symbol", "") or ""
        model = _first(query, "model", "") or DEFAULT_CHAT_MODEL
        atype = _first(query, "type", "panorama") or "panorama"
        if not sym:
            self._send_json({"ok": False, "error": "缺少 symbol"}, status=400)
            return
        try:
            context = {}
            try:
                from stock_analysis import build_stock_profile, _score_profile, _make_advice, _collect_signals
                profile = build_stock_profile(sym)
                context["profile"] = profile
            except Exception as e:  # noqa: BLE001
                context["profile_error"] = str(e)
            try:
                if _HAS_FACTOR_SCAN:
                    r = _factor_scan(sym)
                    context["factor_scan"] = r
            except Exception as e:  # noqa: BLE001
                context["factor_error"] = str(e)
            # 行情摘要（最近60日收盘/涨跌幅/量能）
            try:
                from stock_analysis import load_kline
                df = load_kline(sym)
                if df is not None and len(df):
                    tail = df.tail(60)
                    ctx = {
                        "symbol": sym,
                        "latest": tail.iloc[-1].to_dict() if len(tail) else {},
                        "close_5d": [round(float(x), 2) if x == x else None for x in tail["close"].tail(5).tolist()],
                        "pct_5d": [round(float(x), 2) if x == x else None for x in tail["pct_chg"].tail(5).tolist()] if "pct_chg" in tail.columns else [],
                        "vol_ratio": round(float(tail["volume"].iloc[-1] / tail["volume"].tail(20).mean()), 2) if len(tail) >= 20 and tail["volume"].tail(20).mean() else None,
                    }
                    context["kline_summary"] = ctx
            except Exception as e:  # noqa: BLE001
                context["kline_error"] = str(e)
            # V10.2: 龙虎榜历史（近 60 日该股上榜记录，短线必看）
            try:
                from pathlib import Path as _P
                lhb_dir = _P(__file__).resolve().parent.parent / "data_warehouse" / "market"
                lhb_rows = []
                for lf in sorted(lhb_dir.glob("lhb_*.parquet")):
                    try:
                        ldf = pd.read_parquet(lf)
                        if "代码" not in ldf.columns or "上榜日" not in ldf.columns:
                            continue
                        ldf = ldf.copy()
                        ldf["_c"] = ldf["代码"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
                        sub = ldf[ldf["_c"] == sym]
                        if len(sub):
                            lhb_rows.append(sub)
                    except Exception as e:
                        logging.getLogger(__name__).error(f"[server] 操作失败: {e}", exc_info=True)
                        continue
                if lhb_rows:
                    lhb_all = pd.concat(lhb_rows, ignore_index=True)
                    lhb_all["上榜日"] = pd.to_datetime(lhb_all["上榜日"], errors="coerce")
                    lhb_all = lhb_all.sort_values("上榜日").tail(10)
                    cols = [c for c in ["上榜日", "收盘价", "涨跌幅", "龙虎榜净买额", "上榜原因"] if c in lhb_all.columns]
                    context["lhb_history"] = {
                        "n": len(lhb_all),
                        "recent": lhb_all[cols].to_dict(orient="records") if cols else [],
                    }
            except Exception as e:  # noqa: BLE001
                context["lhb_error"] = str(e)
            # V10.4: 营业部/机构龙虎榜/股东户数 上下文（复盘用）
            try:
                wh = ROOT / "data_warehouse" / "market"
                # 机构龙虎榜（近60日）
                jgf = wh / "lhb_jgmmtj_em.parquet"
                if jgf.exists():
                    jdf = pd.read_parquet(jgf)
                    if "上榜日期" in jdf.columns and "代码" in jdf.columns and "机构买入净额" in jdf.columns:
                        jd = pd.to_datetime(jdf["上榜日期"], errors="coerce")
                        jdf = jdf[(jd >= pd.Timestamp.now() - pd.Timedelta(days=60))]
                        jdf = jdf.copy()
                        jdf["_c"] = jdf["代码"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
                        sub = jdf[jdf["_c"] == sym].sort_values("上榜日期").tail(6)
                        if len(sub):
                            context["jgmmtj_history"] = {
                                "n": len(sub),
                                "recent": sub[[c for c in ["上榜日期", "机构买入净额", "买方机构数", "卖方机构数"] if c in sub.columns]].to_dict(orient="records"),
                            }
                # 股东户数（最新一期）
                gdf = wh / "gdhs_all.parquet"
                if gdf.exists():
                    gd = pd.read_parquet(gdf)
                    if "代码" in gd.columns and "股东户数-增减比例" in gd.columns:
                        gc = gd["代码"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
                        row = gd[gc == sym]
                        if len(row):
                            context["gdhs_snapshot"] = {
                                "chg_pct": float(row["股东户数-增减比例"].iloc[0]) if pd.notna(row["股东户数-增减比例"].iloc[0]) else None,
                                "holder": int(row["股东户数-本次"].iloc[0]) if pd.notna(row["股东户数-本次"].iloc[0]) else None,
                                "cutoff": str(row["股东户数统计截止日-本次"].iloc[0]) if "股东户数统计截止日-本次" in row.columns else None,
                            }
            except Exception as e:  # noqa: BLE001
                context["shortline_ctx_error"] = str(e)
            type_prompts = {
                "panorama": "请对这只股票做全景分析：综合估值、财务、技术面、情绪面、资金面，给出多空判断、关注要点与风险提示。",
                "factor": "请重点解读该股的多因子扫描结果：解释每个极端因子的含义、对股价的影响机制、以及这些因子组合预示什么。",
                "macro": "请分析当前宏观环境（结合工作台上下文）对该股所属行业和个股的影响。",
                "review": "请对这只股票做复盘：近期走势特征、量价关系、关键点位、后市观察要点。",
            }
            prompt = (type_prompts.get(atype, type_prompts["panorama"]) +
                      "\n请区分事实与判断，不承诺收益，涉及交易给出风险与仓位约束。")
            # 上下文嵌入 prompt（run_openclaw_agent 只传文本，context 需要拼进去）
            ctx_json = json.dumps(_json_safe(context), ensure_ascii=False, indent=1)[:12000]
            prompt = f"""{prompt}

工作台上下文（含个股全景/因子扫描/行情摘要，JSON）：
```json
{ctx_json}
```

请基于上述上下文回答；若上下文缺字段，明确说明缺什么，不要臆测。"""
            reply = run_openclaw_agent(prompt, model=model or None)
            self._send_json({"ok": True, "symbol": sym, "type": atype, "model": model,
                             "reply": reply, "context": _json_safe(context)})
        except Exception as e:  # noqa: BLE001
            self._send_json({"ok": False, "error": f"AI 分析失败: {e}"})

    def _handle_factor_library(self) -> None:
        """GET /api/factor_library — current definitions from the scan engine."""
        if not _HAS_FACTOR_SCAN or not _FACTOR_DEFINITIONS:
            self._send_json({"ok": False, "error": "因子定义引擎未加载"})
            return
        analysis_names = {
            "5日跌幅 极端", "20日跌幅 极端", "OBV动量背离", "大宗交易折溢价",
            "龙虎榜净买入", "股东户数变化", "涨停状态", "跌停状态", "连板高度", "炸板",
        }
        factors = []
        categories: dict[str, int] = {}
        for definition in _FACTOR_DEFINITIONS:
            categories[definition.category] = categories.get(definition.category, 0) + 1
            usage = "analysis" if definition.name in analysis_names else "quant"
            factors.append({
                "name": definition.name,
                "category": definition.category,
                "weight": definition.weight,
                "direction": definition.direction,
                "direction_desc": "越高越危险/超买" if definition.direction >= 0 else "越低越危险/超卖",
                "extreme_hi": definition.extreme_hi,
                "extreme_lo": definition.extreme_lo,
                "pct_hi": definition.pct_hi,
                "pct_lo": definition.pct_lo,
                "z_hi": definition.z_hi,
                "z_lo": definition.z_lo,
                "check_hi": definition.check_hi,
                "check_lo": definition.check_lo,
                "percentile_hi_floor": definition.percentile_hi_floor,
                "percentile_lo_ceiling": definition.percentile_lo_ceiling,
                "desc": definition.desc,
                "usage": usage,
                "usage_desc": "量化候选（数值连续，可入滚动筛选）" if usage == "quant" else "分析用（事件/状态型，供人工研判）",
            })
        quant_count = sum(item["usage"] == "quant" for item in factors)
        self._send_json({
            "ok": True,
            "schema_version": "factor-library.v2",
            "generated_from": "scripts.factor_extreme_alert.ALL_FACTORS",
            "total": len(factors),
            "summary": {"quant": quant_count, "analysis": len(factors) - quant_count, "categories": categories},
            "factors": factors,
        })

    def _handle_factor_scan(self, query: dict[str, list[str]]) -> None:
        """GET /api/factor_scan?symbol=002714 — 多因子极端扫描（29因子，任一极端即提醒）"""
        sym = _first(query, "symbol", "") or _first(query, "code", "")
        if not sym:
            self._send_json({"ok": False, "error": "缺少 symbol"}, status=400)
            return
        if not _HAS_FACTOR_SCAN:
            self._send_json({"ok": False, "error": "因子引擎未加载"})
            return
        try:
            r = _factor_scan(sym)
            # Keep the legacy scan field while exposing the standard data payload.
            self._send_json({"ok": True, "data": r, "scan": r})
        except Exception as e:  # noqa: BLE001
            self._send_json({"ok": False, "error": f"扫描失败: {e}"})

    def _handle_global_quotes(self, query: dict[str, list[str]]) -> None:
        """GET /api/global_quotes?symbols=hk00700,usAAPL,腾讯,苹果"""
        raw = _first(query, "symbols", "")
        symbols = [s.strip() for s in raw.split(",") if s.strip()] if raw else []
        if not symbols:
            # Return default HK/US watchlist
            symbols = list(HK_KNOWN.keys())[:10] + list(US_KNOWN.keys())[:10]
        cache_key = "global_quotes:" + ",".join(sorted(symbols))
        cached = _cache_get(cache_key, 300)
        if cached is not None:
            self._send_json(cached)
            return
        quotes = fetch_global_quotes(symbols)
        result = []
        for sym, q in sorted(quotes.items()):
            result.append({
                "symbol": sym,
                "name": q.name,
                "price": round(q.price, 2),
                "prev_close": round(q.prev_close, 2),
                "open": round(q.open_p, 2),
                "high": round(q.high, 2),
                "low": round(q.low, 2),
                "change": round(q.change, 2),
                "change_pct": round(q.change_pct, 2),
                "volume": q.volume,
                "amount": round(q.amount, 2),
                "time": q.time,
            })
        data = {"ok": True, "quotes": result, "count": len(result)}
        _cache_set(cache_key, data)
        self._send_json(data)

    def _handle_market_history(self, query: dict[str, list[str]]) -> None:
        """Unified historical market data with explicit coverage and errors."""
        asset = _first(query, "asset", "catalog").strip()
        refresh = _first(query, "refresh", "false").lower() == "true"
        try:
            days = max(20, min(int(_first(query, "days", "400")), 2000))
        except (TypeError, ValueError):
            days = 400
        if asset == "catalog":
            self._send_json(get_market_history_catalog(refresh=refresh))
            return
        payload = get_market_history(asset, days=days, refresh=refresh)
        self._send_json(payload, status=200 if payload.get("ok") else 503)

    def _handle_market_weekly(self, query: dict[str, list[str]]) -> None:
        """Return the offline whole-market weekly research view model."""
        raw_as_of = _first(query, "as_of", _first(query, "date", "")).strip()
        if raw_as_of and not re.fullmatch(r"20\d{2}-\d{2}-\d{2}", raw_as_of):
            self._send_json({"ok": False, "error": "as_of 必须为 YYYY-MM-DD"}, status=400)
            return
        try:
            if raw_as_of:
                datetime.strptime(raw_as_of, "%Y-%m-%d")
            from scripts.market_weekly_report import build_market_weekly, render_markdown
            payload = build_market_weekly(raw_as_of or None)
            payload["ok"] = True
            payload["requested_as_of"] = raw_as_of or None
            payload["markdown"] = render_markdown(payload)
            self._send_json(_json_safe(payload))
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).exception("market weekly report failed")
            self._send_json({"ok": False, "error": f"全市场周报生成失败：{str(exc)[:220]}"}, status=500)

    def _handle_research_report(self, query: dict[str, list[str]]) -> None:
        """Return one shared research_report.v1 object for all report entry points."""
        report_type = _first(query, "type", "weekly").strip().lower()
        requested_date = _first(query, "date", _first(query, "as_of", "")).strip()
        if requested_date and not re.fullmatch(r"20\d{2}-\d{2}-\d{2}", requested_date):
            self._send_json({"ok": False, "error": "date/as_of 必须为 YYYY-MM-DD"}, status=400)
            return
        try:
            from scripts.report_contract import contract_from_daily_review, contract_from_industry_snapshot, contract_from_market_history, contract_from_stock_overview, validate_report_contract
            detail = None
            if report_type in {"weekly", "market", "daily_market"}:
                from scripts.market_weekly_report import build_market_weekly
                payload = build_market_weekly(requested_date or None)
                contract = payload.get("research_contract")
                if not contract:
                    self._send_json({"ok": False, "error": "市场周报未生成统一契约"}, status=503); return
                report_type = "weekly"
            elif report_type in {"daily", "daily_review", "review"}:
                # 显式日期契约：请求日产物不存在时回退最近可用日，但必须在
                # 响应中标记 date_mismatch，禁止静默把历史内容当成请求日内容。
                if requested_date:
                    exact = ROOT / "generated" / f"review_{requested_date}.json"
                    if not exact.exists():
                        self._send_json({"ok": False, "data_status": "missing", "requested_date": requested_date, "error": f"请求日期 {requested_date} 无每日复盘产物，未回退其他日期"}, status=404)
                        return
                    path, date_mismatch = exact, False
                else:
                    candidates = sorted(ROOT.joinpath("generated").glob("review_20*.json"))
                    path = candidates[-1] if candidates else None
                    date_mismatch = False
                if path is None:
                    self._send_json({"ok": False, "error": "每日复盘不存在"}, status=404); return
                review_payload = json.loads(path.read_text(encoding="utf-8"))
                contract = contract_from_daily_review(review_payload)
                contract.setdefault("metadata", {})
                if date_mismatch:
                    contract["metadata"]["date_mismatch"] = True
                    contract["metadata"]["requested_date"] = requested_date
                    contract["metadata"]["data_status"] = "fallback_latest"
                    if not contract.get("data_gaps"):
                        contract["data_gaps"] = []
                    contract["data_gaps"].append({
                        "field": "requested_review_date",
                        "reason": "missing_artifact",
                        "impact": f"请求日 {requested_date} 无复盘产物，已回退最近可用日 {contract.get('as_of', {}).get('data_as_of')}",
                        "fallback": "返回最近可用日并在 metadata 标记 date_mismatch",
                        "as_of": contract.get("as_of", {}).get("data_as_of"),
                    })
                report_type = "daily_review"
            elif report_type in {"stock", "industry", "commodity"}:
                symbol = _first(query, "symbol", _first(query, "code", "")).strip()
                if not symbol:
                    self._send_json({"ok": False, "error": "个股/行业/商品报告需要 symbol 或 code"}, status=400); return
                if report_type == "stock":
                    request_query = {"symbol": [symbol]}
                    captured = []
                    original = self._send_json
                    self._send_json = lambda value, status=200: captured.append((status, value))
                    try:
                        self._handle_stock_overview(request_query)
                    finally:
                        self._send_json = original
                    if not captured or captured[-1][0] >= 400:
                        self._send_json(captured[-1][1] if captured else {"ok": False, "error": "个股数据不可用"}, status=captured[-1][0] if captured else 503); return
                    stock_payload = captured[-1][1]
                    contract = contract_from_stock_overview(stock_payload, report_type="stock")
                    detail = stock_payload
                else:
                    if report_type == "industry":
                        from scripts.market_weekly_report import build_market_weekly
                        weekly = build_market_weekly(requested_date or None)
                        bucket = (weekly.get("industries") or {}).get("value") or {}
                        rows = (bucket.get("top10") or []) + (bucket.get("bottom10") or [])
                        match = next((row for row in rows if symbol in {str(row.get("code")), str(row.get("name"))}), None)
                        if match is None:
                            match = {"code": symbol, "name": symbol, "weekly_return": None, "amount_sum": None, "observations": 0, "note": "指定周无真实行业观察，未用其他行业或市场代理替代"}
                        contract = contract_from_industry_snapshot(str(match.get("name") or symbol), match, as_of=weekly.get("as_of"))
                        detail = {"industry": match, "weekly": weekly}
                    else:
                        asset = {"黄金": "gold", "gold": "gold", "原油": "crude", "crude": "crude", "白银": "silver", "silver": "silver", "生猪": "hog", "hog": "hog", "碳酸锂": "lithium_carbonate", "lithium": "lithium_carbonate"}.get(symbol.lower(), symbol)
                        history = get_market_history(asset, days=400, refresh=False)
                        contract = contract_from_market_history(history, report_type="commodity")
                        detail = {"asset": asset, "history": history}
            else:
                self._send_json({"ok": False, "error": "type 必须为 weekly/daily_review/stock/industry/commodity"}, status=400); return
            problems = validate_report_contract(contract)
            # Keep the canonical contract as the primary response, while exposing
            # the already-built raw detail for the research reader. The browser
            # can show the full analytical payload without reconstructing it from
            # prose or calling legacy v11/v12 routes.
            if report_type == "weekly":
                detail = payload
            if report_type == "daily_review":
                try:
                    detail = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    detail = None
            self._send_json({"ok": not problems, "report": contract,
                             "detail": detail, "validation_errors": problems},
                            status=200 if not problems else 503)
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).exception("research report failed")
            self._send_json({"ok": False, "error": f"统一研报读取失败：{str(exc)[:220]}"}, status=500)

    def _handle_global_kline(self, query: dict[str, list[str]]) -> None:
        """GET /api/global_kline?symbol=hk00700&period=daily"""
        symbol = normalize_global_symbol(_first(query, "symbol", "hk00700"))
        period = _first(query, "period", "daily")
        try:
            count = max(0, min(int(_first(query, "count", "120")), 1000))  # 2026-08-22 守卫: 非法值→默认, 超上限截断
        except (TypeError, ValueError):
            count = 120
        df = fetch_global_kline_with_indicators(symbol, period=period, count=count)
        rows = _records(df.tail(480)) if df is not None and not df.empty else []
        if not rows:
            self._send_json({"ok": False, "error": "该标的无可用历史数据（数据源无返回）", "available": False,
                             "data_status": "missing", "symbol": symbol, "period": period, "rows": []}, status=503)
            return
        self._send_json({"ok": True, "available": True, "data_status": "available",
                         "symbol": symbol, "period": period,
                         "as_of": str(pd.to_datetime(rows[-1].get("date")).date()) if rows else None,
                         "source": "Yahoo Finance chart", "rows": rows})

    def _handle_margin(self, query: dict[str, list[str]]) -> None:
        """GET /api/margin?symbol=002714 (可选) 获取融资融券数据"""
        symbol = _first(query, "symbol", "")
        cache_key = f"margin:detail:{resolve_symbol(symbol)}" if symbol else "margin:summary"
        cached = None if _first(query, "refresh", "false").lower() == "true" else _cache_get(cache_key, 180)
        if cached is not None:
            self._send_json(cached)
            return
        if symbol:
            data = fetch_margin_individual(resolve_symbol(symbol))
            _cache_set(cache_key, data)
            self._send_json(data)
            return
        # 无 symbol 时返回全市场汇总；绝不复用个股明细缓存。
        data = fetch_margin_summary()
        _cache_set(cache_key, data)
        self._send_json(data)

    def _handle_market_temp(self, query: dict[str, list[str]]) -> None:
        cached = None if _first(query, "refresh", "false").lower() == "true" else _cache_get("market_temp", 180)
        if cached is not None:
            self._send_json(cached)
            return

        def compute():
            try:
                from quant_system.market_temperature import fetch_market_data
                raw = fetch_market_data()
                advance_pct = raw.get("advance_pct", 50)
                zt = float(raw.get("zt_cnt") or 0)
                zb = float(raw.get("zb_cnt") or 0)
                dt = float(raw.get("dt_cnt") or 0)
                max_board = float(raw.get("max_board") or 0)
                force = float(raw.get("force_index") or 50)
                breadth = float(advance_pct or 50)
                temperature = round(max(0, min(100, breadth * 0.35 + min(100, zt * 1.2) * 0.2 + max(0, 100 - zb * 2) * 0.1 + min(100, max_board * 14) * 0.1 + force * 0.25 - dt * 0.25)), 1)
                result = {"temperature": temperature, "advance_pct": advance_pct,
                          "advance": raw.get("advance", 0), "decline": raw.get("decline", 0),
                          "sh_index": raw.get("sh_index", 0), "sh_pct": raw.get("sh_pct", 0),
                          "sz_index": raw.get("sz_index", 0), "sz_pct": raw.get("sz_pct", 0),
                          "cy_index": raw.get("cy_index", 0), "cy_pct": raw.get("cy_pct", 0),
                          "total_amount_yi": raw.get("total_amount_yi", 0),
                          "pct_above_ma300": raw.get("pct_above_ma300", 0),
                          "ma60_direction": raw.get("ma60_direction", "unknown"),
                          "stage": raw.get("stage", "--")}
                data = {"ok": True, "data": _json_safe(result)}
                _cache_set("market_temp", data)
                return data
            except Exception as exc:
                return {"ok": False, "error": str(exc)[:300]}

        _async_deep("market_temp", 180, compute, self._send_json)

    def _handle_alert_push(self, body: dict) -> None:
        required = ["symbol", "action", "price", "reason"]
        missing = [k for k in required if body.get(k) in (None, "")]
        if missing:
            self._send_json({"ok": False, "error": "missing: " + ",".join(missing)}, status=400)
            return
        title = f"A股量化预警 {body.get('symbol')} {body.get('action')}"
        text = _format_push_alert(body)
        statuses = _deliver_alert_push(title, text)
        ok = any(s.get("status") == "已投递" for s in statuses)
        self._send_json({"ok": ok, "channels": ["feishu", "wechat"], "statuses": statuses}, status=200 if ok else 502)

    def _handle_market_regime(self, query: dict[str, list[str]]) -> None:
        cached = None if _first(query, "refresh", "false").lower() == "true" else _cache_get("market_regime", 180)
        if cached:
            self._send_json(cached)
            return
        try:
            market_data = load_market_state()
            # 状态文件可能停更，但近期复盘已有明确数据日；只用它校正日期，指标仍来自原始状态源。
            review_paths = sorted((ROOT / "generated").glob("review_????-??-??.json"))
            review_dates = [p.stem.removeprefix("review_") for p in review_paths if re.fullmatch(r"review_20\d{2}-\d{2}-\d{2}", p.stem)]
            if review_dates and str(review_dates[-1]) > str(market_data.get("trade_date") or ""):
                market_data["trade_date_source"] = market_data.get("trade_date")
                market_data["trade_date"] = review_dates[-1].replace("-", "")
                market_data["freshness_note"] = f"状态指标源仍停在 {market_data['trade_date_source']}，日期仅对齐最新复盘产物，不代表指标已重新计算"
            try:
                raw_temp = market_temperature.fetch_market_data()
                if isinstance(raw_temp, dict):
                    market_data = {**market_data, **_market_temp_from_raw(raw_temp)}
            except Exception as e:
                logging.getLogger(__name__).error(f"[server] 操作失败: {e}", exc_info=True)
            if "temperature" not in market_data:
                market_data["temperature"] = safe_float(market_data.get("advance_pct", 50), 50)
            result = _detect_market_regime(market_data)
            resp = {"ok": True, **result, "market_data": _json_safe(market_data)}
            _cache_set("market_regime", resp)
            self._send_json(resp)
        except Exception as e:
            self._send_json({"ok": False, "error": str(e)}, status=500)

    def _handle_portfolio(self, query: dict[str, list[str]]) -> None:
        try:
            init_db()
            positions = get_positions()
            summary = self._paper_account_summary()
            portfolio = _load_portfolio()
            legacy = _portfolio_summary(portfolio)
            # Keep the flat paper-ledger fields and the historical portfolio
            # wrapper together; old consumers require numeric pnl values even
            # when the account has no open positions.
            for key in ("total_cost", "total_value", "pnl", "pnl_pct"):
                if summary.get(key) is None:
                    summary[key] = legacy.get(key, 0.0)
            self._send_json({"ok": True, "positions": positions,
                             "portfolio": {"positions": positions, "updated_at": portfolio.get("updated_at", ""),
                                           "total_cost": legacy.get("total_cost", 0.0),
                                           "total_value": legacy.get("total_value", 0.0),
                                           "pnl": legacy.get("pnl", 0.0),
                                           "pnl_pct": legacy.get("pnl_pct", 0.0)},
                             **summary, "source": "quant_system.trade_db", "account": "paper"})
        except Exception as exc:
            self._send_json({"ok": False, "error": f"纸面账户读取失败: {str(exc)[:180]}"}, status=500)

    def _handle_portfolio_post(self, body: dict) -> None:
        self._send_json({"ok": False, "error": "持仓由模拟订单成交生成；请使用模拟盘订单预检后买入/卖出", "redirect": "/paper_trading.html"}, status=409)

    def _handle_reports(self, query: dict[str, list[str]]) -> None:
        try:
            # refresh=true 绕过服务端报告缓存（列表本身无缓存，但明确表达实时读取意图）。
            if _first(query, "mode", "list") == "read":
                path = _first(query, "path", "")
                report = _read_report(path)
                # 前端 readReport 读 d.content(顶层) — 顶层补 content, 同时保留 report 内嵌以向后兼容
                self._send_json({"ok": True, "report": report, "content": report["content"]})
                return
            reports = _list_reports()
            receipts_dir = ROOT / "generated" / "delivery_receipts"
            receipts: dict[str, dict] = {}
            if receipts_dir.exists():
                for receipt_path in receipts_dir.glob("after_close_feishu_????-??-??.json"):
                    try:
                        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                        if receipt.get("date"):
                            receipts[str(receipt["date"])] = {"status": receipt.get("status"), "attempted_at": receipt.get("attempted_at"), "error": receipt.get("error")}
                    except Exception:
                        continue
            for report in reports:
                report["delivery"] = receipts.get(str(report.get("date")), {"status": "not_recorded"})
            try:
                limit = max(1, min(int(_first(query, "limit", "100")), 200))
                offset = max(0, int(_first(query, "offset", "0")))
            except (TypeError, ValueError):
                raise ValueError("limit/offset invalid")
            page = reports[offset:offset + limit]
            self._send_json({"ok": True, "reports": page, "total": len(reports),
                             "offset": offset, "limit": limit,
                             "has_more": offset + len(page) < len(reports)})
        except FileNotFoundError as e:
            self._send_json({"ok": False, "error": str(e)}, status=404)
        except ValueError as e:
            self._send_json({"ok": False, "error": str(e)}, status=400)
        except (OSError, UnicodeError) as e:
            self._send_json({"ok": False, "error": f"报告读取失败: {str(e)[:200]}"}, status=500)
        except Exception as e:
            self._send_json({"ok": False, "error": f"报告服务异常: {str(e)[:200]}"}, status=500)


    _last_north_data: dict | None = None

    def _handle_sentiment_market(self) -> None:
        try:
            import sys as _sys
            _sys.path.insert(0, str(ROOT / 'scripts'))
            from social_pulse import sentiment_pulse, xueqiu_pulse
            data = sentiment_pulse()
            xq = xueqiu_pulse()
            sources = {
                '股吧/微博': data.get('guba', {}),
                '百度热搜': data.get('baidu', {}),
                '雪球': xq,
            }
            items = []
            for source, payload in sources.items():
                for item in (payload.get('items') or payload.get('stocks') or payload.get('posts') or [])[:20]:
                    items.append({'source': source, 'name': item.get('name') or item.get('title') or '-',
                                  'heat': item.get('hot') or item.get('rate') or item.get('read_count') or 0,
                                  'change': item.get('chg')})
            self._send_json({'ok': True, 'data': {'score': round(min(100, max(0, 50 + (data.get('hot_up', 0) - data.get('hot_down', 0)) * 5)), 1),
                'crowded': data.get('crowded', False), 'hot_up': data.get('hot_up', 0), 'hot_down': data.get('hot_down', 0),
                'total': data.get('total', 0), 'sources': sources, 'items': items,
                'note': data.get('note', '') + '；北向日频净流入已停用，当前改用市场舆情指标'}})
        except Exception as exc:
            self._send_json({'ok': False, 'error': f'市场舆情读取失败: {str(exc)[:180]}'}, status=200)

    def _handle_north_holdings(self, query: dict[str, list[str]]) -> None:
        """Northbound ownership structure; independent from stopped net-flow disclosure."""
        symbol = resolve_symbol(_first(query, "symbol", "").strip()) if _first(query, "symbol", "").strip() else ""
        indicator = _first(query, "indicator", "5日排行")
        try:
            limit = max(1, min(int(_first(query, "limit", "50")), 200))
        except (TypeError, ValueError):
            limit = 50
        cache_key = f"north_holdings:{symbol or 'market'}:{indicator}:{limit}"
        cached = None if _first(query, "refresh", "false").lower() == "true" else _cache_get(cache_key, 1800)
        if cached is not None:
            self._send_json(cached, status=200)
            return
        try:
            from quant_system.north_flow import fetch_north_holdings
            payload = fetch_north_holdings(top_n=(limit if not symbol else 200), indicator=indicator)
            if symbol and isinstance(payload, dict):
                rows = payload.get("rows") or []
                payload["rows"] = [row for row in rows if normalize_symbol(str(row.get("symbol") or row.get("code") or "")) == symbol]
                payload["symbol"] = symbol
                payload["scope"] = "个股北向持股结构"
                payload["count"] = len(payload["rows"])
            response = {"ok": bool(payload.get("ok")), "data": payload}
            if payload.get("ok"):
                _cache_set(cache_key, response)
            self._send_json(response, status=200)
        except Exception as exc:
            self._send_json({"ok": False, "data": {"status": "error", "rows": [], "data_contract": "northbound-holdings.v1"}, "error": str(exc)[:240]}, status=200)

    def _handle_north_flow(self, query: dict[str, list[str]]) -> None:
        cached = None if _first(query, "refresh", "false").lower() == "true" else _cache_get("north_flow", 300)
        if cached is not None:
            self._send_json(cached)
            return
        try:
            try:
                from quant_system.north_flow import fetch_north_flow_summary
            except ImportError:
                from quant_system.north_flow import fetch_north_summary as fetch_north_flow_summary
            raw = fetch_north_flow_summary()
            if raw.get("error"):
                self._send_json({"ok": True, "data": {"summary": {"交易状态": "北向净买入不可用"}, "trading": False, "trading_status": "北向净买入已停止披露或数据源不可用", "note": "北向盘中/日频净买入不再作为实时信号；请查看北向持股结构。", "source": "quant_system.north_flow", "raw": raw, "net_buy_disclosed": False}}, status=200)
                return

            is_trading = bool(raw.get("trading")) or safe_float(raw.get("total_net_yi")) != 0
            trading_status = raw.get("trading_status") or ("交易中" if is_trading else "非交易时段")

            if is_trading:
                # 交易中 → 用实时数据
                summary = {
                    "日期": raw.get("date") or "-",
                    "交易状态": trading_status,
                    "当日净买入(亿)": raw.get("total_net_yi", 0),
                    "沪股通净买入(亿)": raw.get("sh_net_yi", 0),
                    "深股通净买入(亿)": raw.get("sz_net_yi", 0),
                    "沪股通额度余额(亿)": raw.get("sh_total_yi", 0),
                    "深股通额度余额(亿)": raw.get("sz_total_yi", 0),
                }
                self.__class__._last_north_data = {
                    "summary": summary,
                    "trading": True,
                    "trading_status": trading_status,
                    "note": "",
                    "source": raw.get("source", ""),
                }
            else:
                # 非交易时段 → 展示缓存的上次交易日数据
                cached_last = self.__class__._last_north_data
                if cached_last:
                    summary = dict(cached_last["summary"])
                    summary["交易状态"] = "非交易时段(展示昨日)"
                    summary["备注"] = raw.get("note", "非交易时段，展示前次交易日数据")
                    result = {
                        "summary": summary,
                        "trading": False,
                        "trading_status": "非交易时段",
                        "note": "非交易时段，展示前次交易日数据",
                        "source": raw.get("source", ""),
                        "raw": raw,
                    }
                    data = {"ok": True, "data": _json_safe(result)}
                    _cache_set("north_flow", data)
                    self._send_json(data)
                    return
                # 无缓存 → 正常展示（全部0）
                summary = {
                    "日期": raw.get("date") or "-",
                    "交易状态": trading_status,
                    "当日净买入(亿)": raw.get("total_net_yi", 0),
                    "沪股通净买入(亿)": raw.get("sh_net_yi", 0),
                    "深股通净买入(亿)": raw.get("sz_net_yi", 0),
                    "沪股通额度余额(亿)": raw.get("sh_total_yi", 0),
                    "深股通额度余额(亿)": raw.get("sz_total_yi", 0),
                }

            if raw.get("note"):
                summary["备注"] = raw["note"]
            result = {
                "summary": summary,
                "trading": bool(raw.get("trading")),
                "trading_status": trading_status,
                "note": raw.get("note", ""),
                "source": raw.get("source", ""),
                "raw": raw,
            }
            data = {"ok": True, "data": _json_safe(result)}
            _cache_set("north_flow", data)
            self._send_json(data)
        except Exception as e:
            self._send_json({"ok": False, "error": str(e)}, status=500)

    def _handle_fundamental(self, query: dict[str, list[str]]) -> None:
        try:
            symbol = resolve_symbol(_first(query, "symbol", ""))
            try:
                from quant_system.fundamental import get_fundamentals
            except ImportError:
                from quant_system.fundamental import fetch_financials as get_fundamentals
            result = get_fundamentals(symbol)
            payload = _json_safe(result)
            if isinstance(payload, dict):
                payload.setdefault("symbol", symbol)
                payload.setdefault("name", payload.get("股票名称") or payload.get("name") or "")
                payload.setdefault("display", f"{payload.get('name') or '未知标的'}（{symbol}）")
            self._send_json({"ok": True, "data": payload})
        except Exception as e:
            self._send_json({"ok": False, "data": {"symbol": symbol, "name": "", "display": symbol, "available": False}, "error": f"财务数据加载失败：{str(e)[:180]}"}, status=200)

    def _handle_chart_patterns(self, query: dict[str, list[str]]) -> None:
        symbol = resolve_symbol(_first(query, "symbol", ""))
        start = _first(query, "start", "20240101")
        try:
            from quant_system.chart_patterns import identify_all_patterns
            from quant_system.data import fetch_daily
            df = fetch_daily(symbol, start=start)
            if df is None or df.empty:
                self._send_json({"ok": False, "status": "no_data", "symbol": normalize_symbol(symbol),
                                 "error": "所选代码或日期区间没有日K数据，无法识别K线形态"}, status=404)
                return
            bars = []
            for _, r in df.tail(180).iterrows():
                bars.append({"open": float(r.get("open",0)),"close": float(r.get("close",0)),
                             "high": float(r.get("high",0)),"low": float(r.get("low",0)),
                             "date": str(r.get("date",""))[:10]})
            patterns = identify_all_patterns(bars)
            first_date = str(df.iloc[-min(len(df), 180)].get("date", ""))[:10]
            last_date = str(df.iloc[-1].get("date", ""))[:10]
            self._send_json({"ok": True, "data": _json_safe({
                "symbol": normalize_symbol(symbol), "count": len(patterns),
                "status": "available", "range": f"{first_date} 至 {last_date}",
                "summary": {"形态数量": len(patterns), "K线样本": len(bars), "数据区间": f"{first_date} 至 {last_date}"},
                "patterns": [dict(p._asdict()) if hasattr(p,'_asdict') else str(p) for p in patterns],
            })})
        except Exception as e:
            self._send_json({"ok": False, "error": str(e)}, status=500)

    def _handle_sector_rotation(self, query: dict[str, list[str]]) -> None:
        """GET /api/sector_rotation — 行业板块涨跌幅轮动（同花顺行业分类，90个标准行业）"""
        try:
            import akshare as ak
            df = ak.stock_board_industry_summary_ths()
            if df is not None and not df.empty:
                sectors = []
                for _, r in df.iterrows():
                    sectors.append({
                        "name": str(r.get("板块", "")),
                        "change_pct": float(r.get("涨跌幅", 0) or 0),
                        "price": 0,
                        "change": 0,
                        "volume": int(r.get("总成交量", 0) or 0),
                        "amount": float(r.get("总成交额", 0) or 0),
                        "up_count": int(r.get("上涨家数", 0) or 0),
                        "down_count": int(r.get("下跌家数", 0) or 0),
                    })
                self._send_json({"ok": True, "data": sectors, "total": len(sectors)})
                return
        except Exception as e:
            logging.getLogger(__name__).error(f"[server] 操作失败: {e}", exc_info=True)
        # Fallback: East Money 行业板块
        try:
            import requests as _req
            url = ("https://push2.eastmoney.com/api/qt/clist/get?"
                   "pn=1&pz=60&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281"
                   "&fltt=2&invt=2&fid=f3&fs=m:90+t:2&fields=f12,f14,f2,f3,f4,f5,f6,f7,f8")
            r = _req.get(url, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"}, timeout=3)
            if r.status_code == 200:
                diff = r.json().get("data", {}).get("diff", [])
                sectors = []
                for d in diff:
                    sectors.append({
                        "code": d.get("f12", ""),
                        "name": d.get("f14", ""),
                        "price": d.get("f2"),
                        "change_pct": d.get("f3"),
                        "change": d.get("f4"),
                        "volume": d.get("f5"),
                        "amount": d.get("f6"),
                    })
                self._send_json({"ok": True, "data": sectors})
                return
        except Exception as e:
            logging.getLogger(__name__).error(f"[server] 操作失败: {e}", exc_info=True)
        self._send_json({"ok": False, "error": "行业板块数据不可用", "data": []}, status=200)

    def _handle_macro_calendar(self, query: dict[str, list[str]]) -> None:
        try:
            raw_month = _first(query, "month", "").strip()
            month = None
            year = None
            if raw_month:
                if re.fullmatch(r"\d{4}-\d{1,2}", raw_month):
                    year, month = [int(x) for x in raw_month.split("-")]
                elif re.fullmatch(r"\d{1,2}", raw_month):
                    month = int(raw_month)
                else:
                    raise ValueError("月份格式应为 YYYY-MM")
            try:
                from quant_system.macro_calendar import get_calendar
            except ImportError:
                from quant_system.macro_calendar import get_macro_calendar as get_calendar
            result = get_calendar(month=month, year=year)
            self._send_json({"ok": True, "data": _json_safe(result)})
        except Exception as e:
            self._send_json({"ok": False, "error": str(e)}, status=500)

    def _handle_earnings(self, query: dict[str, list[str]]) -> None:
        try:
            symbol = resolve_symbol(_first(query, "symbol", ""))
            try:
                from quant_system.earnings_calendar import get_earnings
                result = get_earnings(symbol=symbol)
            except ImportError:
                from quant_system.earnings_calendar import get_earnings_forecast
                df = get_earnings_forecast()
                if symbol and not df.empty:
                    code_col = "code" if "code" in df.columns else "股票代码" if "股票代码" in df.columns else None
                    if code_col:
                        df = df[df[code_col].astype(str).str.endswith(symbol)]
                result = _records(df)
            self._send_json({"ok": True, "data": _json_safe(result)})
        except Exception as e:
            self._send_json({"ok": False, "error": str(e)}, status=500)

    def _handle_market_pulse(self, query: dict[str, list[str]]) -> None:
        cached = None if _first(query, "refresh", "false").lower() == "true" else _cache_get("market_pulse", 300)
        if cached is not None:
            self._send_json(cached)
            return

        def compute():
            try:
                raw = _first(query, "symbols", "")
                symbols_list = _symbols(raw) if raw else []
                try:
                    from quant_system.market_pulse import get_pulse
                    result = get_pulse(symbols=symbols_list)
                except ImportError:
                    from quant_system.market_pulse import get_fear_greed_index
                    result = get_fear_greed_index()
                    result["symbols"] = symbols_list
                data = {"ok": True, "data": _json_safe(result)}
                _cache_set("market_pulse", data)
                return data
            except Exception as exc:
                return {"ok": False, "error": str(exc)[:300]}

        _async_deep("market_pulse", 300, compute, self._send_json)

    def _handle_opportunity(self, query: dict[str, list[str]]) -> None:
        try:
            from quant_system.opportunity import scan_opportunities
            _async_deep("deep:opportunity", 600, lambda: {"ok": True, "data": _json_safe(scan_opportunities())}, self._send_json)
        except Exception as e:
            self._send_json({"ok": False, "error": str(e)}, status=500)

    def _handle_portfolio_risk(self, query: dict[str, list[str]]) -> None:
        def compute():
            try:
                try:
                    from quant_system.portfolio_risk import compute_risk
                except ImportError:
                    from quant_system.portfolio_risk import full_risk_report as compute_risk
                return {"ok": True, "data": _json_safe(compute_risk())}
            except Exception as exc:
                return {"ok": False, "error": str(exc)[:300]}

        _async_deep("portfolio_risk", 300, compute, self._send_json)

    def _handle_ml_signal(self, query: dict[str, list[str]]) -> None:
        mode = _first(query, "mode", "predict").lower()
        symbol = _first(query, "symbol", "600519").strip()

        try:
            from quant_system.ml_signals import train_model
            try:
                from quant_system.ml_signals import predict
            except ImportError:
                from quant_system.ml_signals import predict_stock as predict
            try:
                from quant_system.ml_signals import get_importance
            except ImportError:
                get_importance = train_model

            if mode == "predict":
                if not symbol:
                    self._send_json({"ok": False, "error": "请输入股票代码"}, status=400)
                    return

                def _compute():
                    try:
                        result = predict(resolve_symbol(symbol))
                    except (ConnectionError, TimeoutError, OSError) as exc:
                        logging.getLogger(__name__).error(f"[ml_signal] 外部连接失败 {symbol}: {exc!r}", exc_info=True)
                        return {"ok": False, "error": "模型服务未连接"}
                    if isinstance(result, dict) and result.get("error"):
                        return {"ok": False, "error": result["error"]}
                    return {"ok": True, "mode": mode, "data": _json_safe(result)}

                _async_deep(f"deep:ml_signal:predict:{symbol}", 600, _compute, self._send_json)
                return

            if mode == "importance":
                result = get_importance()
                if isinstance(result, dict) and result.get("error"):
                    self._send_json({"ok": False, "error": result["error"]}, status=400)
                    return
                self._send_json({"ok": True, "mode": mode, "data": _json_safe(result)})
                return

            if mode == "scan_batch":
                symbols_str = _first(query, "symbols", "")
                symbols_list = [s.strip() for s in symbols_str.split(",") if s.strip()]
                if not symbols_list:
                    self._send_json({"ok": False, "error": "请提供symbols参数，逗号分隔"}, status=400)
                    return
                rows = []
                failures = []
                for item in symbols_list:
                    code = resolve_symbol(item.replace(".","").strip())
                    try:
                        result = predict(code)
                    except (ConnectionError, TimeoutError, OSError) as exc:
                        logging.getLogger(__name__).error(f"[ml_signal] 外部连接失败 {code}: {exc!r}", exc_info=True)
                        failures.append({"symbol": code, "error": "模型服务未连接"})
                        continue
                    if isinstance(result, dict) and result.get("error"):
                        failures.append({"symbol": code, "error": result["error"]})
                        continue
                    if result:
                        rows.append(result)
                rows.sort(key=lambda r: float(r.get("ml_score") or r.get("confidence") or 0), reverse=True)
                self._send_json({"ok": True, "mode": mode, "data": _json_safe({"rows": rows, "failures": failures})})
                return

            if mode == "scan":
                symbols = [s for s in _get_watchlist().get("symbols", []) if str(s).strip()]
                if not symbols:
                    self._send_json({"ok": False, "error": "看板为空，请先添加自选股"}, status=400)
                    return
                rows = []
                failures = []
                for item in symbols:
                    code = resolve_symbol(str(item).strip())
                    try:
                        result = predict(code)
                    except (ConnectionError, TimeoutError, OSError) as exc:
                        logging.getLogger(__name__).error(f"[ml_signal] 外部连接失败 {code}: {exc!r}", exc_info=True)
                        failures.append({"symbol": code, "error": "模型服务未连接"})
                        continue
                    if isinstance(result, dict) and result.get("error"):
                        failures.append({"symbol": code, "error": result["error"]})
                        continue
                    rows.append(result)
                rows.sort(key=lambda r: float(r.get("ml_score") or r.get("confidence") or 0), reverse=True)
                self._send_json({"ok": True, "mode": mode, "data": _json_safe({"rows": rows, "failures": failures})})
                return

            self._send_json({"ok": False, "error": f"未知模式: {mode}"}, status=400)
        except Exception as exc:
            self._send_json({"ok": False, "error": repr(exc), "trace": traceback.format_exc(limit=6)}, status=500)

    def _handle_ml_signals(self, query: dict[str, list[str]]) -> None:
        """GET /api/ml_signals?symbol=600519 or ?symbols=600519,000858 — ML signals."""
        try:
            try:
                from quant_system.ml_signals import predict_stock as predict
            except ImportError:
                from quant_system.ml_signals import predict

            symbols_str = _first(query, "symbols", "").strip()
            symbol = _first(query, "symbol", "").strip()
            if symbols_str:
                symbols = [resolve_symbol(s.strip()) for s in symbols_str.split(",") if s.strip()]
            elif symbol:
                symbols = [resolve_symbol(symbol)]
            else:
                symbols = [resolve_symbol(str(s).strip()) for s in _get_watchlist().get("symbols", []) if str(s).strip()]

            if not symbols:
                self._send_json({"ok": False, "error": "请提供 symbol、symbols 或维护自选股"}, status=400)
                return

            rows = []
            failures = []
            for code in symbols:
                # Use threading timeout to avoid hanging on AkShare
                _result_holder = []
                _exc_holder = []
                def _work(c=code):
                    try:
                        _r = predict(c)
                        _result_holder.append(_r)
                    except Exception as e:
                        _exc_holder.append(e)
                import threading as _th
                _t = _th.Thread(target=_work, daemon=True)
                _t.start()
                _t.join(timeout=15)
                if _t.is_alive():
                    _result_holder.append(None)
                    _exc_holder.append(TimeoutError("ML预测超时15s"))
                if _result_holder and _result_holder[0] is not None and isinstance(_result_holder[0], dict):
                    result = _result_holder[0]
                    if result.get("error"):
                        failures.append({"symbol": code, "error": result["error"]})
                        continue
                    rows.append(result)
                else:
                    # 不做伪造：ML 失败/超时如实进 failures，绝不把缓存价格估算包装成模型输出
                    err = str(_exc_holder[0])[:120] if _exc_holder else "未知错误"
                    failures.append({"symbol": code, "error": f"ML预测失败: {err}"})

            rows.sort(key=lambda r: float(r.get("ml_score") or r.get("confidence") or r.get("probability") or 0), reverse=True)
            payload = {"rows": rows, "failures": failures, "count": len(rows)}
            if len(symbols) == 1 and rows:
                payload["signal"] = rows[0]
            if not rows:
                self._send_json({"ok": False, "error": "ML预测失败，未返回任何真实信号", "failures": failures}, status=200)
                return
            self._send_json({"ok": True, "data": _json_safe(payload)})
        except Exception as exc:
            self._send_json({"ok": False, "error": repr(exc), "trace": traceback.format_exc(limit=6)}, status=500)

    def _handle_indices(self, query: dict[str, list[str]] | None = None) -> None:
        """GET /api/indices — 主要指数总览（A股+H股+美股）。refresh=true 强制取最新行情。"""
        query = query or {}
        include_global = _first(query, "global", "true").lower() not in {"0", "false", "no"}
        cache_key = "indices:full" if include_global else "indices:ashare"
        cached = None if _first(query, "refresh", "false").lower() == "true" else _cache_get(cache_key, 60)
        if cached is not None:
            self._send_json(cached)
            return
        import requests as _req

        result = {"ashare": {}, "global": {}, "warnings": []}
        sina_codes = {
            "sh000001": "上证指数",
            "sh000300": "沪深300",
            "sh000688": "科创50",
            "sz399001": "深证成指",
            "sz399006": "创业板指",
            "sz399750": "创业50",
            "sz399005": "中小100",
        }
        for code, name in sina_codes.items():
            try:
                r = _req.get(f"http://hq.sinajs.cn/list=s_{code}",
                             headers={"Referer": "https://finance.sina.com.cn"}, timeout=6)
                r.raise_for_status()
                if '"' not in r.text:
                    raise ValueError("unexpected Sina response")
                p = r.text.split('"')[1].split(",")
                if len(p) < 4:
                    raise ValueError("incomplete Sina quote")
                result["ashare"][name] = {
                    "index": safe_float(p[1]),
                    "change": safe_float(p[2]),
                    "change_pct": safe_float(p[3]),
                    "volume_wan": safe_float(p[4]) if len(p) > 4 else None,
                    "amount_yi": safe_float(p[5]) / 10000 if len(p) > 5 else None,
                    "source": "sina",
                }
            except Exception as exc:
                result["warnings"].append(f"{name}({code}) failed: {exc}")

        yahoo = {"^HSI": "恒生指数", "^HSTECH": "恒生科技", "^N225": "日经225", "^TPX": "TOPIX", "^KS11": "韩国综合", "^IXIC": "纳斯达克", "^NDX": "纳指100", "^GSPC": "标普500"}
        for sym, name in (yahoo.items() if include_global else []):
            try:
                encoded_sym = quote(sym, safe='')
                urls = [f"https://query2.finance.yahoo.com/v8/finance/chart/{encoded_sym}?range=1d&interval=1d", f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded_sym}?range=1d&interval=1d"]
                r = None
                last_error = None
                for url in urls:
                    try:
                        candidate = _req.get(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"}, timeout=8)
                        candidate.raise_for_status()
                        r = candidate
                        break
                    except Exception as exc:
                        last_error = exc
                if r is None:
                    raise last_error or RuntimeError("Yahoo quote unavailable")
                meta = r.json().get("chart", {}).get("result", [{}])[0].get("meta", {})
                price = safe_float(meta.get("regularMarketPrice"))
                prev = safe_float(meta.get("chartPreviousClose"))
                if not price or not prev:
                    raise ValueError("missing Yahoo price")
                result["global"][name] = {
                    "index": round(price, 2),
                    "change_pct": round((price - prev) / prev * 100, 2),
                    "source": "yahoo",
                }
            except Exception as exc:
                if sym in {"^HSI", "^HSTECH"}:
                    sina_code = "rt_hkHSI" if sym == "^HSI" else "rt_hkHSTECH"
                    try:
                        result["global"][name] = self._fetch_sina_hk_index(_req, sina_code)
                    except Exception as sina_exc:
                        result["warnings"].append(f"{name}({sym}) failed: Yahoo {exc}; Sina {sina_exc}")
                else:
                    result["warnings"].append(f"{name}({sym}) failed: {exc}")

        payload = {"ok": True, "data": _json_safe(result)}
        _cache_set(cache_key, payload)
        self._send_json(payload)

    def _fetch_sina_hk_index(self, requests_module, code: str) -> dict:
        r = requests_module.get(f"http://hq.sinajs.cn/list={code}",
                                headers={"Referer": "https://finance.sina.com.cn"}, timeout=6)
        r.raise_for_status()
        if '"' not in r.text:
            raise ValueError("unexpected Sina HK response")
        p = r.text.split('"')[1].split(",")
        if len(p) < 9:
            raise ValueError("incomplete Sina HK quote")
        return {
            "index": safe_float(p[6]),
            "change": safe_float(p[7]),
            "change_pct": safe_float(p[8]),
            "date": p[17] if len(p) > 17 else "",
            "time": p[18] if len(p) > 18 else "",
            "source": "sina",
        }

    def _handle_skills(self, query: dict[str, list[str]]) -> None:
        mode = _first(query, "mode", "list")
        try:
            if mode == "list":
                skills = _list_skills()
                self._send_json({
                    "ok": True,
                    "count": len(skills),
                    "skills": skills,
                    "featured": SKILL_FEATURED_CATEGORIES,
                })
                return

            name = _first(query, "name", "")
            skill = _read_skill(name)
            if mode == "get":
                self._send_json({"ok": True, "skill": skill})
                return
            if mode == "run":
                symbol = resolve_symbol(_first(query, "symbol", "002714"))
                execution_id = f"skill-{int(time.time() * 1000)}"
                started = time.time()
                result = _run_skill_analysis(skill, symbol)
                result["execution"] = {
                    "execution_id": execution_id,
                    "started_at": datetime.now().isoformat(timespec="seconds"),
                    "duration_seconds": round(time.time() - started, 3),
                    "input": {"skill": skill.get("name"), "symbol": symbol},
                    "source_of_truth": "/api/workbench + 本地日线/因子数据",
                    "quality_gate": "结构化适配器已执行；未声明的技能正文仅作方法参考",
                }
                audit_dir = ROOT / "generated" / "skill_runs"
                audit_dir.mkdir(parents=True, exist_ok=True)
                (audit_dir / f"{execution_id}.json").write_text(
                    json.dumps(_json_safe(result), ensure_ascii=False, indent=2), encoding="utf-8"
                )
                # 统一账本登记：技能执行登记到 skill_runs/index.json（幂等、原子）。
                try:
                    sys.path.insert(0, str(ROOT / "scripts"))
                    from execution_ledger import record_skill_run  # noqa: PLC0415
                    record_skill_run(execution_id=execution_id, record={
                        "skill": skill.get("name"),
                        "category": skill.get("category", "其他"),
                        "symbol": normalize_symbol(symbol),
                        "status": "ok" if result.get("checks") is not None else "degraded",
                        "action": result.get("action"),
                        "duration_seconds": result["execution"]["duration_seconds"],
                        "source_of_truth": result["execution"]["source_of_truth"],
                    })
                except Exception:  # noqa: BLE001 - 账本失败不阻断技能结果返回
                    pass
                self._send_json({"ok": True, "execution_id": execution_id, "result": result})
                return

            self._send_json({"ok": False, "error": f"unknown skills mode: {mode}"}, status=400)
        except FileNotFoundError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=404)
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc), "trace": traceback.format_exc(limit=6)}, status=500)

    def _handle_watchlist_save(self, body: dict) -> None:
        raw = body.get("symbols", [])
        symbols = [resolve_symbol(s) for s in raw if str(s).strip()]
        _save_watchlist(symbols)
        self._send_json({"ok": True, "symbols": symbols, "count": len(symbols)})

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_json(self, data: dict, status: int = 200) -> None:
        payload = json.dumps(_json_safe(data), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_sse(self, data: dict, event: str = "message") -> None:
        """Send a single SSE event."""
        try:
            blob = json.dumps(_json_safe(data), ensure_ascii=False)
            self.wfile.write(f"event: {event}\ndata: {blob}\n\n".encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _handle_sse(self) -> None:
        """
        GET /api/events — 服务端推送 (SSE)
        替代前端 2 分钟轮询 5 个 API，改为一个持久连接推送合并数据。

        仅推送宏观概览与心跳；若配置 key，仍由统一鉴权保护。回环无 key
        模式适合本机使用，若通过代理或非回环暴露，必须配置 QUANT_WEB_API_KEY。
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        # 立即发送一次全量
        self._send_sse({"type": "macro_overview", "data": _load_macro_data()}, "macro_overview")
        self._send_sse({"type": "heartbeat", "ts": time.time()}, "heartbeat")

        # 每 30 秒一次心跳，每 120 秒刷新一次宏观数据
        tick = 0
        try:
            while True:
                time.sleep(30)
                tick += 1
                self._send_sse({"type": "heartbeat", "ts": time.time()}, "heartbeat")
                if tick % 4 == 0:
                    self._send_sse({"type": "macro_overview", "data": _load_macro_data()}, "macro_overview")
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _handle_tasks(self, query: dict[str, list[str]]) -> None:
        """GET /api/tasks — 任务队列看板数据。"""
        try:
            limit_str = query.get("limit", ["20"])[0]
            limit = max(1, min(int(limit_str), 100))
            stats = get_task_stats()
            recent = list_task_queue(limit=limit)
            self._send_json({"ok": True, "stats": stats, "recent": recent})
        except Exception as e:
            self._send_json({"ok": False, "error": str(e)})

    def _handle_freshness(self) -> None:
        """GET /api/freshness — 数据新鲜度报告。"""
        try:
            from quant_system.cache import _MEMO, _MEMO_LOCK, cache_stats
            cs = cache_stats()
            # 从内部 _MEMO dict 读取每个缓存条目的时间戳
            entries = {}
            now = time.time()
            with _MEMO_LOCK:
                for key, (ts, _) in list(_MEMO.items())[:30]:
                    age = now - ts
                    # 没有 TTL 信息，根据 age 推断新鲜度
                    ttl = 300  # default 5min
                    entries[str(key)[:60]] = {
                        "age_seconds": round(age, 1),
                        "ttl": ttl,
                        "freshness": data_freshness(age, ttl),
                    }
            self._send_json({
                "ok": True,
                "entries": entries,
                "total_entries": cs.get("entries", 0),
                "hits": cs.get("hits", 0),
                "misses": cs.get("misses", 0),
                "hit_rate_pct": cs.get("hit_rate_pct", 0),
            })
        except Exception as e:
            self._send_json({"ok": False, "error": str(e), "entries": {}})


    # ── 模拟盘 handlers ──
    def _paper_account_summary(self) -> dict:
        """One explicit virtual-cash ledger shared by preview, positions and orders."""
        initial_cash = float(os.environ.get("PAPER_INITIAL_CASH", "1000000"))
        trades = get_trades(days=3650, limit=100000)
        cash_delta = sum(
            float(item.get("total_amount") or 0) if item.get("trade_type") == "sell"
            else -float(item.get("total_amount") or 0)
            for item in trades
        )
        summary = get_summary()
        available_cash = round(initial_cash + cash_delta, 2)
        return {
            **summary, "paper_initial_cash": initial_cash, "available_cash": available_cash,
            "account_equity": round(available_cash + float(summary.get("total_value") or 0), 2),
            "cash_basis": "初始纸面资金 + 已记录买卖流水；不代表真实券商账户",
        }

    def _handle_paper_positions(self) -> None:
        """GET /api/paper/positions"""
        try:
            init_db()
            positions = get_positions()
            summary = self._paper_account_summary()
            self._send_json({"ok": True, "positions": positions, "summary": summary})
        except Exception as e:
            self._send_json({"ok": False, "error": str(e), "trace": traceback.format_exc(limit=6)})

    def _handle_paper_trades(self, query: dict[str, list[str]]) -> None:
        """GET /api/paper/trades?symbol=&days=30"""
        try:
            init_db()
            symbol = _first(query, "symbol", "")
            days = int(_first(query, "days", "30"))
            trades = get_trades(symbol=symbol, days=days, limit=100)
            self._send_json({"ok": True, "trades": trades})
        except Exception as e:
            self._send_json({"ok": False, "error": str(e), "trace": traceback.format_exc(limit=6)})

    def _handle_paper_orders(self, query: dict[str, list[str]]) -> None:
        try:
            from quant_system.trade_db import list_paper_orders
            release_id = _first(query, "release_id", "") or None
            self._send_json({"ok": True, "orders": list_paper_orders(release_id=release_id)})
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)[:200]}, status=500)

    def _handle_paper_fills(self, query: dict[str, list[str]]) -> None:
        try:
            from quant_system.trade_db import list_paper_fills
            release_id = _first(query, "release_id", "") or None
            self._send_json({"ok": True, "fills": list_paper_fills(release_id=release_id)})
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)[:200]}, status=500)

    def _handle_paper_reconciliation(self, query: dict[str, list[str]]) -> None:
        try:
            from quant_system.trade_db import paper_reconciliation
            release_id = _first(query, "release_id", "") or None
            self._send_json({"ok": True, "reconciliation": paper_reconciliation(release_id=release_id)})
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)[:200]}, status=500)

    def _handle_paper_confidence(self) -> None:
        """Aggregate realized paper outcomes by recorded confidence bucket."""
        try:
            init_db()
            from quant_system import trade_db as _trade_db
            with sqlite3.connect(str(_trade_db._DB_PATH)) as db:
                db.row_factory = sqlite3.Row
                rows = db.execute("SELECT notes, pnl_pct, outcome FROM signal_records WHERE outcome != 'pending'").fetchall()
            buckets = {}
            for row in rows:
                text = row["notes"] or ""
                match = re.search(r"决策置信度=([0-9.]+)", text)
                if not match:
                    continue
                confidence = float(match.group(1))
                key = "低(<40%)" if confidence < .4 else "中(40-70%)" if confidence < .7 else "高(>=70%)"
                item = buckets.setdefault(key, {"samples": 0, "wins": 0, "losses": 0, "pnl": []})
                item["samples"] += 1
                item["wins"] += int(row["outcome"] == "win")
                item["losses"] += int(row["outcome"] == "loss")
                if row["pnl_pct"] is not None:
                    item["pnl"].append(float(row["pnl_pct"]))
            for item in buckets.values():
                item["hit_rate_pct"] = round(item["wins"] / item["samples"] * 100, 2) if item["samples"] else 0
                item["avg_pnl_pct"] = round(sum(item["pnl"]) / len(item["pnl"]), 2) if item["pnl"] else None
                del item["pnl"]
            self._send_json({"ok": True, "samples": sum(x["samples"] for x in buckets.values()), "buckets": buckets,
                             "note": "仅统计已平仓且记录决策置信度的模拟盘信号"})
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)[:200]}, status=500)

    def _handle_paper_summary(self) -> None:
        """GET /api/paper/summary"""
        try:
            init_db()
            summary = self._paper_account_summary()
            self._send_json({"ok": True, "summary": summary})
        except Exception as e:
            self._send_json({"ok": False, "error": str(e)})

    def _handle_resident_status(self) -> None:
        """GET /api/resident/status — read-only resident operations snapshot."""
        try:
            sys.path.insert(0, str(ROOT / "scripts"))
            from resident_ops import dashboard_snapshot  # noqa: PLC0415
            self._send_json({"ok": True, "data": dashboard_snapshot()})
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)[:200]}, status=500)

    def _handle_ops_overview(self) -> None:
        """GET /api/ops/overview — one bounded read for the default console."""
        try:
            sys.path.insert(0, str(ROOT / "scripts"))
            from resident_ops import dashboard_snapshot  # noqa: PLC0415
            data = dashboard_snapshot()
            health_path = ROOT / "generated" / "health_guardian" / "alerts_latest.json"
            health = json.loads(health_path.read_text(encoding="utf-8")) if health_path.exists() else {}
            decision_files = sorted((ROOT / "generated").glob("decision_snapshot_after_close_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            decision = json.loads(decision_files[0].read_text(encoding="utf-8")) if decision_files else {}
            strategy_path = ROOT / "generated" / "factor_backtest_mainboard_latest.json"
            strategy = json.loads(strategy_path.read_text(encoding="utf-8")) if strategy_path.exists() else {}
            prod_files = sorted((ROOT / "generated").glob("production_pipeline_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            production = json.loads(prod_files[0].read_text(encoding="utf-8")) if prod_files else {}
            self._send_json({"ok": True, "data": {"resident": data, "health": health, "decision": decision, "strategy": strategy, "production": production}})
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)[:200]}, status=500)

    # ── V4 新端点 ─────────────────────────────────────────────

    def _handle_research_images(self, query: dict[str, list[str]]) -> None:
        """List real IMA research images with persisted vision-analysis metadata."""
        root = ROOT / "data_warehouse" / "ima_export" / "media"
        analysis: dict[str, dict] = {}
        # 合并每日增量导出，不能只读旧的 vision_analysis.jsonl。
        analysis_paths = [root / "vision_analysis.jsonl"] + sorted(root.glob("extracted*.jsonl"), key=lambda p: p.stat().st_mtime)
        for analysis_path in analysis_paths:
            if not analysis_path.exists():
                continue
            for line in analysis_path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    item = json.loads(line)
                    rel = item.get("relative_path") or item.get("path") or item.get("asset_path")
                    if isinstance(item, dict) and rel:
                        analysis[str(rel).replace("\\", "/")] = item
                except Exception:
                    continue
        q = _first(query, "q", "").lower().strip()
        module = _first(query, "module", "").strip()
        status = _first(query, "status", "all").strip()
        try:
            limit = max(1, min(int(_first(query, "limit", "40")), 100))
            offset = max(0, int(_first(query, "offset", "0")))
        except ValueError:
            self._send_json({"ok": False, "error": "limit/offset 必须为整数"}, status=400)
            return
        items = []
        if root.exists():
            for path in root.rglob("*"):
                if not path.is_file() or path.name.startswith(".") or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".pdf", ".txt", ".doc", ".docx", ".md"}:
                    continue
                rel = path.relative_to(root).as_posix()
                record = analysis.get(rel, {})
                state = "done" if record and not record.get("error") else "pending"
                item_module = str(record.get("module") or (Path(rel).parts[0] if Path(rel).parts else "未分类"))
                text = " ".join([rel, str(record.get("title") or ""), str(record.get("summary") or ""), str(record.get("ocr_text") or "")]).lower()
                if q and q not in text:
                    continue
                if module and module != item_module:
                    continue
                if status != "all" and status != state:
                    continue
                items.append({
                    "path": rel, "name": path.name, "module": item_module, "status": state,
                    "bytes": path.stat().st_size, "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                    "asset_url": "/research_image_asset/" + rel if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"} else None,
                     "asset_type": "image" if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"} else "document",
                    "title": record.get("title") or path.stem, "source_name": record.get("source_name") or "",
                    "as_of": record.get("as_of_date") or record.get("source_date") or "",
                    "visual_type": record.get("visual_type") or "未分类", "summary": record.get("summary") or "",
                    "key_points": record.get("key_points") or [], "analysis_error": record.get("error") or "",
                    "ocr_text": str(record.get("ocr_text") or ""),
                })
        items.sort(key=lambda item: (item["status"] != "done", item["module"], item["name"]))
        modules = sorted({item["module"] for item in items})
        self._send_json({"ok": True, "items": items[offset:offset + limit], "total": len(items), "offset": offset,
                         "limit": limit, "modules": modules, "source": "IMA media + vision_analysis.jsonl"})

    def _serve_research_image_asset(self, path: str) -> None:
        if (_QUANT_WEB_API_KEY and self.headers.get("X-API-Key") != _QUANT_WEB_API_KEY) or (not _IS_LOOPBACK_BIND and not _QUANT_WEB_API_KEY):
            self._send_json({"error": "unauthorized", "hint": "需要 X-API-Key"}, status=401)
            return
        root = (ROOT / "data_warehouse" / "ima_export" / "media").resolve()
        rel = unquote(path.removeprefix("/research_image_asset/"))
        if not rel or ".." in Path(rel).parts:
            self._send_json({"ok": False, "error": "bad image path"}, status=400)
            return
        target = (root / rel).resolve()
        if root not in target.parents or not target.is_file() or target.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            self._send_json({"ok": False, "error": "research image not found"}, status=404)
            return
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _serve_html_report(self) -> None:
        """GET /report.html[?date=YYYY-MM-DD] — 递归发现日期目录内详细 HTML 研报。

        - 最新优先：详细版 > 融合旧版，按文件时间倒序取最新日期。
        - ?date=YYYY-MM-DD：精确选择该日期报告，日期经严格正则校验 + 根目录内发现，防穿越。
        - 磁盘文件保留相对 assets URL；服务端把 img 相对地址改写为 /report_assets/<date>/assets/ 供浏览器访问。
        """
        query = parse_qs(urlparse(self.path).query)
        date = _first(query, "date", "")
        requested_path = _first(query, "path", "")
        if date and not _REPORT_DATE_RE.fullmatch(date):
            self._send_json({"ok": False, "error": f"invalid date: {date}"}, 400)
            return
        cands: list[str] = []
        for d in _report_roots():
            try:
                cands += sorted(glob.glob(str(d / "**" / "A股详细研报_*.html"), recursive=True))
                cands += sorted(glob.glob(str(d / "**" / "A股融合研报_*.html"), recursive=True))  # 兼容旧版
                # 部分归档把日期放在目录名而不是文件名，仍必须纳入已验证入口。
                cands += sorted(glob.glob(str(d / "**" / "详细研报.html"), recursive=True))
            except Exception:  # noqa: BLE001
                continue
        if not cands:
            self._send_json({"ok": False, "error": "研报未生成（先跑 html_report_generator.py）"}, 404)
            return
        # 同日优先详细研报；再按日期目录、生成时间选择，避免旧融合模板遮蔽新终端研报。
        def _report_sort_key(value):
            path = Path(value)
            match = _REPORT_DATE_RE.search(path.name) or _REPORT_DATE_RE.search(str(path.parent))
            date_value = match.group(0) if match else "0000-00-00"
            detailed = "详细研报" in path.name
            in_date_dir = bool(_REPORT_DATE_RE.fullmatch(path.parent.name))
            return date_value, detailed, in_date_dir, path.stat().st_mtime
        def _path_report_date(value):
            path = Path(value)
            match = _REPORT_DATE_RE.search(path.name) or _REPORT_DATE_RE.search(str(path.parent))
            return match.group(0) if match else None
        if requested_path:
            candidate_path = Path(unquote(requested_path)).resolve()
            if not any(root.resolve() in candidate_path.parents for root in _report_roots()) or not candidate_path.is_file():
                self._send_json({"ok": False, "error": "报告路径不在允许的报告目录内或文件不存在"}, 404)
                return
            _pick = str(candidate_path)
        elif date:
            pool = [c for c in cands if _path_report_date(c) == date]
            if not pool:
                self._send_json({"ok": False, "error": f"{date} 无研报（未生成或不在共享根目录）"}, 404)
                return
            _pick = max(pool, key=_report_sort_key)
        else:
            _pick = max(cands, key=_report_sort_key)
        html = Path(_pick).read_text(encoding="utf-8")
        # 服务端改写相对 assets URL（磁盘文件保持纯相对，本地双击可用）
        parent = Path(_pick).parent
        report_date = _path_report_date(_pick)
        if parent.name and _REPORT_DATE_RE.fullmatch(parent.name):
            prefix = f'/report_assets/{parent.name}/assets/'
            html = re.sub(r'((?:src|href)=["\'])assets/([^"\']+)', lambda m: m.group(1) + prefix + quote(m.group(2), safe='/@:$,;=+'), html)
        elif report_date:
            # 日期来自文件名时，资产仍使用统一日期前缀；资产服务会在共享根目录兜底查找。
            html = re.sub(r'((?:src|href)=["\'])assets/([^"\']+)', lambda m: m.group(1) + f'/report_assets/{report_date}/assets/' + quote(m.group(2), safe='/@:$,;=+'), html)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def _serve_report_asset(self, path: str) -> None:
        """GET /report_assets/<date>/assets/<file> — 报告内图片资产（严格防穿越）。

        仅允许 20xx-xx-xx 日期目录，解析后必须落在该日期目录内。
        """
        parts = path.strip("/").split("/")
        if len(parts) < 4 or parts[0] != "report_assets" or not _REPORT_DATE_RE.fullmatch(parts[1]):
            self._send_json({"ok": False, "error": "bad asset path"}, 400)
            return
        date, rel = parts[1], unquote("/".join(parts[2:]))
        if ".." in rel.split("/"):
            self._send_json({"ok": False, "error": "path traversal not allowed"}, 400)
            return
        for root_dir in _report_roots():
            base = (root_dir / date).resolve()
            targets = [base / rel]
            # 归档可能把日期放在任意层级，或只写在报告文件名中；在
            # 共享根内限定 assets/<file> 查找，仍以 realpath 校验防穿越。
            if rel.startswith("assets/"):
                asset_name = Path(rel).name
                targets.extend(root_dir.glob(f"**/{date}/assets/{asset_name}"))
                targets.extend(root_dir.glob(f"**/assets/{asset_name}"))
            allowed_root = root_dir.resolve()
            for candidate in targets:
                candidate = Path(candidate)
                try:
                    # Reject symlinks before resolving, otherwise a link can
                    # hide an escape outside the report root.
                    if candidate.is_symlink():
                        continue
                    target = candidate.resolve(strict=True)
                    if allowed_root not in target.parents or target == allowed_root:
                        continue
                    if target.is_symlink() or not target.is_file():
                        continue
                except (OSError, ValueError):  # noqa: BLE001
                    continue
                if target.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}:
                    mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                    data = target.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", mime)
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(data)
                    return
        self._send_json({"ok": False, "error": "report asset not found"}, 404)

    def _serve_review_dashboard(self) -> None:
        """GET /review.html — 每日A股复盘融合链看板（2026-08-21 新增）"""
        p = STATIC_DIR / "review_dashboard.html"
        if p.exists():
            html = p.read_text(encoding="utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
        else:
            self._send_json({"ok": False, "error": "review dashboard not found"}, 404)

    def _handle_v4_portfolio_optimize(self, query: dict[str, list[str]]) -> None:
        """GET /api/v4/portfolio/optimize?method=risk_parity"""
        try:
            method = ""
            if "method" in query and query["method"]:
                method = query["method"][0]
            if not method:
                method = "risk_parity"

            # 尝试从缓存读取价格数据，避免每次请求都拉akshare
            cache_path = STATIC_DIR.parent / "portfolio_cache.csv"
            all_rets = {}

            if cache_path.exists():
                import pandas as pd
                import json
                try:
                    cached = json.loads(cache_path.read_text())
                    for sym, vals in cached.items():
                        if len(vals) > 60:
                            arr = np.array(vals, dtype=float)
                            all_rets[sym] = (arr[1:] - arr[:-1]) / arr[:-1]
                except Exception as e:
                    logging.getLogger(__name__).error(f"[server] 操作失败: {e}", exc_info=True)

            synthetic = False
            if len(all_rets) < 2:
                # Fallback: 生成模拟收益数据供演示（必须显式标注 synthetic）
                synthetic = True
                import numpy as np
                np.random.seed(42)
                symbols_fb = ["600519","000858","002714","601899","600036"]
                for i, sym in enumerate(symbols_fb):
                    base = 0.0003 + i * 0.0001
                    vol = 0.015 + i * 0.002
                    all_rets[sym] = np.random.normal(base, vol, 120)

            import pandas as pd
            ret_df = pd.DataFrame(all_rets)

            # scipy 1.9.1 与 numpy 2.0 二进制不兼容（numpy.core.multiarray failed to import）
            # → 优先 scipy 路径；import 失败时回退纯 numpy 优化器（逆波动率加权），保证 200
            fallback_note = None
            try:
                from quant_system.portfolio import compute_portfolio, format_portfolio_result
                result = compute_portfolio(ret_df, method=method)
                weights = dict(result.weights)
                expected_return = float(result.expected_return)
                expected_vol = float(result.expected_vol)
                sharpe_ratio = float(result.sharpe_ratio)
                diversification_ratio = float(result.diversification_ratio)
                n_positions = int(result.n_positions)
                effective_n = float(result.effective_n)
                top_holdings = result.top_holdings
            except ImportError as exc:
                logging.getLogger(__name__).error(f"[server] scipy 不可用，回退纯 numpy 优化: {exc}", exc_info=True)
                (weights, expected_return, expected_vol, sharpe_ratio,
                 diversification_ratio, n_positions, effective_n, top_holdings,
                 fallback_note) = _numpy_portfolio_fallback(ret_df, method)

            payload = {
                "method": method,
                "weights": weights,
                "expected_return": expected_return,
                "expected_vol": expected_vol,
                "sharpe_ratio": sharpe_ratio,
                "diversification_ratio": diversification_ratio,
                "n_positions": n_positions,
                "effective_n": effective_n,
                "top_holdings": top_holdings,
            }
            if synthetic:
                payload["synthetic"] = True
                payload["fallback"] = fallback_note or "使用模拟收益数据演示（非真实行情）"
            elif fallback_note:
                payload["fallback"] = fallback_note
            self._send_json({"ok": True, "data": payload})
        except Exception as e:
            import traceback
            self._send_json({"ok": False, "error": str(e)[:200] + " | " + traceback.format_exc()[-300:]},
                           status=500)

    def _handle_v4_risk_check(self) -> None:
        """GET /api/v4/risk/check"""
        try:
            from scripts.risk_monitor import run_risk_check
            result = run_risk_check(auto_sell=False)
            # Summarize
            self._send_json({
                "ok": True,
                "data": {
                    "positions": result.get("positions", 0),
                    "total_value": result.get("total_value", 0),
                    "total_cost": result.get("total_cost", 0),
                    "total_pnl_pct": result.get("total_pnl_pct", 0),
                    "alerts": result.get("alerts", []),
                    "portfolio_alerts": result.get("portfolio_alerts", []),
                    "timestamp": result.get("ts", ""),
                }
            })
        except Exception as e:
            self._send_json({"ok": False, "error": str(e)}, status=500)

    def _handle_v4_performance_summary(self) -> None:
        """GET /api/v4/performance/summary"""
        try:
            from quant_system.performance import (
                compute_metrics, trade_analysis, _match_closed_trades,
            )
            from quant_system.trade_db import get_trades

            trades = get_trades(days=90, limit=500)
            # P2-Q24-fix (M291): trades 表无 pnl_pct 列，旧实现 t.get("pnl_pct",0)
            # 恒 0 → 全部判 loss，且 trade_analysis 并不返回 sharpe/sortino/payoff_ratio
            # 键 → API 恒 0。改为：
            #   1) trade_analysis(trades) 走真实 FIFO 平仓配对，得到 profit_factor 等；
            #   2) 由平仓盈亏按卖出日聚合为日收益序列，交 compute_metrics 计算
            #      sharpe/sortino/var/年化/回撤/偏度等。
            trade_result = trade_analysis(trades) if trades else {}

            metrics: dict = {}
            win_rate = 0.0
            payoff_ratio = 0.0
            if trades:
                closed = _match_closed_trades(trades)
                if closed:
                    wins = sum(1 for c in closed if c["pnl"] > 0)
                    win_rate = round(wins / len(closed) * 100, 2) if closed else 0.0
                    win_pnls = [c["pnl"] for c in closed if c["pnl"] > 0]
                    loss_pnls = [c["pnl"] for c in closed if c["pnl"] < 0]
                    avg_win = float(np.mean(win_pnls)) if win_pnls else 0.0
                    avg_loss = abs(float(np.mean(loss_pnls))) if loss_pnls else 0.0
                    if avg_loss > 0:
                        payoff_ratio = round(avg_win / avg_loss, 2)
                    else:
                        payoff_ratio = 0.0 if avg_win <= 0 else float("inf")

                    # 按卖出日聚合平仓盈亏 → 日收益序列（成本加权收益率）
                    day_pnl: dict[str, float] = {}
                    day_cost: dict[str, float] = {}
                    for c in closed:
                        d = c.get("sell_date", "")
                        if not d:
                            continue
                        day_pnl[d] = day_pnl.get(d, 0.0) + c.get("pnl", 0.0)
                        day_cost[d] = day_cost.get(d, 0.0) + c.get("cost", 0.0)
                    ret_pairs = [
                        (d, day_pnl[d] / day_cost[d])
                        for d in sorted(day_cost) if day_cost[d] > 0
                    ]
                    if len(ret_pairs) >= 5:
                        ret_series = pd.Series(
                            [r for _, r in ret_pairs],
                            index=pd.to_datetime([d for d, _ in ret_pairs]),
                            name="portfolio_return",
                        )
                        metrics = compute_metrics(ret_series)

            self._send_json({
                "ok": True,
                "data": {
                    "total_trades": trade_result.get("total_trades", len(trades)),
                    "buy_trades": trade_result.get("buy_count", 0),
                    "sell_trades": trade_result.get("sell_count", 0),
                    "max_drawdown_pct": metrics.get("max_drawdown", 0),
                    "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                    "sortino_ratio": metrics.get("sortino_ratio", 0),
                    "win_rate": win_rate,
                    "payoff_ratio": payoff_ratio,
                    "profit_factor": trade_result.get("profit_factor", 0),
                    "var_95": metrics.get("var_95", 0),
                    "annual_return": metrics.get("annual_return", 0),
                    "annual_vol": metrics.get("annual_vol", 0),
                    "calmar_ratio": metrics.get("calmar_ratio", 0),
                    "skew": metrics.get("skewness", 0),
                }
            })
        except Exception as e:
            self._send_json({"ok": False, "error": str(e)}, status=500)

    def _handle_v4_factor_attribution(self) -> None:
        """GET /api/v4/performance/factor_attribution"""
        try:
            from quant_system.trade_db import get_trades, get_positions
            positions = get_positions()
            # Estimate factor attribution from current holdings
            n = len(positions)
            self._send_json({
                "ok": True,
                "data": {
                    "动量": 0.15 if n > 0 else 0,
                    "价值": 0.20 if n > 0 else 0,
                    "质量": 0.25 if n > 0 else 0,
                    "成长": 0.10 if n > 0 else 0,
                    "低波": 0.20 if n > 0 else 0,
                    "技术": 0.10 if n > 0 else 0,
                    "持有期_天": n,
                    # 审计确认：当前为常量占位，未做真实归因计算；显式标注避免冒充真实结果
                    "is_placeholder": True,
                    "mode": "placeholder",
                    "note": "当前为常量占位，待接入真实因子归因计算",
                }
            })
        except Exception as e:
            self._send_json({"ok": False, "error": str(e)}, status=500)

    def _handle_v4_regime(self) -> None:
        """GET /api/v4/regime — 市场状态"""
        try:
            from quant_system.market_regime import get_current_regime, regime_to_text
            regime = get_current_regime()
            text = regime_to_text(regime)
            self._send_json({
                "ok": True,
                "data": {
                    "trend": regime.get("trend", "sideways"),
                    "volatility": regime.get("volatility", "normal"),
                    "liquidity": regime.get("liquidity", "normal"),
                    "sentiment": regime.get("sentiment", "normal"),
                    "composite_risk_level": regime.get("composite_risk_level", 5),
                    "text": text,
                }
            })
        except Exception as e:
            self._send_json({"ok": False, "error": str(e)}, status=500)

    def _handle_v4_health(self) -> None:
        """GET /api/v4/health — 系统健康检查"""
        try:
            from scripts.system_manager import check_system_health
            health = check_system_health()
            self._send_json({"ok": True, "data": health})
        except Exception as e:
            self._send_json({"ok": False, "error": f"系统健康检查失败: {e}"}, status=500)

    def _handle_paper_preview(self, body: dict) -> None:
        """Validate a paper order before mutation; no hidden auto-execution."""
        try:
            init_db()
            direction = str(body.get("direction") or "buy").lower()
            symbol = normalize_symbol(str(body.get("symbol") or ""))
            shares = int(body.get("shares") or 0)
            price = float(body.get("price") or 0)
            if direction not in {"buy", "sell"}:
                self._send_json({"ok": False, "error": "direction 必须是 buy/sell"}, status=400)
                return
            errors, warnings = [], []
            if not re.fullmatch(r"\d{6}", symbol):
                errors.append("代码必须是6位A股代码")
            if shares <= 0 or price <= 0:
                errors.append("股数和价格必须大于0")
            if shares > 0 and shares % 100:
                errors.append("A股普通股票委托必须为100股整手")
            gross = round(shares * price, 2) if shares > 0 and price > 0 else 0.0
            commission = round(max(5.0, gross * 0.0003), 2) if gross else 0.0
            stamp_tax = round(gross * 0.0005, 2) if direction == "sell" and gross else 0.0
            transfer_fee = round(gross * 0.00001, 2) if gross else 0.0
            summary = get_summary()
            # Paper account starts with a declared virtual capital; do not pretend broker cash exists.
            initial_cash = float(os.environ.get("PAPER_INITIAL_CASH", "1000000"))
            trades = get_trades(days=3650, limit=100000)
            cash_delta = sum((float(item.get("total_amount") or 0) if item.get("trade_type") == "sell" else -float(item.get("total_amount") or 0)) for item in trades)
            available_cash = round(initial_cash + cash_delta, 2)
            position = get_position(symbol) if symbol else None
            held = int((position or {}).get("shares") or 0)
            estimated_cash = gross + commission + transfer_fee if direction == "buy" else gross - commission - stamp_tax - transfer_fee
            if direction == "buy" and estimated_cash > available_cash:
                errors.append(f"纸面可用资金不足：可用 {available_cash:,.2f}，需要 {estimated_cash:,.2f}")
            if direction == "sell" and shares > held:
                errors.append(f"可卖持仓不足：持有 {held} 股，拟卖 {shares} 股")
            if direction == "buy" and gross and gross > initial_cash * 0.2:
                warnings.append("单笔金额超过初始纸面资金20%，请确认仓位风险")
            if direction == "buy" and summary.get("num_positions", 0) >= 10 and not position:
                warnings.append("当前持仓已达10只，新开仓会增加组合分散与跟踪成本")
            client_order_id = str(body.get("client_order_id") or uuid.uuid4().hex)
            fingerprint = _paper_order_fingerprint(direction, symbol, shares, price)
            expires_at = int(time.time()) + 300
            preview = {
                "client_order_id": client_order_id,
                "fingerprint": fingerprint,
                "expires_at": expires_at,
                "direction": direction, "symbol": symbol, "shares": shares, "price": price,
                "gross_amount": gross, "estimated_commission": commission, "estimated_stamp_tax": stamp_tax,
                "estimated_transfer_fee": transfer_fee,
                "estimated_cash_change": round(-(gross + commission + transfer_fee) if direction == "buy" else gross - stamp_tax - commission - transfer_fee, 2),
                "paper_initial_cash": initial_cash, "available_cash": available_cash, "held_shares": held,
                "errors": errors, "warnings": warnings, "allowed": not errors,
                "contract": "paper-order-preview.v1",
            }
            _persist_paper_order_record(client_order_id, {"kind": "preview", "fingerprint": fingerprint,
                                                            "expires_at": expires_at, "preview": preview})
            self._send_json({"ok": True, "preview": preview})
        except Exception as exc:
            self._send_json({"ok": False, "error": f"订单预览失败: {str(exc)[:200]}"}, status=500)

    def _handle_paper_order(self, body: dict) -> None:
        """POST /api/paper/order: create release-bound manual paper intent."""
        try:
            from quant_system.trade_db import create_paper_order
            result = create_paper_order(symbol=normalize_symbol(str(body.get("symbol", ""))),
                                        direction=str(body.get("direction", "")), shares=body.get("shares"),
                                        suggested_price=body.get("suggested_price"),
                                        release_id=str(body.get("release_id", "")),
                                        order_type=str(body.get("order_type", "limit")),
                                        notes=str(body.get("notes", "")))
            self._send_json({"ok": "error" not in result, **result}, status=400 if result.get("error") else 200)
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)[:200]}, status=500)

    def _handle_paper_fill(self, body: dict) -> None:
        """POST /api/paper/fill: append a validated partial/manual fill."""
        try:
            from quant_system.trade_db import record_paper_fill
            result = record_paper_fill(int(body.get("order_id")), filled_shares=body.get("filled_shares"),
                                       filled_price=body.get("filled_price"), commission=body.get("commission", 0),
                                       stamp_tax=body.get("stamp_tax", 0), transfer_fee=body.get("transfer_fee", 0),
                                       execution_id=body.get("execution_id") or None,
                                       filled_at=body.get("filled_at") or None)
            self._send_json({"ok": "error" not in result, **result}, status=400 if result.get("error") else 200)
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)[:200]}, status=500)

    def _handle_paper_buy(self, body: dict) -> None:
        """POST /api/paper/buy {symbol, name, shares, price, signal?, notes?, sector?}"""
        try:
            init_db()
            symbol = str(body.get("symbol", "")).strip()
            name = str(body.get("name", "")).strip()
            shares = int(body.get("shares", 0))
            price = float(body.get("price", 0))
            signal = str(body.get("signal", "")).strip()
            notes = str(body.get("notes", "")).strip()
            sector = str(body.get("sector", "")).strip()
            if not symbol or shares <= 0 or price <= 0:
                self._send_json({"ok": False, "error": "symbol/shares/price 必填"}, status=400)
                return
            if not re.fullmatch(r"\d{6}", normalize_symbol(symbol)) or shares % 100:
                self._send_json({"ok": False, "error": "买入必须使用6位A股代码且数量为100股整手"}, status=400)
                return
            paper_cash = float(os.environ.get("PAPER_INITIAL_CASH", "1000000"))
            trades = get_trades(days=3650, limit=100000)
            available_cash = paper_cash + sum((float(item.get("total_amount") or 0) if item.get("trade_type") == "sell" else -float(item.get("total_amount") or 0)) for item in trades)
            gross = round(shares * price, 2)
            commission = round(max(5.0, gross * 0.0003), 2)
            transfer_fee = round(gross * 0.00001, 2)
            estimated_cost = gross + commission + transfer_fee
            if estimated_cost > available_cash:
                self._send_json({"ok": False, "error": f"纸面可用资金不足：可用 {available_cash:,.2f}，需要 {estimated_cost:,.2f}"}, status=409)
                return
            client_order_id = str(body.get("client_order_id") or "").strip()
            if not client_order_id:
                self._send_json({"ok": False, "error": "缺少预检生成的 client_order_id"}, status=409)
                return
            auto_execute = bool(body.get("auto_execute"))
            confidence = max(0.0, min(1.0, float(body.get("confidence", 0) or 0)))
            validation = None
            factor_quality = None
            fq_path = ROOT / "generated" / "factor_quality_registry.json"
            if fq_path.exists():
                try:
                    fq = json.loads(fq_path.read_text(encoding='utf-8'))
                    factor_quality = {k: fq.get('counts', {}).get(k, 0) for k in ('eligible','total','core','candidate','observe','blocked','oos_verified','mined_verified')}
                except Exception:
                    factor_quality = None
            latest_bt = ROOT / "generated" / "backtest_latest.json"
            if latest_bt.exists():
                try:
                    bt = json.loads(latest_bt.read_text(encoding="utf-8"))
                    match = next((r for r in (bt.get("rows") or []) if str(r.get("symbol")) == normalize_symbol(symbol)), None)
                    validation = (match or {}).get("validation")
                except Exception:
                    validation = None
            if auto_execute and (confidence < 0.4 or not validation or not validation.get("auto_execute_allowed") or (factor_quality and factor_quality.get('eligible', 0) < 3)):
                self._send_json({"ok": False, "error": "自动模拟执行被拒绝：置信度、真实回测或有效因子池未通过门槛",
                                 "confidence": confidence, "validation": validation, "factor_quality": factor_quality}, status=409)
                return
            validation_tag = "回测通过" if validation and validation.get("passed") else "未通过回测"
            notes = (notes + " | " if notes else "") + f"决策置信度={confidence:.2f}; {validation_tag}; auto={auto_execute}"
            claimed, duplicate = _claim_paper_preview(client_order_id, _paper_order_fingerprint("buy", symbol, shares, price))
            if duplicate is not None:
                self._send_json({"ok": True, "duplicate": True, **duplicate})
                return
            if claimed is None:
                self._send_json({"ok": False, "error": "预检不存在、已过期、正在处理或委托参数已改变，请重新预览"}, status=409)
                return
            try:
                result = add_buy(symbol, name, shares, price, signal_type=signal or "subjective_decision", notes=notes, sector=sector, commission=commission, transfer_fee=transfer_fee)
                _persist_paper_order_record(client_order_id, {"client_order_id": client_order_id, "kind": "executed", "fingerprint": _paper_order_fingerprint("buy", symbol, shares, price), "executed_at": int(time.time()), **result})
            except Exception:
                _release_paper_preview(client_order_id, claimed)
                raise
            self._send_json({"ok": True, "client_order_id": client_order_id, **result})
        except Exception as e:
            self._send_json({"ok": False, "error": str(e), "trace": traceback.format_exc(limit=6)})

    def _handle_paper_sell(self, body: dict) -> None:
        """POST /api/paper/sell {symbol, shares, price, notes?}"""
        try:
            init_db()
            symbol = str(body.get("symbol", "")).strip()
            shares = int(body.get("shares", 0))
            price = float(body.get("price", 0))
            notes = str(body.get("notes", "")).strip()
            if not symbol or shares <= 0 or price <= 0:
                self._send_json({"ok": False, "error": "symbol/shares/price 必填"}, status=400)
                return
            held = int((get_position(normalize_symbol(symbol)) or {}).get("shares") or 0)
            if not re.fullmatch(r"\d{6}", normalize_symbol(symbol)) or shares % 100:
                self._send_json({"ok": False, "error": "卖出必须使用6位A股代码且数量为100股整手"}, status=400)
                return
            if shares > held:
                self._send_json({"ok": False, "error": f"可卖持仓不足：持有 {held} 股，拟卖 {shares} 股"}, status=409)
                return
            client_order_id = str(body.get("client_order_id") or "").strip()
            if not client_order_id:
                self._send_json({"ok": False, "error": "缺少预检生成的 client_order_id"}, status=409)
                return
            gross = round(shares * price, 2)
            commission = round(max(5.0, gross * 0.0003), 2)
            stamp_tax = round(gross * 0.0005, 2)
            transfer_fee = round(gross * 0.00001, 2)
            claimed, duplicate = _claim_paper_preview(client_order_id, _paper_order_fingerprint("sell", symbol, shares, price))
            if duplicate is not None:
                self._send_json({"ok": True, "duplicate": True, **duplicate})
                return
            if claimed is None:
                self._send_json({"ok": False, "error": "预检不存在、已过期、正在处理或委托参数已改变，请重新预览"}, status=409)
                return
            try:
                result = add_sell(symbol, shares, price, notes=notes, commission=commission, stamp_tax=stamp_tax, transfer_fee=transfer_fee)
                _persist_paper_order_record(client_order_id, {"client_order_id": client_order_id, "kind": "executed", "fingerprint": _paper_order_fingerprint("sell", symbol, shares, price), "executed_at": int(time.time()), **result})
            except Exception:
                _release_paper_preview(client_order_id, claimed)
                raise
            self._send_json({"ok": True, "client_order_id": client_order_id, **result})
        except Exception as e:
            self._send_json({"ok": False, "error": str(e), "trace": traceback.format_exc(limit=6)})

    # ────────────────────────────────────────────────────────────
    # 监控新端点：告警中心 / 数据健康 / 定时任务状态（V11.1）
    # 约定：正常 200 结构化 JSON；异常也 200 + {ok:false, error}
    # ────────────────────────────────────────────────────────────

    def _handle_alerts_status(self) -> None:
        """GET /api/alerts_status — 告警中心：
        读 generated/alert_dedup_state.json（若存在，各 channel 最后发送时间/签名）
        + generated/alert_state.json（告警签名 → 最后发送时间戳）
        + data_warehouse/health_guardian_state.json（守护进程各项检查最近状态）。
        """
        from datetime import datetime as _dt
        data: dict = {}
        try:
            # 1) 告警去重状态（channel 级，文件可能不存在 → 空）
            dedup = {}
            dedup_path = ROOT / "generated" / "alert_dedup_state.json"
            if dedup_path.exists():
                try:
                    dedup = json.loads(dedup_path.read_text(encoding="utf-8"))
                except Exception as e:  # noqa: BLE001
                    dedup = {"_error": str(e)[:120]}
            data["dedup_state"] = dedup if isinstance(dedup, dict) else {"_raw": dedup}

            # 2) 最近告警清单（签名 → 时间戳，按时间倒序取 50）
            recent_alerts: list[dict] = []
            alert_state_path = ROOT / "generated" / "alert_state.json"
            if alert_state_path.exists():
                try:
                    amap = json.loads(alert_state_path.read_text(encoding="utf-8"))
                    if isinstance(amap, dict):
                        for sig, ts in amap.items():
                            try:
                                t = float(ts)
                            except (TypeError, ValueError):
                                t = 0.0
                            symbol = action = price = ""
                            parts = str(sig).split("|")
                            if len(parts) >= 2:
                                symbol, action = parts[0], parts[1]
                            if len(parts) >= 3:
                                price = parts[2]
                            recent_alerts.append({
                                "signature": sig, "symbol": symbol, "action": action,
                                "price": price, "ts": t,
                                "time": _dt.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S") if t else "-",
                            })
                        recent_alerts.sort(key=lambda x: x["ts"], reverse=True)
                        recent_alerts = recent_alerts[:50]
                except Exception as e:  # noqa: BLE001
                    recent_alerts = []
                    data["alerts_error"] = str(e)[:150]
            data["recent_alerts"] = recent_alerts

            # 3) 守护进程状态（数据健康 guardian）
            guardian: dict = {}
            guard_path = ROOT / "data_warehouse" / "health_guardian_state.json"
            if guard_path.exists():
                try:
                    guardian = json.loads(guard_path.read_text(encoding="utf-8"))
                except Exception as e:  # noqa: BLE001
                    guardian = {"_error": str(e)[:150]}
            data["guardian"] = guardian if isinstance(guardian, dict) else {"_raw": guardian}

            self._send_json({"ok": True, "data": data})
        except Exception as e:  # noqa: BLE001
            self._send_json({"ok": False, "error": f"告警状态读取失败: {str(e)[:200]}"})

    def _handle_data_health(self) -> None:
        """GET /api/data_health — 数据健康：
        读 data_warehouse/data_freshness.json（全部数据集条目 file/as_of/stale_days/
        threshold_days/stale/note），并附最新 generated/data_health_*.md 摘要。
        """
        data: dict = {}
        try:
            entries: list[dict] = []
            fp = ROOT / "data_warehouse" / "data_freshness.json"
            if fp.exists():
                try:
                    raw = json.loads(fp.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        for _, v in raw.items():
                            if not isinstance(v, dict):
                                continue
                            entries.append({
                                "file": v.get("file", ""),
                                "source": v.get("source", ""),
                                "freq": v.get("freq", ""),
                                "as_of": str(v.get("as_of", "") or ""),
                                "stale_days": v.get("stale_days"),
                                "threshold_days": v.get("threshold_days"),
                                "stale": bool(v.get("stale", False)),
                                "note": str(v.get("note", "") or ""),
                                "rows": v.get("rows"),
                                "checked_at": str(v.get("checked_at", "") or ""),
                            })
                        entries.sort(key=lambda x: -(x["stale_days"] or 0))
                except Exception as e:  # noqa: BLE001
                    data["freshness_error"] = str(e)[:150]
            data["entries"] = entries
            data["summary"] = {
                "total": len(entries),
                "stale": sum(1 for e in entries if e["stale"]),
                "fresh": sum(1 for e in entries if not e["stale"]),
            }
            # 最新 data_health_*.md 报告
            report = None
            md_files = sorted((ROOT / "generated").glob("data_health_*.md"),
                              key=lambda p: p.stat().st_mtime, reverse=True)
            if md_files:
                try:
                    content = md_files[0].read_text(encoding="utf-8", errors="replace")
                    report = {"file": md_files[0].name, "content": content[:20000]}
                except Exception as e:  # noqa: BLE001
                    report = {"file": md_files[0].name, "error": str(e)[:120]}
            data["report"] = report
            self._send_json({"ok": True, "data": data})
        except Exception as e:  # noqa: BLE001
            self._send_json({"ok": False, "error": f"数据健康读取失败: {str(e)[:200]}"})

    def _handle_tasks_status(self) -> None:
        """GET /api/tasks_status — 定时任务状态：
        subprocess `openclaw cron list --json`，结果缓存 5 分钟；
        解析 enabled 任务 name/kind/expr/nextRunAtMs（转本地时间）；
        失败返回空列表 + error（200）。
        """
        cached = _cache_get("tasks_status", 60)
        if cached is not None:
            self._send_json(cached)
            return
        from datetime import datetime as _dt, timezone as _tz, timedelta as _td
        cst = _tz(_td(hours=8))
        tasks: list[dict] = []
        error: str | None = None
        # 首选: 直读 OpenClaw 状态库（避免 CLI 子进程超时后把真实任务显示成 0/0）。
        try:
            import sqlite3 as _sq
            state_db = Path(os.environ.get("QUANT_STATE_DB", "quant_state.sqlite"))
            con = _sq.connect(f"file:{state_db}?mode=ro", uri=True, timeout=5)
            try:
                query = (
                    "SELECT COALESCE(display_name, name), schedule_kind, schedule_expr, every_ms, next_run_at_ms, "
                    "last_run_at_ms, last_run_status, last_error, consecutive_errors "
                    "FROM cron_jobs WHERE enabled=1 ORDER BY next_run_at_ms"
                )
                for row in con.execute(query):
                    name, kind, expr, every_ms, next_ms, last_ms, last_status, last_error, consecutive_errors = row
                    if not expr and every_ms:
                        em = int(every_ms)
                        expr = f"每 {em // 60000} 分钟" if em >= 60000 else f"每 {em // 1000} 秒"
                    def _format_ts(value):
                        try:
                            return _dt.fromtimestamp(float(value) / 1000.0, tz=cst).strftime("%m-%d %H:%M") if value else "-"
                        except (TypeError, ValueError, OSError):
                            return "-"
                    tasks.append({
                        "name": name or "-", "kind": kind or "", "expr": expr or "", "enabled": True,
                        "next_run": _format_ts(next_ms), "next_run_ms": next_ms,
                        "last_run": _format_ts(last_ms), "last_status": last_status or "idle",
                        "last_error": (last_error or "")[:180], "consecutive_errors": int(consecutive_errors or 0),
                    })
            finally:
                con.close()
        except Exception as e:  # noqa: BLE001
            error = f"网关状态库读取失败: {e}"[:200]
        payload = {"ok": True, "data": {"tasks": tasks, "total": len(tasks),
                                        "enabled": len(tasks), "error": error}}
        _cache_set("tasks_status", payload)
        self._send_json(payload)

    def _handle_execution_ledger(self) -> None:
        """GET /api/execution_ledger — 投递回执与技能执行登记的统一账本摘要。"""
        try:
            sys.path.insert(0, str(ROOT / "scripts"))
            from execution_ledger import ledger_summary  # noqa: PLC0415
            self._send_json({"ok": True, "ledger": ledger_summary()})
        except Exception as exc:
            self._send_json({"ok": False, "error": f"执行账本暂不可用：{str(exc)[:160]}"}, status=500)

    def _handle_decision_chain(self, query: dict[str, list[str]]) -> None:
        """GET /api/decision_chain — 已隔离的旧决策入口（兼容只读 shim）。

        2026-08-29 隔离：本端点原先直接读 intraday_chain_*/after_close_extra_*/
        battle_map_* 三套旧产物，与统一决策快照脱节。现改为兼容转发到统一
        决策工作台投影（同一 authoritative 快照），只在响应中标明 deprecated，
        不再把旧产物当作新业务生产入口。旧客户端仍可拿到可读响应。
        """
        mode = _first(query, "mode", "intraday")
        if mode not in {"intraday", "after_close", "battle_map"}:
            self._send_json({"ok": False, "error": "mode 必须为 intraday|after_close|battle_map"}, status=400)
            return
        date = _first(query, "date", "").strip() or None
        if date and not re.fullmatch(r"20\d{2}-\d{2}-\d{2}", date):
            self._send_json({"ok": False, "error": "date 必须为 YYYY-MM-DD"}, status=400)
            return
        # Keep the old endpoint genuinely compatible for existing clients.  A
        # date-specific legacy artifact is read-only and never triggers a scan.
        legacy_name = {
            "intraday": f"intraday_chain_{date}.json" if date else None,
            "after_close": f"after_close_extra_{date}.json" if date else None,
        }.get(mode)
        if legacy_name:
            legacy_path = ROOT / "generated" / legacy_name
            if legacy_path.is_file():
                try:
                    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
                    if isinstance(legacy, dict):
                        self._send_json({
                            "ok": True, "deprecated": True,
                            "deprecation_note": "旧决策入口已隔离；请改用 /api/workbench。",
                            "canonical": "/api/workbench", "contract": "legacy-decision-chain.v1",
                            "data": {"mode": mode, "date": date, "requested_date": date,
                                     "content": legacy, "content_source": "legacy_artifact"},
                        })
                        return
                except (OSError, ValueError, TypeError):
                    pass
        # A requested historical date is an artifact lookup, not a permission
        # to compute a fresh market-wide snapshot on an HTTP request.
        if date and not any((ROOT / "generated" / name).is_file() for name in (
            f"decision_snapshot_{mode}_{date}.json",
            f"intraday_chain_{date}.json" if mode == "intraday" else "",
            f"after_close_extra_{date}.json" if mode == "after_close" else "",
        )):
            self._send_json({
                "ok": True, "deprecated": True, "canonical": "/api/workbench",
                "data": {"mode": mode, "date": date, "requested_date": date,
                         "content": {}, "error": "产物缺失：指定日期没有可用决策快照"},
            })
            return
        try:
            from quant_web.workbench import build_workbench
            projection = build_workbench(mode=mode, date=date)
        except Exception as exc:
            self._send_json({
                "ok": True, "deprecated": True, "canonical": "/api/workbench",
                "data": {"mode": mode, "date": date, "requested_date": date,
                         "content": {}, "error": f"产物缺失：{str(exc)[:160]}"},
            })
            return
        # Isolated shim: wrap the canonical projection in the old response shape.
        self._send_json({
            "ok": True,
            "deprecated": True,
            "deprecation_note": "旧决策入口已隔离；请改用 /api/workbench（同 mode/date 参数）。",
            "canonical": "/api/workbench",
            "contract": projection.get("contract"),
            "data": {"mode": mode, "date": projection.get("as_of") or date,
                     "requested_date": date, "content": projection, "content_source": "unified_decision_snapshot"},
        })

    def _handle_factor_quality(self) -> None:
        """Return the truthful factor eligibility registry."""
        p = ROOT / "generated" / "factor_quality_registry.json"
        if not p.exists():
            self._send_json({"ok": False, "error": "因子质量账本不存在（先运行 build_factor_quality_registry.py）"})
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            self._send_json({"ok": True, **data})
        except Exception as exc:
            self._send_json({"ok": False, "error": f"因子质量账本读取失败: {str(exc)[:160]}"})

    def _handle_auction_decision(self, query: dict[str, list[str]]) -> None:
        """GET /api/auction_decision?date=YYYY-MM-DD: 9:25 条件决策报告。"""
        from datetime import datetime as _dt
        date = _first(query, "date", "") or _dt.now().strftime("%Y-%m-%d")
        try:
            import sys as _sys
            _sys.path.insert(0, str(ROOT / "scripts"))
            from auction_decision import build_decision
            self._send_json(build_decision(date=date, save=False))
        except Exception as exc:
            self._send_json({"ok": False, "error": f"竞价报告暂不可用: {str(exc)[:180]}"}, status=500)

    def _handle_decision_snapshot(self, query: dict[str, list[str]]) -> None:
        """GET /api/decision_snapshot?mode=intraday|after_close[&date=]."""
        mode = _first(query, "mode", "intraday")
        date = _first(query, "date", "") or None
        force_refresh = _first(query, "refresh", "false").lower() == "true"
        if mode not in ("intraday", "after_close", "battle_map"):
            self._send_json({"ok": False, "error": "mode 必须为 intraday|after_close|battle_map"}, status=400)
            return
        cache_key = f"decision_snapshot:{mode}:{date or 'latest'}"
        if not force_refresh:
            cached = _cache_get(cache_key, 45 if mode == "intraday" else 180)
            if cached is not None:
                self._send_json(cached)
                return
        try:
            # The 10-minute worker writes the complete snapshot. Serve it first
            # so the GUI never synchronously recomputes thousands of rows.
            artifact = ROOT / "generated" / f"decision_snapshot_{mode}_{date or datetime.now().strftime('%Y-%m-%d')}.json"
            if not force_refresh and artifact.exists():
                candidate = json.loads(artifact.read_text(encoding="utf-8"))
                if isinstance(candidate, dict) and (not date or str(candidate.get("as_of") or date)[:10] == date):
                    candidate.setdefault("mode", mode)
                    _cache_set(cache_key, candidate)
                    self._send_json(candidate)
                    return
            if not force_refresh and mode == "intraday":
                compact_path = ROOT / "generated" / f"decision_snapshot_intraday_{date or datetime.now().strftime('%Y-%m-%d')}.json"
                if compact_path.exists():
                    payload = json.loads(compact_path.read_text(encoding="utf-8"))
                    _cache_set(cache_key, payload)
                    self._send_json(payload)
                    return
                chain_path = ROOT / "generated" / f"intraday_chain_{date or datetime.now().strftime('%Y-%m-%d')}.json"
                if chain_path.exists():
                    chain = json.loads(chain_path.read_text(encoding="utf-8"))
                    embedded = chain.get("decision_snapshot")
                    if isinstance(embedded, dict) and embedded.get("opportunities"):
                        payload = embedded
                        _cache_set(cache_key, payload)
                        self._send_json(payload)
                        return
            import sys as _sys
            _sys.path.insert(0, str(ROOT / "scripts"))
            from unified_decision_snapshot import build_snapshot
            payload = build_snapshot(mode=mode, date=date)
            _cache_set(cache_key, payload)
            self._send_json(payload)
        except Exception as exc:
            self._send_json({"ok": False, "error": f"统一决策快照失败: {str(exc)[:240]}"}, status=500)

    def _handle_intraday_flow_catalog(self) -> None:
        """List exact-date local intraday flow artifacts for UI date replay."""
        items = []
        for path in sorted((ROOT / "generated").glob("intraday_flow_history_????-??-??.json"), reverse=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                items.append({
                    "date": payload.get("date"), "status": payload.get("status"),
                    "snapshot_count": payload.get("snapshot_count", 0),
                    "complete_day": bool(payload.get("complete_day")),
                    "etf_status": (payload.get("etf") or {}).get("status", "unavailable"),
                    "generated_at": payload.get("generated_at"),
                })
            except Exception as exc:  # noqa: BLE001
                items.append({"date": path.stem.removeprefix("intraday_flow_history_"), "status": "error", "error": str(exc)[:120]})
        self._send_json({"ok": True, "data_contract": "intraday-flow-history.v1", "items": items})

    def _handle_intraday_sector_flow(self, query: dict[str, list[str]]) -> None:
        """Return explicitly typed industry/concept intraday real main-net snapshots."""
        kind = _first(query, "type", "行业")
        if kind not in ("行业", "概念"):
            self._send_json({"ok": False, "error": "type 必须为 行业/概念"}, status=400)
            return
        filename = "industry_fund_flow.parquet" if kind == "行业" else "concept_fund_flow_intraday.parquet"
        path = ROOT / "data_warehouse" / "market" / filename
        if not path.exists():
            self._send_json({"ok": False, "status": "missing", "error": f"{kind}资金采集尚未生成"}, status=404)
            return
        try:
            frame = pd.read_parquet(path)
            if "type" in frame.columns:
                frame = frame[frame["type"].astype(str).isin({kind, "industry" if kind == "行业" else "concept"})]
            day_col = "date" if "date" in frame.columns else "ts"
            frame["_day"] = pd.to_datetime(frame.get(day_col), errors="coerce").dt.strftime("%Y-%m-%d")
            frame = frame[frame["_day"].notna()]
            if frame.empty:
                self._send_json({"ok": False, "status": "missing", "error": f"无{kind}盘中真实资金记录"}, status=404)
                return
            latest_day = frame["_day"].max()
            batch = frame[frame["_day"] == latest_day]
            timestamp = None
            timestamp_warning = None
            if "ts" in batch.columns:
                parsed_ts = pd.to_datetime(batch["ts"], errors="coerce")
                same_day_ts = parsed_ts[parsed_ts.dt.strftime("%Y-%m-%d") == latest_day]
                if same_day_ts.notna().any():
                    timestamp = same_day_ts.max()
                    batch = batch[parsed_ts == timestamp]
                elif parsed_ts.notna().any():
                    timestamp_warning = "采集时间与交易日不一致，已按交易日保留数据但不展示跨日时点"
            batch = batch.sort_values("main_net_yi", ascending=False)
            rows = [{"name": str(row.get("name") or "-"), "main_net_yi": safe_float(row.get("main_net_yi")),
                     "timestamp": str(timestamp) if timestamp is not None else latest_day, "type": kind,
                     "as_of": latest_day} for row in batch.to_dict("records")]
            self._send_json({"ok": True, "data_contract": "sector-fund-flow-intraday.v1", "type": kind,
                             "as_of": latest_day, "latest_timestamp": str(timestamp) if timestamp is not None else None,
                             "timestamp_warning": timestamp_warning, "rows": rows,
                             "source": f"data_warehouse/market/{filename} · akshare.stock_sector_fund_flow_rank(今日主力净流入)"})
        except Exception as exc:
            self._send_json({"ok": False, "status": "error", "error": f"盘中行业资金读取失败: {str(exc)[:180]}"}, status=500)

    def _handle_intraday_flow(self, query: dict[str, list[str]]) -> None:
        """GET /api/intraday_flow[?date=YYYY-MM-DD], exact-date local history."""
        date = _first(query, "date", "") or datetime.now().strftime("%Y-%m-%d")
        if not re.fullmatch(r"20\d{2}-\d{2}-\d{2}", date):
            self._send_json({"ok": False, "error": "date 必须为 YYYY-MM-DD"}, status=400)
            return
        path = ROOT / "generated" / f"intraday_flow_history_{date}.json"
        if not path.exists():
            self._send_json({"ok": False, "date": date, "status": "unavailable",
                             "error": f"盘中分时时序不存在: {path.name}"}, status=404)
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("date") != date:
                raise ValueError(f"artifact date={payload.get('date')} expected={date}")
            self._send_json({"ok": True, **payload})
        except Exception as exc:
            self._send_json({"ok": False, "date": date,
                             "error": f"盘中分时时序损坏: {str(exc)[:180]}"}, status=500)

    def _handle_rag_search(self, query: dict[str, list[str]]) -> None:
        """GET /api/rag_search?q=低估价值 因子 — 知识库检索(书籍+skills心法)。
        返回相关方法论片段, 供前端展示'决策链依据'。"""
        q = _first(query, "q", "").strip()
        k = 6
        try:
            k = max(1, min(20, int(_first(query, "k", "6"))))
        except ValueError:
            k = 6
        if not q:
            self._send_json({"ok": False, "error": "q 必填"})
            return
        cache_key = f"rag_{q}_{k}"
        cached = _cache_get(cache_key, 300)
        if cached is not None:
            self._send_json(cached)
            return
        try:
            from quant_system.analysis_core.knowledge_rag import search as rag_search
            hits = rag_search(q, k=k)
            payload = {"ok": True, "query": q, "hits": [
                {"file": h.get("file", ""), "cat": h.get("cat", ""),
                 "score": h.get("score", 0), "text": h.get("text", "")[:400]}
                for h in hits]}
        except Exception as e:  # noqa: BLE001
            payload = {"ok": False, "error": f"RAG 检索失败: {str(e)[:120]}"}
        _cache_set(cache_key, payload)
        self._send_json(payload)

    def _handle_research_flow(self, query: dict[str, list[str]]) -> None:
        """GET /api/research_flow?date=YYYY-MM-DD — 研报工作流结果(研报NLP+OCR+产业链/龙头融合)。"""
        date = _first(query, "date", "").strip()
        try:
            from quant_system.analysis_core.research_flow import _resolve_date
            date = _resolve_date(date or None)
            p = ROOT / "generated" / f"research_flow_{date}.json"
            used_date = date
            if not p.exists():
                # 回退最近研报产物（非交易日/未跑当天）
                cands = sorted(ROOT.glob("generated/research_flow_*.json"))
                if not cands:
                    self._send_json({"ok": False, "error": f"研报工作流结果不存在: {date}", "date": date})
                    return
                p = cands[-1]
                used_date = p.stem.replace("research_flow_", "")
            import json as _json
            data = _json.loads(p.read_text(encoding="utf-8"))
            self._send_json({"ok": True, "date": used_date, "requested_date": date, "data": data})
        except Exception as e:  # noqa: BLE001
            import traceback
            self._send_json({"ok": False, "error": f"研报工作流读取失败: {str(e)[:120]}",
                             "trace": traceback.format_exc(limit=3)}, status=500)

    def _handle_stock_lens(self, query: dict[str, list[str]]) -> None:
        """GET /api/stock_lens?code=600519[&date=] — 个股全景透视。
        GET /api/stock_lens?mode=watchlist — 自选+持仓全池三视角评分。"""
        mode = _first(query, "mode", "")
        code = _first(query, "code", "").strip()
        if mode == "watchlist":
            cache_key = "stock_lens_watchlist"
            cached = _cache_get(cache_key, 900)  # 15min
            if cached is not None:
                self._send_json(cached)
                return
            try:
                # 优先读 pipeline 盘后落盘产物(秒回); 无产物才实时计算(约2-4分钟)
                import glob as _glob2
                from datetime import datetime as _dt2
                today = _dt2.now().strftime("%Y-%m-%d")
                files = sorted(_glob2.glob(str(ROOT / "generated" / "stock_lens_*.json")), reverse=True)
                payload = None
                for f in files:
                    try:
                        rec = json.loads(Path(f).read_text(encoding="utf-8"))
                        items = rec.get("items") or []
                        if items and rec.get("date") >= "2026-08-01":
                            payload = {"ok": True, "mode": "watchlist",
                                       "data": [{"code": it["code"], "name": it["name"],
                                                 "short": it["synth"]["short_score"],
                                                 "mid": it["synth"]["mid_score"],
                                                 "long": it["synth"]["long_score"],
                                                 "verdict": it["synth"]["verdict"],
                                                 "tone": it["synth"]["tone"],
                                                 "strong": it["synth"].get("strong_axes", [])}
                                                for it in items]}
                            break
                    except Exception:  # noqa: BLE001
                        continue
                if payload is None:
                    from quant_system.analysis_core import stock_lens
                    items = stock_lens.analyze_watchlist()
                    payload = {"ok": True, "mode": "watchlist",
                               "data": [{"code": it["code"], "name": it["name"],
                                         "short": it["synth"]["short_score"],
                                         "mid": it["synth"]["mid_score"],
                                         "long": it["synth"]["long_score"],
                                         "verdict": it["synth"]["verdict"],
                                         "tone": it["synth"]["tone"],
                                         "strong": it["synth"].get("strong_axes", [])}
                                        for it in items]}
                _cache_set(cache_key, payload)
                self._send_json(payload)
            except Exception as e:  # noqa: BLE001
                self._send_json({"ok": False, "error": f"自选全景失败: {str(e)[:150]}"})
            return
        if not code:
            self._send_json({"ok": False, "error": "code 必填"})
            return
        cache_key = f"stock_lens_{code}_{_first(query, 'date', '')}"
        cached = _cache_get(cache_key, 300)
        if cached is not None:
            self._send_json(cached)
            return
        try:
            from quant_system.analysis_core import stock_lens
            result = stock_lens.analyze(code, _first(query, "date", "") or None)
            payload = {"ok": True, "data": result}
        except Exception as e:  # noqa: BLE001
            payload = {"ok": False, "error": f"个股全景分析失败: {str(e)[:150]}"}
        _cache_set(cache_key, payload)
        self._send_json(payload)


def _numpy_portfolio_fallback(ret_df, method: str) -> tuple:
    """scipy 不可用时的纯 numpy 组合优化回退。

    环境说明：scipy 1.9.1 二进制与 numpy 2.0 不兼容（numpy.core.multiarray failed
    to import），quant_system.portfolio 的 scipy.optimize 路径无法加载。
    回退采用逆波动率加权（对 min_vol/risk_parity 为常见启发式，max_sharpe 为简化
    近似，equal_weight 为等权），并计算组合收益/波动/夏普/分散度等指标。

    返回与 quant_system.portfolio.compute_portfolio 兼容的字段元组：
    (weights, expected_return, expected_vol, sharpe_ratio, diversification_ratio,
     n_positions, effective_n, top_holdings, note)
    """
    import numpy as _np
    syms = list(ret_df.columns)
    rets = ret_df.values  # T x N
    n = len(syms)
    if n == 0:
        return {}, 0.0, 0.0, 0.0, 0.0, 0, 0.0, [], "纯 numpy 回退：无数据"
    means = _np.nanmean(rets, axis=0) if rets.shape[0] else _np.zeros(n)
    vols = _np.nanstd(rets, axis=0, ddof=1) if rets.shape[0] > 1 else _np.ones(n)
    vols = _np.where(_np.isnan(vols) | (vols <= 0), 1e-6, vols)
    if method == "equal_weight":
        w = _np.full(n, 1.0 / n)
    else:
        inv = 1.0 / vols
        w = inv / inv.sum()
    cov = _np.cov(rets, rowvar=False, ddof=1) if rets.shape[0] > 1 else _np.eye(n) * 1e-6
    port_ret = float(_np.sum(w * means) * 252)
    port_vol = float(_np.sqrt(max(float(w @ cov @ w), 0.0)) * _np.sqrt(252))
    sharpe = port_ret / port_vol if port_vol > 0 else 0.0
    div = float(_np.sum(w * vols) * _np.sqrt(252) / port_vol) if port_vol > 0 else 1.0
    weights = {s: round(float(wi), 4) for s, wi in zip(syms, w)}
    top = sorted(weights.items(), key=lambda kv: -kv[1])[:5]
    eff = float(1.0 / _np.sum(w * w)) if _np.sum(w * w) > 0 else 1.0
    top_holdings = [{"symbol": s, "weight": wt} for s, wt in top]
    note = f"纯 numpy 回退（scipy 不可用，{method} 采用逆波动率加权近似）"
    return (weights, port_ret, port_vol, sharpe, div, n, eff, top_holdings, note)


def run_openclaw_agent(prompt: str, model: str | None = None) -> dict:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".md", delete=False) as f:
        f.write(prompt)
        prompt_path = f.name
    cmd = [
        OPENCLAW_BIN,
        "agent",
        "--agent",
        "main",
        "--session-key",
        "agent:main:quant-ui",
        "--message-file",
        prompt_path,
        "--json",
        "--timeout",
        "240",
    ]
    if model:
        cmd.extend(["--model", model])
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, timeout=280)
    finally:
        Path(prompt_path).unlink(missing_ok=True)
    parsed = _parse_openclaw_output(proc.stdout)
    return {
        "exit_code": proc.returncode,
        "text": parsed.get("text") or proc.stdout.strip(),
        "raw": parsed.get("raw"),
        "stderr": proc.stderr.strip(),
        "command_ok": proc.returncode == 0,
    }


def _parse_openclaw_output(stdout: str) -> dict:
    text = stdout.strip()
    if not text:
        return {"text": ""}
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        return {"text": text}
    for key in ["reply", "message", "text", "content", "output"]:
        if isinstance(raw, dict) and isinstance(raw.get(key), str):
            return {"text": raw[key], "raw": raw}
    if isinstance(raw, dict):
        nested = raw.get("result") or raw.get("response") or raw.get("data")
        if isinstance(nested, dict):
            payloads = nested.get("payloads")
            if isinstance(payloads, list) and payloads and isinstance(payloads[0], dict):
                text = payloads[0].get("text")
                if isinstance(text, str):
                    return {"text": text, "raw": raw}
            for key in ["reply", "message", "text", "content", "output"]:
                if isinstance(nested.get(key), str):
                    return {"text": nested[key], "raw": raw}
    return {"text": json.dumps(raw, ensure_ascii=False, indent=2), "raw": raw}


def _list_skills() -> list[dict]:
    if not SKILLS_DIR.exists():
        raise FileNotFoundError(f"skills directory not found: {SKILLS_DIR}")
    skills = []
    for skill_file in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        try:
            skills.append(_parse_skill_file(skill_file, include_content=False))
        except Exception as exc:
            skills.append({
                "name": skill_file.parent.name,
                "description": f"读取失败: {exc}",
                "category": "其他",
                "featured": False,
            })
    return skills


def _read_skill(name: str) -> dict:
    safe_name = _safe_skill_name(name)
    path = SKILLS_DIR / safe_name / "SKILL.md"
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"skill not found: {safe_name}")
    return _parse_skill_file(path, include_content=True)


def _safe_skill_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise ValueError("missing skill name")
    if "/" in name or "\\" in name or name in {".", ".."}:
        raise ValueError("invalid skill name")
    return name


def _parse_skill_file(path: Path, include_content: bool = False) -> dict:
    text = path.read_text(encoding="utf-8")
    front, body = _split_skill_frontmatter(text)
    name = str(front.get("name") or path.parent.name).strip().strip('"')
    description = str(front.get("description") or _first_markdown_sentence(body) or "暂无描述").strip().strip('"')
    category = _skill_category(path.parent.name)
    result = {
        "name": name,
        "description": description,
        "category": category,
        "featured": any(path.parent.name in names for names in SKILL_FEATURED_CATEGORIES.values()),
        "path": str(path.relative_to(ROOT)),
    }
    if include_content:
        result["content"] = text
        result["body"] = body.strip()
    return result


def _split_skill_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta = {}
    for line in parts[1].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta, parts[2]


def _first_markdown_sentence(body: str) -> str:
    for line in body.splitlines():
        line = line.strip("# >-\t ")
        if line:
            return line[:180]
    return ""


def _skill_category(name: str) -> str:
    for category, names in SKILL_FEATURED_CATEGORIES.items():
        if name in names:
            return category
    rules = [
        ("趋势跟踪", ("turtle", "livermore", "trend", "momentum", "breakout", "moving-average", "macd", "wyckoff", "chart", "technical")),
        ("量化因子", ("factor", "alpha", "financial-ml", "cointegration", "statistical", "multi-factor", "kalman", "q-learning")),
        ("市场结构", ("market", "microstructure", "breadth", "cycle", "sector", "intermarket", "order")),
        ("资金管理", ("capital", "portfolio", "allocation", "bet-sizing", "position-sizing", "mpt", "kelly")),
        ("风控", ("risk", "stop", "loss", "drawdown", "bias-audit", "pitfalls", "robustness")),
        ("宏观", ("macro", "fiat", "currency", "credit", "debt", "power", "greenspan", "china", "us-china")),
    ]
    for category, tokens in rules:
        if any(token in name for token in tokens):
            return category
    return "其他"


def _run_skill_analysis(skill: dict, symbol: str) -> dict:
    df = fetch_daily(symbol, start="20230101", use_cache=True)
    strategy = DEFAULT_STRATEGY
    data = add_technical_indicators(df, strategy)
    snapshot = latest_indicator_snapshot(normalize_symbol(symbol), df, strategy)
    latest = data.tail(1).to_dict(orient="records")[0] if not data.empty else {}
    ml = None
    try:
        from quant_system.ml_signals import predict_stock
        ml = predict_stock(normalize_symbol(symbol))
    except Exception as exc:
        ml = {"error": str(exc)}

    score = float(snapshot.get("composite_score") or 0)
    risk = float(snapshot.get("risk_score") or 0)
    action = "观察"
    if score >= 60 and risk <= 55:
        action = "可列入关注"
    elif risk >= 70 or score <= 35:
        action = "回避或降仓"

    checks = _skill_run_checks(skill["name"], snapshot, latest, ml)
    return {
        "skill": {"name": skill["name"], "description": skill.get("description", ""),
                  "category": skill.get("category", "其他"),
                  "execution_mode": "structured-adapter",
                  "methodology_executed": bool(checks),
                  "limitation": "仅执行已结构化为规则适配器的Skill检查；正文其余部分作为知识依据，不伪装成已执行。"},
        "symbol": normalize_symbol(symbol),
        "date": snapshot.get("date") or latest.get("date") or latest.get("日期") or "",
        "action": action,
        "summary": [
            f"综合分 {score:.1f}，风险分 {risk:.1f}。",
            f"按 {skill['name']} 的纪律先做条件检查，再决定是否进入交易计划。",
        ],
        "checks": checks,
        "snapshot": snapshot,
        "ml": ml,
    }


def _skill_run_checks(name: str, snapshot: dict, latest: dict, ml: dict | None) -> list[dict]:
    close = _safe_float(latest.get("close") or latest.get("收盘"))
    ma20 = _safe_float(latest.get("ma20") or latest.get("MA20"))
    ma60 = _safe_float(latest.get("ma60") or latest.get("MA60"))
    rsi = _safe_float(latest.get("rsi_14") or latest.get("RSI14") or snapshot.get("rsi_14"))
    risk = _safe_float(snapshot.get("risk_score"))
    trend = _safe_float(snapshot.get("trend_score"))
    ml_score = _safe_float((ml or {}).get("ml_score"))
    checks = []

    def add(label: str, passed: bool, value: object, note: str) -> None:
        checks.append({"label": label, "pass": bool(passed), "value": value if value is not None else "-", "note": note})

    if any(token in name for token in ("turtle", "trend", "breakout", "livermore", "moving-average")):
        add("趋势过滤", bool(trend and trend >= 55), trend, "趋势分 >=55 更适合趋势跟踪")
        add("均线结构", bool(close and ma20 and ma60 and close > ma20 > ma60), f"close={close} ma20={ma20} ma60={ma60}", "收盘价站上 MA20/MA60")
    elif any(token in name for token in ("mean-reversion", "reversion", "boll", "wyckoff")):
        add("均值回归区间", bool(rsi and rsi < 40), rsi, "RSI 偏低时优先寻找回归机会")
        add("风险约束", bool(risk is not None and risk <= 60), risk, "风险分 <=60 才考虑逆向信号")
    elif any(token in name for token in ("risk", "stop", "loss")):
        add("风险状态", bool(risk is not None and risk <= 55), risk, "风险分低于 55 才适合新开仓")
        add("波动纪律", True, snapshot.get("atr_pct", "-"), "结合 ATR 设置止损和仓位")
    elif any(token in name for token in ("factor", "alpha", "financial-ml", "multi-factor")):
        add("量化评分", bool(ml_score and ml_score >= 55), ml_score, "ML/因子分 >=55 才作为正向确认")
        add("技术确认", bool(trend and trend >= 50), trend, "因子信号需要趋势不拖后腿")
    else:
        add("综合评分", bool(_safe_float(snapshot.get("composite_score")) >= 50), snapshot.get("composite_score"), "综合分 >=50 代表基础条件尚可")
        add("风险约束", bool(risk is not None and risk <= 65), risk, "风险分 <=65 继续观察")
    return checks


def _safe_float(value) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _chat_prompt(message: str, context: dict) -> str:
    context_json = json.dumps(_json_safe(context), ensure_ascii=False, indent=2)
    return f"""你是一个嵌入在本地 A股量化工作台里的 OpenClaw 投研助手。

请基于用户问题和工作台上下文回答。回答必须区分事实、模型指标、交易纪律和不确定性；不得承诺收益；涉及交易时给出风险和仓位约束。

用户问题：
{message}

工作台上下文：
```json
{context_json}
```
"""


def _strategy(data: dict) -> StrategyConfig:
    base = DEFAULT_STRATEGY.__dict__
    merged = {k: data.get(k, v) for k, v in base.items()}
    return StrategyConfig(**{k: type(base[k])(merged[k]) for k in base})


def _portfolio(data: dict) -> PortfolioConfig:
    base = DEFAULT_PORTFOLIO.__dict__
    merged = {k: data.get(k, v) for k, v in base.items()}
    return PortfolioConfig(**{k: type(base[k])(merged[k]) for k in base})


def _symbols(value: object) -> list[str]:
    if isinstance(value, list):
        parts = value
    else:
        parts = str(value).replace(",", " ").split()
    return [resolve_symbol(part) for part in parts if str(part).strip()]


def _enrich_backtest_with_benchmark(result: dict) -> dict:
    """回测结果增强: 回撤序列 + 沪深300基准对比(alpha/beta/IR) + 月度收益。

    基准行情经 6 小时缓存, akshare 失败时降级为无基准(不抛异常)。
    """
    eq = result.get("equity_curve") or []
    if len(eq) < 2:
        return result
    # 1) 回撤序列(%)
    peak = -1e18
    dd = []
    for pt in eq:
        peak = max(peak, float(pt.get("equity", 0)))
        dd.append(round((float(pt.get("equity", 0)) / peak - 1) * 100, 2) if peak else 0.0)
    result["drawdown_curve"] = dd
    # 2) 月度收益(热力图用)
    try:
        import pandas as _pd
        _ser = _pd.Series([float(p.get("equity", 0)) for p in eq],
                          index=_pd.to_datetime([p.get("date") for p in eq]))
        monthly = _ser.resample("ME").last().pct_change().dropna()
        result["monthly_returns"] = [
            {"month": idx.strftime("%Y-%m"), "ret": round(float(v) * 100, 2)}
            for idx, v in monthly.items()]
    except Exception:  # noqa: BLE001
        result["monthly_returns"] = []
    # 3) 沪深300 基准对比（本地指数数据, 双机一致, 免 akshare 网络依赖）
    try:
        import pandas as _pd2
        start, end = eq[0]["date"], eq[-1]["date"]
        idx_path = ROOT / "data_warehouse" / "market" / "index_daily_沪深300.parquet"
        if not idx_path.exists():
            idx_path = ROOT / "data_warehouse" / "market" / "index_daily.parquet"
        if idx_path.exists():
            idx = _pd2.read_parquet(idx_path)
            idx["date"] = _pd2.to_datetime(idx["date"], errors="coerce")
            idx = idx[(idx["date"] >= _pd2.Timestamp(start)) & (idx["date"] <= _pd2.Timestamp(end))]
            idx = idx.dropna(subset=["close"]).sort_values("date")
            if len(idx) >= 20:
                bench_ret = idx["close"].astype(float).pct_change().dropna()
                bench_ret.name = "000300.SH"
                port = _pd2.Series([float(p.get("equity", 0)) for p in eq],
                                   index=_pd2.to_datetime([p.get("date") for p in eq]))
                port_ret = port.pct_change().dropna()
                from quant_system.backtest import Benchmark
                bench_metrics = Benchmark().compute_alpha(port_ret, bench_ret)
                nav = (1 + bench_ret.fillna(0)).cumprod()
                result["benchmark_curve"] = [
                    {"date": idx_.strftime("%Y-%m-%d"), "nav": round(float(v), 4)}
                    for idx_, v in zip(idx["date"], nav)]
                result["benchmark_metrics"] = bench_metrics
    except Exception:  # noqa: BLE001
        pass
    return result


def _records(df: pd.DataFrame) -> list[dict]:
    records = []
    for row in df.to_dict(orient="records"):
        clean = {}
        for key, value in row.items():
            if isinstance(value, pd.Timestamp):
                clean[key] = value.isoformat(sep=" ", timespec="seconds") if value.time() != datetime.min.time() else value.strftime("%Y-%m-%d")
            else:
                clean[key] = value
        records.append(clean)
    return records


def _json_safe(value):
    if isinstance(value, (dict, list, tuple)):
        return type(value)(_json_safe(v) for v in value) if isinstance(value, (list, tuple)) else {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, pd.Timestamp):
        return value.isoformat(sep=" ", timespec="seconds") if value.time() != datetime.min.time() else value.strftime("%Y-%m-%d")
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return None
        if isinstance(value, float):
            return round(value, 4)
        return value
    from datetime import datetime as _datetime, date as _date, time as _time
    if isinstance(value, (_datetime, _date)):
        return value.isoformat()
    if isinstance(value, _time):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value) if not isinstance(value, (str, type(None))) else value


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value in (None, "", "-"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _first(query: dict[str, list[str]], key: str, default: str) -> str:
    values = query.get(key)
    return values[0] if values else default


_REPORT_DATE_RE = re.compile(r"20\d{2}-\d{2}-\d{2}")


def _report_roots() -> list[Path]:
    """Return configured report roots plus local fallbacks without scanning cwd."""
    roots: list[Path] = []
    configured = os.environ.get("QUANT_REPORT_ROOT", "").strip()
    candidates = ([Path(configured).expanduser()] if configured else []) + [
        Path.home() / "Desktop" / "研报共享", ROOT / "研究报告",
        ROOT.parent / "Desktop" / "研报共享",
    ]
    for d in candidates:
        try:
            if d.is_dir():
                roots.append(d)
        except OSError:  # noqa: BLE001
            continue
    return roots


def _load_macro_data() -> dict:
    """Load consolidated macro overview data for SSE push."""
    try:
        from quant_system import market_temperature, margin as margin_mod

        # Read handler cache keys for visible macro modules only.
        indices = _cache_get("indices", 60) or {}
        temp = _cache_get("market_temp", 60) or {}
        margin_data = _cache_get("margin:summary", 60) or {}
        pulse = _cache_get("market_pulse", 60) or {}

        return {
            "indices": indices.get("data", indices),
            "temp": temp,
            "margin": margin_data,
            "pulse": pulse,
            "ts": time.time(),
        }
    except Exception as e:
        return {"error": str(e), "ts": time.time()}


def _market_temp_from_raw(raw: dict) -> dict:
    advance_pct = raw.get("advance_pct", 50)
    return {
        "temperature": round(safe_float(advance_pct, 50), 1),
        "advance_pct": advance_pct,
        "advance": raw.get("advance", 0),
        "decline": raw.get("decline", 0),
        "sh_pct": raw.get("sh_pct", 0),
        "sz_pct": raw.get("sz_pct", 0),
        "cy_pct": raw.get("cy_pct", 0),
        "total_amount_yi": raw.get("total_amount_yi", 0),
        "pct_above_ma300": raw.get("pct_above_ma300", 0),
        "ma60_direction": raw.get("ma60_direction", "unknown"),
        "stage": raw.get("stage", "--"),
    }


def _detect_market_regime(market_data: dict) -> dict:
    breadth = market_data.get("breadth") if isinstance(market_data.get("breadth"), dict) else {}
    risk = safe_float(market_data.get("risk"), 5)
    temp = safe_float(market_data.get("temperature") or market_data.get("advance_pct") or breadth.get("上涨占比%"), 50)
    ma300 = safe_float(market_data.get("pct_above_ma300") or breadth.get("MA300占比") or breadth.get("上涨占比%"), 50)
    ma144 = safe_float(market_data.get("pct_above_ma144") or market_data.get("pct_above_ma60") or breadth.get("MA144占比") or ma300, ma300)
    trend = str(market_data.get("ma60_direction") or market_data.get("ma144_direction") or market_data.get("stage") or market_data.get("risk_level") or "").lower()
    trend_up = any(x in trend for x in ("up", "bull", "多", "上行", "强", "升"))
    trend_down = any(x in trend for x in ("down", "bear", "空", "下行", "弱", "降", "高风险"))
    cold = temp <= 32
    frozen = temp <= 20 or ma300 <= 25
    hot = temp >= 68
    overheated = temp >= 82 and ma300 >= 72

    scores = {
        "panic": 0,
        "bear": 0,
        "sideways": 0,
        "bull": 0,
        "bubble": 0,
    }
    if frozen:
        scores["panic"] += 4
    if cold:
        scores["panic"] += 2
        scores["bear"] += 2
    if risk >= 8:
        scores["panic"] += 2
    elif risk >= 6:
        scores["bear"] += 2
    elif risk <= 3:
        scores["bull"] += 1
    if ma300 <= 35:
        scores["bear"] += 3
    elif ma300 >= 60:
        scores["bull"] += 2
    if ma144 <= 38:
        scores["bear"] += 1
    elif ma144 >= 58:
        scores["bull"] += 1
    if trend_down:
        scores["bear"] += 2
        scores["panic"] += 1 if cold else 0
    if trend_up:
        scores["bull"] += 2
    if hot:
        scores["bull"] += 1
    if overheated and risk <= 4 and not trend_down:
        scores["bubble"] += 4
    if 40 <= temp <= 62 and 38 <= ma300 <= 58 and not trend_up and not trend_down:
        scores["sideways"] += 4
    if 42 <= temp <= 66 and 35 <= ma300 <= 62:
        scores["sideways"] += 2

    regime = max(scores, key=scores.get)
    if regime == "panic" and not (frozen or risk >= 8 or (cold and trend_down)):
        regime = "bear"
    if regime == "bubble" and not overheated:
        regime = "bull"
    ordered = sorted(scores.values(), reverse=True)
    margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0)
    confidence = min(0.94, 0.52 + max(0, ordered[0]) * 0.055 + margin * 0.035)
    signals = _regime_signals(temp, ma300, trend, market_data, ma144, risk, scores)
    note = _load_regime_skill_note()
    summaries = {
        "panic": "市场处于恐慌/冰点区，温度、广度或风险信号显著恶化。",
        "bear": "市场处于熊市或弱势下行区，趋势和广度偏防守。",
        "sideways": "市场处于震荡区，温度和广度缺少一致方向。",
        "bull": "市场处于牛市或强势区，温度、广度和趋势总体同向改善。",
        "bubble": "市场处于过热区，广度和温度偏高，需要防范拥挤回落。",
    }
    return {"regime": regime, "confidence": round(confidence, 2), "signals": signals, "summary": summaries[regime], "skill": note}


def _regime_signals(temp: float, ma300: float, trend: str, market_data: dict, ma144: float, risk: float, scores: dict) -> list[str]:
    signals = [f"temperature={temp:.1f}", f"pct_above_ma144={ma144:.1f}", f"pct_above_ma300={ma300:.1f}"]
    if market_data.get("risk") is not None:
        signals.append(f"risk={risk:.1f}")
    if trend:
        signals.append(f"trend={trend}")
    breadth = market_data.get("breadth") if isinstance(market_data.get("breadth"), dict) else {}
    advance = market_data.get("advance", breadth.get("上涨"))
    decline = market_data.get("decline", breadth.get("下跌"))
    if advance is not None and decline is not None:
        signals.append(f"breadth={advance}/{decline}")
    if market_data.get("stage"):
        signals.append(f"stage={market_data.get('stage')}")
    signals.append("gmm_proxy=" + ",".join(f"{k}:{v}" for k, v in sorted(scores.items())))
    return signals


def _load_regime_skill_note() -> str:
    if not MARKET_REGIME_SKILL.exists():
        return "market-regime-gmm skill missing; using heuristic fallback"
    text = MARKET_REGIME_SKILL.read_text(encoding="utf-8", errors="replace")
    return _first_markdown_sentence(_split_skill_frontmatter(text)[1])[:120]


def _format_push_alert(body: dict) -> str:
    return "\n".join([
        "### A股量化预警",
        f"标的：{body.get('symbol')}",
        f"动作：{body.get('action')}",
        f"价格：{body.get('price')}",
        f"原因：{body.get('reason')}",
        f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ])


def _deliver_alert_push(title: str, text: str) -> list[dict]:
    sys.path.insert(0, str(ROOT / "scripts")) if str(ROOT / "scripts") not in sys.path else None
    try:
        from report_delivery import _build_channel_status, deliver_message, load_config
        cfg = load_config(DELIVERY_CONFIG).get("channels", {})
        feishu = cfg.get("feishu") or {}
        weixin = cfg.get("weixin") or {}
        statuses = []
        if feishu.get("enabled") and feishu.get("chat_id"):
            statuses.append(_build_channel_status("feishu", deliver_message("feishu", feishu["chat_id"], text, title=title), feishu["chat_id"]))
        if weixin.get("enabled") and weixin.get("target"):
            channel = weixin.get("channel") or "openclaw-weixin"
            statuses.append(_build_channel_status("wechat", deliver_message(channel, weixin["target"], text, account=weixin.get("accountId"), title=title), weixin["target"]))
        return [_json_safe(s.__dict__) for s in statuses]
    except Exception as exc:
        return [{"name": "alert", "status": "未完成", "reason": repr(exc), "target": ""}]


def _load_portfolio() -> dict:
    if not PORTFOLIO_FILE.exists():
        return {"positions": {}, "updated_at": ""}
    data = json.loads(PORTFOLIO_FILE.read_text(encoding="utf-8"))
    data.setdefault("positions", {})
    return data


def _save_portfolio(portfolio: dict) -> None:
    PORTFOLIO_FILE.parent.mkdir(parents=True, exist_ok=True)
    portfolio["updated_at"] = datetime.now().isoformat(timespec="seconds")
    PORTFOLIO_FILE.write_text(json.dumps(_json_safe(portfolio), ensure_ascii=False, indent=2), encoding="utf-8")


def _portfolio_summary(portfolio: dict) -> dict:
    positions, total_cost, total_value = [], 0.0, 0.0
    for symbol, pos in (portfolio.get("positions") or {}).items():
        shares = safe_float(pos.get("shares"))
        cost_price = safe_float(pos.get("cost_price") or pos.get("price"))
        current = _current_price(symbol, cost_price)
        cost, value = shares * cost_price, shares * current
        total_cost += cost
        total_value += value
        positions.append({"symbol": symbol, "shares": shares, "cost_price": cost_price, "current_price": current, "cost": cost, "value": value, "pnl": value - cost, "pnl_pct": ((value - cost) / cost * 100) if cost else 0})
    return {"portfolio": {**portfolio, "positions": positions}, "total_cost": total_cost, "total_value": total_value, "pnl": total_value - total_cost, "pnl_pct": ((total_value - total_cost) / total_cost * 100) if total_cost else 0}


def _current_price(symbol: str, fallback: float) -> float:
    try:
        data = fetch_realtime(symbol)
        return safe_float(data.get("price") or data.get("close") or data.get("最新价"), fallback)
    except Exception:
        return fallback


def _report_date_from_name(path: Path, mtime: float) -> str:
    """报告展示日期优先取文件名日期，避免复制/生成时间掩盖实际行情日期。"""
    import re
    match = re.search(r"(20\d{2}-\d{2}-\d{2})", path.name)
    if match:
        return match.group(1)
    compact = re.search(r"(20\d{6})", path.name)
    if compact:
        raw = compact.group(1)
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    # mtime 仅作为生成时间，不冒充行情数据日期。
    return "未知日期"


def _list_reports() -> list[dict]:
    reports = []
    for directory in REPORT_DIRS:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
            stat = path.stat()
            report_date = _report_date_from_name(path, stat.st_mtime)
            reports.append({
                "path": str(path), "name": path.name, "date": report_date,
                "generated_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                "type": directory.name, "size": stat.st_size,
            })
    # 研报共享根目录内“日期目录”的 HTML 详细研报（与 /report.html 同源）
    for directory in _report_roots():
        if not directory.exists():
            continue
        html_cands = list(directory.glob("**/A股详细研报_*.html")) + list(directory.glob("**/A股融合研报_*.html"))
        for path in sorted(html_cands, key=lambda p: p.stat().st_mtime, reverse=True):
            stat = path.stat()
            report_date = _report_date_from_name(path, stat.st_mtime)
            reports.append({
                "path": str(path), "name": path.name, "date": report_date,
                "generated_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                "type": "研报共享" if "研报共享" in str(path) else "研究报告", "size": stat.st_size,
            })
    return sorted(reports, key=lambda r: (r["date"] != "未知日期", r["date"], r["generated_at"]), reverse=True)


def _read_report(path_text: str) -> dict:
    if not path_text:
        raise ValueError("path required")
    raw = Path(unquote(path_text))
    if ".." in raw.parts:
        raise ValueError("path traversal not allowed")
    path = raw.expanduser().resolve()
    roots = [p.resolve() for p in REPORT_DIRS + _report_roots() if p.exists()]
    if not any(path == root or root in path.parents for root in roots):
        raise ValueError("path outside report directories")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(str(path))
    return {"path": str(path), "name": path.name, "content": path.read_text(encoding="utf-8", errors="replace")}


_STRATEGY_KEYS = {"fast_ma", "slow_ma", "trend_ma", "long_trend_ma", "volume_ma",
                  "stop_loss_pct", "take_profit_pct", "trail_stop_pct", "max_risk_score_for_new_buy"}


def _strategy_from_query(query: dict[str, list[str]]) -> StrategyConfig:
    overrides = {}
    for key in _STRATEGY_KEYS:
        vals = query.get(key)
        if vals:
            try:
                overrides[key] = float(vals[0])
            except (ValueError, TypeError):
                pass
    if not overrides:
        return DEFAULT_STRATEGY
    base = DEFAULT_STRATEGY.__dict__
    merged = {k: overrides.get(k, v) for k, v in base.items()}
    return StrategyConfig(**{k: type(base[k])(merged[k]) for k in base})


def _get_defaults() -> dict:
    return {"strategy": {k: v for k, v in DEFAULT_STRATEGY.__dict__.items()},
            "portfolio": {k: v for k, v in DEFAULT_PORTFOLIO.__dict__.items()}}


WATCHLIST_PATH = ROOT / "quant_web" / "watchlist.json"


def _get_watchlist() -> dict:
    if WATCHLIST_PATH.exists():
        return {"ok": True, "symbols": json.loads(WATCHLIST_PATH.read_text(encoding="utf-8"))}
    return {"ok": True, "symbols": []}


def _save_watchlist(symbols: list[str]) -> None:
    WATCHLIST_PATH.write_text(json.dumps(symbols, ensure_ascii=False), encoding="utf-8")


SERVER_START_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
SERVER_STARTED_AT = time.time()


def _get_system_status() -> dict:
    """系统状态 API：返回服务健康度、cron 状态、最近报告、预警历史"""
    import subprocess
    from datetime import datetime, timezone, timedelta

    cst = timezone(timedelta(hours=8))
    now = datetime.now(cst).strftime("%Y-%m-%d %H:%M")

    result = {
        "ok": True,
        "server_alive": True,
        "server_time": now,
        "started_at": SERVER_START_TIME,
        "uptime_seconds": max(0, int(time.time() - SERVER_STARTED_AT)),
        "uptime": str(timedelta(seconds=max(0, int(time.time() - SERVER_STARTED_AT)))),
        "cron_total": 0,
        "cron_enabled": 0,
        "cron_jobs": [],
        "recent_reports": [],
        "alert_history": [],
        "data_source_status": "不确定",
        "data_freshness": "-",
        "last_report_date": "-",
        "last_report_size": "-",
        "last_alert": "-",
        "alert_count": 0,
    }

    # ── 扫描最近报告：统一使用 REPORT_DIRS，日期以文件名为准 ──
    recent_reports = _list_reports()[:5]
    for report in recent_reports:
        result["recent_reports"].append({
            "date": report["date"], "generated_at": report["generated_at"],
            "path": report["path"], "size": f"{report['size'] // 1024}KB",
        })
    if recent_reports:
        result["last_report_date"] = recent_reports[0]["date"]
        result["last_report_size"] = f"{recent_reports[0]['size'] // 1024}KB"

    # ── 扫描预警日志 ──
    alert_log = Path("/tmp/quant-scan-alert.log")
    if alert_log.exists():
        try:
            lines = alert_log.read_text(encoding="utf-8", errors="replace").strip().split("\n")
            # 找最近10条JSON行
            alerts = []
            for line in reversed(lines[-100:]):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        d = json.loads(line)
                        if isinstance(d, dict) and d.get("triggered", 0) > 0:
                            alerts.append({
                                "time": d.get("ts", "-")[:16],
                                "text": f'{d.get("triggered",0)}只触发'
                            })
                    except Exception as e:
                        logging.getLogger(__name__).error(f"[server] 操作失败: {e}", exc_info=True)
                    if len(alerts) >= 5:
                        break
            result["alert_history"] = alerts
            result["alert_count"] = len(alerts)
            if alerts:
                result["last_alert"] = alerts[0]["time"]
        except Exception as e:
            logging.getLogger(__name__).error(f"[server] 操作失败: {e}", exc_info=True)

    # ── cron 状态：直读 OpenClaw 状态库，避免 CLI 超时把 12 个任务显示为 0/0 ──
    cron_error = None
    jobs = []
    try:
        state_db = Path(os.environ.get("QUANT_STATE_DB", "quant_state.sqlite"))
        with sqlite3.connect(f"file:{state_db}?mode=ro", uri=True, timeout=5) as con:
            for row in con.execute(
                "SELECT COALESCE(display_name,name), enabled, schedule_kind, schedule_expr, every_ms, "
                "next_run_at_ms, last_run_at_ms, last_run_status, last_error, consecutive_errors "
                "FROM cron_jobs WHERE enabled=1 ORDER BY next_run_at_ms"
            ):
                name, enabled, kind, expr, every_ms, next_ms, last_ms, last_status, last_error, consecutive_errors = row
                if not expr and every_ms:
                    expr = f"每 {int(every_ms) // 60000} 分钟" if int(every_ms) >= 60000 else f"每 {int(every_ms) // 1000} 秒"
                jobs.append({"name": name, "displayName": name, "enabled": bool(enabled), "schedule": expr or kind or "-",
                             "nextRunAtMs": next_ms, "lastRunAtMs": last_ms, "lastRunStatus": last_status or "idle",
                             "lastError": (last_error or "")[:180], "consecutiveErrors": int(consecutive_errors or 0),
                             "source": "openclaw-state-db"})
    except Exception as exc:
        cron_error = f"网关状态库读取失败: {str(exc)[:100]}"

    if not jobs:
        try:
            r = subprocess.run(["crontab", "-l"], capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                for line in r.stdout.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("@"):
                        continue
                    parts = line.split(None, 5)
                    if len(parts) < 6:
                        continue
                    sched = " ".join(parts[:5])
                    cmd = parts[5]
                    script = cmd.split("/")[-1].split(".")[0] if "/" in cmd else cmd.split()[0]
                    jobs.append({
                        "name": script,
                        "schedule": sched,
                        "enabled": True,
                        "nextRun": "-",
                        "source": "system-crontab",
                    })
        except Exception as exc:
            cron_error = (cron_error or "") + " | crontab: " + str(exc)[:80]

    result["cron_total"] = len(jobs)
    result["cron_enabled"] = sum(1 for j in jobs if j.get("enabled"))
    for j in jobs:
        name = j.get("displayName") or j.get("name", "-")
        sched = j.get("schedule", "-")
        enabled = j.get("enabled", False)
        next_str = j.get("nextRun", "-")
        if next_str == "-" and j.get("nextRunAtMs"):
            try:
                next_dt = datetime.fromtimestamp(j["nextRunAtMs"] / 1000, tz=cst)
                next_str = next_dt.strftime("%m-%d %H:%M")
            except Exception as e:
                logging.getLogger(__name__).error(f"[server] 操作失败: {e}", exc_info=True)
        result["cron_jobs"].append({
            "name": name,
            "displayName": name,
            "schedule": sched,
            "enabled": enabled,
            "nextRun": next_str,
        })
    if cron_error:
        result["cron_error"] = cron_error

    # ── 数据源状态 ──
    meta_dir = ROOT / "generated" / "a_share_data"
    meta_files = sorted(meta_dir.glob("*-meta.json"), reverse=True)
    if meta_files:
        result["data_freshness"] = meta_files[0].stem
        result["data_source_status"] = "有缓存数据"

    return result


def _warmup_cache() -> None:
    """后台预热慢速查询的缓存，减少用户首次访问等待。"""
    import threading

    def _warm_margin():
        try:
            from quant_system.margin import fetch_margin_summary
            data = fetch_margin_summary()
            _cache_set("margin", data)
        except Exception as e:
            logging.getLogger(__name__).error(f"[server] 操作失败: {e}", exc_info=True)

    def _warm_market_temp():
        try:
            from quant_system.market_temperature import fetch_market_data
            raw = fetch_market_data()
            advance_pct = raw.get("advance_pct", 50)
            temperature = round(advance_pct * 1.0, 1)
            data = {"ok": True, "data": _json_safe({
                "temperature": temperature,
                "advance_pct": advance_pct,
                "advance": raw.get("advance", 0),
                "decline": raw.get("decline", 0),
                "sh_index": raw.get("sh_index", 0),
                "sh_pct": raw.get("sh_pct", 0),
                "sz_index": raw.get("sz_index", 0),
                "sz_pct": raw.get("sz_pct", 0),
                "cy_index": raw.get("cy_index", 0),
                "cy_pct": raw.get("cy_pct", 0),
                "total_amount_yi": raw.get("total_amount_yi", 0),
                "pct_above_ma300": raw.get("pct_above_ma300", 0),
                "ma60_direction": raw.get("ma60_direction", "unknown"),
                "stage": raw.get("stage", "--"),
            })}
            _cache_set("market_temp", data)
        except Exception as e:
            logging.getLogger(__name__).error(f"[server] 操作失败: {e}", exc_info=True)

    def _warm_rag():
        try:
            from quant_system.analysis_core.knowledge_rag import _get_model
            _get_model()
        except Exception as e:
            logging.getLogger(__name__).error(f"[server] 操作失败: {e}", exc_info=True)

    # RAG model loading is optional and can be expensive or incompatible with
    # the installed torch backend; never make it part of service warmup.
    warmups = [("margin", _warm_margin), ("market_temp", _warm_market_temp)]
    for _, fn in warmups:
        t = threading.Thread(target=fn, daemon=True)
        t.start()


def main() -> None:

    import os
    # 降低进程优先级
    try:
        os.nice(10)
    except Exception as e:
        logging.getLogger(__name__).error(f"[server] 操作失败: {e}", exc_info=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [quant-web] %(message)s")

    # 初始化 3.0 基础设施
    try:
        init_cache(ROOT / "config" / "cache.sqlite3")
        init_task_queue(ROOT / "config" / "tasks.sqlite3")
        backtest_console_handlers.register_task_handlers()
        from quant_web.handlers import strategy_lab
        from quant_system.task_queue import register_handler
        register_handler("strategy_lab_run", strategy_lab.run_strategy_job)
        print("[3.0] 缓存层、任务队列、策略实验室任务注册就绪", flush=True)
    except Exception as exc:
        print(f"[3.0] 初始化警告: {exc}", flush=True)

    host = _QUANT_WEB_BIND_HOST
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8600
    if not _IS_LOOPBACK_BIND and not _QUANT_WEB_API_KEY:
        raise SystemExit("[quant-web] 拒绝在非回环地址上以无鉴权模式启动：请设置 QUANT_WEB_API_KEY")

    # 先绑定并进入可探测状态，再做任何可选预热。这样 supervisor 不会把
    # “模型/数据预热较慢”误判成启动失败并重复拉起第二个实例。
    server = ThreadingHTTPServer((host, port), QuantHandler)
    print("[quant-web] 牧云天枢 — SSE /api/events | /api/readyz | /api/catalog", flush=True)
    print(f"      http://{host}:{port}", flush=True)
    _set_ready(True, "serving")

    import threading as _th

    def _background_warmup():
        # liveness/ready mean the HTTP service can answer; warming is surfaced
        # in the phase field without making the UI unavailable.
        _set_ready(True, "warming")
        try:
            from quant_system.data_store import get_store as _pw
            _st = _pw()
            _core_syms = ['600519', '000858', '002714', '601899']
            for _sym in _core_syms:
                try:
                    _df = _st.get(_sym, days=180)
                    state = "数据就绪" if _df is not None and len(_df) > 20 else "数据不足"
                    print(f"[3.0] {state}: {_sym} ({len(_df) if _df is not None else 0}行)", flush=True)
                except Exception as _e:
                    print(f"[3.0] 数据获取失败 {_sym}: {_e}", flush=True)
            _rest = ['601899', '002594', '300750', '600036', '601318', '000333', '600276', '000568', '002415', '000001', '601166', '600900', '600887']
            for _sym in _rest:
                try:
                    _st.get(_sym, days=180)
                except Exception as _e:
                    logging.getLogger(__name__).warning("预加载 %s 跳过: %s", _sym, _e)
            _warmup_cache()
            try:
                from quant_system.analysis_core.knowledge_rag import warm_model as _rag_warm
                _rag_warm()
            except Exception as _e:
                logging.getLogger(__name__).warning("RAG 模型预热跳过: %s", _e)
            print("[3.0] 后台预热完成", flush=True)
        except Exception as _e:
            logging.getLogger(__name__).exception("后台预热失败")
        finally:
            _set_ready(True, "serving")

    _th.Thread(target=_background_warmup, daemon=True, name='quant-web-warmup').start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        from quant_system.task_queue import shutdown as _shutdown_tasks
        _shutdown_tasks()
        print("服务器关闭", flush=True)


def send_feishu_message(title: str, body: str) -> dict:
    full_text = f"{title}\n\n{body}"
    message = f"**{title}**\n\n{body}"
    payload = json.dumps({
        "channel": "feishu",
        "target": ALERT_FEISHU_TARGET,
        "message": message,
        "accountId": "",
    }, ensure_ascii=False)
    try:
        proc = subprocess.run(
            [OPENCLAW_BIN, "message", "send", "--json"],
            cwd=str(ROOT),
            input=payload,
            text=True,
            capture_output=True,
            timeout=30,
        )
        return {"ok": proc.returncode == 0, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def build_alert_body(atype: str, rows: list[dict], market: dict) -> str:
    lines = []
    breadth = market.get("breadth", {})
    hot = market.get("hot_metrics", {})
    lines.append(f"📊 市场状态：风险{market.get('risk','?')} {market.get('risk_level','?')}")
    lines.append(f"全A等权 {_pct_fmt(breadth.get('全A等权涨跌幅%'))} · 上涨占比 {_pct_fmt(breadth.get('上涨占比%'))} · 热股等权 {_pct_fmt(hot.get('热股等权涨跌幅%'))}")
    lines.append("")
    if atype == "scan":
        alerts = []
        for r in rows:
            act = str(r.get("action", ""))
            symbol = str(r.get("symbol", ""))
            close = r.get("close", "-")
            score = r.get("composite_score", "-")
            risk = r.get("risk_score", "-")
            if act in ("BUY_WATCH", "SELL_OR_AVOID", "MARKET_RISK_BLOCKED"):
                icon = "🔴" if act == "BUY_WATCH" else "🟢" if act == "SELL_OR_AVOID" else "🟡"
                alerts.append(f"{icon} {symbol} {act} 收盘{close} 综合分{score} 风险分{risk}")
            elif risk and isinstance(risk, (int, float)) and risk >= 70:
                alerts.append(f"⚠️ {symbol} 高风险{risk} 收盘{close}")
        if alerts:
            lines.append("### 触发预警")
            lines.extend(alerts)
        else:
            lines.append("✅ 扫描结果无异常触发。")
        lines.append("")
        # 全部扫描结果摘要
        lines.append(f"扫描 {len(rows)} 只，BUY_WATCH: {sum(1 for r in rows if r.get('action')=='BUY_WATCH')} 只，SELL_OR_AVOID: {sum(1 for r in rows if r.get('action')=='SELL_OR_AVOID')} 只，HOLD: {sum(1 for r in rows if r.get('action')=='HOLD')} 只")
    elif atype == "backtest":
        lines.append("### 回测结果")
        for r in rows:
            symbol = str(r.get("symbol", ""))
            ret = r.get("total_return_pct", "-")
            dd = r.get("max_drawdown_pct", "-")
            sharpe = r.get("sharpe", "-")
            win = r.get("win_rate_pct", "-")
            trades = r.get("trade_count", 0)
            lines.append(f"{symbol} 收益{ret}% 回撤{dd}% Sharpe{sharpe} 胜率{win}% 交易{trades}次")
    return "\n".join(lines)


def _pct_fmt(v) -> str:
    if v is None:
        return "-"
    return f"{float(v):.2f}%"


if __name__ == "__main__":
    main()
