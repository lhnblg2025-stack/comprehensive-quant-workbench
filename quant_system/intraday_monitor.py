"""A股盘中监控与预警系统 — 409只主板≥300亿全量扫描。

运行逻辑:
- 交易时段每 2.5 分钟轮询全部 409 只自选股
- 每轮：腾讯批量拉取实时行情（~5秒），检测价格/量能/位置异常
- 每 5 轮（~12分钟）：对异动股做技术信号扫描（CCI/均线/布林带）
- 发现机会时推送到飞书

Usage:
  python3 -m quant_system.intraday_monitor                  # 前台交互运行
  python3 -m quant_system.intraday_monitor --daemon         # 后台运行
  python3 -m quant_system.intraday_monitor --status         # 查看状态
  python3 -m quant_system.intraday_monitor --stop           # 停止
"""

from __future__ import annotations
import logging

import os
import pickle
import signal
import subprocess
import sys
import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── Quant system imports ──
# P2-P2-Q10-L036-fix: 清理未使用的 filter_stocks / _52w_position / json / Counter
from quant_system.watchlist import fetch_quotes, _cap_tier
from quant_system.data import fetch_daily
from quant_system.indicators import add_technical_indicators
from quant_system.opportunity import scan_real_time, format_real_time
from quant_system.config import DEFAULT_STRATEGY
from quant_system.utils import now_cst

# ── Config ──
STATE_PATH = ROOT / "generated" / "quant_system" / "monitor_state.pkl"
PID_FILE = ROOT / "generated" / "quant_system" / "monitor.pid"
LOCK_FILE = ROOT / "generated" / "quant_system" / "monitor.lock"
DELIVERY_SCRIPT = ROOT / "scripts" / "report_delivery.py"
CST = timezone(timedelta(hours=8))

# Alert thresholds
ALERT_SURGE_PCT = 3.0           # 单股涨跌幅超3%
ALERT_DANGER_PCT = 5.0          # 严重涨跌幅超5%
ALERT_ACCEL_CHANGE = 0.8        # 相比前次轮询价格变动超0.8%
ALERT_VOLUME_RATIO = 2.0        # 量比 > 2
SCAN_INTERVAL_SEC = 150         # 2.5分钟
DEEP_SCAN_EVERY_N = 5           # 每5轮做一次技术信号深度扫描
DEEP_SCAN_TOP_N = 30            # 深度扫描取涨跌前30只

# ── State ──
class MonitorState:
    def __init__(self):
        self.last_prices: dict[str, float] = {}
        self.last_alerts: dict[str, float] = {}
        # P2-P2-Q10-L038-fix: 移除从未读写的 daily_cache/daily_signal 死字段
        # (实际日线缓存用的是模块级 _daily_cache)

    def save(self):
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_PATH, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls):
        if STATE_PATH.exists():
            try:
                with open(STATE_PATH, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logging.getLogger(__name__).error(f"[intraday_monitor] 操作失败: {e}", exc_info=True)
        return cls()


# ── Trading time checks ──

def is_trading_hours() -> bool:
    t = now_cst()
    if t.weekday() >= 5:
        return False
    hm = t.hour * 100 + t.minute
    return (930 <= hm <= 1130) or (1300 <= hm <= 1457)


def seconds_until_next_open() -> int:
    t = now_cst()
    hm = t.hour * 100 + t.minute
    # Q10-fix: datetime(t.year,...) 构造的是 naive datetime，与 aware 的 t 相减
    # 抛 TypeError（午休/盘前/盘后守护进程必崩）。统一用 replace() 构造 aware。
    if hm < 930:
        return int((t.replace(hour=9, minute=30, second=0, microsecond=0) - t).total_seconds())
    elif hm < 1130:
        return 0  # Already trading
    elif hm < 1300:
        return int((t.replace(hour=13, minute=0, second=0, microsecond=0) - t).total_seconds())
    elif hm < 1500:
        return 0  # Already trading
    else:
        next_day = t + timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        next_open = next_day.replace(hour=9, minute=30, second=0, microsecond=0)
        return int((next_open - t).total_seconds())


# ── Daily data cache with LRU ──

_daily_cache: dict[str, tuple[float, Any]] = {}  # symbol -> (timestamp, df_with_indicators)


def _get_daily(symbol: str, refresh: bool = False) -> Any:
    """Get daily K-line with indicators. Cached 30 min."""
    now = _time.time()
    if not refresh and symbol in _daily_cache:
        ts, df = _daily_cache[symbol]
        if now - ts < 1800:  # 30 min cache
            return df
    try:
        # P1-Q10-fix: 原 start="" 使 data.py 中 strptime("","%Y%m%d") 抛 ValueError，
        # 且 add_technical_indicators(df) 缺必填 config 抛 TypeError —— 双重必挂，
        # 深度技术扫描（Tier-2）恒 None。改传实际起始日期并补 DEFAULT_STRATEGY。
        df = fetch_daily(symbol, start="20200101", use_cache=not refresh)
        if df is not None and len(df) > 0:
            df = add_technical_indicators(df, DEFAULT_STRATEGY)
            _daily_cache[symbol] = (now, df)
            return df
    except Exception as e:
        # P1-Q10-fix: 降级必须可见 —— 不再静默吞异常
        logger.warning(f"[monitor] ⚠️ {symbol} 日线获取失败: {e}")
    return None


# ── Alert dispatch via report_delivery.py ──

def send_alert(title: str, body: str, level: str = "info"):
    """Send alert via report_delivery.py -> Feishu/WeChat.

    2026-08-14 审计修复: Tier2 技术信号扫描/Tier3 买入机会此前无冷却，
    每 5-12.5 分钟同内容重推刷屏。现经 alert_dedup 网关按 (title+body) 签名
    去重（同签名 30 分钟静默）；danger（止损）直通不静默。"""
    if not DELIVERY_SCRIPT.exists():
        logger.error(f"[monitor] ❌ Delivery script not found: {DELIVERY_SCRIPT}")
        return
    try:
        # 去重网关（可用时）；失败则直发兜底
        if level != "danger":
            try:
                sys.path.insert(0, str(ROOT / "scripts"))
                from alert_dedup import get_deduper, make_sig, normalize_price  # noqa: PLC0415
                sig = make_sig("intraday", normalize_price(title + "\n" + body))
                if not get_deduper().should_send("intraday", sig, level=level,
                                                 same_sig_min_gap_s=1800):
                    logger.info(f"[monitor] 🔇 去重静默: {title}")
                    return
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[monitor] 去重失败直发兜底: {e}")

        prefix = {"info": "ℹ️", "warn": "⚠️", "danger": "🚨"}.get(level, "")
        msg = f"{prefix} **{title}**\n{body}"
        subprocess.run(
            [sys.executable, str(DELIVERY_SCRIPT), "-",
             "--deliver", "--title", f"盘中预警: {title}"],
            input=msg, text=True, timeout=30,
            capture_output=True,
        )
        logger.info(f"[monitor] ✅ Alert sent: {title}")
    except Exception as e:
        logger.error(f"[monitor] ❌ Alert failed: {e}")


# ── Opportunity detection (lightweight, Tencent data only) ──

def quick_scan(symbol: str, q: dict, state: MonitorState) -> list[dict]:
    """
    Quick check on a single quote for immediate alerts.
    Uses only Tencent quote data (no K-line fetch needed).
    """
    alerts = []
    sym = q["symbol"]
    price = q["price"]
    prev_close = q.get("prev_close", price)
    change_pct = q.get("change_pct", 0)
    volume = q.get("volume", 0)
    last_price = state.last_prices.get(sym, prev_close)
    now = _time.time()
    cooldown = 1800  # 30 min between alerts for same stock
    last_alert = state.last_alerts.get(sym, 0)

    def _cooldown_ok(alert_key: str) -> bool:
        """P2-Q10-M030-fix: 统一冷却——52周新高/新低/放量此前无冷却, 每 2.5 分钟
        可重复推送轰炸飞书; 现按 类型+sym 维度在 state.last_alerts 中统一冷却。"""
        return now - state.last_alerts.get(alert_key, 0) > cooldown

    # 1. Large move
    if abs(change_pct) >= ALERT_SURGE_PCT and now - last_alert > cooldown:
        direction = "🚀大涨" if change_pct > 0 else "💀大跌"
        level = "danger" if abs(change_pct) >= ALERT_DANGER_PCT else "warn"
        tier = _cap_tier(q.get("market_cap_yi", 0))
        alerts.append({
            "sym": sym, "name": q["name"], "type": "large_move",
            "msg": f"{direction} {q['name']}({sym}) {change_pct:+.2f}%  现¥{price:.2f}  市值{tier}",
            "level": level,
        })
        state.last_alerts[sym] = now

    # 2. Acceleration
    if last_price > 0 and abs(change_pct) < ALERT_SURGE_PCT:
        accel = abs(price - last_price) / last_price * 100
        key_accel = f"{sym}:acceleration"
        # V11 审计修复（Medium）: acceleration 原无冷却，价格每轮波动 ≥0.8% 就重复
        # 触发刷屏（52w_high 等已有 _cooldown_ok，唯独此分支漏加）。
        if accel >= ALERT_ACCEL_CHANGE and _cooldown_ok(key_accel):
            alerts.append({
                "sym": sym, "name": q["name"], "type": "acceleration",
                "msg": f"⚡ {q['name']}({sym}) 快速波动 {accel:+.2f}% (上轮¥{last_price:.2f}→本轮¥{price:.2f})",
                "level": "warn",
            })
            state.last_alerts[key_accel] = now

    # 3. 52-week extreme (P2-Q10-M030-fix: 新增冷却, 防每轮重复推送)
    high52 = q.get("high_52w", 0)
    low52 = q.get("low_52w", 0)
    key_52h = f"{sym}:52w_high"
    if high52 > 0 and price >= high52 * 0.99 and price > 5 and _cooldown_ok(key_52h):
        alerts.append({
            "sym": sym, "name": q["name"], "type": "52w_high",
            "msg": f"🔝 {q['name']}({sym}) 接近52周新高 ¥{price:.2f} (高{high52:.2f}) PE={q.get('pe_ttm',0):.1f}",
            "level": "info",
        })
        state.last_alerts[key_52h] = now
    key_52l = f"{sym}:52w_low"
    if low52 > 0 and price <= low52 * 1.02 and _cooldown_ok(key_52l):
        alerts.append({
            "sym": sym, "name": q["name"], "type": "52w_low",
            "msg": f"🔻 {q['name']}({sym}) 接近52周新低 ¥{price:.2f} (低{low52:.2f}) PB={q.get('pb',0):.2f}",
            "level": "warn",
        })
        state.last_alerts[key_52l] = now

    # 4. Volume anomaly
    # P1-Q10-fix: fetch_quotes 返回字段中无 volume_ratio（watchlist.py 已核实），
    # 原 `q.get("volume_ratio", 0)` 恒 0 → 量比≥2 预警形同虚设。
    # 改用当日累计量 / 过去20日日均量，并按盘中已开盘分钟折算为全天量比。
    vol_ratio = 0.0
    ddf = _get_daily(sym)
    if ddf is not None and len(ddf) >= 21 and volume > 0:
        avg_vol = float(ddf["volume"].iloc[-21:-1].mean())  # 过去20个已完成交易日
        if avg_vol > 0:
            _t = now_cst()
            _hm = _t.hour * 100 + _t.minute
            if 930 <= _hm <= 1130:
                _elapsed = _t.hour * 60 + _t.minute - 570   # 09:30起
            elif 1300 <= _hm <= 1457:
                _elapsed = 120 + (_t.hour * 60 + _t.minute - 780)  # 上午120分+午后
            else:
                _elapsed = 240  # 非交易时段按全天估算
            vol_ratio = volume / (avg_vol * max(_elapsed, 1) / 240)
    key_vol = f"{sym}:volume"
    if vol_ratio >= ALERT_VOLUME_RATIO and _cooldown_ok(key_vol):
        alerts.append({
            "sym": sym, "name": q["name"], "type": "volume",
            "msg": f"📊 {q['name']}({sym}) 放量异常 量比{vol_ratio:.1f} 换手{q.get('turnover',0):.1f}%  {change_pct:+.2f}%",
            "level": "warn",
        })
        state.last_alerts[key_vol] = now

    return alerts


def deep_signal(symbol: str) -> dict | None:
    """
    Run technical indicator check on a stock (needs daily K-line).
    Returns signal dict or None if data unavailable.
    """
    df = _get_daily(symbol)
    if df is None or len(df) < 30:
        return None

    row = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else row

    signal = {
        "symbol": symbol,
        "price": float(row.get("close", 0)),
        "change_pct": float(row.get("pct_chg", 0)),
    }

    # MA position
    # P1-Q10-fix: 日线指标实际列名为 ma_fast/ma_slow/ma_trend/ma_long_trend
    # （indicators.py 中 DEFAULT_STRATEGY 的 20/60/144/300 周期），原 `ma_20/ma_60/...`
    # 列不存在 → `if col in row` 恒 False，MA 位置段永远为空。
    _ma_cols = {20: "ma_fast", 60: "ma_slow", 144: "ma_trend", 300: "ma_long_trend"}
    for _ma, _col in _ma_cols.items():
        if _col in row:
            signal[f"ma{_ma}"] = float(row[_col])

    # CCI
    if "cci_20" in row:
        cci = float(row["cci_20"])
        cci_prev = float(prev.get("cci_20", cci))
        signal["cci"] = cci
        if cci > 100 and cci_prev <= 100:
            signal["cci_signal"] = "进入超买"
        elif cci < -100 and cci_prev >= -100:
            signal["cci_signal"] = "进入超卖"
        elif cci > 100:
            signal["cci_signal"] = "超买区"
        elif cci < -100:
            signal["cci_signal"] = "超卖区"
        else:
            signal["cci_signal"] = "正常"

    # RSI
    if "rsi_14" in row:
        signal["rsi"] = float(row["rsi_14"])
        signal["rsi_signal"] = "超买" if signal["rsi"] > 70 else ("超卖" if signal["rsi"] < 30 else "正常")

    # MACD
    if "macd_hist" in row:
        hist = float(row["macd_hist"])
        hist_prev = float(prev.get("macd_hist", 0))
        signal["macd_hist"] = hist
        if hist > 0 and hist_prev <= 0:
            signal["macd_signal"] = "MACD转正(🟢)"
        elif hist < 0 and hist_prev >= 0:
            signal["macd_signal"] = "MACD转负(🔴)"

    # BOLL position
    if "boll_upper" in row and "boll_lower" in row:
        upper = float(row["boll_upper"])
        lower = float(row["boll_lower"])
        mid = float(row.get("boll_mid", (upper + lower) / 2))
        price = signal["price"]
        if price >= upper:
            signal["boll_signal"] = "突破BOLL上轨"
        elif price <= lower:
            signal["boll_signal"] = "跌破BOLL下轨"
        elif price > mid:
            signal["boll_signal"] = "BOLL中轨上方"
        else:
            signal["boll_signal"] = "BOLL中轨下方"

    # Volume ratio (from daily data)
    if "volume_ratio" in row and row["volume_ratio"] != 0:
        signal["volume_ratio"] = float(row["volume_ratio"])

    return signal


# ════════════════════════════════════════════════════════════════
# Main monitoring loop
# ════════════════════════════════════════════════════════════════

def monitor_loop(state: MonitorState):
    """Main loop — runs during trading hours."""
    loop_count = 0
    logger.info(f"[monitor] 🟢 启动 409只主板≥300亿 盘中监控")
    logger.info(f"[monitor]    每{SCAN_INTERVAL_SEC//60}分钟轮询, 每{(SCAN_INTERVAL_SEC*DEEP_SCAN_EVERY_N)//60}分钟深度扫描")
    logger.info(f"[monitor]    预警: 涨跌≥{ALERT_SURGE_PCT}%/加速≥{ALERT_ACCEL_CHANGE}%/量比≥{ALERT_VOLUME_RATIO}")
    logger.info("")

    while True:
        if not is_trading_hours():
            secs = seconds_until_next_open()
            if secs > 0:
                logger.info(f"[monitor] 🌙 非交易时段, 距开盘 {secs//60} 分钟")
                for _ in range(min(secs, 600)):
                    _time.sleep(1)
                continue
            else:
                # Between sessions, check more frequently
                _time.sleep(60)
                continue

        # ── Trading hours ──
        loop_count += 1
        now_str = datetime.now(CST).strftime("%H:%M:%S")
        logger.info(f"\n{'='*50}")
        logger.info(f"[monitor] 🔄 第{loop_count}轮 — {now_str}")

        # Fetch all 409 stocks (batch, ~5s)
        t0 = _time.time()
        quotes = fetch_quotes(force=True)
        valid = [q for q in quotes if q.get("price", 0) > 0]
        fetch_time = _time.time() - t0
        logger.info(f"[monitor] 📡 {len(valid)}只行情 ({fetch_time:.1f}s)")

        if not valid:
            logger.warning("[monitor] ⚠️ 无有效数据")
            _time.sleep(SCAN_INTERVAL_SEC)
            continue

        # Market pulse
        up = sum(1 for q in valid if q["change_pct"] > 0)
        down = sum(1 for q in valid if q["change_pct"] < 0)
        avg = sum(q["change_pct"] for q in valid) / len(valid)
        logger.info(f"[monitor] 📊 自选: 🟢{up}涨 🔴{down}跌 ⚪{len(valid)-up-down}平 平均{avg:+.2f}%")

        # === Tier 1: Quick scan all stocks ===
        all_alerts = []
        for q in valid:
            alerts = quick_scan(q["symbol"], q, state)
            all_alerts.extend(alerts)

        # Update last prices
        for q in valid:
            if q["price"] > 0:
                state.last_prices[q["symbol"]] = q["price"]

        # === Tier 2: Deep signal scan (every N cycles) ===
        if loop_count % DEEP_SCAN_EVERY_N == 0:
            logger.info(f"[monitor] 🔬 深度技术扫描 — 涨跌前{DEEP_SCAN_TOP_N}只")
            # Pick top movers for signal analysis
            sorted_q = sorted(valid, key=lambda q: abs(q["change_pct"]), reverse=True)
            deep_targets = [q["symbol"] for q in sorted_q[:DEEP_SCAN_TOP_N]]

            signal_results = {}
            for sym in deep_targets:
                sig = deep_signal(sym)
                if sig:
                    signal_results[sym] = sig

            # Generate signal-based alerts
            cci_entries = []
            macd_changes = []
            boll_breaks = []
            for sym, sig in signal_results.items():
                price = sig.get("price", 0)
                name = next((q["name"] for q in valid if q["symbol"] == sym), sym)
                mv = next((q["market_cap_yi"] for q in valid if q["symbol"] == sym), 0)
                pct = sig.get("change_pct", 0)

                # CCI zone transition
                if sig.get("cci_signal") in ("进入超买", "进入超卖"):
                    cci_entries.append(f"  {sym} {name} CCI{sig['cci_signal']} ({sig.get('cci',0):.0f}) ¥{price:.2f} {pct:+.2f}%")

                # MACD crossover
                macd_sig = sig.get("macd_signal", "")
                if "转正" in macd_sig or "转负" in macd_sig:
                    macd_changes.append(f"  {sym} {name} {macd_sig} ¥{price:.2f} {mv:.0f}亿")

                # BOLL breakout
                boll_sig = sig.get("boll_signal", "")
                if "突破" in boll_sig or "跌破" in boll_sig:
                    boll_breaks.append(f"  {sym} {name} {boll_sig} ¥{price:.2f} {mv:.0f}亿")

            # Build deep scan alert body
            deep_lines = []
            if cci_entries:
                deep_lines.append(f"📊 CCI信号 ({len(cci_entries)}只):")
                deep_lines.extend(cci_entries[:8])
            if macd_changes:
                deep_lines.append(f"\n📈 MACD交叉 ({len(macd_changes)}只):")
                deep_lines.extend(macd_changes[:8])
            if boll_breaks:
                deep_lines.append(f"\n📉 BOLL突破 ({len(boll_breaks)}只):")
                deep_lines.extend(boll_breaks[:8])

            if deep_lines:
                body = "\n".join(deep_lines)
                send_alert(f"🔬 技术信号扫描 ({len(signal_results)}只)", body, level="info")
                logger.info(f"[monitor] 🔬 深度扫描: {len(cci_entries)}只CCI, {len(macd_changes)}只MACD, {len(boll_breaks)}只BOLL")

        # === Stop-loss monitoring (every cycle) ===
        try:
            from quant_system.trade_db import check_stops
            stop_alerts = check_stops()
            if stop_alerts:
                for a in stop_alerts:
                    all_alerts.append({
                        "sym": a["symbol"],
                        "name": a["name"],
                        "type": a["type"].lower(),
                        "msg": a["message"],
                        "level": "danger",
                    })
        except Exception as e:
            logging.getLogger(__name__).error(f"[intraday_monitor] 操作失败: {e}", exc_info=True)

        # === Dispatch tier-1 alerts ===
        if all_alerts:
            danger = [a for a in all_alerts if a["level"] == "danger"]
            warn = [a for a in all_alerts if a["level"] == "warn"]
            info = [a for a in all_alerts if a["level"] == "info"]

            for tlist, level, prefix in [
                (danger, "danger", "🚨 严重预警"),
                (warn, "warn", "⚠️ 波动预警"),
                (info, "info", "ℹ️ 信号提示"),
            ]:
                if not tlist:
                    continue
                body = "\n".join(f"• {a['msg']}" for a in tlist)
                send_alert(f"{prefix} ({len(tlist)}条)", body, level=level)

            logger.info(f"[monitor] 🔔 发送{len(all_alerts)}条预警 (严重{len(danger)} 波动{len(warn)} 提示{len(info)})")
        else:
            logger.info("[monitor] ✅ 无预警")

        # Market pulse summary every cycle
        if loop_count % 2 == 0:
            best = max(valid, key=lambda q: q["change_pct"])
            worst = min(valid, key=lambda q: q["change_pct"])
            logger.info(f"[monitor] 📊 最佳: {best['name']} +{best['change_pct']:.2f}%  |  最差: {worst['name']} {worst['change_pct']:.2f}%")

        # === Tier 3: Real-time opportunity scan (every cycle, ~8s) ===
        if is_trading_hours() and valid:
            try:
                # P2-P2-Q10-M031-fix: 复用本轮已拉取的 quotes, 不再二次全量拉取 409 只;
                # K线预热已改后台线程(见 scan_real_time), 不再阻塞监控循环。
                opps_data = scan_real_time(min_score=2, quotes=valid)
                opps_list = opps_data.get('results', []) if isinstance(opps_data, dict) else opps_data
                if opps_list:
                    rt_text = format_real_time(opps_list, full=False)
                    logger.info(f"[monitor] 🎯 {len(opps_list)}个买入机会")
                    # Send as info alert
                    if loop_count % 2 == 0:  # only every other cycle to reduce noise
                        first_5 = "\n".join(rt_text.split("\n")[:7])
                        send_alert(f"🎯 买入机会 ({len(opps_list)}个)", first_5, level="info")
            except Exception as e:
                logger.warning(f"[monitor] ⚠️ 机会扫描失败: {e}")

        # Save state
        state.save()

        # Sleep
        _time.sleep(SCAN_INTERVAL_SEC)


# ════════════════════════════════════════════════════════════════
# Daemon management
# ════════════════════════════════════════════════════════════════

def _is_running() -> int | None:
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)
            return pid
        except (ValueError, OSError, ProcessLookupError):
            PID_FILE.unlink(missing_ok=True)
    return None


def start_daemon():
    if _is_running():
        logger.info("[monitor] Already running")
        return
    pid = os.fork()
    if pid > 0:
        PID_FILE.write_text(str(pid))
        logger.info(f"[monitor] Started (PID {pid})")
        return
    os.setsid()
    pid2 = os.fork()
    if pid2 > 0:
        sys.exit(0)
    PID_FILE.write_text(str(os.getpid()))
    _run()


def _run():
    state = MonitorState.load()
    try:
        monitor_loop(state)
    except KeyboardInterrupt:
        logger.info("\n[monitor] Stopped")
    except Exception:
        logger.error("[monitor] 监控循环异常", exc_info=True)
    finally:
        PID_FILE.unlink(missing_ok=True)
        LOCK_FILE.unlink(missing_ok=True)


def stop_daemon():
    pid = _is_running()
    if not pid:
        logger.info("[monitor] Not running")
        return
    os.kill(pid, signal.SIGTERM)
    logger.info(f"[monitor] Stopped (PID {pid})")
    PID_FILE.unlink(missing_ok=True)


def show_status():
    pid = _is_running()
    if pid:
        logger.info(f"[monitor] Running (PID {pid})")
        if STATE_PATH.exists():
            size = STATE_PATH.stat().st_size
            logger.info(f"[monitor] State: {STATE_PATH} ({size//1024}KB)")
    else:
        logger.info("[monitor] Not running")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import argparse
    parser = argparse.ArgumentParser(description="A股409只盘中监控")
    parser.add_argument("--daemon", action="store_true", help="后台运行")
    parser.add_argument("--status", action="store_true", help="查看状态")
    parser.add_argument("--stop", action="store_true", help="停止")
    args = parser.parse_args()

    if args.stop:
        stop_daemon()
    elif args.status:
        show_status()
    elif args.daemon:
        start_daemon()
    else:
        print("[monitor] Starting foreground (Ctrl+C to stop)...\n")
        _run()
