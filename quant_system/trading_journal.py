"""
交易日志分析层 — 高于 trade_db 的分析功能。

功能:
- 月度/年度盈亏报告
- 最佳/最差交易分析
- 持仓时间分布
- 信号成功率趋势
- 海龟式仓位建议

用法:
  python3 -m quant_system.trading_journal --report
  python3 -m quant_system.trading_journal --sizing --symbol 600519 --price 1500
"""

from __future__ import annotations
import logging

import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

# P2-Q18-fix: numpy 延迟导入——缺失时 np=None，整个模块（含 CLI）仍可导入，
# 与 trading_system「numpy/sklearn 延迟导入」约定保持一致；使用点走纯 Python 降级。
try:
    import numpy as np
except Exception:
    np = None

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

CST = timezone(timedelta(hours=8))

# Ensure DB initialized
from quant_system.trade_db import (
    init_db, get_trades,
    get_signal_performance, get_summary,
    format_positions, format_trades, format_signal_performance,
)


def _init_check():
    init_db()


# ════════════════════════════════════════════════════════════════
# 仓位建议 (海龟交易法则核心公式改良)
# ════════════════════════════════════════════════════════════════

def suggest_position_size(
    symbol: str,
    current_price: float,
    account_total: float = 0,
    risk_percent: float = 2.0,
    atr_percent: float = None,
    max_shares: int = None,
) -> dict:
    """
    海龟交易法则改良版 — A股适用仓位建议。

    Args:
        symbol: 股票代码
        current_price: 当前价格
        account_total: 总资金（默认从持仓总成本估算）
        risk_percent: 单笔风险占总资金百分比 (默认2%)
        atr_percent: 平均真实波幅%(如不提供则估算)
        max_shares: 最大股数限制

    Returns:
        {price, position_size, risk_amount, risk_percent,
         atr_pct, target_stop, explanation}
    """
    if account_total <= 0:
        summary = get_summary()
        account_total = max(summary["total_value"], summary["total_cost"], 100000)

    # ATR estimate: use historical data if possible
    if atr_percent is None:
        try:
            import akshare as ak
            # P2-Q18-fix: 起始日期动态生成（近 120 天）。硬编码 start_date="20260101"
            # 会在 2027+ 数据窗口漂移、2025 运行时取到空数据导致 ATR 静默回退 3%。
            start_date = (datetime.now() - timedelta(days=120)).strftime("%Y%m%d")
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                    start_date=start_date, adjust="qfq")
            if df is not None and len(df) > 20:
                closes = df["收盘"].values
                highs = df["最高"].values
                lows = df["最低"].values

                # Calculate ATR as percentage of price
                tr_values = []
                for i in range(1, min(21, len(closes))):
                    tr = max(
                        highs[-i] - lows[-i],
                        abs(highs[-i] - closes[-i-1]),
                        abs(lows[-i] - closes[-i-1])
                    )
                    tr_values.append(tr)

                if tr_values:
                    sample = tr_values[-20:]
                    # P2-Q18-fix: np 缺失时降级为纯 Python 均值，保持行为一致。
                    atr_val = float(np.mean(sample)) if np is not None else sum(sample) / len(sample)
                    atr_percent = round(atr_val / current_price * 100, 2)
        except Exception as exc:
            # P2-Q18-fix: ATR 估算失败记录告警，避免静默吞错回退默认波动率。
            import logging
            logging.getLogger(__name__).warning(
                f"ATR 估算失败({symbol})，将回退默认波动率 3%: {exc}"
            )

    if atr_percent is None:
        # Fallback: use 3% as default volatility for large-cap A-shares
        atr_percent = 3.0

    # Risk per trade in currency
    risk_amount = account_total * (risk_percent / 100)

    # Stop loss distance (2x ATR)
    stop_distance_pct = atr_percent * 2
    stop_distance = current_price * (stop_distance_pct / 100)

    # Position size (海龟 formula: Unit = Risk / (N × Price))
    # Simplified for stocks: shares = risk_amount / stop_distance
    if stop_distance > 0:
        suggested_shares = max(100, int(risk_amount / stop_distance / 100) * 100)
    else:
        suggested_shares = 100

    # Cap by max_shares
    if max_shares:
        suggested_shares = min(suggested_shares, max_shares)

    # Also cap by single-stock concentration (max 20% of account)
    concentration_cap = int(account_total * 0.20 / current_price / 100) * 100
    if concentration_cap > 0:
        suggested_shares = min(suggested_shares, concentration_cap)

    position_value = suggested_shares * current_price
    allocation_pct = round(position_value / account_total * 100, 1)

    target_stop = round(current_price - stop_distance, 2)

    # Check existing position
    from quant_system.trade_db import get_position
    existing = get_position(symbol)

    return {
        "symbol": symbol,
        "price": current_price,
        "account_total": int(account_total),
        "risk_percent": risk_percent,
        "risk_amount": int(risk_amount),
        "atr_percent": atr_percent,
        "stop_distance_pct": round(stop_distance_pct, 2),
        "target_stop": target_stop,
        "suggested_shares": suggested_shares,
        "position_value": int(position_value),
        "allocation_pct": allocation_pct,
        "existing_shares": existing["shares"] if existing else 0,
        "existing_cost": existing["total_cost"] if existing else 0,
    }


def format_sizing(s: dict) -> str:
    """Format position sizing advice."""
    lines = [
        f"📐 **仓位建议 — {s['symbol']} @{s['price']:.2f}**",
        f"  总资金: {s['account_total']:,.0f}",
        f"  单笔风险: {s['risk_percent']:.1f}% = {s['risk_amount']:,}",
        f"  ATR(20): {s['atr_percent']:.2f}% (波动率)",
        f"  止损距离: {s['stop_distance_pct']:.2f}% → 止损价{s['target_stop']:.2f}",
        f"",
        f"  建议仓位: **{s['suggested_shares']}股** (约{s['allocation_pct']:.1f}%仓位)",
        f"  仓位金额: {s['position_value']:,}",
    ]
    if s['existing_shares'] > 0:
        lines.append(f"")
        lines.append(f"  当前已有: {s['existing_shares']}股 (成本{s['existing_cost']:,.0f})")
        if s['existing_shares'] >= s['suggested_shares']:
            lines.append(f"  ⚠️ 已超过建议仓位,不建议再加")
        else:
            add = s['suggested_shares'] - s['existing_shares']
            if add >= 100:
                lines.append(f"  💡 还可加仓 {add}股 (到建议仓位)")
            else:
                lines.append(f"  ➖ 仓位已接近建议值")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# 交易表现分析
# ════════════════════════════════════════════════════════════════

def get_best_worst_trades(top_n: int = 5) -> dict:
    """Get best and worst trades by P&L.

    P2-Q18-fix(M165/L169): 原实现按 `ORDER BY trade_date DESC` 取 buys[-1]（卖出前
    最早一笔买入价）作为全部卖出成本，多轮买卖/部分平仓时归因错误，且盈亏忽略
    持仓数量匹配；原 SELECT 中一个从未被使用的关联子查询（死代码）已删除。
    现改为逐 symbol 按时间顺序做 FIFO 配对：买入建仓（先进先出），卖出按最早
    剩余成本匹配，盈亏 = Σ(卖出价 - 对应买入价) × 匹配股数。
    """
    from quant_system.trade_db import _conn

    with _conn() as db:
        rows = db.execute("""
            SELECT symbol, name, trade_type, shares, price, trade_date
            FROM trades
            WHERE trade_type IN ('buy', 'sell')
            ORDER BY symbol, trade_date, id
        """).fetchall()

    # FIFO 逐 symbol 配对：买入入队，卖出按最早成本出队匹配
    sells = []
    unmatched = 0  # V11: 未匹配卖出（数据不完整告警）—— 无买入记录可匹配的卖出笔数
    fifo_queue: dict[str, list[tuple[float, int]]] = defaultdict(list)  # symbol -> [(price, qty)]
    name_by_symbol: dict[str, str] = {}
    for r in rows:
        r = dict(r)
        sym = r["symbol"]
        name_by_symbol[sym] = r["name"] or sym
        qty = int(r["shares"])
        price = float(r["price"])
        if r["trade_type"] == "buy":
            fifo_queue[sym].append((price, qty))
            continue
        remaining = qty
        matched_cost = 0.0
        matched_qty = 0
        while remaining > 0 and fifo_queue[sym]:
            lot_price, lot_qty = fifo_queue[sym][0]
            take = min(lot_qty, remaining)
            matched_cost += take * lot_price
            matched_qty += take
            remaining -= take
            if lot_qty - take > 0:
                fifo_queue[sym][0] = (lot_price, lot_qty - take)
            else:
                fifo_queue[sym].pop(0)
        if matched_qty == 0:
            unmatched += 1
        if matched_qty > 0:
            avg_buy = matched_cost / matched_qty
            pnl = (price - avg_buy) * matched_qty
            pnl_pct = (price / avg_buy - 1) * 100 if avg_buy else 0.0
            sells.append({
                "symbol": sym,
                "name": name_by_symbol[sym],
                "date": r["trade_date"],
                "buy_price": round(avg_buy, 2),
                "sell_price": price,
                "shares": matched_qty,
                "pnl_pct": round(pnl_pct, 2),
                "pnl": round(pnl, 2),
            })

    sells.sort(key=lambda x: x["pnl_pct"], reverse=True)

    return {
        "best": sells[:top_n],
        "worst": sells[-top_n:][::-1] if sells else [],
        "total_closed": len(sells),
        "unmatched": unmatched,  # V11: 未匹配卖出（数据不完整告警）
    }


def get_monthly_performance(year: int = None) -> list[dict]:
    """Monthly P&L breakdown.

    P2-Q18-fix(M166): 原 `net = sell_total - buy_total` 是「买卖总额差」，即月度
    现金流差（未平仓部分与持仓成本未计入），标注为净盈亏具有误导性。现重命名为
    `net_cash_flow` 并保留 `net` 兼容别名；月度已实现盈亏请结合 FIFO 平仓分析
    （见 get_best_worst_trades）。
    """
    if year is None:
        year = datetime.now(CST).year

    from quant_system.trade_db import _conn
    with _conn() as db:
        rows = db.execute("""
            SELECT trade_type, shares, price, total_amount, trade_date, notes
            FROM trades
            WHERE strftime('%Y', trade_date) = ?
            ORDER BY trade_date
        """, (str(year),)).fetchall()

    months = defaultdict(lambda: {"buy_total": 0, "sell_total": 0, "net_pnl": 0})
    for r in rows:
        r = dict(r)
        month = r["trade_date"][:7]  # YYYY-MM
        if r["trade_type"] == "buy":
            months[month]["buy_total"] += r["total_amount"]
        else:
            months[month]["sell_total"] += r["total_amount"]

    results = []
    for m in sorted(months.keys()):
        data = months[m]
        net_cash_flow = data["sell_total"] - data["buy_total"]  # 现金流差，非已实现盈亏
        results.append({
            "month": m,
            "buy_amount": round(data["buy_total"], 2),
            "sell_amount": round(data["sell_total"], 2),
            "net_cash_flow": round(net_cash_flow, 2),
            "net": round(net_cash_flow, 2),  # 兼容别名，避免破坏既有调用方
        })

    return results


def generate_report() -> str:
    """Full trading journal report."""
    _init_check()
    parts = []

    # 1. Portfolio summary
    parts.append("═" * 55)
    parts.append("📊 持仓与交易日志报告")
    parts.append("═" * 55)
    parts.append(f"  {datetime.now(CST).strftime('%Y-%m-%d %H:%M')}")
    parts.append("")

    parts.append(format_positions() + "\n")

    # 2. Signal performance
    sig_stats = get_signal_performance()
    if sig_stats:
        parts.append(format_signal_performance(sig_stats) + "\n")

    # 3. Best/worst trades
    bwt = get_best_worst_trades(5)
    if bwt["best"]:
        parts.append(f"🏆 **最佳交易 (Top {len(bwt['best'])}):**")
        for t in bwt["best"]:
            pnl_emoji = "🟢" if t["pnl_pct"] > 0 else "🔴"
            parts.append(f"  {pnl_emoji} {t['date']} {t['symbol']} {t['name'][:8]} "
                         f"买入{t['buy_price']}→卖出{t['sell_price']} "
                         f"({t['pnl_pct']:+.2f}% {t['pnl']:+.0f})")
        parts.append("")

    if bwt["worst"]:
        parts.append(f"💀 **最差交易 (Bottom {len(bwt['worst'])}):**")
        for t in bwt["worst"]:
            pnl_emoji = "🟢" if t["pnl_pct"] > 0 else "🔴"
            parts.append(f"  {pnl_emoji} {t['date']} {t['symbol']} {t['name'][:8]} "
                         f"买入{t['buy_price']}→卖出{t['sell_price']} "
                         f"({t['pnl_pct']:+.2f}% {t['pnl']:+.0f})")
        parts.append("")
        parts.append(f"  已平仓交易: {bwt['total_closed']}笔\n")

    # 4. Monthly performance
    mp = get_monthly_performance()
    if mp:
        parts.append("📅 **月度交易量:**")
        for m in mp:
            net_emoji = "🟢" if m["net_cash_flow"] >= 0 else "🔴"
            parts.append(f"  {m['month']}: 买入{m['buy_amount']:>10,.0f} "
                         f"卖出{m['sell_amount']:>10,.0f} "
                         f"现金流差{net_emoji}{m['net_cash_flow']:+.0f}"
                         f"(买卖总额差,非已实现盈亏)")
        parts.append("")

    # 5. Recent trades
    recent = get_trades(days=7)
    if recent:
        parts.append(format_trades(recent))

    return "\n".join(parts)


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="交易日志与仓位管理")
    parser.add_argument("--report", action="store_true", help="完整报告")
    parser.add_argument("--status", action="store_true", help="持仓概览")
    parser.add_argument("--trades", action="store_true", help="最近交易")
    parser.add_argument("--signals", action="store_true", help="信号胜率")
    parser.add_argument("--sizing", action="store_true", help="仓位建议")
    parser.add_argument("--symbol", default="600519")
    parser.add_argument("--price", type=float, default=0)
    parser.add_argument("--account", type=float, default=0, help="总资金")
    args = parser.parse_args()

    _init_check()

    if args.report:
        print(generate_report())

    elif args.status:
        print(format_positions())
        print(f"\n💡 建议: python3 -m quant_system.trading_journal --report 查看完整报告")

    elif args.trades:
        trades = get_trades(days=30)
        print(format_trades(trades))

    elif args.signals:
        stats = get_signal_performance()
        print(format_signal_performance(stats))

    elif args.sizing:
        price = args.price
        if price <= 0:
            # Try to get current price
            try:
                from quant_system.watchlist import fetch_quotes
                quotes = fetch_quotes([args.symbol])
                if quotes:
                    price = quotes[0].get("price") or quotes[0].get("close", 0)
            except Exception as e:
                logging.getLogger(__name__).error(f"[trading_journal] 操作失败: {e}", exc_info=True)
        if price <= 0:
            print("❌ 需要 --price 指定价格或先刷新价格")
            sys.exit(1)
        s = suggest_position_size(args.symbol, price, args.account)
        print(format_sizing(s))

    else:
        # Default: quick overview
        summary = get_summary()
        print(f"📦 持仓: {summary['num_positions']}只 | "
              f"成本{summary['total_cost']:,.0f} | "
              f"市值{summary['total_value']:,.0f} | "
              f"盈亏{summary['total_pnl_pct']:+.2f}%")
        print(f"\n💡 可用命令:")
        print(f"  --status    查看持仓明细")
        print(f"  --report    完整报告(含信号胜率+最佳/最差交易)")
        print(f"  --trades    最近交易记录")
        print(f"  --signals   信号胜率统计")
        print(f"  --sizing    仓位建议(--symbol --price)")
