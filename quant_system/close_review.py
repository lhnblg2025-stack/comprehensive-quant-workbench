"""
收盘复盘报告 — 整合市场、持仓、信号、风险的一站式每日复盘。

用法:
  python3 -m quant_system.close_review             # 完整版(~5s)
  python3 -m quant_system.close_review --brief     # 简要版(~2s)
  python3 -m quant_system.close_review --file ~/复盘.md
"""

from __future__ import annotations
import logging

import sys
import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

CST = timezone(timedelta(hours=8))


def generate_review(brief: bool = False) -> str:
    """Generate close review report."""
    parts = []
    now = datetime.now(CST)
    date_str = now.strftime("%Y-%m-%d %H:%M")
    # V4.1 fix: now.hour是整数, 11.5比较无意义。精确判断交易时段含午休排除
    t = now.hour * 60 + now.minute
    is_open = now.weekday() < 5 and ((9*60+30 <= t <= 11*60+30) or (13*60 <= t <= 15*60))
    session = "🟢 盘中快照" if is_open else "🔴 盘后复盘"
    sep = lambda: parts.append("")

    parts.append(f"# {session} — {now.strftime('%Y-%m-%d')} ({now.strftime('%A')})")

    # ── 1. Market Temperature ──
    parts.append("\n## 🌡️ 市场温度\n")
    try:
        from quant_system.market_temperature import fetch_market_data, classify_cycle
        t0 = _time.time()
        data = fetch_market_data()
        if data:
            cycle = classify_cycle(data)
            parts.append(f"  周期: {cycle['cycle_segment']}  |  风险: {cycle['risk_posture']}  |  "
                        f"温度: {cycle['temperature']}/100")
            parts.append(f"  上证: {data.get('sh_pct',0):+.2f}%  |  "
                        f"深证: {data.get('sz_pct',0):+.2f}%  |  "
                        f"成交: {data.get('total_amount_yi',0):.0f}亿  |  "
                        f"涨跌: {data.get('advance',0)}:{data.get('decline',0)}")
            print(f"[review] market data: {_time.time()-t0:.1f}s")
        else:
            parts.append("  ⚠️ 市场数据获取失败\n")
    except Exception as e:
        parts.append(f"  ⚠️ 市场: {str(e)[:60]}\n")
    sep()

    # ── 2. Portfolio ──
    parts.append("## 📦 持仓表现\n")
    try:
        from quant_system.trade_db import get_positions, get_summary, refresh_prices
        refresh_prices()
        positions = get_positions()
        summary = get_summary()
        if not positions:
            parts.append("📭 **当前空仓**\n")
        else:
            pnl = summary["total_pnl_pct"]
            pnl_s = f"🟢 +{pnl:.2f}%" if pnl >= 0 else f"🔴 {pnl:.2f}%"
            parts.append(f"📊 {summary['num_positions']}只 | "
                        f"成本{summary['total_cost']:,.0f} | "
                        f"市值{summary['total_value']:,.0f} | "
                        f"盈亏 {pnl_s}\n")
            if not brief:
                parts.append("| 股票 | 持股 | 成本→现价 | 盈亏 | 信号 |")
                parts.append("|:----|:---:|:--------:|:---:|:----:|")
                for p in positions[:8]:
                    curr = p.get("current_price") or p["cost_price"]
                    pp = p.get("pnl_pct", 0) or 0
                    e = "🟢" if pp >= 0 else "🔴"
                    s = p.get("signal_type", "") or "-"
                    parts.append(f"| {p['symbol']} {p['name'][:6]} | {p['shares']} | "
                                f"{p['cost_price']:.2f}→{curr:.2f} | {e}{pp:+.2f}% | {s} |")
    except Exception as e:
        parts.append(f"  ⚠️ 持仓: {str(e)[:60]}\n")
    sep()

    # ── 3. Today's Trades ──
    if not brief:
        parts.append("## 📋 今日交易\n")
        try:
            from quant_system.trade_db import get_trades
            trades = get_trades(days=1)
            if trades:
                for t in trades:
                    tt = "🟢买" if t["trade_type"] == "buy" else "🔴卖"
                    parts.append(f"  {t['trade_date']} {tt} {t['symbol']} {t['name'][:6]} "
                                f"{t['shares']}股 @{t['price']:.2f} 金额{t['total_amount']:,.0f}")
            else:
                parts.append("  📭 今日无交易\n")
        except Exception as e:
            parts.append(f"  ⚠️ 交易: {str(e)[:60]}\n")
        sep()

    # ── 4. Risk Check ──
    parts.append("## 🛡️ 风险检查\n")
    try:
        from quant_system.risk_budget import compute_risk_dashboard, run_scenario
        risk = compute_risk_dashboard()
        parts.append(f"  评分: {risk['score']}/100  |  状态: {risk['status']}")
        for w in risk.get("warnings", [])[:3]:
            parts.append(f"  {w}")
        if not risk.get("warnings") and risk["num_positions"] > 0:
            parts.append("  ✅ 无异常预警")
        # Stop-monitoring
        from quant_system.trade_db import check_stops
        alerts = check_stops()
        if alerts:
            parts.append(f"  🚨 **{len(alerts)}个止损/止盈!**")
            for a in alerts[:3]:
                parts.append(f"    {a['message']}")
        # Scenario
        if risk["num_positions"] > 0:
            s = run_scenario(10)
            parts.append(f"\n  📉 大盘跌10%: {s['total_loss_pct']:+.2f}% (≈¥{s['total_loss']:+,.0f})"
                        f"{' 🔴 注意' if s['total_loss_pct'] < -5 else ' ✅ 可控'}")
    except Exception as e:
        parts.append(f"  ⚠️ 风险: {str(e)[:60]}\n")
    sep()

    # ── 5. Action Plan ──
    parts.append("## 📋 明日计划\n")
    try:
        from quant_system.trade_db import get_positions
        positions = get_positions()
        if positions:
            # Check near stop
            near_stop = []
            for p in positions:
                curr = p.get("current_price")
                cost = p["cost_price"]
                if curr and curr > 0:
                    pnl_pct = (curr / cost - 1) * 100
                    sp = p.get("stop_pct", -8)
                    if pnl_pct <= sp + 2:
                        near_stop.append((p, pnl_pct, sp))
            if near_stop:
                parts.append("  🚨 **接近止损:**")
                for p, pp, sp in near_stop:
                    parts.append(f"    {p['symbol']} {p['name'][:6]} 盈亏{pp:.2f}% ≤ 止损{sp:.0f}%")
            else:
                parts.append("  ✅ 各持仓止损空间充足\n")
            parts.append("\n  💡 关注盘中监控预警，按信号执行")
        else:
            parts.append("  💡 空仓期重点关注买入机会清单和行业轮动\n")
    except Exception as e:
        logging.getLogger(__name__).error(f"[close_review] 操作失败: {e}", exc_info=True)

    parts.append("\n\n---\n")
    parts.append(f"*报告: {date_str} | 量化系统自动复盘*\n")

    return "\n".join(parts)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="收盘复盘")
    parser.add_argument("--brief", action="store_true", help="简要版(~2s)")
    parser.add_argument("--file", type=str, default="", help="输出到文件")
    args = parser.parse_args()
    report = generate_review(args.brief)
    if args.file:
        Path(args.file).write_text(report, encoding="utf-8")
        print(f"✅ 已保存: {args.file}")
    else:
        print(report)
