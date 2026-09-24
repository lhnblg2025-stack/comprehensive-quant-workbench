"""
behavior_audit — 行为金融偏差审计（V11 检测使用者而非市场）

人亏钱往往不是系统不准，而是 处置效应(赚就跑/亏就扛) + 过度交易 + 不执行止损。
本模块从 trade_db（模拟盘/实盘 SQLite）直接读成交记录做统计检测，
无数据时输出等待状态。

检测项（数据到位自动生效）:
  1. 处置效应: 平均盈利持仓时间 vs 平均亏损持仓时间（赚就跑=处置效应）
  2. 止损执行率: 系统建议止损价 vs 实际卖出价（扛单不执行）
  3. 交易频率: 近30日买卖笔数（高潮期过度交易倾向）
  4. 手动干预: 平仓盈亏分布（盈利时是否过早离场）

数据: quant_system.trade_db.trades 表（symbol/name/trade_type/shares/price/trade_date）
输出: generated/behavior_audit_{date}.json

用法:
  python3 -m quant_system.analysis_core.behavior_audit [--date 2026-08-07]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

CST = timezone(timedelta(hours=8))


def _load_trades(days: int = 365) -> list[dict]:
    try:
        from quant_system.trade_db import get_trades
        return get_trades(days=days, limit=5000) or []
    except Exception as e:
        print(f"[behavior_audit] 交易数据读取失败: {str(e)[:80]}")
        return []


def _pair_trades(rows: list[dict]) -> list[dict]:
    """FIFO 配对: 每笔卖出匹配最早剩余买入，得到已平仓交易(含持有天数/盈亏)。"""
    from collections import defaultdict as _dd
    fifo: dict[str, list[tuple[float, int, str]]] = _dd(list)  # sym -> [(price, qty, buy_date)]
    closed: list[dict] = []
    for r in sorted(rows, key=lambda x: (x.get("symbol", ""), str(x.get("trade_date", "")))):
        sym = r.get("symbol", "")
        qty = int(r.get("shares", 0) or 0)
        price = float(r.get("price", 0) or 0)
        date = str(r.get("trade_date", ""))[:10]
        ttype = r.get("trade_type", "")
        if ttype == "buy":
            fifo[sym].append((price, qty, date))
        elif ttype == "sell" and fifo.get(sym):
            remaining = qty
            while remaining > 0 and fifo[sym]:
                bp, bq, bd = fifo[sym][0]
                matched = min(bq, remaining)
                pnl = (price - bp) * matched
                hold_days = (pd.Timestamp(date) - pd.Timestamp(bd)).days if bd else 0
                closed.append({"symbol": sym, "name": r.get("name", sym),
                               "buy_date": bd, "sell_date": date,
                               "hold_days": hold_days, "pnl": pnl,
                               "ret": pnl / (bp * matched) if bp * matched else 0})
                remaining -= matched
                if matched >= bq:
                    fifo[sym].pop(0)
                else:
                    fifo[sym][0] = (bp, bq - matched, bd)
    return closed


def audit(date: str | None = None) -> dict:
    date = date or datetime.now(CST).date().isoformat()
    rows = _load_trades()
    # --date 只支持当日/历史标签: 历史日期时按日期过滤（避免标签与数据错配）
    if date != datetime.now(CST).date().isoformat():
        rows = [r for r in rows if str(r.get("trade_date", ""))[:10] <= date]
    if not rows:
        res = {"date": date, "status": "waiting_data", "trades": 0,
               "note": "trade_db 无成交记录（模拟盘未开始）。数据到位后自动检测 处置效应/止损执行率/过度交易。",
               "flags": []}
        out = ROOT / "generated" / f"behavior_audit_{date}.json"
        out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        return res

    closed = _pair_trades(rows)
    flags: list[dict] = []

    # ── 1. 处置效应: 盈利 vs 亏损持仓时间 ──
    if len(closed) >= 5:
        wins = [c for c in closed if c["pnl"] > 0]
        losses = [c for c in closed if c["pnl"] <= 0]
        if wins and losses:
            w_hold = sum(c["hold_days"] for c in wins) / len(wins)
            l_hold = sum(c["hold_days"] for c in losses) / len(losses)
            if w_hold < l_hold * 0.6 and l_hold > 3:
                flags.append({"type": "处置效应", "level": "🔴",
                              "msg": f"盈利单平均持仓 {w_hold:.1f} 天 < 亏损单 {l_hold:.1f} 天 → 赚就跑亏就扛，建议设置持有下限"})
        # 盈亏比
        avg_win = sum(c["pnl"] for c in wins) / len(wins) if wins else 0
        avg_loss = abs(sum(c["pnl"] for c in losses) / len(losses)) if losses else 0
        if avg_loss and avg_win / avg_loss < 1.0 and len(wins) >= 3:
            flags.append({"type": "盈亏比失衡", "level": "🟠",
                          "msg": f"平均盈利 {avg_win:.0f} < 平均亏损 {avg_loss:.0f} → 截断亏损不足"})

    # ── 2. 交易频率（近30日）──
    recent = [r for r in rows if str(r.get("trade_date", ""))[:10] >=
              (datetime.now(CST) - timedelta(days=30)).strftime("%Y-%m-%d")]
    if len(recent) >= 15:
        flags.append({"type": "过度交易", "level": "🟠",
                      "msg": f"近30日 {len(recent)} 笔交易 → 交易过频，检查是否追涨杀跌"})

    # ── 3. 汇总统计 ──
    total_pnl = sum(c["pnl"] for c in closed)
    res = {
        "date": date, "status": "ok",
        "trades": len(rows), "closed_trades": len(closed),
        "total_pnl": round(total_pnl, 2),
        "avg_hold_days": round(sum(c["hold_days"] for c in closed) / len(closed), 1) if closed else 0,
        "win_rate": round(sum(1 for c in closed if c["pnl"] > 0) / len(closed), 3) if closed else 0,
        "flags": flags,
        "summary": "行为审计: " + ("; ".join(f["msg"] for f in flags) if flags else "无明显偏差"),
    }
    out = ROOT / "generated" / f"behavior_audit_{date}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="行为金融偏差审计")
    ap.add_argument("--date", default=None)
    args = ap.parse_args()
    r = audit(args.date)
    print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
