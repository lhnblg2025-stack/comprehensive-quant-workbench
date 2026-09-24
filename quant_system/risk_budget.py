"""
风险预算管理 — 持仓风险全景仪表盘。

功能:
  - 组合风险总览 (β, VaR, 最大回撤, 集中度)
  - 止损止盈自动检查 + 预警
  - 情景分析 ("如果大盘跌10%会怎样")
  - 海龟式仓位校验 (当前仓位 vs 建议仓位)
  - 精美报告输出

用法:
  python3 -m quant_system.risk_budget
  python3 -m quant_system.risk_budget --scenario --drop 10
  python3 -m quant_system.risk_budget --stops
  python3 -m quant_system.risk_budget --report
"""

from __future__ import annotations
import logging

import sys
import time as _time
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

CST = timezone(timedelta(hours=8))


# ════════════════════════════════════════════════════════════════
# 底层数据读取
# ════════════════════════════════════════════════════════════════

def _load_portfolio() -> tuple[list[dict], dict]:
    """Load positions and summary from trade_db."""
    from quant_system.trade_db import get_positions, get_summary
    positions = get_positions()
    summary = get_summary()
    return positions, summary


# P2-Q21-fix: 组合 Beta 估算缓存（每只持仓 beta + 组合加权 beta），
# 供 compute_risk_dashboard 的 VaR 修正 与 run_scenario 的情景分析复用，
# 避免每次调用都对每只股票抓取历史行情。
_BETA_CACHE: dict[str, Any] = {
    "key": None, "per_stock": {}, "portfolio": 1.0, "note": "", "ts": 0.0,
}
_BETA_TTL = 600.0  # 秒


def _position_value(p: dict) -> float:
    """持仓市值：优先 total_value，否则 shares × cost_price。"""
    return p.get("total_value") or (p.get("shares", 0) * p.get("cost_price", 0))


def _compute_beta_estimates(positions: list[dict]) -> tuple[dict[str, float], float, str]:
    """估算各持仓 Beta 与组合加权 Beta（相对沪深300，按日期对齐）。

    P2-Q21-fix: 原 VaR/情景分析隐含 beta=1.0（用指数波动率直接代理组合），
    高/低 beta 持仓下系统性偏差。此处用个股日收益与沪深300日收益按日期内连接
    对齐后回归/协方差估算 beta，再按持仓市值加权。

    Returns:
        (per_symbol_beta, portfolio_beta, note)
      - per_symbol_beta: {symbol: beta}，仅含估算成功的个股
      - portfolio_beta: 按持仓市值加权；全部不可得时返回 1.0（保守假设）
      - note: 非空说明存在降级假设（数据不足/异常），供输出文案披露
    """
    symbols = sorted(str(p["symbol"]) for p in positions if p.get("symbol"))
    key = "|".join(symbols)
    now = _time.time()
    if _BETA_CACHE["key"] == key and now - _BETA_CACHE["ts"] < _BETA_TTL:
        return _BETA_CACHE["per_stock"], _BETA_CACHE["portfolio"], _BETA_CACHE["note"]

    per_stock: dict[str, float] = {}
    portfolio_beta = 1.0
    note = ""
    try:
        import akshare as ak
        import pandas as pd
        idx = ak.stock_zh_index_daily(symbol="sh000300")
        if idx is None or len(idx) < 60 or "close" not in idx.columns or "date" not in idx.columns:
            note = "基准指数数据不足，按 beta=1 保守估计"
        else:
            idx = idx.copy()
            idx["date"] = pd.to_datetime(idx["date"])
            idx = idx.set_index("date").sort_index()
            mkt = idx["close"].pct_change().dropna()
            total_val = sum(_position_value(p) for p in positions) or 1.0
            for p in positions:
                symbol = str(p["symbol"])
                try:
                    df = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="qfq")
                    if df is None or len(df) < 30 or "收盘" not in df.columns or "日期" not in df.columns:
                        continue
                    df = df.copy()
                    df["日期"] = pd.to_datetime(df["日期"])
                    df = df.set_index("日期").sort_index()
                    s_ret = df["收盘"].pct_change().dropna()
                    aligned = pd.DataFrame({"s": s_ret, "m": mkt}).dropna().iloc[-250:]
                    if len(aligned) < 30:
                        continue
                    var_m = float(np.var(aligned["m"].values))
                    if var_m < 1e-12:
                        continue
                    cov = float(np.cov(aligned["s"].values, aligned["m"].values)[0, 1])
                    per_stock[symbol] = cov / var_m
                except Exception as e:
                    logging.getLogger(__name__).error(f"[risk_budget] 操作失败: {e}", exc_info=True)
                    continue

            if not per_stock:
                note = "个股历史数据不可得，按 beta=1 保守估计"
                portfolio_beta = 1.0
            else:
                weighted = 0.0
                matched_w = 0.0
                for p in positions:
                    symbol = str(p["symbol"])
                    if symbol in per_stock:
                        w = _position_value(p) / total_val
                        weighted += w * per_stock[symbol]
                        matched_w += w
                portfolio_beta = weighted / matched_w if matched_w > 0 else 1.0
    except Exception:
        note = "Beta 估算异常，按 beta=1 保守估计"
        portfolio_beta = 1.0

    _BETA_CACHE.update(key=key, per_stock=per_stock, portfolio=portfolio_beta, note=note, ts=now)
    return per_stock, portfolio_beta, note


# ════════════════════════════════════════════════════════════════
# 风险计算
# ════════════════════════════════════════════════════════════════

def compute_risk_dashboard() -> dict:
    """
    Full risk dashboard.
    
    Returns:
        {portfolio, positions[], scenarios, concentration, warnings[], score}
    """
    positions, summary = _load_portfolio()

    if not positions:
        return {
            "num_positions": 0,
            "status": "空仓",
            "total_cost": 0,
            "total_value": 0,
            "score": 100,  # No risk = perfect score
            "warnings": [],
            "suggestions": ["当前空仓，无风险暴露"],
        }

    total_cost = summary["total_cost"]
    total_value = summary["total_value"]
    total_pnl_pct = summary["total_pnl_pct"]

    # --- Concentration risk ---
    total = total_value or total_cost
    concentration = []
    herfindahl = 0  # Herfindahl-Hirschman Index
    for p in positions:
        val = p.get("total_value") or (p["shares"] * p["cost_price"])
        weight = val / total if total > 0 else 0
        herfindahl += weight ** 2
        concentration.append({
            "symbol": p["symbol"],
            "name": p["name"],
            "weight_pct": round(weight * 100, 1),
            "val": val,
        })
    concentration.sort(key=lambda x: -x["weight_pct"])
    hhi = round(herfindahl * 10000, 0)  # Scale to 0-10000

    # --- VaR estimation (historical simulation, simplified) ---
    # P2-Q21-fix: 原实现用沪深300指数波动率直接代理组合波动率（隐含 beta=1），
    # 高/低 beta 持仓下 VaR 系统性偏差。改用持仓加权 beta 修正：
    # vol_eff = vol × β_portfolio，并在输出文案注明"基于指数波动率近似"。
    var_95_daily = 1.645  # V4.1 fix: 单尾 z-score Φ(1.645)≈0.95 (确认95%VaR正确)
    var_99_daily = 2.326  # V4.1 fix: 单尾 z-score Φ(2.326)≈0.99 (确认99%VaR正确)

    # Estimate daily portfolio volatility from CSI 300
    try:
        import akshare as ak
        import pandas as pd
        idx = ak.stock_zh_index_daily(symbol="sh000300")
        if idx is not None and len(idx) > 20:
            rets = np.diff(idx["close"].values) / idx["close"].values[:-1]
            vol = np.std(rets[-60:])  # 60-day rolling vol
        else:
            vol = 0.015  # 1.5% daily vol default
    except Exception:
        vol = 0.015

    # P2-Q21-fix: 持仓加权 beta 修正（beta<0 罕见，按 0 截断避免 VaR 为负）
    _, port_beta, beta_note = _compute_beta_estimates(positions)
    vol_eff = vol * max(float(port_beta), 0.0)

    var_95 = round(vol_eff * var_95_daily * total * 100, 0) / 100
    var_99 = round(vol_eff * var_99_daily * total * 100, 0) / 100

    # --- Max sector concentration ---
    sector_alloc = summary.get("sector_allocation", {})
    max_sector = max(sector_alloc.values()) if sector_alloc else 0

    # --- Warnings ---
    warnings = []
    if hhi > 2500:
        warnings.append(f"🔴 持仓高度集中(HHI={hhi:.0f})，单一股票占比过高")
    elif hhi > 1500:
        warnings.append(f"🟡 持仓集中度偏高(HHI={hhi:.0f})")
    if max_sector > 50:
        warnings.append(f"🔴 {list(sector_alloc.keys())[list(sector_alloc.values()).index(max_sector)]} 行业占比{max_sector:.0f}%，行业集中风险")

    # --- Score (0-100) ---
    score = 100
    if hhi > 1500:
        score -= 15
    if hhi > 2500:
        score -= 10
    if max_sector > 50:
        score -= 10
    if total_pnl_pct < -10:
        score -= 15
    elif total_pnl_pct < -5:
        score -= 8
    score = max(0, min(100, score))

    # Check stops
    from quant_system.trade_db import check_stops
    stop_alerts = check_stops()
    if stop_alerts:
        warnings.append(f"🚨 {len(stop_alerts)} 个持仓触发止损/止盈")

    # Suggestions
    suggestions = []
    if hhi > 2500:
        suggestions.append("建议分散持仓，将单票最大仓位控制在20%以内")
    if max_sector > 50:
        suggestions.append(f"行业集中度偏高，建议控制在40%以内")
    if score < 60:
        suggestions.append("整体风险偏高，建议减仓或增加对冲")
    if not suggestions:
        suggestions.append("✅ 组合结构健康，继续按计划执行")

    return {
        "num_positions": len(positions),
        "status": "🟢 健康" if score >= 70 else ("🟡 关注" if score >= 50 else "🔴 高风险"),
        "total_cost": total_cost,
        "total_value": total_value,
        "total_pnl_pct": total_pnl_pct,
        "score": score,
        "hhi": hhi,
        "var_95": var_95,
        "var_99": var_99,
        # P2-Q21-fix: 日波动率为 beta 修正后的组合有效波动率；附组合 beta 与降级说明
        "vol_daily_pct": round(vol_eff * 100, 2),
        "portfolio_beta": round(port_beta, 2),
        "beta_note": beta_note,
        "concentration": concentration[:5],  # Top 5
        "max_sector_pct": max_sector,
        "sector_allocation": sector_alloc,
        "warnings": warnings,
        "suggestions": suggestions,
        "stop_alerts": [] if not stop_alerts else stop_alerts,
    }


def run_scenario(market_drop_pct: float = 10) -> dict:
    """
    Scenario analysis: what happens if market drops X%.
    """
    positions, summary = _load_portfolio()
    if not positions:
        return {"status": "空仓，不影响"}

    total = summary["total_value"] or summary["total_cost"]
    # P2-Q21-fix: 用持仓加权 beta 修正情景影响（原实现注释"Assume beta = 1.0"，
    # 大盘跌10%情景对高/低 beta 持仓失真）。全部 beta 不可得时降级为 1.0 并披露。
    per_stock_beta, port_beta, beta_note = _compute_beta_estimates(positions)
    results = []

    for p in positions:
        val = p.get("total_value") or (p["shares"] * p["cost_price"])
        cost = p["total_cost"]
        curr_pnl_pct = (val / cost - 1) * 100 if cost > 0 else 0

        # P2-Q21-fix: 个股跌幅 = 大盘跌幅 × 个股 beta（无个股 beta 时用组合 beta）
        beta = per_stock_beta.get(str(p["symbol"]), port_beta)
        new_val = val * (1 - market_drop_pct / 100 * beta)
        new_pnl = new_val - cost
        new_pnl_pct = (new_val / cost - 1) * 100 if cost > 0 else 0

        results.append({
            "symbol": p["symbol"],
            "name": p["name"],
            "shares": p["shares"],
            "beta": round(beta, 2),
            "current_val": val,
            "current_pnl_pct": round(curr_pnl_pct, 2),
            "after_drop_val": round(new_val, 2),
            "after_drop_pnl": round(new_pnl, 2),
            "after_drop_pnl_pct": round(new_pnl_pct, 2),
        })

    total_after = sum(r["after_drop_val"] for r in results)
    total_cost = sum(p["total_cost"] for p in positions)
    total_loss = total_after - total_cost
    total_loss_pct = (total_after / total_cost - 1) * 100 if total_cost > 0 else 0

    # P2-Q21-fix: 情景结果补充对冲建议（原实现无）
    hedging_suggestions: list[str] = []
    if port_beta > 1.2:
        hedging_suggestions.append(
            f"组合β={port_beta:.2f}>1.2，大盘下跌情景受损放大，"
            f"可考虑股指期货空单/买入看跌期权对冲"
        )
    if port_beta > 1.5:
        hedging_suggestions.append("高β持仓偏重，建议降低高β个股权重或增加对冲头寸")
    elif port_beta < 0.8:
        hedging_suggestions.append(
            f"组合β={port_beta:.2f}<0.8，防御性较强，大盘下跌情景冲击有限"
        )

    return {
        "scenario": f"大盘下跌{market_drop_pct:.0f}%",
        "current_total": total,
        "after_total": round(total_after, 2),
        "total_loss": round(total_loss, 2),
        "total_loss_pct": round(total_loss_pct, 2),
        "portfolio_beta": round(port_beta, 2),
        "beta_note": beta_note,
        "hedging_suggestions": hedging_suggestions,
        "positions": results,
    }


# ════════════════════════════════════════════════════════════════
# 精美报告输出
# ════════════════════════════════════════════════════════════════

def _score_bar(score: int) -> str:
    """Generate visual score bar."""
    filled = max(1, score // 10)
    empty = 10 - filled
    return "█" * filled + "░" * empty


def format_dashboard(d: dict) -> str:
    """Beautiful risk dashboard report."""
    lines = [
        "═" * 56,
        "🛡️  风险预算仪表盘",
        "═" * 56,
        f"  状态: {d['status']}  |  持仓: {d['num_positions']}只  |  "
        f"风险评分: {d['score']}/100",
    ]

    if d["num_positions"] == 0:
        lines.append("")
        lines.append("  📭 当前空仓，无风险暴露")
        lines.append("")
        lines.append("💡 小贴士：空仓也有空仓的风险——踏空风险。")
        lines.append("   建议保持对市场的关注，准备好买入清单。")
        return "\n".join(lines)

    # Portfolio overview
    lines += [
        "",
        f"📊 **组合概览**",
        f"  总成本: ¥{d['total_cost']:,.2f}",
        f"  总市值: ¥{d['total_value']:,.2f}",
    ]
    pnl = d.get("total_pnl_pct", 0)
    if pnl >= 0:
        lines.append(f"  总盈亏: 🟢 **+{pnl:.2f}%**")
    else:
        lines.append(f"  总盈亏: 🔴 **{pnl:.2f}%**")
    lines.append("")

    # Risk score
    lines += [
        f"🏆 **风险评分: {d['score']}/100**",
        f"  {_score_bar(d['score'])}",
    ]
    if d['score'] >= 80:
        lines.append("  ✅ 组合风险可控")
    elif d['score'] >= 60:
        lines.append("  ⚠️ 风险适中，关注集中度")
    else:
        lines.append("  🚨 风险偏高，建议减仓")
    lines.append("")

    # VaR
    # P2-Q21-fix: 文案注明"基于指数波动率×β近似"，并披露组合 beta 与降级说明
    port_beta = d.get("portfolio_beta", 1.0)
    lines += [
        "📉 **在险价值 (VaR)**",
        f"  ⚡ 日波动率: {d['vol_daily_pct']:.2f}% (沪深300近60日 × 组合β={port_beta:.2f})",
        f"  95% VaR: **¥{d['var_95']:,.0f}** (单日最大预期亏损)",
        f"  99% VaR: **¥{d['var_99']:,.0f}** (极端情况下)",
        f"  💡 意味: 95%概率单日亏损不超过 ¥{d['var_95']:,.0f} (基于指数波动率×β近似)",
        "",
    ]
    if d.get("beta_note"):
        lines.append(f"  ⚠️ {d['beta_note']}")
        lines.append("")

    # Concentration
    lines += [
        "🎯 **持仓集中度**",
        f"  Herfindahl指数: {d['hhi']:.0f}",
    ]
    hhi = d['hhi']
    if hhi > 2500:
        lines.append(f"    🚨 高度集中 (风险!)")
    elif hhi > 1500:
        lines.append(f"    ⚠️ 中度集中")
    else:
        lines.append(f"    ✅ 分散良好")

    lines.append(f"  最大行业占比: {d['max_sector_pct']:.1f}%")
    lines.append("")

    # Top holdings
    lines.append(f"📋 **前5大持仓**")
    for i, c in enumerate(d["concentration"]):
        bar = "█" * max(1, int(c["weight_pct"] / 3))
        lines.append(f"  {i+1}. {c['symbol']} {c['name'][:8]} "
                     f"{c['weight_pct']:.1f}% {bar}")
    lines.append("")

    # Warnings
    if d["warnings"]:
        lines.append(f"⚠️ **预警 ({len(d['warnings'])}项)**")
        for w in d["warnings"]:
            lines.append(f"  {w}")
        lines.append("")

    # Suggestions
    lines.append(f"💡 **建议**")
    for s in d["suggestions"]:
        lines.append(f"  {s}")
    lines.append("")

    # Stop alerts
    if d.get("stop_alerts"):
        lines.append(f"🚨 **止损/止盈触发**")
        for a in d["stop_alerts"]:
            lines.append(f"  {a['message']}")
        lines.append("")

    # Sector distribution mini-bar
    sectors = d.get("sector_allocation", {})
    if sectors:
        lines.append("📊 **行业分布**")
        for sec, pct in sorted(sectors.items(), key=lambda x: -x[1]):
            bar = "█" * max(1, int(pct / 3))
            marker = "⚠️" if pct > 40 else "  "
            lines.append(f"  {marker} {sec}: {pct:.1f}% {bar}")
        lines.append("")

    lines.append("═" * 56)
    lines.append("💡 运行 --scenario --drop 10 查看大盘下跌情景分析")
    lines.append("💡 运行 --stops 查看止损/止盈状态")
    lines.append("💡 运行 python3 -m quant_system.trading_journal --report 查看完整报告")

    return "\n".join(lines)


def format_scenario(s: dict) -> str:
    """Format scenario analysis."""
    lines = [
        "═" * 56,
        f"🌀 **情景分析: {s['scenario']}**",
        "═" * 56,
        f"  当前总市值: ¥{s['current_total']:,.0f}",
        f"  情景后市値: ¥{s['after_total']:,.0f}",
    ]
    loss = s["total_loss"]
    loss_pct = s["total_loss_pct"]
    if loss >= 0:
        lines.append(f"  影响: 🟢 +¥{loss:+,.0f} ({loss_pct:+.2f}%)")
    else:
        impact = "较大" if loss_pct < -5 else "可控"
        lines.append(f"  影响: 🔴 ¥{loss:+,.0f} ({loss_pct:+.2f}%) → {impact}")
    lines.append("")

    for p in s["positions"]:
        before = p["current_pnl_pct"]
        after = p["after_drop_pnl_pct"]
        arrow = "⬇️" if after < before else "⬆️"
        beta_txt = f" β={p.get('beta', 1.0):.2f}"
        lines.append(
            f"  {p['symbol']} {p['name'][:8]}{beta_txt} "
            f"¥{p['current_val']:,.0f} → ¥{p['after_drop_val']:,.0f} "
            f"({p['current_pnl_pct']:+.1f}% → {after:+.1f}%) {arrow}"
        )

    # P2-Q21-fix: 披露 beta 降级说明与对冲建议
    if s.get("beta_note"):
        lines.append("")
        lines.append(f"  ⚠️ {s['beta_note']}")
    if s.get("hedging_suggestions"):
        lines.append("")
        lines.append("  🛡️ **对冲建议**")
        for h in s["hedging_suggestions"]:
            lines.append(f"    • {h}")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="🛡️ 风险预算管理")
    parser.add_argument("--report", action="store_true", help="完整风险报告")
    parser.add_argument("--scenario", action="store_true", help="情景分析")
    parser.add_argument("--drop", type=float, default=10, help="大盘跌幅百分比")
    parser.add_argument("--stops", action="store_true", help="止损止盈检查")
    args = parser.parse_args()

    if args.stops:
        from quant_system.trade_db import check_stops
        alerts = check_stops()
        if not alerts:
            print("✅ 所有持仓在止损/止盈范围内，无触发")
        else:
            for a in alerts:
                print(a["message"])
        sys.exit(0)

    if args.scenario:
        s = run_scenario(args.drop)
        print(format_scenario(s))
        sys.exit(0)

    # Default: full dashboard
    d = compute_risk_dashboard()
    print(format_dashboard(d))
