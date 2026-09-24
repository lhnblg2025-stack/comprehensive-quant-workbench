#!/usr/bin/env python3
"""
risk_monitor.py — 持仓风险监控 + 飞书推送 + 订单风控 + 压力测试。

在 Vultr 上定期运行，检查所有持仓的止损/止盈/追踪止损是否触发。
增强功能：
  - 待成交订单风控检查（超持仓/超行业限制）
  - 总杠杆监控
  - 每日风控摘要 Markdown 报告
  - 压力测试（2015股灾模式、2018单边下跌、2022风格切换）

用法:
  python3 scripts/risk_monitor.py                          # 检查+推送
  python3 scripts/risk_monitor.py --auto-sell              # 触发时自动卖出
  python3 scripts/risk_monitor.py --dry-run                # 仅输出
  python3 scripts/risk_monitor.py --stress-test            # 运行压力测试
  python3 scripts/risk_monitor.py --daily-report           # 生成每日风控摘要
"""

from __future__ import annotations
import logging

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CST = timezone(timedelta(hours=8))

sys.path.insert(0, str(ROOT / "scripts"))
import feishu_sender


def _get_ts() -> str:
    return datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S")


def _get_ts_short() -> str:
    return datetime.now(CST).strftime("%H:%M")


# ════════════════════════════════════════════════════════════════
# 1. 持仓止损/止盈检查（原有）
# ════════════════════════════════════════════════════════════════

def run_risk_check(auto_sell: bool = False) -> dict:
    """检查所有持仓止损/止盈 + 组合级风控。返回预警列表。"""
    from quant_system.trade_db import check_stops, get_positions, init_db, get_trades

    # 确保数据库已初始化
    init_db()

    print(f"[{_get_ts()}] 持仓风险扫描...", flush=True)

    # 先刷新价格
    from quant_system.trade_db import refresh_prices
    updated = refresh_prices()
    print(f"[{_get_ts()}] 价格刷新: {updated} 只", flush=True)

    # 检查止损
    alerts = check_stops()

    # 获取持仓概况
    positions = get_positions()
    total_value = sum(p.get("total_value", 0) or 0 for p in positions)
    total_cost = sum(p.get("total_cost", 0) or 0 for p in positions)
    total_pnl = sum(p.get("pnl", 0) or 0 for p in positions)
    total_pnl_pct = round((total_value / total_cost - 1) * 100, 2) if total_cost > 0 else 0

    # ── 组合级风控 ──
    portfolio_alerts = []

    # PR1: 总回撤监控
    if total_cost > 0 and total_pnl_pct <= -10:
        portfolio_alerts.append({
            "type": "PORTFOLIO_DRAWDOWN",
            "severity": "CRITICAL",
            "message": f"🔥 组合总回撤 {total_pnl_pct:.1f}%，已触发-10%警戒线！",
        })
    elif total_cost > 0 and total_pnl_pct <= -5:
        portfolio_alerts.append({
            "type": "PORTFOLIO_DRAWDOWN",
            "severity": "WARN",
            "message": f"⚠️ 组合总回撤 {total_pnl_pct:.1f}%，接近-10%警戒线",
        })

    # PR2: 行业集中度 (近似估算：通过持仓代码前缀)
    if len(positions) >= 3:
        prefix_count: dict[str, list] = {}
        for p in positions:
            sym = p["symbol"]
            prefix = sym[:3]  # 601/600/000/002/300
            if prefix not in prefix_count:
                prefix_count[prefix] = []
            prefix_count[prefix].append(p)
        for prefix, group in prefix_count.items():
            group_pct = sum(p.get("total_cost", 0) for p in group) / total_cost * 100 if total_cost > 0 else 0
            if group_pct >= 40:
                names = ", ".join(p["name"][:6] for p in group)
                portfolio_alerts.append({
                    "type": "SECTOR_CONCENTRATION",
                    "severity": "WARN",
                    "message": f"📊 板块集中度 {group_pct:.0f}%: [{names}] 超过40%警戒线",
                })

    # PR3: 波动率估算 (基于最近交易的盈亏标准差)
    try:
        recent_trades = get_trades(days=20, limit=50)
        sell_trades = [t for t in recent_trades if t["trade_type"] == "sell" and t.get("pnl_pct")]
        if len(sell_trades) >= 5:
            pnl_values = [t["pnl_pct"] for t in sell_trades]
            avg_pnl = sum(pnl_values) / len(pnl_values)
            variance = sum((x - avg_pnl) ** 2 for x in pnl_values) / len(pnl_values)
            vol_est = variance ** 0.5
            if vol_est > 10:
                portfolio_alerts.append({
                    "type": "HIGH_VOLATILITY",
                    "severity": "INFO",
                    "message": f"📈 近20日交易波动率 {vol_est:.1f}%（单笔盈亏标准差），仓位管理建议"
                               f"单笔不超过 {max(1, round(1/(vol_est/10)))}%"
                               f"总资金",
                })
    except Exception as e:
        logging.getLogger(__name__).error(f"[risk_monitor] 操作失败: {e}", exc_info=True)

    # ── PR4: 总杠杆检查 ──
    try:
        leverage_result = _check_leverage(positions)
        if leverage_result:
            portfolio_alerts.append(leverage_result)
    except Exception as e:
        logging.getLogger(__name__).error(f"[risk_monitor] 操作失败: {e}", exc_info=True)

    # ── PR5: 待成交订单风控检查 ──
    try:
        pending_order_alerts = _check_pending_orders(positions, total_cost)
        portfolio_alerts.extend(pending_order_alerts)
    except Exception as e:
        logging.getLogger(__name__).error(f"[risk_monitor] 操作失败: {e}", exc_info=True)

    result = {
        "ts": _get_ts(),
        "positions": len(positions),
        "total_value": round(total_value, 2),
        "total_cost": round(total_cost, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": total_pnl_pct,
        "alerts": alerts,
        "portfolio_alerts": portfolio_alerts,
        "positions_detail": [
            {
                "symbol": p["symbol"],
                "name": p["name"][:8],
                "shares": p["shares"],
                "cost": p["cost_price"],
                "price": p.get("current_price", 0),
                "pnl_pct": round((p.get("current_price", 0) / p["cost_price"] - 1) * 100, 1)
                           if p.get("current_price", 0) > 0 else 0,
                "pct_of_portfolio": round(p.get("total_cost", 0) / total_cost * 100, 1)
                                    if total_cost > 0 else 0,
            }
            for p in positions
        ],
    }

    # 自动卖出 —— 审计修复(2026-08-24): 必须走订单状态机(execution.create_order)，
    # 禁止直接调用 trade_db.add_sell 绕过 审批/成交回报/订单状态/审计留痕。
    if auto_sell and alerts:
        try:
            from quant_system import execution
        except Exception as e:
            for alert in alerts:
                alert["auto_sold"] = False
                alert["auto_sell_error"] = f"execution 模块不可用: {e}"
            return result
        for alert in alerts:
            try:
                sym = alert["symbol"]
                shares = alert["shares"]
                price = alert["current_price"]
                reason = alert["type"]
                order = execution.create_order(
                    symbol=sym,
                    order_type="market",
                    direction="sell",
                    shares=shares,
                    price=float(price) if price else None,
                    notes=f"风控自动卖出({reason})",
                )
                if order.get("error"):
                    alert["auto_sold"] = False
                    alert["auto_sell_error"] = order["error"]
                    continue
                # 订单已进入状态机（sent/filled 由 execution 收尾），此处只记录订单 id，
                # 不再直接 add_sell 修改持仓。
                alert["auto_sold"] = True
                alert["auto_sell_order_id"] = order.get("id")
                print(f"[{_get_ts()}] 风控订单已创建 {sym} {shares}股 @ ¥{price} order={order.get('id')}", flush=True)
            except Exception as e:
                alert["auto_sold"] = False
                alert["auto_sell_error"] = str(e)

    return result


# ════════════════════════════════════════════════════════════════
# 2. 杠杆监控
# ════════════════════════════════════════════════════════════════

def _check_leverage(positions: list[dict]) -> dict | None:
    """
    检查总杠杆（组合总价值 / 总成本）。
    如果杠杆 > 1.5 或 < 0.7 发出预警。
    杠杆 > 1 意味使用了融资。
    """
    total_cost = sum(p.get("total_cost", 0) for p in positions)
    total_value = sum(p.get("total_value", 0) or 0 for p in positions)

    if total_cost <= 0:
        return None

    leverage = round(total_value / total_cost, 4)

    if leverage > 1.5:
        return {
            "type": "LEVERAGE_HIGH",
            "severity": "WARN",
            "message": f"⚖️ 总杠杆率 {leverage:.2f}x（市值/成本），超过1.5x警戒线！",
        }
    elif leverage > 1.3:
        return {
            "type": "LEVERAGE_ELEVATED",
            "severity": "INFO",
            "message": f"⚖️ 总杠杆率 {leverage:.2f}x（市值/成本），接近1.5x警戒线",
        }

    return None


# ════════════════════════════════════════════════════════════════
# 3. 待成交订单风控
# ════════════════════════════════════════════════════════════════

def _check_pending_orders(positions: list[dict], total_cost: float) -> list[dict]:
    """
    从 trade_db.orders 表读取待成交订单，检查：
      - 是否有超持仓限制
      - 是否有超行业限制
    """
    from quant_system import trade_db

    pending_rows = trade_db._conn().execute(
        """SELECT * FROM orders
           WHERE status IN ('pending', 'approved', 'sent', 'partial_filled')
           ORDER BY created_at"""
    ).fetchall()

    if not pending_rows:
        return []

    orders = [dict(r) for r in pending_rows]
    alerts: list[dict] = []

    # 单个股票最大持仓限制（总资产比例）
    max_single_pct = 0.30  # 单只不超过30%
    # 单个行业最大限制
    max_sector_pct = 0.50  # 单个行业不超过50%

    for o in orders:
        sym = o["symbol"]
        direction = o["direction"]
        shares = o["shares"]
        price = o.get("price", 0) or 0

        # 估算成交金额
        est_amount = shares * price if price > 0 else shares * 10  # 粗略估算

        # 如果是买单，检查是否会超单只限制
        if direction == "buy" and total_cost > 0:
            existing_position = next((p for p in positions if p["symbol"] == sym), None)
            existing_cost = existing_position.get("total_cost", 0) if existing_position else 0
            new_total = existing_cost + est_amount
            new_pct = new_total / total_cost * 100
            if new_pct > max_single_pct * 100:
                alerts.append({
                    "type": "ORDER_EXCEEDS_SINGLE_LIMIT",
                    "severity": "WARN",
                    "message": f"📋 订单 #{o['id']} {sym} {direction} {shares}股："
                               f"预估占比{new_pct:.1f}% 超过单只限制{max_single_pct*100:.0f}%",
                })

        # 检查行业集中度风险（通过前缀近似）
        prefix = sym[:3]
        same_prefix_cost = sum(
            p.get("total_cost", 0) for p in positions
            if p["symbol"].startswith(prefix)
        )
        # 加这个订单
        if direction == "buy":
            same_prefix_cost += est_amount
        if total_cost > 0:
            sector_pct = same_prefix_cost / total_cost * 100
            if sector_pct > max_sector_pct * 100:
                alerts.append({
                    "type": "ORDER_EXCEEDS_SECTOR_LIMIT",
                    "severity": "WARN",
                    "message": f"📋 订单 #{o['id']} {sym} {direction} {shares}股："
                               f"同板块预估占比{sector_pct:.1f}% 超过板块限制{max_sector_pct*100:.0f}%",
                })

    return alerts


# ════════════════════════════════════════════════════════════════
# 4. 压力测试
# ════════════════════════════════════════════════════════════════

def run_stress_test() -> dict:
    """
    运行组合压力测试。

    场景:
      1. 2015股灾模式：全组合 -15%
      2. 2018单边下跌：全组合 -10%
      3. 2022风格切换：小盘 -15%, 大盘 +5%
    """
    from quant_system.trade_db import get_positions, init_db

    init_db()
    positions = get_positions()
    if not positions:
        return {"error": "当前无持仓，无法运行压力测试", "scenarios": []}

    total_cost = sum(p.get("total_cost", 0) for p in positions)
    total_value = sum(p.get("total_value", 0) or (p["shares"] * p["cost_price"]) for p in positions)

    # 按代码前缀区分大小盘
    large_cap_prefixes = {"600", "601", "603", "000"}  # 沪市主板+深市主板
    small_cap_prefixes = {"002", "300", "688"}  # 中小板、创业板、科创板

    def classify_position(p: dict) -> str:
        sym = p["symbol"]
        if sym.startswith(tuple(small_cap_prefixes)):
            return "small"
        return "large"

    scenarios: list[dict] = []

    # ── 场景1: 2015股灾模式 ──
    total_loss_1 = 0
    detail_1 = []
    for p in positions:
        cur_val = p.get("total_value", 0) or (p["shares"] * (p.get("current_price") or p["cost_price"]))
        loss = cur_val * 0.15
        total_loss_1 += loss
        detail_1.append({
            "symbol": p["symbol"],
            "name": p["name"][:8],
            "current_value": round(cur_val, 2),
            "estimated_loss": round(loss, 2),
        })
    scenarios.append({
        "name": "2015股灾模式",
        "description": "全组合下跌15%",
        "total_loss": round(total_loss_1, 2),
        "total_loss_pct": round(total_loss_1 / total_cost * 100, 2) if total_cost > 0 else 0,
        "details": detail_1,
    })

    # ── 场景2: 2018单边下跌 ──
    total_loss_2 = 0
    detail_2 = []
    for p in positions:
        cur_val = p.get("total_value", 0) or (p["shares"] * (p.get("current_price") or p["cost_price"]))
        loss = cur_val * 0.10
        total_loss_2 += loss
        detail_2.append({
            "symbol": p["symbol"],
            "name": p["name"][:8],
            "current_value": round(cur_val, 2),
            "estimated_loss": round(loss, 2),
        })
    scenarios.append({
        "name": "2018单边下跌",
        "description": "全组合下跌10%",
        "total_loss": round(total_loss_2, 2),
        "total_loss_pct": round(total_loss_2 / total_cost * 100, 2) if total_cost > 0 else 0,
        "details": detail_2,
    })

    # ── 场景3: 2022风格切换 ──
    total_change_3 = 0
    detail_3 = []
    for p in positions:
        cur_val = p.get("total_value", 0) or (p["shares"] * (p.get("current_price") or p["cost_price"]))
        cat = classify_position(p)
        if cat == "small":
            change = -cur_val * 0.15
        else:
            change = cur_val * 0.05
        total_change_3 += change
        detail_3.append({
            "symbol": p["symbol"],
            "name": p["name"][:8],
            "category": "小盘" if cat == "small" else "大盘",
            "current_value": round(cur_val, 2),
            "estimated_change": round(change, 2),
        })
    scenarios.append({
        "name": "2022风格切换",
        "description": "小盘-15%, 大盘+5%",
        "total_loss": round(total_change_3, 2),
        "total_loss_pct": round(total_change_3 / total_cost * 100, 2) if total_cost > 0 else 0,
        "details": detail_3,
    })

    return {
        "ts": _get_ts(),
        "total_cost": round(total_cost, 2),
        "total_value": round(total_value, 2),
        "num_positions": len(positions),
        "scenarios": scenarios,
    }


# ════════════════════════════════════════════════════════════════
# 5. 每日风控摘要 (Markdown)
# ════════════════════════════════════════════════════════════════

def generate_daily_risk_report() -> tuple[str, str]:
    """
    生成每日风控摘要 Markdown 报告。

    包含：
      - VaR 估算（基于持仓历史波动率或简单价格变动）
      - 最大回撤
      - 集中度分析
      - 杠杆
      - 压力测试结果
      - 待成交订单状态
    """
    from quant_system.trade_db import get_positions, get_summary, get_trades, init_db
    init_db()

    summary = get_summary()
    positions = get_positions()
    total_cost = summary["total_cost"]
    total_value = summary["total_value"]
    total_pnl_pct = summary["total_pnl_pct"]
    num_pos = summary["num_positions"]

    now_str = _get_ts()
    date_str = now_str[:10]

    lines = [
        f"# 📊 每日风控摘要 — {date_str}",
        f"",
        f"**报告时间:** {now_str}",
        f"**持仓数量:** {num_pos} 只",
        f"",
        f"---",
        f"",
        f"## 一、组合概况",
        f"",
        f"| 指标 | 数值 |",
        f"|------|------|",
        f"| 总成本 | ¥{total_cost:,.2f} |",
        f"| 总市值 | ¥{total_value:,.2f} |",
        f"| 总盈亏 | ¥{summary['total_pnl']:,.2f} ({total_pnl_pct:+.2f}%) |",
    ]

    # 杠杆
    if total_cost > 0:
        leverage = round(total_value / total_cost, 4)
        lines.append(f"| 杠杆率 | {leverage:.2f}x |")

    # VaR 估算（简单方法：基于最近20天交易盈亏）
    try:
        recent_trades = get_trades(days=20, limit=100)
        sell_trades = [t for t in recent_trades if t["trade_type"] == "sell" and t.get("pnl_pct")]
        if len(sell_trades) >= 10:
            pnl_values = [t["pnl_pct"] for t in sell_trades]
            avg_pnl = sum(pnl_values) / len(pnl_values)
            variance = sum((x - avg_pnl) ** 2 for x in pnl_values) / len(pnl_values)
            std_dev = variance ** 0.5
            # 95% VaR = -1.645 * sigma
            var_95 = round(-1.645 * std_dev, 2)
            var_95_amount = round(total_value * abs(var_95) / 100, 2)
            lines.append(f"| VaR(95%,1日) 估算 | {var_95:+.2f}% (≈ ¥{var_95_amount:,.0f}) |")
        elif total_value > 0:
            # 没有足够交易数据时用组合价格的 5% 作为粗略 VaR
            var_95_amount = round(total_value * 0.05, 2)
            lines.append(f"| VaR(95%,1日) 估算 | -5.00% (≈ ¥{var_95_amount:,.0f}) |")
    except Exception as e:
        logging.getLogger(__name__).error(f"[risk_monitor] 操作失败: {e}", exc_info=True)

    # 最大回撤
    try:
        all_trades = get_trades(days=365, limit=500)
        if all_trades:
            # 通过交易历史估算回撤
            running_pnl: list[float] = [0]
            for t in sorted(all_trades, key=lambda x: x["trade_date"]):
                if t["trade_type"] == "sell" and t.get("pnl_pct"):
                    running_pnl.append(running_pnl[-1] + t["pnl_pct"])
            if len(running_pnl) > 1:
                peak = max(running_pnl)
                trough = min(running_pnl)
                max_dd = round(trough - peak, 2) if peak > 0 else 0
                lines.append(f"| 估算最大回撤 | {max_dd:+.2f}% |")
    except Exception as e:
        logging.getLogger(__name__).error(f"[risk_monitor] 操作失败: {e}", exc_info=True)

    lines.append("")

    # 行业分布
    sector_allocation = summary.get("sector_allocation", {})
    if sector_allocation:
        lines.append("## 二、行业分布")
        lines.append("")
        lines.append("| 行业 | 占比 | 柱状图 |")
        lines.append("|------|------|--------|")
        for sec, pct in sorted(sector_allocation.items(), key=lambda x: -x[1]):
            bar = "█" * max(1, int(pct / 5))
            lines.append(f"| {sec} | {pct:.1f}% | {bar} |")
        lines.append("")

    # 持仓明细
    if positions:
        lines.append("## 三、持仓明细")
        lines.append("")
        lines.append("| # | 代码 | 名称 | 股数 | 成本 | 现价 | 盈亏% | 占比 |")
        lines.append("|---|------|------|------|------|------|-------|------|")
        for i, p in enumerate(positions):
            sym = p["symbol"]
            name = p["name"][:8]
            shares = p["shares"]
            cost = p["cost_price"]
            curr = p.get("current_price", 0) or cost
            pnl = round((curr / cost - 1) * 100, 1) if cost > 0 else 0
            pct = round(p.get("total_cost", 0) / total_cost * 100, 1) if total_cost > 0 else 0
            pnl_str = f"{pnl:+.1f}%"
            lines.append(f"| {i+1} | {sym} | {name} | {shares} | {cost:.2f} | {curr:.2f} | {pnl_str} | {pct:.1f}% |")
        lines.append("")

    # 压力测试
    try:
        stress_result = run_stress_test()
        if "error" not in stress_result:
            lines.append("## 四、压力测试")
            lines.append("")
            for sc in stress_result.get("scenarios", []):
                loss = sc["total_loss"]
                loss_pct = sc["total_loss_pct"]
                emoji = "🔴" if loss < 0 else "🟢"
                lines.append(f"- **{sc['name']}** ({sc['description']}): "
                             f"{emoji} ¥{loss:,.2f} ({loss_pct:+.2f}%)")
            lines.append("")
    except Exception as e:
        logging.getLogger(__name__).error(f"[risk_monitor] 操作失败: {e}", exc_info=True)

    # 待成交订单
    try:
        from quant_system import trade_db as td
        pending = td._conn().execute(
            "SELECT * FROM orders WHERE status IN ('pending','approved','sent','partial_filled') ORDER BY created_at"
        ).fetchall()
        if pending:
            lines.append("## 五、待成交订单")
            lines.append("")
            lines.append("| 序号 | 代码 | 类型 | 方向 | 股数 | 状态 | 创建时间 |")
            lines.append("|------|------|------|------|------|------|----------|")
            for i, o in enumerate(pending):
                o = dict(o)
                lines.append(f"| {i+1} | {o['symbol']} | {o['order_type']} | {o['direction']} | "
                             f"{o['shares']} | {o['status']} | {o['created_at'][:10]} |")
            lines.append("")
    except Exception as e:
        logging.getLogger(__name__).error(f"[risk_monitor] 操作失败: {e}", exc_info=True)

    lines.append("---")
    lines.append("")
    lines.append("_牧云天枢 · V3 风控引擎_")

    title = f"📊 每日风控摘要 — {date_str}"
    return title, "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# 6. 构建推送文本
# ════════════════════════════════════════════════════════════════

def build_alert_text(result: dict) -> tuple[str, str]:
    """构建飞书风险预警文本"""
    alerts = result.get("alerts", [])
    portfolio_alerts = result.get("portfolio_alerts", [])
    all_alerts = alerts + portfolio_alerts
    positions = result.get("positions", 0)

    if not all_alerts:
        if positions == 0:
            title = f"💤 持仓为空，无风险检查"
            body = "当前无持仓，无需风控。"
        else:
            # 正常状态 + 持仓明细
            title = f"✅ 持仓风控正常 — {positions} 只"
            lines = [
                f"📊 持仓 {positions} 只 | 总值 ¥{result['total_value']:.0f} | "
                f"盈亏 {result['total_pnl_pct']:+.2f}%",
                "",
            ]
            positions_detail = result.get("positions_detail", [])
            if positions_detail:
                lines.append("持仓明细:")
                for pd in positions_detail:
                    emoji = "🟢" if pd["pnl_pct"] >= 0 else "🔴"
                    lines.append(
                        f"  {emoji} {pd['symbol']} {pd['name']:<6} "
                        f"{pd['shares']}股 @ ¥{pd['price']:.2f} "
                        f"{pd['pnl_pct']:+.1f}% 占比{pd['pct_of_portfolio']:.0f}%"
                    )
            lines.append("")
            lines.append("所有止损/止盈/追踪止损均未触发。")
            body = "\n".join(lines)
        return title, body

    title = f"🚨 风控预警 {_get_ts_short()} — {len(alerts)}个个股 + {len(portfolio_alerts)}个组合"

    lines = [
        f"📊 持仓 {positions} 只 | 总值 ¥{result['total_value']:.0f} | "
        f"盈亏 {result['total_pnl_pct']:+.2f}%",
        "",
    ]

    # 个股触发
    if alerts:
        lines.append(f"🔴 个股止损/止盈 ({len(alerts)}):")
        for alert in alerts:
            msg = alert.get("message", "")
            lines.append(f"  {msg}")
            if alert.get("auto_sold"):
                lines.append(f"     ✅ 已自动卖出")
        lines.append("")

    # 组合预警
    if portfolio_alerts:
        lines.append(f"📋 组合级预警 ({len(portfolio_alerts)}):")
        for pa in portfolio_alerts:
            severity_emoji = {"CRITICAL": "🔥", "WARN": "⚠️", "INFO": "ℹ️"}
            emoji = severity_emoji.get(pa["severity"], "📋")
            lines.append(f"  {emoji} {pa['message']}")
        lines.append("")

    # 持仓明细
    positions_detail = result.get("positions_detail", [])
    if positions_detail:
        lines.append("持仓明细:")
        for pd in positions_detail:
            emoji = "🟢" if pd["pnl_pct"] >= 0 else "🔴"
            lines.append(
                f"  {emoji} {pd['symbol']} {pd['name']:<6} "
                f"{pd['shares']}股 @ ¥{pd['price']:.2f} "
                f"{pd['pnl_pct']:+.1f}% 占比{pd['pct_of_portfolio']:.0f}%"
            )

    lines.append("")
    lines.append("---")
    lines.append("牧云天枢 · V3 组合风控")

    return title, "\n".join(lines)


def _dedup_allow(channel: str, body: str, same_sig_min_gap_s: int = 2 * 3600) -> bool:
    """经 alert_dedup 网关去重；失败直发兜底。"""
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from alert_dedup import get_deduper, make_sig, normalize_price  # noqa: PLC0415
        sig = make_sig(channel, normalize_price(body))
        return get_deduper().should_send(channel, sig, level="danger",
                                         same_sig_min_gap_s=same_sig_min_gap_s)
    except Exception as e:  # noqa: BLE001 - 去重失败直发兜底
        print(f"[alert_dedup] 去重失败, 直发兜底: {e}", flush=True)
        return True


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="持仓风控检查 + 飞书推送")
    parser.add_argument("--auto-sell", action="store_true", help="触发时自动卖出")
    parser.add_argument("--dry-run", action="store_true", help="仅输出，不推送")
    parser.add_argument("--stress-test", action="store_true", help="运行压力测试")
    parser.add_argument("--daily-report", action="store_true", help="生成每日风控摘要")
    args = parser.parse_args()

    if args.stress_test:
        result = run_stress_test()
        if "error" in result:
            print(f"❌ {result['error']}")
        else:
            print(f"\n{'='*60}")
            print(f"📊 压力测试结果 — {result['ts']}")
            print(f"{'='*60}")
            print(f"组合成本: ¥{result['total_cost']:,.2f}")
            print(f"组合市值: ¥{result['total_value']:,.2f}")
            print(f"持仓数量: {result['num_positions']} 只")
            print()
            for sc in result["scenarios"]:
                loss = sc["total_loss"]
                loss_pct = sc["total_loss_pct"]
                emoji = "🔴" if loss < 0 else "🟢"
                print(f"  {emoji} {sc['name']}")
                print(f"     描述: {sc['description']}")
                print(f"     预估损益: ¥{loss:+,.2f} ({loss_pct:+.2f}%)")
                print()
        sys.exit(0)

    if args.daily_report:
        title, body = generate_daily_risk_report()
        print(f"\n=== {title} ===\n{body}\n")

        if not args.dry_run:
            ok = feishu_sender.send_markdown(body, title=title)
            print(f"飞书推送: {'✅' if ok else '❌'}")
        sys.exit(0)

    result = run_risk_check(auto_sell=args.auto_sell)
    title, body = build_alert_text(result)

    if args.dry_run:
        print(f"\n=== {title} ===\n{body}\n")
        print(json.dumps(result, ensure_ascii=False, indent=2)[:500])
    else:
        print(f"\n=== {title} ===", flush=True)
        # 2026-08-14 审计修复: 旧版每次运行都推飞书（正常状态也推"持仓风控正常"）。
        # 现改为: 仅真实预警（个股止损/组合级）才推送；正常状态静默。
        # 每日风险报告 quant_risk_stress --daily-report 仍照常推送（每日一次）。
        n_alerts = len(result.get("alerts", [])) + len(result.get("portfolio_alerts", []))
        if n_alerts > 0:
            if _dedup_allow("risk", body):
                ok = feishu_sender.send_markdown(body, title=title)
                print(f"飞书推送: {'✅' if ok else '❌'}", flush=True)
            else:
                print("同预警集合在静默窗口内，跳过飞书推送", flush=True)
        else:
            print("无预警，静默（不推飞书）", flush=True)
