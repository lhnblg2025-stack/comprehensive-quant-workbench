"""
system_manager.py — 牧云天枢 V4 全系统管理器

一键执行: 数据更新 → 机会扫描 → 组合优化 → 风控检查 → 执行订单 → 生成报告 → 推送

用法:
  python3 scripts/system_manager.py                         # 完整流程 (默认盘中)
  python3 scripts/system_manager.py --mode daily            # 收盘后全流程
  python3 scripts/system_manager.py --mode scan-only        # 仅扫描
  python3 scripts/system_manager.py --mode rebalance        # 仅再平衡
  python3 scripts/system_manager.py --dry-run               # 预览, 不执行
  python3 scripts/system_manager.py --status                # 系统状态报告
  python3 scripts/system_manager.py --backup                # 备份全部数据库
"""

from __future__ import annotations
import logging

import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CST = timezone(timedelta(hours=8))

# 系统路径
QUANT_DIR = ROOT / "quant_system"
DATA_DIR = Path.home() / ".quant_system"
BACKUP_DIR = DATA_DIR / "backups"
CHART_DIR = QUANT_DIR / "charts"

# 时间窗口 (盘中/收盘)
MARKET_OPEN = 9 * 60 + 25  # 09:25
MARKET_CLOSE = 15 * 60 + 5  # 15:05
MARKET_LUNCH_START = 11 * 60 + 30
MARKET_LUNCH_END = 13 * 60 + 0


# ════════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════════

def _ts() -> str:
    return datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S")


def _now_minutes() -> int:
    """当前时间的分钟数 (从00:00起)"""
    now = datetime.now(CST)
    return now.hour * 60 + now.minute


def _is_market_open() -> bool:
    """是否为交易时间"""
    m = _now_minutes()
    if m < MARKET_OPEN or m > MARKET_CLOSE:
        return False
    if MARKET_LUNCH_START <= m < MARKET_LUNCH_END:
        return False
    return True


def _is_trading_day() -> bool:
    """是否为交易日 (周一到周五)"""
    return datetime.now(CST).weekday() < 5


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ════════════════════════════════════════════════════════════════
# 检查器
# ════════════════════════════════════════════════════════════════

def check_system_health() -> dict:
    """
    全面系统健康检查。

    Returns: {
        status: "healthy" | "degraded" | "down",
        services: {...},
        databases: {...},
        data_staleness: {...},
        disk: {...},
    }
    """
    issues = []
    services = {}
    databases = {}
    data_staleness = {}

    # 数据库检查
    for db_name in ["trade_log.db", "financial.db", "cache.sqlite3", "tasks.sqlite3"]:
        db_path = DATA_DIR / db_name
        if db_path.exists():
            size_kb = db_path.stat().st_size / 1024
            databases[db_name] = {"ok": True, "size_kb": round(size_kb, 1)}
        else:
            databases[db_name] = {"ok": False, "size_kb": 0}
            issues.append(f"数据库缺失: {db_name}")

    # 磁盘空间
    try:
        st = DATA_DIR.stat()
        disk_usage = shutil.disk_usage(DATA_DIR)
        free_gb = disk_usage.free / (1024 ** 3)
        disk_info = {"free_gb": round(free_gb, 1), "ok": free_gb > 1.0}
        if free_gb < 1.0:
            issues.append(f"磁盘空间不足: {free_gb:.1f}GB")
    except Exception as e:
        disk_info = {"error": str(e)}

    # 数据新鲜度 (检查data_cache表)
    try:
        import sqlite3
        mp = DATA_DIR / "market_data.db"
        if mp.exists():
            db = sqlite3.connect(str(mp))
            rows = db.execute(
                "SELECT source, updated_at FROM data_cache ORDER BY updated_at DESC LIMIT 10"
            ).fetchall()
            for src, updated in rows:
                try:
                    dt = datetime.strptime(updated, "%Y-%m-%d %H:%M:%S")
                    age = datetime.now() - dt
                    stale = age > timedelta(hours=4)
                    data_staleness[src] = {
                        "updated": updated, "age_hours": round(age.total_seconds() / 3600, 1),
                        "stale": stale,
                    }
                    if stale:
                        issues.append(f"数据源过期: {src} ({updated})")
                except (ValueError, TypeError):
                    logging.getLogger(__name__).warning(
                        f"[system_manager] 数据源 {src} updated_at 无法解析({updated!r})，按时间戳异常处理")
                    data_staleness[src] = {"updated": updated, "parse_error": True}
                    issues.append(f"数据源时间戳异常: {src} ({updated})")
            db.close()
    except Exception as e:
        data_staleness = {"error": str(e)}

    # 进程检查
    try:
        import psutil
        for proc_name in ["python3", "quant"]:
            for p in psutil.process_iter(["name", "cmdline"]):
                try:
                    cmdline = " ".join(p.info.get("cmdline", []) or [])
                    if "quant_web" in cmdline or "server.py" in cmdline:
                        services["quant_web"] = {
                            "ok": True, "pid": p.pid,
                            "cpu": p.cpu_percent(interval=0),
                            "memory_mb": round(p.memory_info().rss / 1024 ** 2, 1),
                        }
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        if "quant_web" not in services:
            services["quant_web"] = {"ok": False}
            issues.append("quant_web 服务未运行")
    except ImportError:
        services["psutil"] = {"ok": False, "note": "psutil not available"}

    status = "healthy" if len(issues) == 0 else ("degraded" if len(issues) < 3 else "down")

    return {
        "timestamp": _ts(),
        "status": status,
        "issues": issues,
        "services": services,
        "databases": databases,
        "data_staleness": data_staleness,
        "disk": disk_info,
    }


# ════════════════════════════════════════════════════════════════
# 备份
# ════════════════════════════════════════════════════════════════

def backup_databases() -> dict:
    """备份所有 SQLite 数据库到 backup 目录。"""
    _ensure_dir(BACKUP_DIR)
    date_str = datetime.now(CST).strftime("%Y%m%d_%H%M%S")
    results = {}

    for db_name in ["trade_log.db", "financial.db", "cache.sqlite3", "tasks.sqlite3"]:
        src = DATA_DIR / db_name
        if not src.exists():
            results[db_name] = {"ok": False, "reason": "不存在"}
            continue
        dst = BACKUP_DIR / f"{date_str}_{db_name}"
        try:
            shutil.copy2(src, dst)
            size_kb = dst.stat().st_size / 1024
            results[db_name] = {"ok": True, "size_kb": round(size_kb, 1), "path": str(dst)}
        except Exception as e:
            results[db_name] = {"ok": False, "reason": str(e)}

    # 清理30天前的备份
    cleaned = 0
    cutoff = datetime.now() - timedelta(days=30)
    for f in BACKUP_DIR.glob("*.db"):
        try:
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            if mtime < cutoff:
                f.unlink()
                cleaned += 1
        except Exception as e:
            logging.getLogger(__name__).error(f"[system_manager] 操作失败: {e}", exc_info=True)

    return {
        "timestamp": _ts(),
        "backup_date": date_str,
        "results": results,
        "old_backups_cleaned": cleaned,
    }


# ════════════════════════════════════════════════════════════════
# 全流程执行
# ════════════════════════════════════════════════════════════════

def run_full_pipeline(dry_run: bool = False) -> dict:
    """
    执行完整量化流水线。

    Steps:
      1. 数据更新 (data_pipeline.update_data_if_stale)
      2. 市场状态检测 (market_regime.get_current_regime)
      3. 机会扫描 (opportunity.scan_opportunities)
      4. 因子计算 (factor_zoo.get_top_factors)
      5. 组合优化 (portfolio.compute_portfolio)
      6. 再平衡检查 (portfolio.compute_rebalance)
      7. 风控检查 (risk_monitor.run_risk_check)
      8. 订单检查 (execution.check_orders)
      9. 报告生成+推送
    """
    pipeline = {
        "timestamp": _ts(),
        "dry_run": dry_run,
        "steps": {},
        "overall": "pending",
    }

    # Step 1: 数据更新
    _print_step(1, "数据更新")
    if not dry_run:
        try:
            from quant_system.data_pipeline import update_data_if_stale
            data_result = update_data_if_stale()
            pipeline["steps"]["data_update"] = {
                "ok": True, "updated": data_result.get("updated_sources", []),
                "skipped": data_result.get("skipped_sources", []),
            }
        except Exception as e:
            pipeline["steps"]["data_update"] = {"ok": False, "error": str(e)}
    else:
        pipeline["steps"]["data_update"] = {"ok": True, "note": "dry-run, 跳过"}

    # Step 2: 市场状态
    _print_step(2, "市场状态检测")
    if not dry_run:
        try:
            from quant_system.market_regime import get_current_regime
            regime = get_current_regime()
            pipeline["steps"]["market_regime"] = {
                "ok": True, "regime": regime.get("trend", "unknown"),
                "risk_level": regime.get("composite_risk_level", 5),
            }
        except Exception as e:
            pipeline["steps"]["market_regime"] = {"ok": False, "error": str(e)}
    else:
        pipeline["steps"]["market_regime"] = {"ok": True, "note": "dry-run"}

    # Step 3: 机会扫描
    _print_step(3, "机会扫描")
    if not dry_run:
        try:
            from quant_system.opportunity import scan_opportunities
            opps = scan_opportunities(top_n=10, min_score=2.5)
            pipeline["steps"]["scan"] = {
                "ok": True,
                "opportunities": len(opps),
                "top": [{"symbol": o["symbol"], "score": o["signal_count"]} for o in opps[:5]],
            }
        except Exception as e:
            pipeline["steps"]["scan"] = {"ok": False, "error": str(e)}
    else:
        pipeline["steps"]["scan"] = {"ok": True, "note": "dry-run"}

    # Step 4: 风控检查
    _print_step(4, "风控检查")
    if not dry_run:
        try:
            from scripts.risk_monitor import run_risk_check
            risk = run_risk_check(auto_sell=False)
            pipeline["steps"]["risk"] = {
                "ok": True,
                "positions": risk.get("positions", 0),
                "alerts": len(risk.get("alerts", [])),
                "portfolio_alerts": len(risk.get("portfolio_alerts", [])),
                "total_pnl_pct": risk.get("total_pnl_pct", 0),
            }
        except Exception as e:
            pipeline["steps"]["risk"] = {"ok": False, "error": str(e)}
    else:
        pipeline["steps"]["risk"] = {"ok": True, "note": "dry-run"}

    # Step 5: 检查待成交订单
    _print_step(5, "订单检查")
    if not dry_run:
        try:
            from quant_system.execution import check_orders
            from quant_system.watchlist import fetch_quotes
            # 获取所有持仓的当前价格
            prices = {q["symbol"]: q["price"] for q in fetch_quotes() if q.get("price", 0) > 0}
            filled = check_orders(prices)
            pipeline["steps"]["orders"] = {
                "ok": True,
                "orders_filled": len(filled),
                "filled_details": filled[:3],
            }
        except Exception as e:
            pipeline["steps"]["orders"] = {"ok": False, "error": str(e)}
    else:
        pipeline["steps"]["orders"] = {"ok": True, "note": "dry-run"}

    pipeline["overall"] = "completed"
    return pipeline


def run_scan_only(dry_run: bool = False) -> dict:
    """仅执行机会扫描。"""
    _print_step(1, "仅扫描模式")
    try:
        from quant_system.opportunity import scan_opportunities
        from quant_system.opportunity import format_opportunities
        opps = scan_opportunities(top_n=15, min_score=2.5)
        text = format_opportunities(opps, brief=False)
        return {"ok": True, "opportunities": len(opps), "text": text}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def run_rebalance(dry_run: bool = False) -> dict:
    """仅执行再平衡检查。"""
    _print_step(1, "再平衡检查")
    try:
        from quant_system.trade_db import get_positions, refresh_prices
        from quant_system.portfolio import compute_rebalance
        refresh_prices()
        positions = get_positions()
        if not positions:
            return {"ok": True, "note": "无持仓, 无需再平衡", "trades": []}

        portfolio_value = sum(p.get("total_value", 0) or 0 for p in positions)
        # 等权作为目标 (可改为其他权重来源)
        n = len(positions)
        target = {p["symbol"]: 1.0 / n for p in positions}

        result = compute_rebalance(positions, target, portfolio_value, threshold=0.05)

        if dry_run:
            return {"ok": True, "note": "dry-run", "trades": result.trades,
                    "turnover": result.turnover}

        # 执行交易
        executed = []
        for t in result.trades:
            if t["action"] == "buy":
                from quant_system.trade_db import add_buy
                shares = int(t["trade_value"] / positions[0].get("current_price", 10))
                if shares > 0:
                    add_buy(t["symbol"], "", shares, positions[0].get("current_price", 10),
                            notes="再平衡买入")
                    executed.append(t)
            elif t["action"] == "sell":
                from quant_system.trade_db import add_sell
                pos = next((p for p in positions if p["symbol"] == t["symbol"]), None)
                if pos:
                    shares = min(pos["shares"], int(t["trade_value"] / pos.get("current_price", 10)))
                    if shares > 0:
                        add_sell(t["symbol"], shares, pos.get("current_price", 10),
                                 notes="再平衡卖出")
                        executed.append(t)

        return {"ok": True, "trades": result.trades, "executed": len(executed),
                "turnover": result.turnover}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ════════════════════════════════════════════════════════════════
# 报告生成
# ════════════════════════════════════════════════════════════════

def generate_system_report(pipeline_result: Optional[dict] = None) -> str:
    """生成飞书推送文本。"""
    lines = ["📊 **牧云天枢 V4 — 系统运行报告**", f"⏰ {_ts()}", ""]

    if pipeline_result:
        steps = pipeline_result.get("steps", {})
        for step_name, step_data in steps.items():
            icon = "✅" if step_data.get("ok") else "❌"
            lines.append(f"{icon} {step_name}")

        lines.append("")
        scan = steps.get("scan", {})
        if scan.get("ok"):
            lines.append(f"🔍 机会: {scan.get('opportunities', 0)} 个")
            for o in scan.get("top", []):
                lines.append(f"  {o['symbol']}: 评分 {o['score']}")

        risk = steps.get("risk", {})
        if risk.get("ok"):
            lines.append(f"🛡️ 持仓: {risk.get('positions', 0)} 只 | "
                         f"盈亏: {risk.get('total_pnl_pct', 0):+.2f}% | "
                         f"预警: {risk.get('alerts', 0)} 个")

        orders = steps.get("orders", {})
        if orders.get("ok"):
            lines.append(f"📋 订单: {orders.get('orders_filled', 0)} 笔成交")
    else:
        lines.append("无流水线数据")

    # 系统状态摘要
    health = check_system_health()
    if health["status"] != "healthy":
        lines.append("")
        lines.append(f"⚠️ 健康状态: {health['status']}")
        for issue in health.get("issues", [])[:3]:
            lines.append(f"  • {issue}")

    lines.append("")
    lines.append("---")
    lines.append("牧云天枢 V4 · 自动报告")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════

def _print_step(n: int, name: str):
    print(f"[{_ts()}] [{n}/5] {name}...", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="牧云天枢 V4 系统管理器")
    parser.add_argument("--mode", choices=["full", "daily", "scan-only", "rebalance",
                                           "status", "backup", "report"],
                        default="full", help="运行模式")
    parser.add_argument("--dry-run", action="store_true", help="预览模式")
    parser.add_argument("--push", action="store_true", help="推送结果到飞书")
    args = parser.parse_args()

    print(f"=== 牧云天枢 V4 系统管理器 ===")
    print(f"模式: {args.mode} | 时间: {_ts()}")
    if _is_trading_day():
        print(f"市场: {'🟢 交易中' if _is_market_open() else '🔴 已收盘'}")
    else:
        print(f"市场: ⚪ 非交易日")
    print()

    result = None

    if args.mode == "status":
        health = check_system_health()
        print(json.dumps(health, ensure_ascii=False, indent=2))
        result = health

    elif args.mode == "backup":
        bk = backup_databases()
        print(json.dumps(bk, ensure_ascii=False, indent=2))
        result = bk

    elif args.mode == "scan-only":
        result = run_scan_only(dry_run=args.dry_run)
        if result.get("ok"):
            print(result.get("text", ""))
        else:
            print(f"❌ {result.get('error', '')}")

    elif args.mode == "rebalance":
        result = run_rebalance(dry_run=args.dry_run)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.mode == "report":
        report = generate_system_report()
        print(report)
        result = {"text": report}

    else:  # full / daily
        result = run_full_pipeline(dry_run=args.dry_run)

    # 推送
    if args.push and result:
        if "text" not in result:
            report = generate_system_report(
                result if "steps" in result else None
            )
        else:
            report = result["text"]

        try:
            sys.path.insert(0, str(ROOT / "scripts"))
            import feishu_sender
            feishu_sender.send_markdown(report, title="📊 牧云天枢 V4 运行报告")
            print(f"\n飞书推送 ✅")
        except Exception as e:
            print(f"\n飞书推送 ❌ {e}")

    print(f"\n完成: {_ts()}")
