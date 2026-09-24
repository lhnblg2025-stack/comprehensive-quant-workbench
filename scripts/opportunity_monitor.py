#!/usr/bin/env python3
"""
opportunity_monitor.py — 主板大市值机会扫描与本地产物维护。

默认只更新扫描缓存，不再发送“买入信号/全量扫描”飞书播报。
如需人工调试消息内容，使用 --dry-run；只有显式 --notify 才允许发送。

用法:
  python3 scripts/opportunity_monitor.py                         # 全量扫描+推送
  python3 scripts/opportunity_monitor.py --min-score 4            # 最低4分才推送
  python3 scripts/opportunity_monitor.py --top 10                 # 只推前10
  python3 scripts/opportunity_monitor.py --dry-run                # 只输出，不推送
  python3 scripts/opportunity_monitor.py --interval               # 高频模式（增量缓存）
"""

from __future__ import annotations

import json
import os
import re
import sys
import time as _time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_system.opportunity import (
    scan_opportunities,
)
from quant_system.watchlist import get_watchlist

CST = timezone(timedelta(hours=8))
CACHE_DIR = ROOT / "generated" / "opportunity_monitor"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 飞书发送模块
sys.path.insert(0, str(ROOT / "scripts"))
import feishu_sender


# Windows gbk stdout 无法输出 emoji（UnicodeEncodeError），统一替换为 ASCII。
# build_alert_text 返回值同时用于本地 print 与飞书推送，替换后不影响可读性。
_EMOJI_MAP = {
    "\u2705": "[OK]",          # check mark
    "\u274c": "[失败]",        # cross mark
    "\U0001f534": "[信号]",    # red circle
    "\U0001f4ca": "[扫描]",    # bar chart
    "\u26a0\ufe0f": "[!]",     # warning sign
    "\U0001f53a": "[上升]",    # up triangle
    "\U0001f53b": "[下降]",    # down triangle
    "\u26a1": "[FIB]",         # high voltage
    "\U0001f30a": "[波幅]",    # water wave
    "\U0001f6e1": "[防御]",    # shield
    "\u00a5": "\uffe5",        # half-width yen -> full-width yen (gbk-safe)
}
_EMOJI_RE = re.compile(
    "["
    "\U0001f000-\U0001faff"   # misc symbols and pictographs
    "\u2600-\u27bf"           # misc symbols (warnings, stars, checks)
    "\u2190-\u21ff"           # arrows
    "\ufe0f\ufe0e"            # variation selectors
    "]"
)


def _sanitize_emoji(text: str) -> str:
    """替换常见 emoji 为 ASCII 标签，其余 emoji 字符剔除，防 gbk print 崩溃。"""
    if not isinstance(text, str):
        return text
    for emoji, ascii_ in _EMOJI_MAP.items():
        text = text.replace(emoji, ascii_)
    return _EMOJI_RE.sub("", text)


def run_monitor(min_score: float = 3.0, top_n: int = 30, intraday: bool = False) -> dict:
    """
    全量扫描主板≥300亿股票，返回机会列表并推送。

    V6.1-fix: intraday=True 时用**分钟线分时信号**盘中扫描（用户硬性要求:
    定时监控必须按分时检测, 不是日K; 收盘后才发日K）。
    """
    print(f"[{_ts()}] 开始主板≥300亿{'盘中分时' if intraday else '日K'}扫描...", flush=True)
    t0 = _time.time()

    if intraday:
        # ── 盘中模式: 分时信号 (新浪分钟源, Vultr 可用) ──
        # 单栈化(V3)：改走 quant_platform.legacy 薄转发，不再直接 import quant_v6
        try:
            sys.path.insert(0, str(ROOT))
            from quant_platform.legacy import scan_intraday, compute_intraday_signals, is_trading_time  # noqa: F401
            from quant_system.opportunity import fetch_quotes, _cap_tier, _limit_state
            quotes = fetch_quotes(force=True)
            valid = [q for q in quotes if q.get("price", 0) > 0]
            # 复用日K快速过滤: 剔除低价/负PB/高PE/涨跌停/追高
            # V6.1-fix: 不调 _get_stock_sector——东财接口在 Vultr 被风控,
            # 240 只行业查询会全部等超时卡死; sector 仅展示字段, 盘中可省。
            cands = []
            qmap = {}
            for q in valid:
                px, pb, pe = q.get("price", 0), q.get("pb", 0), q.get("pe_ttm", 0)
                h52 = q.get("high_52w", 0)
                if px < 3 or pb <= 0 or pe > 80:
                    continue
                if h52 > 0 and px / h52 > 0.95:
                    continue
                if _limit_state(q) in ("limit_up", "limit_down"):
                    continue
                q["tier"] = _cap_tier(q.get("market_cap_yi", 0))
                q["sector"] = ""
                cands.append(q["symbol"])
                qmap[q["symbol"]] = q
            scan = scan_intraday(cands, min_score=max(min_score, 5.0), scale=5, quotes=qmap)
            opps = scan["results"]
            scan_summary = scan.get("summary") or {}
            elapsed = _time.time() - t0
            print(f"[{_ts()}] 盘中分时扫描: {scan['summary']['total']}候选 "
                  f"K线成功{scan['summary']['kline_ok']} 命中{len(opps)} 耗时{elapsed:.0f}s", flush=True)
        except Exception as exc:
            print(f"[{_ts()}] [!] 盘中扫描失败, 回退日K: {str(exc)[:120]}", flush=True)
            opps = scan_opportunities(top_n=top_n, min_score=min_score)
            elapsed = _time.time() - t0
    else:
        # ── 盘后模式: 日K全量扫描 ──
        opps = scan_opportunities(top_n=top_n, min_score=min_score)
        elapsed = _time.time() - t0

    print(f"[{_ts()}] 扫描完成: {len(opps)} 只机会, 耗时 {elapsed:.0f}s", flush=True)

    # 按分数排序
    opps.sort(key=lambda r: r.get("signal_count", 0) or r.get("score", 0), reverse=True)

    total_stocks = len(get_watchlist())
    result = {
        "ts": _ts(),
        "scan_summary": locals().get("scan_summary", {}),
        "total_scanned": total_stocks,
        "opportunities_found": len(opps),
        "elapsed_seconds": round(elapsed),
        "opportunities": [
            {
                "symbol": r.get("symbol"),
                "name": r.get("name"),
                "price": r.get("price"),
                "change_pct": r.get("change_pct"),
                "score": r.get("signal_count", 0),
                "market_cap_yi": r.get("market_cap_yi"),
                "tier": r.get("tier"),
                "signals": r.get("signals", {}),
                "signal_reasons": r.get("signal_reasons", {}),
                "signal_summary": r.get("signal_summary", ""),
                "trend": r.get("trend", "side"),
                "fib_382": r.get("fib_382"),
                "fib_618": r.get("fib_618"),
                "last_swing_low": r.get("last_swing_low"),
                "last_swing_high": r.get("last_swing_high"),
                "rsi": r.get("rsi"),
                "cci": r.get("cci"),
                "ma20": r.get("ma20"),
                "ma60": r.get("ma60"),
                "ma144": r.get("ma144"),
                "boll_lower": r.get("boll_lower"),
                "vol_ratio": r.get("vol_ratio"),
                "pe_ttm": r.get("pe_ttm"),
                "pb": r.get("pb"),
                "full_stack_score": r.get("full_stack_score"),
                "full_stack_reasons": r.get("full_stack_reasons", []),
                "research_evidence": r.get("research_evidence", {}),
                "regulatory_risk": r.get("regulatory_risk", {}),
                "mainline_match": r.get("mainline_match"),
                "strategy": r.get("strategy", ""),
                "trade_allowed": r.get("trade_allowed", False),
                "decision": r.get("decision", "观察候选/等待全栈确认"),
            }
            for r in opps
        ],
    }

    # 统一决策链门控：全市场环境/主线/资金/风险作为技术候选的上层上下文。
    # 技术扫描仍负责发现异动，但不会单独决定是否进入可执行清单。
    try:
        from unified_decision_snapshot import build_snapshot
        decision = build_snapshot(mode="intraday" if intraday else "after_close")
        gate = decision.get("market") or {}
        mainlines = decision.get("mainlines") or []
        risk_flags = list(gate.get("risk_flags") or [])
        summary = result.get("scan_summary") or {}
        attempted = int(summary.get("total") or 0)
        succeeded = int(summary.get("kline_ok") or 0)
        if attempted and succeeded / attempted < 0.6:
            risk_flags.append(f"分钟数据覆盖不足({succeeded}/{attempted})，候选仅作观察，不推执行信号")
        result["decision_snapshot"] = {
            "as_of": decision.get("as_of"),
            "scope": decision.get("scope"),
            "market": gate,
            "mainlines": mainlines[:8],
            "risk_flags": risk_flags,
            "source_chain": decision.get("source_chain"),
        }
        # 全栈门控：技术扫描只负责发现异动；必须通过市场、主线、研究/公告和策略门槛，才可进入执行池。
        evidence = decision.get("evidence") or {}
        research_codes = ((evidence.get("research") or {}).get("codes") or {})
        regulatory_codes = ((evidence.get("regulatory") or {}).get("by_code") or {})
        force = float(gate.get("force_index") or 0)
        breadth = float(gate.get("breadth") or 0)
        hard_market_block = force < 35 or breadth < 0.50
        gated = []
        for row in result["opportunities"]:
            code = str(row.get("symbol") or "").zfill(6)
            name = str(row.get("name") or "")
            is_main = code.startswith(("600", "601", "603", "605", "000", "001", "002", "003"))
            base_allowed = is_main and "st" not in name.lower() and float(row.get("price") or 0) > 0
            research_ev = dict(research_codes.get(code) or {})
            regulatory_ev = dict(regulatory_codes.get(code) or {})
            text = " ".join(str(row.get(k) or "") for k in ("name", "signal_summary", "trend"))
            mainline_match = next((str(m.get("industry") or m.get("name") or "") for m in mainlines
                                   if str(m.get("industry") or m.get("name") or "") and str(m.get("industry") or m.get("name")) in text), None)
            technical = min(40.0, float(row.get("score") or 0) * 6.0)
            stack_score = technical
            stack_reasons = [f"技术信号{row.get('score', 0)}项"]
            if force >= 35:
                stack_score += 15; stack_reasons.append(f"资金合力{force:.1f}通过")
            else:
                stack_reasons.append(f"资金合力{force:.1f}<35")
            if breadth >= 0.50:
                stack_score += 10; stack_reasons.append(f"市场宽度{breadth:.0%}通过")
            else:
                stack_reasons.append(f"市场宽度{breadth:.0%}<50%")
            if mainline_match:
                stack_score += 20; stack_reasons.append(f"主线匹配:{mainline_match}")
            else:
                stack_reasons.append("未匹配当前主线")
            reports = int(research_ev.get("reports") or 0)
            if reports:
                stack_score += min(10.0, reports * 2); stack_reasons.append(f"研报证据{reports}篇")
            else:
                stack_reasons.append("无直接研报覆盖")
            tier1 = int(regulatory_ev.get("tier1") or 0)
            if tier1:
                stack_score -= 60; stack_reasons.append(f"监管硬风险{tier1}条")
            row["full_stack_score"] = round(max(0.0, min(100.0, stack_score)), 1)
            row["research_evidence"] = {"reports": reports, "top_concepts": research_ev.get("top_concepts", [])}
            row["regulatory_risk"] = {"count": regulatory_ev.get("count", 0), "tier1": tier1, "latest": regulatory_ev.get("latest")}
            row["mainline_match"] = mainline_match
            row["full_stack_reasons"] = stack_reasons
            row["strategy"] = "只在次日竞价不弱、主线成交额放大且核心股有承接时试探；否则观察"
            row["trade_allowed"] = bool(base_allowed and not hard_market_block and not tier1 and mainline_match and row["full_stack_score"] >= 65)
            row["decision"] = "全栈可执行候选" if row["trade_allowed"] else "观察候选/等待全栈确认"
            if row["trade_allowed"]:
                gated.append(row)
        result["execution_opportunities"] = gated
        result["execution_found"] = len(gated)
        result["technical_opportunities_found"] = len(result["opportunities"])
        result["observation_opportunities"] = [r for r in result["opportunities"] if not r.get("trade_allowed")]
        result["execution_gate"] = {"market_block": hard_market_block, "force_index": force, "breadth": breadth,
                                     "requires_mainline": True, "requires_research_or_price": True,
                                     "note": "技术机会不等于买入；全栈门控未通过时禁止发送买入信号"}
    except Exception as exc:
        result["decision_snapshot_error"] = str(exc)[:160]
        result["execution_opportunities"] = []
        result["execution_found"] = 0

    # 保存缓存
    cache_path = CACHE_DIR / "latest_scan.json"
    cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    return result


def _describe_signals(r: dict) -> str:
    """
    将加权信号转为详细的中文理由，含波浪/斐波那契/数值。
    使用 compute_indicators 返回的 signal_reasons。
    """
    signal_reasons = r.get("signal_reasons", {})
    signals = r.get("signals", {})

    parts = []

    # 趋势背景
    trend = r.get("trend", "side")
    if trend == "up":
        parts.append("[上升]趋势回调")
    elif trend == "down":
        parts.append("[下降]趋势")

    # 用 signal_reasons 里已有的中文描述
    for key in ["ma144_support", "ma60_support", "fib_support", "swing_low_support",
                "boll_lower_band", "boll_near_lower",
                "rsi_oversold", "rsi_near_oversold", "cci_oversold",
                "macd_bullish_cross", "macd_bullish_diverging", "bullish_divergence",
                "rsi_turning_up", "volume_price_up", "volume_price_down",
                # V6.1: 盘中分时信号 key
                "m5_support", "m10_support", "m20_support", "m_trend_up",
                "m_rsi_oversold", "m_rsi_near_oversold", "m_rsi_turn_up",
                "m_cci_oversold", "m_cci_near_oversold", "m_cci_turn_up",
                "m_macd_positive", "m_macd_cross", "m_macd_shrink",
                "m_vol_surge", "m_vol_active", "m_day_low", "m_day_high", "m_late_rise"]:
        if key in signal_reasons and signals.get(key, 0) > 0:
            parts.append(signal_reasons[key])

    # 斐波那契支撑信息
    fib_str = ""
    fib_382 = r.get("fib_382")
    fib_618 = r.get("fib_618")
    if fib_382 and fib_618:
        price = r.get("price", 0)
        # 检查价格离哪个 fib 最近
        dist_382 = abs(price - fib_382)
        dist_618 = abs(price - fib_618)
        nearest = "0.382" if dist_382 < dist_618 else "0.618"
        nearest_val = fib_382 if dist_382 < dist_618 else fib_618
        fib_str = f"[FIB]斐波那契{nearest}=¥{nearest_val:.2f}"
        if nearest not in str(parts):
            parts.append(fib_str)

    # 波浪结构
    swing_low = r.get("last_swing_low")
    swing_high = r.get("last_swing_high")
    if swing_low and swing_high:
        parts.append(f"[波幅]¥{swing_low:.2f}~¥{swing_high:.2f}")

    # 安全边际
    pb_val = r.get("pb")
    if pb_val and pb_val < 1.5:
        parts.append(f"[防御]低PB={pb_val}")
    pe_val = r.get("pe_ttm")
    if pe_val and 0 < pe_val < 15:
        parts.append(f"[扫描]低PE={pe_val}")

    return " | ".join(parts) if parts else "暂无明确信号"


def build_alert_text(result: dict) -> tuple[str, str]:
    """构建飞书可读的预警文本，带波浪/斐波那契/多维度买入理由"""
    opps = result.get("execution_opportunities", result.get("opportunities", []))
    decision = result.get("decision_snapshot") or {}
    if not opps:
        technical_n = result.get("technical_opportunities_found", result.get("opportunities_found", 0))
        title = f"[OK] 全栈决策复核完成 — 无可执行信号"
        observations = result.get("observation_opportunities") or result.get("opportunities") or []
        detail = []
        for row in observations[:5]:
            detail.append(f"{row.get('name')}({row.get('symbol')}) 全栈分{row.get('full_stack_score', '-')}: {'；'.join(row.get('full_stack_reasons') or [])}")
        body = (
            f"全量扫描 {result['total_scanned']} 只，技术发现 {technical_n} 只，统一链复核后无可执行候选，耗时 {result['elapsed_seconds']}s。\n"
            "技术异动不等于买入；请以盘后决策链中的市场、主线、公告、研究证据和策略触发条件为准。\n"
            + ("前5个观察候选：\n" + "\n".join(detail) if detail else "当前无足够数据生成观察候选。")
        )
        return _sanitize_emoji(title), _sanitize_emoji(body)

    top = opps[:10]
    title = f"[信号] 全栈执行候选 {_ts_short()} — {len(opps)} 只（需按策略触发）"

    market = decision.get("market") or {}
    scope = decision.get("scope") or {}
    mains = decision.get("mainlines") or []
    risk_flags = decision.get("risk_flags") or []
    lines = [
        f"[扫描] 全市场分析 | {result['total_scanned']} 只 | {result['elapsed_seconds']}s",
        f"[执行] {scope.get('execution', '沪深主板，剔除ST')} | 可执行候选 {len(opps)} 只",
        f"[环境] {market.get('emotion_stage', '未知')} | 温度 {market.get('temperature', '-')} | 资金合力 {market.get('force_index', '-')}",
        f"[主线] " + ("、".join(str(x.get('industry') or x.get('name') or '-') for x in mains[:5]) or "未识别"),
        f"[风险] " + ("；".join(str(x) for x in risk_flags) or "风险门控通过"),
        "",
        f"前{len(top)}个主板候选：",
        "",
    ]

    for i, r in enumerate(top, 1):
        name = r.get("name", "")
        symbol = r.get("symbol", "")
        price = r.get("price", "-")
        chg = r.get("change_pct", "-")
        score = r.get("score", 0)
        cap = r.get("market_cap_yi", 0)
        tier = r.get("tier", "")

        lines.append(f"{i}. {name}({symbol})")
        lines.append(f"   现价 ¥{price}  {chg}%  |  {cap:.0f}亿  {tier}")
        full_score = r.get("full_stack_score", score)
        stack_reasons = "；".join(r.get("full_stack_reasons") or []) or "统一链未返回逐股理由"
        strategy = r.get("strategy") or "仅在盘后复核通过、次日盘口承接确认后执行"
        lines.append(f"   技术评分 {score} | 全栈决策分 {full_score} | {_describe_signals(r)}")
        lines.append(f"   原因：{stack_reasons}")
        lines.append(f"   策略：{strategy}")
        lines.append("")

    if len(opps) > 10:
        lines.append(f"... 还有 {len(opps)-10} 只（http://192.248.144.249:8600）")

    lines.append("")
    lines.append("---")
    lines.append("牧云天枢 · 自动监控")

    body = "\n".join(lines)
    return _sanitize_emoji(title), _sanitize_emoji(body)


def _ts() -> str:
    return datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S")


def _ts_short() -> str:
    return datetime.now(CST).strftime("%H:%M")


def _dedup_allow(result: dict) -> bool:
    """同机会集合 2h 内静默；集合变化（新股/新趋势）立即放行。

    2026-08-14 审计修复: 盘中每15分钟一轮扫描，同一批机会每轮重推 → 刷屏。
    签名 = 机会股票+趋势方向集合（剔除价格等易变字段）。"""
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from alert_dedup import get_deduper, make_sig  # noqa: PLC0415
        opps = result.get("opportunities", [])
        parts = sorted(f"{o.get('symbol')}:{o.get('trend', 'side')}" for o in opps)
        sig = make_sig("opp_set", *parts)
        return get_deduper().should_send("opportunity", sig, level="warn",
                                         same_sig_min_gap_s=2 * 3600)
    except Exception as e:  # noqa: BLE001 - 去重失败直发兜底，不吞机会
        print(f"[alert_dedup] 去重失败, 直发兜底: {e}", flush=True)
        return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="主板大市值机会扫描与本地产物维护")
    parser.add_argument("--min-score", type=float, default=3.0, help="最低加权信号分（默认3.0）")
    parser.add_argument("--top", type=int, default=30, help="scan_opportunities top_n参数")
    parser.add_argument("--dry-run", action="store_true", help="输出消息预览和扫描摘要，不发送")
    parser.add_argument("--notify", action="store_true", help="显式允许发送飞书通知；定时任务默认静默")
    parser.add_argument("--interval", action="store_true", help="高频模式（维护缓存）")
    parser.add_argument("--intraday", action="store_true",
                        help="盘中模式: 用分钟线分时信号检测(不是日K); 默认按当前时间自动选择")
    args = parser.parse_args()

    # V6.1: 盘中自动用分时, 盘后/盘前用日K
    intraday = args.intraday
    if not intraday and not args.interval:
        try:
            sys.path.insert(0, str(ROOT))
            from quant_platform.legacy import is_trading_time
            intraday = is_trading_time()
        except Exception:
            intraday = False

    # 运行扫描
    result = run_monitor(min_score=args.min_score, top_n=args.top, intraday=intraday)

    # 构建消息
    title, body = build_alert_text(result)

    if args.dry_run:
        print(f"\n=== {title} ===\n{body}\n")
        print(_sanitize_emoji(json.dumps(result, ensure_ascii=False, indent=2, default=str))[:500])
    elif not args.notify:
        print("扫描缓存已更新；买入信号/全量扫描播报默认关闭", flush=True)
    else:
        print(f"\n=== {title} ===", flush=True)
        n_opp = result.get("opportunities_found", 0)
        if n_opp == 0:
            # 2026-08-14 审计修复: 旧版"无机会也推送确认" → 盘中16轮×纯确认消息刷屏飞书。
            # 无机会静默（仅日志）；需调试时用 --dry-run。
            print("无机会，静默（不推飞书）", flush=True)
        elif _dedup_allow(result):
            ok = feishu_sender.send_markdown(body, title=title)
            print(f"飞书推送: {'[OK]' if ok else '[失败]'}", flush=True)
        else:
            print("同机会集合在静默窗口内，跳过飞书推送", flush=True)
