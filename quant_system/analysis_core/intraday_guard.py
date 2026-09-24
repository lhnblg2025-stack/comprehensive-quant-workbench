"""
intraday_guard — 盘中预警（V11 实时风控，交易日 5 分钟轮询）

监控作战地图的条件单是否盘中触发，命中即推飞书/写告警文件:

  条件1: 炸板率 > 40%                    → 清非龙头
  条件2: 最高板(地图锚点) 竞价低开 >2%    → 减仓
  条件3: 主线概念竞价涨停家数 < 3        → 主线退潮预警
  条件4: 空间龙头炸板                    → 情绪退潮预警
  条件5: 个股风险清单股 盘中跌 >5%       → 风险确认

数据: akshare 实时涨停池/实时行情（本地已有工具链）;
告警: generated/intraday_alerts_{date}.jsonl + 可推送飞书。

用法:
  python3 -m quant_system.analysis_core.intraday_guard --check [--push]
"""

from __future__ import annotations
import logging

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.common import today  # noqa: E402

CST = timezone(timedelta(hours=8))


def _now() -> str:
    return datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S")


def _load_battle_map() -> dict:
    p = ROOT / "generated" / f"battle_map_{today()}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    # 回退到最近一份
    maps = sorted(ROOT.glob("generated/battle_map_*.json"))
    if maps:
        return json.loads(maps[-1].read_text(encoding="utf-8"))
    return {}


def _live_zt_count() -> tuple[int, int]:
    """实时涨停家数/炸板家数（akshare 东财涨停池）。失败返回 (-1,-1)。"""
    try:
        import akshare as ak
        df = ak.stock_zt_pool_em(date=today())
        zt = len(df)
        zb_df = ak.stock_zt_pool_zbgc_em(date=today())
        zb = len(zb_df)
        return zt, zb
    except Exception:
        return -1, -1


def _live_top_board() -> dict | None:
    """实时最高板个股（东财涨停池按连板数排序）。失败返回 None 并打日志。"""
    try:
        import akshare as ak
        df = ak.stock_zt_pool_em(date=today())
        if df is None or df.empty:
            return None
        if "连板数" in df.columns:
            top = df.sort_values("连板数", ascending=False).iloc[0]
            return {"name": top.get("名称"), "code": str(top.get("代码")).zfill(6),
                    "boards": int(top["连板数"])}
    except Exception as e:
        print(f"[intraday_guard] 最高板实时数据不可用: {str(e)[:80]}")
    return None


def _holdings() -> list[dict]:
    """持仓注入: trade_db 持仓 + 前端自选 watchlist。"""
    out = []
    try:
        from quant_system.trade_db import get_positions
        ps = get_positions()
        if isinstance(ps, list):
            for p in ps:
                out.append({"code": str(p.get("symbol", "")).zfill(6), "name": p.get("name", "")})
    except Exception as e:
        logging.getLogger(__name__).error(f"[intraday_guard] 操作失败: {e}", exc_info=True)
    try:
        wl = ROOT / "quant_web" / "watchlist.json"
        if wl.exists():
            import json as _json
            data = _json.loads(wl.read_text(encoding="utf-8"))
            items = data if isinstance(data, list) else data.get("stocks", [])
            for it in items:
                # 2026-08-14: watchlist.json 元素可能是裸字符串("002714")或 dict
                if isinstance(it, str):
                    code, name = it, it
                else:
                    code = str(it.get("code", it.get("symbol", "")))
                    name = it.get("name", code)
                code = code.strip()
                if code:
                    out.append({"code": code.zfill(6), "name": name})
    except Exception as e:
        logging.getLogger(__name__).error(f"[intraday_guard] 操作失败: {e}", exc_info=True)
    # 去重
    seen, dedup = set(), []
    for x in out:
        if x["code"] not in seen:
            seen.add(x["code"])
            dedup.append(x)
    return dedup


_quote_cache: dict[str, object] = {"path": None, "prices": {}}


def _realtime_price(symbol: str) -> float | None:
    """Read the current price from the same Tencent full-market snapshot.

    Risk checks and opportunity scans must share one quote timestamp. Falling
    back to per-stock Eastmoney requests caused noisy failures and mixed-time
    decisions, so a missing snapshot now degrades explicitly to ``None``.
    """
    snap_dir = ROOT / "data_warehouse" / "realtime_snapshot" / datetime.now(CST).strftime("%Y%m%d")
    files = sorted(snap_dir.glob("*.parquet")) if snap_dir.is_dir() else []
    if not files:
        return None
    path = files[-1]
    if _quote_cache.get("path") != path:
        try:
            frame = pd.read_parquet(path, columns=["code", "price"])
            prices = {
                str(row.get("code") or "")[-6:].zfill(6): float(row.get("price"))
                for row in frame.to_dict("records") if pd.notna(row.get("price"))
            }
            _quote_cache.update(path=path, prices=prices)
        except Exception as exc:
            logging.getLogger(__name__).warning("腾讯全市场快照价格缓存失败: %s", str(exc)[:120])
            return None
    value = (_quote_cache.get("prices") or {}).get(str(symbol).zfill(6))
    return float(value) if value is not None else None


def check(push: bool = False) -> list[dict]:
    # 交易时段保护: 仅 9:25-15:05 内执行（午休 11:35-12:55 跳过）
    hm = datetime.now(CST).strftime("%H:%M")
    in_morning = "09:25" <= hm <= "11:35"
    in_afternoon = "12:55" <= hm <= "15:05"
    if not (in_morning or in_afternoon):
        return []
    bm = _load_battle_map()
    alerts: list[dict] = []
    t = _now()

    # ── 条件1/2: 实时涨停与炸板 ──
    zt, zb = _live_zt_count()
    if zt >= 0:
        zb_rate = zb / zt if zt > 0 else 0.0
        if zb_rate > 0.40:
            alerts.append({"time": t, "level": "🔴", "cond": "炸板率>40%",
                           "msg": f"实时炸板率 {zb_rate:.0%} (炸{zb}/涨{zt}) → 清非龙头"})
        elif zb_rate > 0.30:
            alerts.append({"time": t, "level": "🟠", "cond": "炸板率>30%",
                           "msg": f"炸板率 {zb_rate:.0%} 抬升 → 防分歧"})

    # ── 条件3: 最高板/主线锚点检查（battle_map 竞价锚点 → 实时验证）──
    top = _live_top_board()
    if top:
        # 条件3a: 空间龙头低开/炸板
        px = _realtime_price(top["code"])
        prev = _prev_close(top["code"])
        if px and prev:
            chg = px / prev - 1
            # V12.3 审计 P1-5: 分支顺序对调——先判 -0.03(🔴红线) 再判 -0.02(🟠橙线),
            # 原实现 -0.02 在前, -0.03 分支永不触发("空间龙头跳水"红线漏报)。
            if chg < -0.03:
                alerts.append({"time": t, "level": "🔴", "cond": "空间龙头跳水",
                               "msg": f"最高板 {top['name']}({top['boards']}板) 现跌 {chg*100:.1f}% → 情绪退潮预警"})
            elif chg < -0.02:
                alerts.append({"time": t, "level": "🟠", "cond": "最高板低开",
                               "msg": f"最高板 {top['name']}({top['boards']}板) 竞价/现价低开 {chg*100:.1f}% → 减仓预警"})

    # ── 条件3.8: 持仓股个股级风险（注入的持仓逐个盯）──
    for h in _holdings():
        px = _realtime_price(h["code"])
        prev = _prev_close(h["code"])
        if px and prev and px / prev - 1 < -0.05:
            alerts.append({"time": t, "level": "🔴", "cond": "持仓跳水",
                           "msg": f"持仓 {h.get('name', h['code'])} 现跌 {(px/prev-1)*100:.1f}% → 触及止损线"})

    # ── 条件4: 风险清单股盘中跌幅 ──
    for r in bm.get("risk_watch", []):
        code = r.get("code", "")
        if not code:
            continue
        px = _realtime_price(code)
        if px is None:
            continue
        # 用 EM 昨日收盘近似昨收（简化: 用昨日收盘价）
        prev = _prev_close(code)
        if prev and prev > 0 and px / prev - 1 < -0.05:
            alerts.append({"time": t, "level": "🔴", "cond": "风险股跳水",
                           "msg": f"{r.get('name', code)} 现跌 {(px/prev-1)*100:.1f}% → 兑现确认,回避"})

    # 落盘
    if alerts:
        f = ROOT / "generated" / f"intraday_alerts_{today()}.jsonl"
        with open(f, "a", encoding="utf-8") as fh:
            for a in alerts:
                fh.write(json.dumps(a, ensure_ascii=False) + "\n")
        if push:
            _push_feishu("\n".join(f"{a['level']} {a['msg']}" for a in alerts),
                         title="🚨 盘中预警")
    return alerts


def _push_feishu(text: str, title: str) -> bool:
    """飞书推送（2026-08-14 修复: 原 quant_system.report_delivery 不存在,
    每次推送 ImportError 静默失败 → 改走 scripts/feishu_sender 独立机器人）。"""
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from feishu_sender import send_markdown  # noqa: PLC0415
        ok = send_markdown(text, title=title)
        if not ok:
            print(f"[intraday_guard] 飞书推送失败(发送器返回False)", file=sys.stderr)
        return ok
    except Exception as e:
        print(f"[intraday_guard] 推送失败: {e}", file=sys.stderr)
        return False


def _prev_close(code: str) -> float | None:
    """最近已收盘日收盘价（昨收）。末行若不是今天则末行即昨收，避免索引错位。"""
    # 条件单/自选+持仓盘中轮询里对每个标的反复 _prev_close 读本地 store;
    # V12.3 审计 P2-10: 加 30s TTL 内存缓存, 避免一轮 5min 轮询重复读同一标的历史,
    # 减少 IO 风暴(本地读与网络实时价串行时尤其明显)。
    _prev_cache = getattr(_prev_close, "_cache", None)
    if _prev_cache is None:
        _prev_close._cache = {"ts": 0.0, "m": {}}
        _prev_cache = _prev_close._cache
    if time.time() - _prev_cache["ts"] > 30:
        _prev_cache["m"].clear()
        _prev_cache["ts"] = time.time()
    if code in _prev_cache["m"]:
        return _prev_cache["m"][code]
    try:
        from quant_system.data_store import get_store
        s = get_store()
        df = s.get(code, days=5)
        if df is not None and len(df):
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            today = datetime.now(CST).strftime("%Y-%m-%d")
            closed = df[df["date"].dt.strftime("%Y-%m-%d") < today]
            if closed.empty:
                closed = df
            val = float(closed.iloc[-1]["close"])
            _prev_cache["m"][code] = val
            return val
    except Exception as e:
        logging.getLogger(__name__).error(f"[intraday_guard] 操作失败: {e}", exc_info=True)
    return None


# ════════════════════════════════════════════════════════════════
# 盘中决策链 (2026-08-14 增强 — 接续成熟盘中 agent，五视角评估)
# 大盘环境 → 基调(防御/谨慎/进攻) → 自选+持仓五视角 → 机会/风险清单
# 用户硬诉求: 机会提示要分角度(行情防御/短线趋势/中长线价值/资金/动量)并突出侧重
# ════════════════════════════════════════════════════════════════

def _market_bias() -> dict:
    """大盘环境基调: 指数涨跌 + 涨跌家数广度 → 防御/谨慎/进攻 + 证据。"""
    try:
        from quant_system.market_context import get_market_context
        ctx = get_market_context()
        ad = ctx.get("advance_decline", {}) or {}
        advance = int(ad.get("advance", 0) or 0)
        decline = int(ad.get("decline", 0) or 0)
        indices = ctx.get("indices", {}) or {}
        sh = indices.get("上证指数", {}) or {}
        pct = float(sh.get("change_pct", 0) or 0)
        total = advance + decline
        breadth = advance / total if total else 0.5
        if breadth < 0.35 or pct < -0.5:
            tone = "防御"
        elif breadth > 0.6 and pct >= 0:
            tone = "进攻"
        else:
            tone = "谨慎"
        return {"tone": tone, "advance": advance, "decline": decline,
                "breadth": round(breadth, 2), "index_pct": round(pct, 2),
                "summary": (ctx.get("_summary") or "")[:100]}
    except Exception as e:  # noqa: BLE001
        return {"tone": "未知", "error": str(e)[:80]}


def _watch_symbols() -> list[str]:
    """返回持仓+自选，供个股级风控检查使用。

    盘中决策链不再使用这个窄池；它使用 ``_full_market_quotes`` 扫描最新
    全市场快照后再输出高信号前排。保留本函数是为了不扩大逐只实时价风控请求。
    """
    out = []
    for h in _holdings():
        if h.get("code"):
            out.append(h["code"])
    try:
        wl = ROOT / "quant_web" / "watchlist.json"
        if wl.exists():
            import json as _json
            data = _json.loads(wl.read_text(encoding="utf-8"))
            items = data if isinstance(data, list) else data.get("stocks", [])
            for it in items:
                if isinstance(it, str):
                    code = it
                else:
                    code = str(it.get("code", it.get("symbol", "")))
                code = code.strip()
                if code:
                    out.append(code.zfill(6))
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).error(f"[intraday_guard] 操作失败: {e}", exc_info=True)
    # 空持仓/空自选必须诚实返回空。逐只风控没有标的时应跳过，不能用固定股票
    # 冒充扫描范围；全市场机会扫描由 _full_market_quotes() 独立负责。
    return sorted(set(out))


def _full_market_quotes() -> tuple[list[dict], str]:
    """读取最新全市场快照，返回行情行和真实快照时间。

    快照由 realtime_snapshot.py 批量腾讯接口生成，单轮约4-5秒，避免为5200只
    股票逐只调用报价/K线接口。没有快照时才使用 watchlist.fetch_quotes(None)
    作为明确的实时替代；两者都失败则返回空并标记原因。
    """
    snap_dir = ROOT / "data_warehouse" / "realtime_snapshot" / datetime.now(CST).strftime("%Y%m%d")
    files = sorted(snap_dir.glob("*.parquet")) if snap_dir.is_dir() else []
    if files:
        try:
            df = pd.read_parquet(files[-1])
            rows = df.to_dict("records")
            return rows, f"snapshot:{files[-1].name}"
        except Exception as exc:
            logging.getLogger(__name__).warning("全市场快照读取失败: %s", str(exc)[:100])
    try:
        from quant_system.watchlist import fetch_quotes
        rows = fetch_quotes(None, force=True)
        return rows, "watchlist.fetch_quotes(all)"
    except Exception as exc:
        logging.getLogger(__name__).error("全市场实时替代失败: %s", str(exc)[:120])
        return [], f"unavailable:{str(exc)[:80]}"


def _market_row_signal(q: dict, tone: str) -> dict | None:
    """Normalize every valid quote for the full short-term decision chain.

    This is a universe normalizer, not a signal filter. ST/BSE names remain in
    the analysis universe and are marked by downstream trading rules.
    """
    code = str(q.get("code") or q.get("symbol") or "")
    code = code[-6:] if len(code) > 6 else code.zfill(6)
    name = str(q.get("name") or code)
    price = float(q.get("price") or 0)
    pct = float(q.get("pct_chg") if q.get("pct_chg") is not None else q.get("change_pct") or 0)
    amount = float(q.get("amount_wan") or 0) * 10000
    if not price or not code.isdigit():
        return None
    turnover = float(q.get("turnover") or 0)
    pe = q.get("pe_ttm")
    pb = q.get("pb")
    # 量价只做初筛：强势、放量、低估和回撤各给透明分，不声称完整K线判断。
    score = 0.0
    reasons = []
    if pct >= 5:
        score += 3.0; reasons.append(f"涨幅{pct:+.2f}%")
    elif pct <= -5:
        score += 2.0; reasons.append(f"下跌预警{pct:+.2f}%")
    if amount >= 1e8:
        score += min(2.0, amount / 5e9); reasons.append(f"成交额{amount/1e8:.1f}亿")
    if turnover >= 3:
        score += min(1.5, turnover / 10); reasons.append(f"换手{turnover:.1f}%")
    try:
        if 0 < float(pe) <= 12 and 0 < float(pb) <= 1.5:
            score += 1.5; reasons.append(f"低估PE{float(pe):.1f}/PB{float(pb):.2f}")
    except (TypeError, ValueError):
        pass
    # No score threshold here. Full-universe analysis must retain weak rows so
    # they are explicitly ranked below stronger rows instead of disappearing.
    dominant = "防御" if pct <= -5 else ("价值" if any("低估" in x for x in reasons) else "动量")
    open_px = float(q.get("open") or 0)
    high = float(q.get("high") or 0)
    low = float(q.get("low") or 0)
    prev_close = float(q.get("pre_close") or q.get("prev_close") or 0)
    return {"symbol": code, "name": name, "price": price, "change_pct": pct,
            "open": open_px, "high": high, "low": low, "prev_close": prev_close,
            "amount": amount, "turnover": turnover, "pe": pe, "pb": pb,
            "dominant": dominant, "dominant_label": {"防御": "行情防御", "价值": "中长线价值", "动量": "技术动量"}[dominant],
            "summary": "；".join(reasons), "composite_conf": round(score, 2), "source": "full_market_snapshot"}


def _decision_evidence_for_codes(rows: list[dict]) -> dict:
    """Attach research, regulatory, holder and factor evidence to every row."""
    try:
        from scripts.unified_decision_snapshot import (_load_research, _research_evidence, _holder_evidence,
                                                       _load_holders, _regulatory_evidence, _load_inquiry,
                                                       _load_factor_decay, build_snapshot)
        research_raw, research_date = _load_research(datetime.now(CST).strftime("%Y-%m-%d"))
        research = _research_evidence(research_raw)
        holders = _holder_evidence(_load_holders(), datetime.now(CST).strftime("%Y-%m-%d"))
        regulatory = _regulatory_evidence(_load_inquiry())
        decay, decay_date = _load_factor_decay(datetime.now(CST).strftime("%Y-%m-%d"))
        decision = build_snapshot(mode="intraday", date=datetime.now(CST).strftime("%Y-%m-%d"))
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc)[:120], "research_date": None, "decay": {}}
    covered = hard = 0
    for row in rows:
        code = str(row.get("symbol") or "").zfill(6)
        ev = (research.get("codes") or {}).get(code, {"reports": 0, "top_concepts": []})
        reg = (regulatory.get("by_code") or {}).get(code, {})
        holder = (holders.get("by_code") or {}).get(code, {})
        row["research_evidence"] = {"reports": ev.get("reports", 0), "top_concepts": ev.get("top_concepts", [])}
        row["regulatory_risk"] = {"count": reg.get("count", 0), "tier1": reg.get("tier1", 0), "latest": reg.get("latest")}
        row["holder_evidence"] = {"as_of": holder.get("as_of"), "change_pct": holder.get("change_pct"), "signal": holder.get("signal")}
        row["hard_risk"] = bool(float(reg.get("tier1") or 0) > 0)
        if row["research_evidence"]["reports"]: covered += 1
        if row["hard_risk"]:
            hard += 1
            row["decision"] = "禁止交易/仅作风险观察"
            row["trade_reasons"] = [f"监管硬风险{reg.get('tier1')}条"]
        else:
            row["decision"] = "观察/等待盘口确认"
            row["trade_reasons"] = []
    return {"status": "available", "research_date": research_date, "research_reports": research.get("reports", 0),
            "research_codes": research.get("all_codes_count", 0), "covered_candidates": covered, "hard_risk_candidates": hard,
            "regulatory_total_hits": regulatory.get("total_hits", 0), "mainlines": decision.get("mainlines", [])[:8],
            "risk_flags": (decision.get("market") or {}).get("risk_flags", []),
            "factor_decay": {"date": decay_date, "degraded": decay.get("degraded", True), "reason": decay.get("reason"), "n_windows": decay.get("n_windows", 0)}}


def build_decision_chain(push: bool = False) -> dict:
    """Build the full-universe short-term decision chain.

    Every valid quote is retained and sent through the unified factor chain.
    ``push`` is kept for API compatibility but intentionally ignored: this
    chain is a local decision artifact for the GUI, never a stock-scan signal.
    """
    hm = datetime.now(CST).strftime("%H:%M")
    in_morning = "09:30" <= hm <= "11:35"
    in_afternoon = "12:55" <= hm <= "15:05"
    # A scheduled full-decision round may finish after 15:00. Reuse the last
    # same-day snapshot for completion instead of overwriting the day with 0 rows.
    if not (in_morning or in_afternoon):
        bias = _market_bias()
    else:
        bias = _market_bias()
    quotes, scan_source = _full_market_quotes()
    if not quotes and not (in_morning or in_afternoon):
        return {"time": _now(), "bias": {"tone": "休市"},
                "rows": [], "n_scanned": 0, "n_opportunities": 0,
                "skipped": "没有可复用的当日全市场快照"}
    candidate_rows = []
    for quote in quotes:
        try:
            row = _market_row_signal(quote, bias.get("tone", "谨慎"))
            if row is not None:
                candidate_rows.append(row)
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).warning("全市场行解析失败: %s", str(exc)[:100])
    candidate_rows.sort(key=lambda r: (-float(r.get("composite_conf", 0)), -float(r.get("amount", 0))))
    # 全部候选都进入本轮本地证据门控；页面/通知只截断展示，不能截断决策对象。
    visible_rows = candidate_rows
    evidence = _decision_evidence_for_codes(visible_rows)
    rows = visible_rows[:12]
    parsed_count = len(quotes)
    signal_evaluated = len(quotes)
    candidate_count = len(candidate_rows)
    evidence_evaluated = len(visible_rows)
    chain = {
        "schema_version": "intraday-full-decision.v1",
        "time": _now(), "bias": bias, "rows": rows,
        "candidate_rows": visible_rows,
        "decision_rows": visible_rows,
        "n_scanned": parsed_count, "n_candidates": candidate_count,
        "n_opportunities": candidate_count, "scan_source": scan_source,
        "evidence": evidence,
        "coverage": {
            "snapshot_rows": parsed_count,
            "signal_evaluated": signal_evaluated,
            "candidate_count": candidate_count,
            "evidence_evaluated": evidence_evaluated,
            "displayed": len(rows),
            "ratio": round(signal_evaluated / parsed_count, 4) if parsed_count else 0.0,
            "status": "complete" if parsed_count and signal_evaluated == parsed_count else "unavailable",
        },
        "analysis_scope": "全市场快照逐行轻量评估；研究/监管/股东证据附于前30条候选；页面仅展示前12条",
    }
    # 先落盘全市场链，再由统一短线快照生成最终5-10只主板候选。
    try:
        f = ROOT / "generated" / f"intraday_chain_{today()}.json"
        f.write_text(json.dumps(chain, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        from scripts.unified_decision_snapshot import build_snapshot
        decision = build_snapshot(mode="intraday", date=today(), live_rows=visible_rows)
        ranked = decision.get("opportunities") or []
        chain["rows"] = ranked[:12]
        chain["candidate_rows"] = ranked
        chain["decision_rows"] = ranked
        chain["short_term_candidates"] = []
        chain["decision_counts"] = decision.get("counts") or {}
        chain["market_risk_flags"] = (decision.get("market") or {}).get("risk_flags") or []
        chain["mainlines"] = decision.get("mainlines") or []
        chain["decision_snapshot"] = decision
        chain["analysis_scope"] = "全市场全部有效行情进入完整短线分析链；输出行情量价、技术动量、概念/题材、行业资金、个股资金、涨停梯队、龙虎榜、研报/催化、估值财务、公告监管、股东筹码、市场环境与风险分项；仅落盘，不发送股票扫描信号。"
        f.write_text(json.dumps(chain, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        # Keep a compact GUI artifact while retaining the full 5200-row chain above.
        compact = dict(decision)
        compact["opportunities"] = ranked[:120]
        compact["decision_rows"] = ranked[:120]
        compact["full_artifact"] = f.name
        (ROOT / "generated" / f"decision_snapshot_intraday_{today()}.json").write_text(
            json.dumps(compact, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
        )
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).error(f"[intraday_guard] 统一短线快照失败: {e}", exc_info=True)
    # Full decision chains are local artifacts for GUI/review only. Never push
    # a stock scan or a main-board buy signal from this path.
    return chain


def _push_chain_deduped(chain: dict) -> None:
    """Deprecated: stock decision chains are never pushed."""
    return



def _format_chain(chain: dict) -> str:
    """决策链文本（飞书/控制台）: 大盘基调 → 视角侧重分组的机会/风险清单。"""
    b = chain.get("bias", {}) or {}
    tone = b.get("tone", "?")
    icon = {"防御": "🛡", "谨慎": "⚖️", "进攻": "🚀"}.get(tone, "❓")
    ev = chain.get("evidence") or {}
    lines = [f"**盘中全市场分析链** {chain.get('time', '')} | 全部有效行情{chain.get('n_scanned', 0)}只 | 分析对象{chain.get('n_opportunities', chain.get('n_candidates', 0))}只",
             f"{icon} 大盘基调: **{tone}**",
             f"   数据源: {chain.get('scan_source', '-')} | 研究覆盖候选 {ev.get('covered_candidates', 0)} | 监管硬风险拦截 {ev.get('hard_risk_candidates', 0)} | 因子状态 {(ev.get('factor_decay') or {}).get('reason') or ('有效' if not (ev.get('factor_decay') or {}).get('degraded') else '降级')}",
             f"   主线参考: {'、'.join(str(x.get('industry') or x.get('name') or '') for x in (ev.get('mainlines') or [])[:5]) or '暂无'}" ]
    if b.get("index_pct") is not None:
        lines.append(f"   上证 {b.get('index_pct'):+.2f}% | 涨跌 {b.get('advance')}/{b.get('decline')} "
                     f"| 广度 {b.get('breadth')}")
    if tone == "防御":
        lines.append("   → 防御优先: 轻仓/不追高, 机会侧重中长线低估价值")
    elif tone == "谨慎":
        lines.append("   → 谨慎: 控制仓位, 只做高置信机会")
    rows = (chain.get("decision_rows") or chain.get("rows") or [])[:10]
    if not rows:
        lines.append("\n当前没有达到短线精选门槛的主板候选，不为凑数降低标准。")
    else:
        lines.append(f"\n📊 全市场短线分析排名前列 {len(rows)} 只（仅供观察，不是买入信号）")
        for r in rows[:10]:
            code = r.get("code") or r.get("symbol") or ""
            if "short_term_score" not in r:
                lines.append(f"\n   {code} {r.get('name', '')} | {r.get('dominant_label', '历史观察')} | {r.get('summary', '')}")
                continue
            reasons = "；".join(r.get("short_term_reasons") or [])
            blockers = "；".join(r.get("short_term_blockers") or []) or "无硬阻断"
            strategy = r.get("execution_strategy") or {}
            lines.append(f"\n   {code} {r.get('name', '')} | {r.get('short_term_score', 0)}分 | {r.get('trade_label', '观察')}")
            lines.append(f"   理由: {reasons}")
            lines.append(f"   限制: {blockers}")
            lines.append(f"   策略: {strategy.get('entry', '-')}；失效: {strategy.get('stop', '-')}")
    lines.append("\n本轮结果仅保存为全量分析产物，未发送主板扫描或买入信号。")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="盘中预警 + 决策链")
    ap.add_argument("--check", action="store_true", help="风控条件检查")
    ap.add_argument("--scan", action="store_true", help="兼容旧参数：运行全量分析并仅落盘")
    ap.add_argument("--full-decision", action="store_true", help="运行全量短线分析链并仅落盘")
    ap.add_argument("--push", action="store_true", help="已停用：不推送股票扫描或买入信号")
    ap.add_argument("--push-alerts", action="store_true", help="只推送命中的低密度风险告警")
    ap.add_argument("--push-scan", action="store_true", help="已停用：不推送盘中扫描报告")
    args = ap.parse_args()
    if args.scan or args.full_decision:
        chain = build_decision_chain(push=False)
        print(_format_chain(chain))
        print(f"\n[full-decision] 全市场分析{chain['n_scanned']}只, 排名对象{chain['n_opportunities']}只, "
              f"基调={chain['bias'].get('tone')}")
    if args.check:
        als = check(push=args.push_alerts)
        if als:
            for a in als:
                print(f"{a['level']} [{a['cond']}] {a['msg']}")
        else:
            print("✅ 无盘中预警触发")
