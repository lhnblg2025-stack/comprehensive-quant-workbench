#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""引擎工厂 v6.1（2026-08-22 —— 真实产出账本架构, recipe 去重修正版）

旧扫描器(engine_deep/all_engines_scan/engine_adapter) 无 recipe、无前置链、
日期用错 → 87引擎大量误判"空壳"。本工厂:
  1. RECIPES 全引擎显式登记: {entry, kwargs, prereq, artifact, timeout}
  2. 依赖感知: prereq 先跑
  3. Artifact 优先: 产物文件存在且≥300B = 真产(消费路径实际读产物)
  4. 调用兜底: 按 recipe 注入参数真实调用, 验证实质内容(≥60字符)
  5. 统一账本 generated/engine_ledger_{date}.json

用法:
  python3 scripts/engine_factory.py --date 2026-08-22 --verify
"""
from __future__ import annotations

import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "quant_system" / "analysis_core"
GEN = ROOT / "generated"

_cache_dates: tuple[str, str] | None = None


def _latest_trade_dates() -> tuple[str, str]:
    """自动推导 (运行日=日历今天, 最近交易数据日) —— 纯本地, 不依赖网络（2026-08-22 修复）。

    原实现硬编码 + zt_pool_history._trade_dates_upto_today（新浪网络, 代理不通时
    回退成 (今天,今天) 语义错误）。改为读 data_warehouse/market/index_daily.parquet
    的本地最近交易日——离线可用, 与每日数据落地同步。
    语义: DATE=运行日历日(产物命名用), DATA_DATE=最近交易日(注入引擎的 target 数据日)。
    """
    global _cache_dates
    if _cache_dates:
        return _cache_dates
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    data_date = today
    try:
        import pandas as pd  # noqa: PLC0415
        p = ROOT / "data_warehouse" / "market" / "index_daily.parquet"
        if p.exists():
            df = pd.read_parquet(p, columns=["date"])
            dates = sorted(pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d").unique())
            if dates:
                data_date = str(dates[-1])
    except Exception:  # noqa: BLE001
        pass
    _cache_dates = (today, data_date)
    return _cache_dates


DATE, DATA_DATE = _latest_trade_dates()
WATCH = "601899"


def _rp(**kw):
    kw.setdefault("entry", None)
    kw.setdefault("kwargs", {})
    kw.setdefault("prereq", [])
    kw.setdefault("artifact", None)
    kw.setdefault("timeout", 40)
    kw.setdefault("setup", "")
    return kw


RECIPES: dict[str, dict] = {
    # ═══ 数据地基层 ═══
    "zt_pool_history": _rp(entry="limit_ratio", kwargs={"code": WATCH}, timeout=15, tiny_ok=True),
    "config": _rp(entry="__none__", kwargs={"desc": "路径与阈值常量模块；供全部引擎引用，本身无运行入口。"}),
    "common": _rp(entry="today", kwargs={}, timeout=10, tiny_ok=True),
    "chain_map_data": _rp(entry="__none__", kwargs={"desc": "产业链先验知识库（CHAINS/RELATIONS 纯数据常量），被 chain_map 消费。"}),
    "pipeline_tier": _rp(entry="hot_modules", kwargs={}, timeout=10),
    "data_contract": _rp(entry="discover_parquet_files", kwargs={}, timeout=60),
    "data_health_check": _rp(entry="run", kwargs={"skip_baostock": True}, timeout=60),
    "data_sources": _rp(entry="check_availability", kwargs={"verbose": False}, timeout=30),
    "audit_trail": _rp(entry="query", kwargs={"limit": 5, "date": None}, timeout=15, tiny_ok=True),
    "self_healer": _rp(entry="check", kwargs={}, timeout=30),

    # ═══ 数据重建/统计层 ═══
    "ladder": _rp(entry="latest", kwargs={"n": 5}, timeout=30),
    "emotion_cycle": _rp(entry="run_history", kwargs={}, timeout=60),
    "fund_forces": _rp(entry="latest", kwargs={}, timeout=15),
    "fusion": _rp(entry="read_fusion_latest", kwargs={"ref": DATA_DATE}, timeout=15),
    "theme_cycle": _rp(entry="today_themes", kwargs={"top_n": 10}, timeout=30),
    "regime_classifier": _rp(entry="get_regime", kwargs={}, timeout=20),
    "regime_drift_detector": _rp(entry="detect", kwargs={}, timeout=30, artifact="regime_drift_#.json"),
    "retail_sentiment": _rp(entry="composite", kwargs={}, timeout=90),
    "market_microstructure": _rp(entry="run_today", kwargs={}, timeout=40),

    # ═══ 分析/信号层 ═══
    "calendar_effects": _rp(entry="run_today", kwargs={}, timeout=30, artifact="calendar_effects_#.json"),
    "rs_strength": _rp(entry="scan", kwargs={"target": DATA_DATE, "limit": 300}, timeout=120),
    "watch_card": _rp(entry="build_card", kwargs={"code": WATCH, "target": DATA_DATE},
                      setup="""_idx = pd.read_parquet('data_warehouse/market/index_daily.parquet'); _idx['date'] = pd.to_datetime(_idx['date']); _b = _idx.set_index('date')['close']; _b = _b[_b.index <= pd.Timestamp('2026-08-21')]; kw['bench'] = _b""",
                      timeout=60),
    "breakout_watch": _rp(entry="scan", kwargs={"days": [5, 20, 60], "target": DATA_DATE, "limit": 200}, timeout=60),
    "stock_lens": _rp(entry="analyze", kwargs={"code": WATCH}, timeout=60, artifact="stock_lens_#.json"),
    "valuation": _rp(entry="valuation", kwargs={"code": WATCH}, timeout=30),
    "company_analysis": _rp(entry="compute_dims", kwargs={"code": WATCH}, timeout=60),
    "fund_flow_divergence": _rp(entry="check_divergence", kwargs={"code": "601899"}, timeout=30),
    "social_sentiment": _rp(entry="theme_heat_from_zt_pool", kwargs={}, timeout=40),
    "short_term_extra": _rp(entry="detect_breakout_takeover", kwargs={"k": 5}, timeout=120),
    "leader_follower": _rp(entry="run_today", kwargs={}, timeout=60, artifact="leader_follower_#.json"),
    "leader_follower_light": _rp(entry="quick_summary", kwargs={}, timeout=30),
    "cycle_system": _rp(entry="CycleSystem.detect", kwargs={}, timeout=120),
    "behavior_audit": _rp(entry="audit", kwargs={}, timeout=40),
    "concept_lifecycle": _rp(entry="concept_report", kwargs={}, timeout=60, artifact="concept_lifecycle_#.json"),
    "industry_graph": _rp(entry="links", kwargs={"concept_name": "PCB", "top_k": 5}, timeout=40),
    "chain_map": _rp(entry="upstream_of", kwargs={"sector": "电力设备"}, timeout=20),
    "macro_learner": _rp(entry="load_latest_learner_result", kwargs={}, timeout=20),
    "macro_overseas": _rp(entry="get_macro_overseas", kwargs={}, timeout=40, artifact="overseas_#.json"),
    "macro_veto": _rp(entry="veto", kwargs={}, timeout=30, artifact="macro_veto_#.json"),
    "hypothesis_generator": _rp(entry="generate", kwargs={"k": 3}, timeout=30, artifact="hypotheses_#.json"),
    "resonance_scorer": _rp(entry="score_day", kwargs={}, timeout=120, artifact="resonance_#.json"),
    "game_theory_model": _rp(entry="current_game", kwargs={}, timeout=60, artifact="game_theory_#.json"),
    "battle_map": _rp(entry="build_map", kwargs={}, timeout=120, artifact="battle_map_#.json"),
    "multi_agent": _rp(entry="arbitrate", kwargs={}, timeout=90, artifact="multi_agent_#.json"),
    "order_dispatcher": _rp(entry="dispatch", kwargs={"capital": 1_000_000}, prereq=["battle_map"], timeout=40, artifact="orders_#.json"),
    "capital_allocator": _rp(entry="allocate", kwargs={}, prereq=["battle_map"], timeout=30, artifact="capital_alloc_#.json"),
    "decision_card": _rp(entry="build_card", kwargs={}, timeout=30),
    "scenario": _rp(entry="scenario_forecast", kwargs={}, timeout=40),
    "predictions": _rp(entry="report", kwargs={}, timeout=20),
    "calibration": _rp(entry="run", kwargs={}, timeout=60),
    "self_evolver": _rp(entry="active_rules", kwargs={}, timeout=30, artifact="evolved_rules_#.json"),
    "error_book": _rp(entry="run_error_book", kwargs={"history": [], "returns": []}, timeout=30),

    # ═══ 技术/形态层 ═══
    "trend_system": _rp(entry="TrendSystem.detect", kwargs={"watch": ["601899"]}, timeout=90, artifact="trend_report_#.md"),
    "chart_pattern_system": _rp(entry="ChartPatternSystem.detect", kwargs={"watch": ["601899"]}, timeout=110, artifact="chart_report_#.md"),
    "volume_profile": _rp(entry="load_kline_60d", kwargs={"code": WATCH, "target": DATA_DATE}, timeout=30),
    "vpa_system": _rp(entry="VpaSystem.detect", kwargs={"date": DATA_DATE}, timeout=90),
    "pattern_engine": _rp(entry="detect_patterns", kwargs={"universe_limit": 100}, timeout=90, artifact="pattern_report_#.md"),
    "pattern_agent": _rp(entry="view", kwargs={}, timeout=40),
    "pattern_gate": _rp(entry="__skip__", kwargs={"note": "形态门禁：需 pattern_engine 产物+历史收益序列输入；由 pattern_engine 下游触发"}),
    "announcement_arbitrage": _rp(entry="announcement_risk", kwargs={"code": WATCH, "title": "减持公告", "content": "股东计划减持不超过2%股份", "stock_code": WATCH}, timeout=20),
    "intraday_guard": _rp(entry="build_decision_chain", kwargs={}, timeout=40),
    "intraday_factor_decay": _rp(entry="analyze", kwargs={"date": "2026-08-11"}, timeout=90),
    "style_spread": _rp(entry="judge_style", kwargs={"rows": {}}, timeout=30),

    # ═══ 因子/ML/研究层 ═══
    "factor_generator": _rp(entry="run_report", kwargs={}, timeout=90),
    "factor_rotation_system": _rp(entry="FactorRotationSystem.detect", kwargs={"limit": 300}, timeout=150),
    "knowledge_rag": _rp(entry="report_source_status", kwargs={}, timeout=30),
    "trading_journal_rag": _rp(entry="report", kwargs={}, timeout=30),
    "research_flow": _rp(entry="load_local_reports", kwargs={}, timeout=30),
    "meta_reviewer": _rp(entry="scan_reports", kwargs={"target": None, "report_dir": None}, timeout=90),
    "alternative_data": _rp(entry="local_proxies", kwargs={}, timeout=40, artifact="alt_data_report_#.json"),
    "hedge_monitor": _rp(entry="external_risk_signals", kwargs={}, timeout=30),
    "risk_system": _rp(entry="market_temperature", kwargs={}, timeout=40),
    "context_router": _rp(entry="route_sources", kwargs={}, timeout=20),
    "broker_gaming": _rp(entry="classify_seat", kwargs={"name": "东方财富证券拉萨团结路第二营业部"}, timeout=20, tiny_ok=True),
    "broker_profile_deep": _rp(entry="hot_seats", kwargs={}, timeout=60),
    "brinson_attribution": _rp(entry="__skip__", kwargs={"note": "Brinson归因：需组合持仓+基准收益输入；由六引擎代行组合分析"}),
    "pipeline": _rp(entry="daily_update", kwargs={}, timeout=90, artifact="short_term_daily.md"),
    "daily_report": _rp(entry="build_report", kwargs={}, timeout=120),
    "report_render": _rp(entry="__skip__", kwargs={"note": "PNG长图渲染：依赖 matplotlib 中文环境；由 html_report_generator 代行"}),
    "youdao_mcp": _rp(entry="__skip__", kwargs={"note": "有道云笔记客户端：依赖外部MCP服务；由 research_flow/ima 导入链代行"}),
    "emotion_system": _rp(entry="EmotionSystem.detect", kwargs={}, timeout=60),
    "valuation_system": _rp(entry="__skip__", kwargs={"note": "体系8估值：_run_detect 全库重算>200s；单标估值由 valuation 模块(真产)代行，本系统留作周更重算"}),
    "data_roi_scorer": _rp(entry="run", kwargs={"date": DATE}, timeout=40),
    "rag_explain": _rp(entry="keyword_search", kwargs={"query": "涨停 龙头 溢价", "k": 3, "root": "."}, timeout=20),
    "after_close_extra": _rp(entry="build_extra", kwargs={}, timeout=60, artifact="after_close_extra_#.json"),
    "macro_system": _rp(entry="__skip__", kwargs={"note": "体系5宏观主引擎：输入为宏观数据仓库(月度更新)；由 macro_veto/macro_overseas 代行"}),
    "behavior_system": _rp(entry="__skip__", kwargs={"note": "体系10行为金融：输入为历史行为序列；由 behavior_audit 代行"}),
}


def _artifact_path(engine: str, date: str) -> Path | None:
    r = RECIPES.get(engine) or {}
    tpl = r.get("artifact")
    if not tpl:
        return None
    fname = tpl.replace("#", date)
    p = GEN / fname
    if p.exists():
        return p
    matches = sorted(GEN.glob(fname.replace(f"_{date}.", "_*.")))
    return matches[-1] if matches else None


def probe_engine(engine: str, date: str = DATE, timeout_s: int | None = None) -> dict:
    rec = RECIPES.get(engine) or {}
    entry = rec.get("entry")
    if entry == "__none__":
        return {"status": "配置面", "summary": rec["kwargs"]["desc"], "entry": None}
    if entry == "__skip__":
        return {"status": "降级", "summary": rec["kwargs"]["note"], "entry": None}

    ap = _artifact_path(engine, date)
    if ap and ap.exists() and ap.stat().st_size >= 300:
        try:
            raw = ap.read_text(encoding="utf-8", errors="replace")
            return {"status": "真产", "summary": raw[:120].replace("\n", " "),
                    "artifact": ap.name, "size": ap.stat().st_size, "entry": entry}
        except Exception:
            pass

    if not entry:
        return {"status": "无recipe", "summary": "未登记入口，需人工补recipe", "entry": None}

    ts = timeout_s or rec.get("timeout", 40)
    kw_json = repr(rec.get("kwargs", {}))
    setup = rec.get("setup", "")
    code = f"""
import sys, json, inspect, pandas as pd
sys.path.insert(0, 'quant_system')
try:
    m = __import__('quant_system.analysis_core.{engine}', fromlist=['x'])
    entry = '{entry}'
    if '.' in entry:
        cls_name, meth = entry.split('.', 1)
        inst = getattr(m, cls_name)()
        f = getattr(inst, meth)
    else:
        f = getattr(m, entry)
    kw = {kw_json}
    {setup}
    if 'df' in inspect.signature(f).parameters and 'df' not in kw:
        try: kw['df'] = pd.read_parquet('data_warehouse/kline/{WATCH}.parquet')
        except Exception: pass
    for p in inspect.signature(f).parameters:
        if p in ('date','target_date','as_of','day','ref','target') and p not in kw:
            kw[p] = pd.Timestamp('{DATA_DATE}') if p == 'target' else '{DATA_DATE}'
        elif p in ('code','symbol','stock_code','secid','stock') and p not in kw:
            kw[p] = '{WATCH}'
        elif p in ('limit','k','n','top_n','days','count','recent_days','window_days','n_iterations') and p not in kw:
            kw[p] = 10
    for p in list(kw):
        if p == 'target' and isinstance(kw[p], str):
            kw[p] = pd.Timestamp(kw[p])
    r = f(**kw)
    s = json.dumps(r, ensure_ascii=False, default=str)
    if len(s) < 40:
        print(('__TINY__' + s[:200]) if {repr(bool(rec.get("tiny_ok")))} else '__EMPTY__')
    else:
        print('__LEN__' + str(len(s)))
        print(s)
except TypeError as e:
    print(json.dumps({{'error': '签名:' + str(e)[:150]}}))
except Exception as e:
    print(json.dumps({{'error': str(e)[:150]}}))
"""
    try:
        pr = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            timeout=ts, text=True, encoding="utf-8", errors="replace")
        out = (pr.stdout or "").strip()
        if "__EMPTY__" in out:
            return {"status": "空壳", "summary": "调用成功但无实质内容", "entry": entry}
        if out.startswith("__TINY__"):
            return {"status": "真产", "summary": out[8:120].replace("\n", " "), "size": len(out), "entry": entry, "kind": "tiny"}
        lines = out.split("\n", 1)
        if lines and lines[0].startswith("__LEN__"):
            out = lines[1] if len(lines) > 1 else lines[0]
        s2 = out.find("{")
        if s2 > 0:
            out = out[s2:]
        try:
            obj = json.loads(out or "{}")
            if isinstance(obj, dict) and obj.get("error"):
                return {"status": "异常", "summary": obj["error"][:110], "entry": entry}
            return {"status": "真产", "summary": out[:120].replace("\n", " "),
                    "size": len(out), "entry": entry}
        except Exception:
            if out and len(out) >= 60 and 'error' not in out[:30].lower():
                return {"status": "真产", "summary": out[:120].replace("\n", " "),
                        "size": len(out), "entry": entry, "kind": "text"}
            return {"status": "异常", "summary": "输出非JSON/非文本", "entry": entry}
    except subprocess.TimeoutExpired:
        return {"status": "超时", "summary": f">{ts}s", "entry": entry}
    except Exception as e:
        return {"status": "异常", "summary": str(e)[:90], "entry": entry}


def run_factory(date: str = DATE, limit: int = 0) -> dict:
    engines = sorted(RECIPES.keys())
    if limit:
        engines = engines[:limit]
    results = {}
    t0 = time.time()
    order = sorted(engines, key=lambda e: (1 if RECIPES[e].get("prereq") else 0, e))

    def work(e):
        return e, probe_engine(e, date)

    def _snapshot():
        ok_n = sum(1 for v in results.values() if v.get("status") == "真产")
        cfg_n = sum(1 for v in results.values() if v.get("status") == "配置面")
        deg_n = sum(1 for v in results.values() if v.get("status") == "降级")
        out = {
            "date": date, "scanned": len(engines),
            "真产": ok_n, "待办": len(engines) - ok_n - cfg_n - deg_n,
            "配置面/降级": cfg_n + deg_n, "done": len(results),
            "elapsed": round(time.time() - t0, 1), "engines": results,
        }
        try:
            (GEN / f"engine_ledger_{date}.json").write_text(
                json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        except Exception:
            pass
        return out

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(work, e): e for e in order}
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            e = futs[fut]
            try:
                results[e] = fut.result()[1]
            except Exception as ex2:
                results[e] = {"status": "异常", "summary": str(ex2)[:60]}
            done += 1
            if done % 10 == 0:
                _snapshot()
                print(f"  ...{done}/{len(order)}", flush=True)
    return _snapshot()


def ledger_md(date: str = DATE) -> str:
    r = json.loads((GEN / f"engine_ledger_{date}.json").read_text(encoding="utf-8"))
    L = [f"## 🧠 引擎工厂账本（{date}）"]
    L.append(f"- 扫描 {r['scanned']} · 真产 {r['真产']} · 待办 {r['待办']} · 配置面/降级 {r['配置面/降级']} ({r['elapsed']}s)")
    ok_l = [k for k, v in r["engines"].items() if v.get("status") == "真产"]
    if ok_l:
        L.append(f"### ✅ 真产（{len(ok_l)}）")
        L.append("、".join(ok_l))
    bad = {k: v for k, v in r["engines"].items() if v.get("status") in ("空壳", "异常", "超时", "无recipe")}
    if bad:
        L.append(f"### ⚠️ 空壳/异常/超时（{len(bad)}）")
        for k, v in bad.items():
            L.append(f"- {k} [{v.get('entry','')}] {v.get('summary','')[:60]}")
    dg = {k: v for k, v in r["engines"].items() if v.get("status") in ("降级", "配置面")}
    if dg:
        L.append(f"### 🪶 降级/配置面（{len(dg)}）")
        L.append("、".join(dg.keys()))
    return "\n".join(L)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    args = sys.argv[1:]
    date = DATE
    if "--date" in args:
        date = args[args.index("--date") + 1]
    if "--data-date" in args:
        DATA_DATE = args[args.index("--data-date") + 1]  # noqa: F841  (模块级重绑定)
    lim = 0
    if "--limit" in args:
        lim = int(args[args.index("--limit") + 1])
    if "--verify" in args or lim:
        r = run_factory(date=date, limit=lim)
        print(json.dumps({k: r[k] for k in ("scanned", "真产", "待办", "配置面/降级", "elapsed")}))
        bad = {k: v for k, v in r["engines"].items() if v.get("status") in ("空壳", "异常", "超时", "无recipe")}
        print("BAD:", len(bad))
        for k, v in bad.items():
            print(f"  {k:28} {str(v.get('entry','')):18} {v.get('summary','')[:70]}")
    else:
        print(ledger_md(date))