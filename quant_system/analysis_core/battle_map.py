"""
battle_map — 作战地图（V11 决策大脑，盘前 08:30 生成）

把 13 个短线模块 + 共振评分 + 机制识别 + 宏观否决 聚合成一页可执行指令:

  ❶ 竞价观察锚点 (3条, 盯紧就够)
  ❷ 攻击方向分组 (核心攻击/观察/回避, 带个股+策略)
  ❸ 个股风险清单 (逐股: 席位兑现/公告雷/高位滞涨)
  ❹ 建议仓位 = 决策卡仓位 × 宏观系数 × 机制系数 × 数据健康系数
  ❺ 整体置信度 (0-1)
  ❻ 产业链联动观察 (主线 → 传导关联板块, 次日接力候选)

数据降级: 任何模块失败不阻塞整体, 失败模块在 degraded 中标注。

输出: generated/battle_map_{date}.json + .md

用法:
  python3 -m quant_system.analysis_core.battle_map --date 2026-08-10 [--save]
"""

from __future__ import annotations
import logging

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.config import MARKET_DIR, ZT_EM_DAILY  # noqa: E402
from quant_system.analysis_core.data_contract import format_pct_value  # noqa: E402
from quant_system.analysis_core import resonance_scorer, industry_graph  # noqa: E402
from quant_system.analysis_core.regime_classifier import get_regime  # noqa: E402
from quant_system.analysis_core.macro_veto import get_veto  # noqa: E402

CST = timezone(timedelta(hours=8))


def _safe(fn, default=None):
    try:
        return fn()
    except Exception as e:
        return {"__error__": str(e)[:120]} if default is None else default


def _latest_stats(as_of: str | None = None) -> pd.DataFrame:
    df = pd.read_parquet(MARKET_DIR / "zt_daily_stats.parquet")
    if as_of:
        df = df[pd.to_datetime(df["date"]) <= pd.Timestamp(as_of)]
    return df


def _bid_watch(emo, stats, resonance, regime, risk_stocks) -> list[dict]:
    """3 条竞价锚点（多了记不住）。"""
    out = []
    # 2026-08-21 审计: int(NaN) 会抛 ValueError 导致整段 _bid_watch 静默清空。
    # 用 num() 容错（NaN→0），失败仍进 degraded 而非静默。
    _mb = stats.iloc[-1].get("max_board") if stats is not None and len(stats) else None
    max_board = int(_mb) if pd.notna(_mb) else 0
    # 锚点1: 最高板梯队竞价
    if max_board >= 3:
        out.append({
            "item": f"最高板({max_board}板梯队)竞价高开>3%",
            "signal": "乐观", "action": "持有/加仓前排"})
        out.append({
            "item": f"最高板({max_board}板梯队)竞价低开>2%",
            "signal": "退潮预警", "action": "减半仓"})
    # 锚点2: 主线概念竞价（共振第一主线）
    main = [r for r in resonance if r.get("level") == "主线"]
    if main:
        top = main[0]["board_name"]
        out.append({
            "item": f"主线[{top}]竞价涨停≥3家",
            "signal": "主线延续", "action": "参与二线"})
    # 锚点3: 风险股竞价
    if risk_stocks:
        r = risk_stocks[0]
        out.append({
            "item": f"风险股[{r['name']}]竞价低开>2%",
            "signal": "兑现确认", "action": "回避/减仓"})
    # 锚点4(兜底): 昨日涨停溢价
    if len(out) < 3 and stats is not None and len(stats):
        prem = stats.iloc[-1].get("premium")
        if pd.notna(prem):
            out.append({
                "item": f"昨日涨停平均竞价溢价 {format_pct_value(prem, digits=1, unit='pct')}",
                "signal": "正常" if prem >= 0 else "负溢价禁打板",
                "action": "打板" if prem >= 0 else "只低吸"})
    return out[:3]


def _risk_stocks(stats, em_day) -> list[dict]:
    """个股级风险: 高位兑现/炸板/大额卖出（EM 三池 + 天梯数据）。"""
    risks = []
    if em_day is None or em_day.empty:
        return risks
    zt = em_day[em_day["is_zt"]].copy()
    zb = em_day[em_day["is_zb"]].copy()
    # 高换手涨停（>20% = 筹码交换剧烈, 次日易分歧）
    for _, r in zt.sort_values("turnover", ascending=False).head(3).iterrows():
        if pd.notna(r.get("turnover")) and r["turnover"] > 20:
            risks.append({
                "code": r["code"], "name": r["name"],
                "risk": f"高换手{format_pct_value(r['turnover'], digits=0, unit='pct')}涨停(筹码松动)",
                "action": "次日竞价弱则走"})
    # 炸板股
    for _, r in zb.head(3).iterrows():
        risks.append({
            "code": r["code"], "name": r["name"],
            "risk": "炸板(封板失败)", "action": "不接力"})
    return risks[:5]


def _attack_groups(resonance, em_day, regime, code2con, con2codes) -> dict:
    """按共振分级 + 机制规则分组（带龙头个股）。

    龙头优先取 concept_board 官方领涨股（leader），fallback 成分∩涨停排序。
    """
    groups = {"core": [], "observe": [], "avoid": []}
    if em_day is None or em_day.empty:
        return groups
    zt = em_day[em_day["is_zt"]].copy()
    zt["code"] = zt["code"].astype(str).str.zfill(6)
    ban_zhuo = "禁用" not in regime.get("rules", {}).get("打板", "允许")
    strat_def = "打板(换手充分)" if ban_zhuo else "趋势低吸"

    # 官方领涨股映射 (board_name → leader)
    # 2026-08-21 审计修复: 原用 board_code(BKxxxx) 做键、却用概念名 r["concept"] 查询，
    # 键语义不匹配导致快路径从不命中。改用 board_name→leader_code/leader_name，
    # 并按 ts 取最近交易日行(concept_board 是全历史快照，dict() 会保留最后一行)。
    leader_map = {}
    cb = MARKET_DIR.parent / "classification" / "concept_board.parquet"
    if cb.exists():
        cbd = pd.read_parquet(cb)
        if "board_name" in cbd.columns and "leader_code" in cbd.columns:
            cbd["ts"] = pd.to_datetime(cbd["ts"], errors="coerce")
            cbd = cbd.dropna(subset=["ts"]).sort_values("ts")
            cbd = cbd.drop_duplicates(subset=["board_name"], keep="last")
            leader_map = dict(zip(cbd["board_name"], cbd["leader_code"]))
    name2code = {}
    if len(zt):
        name2code = dict(zip(zt["name"], zt["code"]))

    seen = set()
    for r in resonance:
        if r.get("level") not in ("主线", "支线") or r["board_name"] in seen:
            continue
        seen.add(r["board_name"])
        codes = con2codes.get(r["concept"], set())
        # 龙头: 官方 leader 优先
        leader = None
        ldr_code = str(leader_map.get(r["concept"], "") or "").zfill(6)
        if ldr_code and ldr_code in set(zt["code"]):
            sub = zt[zt["code"] == ldr_code]
            if not sub.empty:
                top = sub.iloc[0]
                leader = {"code": top["code"], "name": top["name"],
                          "boards": int(top["board_count"]) if pd.notna(top["board_count"]) else 1}
        if leader is None:
            sub = zt[zt["code"].isin(codes)]
            if not sub.empty:
                sub = sub.sort_values(["board_count", "amount"], ascending=False)
                top = sub.iloc[0]
                leader = {"code": top["code"], "name": top["name"],
                          "boards": int(top["board_count"]) if pd.notna(top["board_count"]) else 1}
        entry = {"name": r["board_name"], "score": r["score"],
                 "strategy": strat_def if r["level"] == "主线" else "轻仓试错",
                 "leader": leader}
        if r["level"] == "主线":
            groups["core"].append(entry)
        else:
            groups["observe"].append(entry)
    return {
        "core": groups["core"][:8],
        "observe": groups["observe"][:5],
        "avoid": groups["avoid"][:5],
        "core_total": len(groups["core"]),
    }


def _load_pattern_explanations(date: str) -> dict:
    """读取 pattern_report 生成的 RAG 规律解释缓存（只读，不触发检索）。

    缓存结构 {date: {ptype: {query, pattern_count, hits, cached, ...}}}，
    只取当日 date 键，规整为 {date, types: [{type, name, query, pattern_count,
    hits: 前3条, cached}], total_types}；文件缺失/损坏/无当日 → {}（优雅降级）。
    """
    path = ROOT / "data_warehouse" / "patterns" / "pattern_explanations.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    day = raw.get(date)
    if not isinstance(day, dict) or not day:
        return {}
    try:
        from quant_system.analysis_core.pattern_engine import TYPE_SPEC
        type_spec = TYPE_SPEC if isinstance(TYPE_SPEC, dict) else {}
    except Exception:
        type_spec = {}
    types = []
    for ptype, e in day.items():
        if not isinstance(e, dict):
            continue
        spec = type_spec.get(ptype, {})
        types.append({
            "type": ptype,
            "name": spec.get("name", ptype),
            "query": e.get("query", ""),
            "pattern_count": e.get("pattern_count", 0),
            "hits": [h for h in (e.get("hits") or []) if isinstance(h, dict)][:3],
            "cached": True,  # 凡从缓存文件读到的都是已缓存, 不透传文件内可能漂移的 cached 字段
        })
    return {"date": date, "types": types, "total_types": len(types)}


def build_action_tree(bm: dict) -> list[dict]:
    """把作战地图压缩为盘前三行行动卡。

    不编造实时价格：battle_map 无行情快照，价格区间只给相对锚点
    （竞价确认 / 昨收±2%）。
    """
    bm = bm if isinstance(bm, dict) else {}

    def _num(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _core_leader() -> str | None:
        groups = bm.get("attack_groups") if isinstance(bm.get("attack_groups"), dict) else {}
        for group in groups.get("core") or []:
            if not isinstance(group, dict):
                continue
            leader = group.get("leader")
            if isinstance(leader, dict) and leader.get("name"):
                return str(leader["name"])
            if group.get("name"):
                return str(group["name"])
        return None

    def _risk_advice() -> str | None:
        risks = bm.get("risk_watch") if isinstance(bm.get("risk_watch"), list) else []
        pieces: list[str] = []
        for risk in risks:
            if not isinstance(risk, dict):
                continue
            name = str(risk.get("name") or risk.get("code") or "").strip()
            action = str(risk.get("action") or "注意风险").strip()
            if name:
                pieces.append(f"{name} {action}")
            if len(pieces) >= 2:
                break
        return "；".join(pieces) if pieces else None

    def _discipline() -> str:
        recommended = str(bm.get("recommended") or "防守")
        if recommended == "进攻":
            return "主线内汰弱留强，破位止损"
        if recommended == "试错":
            return "小仓试错，破位即止损"
        return "不追高，破位止损"

    confidence = _num(bm.get("confidence"), 0.0)
    macro_veto = str(bm.get("macro_veto") or "").lower()

    if confidence >= 0.6:
        recommended = str(bm.get("recommended") or "防守")
        position_range = str(bm.get("position_range") or "").strip()
        direction_text = (f"{recommended} | 仓位 {position_range}"
                          if position_range else recommended)
        leader = _core_leader()
        price_text = (f"{leader} | 开盘后按龙头竞价确认，参考昨收±2%"
                      if leader else "无新增标的")
    else:
        if macro_veto == "hard":
            reason = "宏观 hard 否决"
        elif macro_veto == "soft":
            reason = "宏观 soft 压力"
        else:
            reason = "置信度低"
        direction_text = f"观望/不操作 | {reason}"
        price_text = "无新增标的"

    risk_advice = _risk_advice()
    stop_text = risk_advice or _discipline()

    return [
        {"line": "① 操作方向", "text": direction_text},
        {"line": "② 价格区间", "text": price_text},
        {"line": "③ 止损止盈", "text": stop_text},
    ]


def _social_sentiment_for_date(req_date: str) -> dict:
    """Load social sentiment without allowing current data into historical maps."""
    composite = __import__(
        "quant_system.analysis_core.social_sentiment",
        fromlist=["market_sentiment_composite"],
    ).market_sentiment_composite
    try:
        return composite(as_of=req_date)
    except TypeError:
        if req_date != datetime.now(CST).date().isoformat():
            return {"__excluded__": "social_sentiment does not support historical dates"}
        return composite()


def build_map(date: str | None = None) -> dict:
    req_date = date or datetime.now(CST).date().isoformat()
    date = req_date
    degraded: list[str] = []

    # ── 1. 数据健康（先查地基）──
    health = _safe(lambda: __import__("quant_system.analysis_core.data_health_check", fromlist=["run"]).run(skip_baostock=True), None)
    data_degraded = False
    if health is None or health == {}:
        degraded.append("数据健康检查自身失败 → 按降级处理")
        data_degraded = True
    elif not health.get("overall_ok", True):
        data_degraded = True
        degraded.append("数据健康降级: " + ", ".join(health.get("degraded_datasets", [])))

    # 非数据类降级（回退/模块失败）不压低仓位，仅影响置信度
    health_coef = 0.8 if data_degraded else 1.0

    # ── 2. 情绪周期 + 天梯 ──
    def _emotion_for_request_date():
        module = __import__("quant_system.analysis_core.emotion_cycle", fromlist=["run_today"])
        try:
            return module.run_today(req_date)
        except TypeError:
            # Keep lightweight/no-argument adapters compatible with the map
            # contract while production emotion_cycle remains date-aware.
            return module.run_today()

    emo = _safe(_emotion_for_request_date, {})
    if "__error__" in emo:
        degraded.append(f"emotion: {emo['__error__']}")
        emo = {"date": date, "stage": "unknown", "stage_cn": "未知", "confidence": 0}

    # V12.3 情绪接入(用户需求): 微博/小红书/B站/新闻 → 市场情绪综合指标。
    # 独立于涨停统计的情绪周期，作为短线情绪的补充轴。逐源适配器失败只记录不阻断。
    social = _safe(lambda: _social_sentiment_for_date(req_date), {})
    if social.get("__excluded__"):
        degraded.append("social_sentiment 不支持历史日期，已排除当前快照")
        social = {}
    if "__error__" in social:
        degraded.append(f"social_sentiment: {social['__error__']}")
        social = {}
    social_sent = social.get("market_sentiment")
    social_ok = social.get("coverage") if isinstance(social.get("coverage"), (int, float)) else 0
    # 社媒情绪置信度加成：只在该源有实际覆盖时计入（避免"空的当乐观"）
    if social_sent is not None and social_ok >= 0.5:
        if social_sent >= 0.05:
            conf_bump = 0.03
        elif social_sent <= -0.05:
            conf_bump = -0.03
        else:
            conf_bump = 0.0
    else:
        conf_bump = 0.0
    def _stats_for_request_date():
        try:
            return _latest_stats(req_date)
        except TypeError:
            # 兼容旧的无参适配器；返回后仍按请求日严格过滤，不能引入未来证据。
            raw = _latest_stats()
            if isinstance(raw, pd.DataFrame) and "date" in raw.columns:
                raw = raw[pd.to_datetime(raw["date"], errors="coerce") <= pd.Timestamp(req_date)]
            return raw

    stats = _safe(_stats_for_request_date, None)
    if stats is None or "__error__" in (stats if isinstance(stats, dict) else {}):
        degraded.append("zt_daily_stats 读取失败")
        stats = None
    elif isinstance(stats, pd.DataFrame) and len(stats):
        # 2026-08-10 审计：涨停统计陈旧 >3 日 → 降级标注且不参与分析，禁止静默用旧数据
        _stats_lag = (pd.Timestamp(req_date)
                      - pd.Timestamp(stats["date"].max())).days
        if _stats_lag < 0:
            degraded.append(f"zt_daily_stats 未来数据{stats['date'].max().date()} > 请求日{req_date}")
            stats = None
            _stats_lag = 0
        if _stats_lag > 3:
            degraded.append(f"zt_daily_stats 落后{_stats_lag}日(实际{stats['date'].max().date()})")
            stats = None

    # ── 3. 资金合力 ──
    def _forces_for_request_date():
        latest = __import__("quant_system.analysis_core.fund_forces", fromlist=["latest"]).latest
        try:
            return latest(as_of=req_date)
        except TypeError:
            # Keep compatibility with legacy/lightweight adapters while production
            # fund_forces remains date-aware. The returned frame is still checked
            # for freshness below, so this fallback cannot hide stale evidence.
            return latest()

    forces = _safe(_forces_for_request_date, None)
    if isinstance(forces, dict) and "__error__" in forces:
        degraded.append(f"资金合力读取失败: {forces['__error__']}")
        forces = None
    force_index = None
    if isinstance(forces, pd.DataFrame) and len(forces):
        _forces_lag = (pd.Timestamp(req_date)
                       - pd.Timestamp(forces["date"].max())).days
        if _forces_lag < 0:
            degraded.append(f"资金合力未来数据{forces['date'].max().date()} > 请求日{req_date}")
            forces = None
        elif _forces_lag > 3:
            degraded.append(f"资金合力落后{_forces_lag}日(实际{forces['date'].max().date()})")
        else:
            force_index = float(forces.iloc[-1]["force_index"])

    # ── 4. 共振评分（主线; 非交易日自动回退最近交易日）──
    res_df = _safe(lambda: resonance_scorer.score_day(date), pd.DataFrame())
    if not isinstance(res_df, pd.DataFrame) or res_df.empty:
        res_df = _safe(lambda: resonance_scorer.score_day(None), pd.DataFrame())
        if isinstance(res_df, pd.DataFrame) and len(res_df):
            analyze_date = str(res_df.iloc[0]["date"])[:10]  # 回退到实际分析日
            date = analyze_date
            degraded.append(f"非交易日 {req_date} → 回退分析 {analyze_date}")
    resonance = res_df.to_dict("records") if isinstance(res_df, pd.DataFrame) and len(res_df) else []
    if not resonance:
        degraded.append("resonance: 无评分(主题数据缺失)")

    # ── 5. 机制 + 宏观 ──
    regime = get_regime(date)
    veto = get_veto(date)

    # ── 6. 产业链联动（主线 → 传导板块 + chain_map 先验链）──
    linkage = []
    # 2026-08-14 审计P0-1: 原仅"主线"触发, 弱市无主线时"无强传导"空白
    # → 放宽为 主线 ∪ (支线且共振分≥0.8) 的前3
    main_lines = [r for r in resonance if r.get("level") == "主线"]
    branch_lines = [r for r in resonance
                    if r.get("level") == "支线" and float(r.get("score", 0) or 0) >= 0.8]
    for m in (main_lines or branch_lines)[:3]:
        ls = _safe(lambda m=m: industry_graph.links(m["board_name"], top_k=3), [])
        to = [f"{x['dst']}(lift{x['lift']})" for x in ls]
        entry = {"from": m["board_name"], "to": to}
        # chain_map 先验链并入（失败降级, 不影响原 linkage）
        try:
            from quant_system.analysis_core import chain_map
            props = chain_map.propagate_from_concept(m["board_name"], date) or []
            if props:
                p = props[0]
                entry.update({"chain": p["chain"], "layer": p["layer"],
                              "sectors": p.get("sectors", [])})
        except Exception as e:  # noqa: BLE001
            degraded.append(f"chain_map: {type(e).__name__}: {str(e)[:80]}")
        # 即使 industry_graph 无 lift 传导，仍允许纯 chain_map 先验链条目入榜（有 ⚙️ 标注区分）
        if to or entry.get("chain"):
            linkage.append(entry)

    # ── 7. 个股风险 + 攻击分组 ──
    em_day = _safe(lambda: (lambda d: pd.read_parquet(ZT_EM_DAILY)) (None), None)
    if isinstance(em_day, pd.DataFrame):
        em_day["date"] = pd.to_datetime(em_day["date"]).dt.strftime("%Y-%m-%d")
        em_day = em_day[em_day["date"] == date]
    risk_stocks = _safe(lambda: _risk_stocks(stats, em_day), [])
    # 涨跌停微观结构: 出货判别追加进风险清单（2026-08-10 新增模块）
    try:
        from quant_system.analysis_core.market_microstructure import run_today as micro_run
        micro = micro_run(date)
        for d in micro.get("distribution_risks", [])[:5]:
            risk_stocks.append({
                "code": d.get("code", ""), "name": d.get("name", ""),
                "risk": d.get("judge", "出货") + "(炸板" + str(d.get("zb_times", "?")) + "次)",
                "action": "不接力/回避",
            })
    except Exception as e:
        logging.getLogger(__name__).error(f"[battle_map] 操作失败: {e}", exc_info=True)
    # 龙头-跟风扩散度: 过热信号追加观察（2026-08-10 新增模块）
    try:
        from quant_system.analysis_core.leader_follower import top_signals
        for s in top_signals(n=4):
            risk_stocks.append({
                "code": "", "name": s.get("board_name", s.get("concept", "")),
                "risk": f"跟风过热({s.get('signal')}): {str(s.get('reason', ''))[:40]}",
                "action": "龙头见顶风险,防追高",
            })
    except Exception as e:
        logging.getLogger(__name__).error(f"[battle_map] 操作失败: {e}", exc_info=True)
    code2con, con2codes = _safe(lambda: resonance_scorer._load_concept_map(), ({}, {}))
    groups = _safe(lambda: _attack_groups(resonance, em_day, regime, code2con, con2codes),
                   {"core": [], "observe": [], "avoid": []})

    # ── 7.5 RAG 规律逻辑依据（读 pattern_report 生成的解释缓存，只读不检索）──
    rag_expl = _safe(lambda: _load_pattern_explanations(date), {})
    if not rag_expl:
        degraded.append("pattern_explanations 缓存缺失(运行 pattern_report 后生成)")

    # ── 7.6 研报工作流（研报 NLP + 图片 OCR + 产业链/龙头融合）──
    research_summary = {"n_reports": 0, "concepts": [], "chain_hits": [], "leader_hits": []}
    try:
        rf_path = ROOT / "generated" / f"research_flow_{date}.json"
        if rf_path.exists():
            rf = json.loads(rf_path.read_text(encoding="utf-8"))
            research_summary = {
                "n_reports": int(rf.get("n_reports", 0)),
                "concepts": rf.get("all_concepts", [])[:10],
                "codes": rf.get("all_codes", [])[:20],
                "companies": rf.get("all_companies", [])[:10],
                "chain_hits": rf.get("chain_hits", [])[:5],
                "leader_hits": rf.get("leader_hits", [])[:5],
                "concept_stocks": {k: v[:10] for k, v in rf.get("concept_stocks", {}).items()},
            }
            # 研报看好的标的后备入观察池（带来源 research，需人工/盘面验证）
            for concept, stocks in rf.get("concept_stocks", {}).items() if isinstance(rf.get("concept_stocks"), dict) else []:
                for st in stocks[:5]:
                    groups.setdefault("observe", []).append({
                        "code": st.get("code", ""), "name": st.get("name", st.get("code", "")),
                        "reason": f"研报概念[{concept}]", "source": "research_flow",
                    })
            # 研报链命中并入产业链联动（增加来源可追溯性）
            for ch in rf.get("chain_hits", [])[:3]:
                linkage.append({
                    "from": ch.get("concept", ""), "to": [],
                    "chain": ch.get("chain"), "layer": ch.get("layer"),
                    "source": "research_flow",
                    "temperature": ch.get("temperature"),
                })
    except Exception as e:  # noqa: BLE001
        degraded.append(f"research_flow: {type(e).__name__}: {str(e)[:80]}")

    # ── 8. 仓位 = 决策卡 × 宏观系数 × 机制系数 × 健康系数 ──
    def _decision_card_for_request_date():
        module = __import__("quant_system.analysis_core.decision_card", fromlist=["build_card"])
        try:
            return module.build_card(req_date)
        except TypeError:
            # Lightweight adapters used by callers/tests may expose the legacy
            # no-argument form; keep the map's output contract stable.
            return module.build_card()

    card = _safe(_decision_card_for_request_date, {})
    pos_range = card.get("position_range", "10-30%")
    try:
        lo, hi = (float(x.strip("%")) for x in pos_range.split("-"))
    except Exception:
        lo, hi = 10.0, 30.0
    # 审计 2026-08-20：macro_veto/regime 缺失时 position_coef/仓位系数 可能为 None
    # （降级可见），float(None) 会崩 → 统一容错为 1.0
    try:
        raw_macro_coef = veto.get("position_coef", 1.0)
        macro_coef = 1.0 if raw_macro_coef is None else float(raw_macro_coef)
    except (TypeError, ValueError):
        macro_coef = 1.0
    try:
        raw_regime_coef = regime.get("rules", {}).get("仓位系数", 1.0)
        regime_coef = 1.0 if raw_regime_coef is None else float(raw_regime_coef)
    except (TypeError, ValueError):
        regime_coef = 1.0
    pos_lo = round(lo * macro_coef * regime_coef * health_coef, 0)
    pos_hi = round(hi * macro_coef * regime_coef * health_coef, 0)

    # ── 9. 置信度（含预测校准权重反哺）──
    mod_w = _safe(lambda: __import__("quant_system.analysis_core.calibration", fromlist=["get_module_weights"]).get_module_weights(date), {})
    stage = emo.get("stage", "")
    emo_w = float(mod_w.get("emotion_cycle", 1.0)) if isinstance(mod_w, dict) else 1.0
    scen_w = float(mod_w.get("scenario", 1.0)) if isinstance(mod_w, dict) else 1.0
    # 置信度分解：保留每个加减项，避免最终数值无法解释或误读为校准概率。
    confidence_components = {
        "base": 0.45,
        "emotion": round((0.15 * emo_w if stage in ("ferment", "repair") else 0.05 * emo_w if stage == "climax" else 0.0), 3),
        "fund_forces": 0.10 if force_index is not None and force_index >= 60 else 0.0,
        "core_mainlines": round(0.10 * scen_w if len(groups["core"]) >= 2 else 0.0, 3),
        "macro_veto": -0.15 if veto.get("level") == "hard" else -0.05 if veto.get("level") == "soft" else 0.0,
        "user_feedback": -0.03 if isinstance(mod_w, dict) and any(k.startswith("user::") for k in mod_w) else 0.0,
        "degraded_sources": -0.05 if degraded else 0.0,
        "social_sentiment": round(conf_bump, 3),
    }
    confidence_components["raw"] = round(sum(confidence_components.values()), 3)
    confidence_components["final"] = round(min(0.95, max(0.2, confidence_components["raw"])), 2)
    conf = confidence_components["final"]

    # 去重观察池（防研报概念重叠标的多余）
    seen_obs = set(); dedup_obs = []
    for x in groups.get("observe", []):
        k = (x.get("code"), x.get("reason"))
        if k not in seen_obs:
            seen_obs.add(k); dedup_obs.append(x)
    groups["observe"] = dedup_obs[:10]

    _bw = _safe(lambda: _bid_watch(emo, stats, resonance, regime, risk_stocks), [])
    if isinstance(_bw, dict) and "__error__" in _bw:
        degraded.append(f"bid_watch: {_bw['__error__']}")
        _bw = []
    _bid_watch_safe = _bw

    bm = {
        "date": req_date,
        "analyze_date": date,
        "emotion_stage": emo.get("stage_cn", "未知"),
        "emotion_conf": emo.get("confidence", 0),
        # V12.3 社媒情绪轴（微博/B站/新闻等综合）: sentiment∈[-1,1], coverage=可用源比例
        "social_sentiment": {
            "value": round(social_sent, 3) if social_sent is not None else None,
            "coverage": round(float(social_ok), 2),
            "per_source": social.get("per_source", {}),
            "note": ("社媒情绪覆盖不足(可能需cookie)" if social_ok < 0.5
                     else "微博/B站/新闻综合情绪"),
        },
        "regime": regime.get("regime", "未知"),
        "macro_veto": veto.get("level", "none"),
        "bid_watch": _bid_watch_safe,
        "attack_groups": groups,
        "risk_watch": risk_stocks,
        "linkage": linkage,
        "rag_explanations": rag_expl,
        "research": research_summary,
        "position_range": f"{pos_lo:.0f}-{pos_hi:.0f}%",
        "position_basis": (f"决策卡{pos_range} × 宏观{macro_coef} × 机制{regime_coef}"
                           f" × 健康{health_coef}"),
        "confidence": conf,
        "confidence_components": confidence_components,
        "recommended": "进攻" if conf >= 0.65 else ("试错" if conf >= 0.5 else "防守"),
        "degraded": degraded,
    }
    # 唯一执行口径：上证MA300/融合门控优先于作战图独立建议。
    if card.get("long_trend_guard", {}).get("state") == "below":
        bm["recommended"] = "防守"
        bm["position_range"] = f"0-{min(float(pos_hi), 20):.0f}%"
        bm["position_authority"] = "fusion_decision+MA300_guard"
        bm["degraded"].append("统一仓位门控: 上证跌破MA300，覆盖作战图试错建议")
    bm["action_tree"] = build_action_tree(bm)
    out = ROOT / "generated" / f"battle_map_{req_date}.json"
    ROOT.joinpath("generated").mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bm, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "generated" / f"battle_map_{req_date}.md").write_text(render_md(bm), encoding="utf-8")
    return bm


def _render_social_sentiment(ss: dict) -> str:
    """社媒情绪单行渲染: 值/Coverage + 覆盖不足提示。"""
    v = ss.get("value")
    cov = ss.get("coverage", 0)
    if v is None:
        return f"社媒情绪(覆盖不足,{ss.get('note','')[:30]})"
    label = "偏多" if v >= 0.05 else ("偏空" if v <= -0.05 else "中性")
    return f"{v:+.2f}({label},覆盖{cov:.0%})"


def render_md(bm: dict) -> str:
    lines = [
        f"# 🗺️ 作战地图 {bm['date']}",
        "",
        f"**情绪**: {bm['emotion_stage']} | **机制**: {bm['regime']} | "
        f"**宏观**: {bm['macro_veto']} | **置信度**: {bm['confidence']} | **建议**: {bm['recommended']}",
        f"**仓位**: {bm['position_range']}  ({bm['position_basis']})",
        f"**社媒情绪**: {_render_social_sentiment(bm.get('social_sentiment', {}))}",
        "",
        "## ⚡ 行动卡",
    ]
    action_tree = bm.get("action_tree")
    if not action_tree:
        action_tree = build_action_tree(bm)
    for action in action_tree[:3]:
        lines.append(f"- **{action.get('line', '')}** {action.get('text', '')}")
    lines += [
        "",
        "## ❶ 竞价观察锚点 (9:25)",
    ]
    for i, b in enumerate(bm["bid_watch"], 1):
        lines.append(f"{i}. {b['item']} → {b['signal']} → {b['action']}")
    lines += ["", "## ❷ 攻击方向"]
    lines.append("**核心攻击**")
    for g in bm["attack_groups"]["core"]:
        ld = f"{g['leader']['name']}({g['leader']['boards']}板)" if g.get("leader") else "龙头待竞价确认"
        lines.append(f"- 🔴 {g['name']} (共振{g['score']}) | {g['strategy']} | 龙头: {ld}")
    lines.append("**观察**")
    for g in bm["attack_groups"]["observe"]:
        lines.append(f"- 🟡 {g['name']} (共振{g['score']}) | {g['strategy']}")
    lines += ["", "## ❸ 个股风险清单"]
    for r in bm["risk_watch"]:
        lines.append(f"- ⚠️ {r['name']}({r['code']}): {r['risk']} → {r['action']}")
    if not bm["risk_watch"]:
        lines.append("- 无（数据未覆盖）")
    lines += ["", "## ❹ 产业链联动（次日接力候选）"]
    _layer_cn = {"upstream": "上游", "midstream": "中游", "downstream": "下游"}
    _layer_word = {"upstream": "原料", "midstream": "制造", "downstream": "消费"}
    for lk in bm["linkage"]:
        line = f"- {lk['from']} → " + ", ".join(lk["to"])
        if lk.get("chain"):
            layer = lk.get("layer")
            cn = _layer_cn.get(layer, "未知")
            word = _layer_word.get(layer, "涉及")
            sec = "/".join((lk.get("sectors") or [])[:6])
            line += f" ⚙️ {lk['chain']}·{cn} {word} {sec}"
        lines.append(line)
    if not bm["linkage"]:
        lines.append("- 无强传导")
    # 💡 今日假设（选择题，hypothesis_generator 2026-08-10 新增）
    try:
        from quant_system.analysis_core.hypothesis_generator import generate as hyp_gen
        hyps = hyp_gen(bm["date"])
        if hyps:
            lines += ["", "## 💡 今日假设（选择题）"]
            for i, h in enumerate(hyps, 1):
                lines.append(f"{i}. **{h.get('hypothesis', '')}**")
                lines.append(f"   - 证据: {h.get('evidence', '')} → 建议: {h.get('action', '')} (置信{h.get('confidence', '')})")
    except Exception as e:
        logging.getLogger(__name__).error(f"[battle_map] 操作失败: {e}", exc_info=True)
    # 🧠 专家委员会（multi_agent 2026-08-10 新增）
    try:
        from quant_system.analysis_core.multi_agent import arbitrate as ma_arb
        ma = ma_arb(bm["date"])
        if ma.get("consensus"):
            lines += ["", "## 🧠 专家委员会"]
            votes = "; ".join(f"{v.get('agent')}:{v.get('view')}({v.get('confidence')})" for v in ma.get("votes", []))
            lines.append(f"- 共识: **{ma['consensus']}** (置信{ma['confidence']}) | 投票: {votes}")
            for dg in ma.get("disagreement", [])[:2]:
                lines.append(f"- ⚔️ 分歧: {dg.get('between', '')} — {dg.get('issue', '')}")
    except Exception as e:
        logging.getLogger(__name__).error(f"[battle_map] 操作失败: {e}", exc_info=True)
    # 📚 规律逻辑依据（RAG 解释缓存, battle_map 只读不检索 2026-08-12 新增）
    try:
        rag = bm.get("rag_explanations") or {}
        rag_types = rag.get("types") or []
        lines += ["", "## 📚 规律逻辑依据（RAG）"]
        if not rag_types:
            lines.append("- 当日无规律解释缓存（运行规律报告后生成）")
        else:
            for t in rag_types:
                if not isinstance(t, dict):
                    continue
                name = t.get("name") or t.get("type", "")
                cached_mark = "（cached）" if t.get("cached") else ""
                lines.append(f"- {name}（{t.get('pattern_count', 0)} 条规律）"
                             f"｜检索式: `{t.get('query', '')}`{cached_mark}")
                hits = t.get("hits") or []
                if not hits:
                    lines.append("  - 无检索命中")
                for h in hits[:3]:
                    lines.append(f"  - 📚 [{h.get('cat', '')}] {h.get('file', '')} "
                                 f"(score {h.get('score', '')})")
                    lines.append(f"    - {str(h.get('summary', ''))[:120]}")
    except Exception as e:
        logging.getLogger(__name__).error(f"[battle_map] 操作失败: {e}", exc_info=True)
    # ⚔️ 游资博弈（game_theory 2026-08-10 新增）
    try:
        from quant_system.analysis_core.game_theory_model import current_game
        gg = current_game(bm["date"])
        if gg.get("strategy"):
            lines += ["", "## ⚔️ 游资博弈格局"]
            lines.append(f"- 最优: **{gg['strategy']}** (EV {gg.get('ev')}, 胜率 {gg.get('win_rate')})")
            lines.append(f"- {gg.get('reason', '')}")
    except Exception as e:
        logging.getLogger(__name__).error(f"[battle_map] 操作失败: {e}", exc_info=True)
    # ❺ 龙虎榜资金 + ❻ 中长线低估池（after_close_extra 2026-08-14 新增,
    # 用户硬诉求: 盘后必须含 龙虎榜短线 + 中长线低估提醒; 缺失时优雅降级）
    try:
        ex = ROOT / "generated" / f"after_close_extra_{bm['date']}.json"
        if ex.exists():
            import json as _json
            extra = _json.loads(ex.read_text(encoding="utf-8"))
            lhb = extra.get("lhb") or {}
            bt, st = lhb.get("broker_top", []), lhb.get("stock_top", [])
            if bt or st:
                lines += ["", "## ❺ 龙虎榜资金（短线）"]
                for b in bt:
                    lines.append(f"- 🏦 {b['name']} 净买 {b['net']:.2f}亿 | 涉及: {b['stocks']}")
                for s in st[:6]:
                    fwd = (f" | 后1日 {s['fwd1']:+.1f}%" if s.get("fwd1") is not None
                           and s["fwd1"] == s["fwd1"] else "")
                    lines.append(f"- 💰 {s['name']}({s['code']}) 净买 {s['net']:.2f}亿 "
                                 f"({s['pct']:+.1f}%) {s['reason']}{fwd}")
            vp = extra.get("value_picks") or []
            if vp:
                lines += ["", "## ❻ 中长线低估池（价值提醒）"]
                for v in vp[:6]:
                    roe = f"ROE {v['roe']:.0f}%" if v.get("roe") is not None else "ROE -"
                    gr = f"增 {v['growth']:.0f}%" if v.get("growth") is not None else ""
                    lines.append(f"- 💎 {v['name']}({v['code']}) PE {v['pe']:.1f} PB {v['pb']:.2f} "
                                 f"{roe} {gr} | 距52周低+{v['dist52w']:.0f}%")
    except Exception as e:
        logging.getLogger(__name__).error(f"[battle_map] 盘后增强章节失败: {e}", exc_info=True)
    if bm["degraded"]:
        lines += ["", "## ⚠️ 数据降级"]
        for d in bm["degraded"]:
            lines.append(f"- {d}")
    lines += ["", "---", "*V11 作战地图 | 概率判断, 需竞价验证*"]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="作战地图")
    ap.add_argument("--date", default=None)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    bm = build_map(args.date)
    print(render_md(bm))
