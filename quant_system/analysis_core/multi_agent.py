"""
multi_agent — 多智能体轻量版（专家委员会）

把现有模块输出包装成 11 个领域专家 Agent 的独立观点 + 置信度，
仲裁者（加权投票）综合成最终决策；观点分歧时触发"辩论模式"。

  Agent 1 情绪面: emotion_cycle.run_today()          → 阶段/置信/明日倾向
  Agent 2 资金面: fund_forces.latest()               → 合力/游资机构北向方向
  Agent 3 题材面: theme_cycle.today_themes() + resonance_scorer.score_day()
                                                     → 主线数/最强主线/题材温度
  Agent 4 技术面: regime_classifier.get_regime()     → 机制/趋势方向
  Agent 5 风险面: macro_veto.get_veto() + self_healer.check()
                                                     → 宏观否决级/数据健康
  Agent 6 规律面: pattern_agent.view()               → 规律库 confirmed 加权投票
                                                     （confirmed 规律作证据折算投票权重）
  Agent 7 三屏趋势: TrendSystem().view()             → 沪深300 三屏潮汐 + 自选股同向率
  Agent 8 量价筹码: VpaSystem().view()               → 威科夫/量价/密集区/主力资金
  Agent 9 情绪周期: EmotionSystem().view()           → 六阶段情绪温度/极端标记
  Agent 10 宏观周期: MacroSystem().view()            → 宏观四驱动/资产映射/风险偏好
  Agent 11 风险纪律: RiskSystem().view()             → 风险等级/回撤/警告

仲裁规则:
  - 加权投票: 观点(多=+1/空=-1/震荡=0) × 置信度 × 权重
  - 权重: 情绪0.12/资金0.12/题材0.10/技术0.07/风险0.12/规律面0.09/
           三屏趋势0.10/量价筹码0.10/情绪周期0.08/宏观周期0.06/风险纪律0.04
  - 风险面 hard veto 时直接输出"防守"
  - 共识: 加权和 >0.3 多 / <-0.3 空 / 否则震荡
  - 分歧检测: 任一 agent 观点与共识相反且置信度>0.5 → 记入 disagreement
  - 魔鬼代言人(Devil's Advocate): 方向性共识(多/空)下，获胜方胜率在 45%-65%（接近分歧）时
    自动启动 3 轮攻讦/答辩/评分；获胜方 ≥2 条有效回应 → 维持原判，否则降级为'分歧'；
    辩论全程写 generated/audit_trail/YYYYMMDD.json 供复盘（零网络，规则拼装不用 LLM）
  - 单 agent 失败不阻塞: degraded 标记 + 权重重分配
  - 零网络，全部复用现有模块；辩论用双方 evidence 拼装，不用 LLM

集成点说明（不改 battle_map）:
  battle_map 置信度将参考 arbitrate().consensus。

输出: generated/multi_agent_{date}.json

用法:
  python3 -m quant_system.analysis_core.multi_agent [--date 2026-08-07]
"""

from __future__ import annotations
import logging

import argparse
import json
import sys
from datetime import date as date_type, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.emotion_cycle import run_today as emotion_run_today  # noqa: E402
from quant_system.analysis_core.fund_forces import latest as fund_latest  # noqa: E402
from quant_system.analysis_core.theme_cycle import today_themes  # noqa: E402
from quant_system.analysis_core.resonance_scorer import score_day as resonance_score_day  # noqa: E402
from quant_system.analysis_core.regime_classifier import get_regime  # noqa: E402
from quant_system.analysis_core.macro_veto import get_veto  # noqa: E402
from quant_system.analysis_core.self_healer import check as healer_check  # noqa: E402
from quant_system.analysis_core.common import num  # noqa: E402

CST = timezone(timedelta(hours=8))

BASE_WEIGHTS = {"情绪面": 0.12, "资金面": 0.12, "题材面": 0.10, "技术面": 0.07,
                "风险面": 0.12, "规律面": 0.09, "三屏趋势": 0.10, "量价筹码": 0.10,
                "情绪周期": 0.08, "宏观周期": 0.06, "风险纪律": 0.04}
OUT_DIR = ROOT / "generated"
VIEW_SCORE = {"多": 1.0, "空": -1.0, "震荡": 0.0, "防守": -1.0}
DEVIL_ADVOCATE_MIN = 0.45
DEVIL_ADVOCATE_MAX = 0.65
DEVIL_ADVOCATE_ROUNDS = 3
VALID_REBUTTALS_TO_UPHOLD = 2
ROLLING_IC_DAYS = 20
ROLLING_IC_MIN_PAIRS = 8
ROLLING_IC_IC_BLEND = 0.7
ROLLING_IC_STATIC_BLEND = 0.3
ROLLING_IC_FLOOR = 0.01
assert abs(ROLLING_IC_IC_BLEND + ROLLING_IC_STATIC_BLEND - 1.0) < 1e-9
VOTE_HIT_THRESHOLD = 0.005
VOTE_IC_BLEND = 0.5
VOTE_HIT_BLEND = 0.5
assert abs(VOTE_IC_BLEND + VOTE_HIT_BLEND - 1.0) < 1e-9


# ─────────────────────────── 各 Agent 观点 ───────────────────────────

def _view_emotion(date: str | None) -> dict:
    # 2026-08-21 审计: emotion_run_today() 不支持按日期取数，date 参数实际被忽略——
    # 历史回填(arbitrate(历史日期))时情绪面票用的是"今天"数据。显式标注数据时点，回填降级。
    raw = emotion_run_today()
    stage = raw.get("stage", "")
    conf = num(raw.get("confidence"), 0.5)
    ev = [f"阶段={raw.get('stage_cn', '?')}({stage}) 置信{conf:.2f}"]
    if date is not None and raw.get("date") not in (None, date):
        ev.append(f"[注:数据时点{raw.get('date','?')}≠{date},回填降级]")
    zt, mb, zb = raw.get("zt_cnt"), raw.get("max_board"), raw.get("zb_rate")
    if zt is not None:
        ev.append(f"涨停{zt}家")
    if mb is not None:
        ev.append(f"最高{mb}板")
    if zb is not None:
        ev.append(f"炸板率{zb:.0%}")
    tp = raw.get("transition_probs") or {}
    if tp:
        top = sorted(tp.items(), key=lambda kv: -kv[1])[0]
        ev.append(f"明日倾向={top[0]}({top[1]:.0%})")
    if stage in ("ice", "ebb"):
        view, conf = "空", max(conf, 0.6)
    elif stage in ("repair", "ferment", "climax"):
        view, conf = "多", max(conf, 0.5)
    else:  # divergence / 未知
        view, conf = "震荡", max(conf, 0.5)
    return {"agent": "情绪面", "view": view, "confidence": round(min(conf, 0.95), 2),
            "evidence": ev, "weight": BASE_WEIGHTS["情绪面"], "status": "ok", "detail": raw}


def _view_fund(date: str | None) -> dict:
    # 2026-08-21 审计: fund_latest() 不支持按日期取数，date 被忽略——历史回填时资金面票用今天数据。
    # 显式标注数据时点，回填降级。
    df = fund_latest()
    row = df.iloc[-1]
    fi = num(row["force_index"], 50)
    ev = [f"合力指数{fi:.0f}"]
    if date is not None and "date" in df.columns and str(df.iloc[-1]["date"])[:10] != str(date)[:10]:
        ev.append(f"[注:数据时点{str(df.iloc[-1]['date'])[:10]}≠{date},回填降级]")
    signs = {}
    try:
        signs = json.loads(row["signs"]) if isinstance(row.get("signs"), str) else {}
    except Exception as e:
        logging.getLogger(__name__).error(f"[multi_agent] 操作失败: {e}", exc_info=True)
    if signs:
        ev.append(" ".join(f"{k}{'+' if v > 0 else ('-' if v < 0 else '0')}"
                           for k, v in signs.items() if v is not None))
    for c, cn in [("youzi_net", "游资净买"), ("jg_net", "机构净买"), ("north_net", "北向净买")]:
        v = num(row.get(c))
        if v is not None:
            ev.append(f"{cn}{v / 1e8:+.1f}亿")
    mv = num(row.get("margin_delta"))
    if mv is not None:
        ev.append(f"融资Δ{mv:+.1f}亿")
    recent = [num(r["force_index"], 50) for _, r in df.tail(3).iterrows()]
    ev.append(f"近3日合力={'/'.join(f'{x:.0f}' for x in recent)}")
    if fi >= 60:
        view, conf = "多", fi / 100
    elif fi <= 40:
        view, conf = "空", (100 - fi) / 100
    else:
        view, conf = "震荡", 0.5
    detail = {"date": str(row["date"].date()), "force_index": fi, "signs": signs,
              "youzi_yi": round(num(row.get("youzi_net"), 0) / 1e8, 2),
              "jg_yi": round(num(row.get("jg_net"), 0) / 1e8, 2),
              "north_yi": round(num(row.get("north_net"), 0) / 1e8, 2),
              "margin_delta_yi": (round(mv, 2) if mv is not None else None)}
    return {"agent": "资金面", "view": view, "confidence": round(min(conf, 0.95), 2),
            "evidence": ev, "weight": BASE_WEIGHTS["资金面"], "status": "ok", "detail": detail}


def _view_theme(date: str | None) -> dict:
    themes_raw = today_themes()
    themes = themes_raw.get("themes", []) or []
    asof = date or themes_raw.get("date")
    main_count = sum(1 for t in themes if t.get("role") == "主线")
    ev = [f"活跃板块{len(themes)}个", f"主线{main_count}条"]
    if themes:
        top = themes[0]
        ev.append(f"最强={top.get('board', '?')}(涨停{top.get('zt_cnt', '?')})")
    reso = resonance_score_day(asof)
    top_score = num(reso.iloc[0]["score"]) if reso is not None and len(reso) else None
    reso_main = int((reso["level"] == "主线").sum()) if reso is not None and len(reso) else None
    if top_score is not None:
        ev.append(f"共振最高分{top_score:.2f}")
    if reso_main is not None:
        ev.append(f"共振主线{reso_main}条")
    hot = (main_count >= 2) or (reso_main is not None and reso_main >= 2) or \
          (top_score is not None and top_score >= 2.5)
    if hot:
        view, conf = "多", min(0.9, 0.55 + 0.08 * max(main_count, reso_main or 0) +
                               (0.1 if top_score is not None and top_score >= 2.5 else 0))
    elif main_count == 0 and (top_score is None or top_score < 1.0):
        view, conf = "空", 0.6
    else:
        view, conf = "震荡", 0.5
    detail = {"date": asof, "main_count": main_count, "top_score": top_score,
              "reso_main": reso_main,
              "top_themes": [t.get("board") for t in themes[:3]]}
    return {"agent": "题材面", "view": view, "confidence": round(min(conf, 0.95), 2),
            "evidence": ev, "weight": BASE_WEIGHTS["题材面"], "status": "ok", "detail": detail}


def _view_technical(date: str | None) -> dict:
    raw = get_regime(date)
    regime = raw.get("regime", "")
    bias = num(raw.get("bias_ma300"))
    ev = [f"机制={regime}"]
    if bias is not None:
        ev.append(f"MA300乖离{bias:+.2f}%")
    vol = num(raw.get("vol20_annual"))
    if vol is not None:
        ev.append(f"年化波动{vol:.1f}%")
    if "趋势市" in regime:
        view, conf = "多", min(0.9, 0.55 + abs(bias or 0) / 8)
    elif "下跌趋势" in regime:
        view, conf = "空", min(0.9, 0.55 + abs(bias or 0) / 8)
    elif "震荡市" in regime or "过渡市" in regime:
        view, conf = "震荡", 0.55
    else:
        view, conf = "震荡", 0.5
    detail = {k: raw.get(k) for k in ("date", "regime", "vol20_annual", "bias_ma300", "slope_ma300")}
    return {"agent": "技术面", "view": view, "confidence": round(min(conf, 0.95), 2),
            "evidence": ev, "weight": BASE_WEIGHTS["技术面"], "status": "ok", "detail": detail}


def _view_risk(date: str | None) -> dict:
    veto = get_veto(date)
    heal = healer_check(push=False)
    level = veto.get("level", "none")
    ev = list(veto.get("reasons", [])) or [f"宏观否决级={level}"]
    issues = heal.get("issues", []) or []
    mode = heal.get("decision_mode", "normal")
    if issues:
        ev.append(f"数据健康={mode}({len(issues)}项异常)")
    else:
        ev.append(f"数据健康={mode}")
    if level == "hard":
        view, conf = "防守", 1.0
    elif level == "soft":
        view, conf = "空", 0.7
    elif mode == "conservative":
        # 2026-08-21 审计: 数据健康降级态是"数据不可信"而非"看跌"，原映射"空"(0.8)
        # 会把委员会整体拉向防守。改为中性"震荡"(VIEW_SCORE=0)并降置信，仅压低该票权重。
        view, conf = "震荡", 0.4
    else:
        view, conf = "多", 0.55
    detail = {"veto_level": level, "position_coef": veto.get("position_coef"),
              "pe_percentile": veto.get("pe_percentile"),
              "buffett_ratio": veto.get("buffett_ratio"),
              "heal_mode": mode, "heal_issues": [i.get("msg") for i in issues[:3]]}
    return {"agent": "风险面", "view": view, "confidence": round(min(conf, 0.95), 2),
            "evidence": ev, "weight": BASE_WEIGHTS["风险面"], "status": "ok", "detail": detail}


def _degraded(agent: str, exc: Exception) -> dict:
    return {"agent": agent, "view": "震荡", "confidence": 0.0,
            "evidence": [f"模块异常: {str(exc)[:120]}"], "weight": BASE_WEIGHTS.get(agent, 0.0),
            "status": "degraded", "detail": {"error": str(exc)[:200]}}


def _system_view(agent: str, raw: dict) -> dict:
    """归一化体系模块 view() 为 multi_agent 标准观点（词表: 多/空/震荡/防守）。"""
    view = raw.get("view") or {"看多": "多", "看空": "空", "中性": "震荡"}.get(raw.get("signal"), "震荡")
    if view not in VIEW_SCORE:
        view = "震荡"
    conf = num(raw.get("confidence"), 0.5)
    ev = raw.get("evidence")
    return {"agent": agent, "view": view,
            "confidence": round(min(conf, 0.95), 2),
            "evidence": ev if isinstance(ev, list) else [],
            "weight": BASE_WEIGHTS[agent],
            "status": "ok" if raw.get("status") == "ok" else "degraded",
            "detail": raw.get("detail")}


def _view_trend(date: str | None) -> dict:
    from quant_system.analysis_core.trend_system import TrendSystem
    return _system_view("三屏趋势", TrendSystem().view(date))


def _view_vpa(date: str | None) -> dict:
    from quant_system.analysis_core.vpa_system import VpaSystem
    return _system_view("量价筹码", VpaSystem().view(date))


def _view_emotion_system(date: str | None) -> dict:
    from quant_system.analysis_core.emotion_system import EmotionSystem
    return _system_view("情绪周期", EmotionSystem().view(date))


def _view_macro(date: str | None) -> dict:
    from quant_system.analysis_core.macro_system import MacroSystem
    v = _system_view("宏观周期", MacroSystem().view(date))
    # 宏观规律（AI 自学习）：evidence 末尾追加 ≤2 条，读取失败静默跳过
    try:
        from quant_system.analysis_core.macro_learner import format_ai_rules, load_latest_learner_result
        rules = format_ai_rules(load_latest_learner_result(date), max_rules=2)
        if rules:
            v["evidence"] = list(v.get("evidence") or []) + rules
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).error(f"[multi_agent] 操作失败: {e}", exc_info=True)
    return v


def _view_risk_discipline(date: str | None) -> dict:
    from quant_system.analysis_core.risk_system import RiskSystem
    return _system_view("风险纪律", RiskSystem().view(date))


def agent_views(date: str | None = None) -> list[dict]:
    """11 个 agent 独立观点，每个独立 try/except，失败标 degraded。"""
    # Agent 1 降级时保持调用方传入 date（None=取最新），下游均以"取最新可用日"处理，不阻塞
    eff_date = date
    views: list[dict] = []
    # 情绪面先跑：其数据日期作为统一 as-of 日期（其他模块的 date 参数）
    try:
        v = _view_emotion(date)
        if not eff_date:
            eff_date = v.get("detail", {}).get("date") or eff_date
        views.append(v)
    except Exception as e:
        views.append(_degraded("情绪面", e))
    for name, fn in [("资金面", _view_fund), ("题材面", _view_theme),
                     ("技术面", _view_technical), ("风险面", _view_risk)]:
        try:
            views.append(fn(eff_date))
        except Exception as e:
            views.append(_degraded(name, e))
    # 规律面：pattern_agent（规律库 confirmed 规律折算投票权重），失败独立降级不阻塞
    try:
        from quant_system.analysis_core.pattern_agent import view as pattern_view
        raw = pattern_view(eff_date)
        if raw.get("view") not in VIEW_SCORE:
            raw["view"] = "震荡"
        views.append(raw)
    except Exception as e:
        views.append(_degraded("规律面", e))
    for name, fn in [("三屏趋势", _view_trend), ("量价筹码", _view_vpa),
                     ("情绪周期", _view_emotion_system), ("宏观周期", _view_macro),
                     ("风险纪律", _view_risk_discipline)]:
        try:
            views.append(fn(eff_date))
        except Exception as e:
            views.append(_degraded(name, e))
    return views


# ─────────────────────────── 仲裁与分歧 ───────────────────────────

def _effective_weights(views: list[dict]) -> dict[str, float]:
    """degraded agent 权重按比例重分配给健康 agent。"""
    base = {v["agent"]: BASE_WEIGHTS[v["agent"]] for v in views}
    healthy = [v for v in views if v["status"] != "degraded"]
    if not healthy:
        return base
    total = sum(base[v["agent"]] for v in healthy)
    if total < 1e-9:
        total = 1.0
    return {v["agent"]: (base[v["agent"]] / total if v["status"] != "degraded" else 0.0)
            for v in views}


def _parse_date(value) -> str | None:
    """把 YYYY-MM-DD / YYYYMMDD / date/datetime 归一为 YYYY-MM-DD；非法返回 None。"""
    if value is None:
        return None
    if isinstance(value, date_type) and not isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.date().isoformat()
    if hasattr(value, "isoformat"):
        try:
            return value.date().isoformat() if hasattr(value, "date") else value.isoformat()
        # 防御分支：非标准日期对象无法 .date()/.isoformat() 时跳过，继续尝试字符串解析。
        except (AttributeError, TypeError, ValueError):  # noqa: BLE001
            pass
    if not isinstance(value, str):
        return None
    s = value.strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            pass
    return None


def _asof_date(views: list[dict], date: str | None = None) -> str:
    """返回 views 的 as-of 日期：显式 date 优先，否则取首个 detail.date。"""
    parsed_date = _parse_date(date)
    if parsed_date:
        return parsed_date
    for v in views:
        detail = v.get("detail") or {}
        if detail.get("date"):
            parsed_detail = _parse_date(detail["date"])
            if parsed_detail:
                return parsed_detail
            return str(detail["date"])
    return datetime.now(CST).date().isoformat()


def _view_history(date: str | None = None, days: int = ROLLING_IC_DAYS) -> list[dict]:
    """读最近 days 个 generated/multi_agent_*.json，合并为 [{date, agent, view, status}]。

    只保留 status == "ok" 且 view 在 VIEW_SCORE 词表中的条目；某日期缺失的 agent
    不生成记录，等价于"无观点"。文件名日期（而非文件内 date）作为对齐日期。
    """
    target = _parse_date(date) or datetime.now(CST).date().isoformat()
    candidates: list[tuple[str, Path]] = []
    for path in OUT_DIR.glob("multi_agent_*.json"):
        stem_date = _parse_date(path.stem.removeprefix("multi_agent_"))
        if stem_date and stem_date <= target:
            candidates.append((stem_date, path))
    candidates.sort(key=lambda kv: kv[0])
    records: list[dict] = []
    for file_date, path in candidates[-days:]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for vote in data.get("votes", []) or []:
            view = vote.get("view")
            if vote.get("status") != "ok" or view not in VIEW_SCORE:
                continue
            agent = vote.get("agent")
            if not agent:
                continue
            records.append({"date": file_date, "agent": agent, "view": view, "status": "ok"})
    records.sort(key=lambda r: (r["date"], r["agent"]))
    return records


def _load_ret_next_map() -> dict[str, float]:
    """读取 generated/temperature_history.parquet 的 date → ret_next 映射。"""
    path = OUT_DIR / "temperature_history.parquet"
    if not path.exists():
        return {}
    try:
        df = pd.read_parquet(path)
    except Exception:
        return {}
    if not {"date", "ret_next"} <= set(df.columns):
        return {}
    out: dict[str, float] = {}
    try:
        dates = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    except Exception:
        return {}
    for day, ret in zip(dates, df["ret_next"]):
        out[str(day)] = float(ret) if not pd.isna(ret) else float("nan")
    return out


def _rank_normalize(values: dict[str, float | None], missing: float = 0.5) -> dict[str, float]:
    """把一组分数做 rank 归一化到 [0, 1]；缺失值使用中性分 missing。

    排序后最小值为 0、最大值为 1，并列项取平均 rank。
    """
    out = {agent: missing for agent in values}
    present = [(agent, value) for agent, value in values.items() if value is not None]
    if not present:
        return out
    present.sort(key=lambda kv: kv[1])
    n = len(present)
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs(present[j][1] - present[i][1]) < 1e-12:
            j += 1
        avg_rank = (i + j - 1) / 2.0
        norm = avg_rank / (n - 1) if n > 1 else 0.5
        for agent, _ in present[i:j]:
            out[agent] = norm
        i = j
    return out


def _vote_hit_rates(history: list[dict], ret_map: dict[str, float]) -> dict[str, float | None]:
    """按最近 20 日窗口统计每个 agent 的方向命中率。

    - ret_next > VOTE_HIT_THRESHOLD 视为实际"多"，< -阈值视为"空"，否则"震荡"。
    - 专家 view 的防守按"空"处理（与 VIEW_SCORE 一致）。
    - 实际"震荡"的日期不参与命中率统计；有效配对不足 ROLLING_IC_MIN_PAIRS 个时
      该 agent 返回 None（回退中性分）。
    """
    agents_seen = {h.get("agent") for h in history if h.get("agent")}
    counts: dict[str, tuple[int, int]] = {agent: (0, 0) for agent in agents_seen}

    records: list[dict] = []
    for h in history:
        day = h.get("date")
        agent = h.get("agent")
        view = h.get("view")
        if not day or not agent or view not in VIEW_SCORE:
            continue
        if day not in ret_map:
            continue
        ret = ret_map[day]
        if pd.isna(ret):
            continue
        if ret > VOTE_HIT_THRESHOLD:
            actual = "多"
        elif ret < -VOTE_HIT_THRESHOLD:
            actual = "空"
        else:
            continue
        records.append({"date": day, "agent": agent, "view": view, "actual": actual})

    dates = sorted({r["date"] for r in records})
    if len(dates) > ROLLING_IC_DAYS:
        allowed = set(dates[-ROLLING_IC_DAYS:])
        records = [r for r in records if r["date"] in allowed]

    for r in records:
        view_dir = "空" if r["view"] == "防守" else r["view"]
        valid, hits = counts.get(r["agent"], (0, 0))
        counts[r["agent"]] = (valid + 1, hits + (1 if view_dir == r["actual"] else 0))

    return {agent: (hits / valid) if valid >= ROLLING_IC_MIN_PAIRS else None
            for agent, (valid, hits) in counts.items()}

def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """归一化为和为 1；全零/空时回退 BASE_WEIGHTS。"""
    total = sum(weights.get(agent, 0.0) for agent in BASE_WEIGHTS)
    if total <= 1e-12:
        return {agent: BASE_WEIGHTS[agent] for agent in BASE_WEIGHTS}
    return {agent: weights.get(agent, 0.0) / total for agent in BASE_WEIGHTS}


def _rolling_ic_weights(history: list[dict], target_date: str | None = None) -> dict[str, float] | None:
    """20 日滚动 IC + 方向命中率 rank 融合动态权重；历史不足时返回 None。

    - 观点日期 T 对应 temperature_history.parquet 的 ret_next[T]（即 T+1 全 A 等权收益）；
      ret_next 为 NaN 或缺失的日期剔除。
    - 每个 agent 至少 8 个有效配对才计算 Spearman IC；不足的 agent 在 IC 权重中
      保持其静态权重占比。
    - IC<=0 的 agent 压到 0.01 而非 0；正 IC 原值参与归一。
    - 随后对 IC 权重和方向命中率分别 rank 归一化，最终滚动权重
      w = VOTE_IC_BLEND*ic_rank + VOTE_HIT_BLEND*hit_rank。
    - 本函数只返回滚动 rank 融合权重；与静态权重的 0.7/0.3 混合由
      _effective_weights_ic 完成。
    """
    target = _parse_date(target_date)
    if not target:
        target = datetime.now(CST).date().isoformat()
    # C-1: target 当日的观点对应 ret_next[T]（即 T+1 收益），是未来信息；
    # 滚动 IC 只允许 date < target 的历史配对进入窗口。
    history = [h for h in history
               if (day := _parse_date(h.get("date"))) and day < target]
    ret_map = _load_ret_next_map()
    if not ret_map:
        return None
    # C-2: 防御性截断，确保 ret_map 中不存在 target 当日或未来日期的收益。
    ret_map = {d: v for d, v in ret_map.items() if d < target}

    joined: list[dict] = []
    for h in history:
        day = _parse_date(h.get("date"))
        if not day or day not in ret_map:
            continue
        ret = ret_map[day]
        if pd.isna(ret):
            continue
        joined.append({"date": day, "agent": h.get("agent"), "view": h.get("view")})

    valid_dates = sorted({h["date"] for h in joined})
    # M-1: 整体窗口需 >=20 个有效观点日期（含 ret_next 的配对）；单 agent 配对 >=8
    # 才计算该 agent IC，不足则该 agent 保持静态占比。
    if len(valid_dates) < ROLLING_IC_DAYS:
        return None

    seqs: dict[str, list[tuple[float, float]]] = {agent: [] for agent in BASE_WEIGHTS}
    for h in joined:
        agent = h.get("agent")
        if agent in seqs:
            seqs[agent].append((VIEW_SCORE[h["view"]], ret_map[h["date"]]))

    ic_raw: dict[str, float | None] = {}
    for agent, pairs in seqs.items():
        if len(pairs) < ROLLING_IC_MIN_PAIRS:
            ic_raw[agent] = None
            continue
        scores = [p[0] for p in pairs]
        rets = [p[1] for p in pairs]
        try:
            res = spearmanr(scores, rets)
            val = float(getattr(res, "statistic", None) or getattr(res, "correlation", 0.0))
            if pd.isna(val):
                val = 0.0
        except Exception:
            val = 0.0
        ic_raw[agent] = val

    # M-3: IC -> 归一化 IC 权重；样本不足的 agent 用静态权重参与 rank。
    ic_weights: dict[str, float] = {}
    for agent, static_w in BASE_WEIGHTS.items():
        val = ic_raw.get(agent)
        if val is None:
            ic_weights[agent] = static_w
            continue
        w = max(val, 0.0)
        if w <= 0.0:
            w = ROLLING_IC_FLOOR
        ic_weights[agent] = w
    ic_weights = _normalize_weights(ic_weights)

    ic_rank = _rank_normalize(ic_weights)
    hit_rates = _vote_hit_rates(history, ret_map)
    hit_rates_full = {agent: hit_rates.get(agent) for agent in BASE_WEIGHTS}
    hit_rank = _rank_normalize(hit_rates_full)

    rolling = {
        agent: VOTE_IC_BLEND * ic_rank[agent] + VOTE_HIT_BLEND * hit_rank[agent]
        for agent in BASE_WEIGHTS
    }
    return _normalize_weights(rolling)

def _redistribute_weights(views: list[dict], base: dict[str, float]) -> dict[str, float]:
    """按 views 的 degraded 状态重分配 base 权重；健康 agent 按比例归一。"""
    base = {v["agent"]: base.get(v["agent"], BASE_WEIGHTS.get(v["agent"], 0.0))
            for v in views}
    healthy = [v for v in views if v["status"] != "degraded"]
    if not healthy:
        return base
    total = sum(base[v["agent"]] for v in healthy)
    if total < 1e-9:
        total = 1.0
    return {v["agent"]: (base[v["agent"]] / total if v["status"] != "degraded" else 0.0)
            for v in views}


def _effective_weights_ic(views: list[dict], date: str | None = None) -> tuple[dict[str, float], str]:
    """滚动 IC + 方向命中率优先的生效权重；窗口不足回退 _effective_weights。

    返回 (weights, note)。滚动 rank 权重按 0.7/0.3 与静态权重混合，再做 degraded 重分配。
    """
    target_date = _asof_date(views, date)
    degraded = [v["agent"] for v in views if v["status"] == "degraded"]
    # _view_history 包含 target 当日文件；滚动 IC 会排除 target，因此多取一天保证
    # 在过滤后仍有至少 ROLLING_IC_DAYS 个目标日之前的观点日期。
    history = _view_history(target_date, ROLLING_IC_DAYS + 1)
    rolling_weights = _rolling_ic_weights(history, target_date)

    base_txt = "/".join(f"{k}{v:.2f}" for k, v in BASE_WEIGHTS.items())
    if rolling_weights is None:
        weights = _effective_weights(views)
        note = "观点历史不足20日，回退静态权重"
        if degraded:
            parts = " ".join(f"{v['agent']}:{round(weights[v['agent']], 3)}" for v in views)
            note += f"；降级重分配[{','.join(degraded)} 失效] → {parts}"
        return weights, note

    # 与静态权重按 0.7/0.3 混合，再归一化和 degraded 重分配。
    blended = {
        agent: ROLLING_IC_IC_BLEND * rolling_weights[agent] +
               ROLLING_IC_STATIC_BLEND * BASE_WEIGHTS[agent]
        for agent in BASE_WEIGHTS
    }
    blended = _normalize_weights(blended)
    weights = _redistribute_weights(views, blended)

    ic_txt = "/".join(f"{k}{rolling_weights.get(k, 0.0):.2f}" for k in BASE_WEIGHTS)
    parts = " ".join(f"{v['agent']}:{round(weights[v['agent']], 3)}" for v in views)

    ret_map = {d: v for d, v in _load_ret_next_map().items() if d < target_date}
    hit_history = [h for h in history
                   if (day := _parse_date(h.get("date"))) and day < target_date]
    hit_rates = _vote_hit_rates(hit_history, ret_map)
    present_hits = {a: r for a, r in hit_rates.items() if r is not None}
    hit_txt = ""
    if present_hits:
        top_agent = max(present_hits, key=present_hits.get)
        avg_hit = sum(present_hits.values()) / len(present_hits)
        hit_txt = f"；方向命中率 top={top_agent} {present_hits[top_agent]:.0%} 均值{avg_hit:.0%}"

    note = f"原静态权重 {base_txt}；滚动IC权重 {ic_txt}{hit_txt} → 生效权重 {parts}"
    if degraded:
        note += f"；降级重分配[{','.join(degraded)} 失效]"
    return weights, note

def _is_conflict(view: str, consensus: str) -> bool:
    """agent 观点是否与共识相反（防守视为空方向）。"""
    if view == consensus:
        return False
    v, c = ("空" if view == "防守" else view), ("空" if consensus == "防守" else consensus)
    if c == "震荡":
        return v in ("多", "空")
    if v == "震荡":
        return False
    return (v == "多" and c == "空") or (v == "空" and c == "多")


def _opposite(a: str, b: str) -> bool:
    av, bv = ("空" if a == "防守" else a), ("空" if b == "防守" else b)
    return (av == "多" and bv == "空") or (av == "空" and bv == "多")


def _implication(view_a: str, view_b: str) -> str:
    a, b = ("空" if view_a == "防守" else view_a), ("空" if view_b == "防守" else view_b)
    table = {
        ("多", "空"): "方向背离：多头与空头证据并存，反弹持续性存疑，宜降仓观望",
        ("空", "多"): "方向背离：空头与多头证据并存，反抽力度存疑，不宜追高",
        ("多", "震荡"): "多头证据与中性信号并存，上行需新资金/题材确认",
        ("空", "震荡"): "空头证据与中性信号并存，反弹力度存疑",
        ("震荡", "多"): "中性信号为主但存在多头声音，方向待确认",
        ("震荡", "空"): "中性信号为主但存在空头声音，防守优先",
    }
    return table.get((a, b), "信号分歧，等待方向确认")


def _build_disagreements(views: list[dict], consensus: str, votes: list[dict],
                         conf_consensus: float) -> list[dict]:
    active = [v for v in views if v["status"] != "degraded"]
    by_name = {v["agent"]: v for v in active}
    dis: list[dict] = []
    seen: set[str] = set()

    def _side(name: str) -> dict:
        if name == "共识":
            ev = []
            for v in votes:
                if v["view"] == consensus and v["agent"] in by_name:
                    src = by_name[v["agent"]]["evidence"]
                    if src:
                        ev.append(f"{v['agent']}: {src[0]}")
            return {"name": "共识", "view": consensus, "confidence": conf_consensus,
                    "evidence": ev[:3] or [f"加权共识={consensus}({conf_consensus:.2f})"]}
        v = by_name[name]
        return {"name": name, "view": v["view"], "confidence": v["confidence"],
                "evidence": list(v["evidence"])}

    def _add(between: str, issue: str, a: str, b: str, implication: str):
        if between in seen:
            return
        seen.add(between)
        dis.append({"between": between, "issue": issue, "implication": implication,
                    "side_a": _side(a), "side_b": _side(b)})

    for v in active:
        if v["confidence"] > 0.5 and _is_conflict(v["view"], consensus):
            _add(f"{v['agent']} vs 共识({consensus})",
                 f"{v['agent']}看{v['view']}(置信{v['confidence']:.0%})，共识为{consensus}({conf_consensus:.2f})",
                 v["agent"], "共识", _implication(v["view"], consensus))
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            a, b = active[i], active[j]
            if a["confidence"] > 0.5 and b["confidence"] > 0.5 and _opposite(a["view"], b["view"]):
                _add(f"{a['agent']} vs {b['agent']}",
                     f"{a['agent']}看{a['view']}(置信{a['confidence']:.0%}) vs "
                     f"{b['agent']}看{b['view']}(置信{b['confidence']:.0%})",
                     a["agent"], b["agent"], _implication(a["view"], b["view"]))
    return dis


# ─────────────────────── 魔鬼代言人（Devil's Advocate）───────────────────────

def _vote_counts(consensus: str, votes: list[dict]) -> tuple[int, int]:
    """方向性票数统计: (与共识同向票, 反方向票)。防守视为空；震荡/degraded 不计。"""
    target = "空" if consensus == "防守" else consensus
    aligned = opposing = 0
    for v in votes:
        if v.get("status") == "degraded" or v.get("view") == "震荡":
            continue
        d = "空" if v.get("view") == "防守" else v.get("view")
        if d == target:
            aligned += 1
        else:
            opposing += 1
    return aligned, opposing


def _win_rate(consensus: str, votes: list[dict]) -> float | None:
    """获胜方胜率 = 同向票 / 方向性票（多/空）。非方向性共识(震荡/防守)返回 None 不触发。"""
    if consensus not in ("多", "空"):
        return None
    aligned, opposing = _vote_counts(consensus, votes)
    total = aligned + opposing
    if total == 0:
        return None
    return aligned / total


def _split_sides(views: list[dict], consensus: str) -> tuple[list[dict], list[dict]]:
    """按共识拆分获胜方/反对侧专家（均按置信度降序，degraded 排除）。"""
    target = "空" if consensus == "防守" else consensus
    winning: list[dict] = []
    opposing: list[dict] = []
    for v in views:
        if v.get("status") == "degraded":
            continue
        d = "空" if v.get("view") == "防守" else v.get("view")
        if d == target:
            winning.append(v)
        elif d in ("多", "空"):
            opposing.append(v)
    winning.sort(key=lambda x: -x["confidence"])
    opposing.sort(key=lambda x: -x["confidence"])
    return winning, opposing


def _evidence_pool(side: list[dict], limit: int | None = None) -> list[str]:
    """侧专家证据池（去重保序；limit 截断，供 top 证据选取）。"""
    pool: list[str] = []
    for v in side:
        for e in v.get("evidence") or []:
            if e not in pool:
                pool.append(e)
                if limit is not None and len(pool) >= limit:
                    return pool
    return pool


def _build_counter_arguments(winning: list[dict], opposing: list[dict],
                             consensus: str) -> list[dict]:
    """魔鬼代言人: 反对侧最强专家对获胜方 top 证据生成 3 条攻讦（零网络规则拼装）。"""
    top = _evidence_pool(winning, limit=DEVIL_ADVOCATE_ROUNDS)
    if not top:
        top = [f"{v['agent']}观点证据" for v in winning] or ["获胜方证据链"]
    opp = opposing[0]
    opp_ev = next(iter(opp.get("evidence") or []), None) or f"{opp['agent']}反对观点"
    opp_dir = "空头" if consensus == "多" else "多头"
    out = []
    for i in range(DEVIL_ADVOCATE_ROUNDS):
        ev = top[i % len(top)]
        out.append({
            "round": i + 1,
            "针对的证据": ev,
            "反驳逻辑": (f"反对侧专家「{opp['agent']}」证据「{opp_ev}」与「{ev}」方向相反，"
                         f"构成对获胜证据链的直接证伪路径"),
            "可能的反向场景": (f"若「{opp_ev}」所代表的{opp_dir}信号在未来1-2个交易日持续强化，"
                              f"则「{ev}」的指向将失效，市场按{opp_dir}演绎"),
        })
    return out


def _build_rebuttals(pool: list[str], attacks: list[dict],
                     consensus: str) -> list[dict]:
    """获胜方逐条答辩: 引用未用过且与被攻讦证据不同的独立证据；无可用证据则逻辑不一致。"""
    used: set[str] = set()
    out = []
    for i, att in enumerate(attacks, 1):
        target = att["针对的证据"]
        candidates = [e for e in pool if e not in used and e != target][:2]
        if candidates:
            used.update(candidates)
            out.append({
                "round": i, "针对的证据": target,
                "回应": f"引用独立证据「{'」「'.join(candidates)}」支撑{consensus}方向，该攻讦未动摇原证据链",
                "引用证据": candidates, "逻辑一致性": "一致", "结论": consensus,
            })
        else:
            out.append({
                "round": i, "针对的证据": target,
                "回应": f"无法引用独立证据回应攻讦，{consensus}方向证据链存在被证伪风险",
                "引用证据": [], "逻辑一致性": "不一致", "结论": consensus,
            })
    return out


def _score_rebuttal(rebuttal: dict) -> dict:
    """答辩评分（规则化）: 引用独立新证据(≠被攻讦证据)且逻辑一致 → 有效。"""
    cited = rebuttal.get("引用证据") or []
    targeted = rebuttal.get("针对的证据")
    new_cited = [e for e in cited if e != targeted]
    consistent = rebuttal.get("逻辑一致性") == "一致"
    valid = bool(new_cited) and consistent
    score = round((1.0 if valid else 0.0) + min(len(new_cited), 2) * 0.5, 2)
    if not new_cited:
        reason = "无新证据支撑（未引用独立证据）"
    elif not consistent:
        reason = "逻辑不一致（回应与获胜方向相悖）"
    else:
        reason = f"引用{len(new_cited)}条独立证据且逻辑一致"
    return {"score": score, "valid": valid, "reason": reason}


def _final_verdict(valid_flags: list[bool]) -> str:
    """终审: ≥2 条有效回应 → 维持原判；否则降级'分歧'。"""
    if sum(1 for f in valid_flags if f) >= VALID_REBUTTALS_TO_UPHOLD:
        return "维持"
    return "分歧"


def _devil_advocate(views: list[dict], consensus: str, votes: list[dict]) -> dict | None:
    """触发条件满足时执行完整魔鬼代言人流程；否则返回 None（旧输出不变）。"""
    win_rate = _win_rate(consensus, votes)
    if win_rate is None or not (DEVIL_ADVOCATE_MIN <= win_rate <= DEVIL_ADVOCATE_MAX):
        return None
    winning, opposing = _split_sides(views, consensus)
    if not winning or not opposing:
        return None
    attacks = _build_counter_arguments(winning, opposing, consensus)
    pool = _evidence_pool(winning)
    rebuttals = _build_rebuttals(pool, attacks, consensus)
    rounds = []
    for att, reb in zip(attacks, rebuttals):
        sc = _score_rebuttal(reb)
        rounds.append({**att, **reb, "score": sc["score"],
                       "valid": sc["valid"], "score_reason": sc["reason"]})
    verdict = _final_verdict([r["valid"] for r in rounds])
    aligned, opposing_n = _vote_counts(consensus, votes)
    return {
        "triggered": True,
        "original_consensus": consensus,
        "final_consensus": consensus if verdict == "维持" else "分歧",
        "win_rate": round(win_rate, 4),
        "directional_votes": {"aligned": aligned, "opposing": opposing_n},
        "winning_side": {"experts": [v["agent"] for v in winning],
                         "top_evidence": _evidence_pool(winning, limit=DEVIL_ADVOCATE_ROUNDS)},
        "opposing_side": {"experts": [v["agent"] for v in opposing],
                          "top_evidence": _evidence_pool(opposing, limit=DEVIL_ADVOCATE_ROUNDS)},
        "rounds": rounds,
        "verdict": verdict,
    }


def _write_devil_advocate_audit(res: dict, dev: dict) -> Path:
    """辩论全程写 generated/audit_trail/YYYYMMDD.json，供复盘当时决策逻辑。"""
    day = (res.get("date") or "").replace("-", "")
    if not day:
        day = datetime.now(CST).strftime("%Y%m%d")
    rec = {
        "ts": res["generated_at"],
        "kind": "devil_advocate",
        "date": res.get("date", ""),
        "vote": {
            "consensus_before": dev["original_consensus"],
            "consensus_after": res["consensus"],
            "win_rate": dev["win_rate"],
            "directional_votes": dev["directional_votes"],
            "weighted_sum": res["weighted_sum"],
            "confidence": res["confidence"],
        },
        "sides": {
            "winning": dev["winning_side"],
            "opposing": dev["opposing_side"],
        },
        "rounds": dev["rounds"],
        "verdict": dev["verdict"],
    }
    p = OUT_DIR / "audit_trail" / f"{day}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return p


def arbitrate(date: str | None = None, devil_advocate: bool = True) -> dict:
    """加权投票仲裁: 输出共识/置信度/votes/disagreement，并落盘 generated/。

    胜率 45%-65% 且 devil_advocate=True 时自动追加魔鬼代言人辩论，终审未扛过则共识降级'分歧'。"""
    views = agent_views(date)
    eff_date = _asof_date(views, date)
    eff_w, weights_note = _effective_weights_ic(views, eff_date)
    hard_veto = any(v["agent"] == "风险面" and v["view"] == "防守" and v["status"] == "ok"
                    for v in views)
    ws = sum(VIEW_SCORE[v["view"]] * v["confidence"] * eff_w[v["agent"]] for v in views)

    if hard_veto:
        consensus, conf = "防守", 0.85
    elif ws > 0.3:
        consensus, conf = "多", abs(ws) / 0.5
    elif ws < -0.3:
        consensus, conf = "空", abs(ws) / 0.5
    else:
        consensus, conf = "震荡", abs(ws) / 0.5
    conf = round(min(conf, 1.0), 2)

    votes = []
    for v in views:
        votes.append({"agent": v["agent"], "view": v["view"], "confidence": v["confidence"],
                      "value": VIEW_SCORE[v["view"]], "weight": round(eff_w[v["agent"]], 4),
                      "status": v["status"]})

    disagreement = _build_disagreements(views, consensus, votes, conf)

    res = {
        "date": eff_date,
        "consensus": consensus,
        "confidence": conf,
        "weighted_sum": round(ws, 4),
        "votes": votes,
        "disagreement": disagreement,
        "weights_note": weights_note,
        "note": "battle_map 置信度将参考 arbitrate().consensus（仅说明，不改 battle_map）",
        "generated_at": datetime.now(CST).isoformat(),
    }
    dev = _devil_advocate(views, consensus, votes) if devil_advocate else None
    if dev is not None:
        res["devil_advocate"] = dev
        if dev["verdict"] == "分歧":
            res["consensus"] = "分歧"
            res["confidence"] = round(min(res["confidence"], 0.5), 2)
        _write_devil_advocate_audit(res, dev)
    out = OUT_DIR / f"multi_agent_{res['date']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return res


def debate(date: str | None = None) -> list[dict]:
    """分歧时生成辩论记录: 每个分歧点输出 正方逻辑/反方逻辑（evidence 拼装，不用 LLM）。"""
    res = arbitrate(date)
    out = []
    for d in res.get("disagreement", []):
        a, b = d["side_a"], d["side_b"]
        out.append({
            "between": d["between"],
            "issue": d["issue"],
            "implication": d["implication"],
            "正方逻辑": {"side": a["name"], "view": a["view"],
                       "logic": "；".join(a.get("evidence", []))},
            "反方逻辑": {"side": b["name"], "view": b["view"],
                       "logic": "；".join(b.get("evidence", []))},
        })
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="多智能体专家委员会（轻量版）")
    ap.add_argument("--date", default=None)
    ap.add_argument("--debate", action="store_true", help="输出辩论记录")
    args = ap.parse_args()
    if args.debate:
        print(json.dumps(debate(args.date), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(arbitrate(args.date), ensure_ascii=False, indent=2, default=str))
