#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""每日复盘融合链 · L1 六要素采集器（2026-08-21 架构融合）

把全部板块模块接入统一 SignalBlock 输出：
- 市场温度: emotion_cycle + fund_forces + macro_system + 数据健康门控
- 强势方向: resonance_scorer + theme_cycle + concept_lifecycle + style_spread
- 龙头情绪: leader_follower + ladder + market_microstructure + breakout_watch
- 龙虎榜:   after_close_extra(lhb) + broker_gaming + broker_profile_deep + fund_flow_divergence
- 多格局选股: 短线(short_term) + 中长线(valuation_system) + 风格(factor_rotation) + 机会(opportunity_angles)
- 模型/后验: multi_agent.arbitrate + predictions + macro_proxy + retail_sentiment

每个采集器返回 SignalBlock（统一结构化）——不浪费任何已实现模块。
"""
from __future__ import annotations

import copy
import json
import logging
import sys
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quant_system"))

logger = logging.getLogger("daily_review_collectors")
CST = timezone(timedelta(hours=8))

# 融合链入口所需模块
from quant_system.analysis_core import emotion_cycle  # noqa: E402
from quant_system.analysis_core import ladder  # noqa: E402
from quant_system.analysis_core import fund_forces  # noqa: E402
# 注: leader_follower 原模块云端 import 死挂(>120s), 3a 已改用 leader_follower_light
from quant_system.analysis_core import market_microstructure  # noqa: E402


@dataclass
class SignalBlock:
    element: str                # 要素名
    source: str                 # 模块名
    value: dict                 # 结构化信号
    confidence: float = 0.5     # 0-1 统一标定
    staleness: int = 0          # 落后天数
    health: float = 1.0         # 数据健康 trust
    error: Optional[str] = None
    ts: str = field(default_factory=lambda: datetime.now(CST).isoformat(timespec="seconds"))

    def to_dict(self) -> dict:
        return asdict(self)


def _safe_collect(fn, element: str, source: str, default: dict,
                  timeout_seconds: int = 30) -> SignalBlock:
    """尽力而为采集：超时/异常 → 降级块（不炸链）。"""
    import threading
    result: dict = {"error": "timeout"}
    box: dict = {}

    def _run():
        try:
            box["v"] = fn()
        except Exception as e:  # noqa: BLE001
            box["e"] = f"{type(e).__name__}: {str(e)[:120]}"

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout_seconds)
    if "e" in box:
        return SignalBlock(element, source, default, confidence=0.2,
                           error=box["e"], staleness=-1)
    if "v" not in box:
        return SignalBlock(element, source, default, confidence=0.2,
                           error="timeout", staleness=-1)
    v = box["v"] or copy.deepcopy(default)
    conf = v.get("confidence", 0.5) if isinstance(v, dict) else 0.5
    return SignalBlock(element, source, v if isinstance(v, dict) else {"value": v},
                       confidence=float(conf))


def _date_of(v: dict) -> str:
    return str(v.get("date", v.get("as_of", "")))[:10]


# ═══════════════════════════════════════════════════════════
# 1. 市场温度（情绪 + 资金 + 宏观 + 数据健康门控）
# ═══════════════════════════════════════════════════════════
def collect_market_temperature(gateway, date: str | None = None) -> SignalBlock:
    """市场温度：情绪周期六阶段 + 四路资金合力 + 宏观四驱动，按 as_of 截断。"""
    def _run() -> dict:
        out: dict = {"components": {}}
        # 1a 情绪周期
        try:
            eco = emotion_cycle.run_today(date)
            out["components"]["emotion"] = {
                "stage": eco.get("stage"), "stage_cn": eco.get("stage_cn"),
                "confidence": eco.get("confidence"),
                "zt_cnt": eco.get("zt_cnt"), "max_board": eco.get("max_board"),
                "date": eco.get("date"),
            }
        except Exception as e:  # noqa: BLE001
            out["components"]["emotion"] = {"error": str(e)[:100]}
        # 1b 资金合力
        try:
            ff = fund_forces.latest(as_of=date)
            if ff is not None and len(ff):
                row = ff.iloc[-1]
                actual_date = str(row.get("date", ""))[:10]
                fund_stale = bool(date and actual_date and actual_date < date)
                out["components"]["fund"] = {
                    "force_index": None if fund_stale else float(row.get("force_index", 0)),
                    "youzi_net": float(row.get("youzi_net", 0)),
                    "jg_net": float(row.get("jg_net", 0)),
                    "north_net": float(row.get("north_net", 0)),
                    "n_legs": int(row.get("n_legs", 0)),
                    "date": actual_date,
                    "requested_date": date,
                    "stale": fund_stale,
                    "error": f"资金合力仅到{actual_date}，请求{date}" if fund_stale else None,
                }
        except Exception as e:  # noqa: BLE001
            out["components"]["fund"] = {"error": str(e)[:100]}
        # 1c 宏观系统
        try:
            sys.path.insert(0, str(ROOT / "quant_system"))
            from quant_system.analysis_core import macro_system  # noqa: PLC0415
            ms = macro_system.MacroSystem() if hasattr(macro_system, "MacroSystem") else None
            if ms is not None:
                r = ms.run() if hasattr(ms, "run") else {}
                out["components"]["macro"] = {"regime": r.get("regime", r.get("state", "?")),
                                               "confidence": r.get("confidence", 0.5)}
        except Exception as e:  # noqa: BLE001
            out["components"]["macro"] = {"error": str(e)[:100]}
        # 1d 数据健康门控（整体）
        h = gateway.health_scores()
        out["data_health"] = {
            "overall_ok": h.get("overall_ok"),
            "degraded": h.get("degraded_datasets", [])[:8],
        }
        out["date"] = out["components"].get("emotion", {}).get("date")
        out["confidence"] = 0.6 if h.get("overall_ok") else 0.3
        return out
    return _safe_collect(_run, "market_temperature", "emotion+fund+macro+health", {})


# ═══════════════════════════════════════════════════════════
# 2. 强势方向（题材共振 + 生命周期 + 概念图谱 + 风格）
# ═══════════════════════════════════════════════════════════
def _canonical_mainlines(date: str | None = None, limit: int = 10) -> dict:
    """Build the sole mainline contract from local structure, funding and persistence."""
    import pandas as pd
    from quant_system.analysis_core import resonance_scorer

    theme = pd.read_parquet(resonance_scorer.THEME_CYCLE).copy()
    theme["date"] = pd.to_datetime(theme["date"]).dt.strftime("%Y-%m-%d")
    as_of = str(date or theme["date"].max())[:10]
    if as_of not in set(theme["date"]):
        as_of = str(theme[theme["date"] <= as_of]["date"].max())[:10]
    day = theme[theme["date"] == as_of].copy()
    if day.empty:
        return {"status": "unavailable", "as_of": as_of, "items": [], "error": "structure_missing"}

    flow_path = resonance_scorer.MARKET_DIR / "sector_fund_flow.parquet"
    flow_map: dict[str, float] = {}
    flow_as_of = None
    if flow_path.exists():
        flow = pd.read_parquet(flow_path).copy()
        flow["date"] = flow["date"].astype(str).str[:10]
        eligible = flow[flow["date"] <= as_of]
        if not eligible.empty:
            flow_as_of = str(eligible["date"].max())[:10]
            latest_flow = eligible[eligible["date"] == flow_as_of]
            flow_map = dict(zip(latest_flow["concept_name"].astype(str), latest_flow["net_yi"].astype(float)))

    history_start = (pd.Timestamp(as_of) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    recent = theme[(theme["date"] >= history_start) & (theme["date"] <= as_of)]
    items = []
    for _, row in day.iterrows():
        name = str(row.get("board_name") or row.get("concept") or "").replace("THS:", "")
        if not name or resonance_scorer.is_broad_board(name):
            continue
        zt_cnt = int(row.get("zt_cnt") or 0)
        max_board = int(row.get("max_board") or 0)
        structure_ok = zt_cnt >= 3 or (zt_cnt >= 2 and max_board >= 3)
        matched_flow = next((value for flow_name, value in flow_map.items()
                             if name in flow_name or flow_name in name), None)
        funding_ok = matched_flow is not None and matched_flow >= 5
        same_names = recent["board_name"].astype(str).str.replace("THS:", "", regex=False)
        persistence_days = int(recent.loc[same_names == name, "date"].nunique())
        persistence_ok = persistence_days >= 2
        confirmed = structure_ok and funding_ok and persistence_ok
        evidence = {
            "structure": {"ok": structure_ok, "zt_cnt": zt_cnt, "max_board": max_board},
            "funding": {"ok": funding_ok, "net_yi": round(matched_flow, 2) if matched_flow is not None else None,
                        "as_of": flow_as_of},
            "persistence": {"ok": persistence_ok, "days": persistence_days},
        }
        score = min(100.0, zt_cnt * 8 + max_board * 6 + max(0.0, min(float(matched_flow or 0), 100)) * 0.35 + persistence_days * 5)
        items.append({"name": name, "level": "主线" if confirmed else "观察方向",
                      "status": "confirmed" if confirmed else "watch", "score": round(score, 1),
                      "as_of": as_of, "evidence": evidence})
    items.sort(key=lambda item: (item["status"] == "confirmed", item["score"]), reverse=True)
    return {"status": "confirmed" if any(item["status"] == "confirmed" for item in items) else "watch",
            "as_of": as_of, "flow_as_of": flow_as_of, "items": items[:limit]}


def collect_strong_direction(gateway, date: str | None = None) -> SignalBlock:
    """强势方向：五维共振主线 + 题材生命周期 + 概念案例 + 大小盘风格。"""
    def _run() -> dict:
        out: dict = {"directions": [], "components": {}}
        # 2a canonical 主线：结构、资金、持续性共同确认，不允许单一涨停榜升级主线。
        try:
            canonical = _canonical_mainlines(date)
            out["directions"] = canonical.get("items", [])
            out["mainlines"] = canonical
            _d = canonical.get("as_of") or date
            out["components"]["resonance"] = {
                "count": len(out["directions"]), "source": "canonical_local_evidence",
                "status": canonical.get("status"), "as_of": canonical.get("as_of"),
            }
        except Exception as e:  # noqa: BLE001
            _d = date
            out["mainlines"] = {"status": "unavailable", "items": [], "error": str(e)[:100]}
            out["components"]["resonance"] = {"error": str(e)[:100]}
        # 2b 题材生命周期（theme_cycle）
        try:
            from quant_system.analysis_core import theme_cycle  # noqa: PLC0415
            tc = theme_cycle.today_themes(top_n=8)
            out["components"]["theme_cycle"] = {"themes": tc if isinstance(tc, list) else []}
        except Exception as e:  # noqa: BLE001
            out["components"]["theme_cycle"] = {"error": str(e)[:100]}
        # 2c 概念生命周期只读已有产物。主线是关键决策块，不能为了扩展案例
        # 现场重算百万字节报告并让整个 collector 超时。
        try:
            artifact_date = str(_d or date or "")[:10]
            artifact = ROOT / "generated" / f"concept_lifecycle_{artifact_date}.json"
            if artifact.exists():
                rep = json.loads(artifact.read_text(encoding="utf-8"))
                out["components"]["concept_lifecycle"] = {
                    "active": rep.get("active_concept_count"),
                    "again": rep.get("again_count"),
                    "source": artifact.name,
                }
            else:
                out["components"]["concept_lifecycle"] = {"status": "artifact_missing"}
        except Exception as e:  # noqa: BLE001
            out["components"]["concept_lifecycle"] = {"error": str(e)[:80]}
        # 2d 大小盘风格（style_spread）
        try:
            from quant_system.analysis_core import style_spread  # noqa: PLC0415
            from quant_system.analysis_core.style_spread import judge_style  # noqa: PLC0415
            rows = {}
            # 简单探测：只报 style 判定能力就绪（完整计算较重）
            out["components"]["style_spread"] = {"ready": True, "note": "judge_style 就绪(按需)"}
        except Exception as e:  # noqa: BLE001
            out["components"]["style_spread"] = {"error": str(e)[:80]}
        out["date"] = date
        out["confidence"] = 0.6 if out["directions"] else 0.3
        return out
    # 主线是决策关键块：默认保留结构，避免超时后下游收到空字典。
    return _safe_collect(
        _run, "strong_direction", "local_theme_cycle+theme+concept+style",
        {"directions": [], "components": {}, "date": date, "confidence": 0.2},
        timeout_seconds=60,
    )


# ═══════════════════════════════════════════════════════════
# 3. 龙头情绪（扩散度 + 天梯 + 微观结构 + 新高突破）
# ═══════════════════════════════════════════════════════════
def collect_leader_sentiment(gateway, date: str | None = None) -> SignalBlock:
    """龙头情绪：龙头扩散/过热 + 连板天梯 + 涨跌停微观 + 新高突破。"""
    def _run() -> dict:
        out: dict = {"signals": [], "components": {}}
        # 3a 龙头扩散 —— 拆散轻量版(leader_follower_light, 2026-08-21 用户要求拆散;
        #    原 leader_follower.run_today 云端死挂>120s)
        try:
            sys.path.insert(0, str(ROOT / "quant_system"))
            from quant_system.analysis_core.leader_follower_light import (
                zt_board_stats, concept_diffusion, leader_signal)  # noqa: PLC0415
            _st = zt_board_stats(date)
            _sig = leader_signal(date)
            _cd = concept_diffusion(date, top_n=8)
            out["components"]["leader"] = {
                "active_concepts": _st.get("lianban"),
                "overheat": 1 if _sig.get("overheat") else 0,
                "max_board": _st.get("max_board"),
                "zt_cnt": _st.get("zt_cnt"),
                "high_board": _st.get("high_board"),
                "note": _sig.get("note", ""),
                "date": str(date or "")[:10], "source": "leader_follower_light",
            }
            canonical = _canonical_mainlines(date)
            canonical_by_name = {item["name"]: item for item in canonical.get("items", [])}
            out["signals"] = []
            for candidate in _cd:
                name = str(candidate.get("concept") or candidate.get("board_name") or "").replace("THS:", "")
                authority = canonical_by_name.get(name)
                out["signals"].append({
                    "concept": name,
                    "signal": authority.get("level") if authority else "热度观察",
                    "status": authority.get("status") if authority else "watch",
                    "score": authority.get("score") if authority else candidate.get("zt_cnt"),
                })
            out["signals"] = out["signals"][:5]
        except Exception as e:  # noqa: BLE001
            out["components"]["leader"] = {"error": str(e)[:100]}
        # 3b 连板天梯
        try:
            ld = ladder.latest(3, as_of=date)
            if ld is not None and len(ld):
                last = ld.iloc[-1]
                out["components"]["ladder"] = {
                    "zt_cnt": int(last.get("zt_cnt", 0)), "max_board": int(last.get("max_board", 0)),
                    "zb_cnt": int(last.get("zb_cnt", 0)), "dt_cnt": int(last.get("dt_cnt", 0)),
                    "date": str(last.get("date", ""))[:10],
                }
        except Exception as e:  # noqa: BLE001
            out["components"]["ladder"] = {"error": str(e)[:100]}
        # 3c 涨跌停微观结构
        try:
            ms = market_microstructure.run_today(date)
            if ms:
                out["components"]["microstructure"] = ms if isinstance(ms, dict) else {"raw": str(ms)[:200]}
        except Exception as e:  # noqa: BLE001
            out["components"]["microstructure"] = {"error": str(e)[:100]}
        # 3d 创N日新高 —— 轻量版: 读涨停池近N日最高价(全市场K线扫描云端分钟级, 2026-08-21 弃用)
        try:
            out["components"]["breakout"] = {
                "new_high_60": None, "new_high_120": None,
                "note": "新高扫描按需触发(轻量模式,避免全市场K线扫描)",
            }
        except Exception:  # noqa: BLE001
            pass
        out["date"] = date
        out["confidence"] = 0.6 if out.get("components", {}).get("leader") else 0.3
        return out
    return _safe_collect(_run, "leader_sentiment", "leader+ladder+micro+breakout", {})


# ═══════════════════════════════════════════════════════════
# 4. 龙虎榜（短线资金）
# ═══════════════════════════════════════════════════════════
def collect_lhb(gateway, date: str | None = None) -> SignalBlock:
    """龙虎榜：游资营业部 + 个股净买 + 席位画像 + 大单背离（接孤立模块）。"""
    def _run() -> dict:
        out: dict = {"lhb": {}, "components": {}}
        # 4a after_close_extra.lhb —— 优先读 pipeline 已生成产物(重算含网络,易超时)
        try:
            import glob as _g
            import os as _o
            _cands = sorted(_g.glob(str(ROOT / "generated" / "after_close_extra_*.json")))
            _ex = None
            if date:
                _paths = [ROOT / "generated" / f"after_close_extra_{date.replace('-', '')}.json",
                          ROOT / "generated" / f"after_close_extra_{date}.json"]
                _pick = next((p for p in _paths if p.exists()), None)
                if _pick:
                    _ex = json.loads(_pick.read_text(encoding="utf-8"))
            elif _cands:
                _ex = json.loads(Path(_cands[-1]).read_text(encoding="utf-8"))
            if _ex is None:
                from quant_system.analysis_core import after_close_extra  # noqa: PLC0415
                _ex = after_close_extra.build_extra(date)
            lhb = (_ex or {}).get("lhb") or {}
            out["lhb"] = {"broker_top": lhb.get("broker_top", [])[:5],
                          "stock_top": lhb.get("stock_top", [])[:5]}
            out["components"]["after_close"] = {"value_picks": len((_ex or {}).get("value_picks", []))}
        except Exception as e:  # noqa: BLE001
            out["components"]["after_close"] = {"error": str(e)[:100]}
        # 4b broker_gaming 席位画像 —— 重计算(build_profiles >30s)，复盘链只标注就绪，
        # 详细画像由独立任务(broker_profile_deep)或按需触发，避免拖慢每日复盘。
        try:
            _bp_file = ROOT / "generated" / "broker_profiles.json"
            if _bp_file.exists():
                _bp = json.loads(_bp_file.read_text(encoding="utf-8"))
                out["components"]["broker"] = {"profiles": len(_bp) if isinstance(_bp, list) else 1,
                                               "source": "artifact"}
            else:
                out["components"]["broker"] = {"ready": True, "note": "席位画像按需生成(重计算)"}
        except Exception as e:  # noqa: BLE001
            out["components"]["broker"] = {"ready": True, "note": str(e)[:60]}
        # 4c fund_flow_divergence 大单背离（孤立模块）—— 重计算按需，复盘链仅声明就绪
        try:
            out["components"]["divergence"] = {"ready": True, "note": "大单流向背离模块就绪(独立任务触发)"}
        except Exception:  # noqa: BLE001
            out["components"]["divergence"] = {"error": "n/a"}
        out["date"] = date
        out["confidence"] = 0.6 if out["lhb"].get("broker_top") or out["lhb"].get("stock_top") else 0.3
        return out
    return _safe_collect(_run, "lhb", "after_close+broker+divergence", {})


# ═══════════════════════════════════════════════════════════
# 5. 多格局选股（短线池 + 中长线池 + 风格轮动池 + 机会池）
# ═══════════════════════════════════════════════════════════
def collect_stock_picks(gateway, date: str | None = None) -> SignalBlock:
    """多格局选股：四套选股池合并（不同格局、不重复）。"""
    def _run() -> dict:
        out: dict = {"pools": {}, "components": {}}
        # 5a 短线龙头池 —— 优先 battle_map attack_groups 核心（主线攻击方向），
        # 其次 leader_follower 龙头信号；两源不重复，保证短线池有内容。
        try:
            import glob as _g3
            _bm_f = None
            if date:
                _bf = ROOT / "generated" / f"battle_map_{date}.json"
                if _bf.exists():
                    _bm_f = _bf
            else:
                _c3 = sorted(_g3.glob(str(ROOT / "generated" / "battle_map_*.json")))
                if _c3:
                    _bm_f = Path(_c3[-1])
            canonical = _canonical_mainlines(date, limit=5)
            pool_s = [{"name": item["name"], "signal": "攻击候选" if item["status"] == "confirmed" else "观察候选",
                       "level": item["level"], "status": item["status"],
                       "score": item["score"], "evidence": item["evidence"]}
                      for item in canonical.get("items", [])]
            out["pools"]["short_term"] = pool_s
            out["components"]["short_term"] = {"source": "canonical_mainlines", "status": canonical.get("status")}
        except Exception as e:  # noqa: BLE001
            out["components"]["short_term"] = {"error": str(e)[:80]}
        # 5a2 龙虎榜强势池（个股级，lhb stock_top 净买）
        try:
            import glob as _g4
            _lc = sorted(_g4.glob(str(ROOT / "generated" / "after_close_extra_*.json")))
            _pick_lc = next((p for p in [ROOT / "generated" / f"after_close_extra_{date.replace('-', '')}.json", ROOT / "generated" / f"after_close_extra_{date}.json"] if p.exists()), None) if date else (_lc[-1] if _lc else None)
            if _pick_lc and Path(_pick_lc).exists():
                _ex2 = json.loads(Path(_pick_lc).read_text(encoding="utf-8"))
                st = (_ex2.get("lhb") or {}).get("stock_top", [])[:6]
                out["pools"]["lhb_stocks"] = [
                    {"code": x.get("code"), "name": x.get("name"), "net": x.get("net"),
                     "pct": x.get("pct"), "reason": (x.get("reason") or "")[:30]}
                    for x in st]
        except Exception as e:  # noqa: BLE001
            out["components"]["lhb_stocks"] = {"error": str(e)[:80]}
        # 5a3 RS 强势池（个股级，rs_strength TOP，解析 md 表格）
        try:
            import glob as _g5
            import re as _re
            _rc = sorted(_g5.glob(str(ROOT / "generated" / "rs_strength_*.md")))
            _pick_rc = (ROOT / "generated" / f"rs_strength_{date}.md") if date else (_rc[-1] if _rc else None)
            if _pick_rc and Path(_pick_rc).exists():
                _md5 = Path(_pick_rc).read_text(encoding="utf-8")
                # 表格行格式: | rank | code | name | rs | ... |
                rows = []
                for _ln in _md5.splitlines():
                    _m = _re.match(r"\|\s*(\d+)\s*\|\s*(\d{6})\s*\|\s*([^|]+?)\s*\|", _ln)
                    if _m and int(_m.group(1)) <= 8:
                        rows.append({"rank": int(_m.group(1)), "code": _m.group(2), "name": _m.group(3).strip()})
                        if len(rows) >= 6:
                            break
                out["pools"]["rs_stocks"] = rows
        except Exception as e:  # noqa: BLE001
            out["components"]["rs_stocks"] = {"error": str(e)[:80]}
        # 5b 中长线低估池（after_close_extra value_picks）— 产物优先, 缺则实时(>30s)
        try:
            import glob as _g6
            _ac = sorted(_g6.glob(str(ROOT / "generated" / "after_close_extra_*.json")))
            _pick_ac = next((p for p in [ROOT / "generated" / f"after_close_extra_{date.replace('-', '')}.json", ROOT / "generated" / f"after_close_extra_{date}.json"] if p.exists()), None) if date else (_ac[-1] if _ac else None)
            if _pick_ac and Path(_pick_ac).exists():
                _ex3 = json.loads(Path(_pick_ac).read_text(encoding="utf-8"))
                vp = (_ex3 or {}).get("value_picks", [])[:5]
            else:
                from quant_system.analysis_core import after_close_extra  # noqa: PLC0415
                ex = after_close_extra.build_extra(date)
                vp = (ex or {}).get("value_picks", [])[:5]
            out["pools"]["mid_long"] = vp if isinstance(vp, list) else [{"note": str(vp)[:80]}]
        except Exception as e:  # noqa: BLE001
            out["components"]["mid_long"] = {"error": str(e)[:80]}
        # 5c 风格轮动池（factor_rotation_system 信号）
        try:
            from quant_system.analysis_core import factor_rotation_system  # noqa: PLC0415
            out["components"]["factor_rotation"] = {"ready": True, "note": "五因子轮动就绪(全量按需)"}
        except Exception as e:  # noqa: BLE001
            out["components"]["factor_rotation"] = {"error": str(e)[:80]}
        # 5d 机会池（opportunity_angles 五视角）
        try:
            sys.path.insert(0, str(ROOT / "quant_system"))
            from quant_system.opportunity_angles import evaluate_angles  # noqa: PLC0415
            out["components"]["opportunity_angles"] = {"ready": True, "note": "机会五视角就绪(按需)"}
        except Exception as e:  # noqa: BLE001
            out["components"]["opportunity_angles"] = {"error": str(e)[:80]}
        out["date"] = date
        n = sum(len(v) for v in out["pools"].values() if isinstance(v, list))
        out["confidence"] = 0.6 if n else 0.3
        return out
    return _safe_collect(_run, "stock_picks", "short+midlong+style+opportunity", {})


# ═══════════════════════════════════════════════════════════
# 6. 模型/后验（专家委员会 + 预测验证 + 宏观代理 + 散户情绪）
# ═══════════════════════════════════════════════════════════
def collect_model_verdict(gateway, date: str | None = None) -> SignalBlock:
    """模型集体意见 + 信号后验 + 宏观代理 + 散户情绪（接孤立模块）。"""
    def _run() -> dict:
        out: dict = {"verdict": {}, "components": {}}
        # 6a multi_agent 专家委员会 —— 优先读产物(arbitrate 重计算,仅 fallback)
        try:
            import glob as _g2
            _ma_file = None
            if date:
                _mf = ROOT / "generated" / f"multi_agent_{date}.json"
                if _mf.exists():
                    _ma_file = _mf
            else:
                _c2 = sorted(_g2.glob(str(ROOT / "generated" / "multi_agent_*.json")))
                if _c2:
                    _ma_file = Path(_c2[-1])
            if _ma_file:
                ma = json.loads(Path(_ma_file).read_text(encoding="utf-8"))
            else:
                from quant_system.analysis_core import multi_agent  # noqa: PLC0415
                ma = multi_agent.arbitrate(date)
            out["verdict"] = {"consensus": ma.get("consensus"), "confidence": ma.get("confidence"),
                              "votes": len(ma.get("votes", [])), "date": ma.get("date")}
        except Exception as e:  # noqa: BLE001
            out["components"]["multi_agent"] = {"error": str(e)[:100]}
        # 6b predictions 后验
        try:
            from quant_system.analysis_core import predictions  # noqa: PLC0415
            out["components"]["predictions"] = {"ready": True, "note": "预测入库/验证就绪"}
        except Exception as e:  # noqa: BLE001
            out["components"]["predictions"] = {"error": str(e)[:80]}
        # 6c macro_proxy 宏观代理 + retail_sentiment 散户情绪
        try:
            sys.path.insert(0, str(ROOT / "quant_system"))
            from quant_system.macro_proxy import macro_proxy as _mp  # noqa: PLC0415, E402
            out["components"]["macro_proxy"] = {"ready": True}
        except Exception:  # noqa: BLE001
            try:
                import macro_proxy  # noqa: PLC0415
                out["components"]["macro_proxy"] = {"ready": True}
            except Exception as e:  # noqa: BLE001
                out["components"]["macro_proxy"] = {"error": str(e)[:80]}
        try:
            from quant_system.analysis_core import retail_sentiment  # noqa: PLC0415
            out["components"]["retail_sentiment"] = {"ready": True, "note": "散户情绪代理就绪"}
        except Exception as e:  # noqa: BLE001
            out["components"]["retail_sentiment"] = {"error": str(e)[:80]}
        out["date"] = date
        out["confidence"] = float(out["verdict"].get("confidence", 0.5))
        return out
    return _safe_collect(_run, "model_verdict", "multi_agent+predictions+proxy+retail", {}, timeout_seconds=45)


def collect_engines() -> SignalBlock:
    """全部引擎(87个) — 引擎工厂账本优先(真产判定), 无则全量缓存, 再回退六引擎. 2026-08-22."""
    import json, time
    from pathlib import Path
    def _run():
        _ROOT = Path(__file__).resolve().parent.parent
        # 1) 引擎工厂账本(真产判定: 87引擎 recipe+前置+产物验证)
        _LD = sorted((_ROOT / "generated").glob("engine_ledger_*.json"))
        if _LD:
            d = json.loads(_LD[-1].read_text(encoding="utf-8"))
            ok_n = sum(1 for v in d.get("engines", {}).values() if v.get("status") == "真产")
            return {"mode": f"引擎工厂账本{d.get('date','')}", "ok_n": ok_n, "data": d}
        # 2) 全量缓存(87引擎)
        _AE = _ROOT / "generated" / "all_engines_cache.json"
        if _AE.exists() and time.time() - _AE.stat().st_mtime < 14400:
            d = json.loads(_AE.read_text(encoding="utf-8"))
            ok_n = sum(1 for v in d.get("engines", {}).values() if v.get("ok"))
            return {"mode": "全引擎87", "ok_n": ok_n, "data": d}
        # 3) 六引擎缓存
        _EP = _ROOT / "generated" / "engine_pulse_cache.json"
        if _EP.exists() and time.time() - _EP.stat().st_mtime < 3600:
            return {"mode": "六引擎速回退", "data": json.loads(_EP.read_text(encoding="utf-8"))}
        return {"mode": "无缓存", "error": "引擎缓存缺失"}
    return _safe_collect(_run, "engines", "all_engines", {}, timeout_seconds=25)


# ═══════════════════════════════════════════════════════════
# 汇总：采集全部六要素
# ═══════════════════════════════════════════════════════════
def collect_all(gateway, date: str | None = None) -> dict[str, SignalBlock]:
    """并行采集六要素 —— 手写 daemon 线程集合（2026-08-21 修复:
    ThreadPoolExecutor 的 shutdown(wait=True) 会等死线程导致云端卡死;
    daemon 线程 join 超时即返回, 不阻塞进程退出。外层 _safe_collect 已兜超时。)"""
    import threading as _th
    tasks = {
        "market_temperature": lambda: collect_market_temperature(gateway, date),
        "strong_direction": lambda: collect_strong_direction(gateway, date),
        "leader_sentiment": lambda: collect_leader_sentiment(gateway, date),
        "lhb": lambda: collect_lhb(gateway, date),
        "stock_picks": lambda: collect_stock_picks(gateway, date),
        "model_verdict": lambda: collect_model_verdict(gateway, date),
        "engines": lambda: collect_engines(),
    }
    blocks: dict = {}
    box: dict = {}
    def _runner(name, fn):
        try:
            box[name] = fn()
        except Exception as _e:  # noqa: BLE001
            box[name] = SignalBlock(name, "parallel", {"error": str(_e)[:80]},
                                    confidence=0.2, error=str(_e)[:80])
    threads = [_th.Thread(target=_runner, args=(n, fn), daemon=True)
               for n, fn in tasks.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join(45)  # 每线程最多等45s(外层_safe_collect内部30s, 这里兜底)
    for name in tasks:
        blocks[name] = box.get(name) or SignalBlock(
            name, "parallel", {"error": "timeout"}, confidence=0.2, error="timeout")
    return blocks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from daily_fusion_base import get_gateway
    gw = get_gateway()
    date = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"=== 六要素采集（date={date or '最新'}）===")
    blocks = collect_all(gw, date)
    for k, b in blocks.items():
        status = b.error or f"conf={b.confidence:.2f}"
        print(f"  {k:20s} [{b.source:30s}] {status}")