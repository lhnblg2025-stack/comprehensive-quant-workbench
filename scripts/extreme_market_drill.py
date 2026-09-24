#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""极端行情注入压力测试（2026-08-22 稳定化专项 —— "压力测试极端行情")

把正常复盘 blocks 变异为极端行情场景，逐场景跑 make_decision 并断言安全约束:
  - 任何场景不抛异常、输出结构完整(含"定调")
  - 千股跌停/恐慌/熔断等极端场景: posture 不得"积极进攻", 仓位上限≤20%
  - veto hard → 强制防守观望
  - 数据全缺失 → 保守 ≥ 非激进(不 fail-open)
  - 畸形/恶毒输入(None/字符串/NaN/嵌套错误) → 不炸链

dry-run 安全: 全部在内存构造, 不碰数据仓库/不写源文件, 只输出报告。

用法:
  python3 scripts/extreme_market_drill.py            # 跑全部场景
  python3 scripts/extreme_market_drill.py --scene 千股跌停   # 单场景
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

CST = timezone(timedelta(hours=8))
OUT = ROOT / "generated" / "extreme_drill"

_MAX_POS_CRASH = 0.20      # 极端行情允许的最高仓位上限
_POSTURE_BANNED = ("积极进攻",)  # 极端行情严禁的定调


def _pos_upper(position: str) -> float:
    """'10-20%' → 0.20；解析失败返回 99 视为不受控。"""
    try:
        return float(position.split("-")[1].replace("%", "").strip()) / 100.0
    except Exception:  # noqa: BLE001
        return 99.0


# ────────────────────────────────────────────────────────────────
# 场景构造（基于 review json 真实结构变异的纯 dict, 无需 SignalBlock）
# ────────────────────────────────────────────────────────────────
def _base_blocks() -> dict:
    """2026-08-21 真实基线的精简结构。"""
    return {
        "market_temperature": {
            "date": "2026-08-21", "confidence": 0.35,
            "components": {
                "emotion": {"stage": "repair", "stage_cn": "修复", "confidence": 0.35,
                            "zt_cnt": 74, "max_board": 3},
                "fund": {"force_index": 25.0, "youzi_net": 3.98e9, "jg_net": 0.0,
                         "north_net": 0.0, "n_legs": 1},
            },
        },
        "leader_sentiment": {
            "date": "2026-08-21", "confidence": 0.5,
            "components": {"ladder": {"zt_cnt": 74, "max_board": 3, "zb_cnt": 11,
                                      "dt_cnt": 17, "zb_rate": 0.13},
                           "leader": {"overheat": 6}},
        },
        "strong_direction": {
            "date": "2026-08-21", "confidence": 0.5,
            "directions": [{"name": "半导体", "level": "主线", "score": 2},
                           {"name": "机器人", "level": "次主线", "score": 1}],
        },
        "stock_picks": {"date": "2026-08-21", "confidence": 0.5, "pools": {
            "short_term": [{"concept": "半导体", "signal": "转强"}],
            "lhb_stocks": [], "rs_stocks": [], "mid_long": []}},
        "overseas": {"quotes": [{"label": "纳指", "chg_pct": 0.8},
                                {"label": "伦敦金", "chg_pct": 0.5}], "count": 2},
        "factor_signal": {"date": "2026-08-21", "groups": {
            "技术动量": {"score": 0.332, "n": 6}, "质量": {"score": 0.397, "n": 4},
            "量价": {"score": 0.398, "n": 4}},
            "top_factors": ["rev_20"], "weak_factors": []},
        "macro_veto": {"date": "2026-08-21", "level": "soft", "position_coef": 0.6,
                       "pe_percentile": 0.931, "regime": "震荡市+低波"},
        "freshness": {"total": 18, "fresh": 13, "score": 72, "scan": {
            "kline": {"lag": 1, "level": "fresh"}, "market": {"lag": 1, "level": "fresh"},
            "zt_history": {"lag": 0, "level": "fresh"}, "lhb_hist": {"lag": 0, "level": "fresh"},
            "financial": {"lag": 1, "level": "fresh"}, "valuation": {"lag": 1, "level": "fresh"},
            "macro": {"lag": 2, "level": "fresh"}, "events": {"lag": 1, "level": "fresh"},
            "industry": {"lag": 12, "level": "expired"}}},
        "engine_fusion": {"date": "2026-08-21", "score": 43.9, "consensus": "中性",
                          "ok_n": 3, "trust": 0.6,
                          "dimensions": {"risk": {"score": 44.4, "ok": True},
                                         "meta": {"score": 47.0, "ok": True},
                                         "card": {"score": 38.0, "ok": True}}},
    }


def _mk(blocks: dict, **overrides) -> dict:
    """浅复制并套用覆盖（deep-ish：只覆盖顶层或指定路径键）。"""
    import copy
    b = copy.deepcopy(blocks)
    for k, v in overrides.items():
        b[k] = v
    return b


def _emo(stage_cn: str, zt: int, mb: int) -> dict:
    return {"date": "2026-08-21", "confidence": 0.5,
            "components": {"emotion": {"stage": stage_cn, "stage_cn": stage_cn,
                                       "confidence": 0.5, "zt_cnt": zt, "max_board": mb},
                           "fund": {"force_index": 25.0, "youzi_net": 0.0,
                                    "jg_net": 0.0, "north_net": 0.0}}}


def _ladder(zt: int, mb: int, zb: int, dt: int, zb_rate: float) -> dict:
    return {"date": "2026-08-21", "confidence": 0.5,
            "components": {"ladder": {"zt_cnt": zt, "max_board": mb, "zb_cnt": zb,
                                      "dt_cnt": dt, "zb_rate": zb_rate},
                           "leader": {"overheat": 0}}}


def _fund(fi: float, youzi: float, north: float) -> dict:
    return {"date": "2026-08-21", "confidence": 0.5,
            "components": {"emotion": {"stage": "repair", "stage_cn": "修复",
                                       "confidence": 0.5, "zt_cnt": 20, "max_board": 2},
                           "fund": {"force_index": fi, "youzi_net": youzi,
                                    "jg_net": 0.0, "north_net": north}}}


def _overseas(items: list) -> dict:
    return {"quotes": items, "count": len(items)}


def _directions(items: list) -> dict:
    return {"date": "2026-08-21", "confidence": 0.5, "directions": items}


def _veto(level: str, coef: float) -> dict:
    return {"date": "2026-08-21", "level": level, "position_coef": coef,
            "pe_percentile": 0.95, "regime": "极端"}


def _factor(score: float = 0.32) -> dict:
    return {"date": "2026-08-21", "groups": {
        "技术动量": {"score": score, "n": 6}, "质量": {"score": score, "n": 4},
        "量价": {"score": score, "n": 4}}, "top_factors": [], "weak_factors": ["全部"]}


def _fresh_crash() -> dict:
    return {"total": 18, "score": 22, "scan": {
        "kline": {"lag": 6, "level": "expired"}, "market": {"lag": 6, "level": "expired"},
        "zt_history": {"lag": 6, "level": "expired"}, "lhb_hist": {"lag": 6, "level": "expired"},
        "financial": {"lag": 9, "level": "expired"}, "valuation": {"lag": 9, "level": "expired"},
        "macro": {"lag": 8, "level": "expired"}, "events": {"lag": 6, "level": "expired"},
        "industry": {"lag": 12, "level": "expired"}}}


# 场景清单: (名字, blocks, 断言)
SCENES: list[dict] = [
    {"name": "千股跌停", "blocks": _mk(
        _base_blocks(),
        market_temperature=_fund(3.0, -1.5e10, -8.0e9),
        leader_sentiment=_ladder(zt=8, mb=2, zb=20, dt=1400, zb_rate=0.9),
        overseas=_overseas([{"label": "纳指", "chg_pct": -3.2}, {"label": "伦敦金", "chg_pct": 2.1}]),
        factor_signal=_factor(0.25), strong_direction=_directions([]),
        macro_veto=_veto("hard", 0.3)),
     "exp_posture": "防守观望", "exp_pos_max": _MAX_POS_CRASH},

    {"name": "2015式股灾", "blocks": _mk(
        _base_blocks(),
        market_temperature=_fund(5.0, -2.0e10, -1.0e10),
        leader_sentiment=_ladder(zt=15, mb=2, zb=30, dt=600, zb_rate=0.8),
        overseas=_overseas([{"label": "纳指", "chg_pct": -2.8}, {"label": "道指", "chg_pct": -3.1},
                           {"label": "伦敦金", "chg_pct": 3.0}]),
        strong_direction=_directions([{"name": "银行", "level": "单独", "score": 1}]),
        factor_signal=_factor(0.28), macro_veto=_veto("hard", 0.25)),
     "exp_posture": "防守观望", "exp_pos_max": _MAX_POS_CRASH},

    {"name": "熔断级跳空", "blocks": _mk(
        _base_blocks(),
        market_temperature=_emo("恐慌", 30, 2),
        leader_sentiment=_ladder(zt=30, mb=2, zb=40, dt=300, zb_rate=0.7),
        overseas=_overseas([{"label": "纳指", "chg_pct": -5.5}, {"label": "标普500", "chg_pct": -4.2}]),
        macro_veto=_veto("hard", 0.2)),
     "exp_posture": "防守观望", "exp_pos_max": _MAX_POS_CRASH},

    {"name": "流动性枯竭", "blocks": _mk(
        _base_blocks(),
        market_temperature=_fund(0.0, 0.0, 0.0),  # 资金全无
        leader_sentiment=_ladder(zt=25, mb=2, zb=15, dt=80, zb_rate=0.5),
        overseas=_overseas([]), factor_signal=_factor(0.35)),
     "exp_pos_max": 0.40},  # 无明确极端信号时允许中性档上限40%, 但不得激进

    {"name": "海外崩盘夜", "blocks": _mk(
        _base_blocks(),
        overseas=_overseas([{"label": "纳指", "chg_pct": -4.8}, {"label": "道指", "chg_pct": -4.1},
                           {"label": "标普500", "chg_pct": -4.5}, {"label": "美10年债殖", "chg_pct": 1.8},
                           {"label": "伦敦金", "chg_pct": 5.2}])),
     "exp_pos_max": _MAX_POS_CRASH + 0.05},  # 海外单维降分有限, 允许略放宽但仍受其它维约束

    {"name": "高潮炸板潮", "blocks": _mk(
        _base_blocks(),
        market_temperature=_emo("高潮", 120, 6),
        leader_sentiment=_ladder(zt=120, mb=6, zb=55, dt=10, zb_rate=0.45),
        strong_direction=_directions([{"name": "算力", "level": "主线", "score": 3}]),
        stock_picks={"date": "2026-08-21", "pools": {}}),
     "exp_posture_banned": _POSTURE_BANNED, "expect_risk": True},  # 高炸板 → 必有风险预案

    {"name": "数据全缺失", "blocks": {},
     "exp_posture_banned": _POSTURE_BANNED, "exp_pos_max": 0.50},  # 全缺失不得 fail-open 积极

    {"name": "毒数据混合", "blocks": _mk(
        _base_blocks(),
        market_temperature=None, leader_sentiment="垃圾字符串",
        strong_direction=[1, 2, 3], stock_picks={"pools": None},
        overseas={"weird": True}, factor_signal=None, macro_veto=float("nan"),
        freshness={"score": "NaN"}, engine_fusion={"ok_n": "?"}),
     "exp_pos_max": 0.60},  # 毒数据只要求不炸、输出保守
]


def _check_all(blocks: dict, exp_posture=None, exp_pos_max=None,
               exp_posture_banned=(), expect_risk: bool = False) -> list[str]:
    from fusion_decision import make_decision  # noqa: PLC0415
    try:
        dec = make_decision(blocks)
    except Exception as e:  # noqa: BLE001
        return [f"抛异常: {type(e).__name__}: {str(e)[:120]}"]
    errs = []
    dz = dec.get("定调") or {}
    posture = dz.get("posture", "?")
    position = dz.get("position", "?-?%")
    up = _pos_upper(position)
    if exp_posture and posture != exp_posture:
        errs.append(f"定调应为[{exp_posture}]实际[{posture}]")
    if exp_posture_banned and posture in exp_posture_banned:
        errs.append(f"极端场景禁止[{posture}]")
    if exp_pos_max is not None and up > exp_pos_max:
        errs.append(f"仓位上限[{up:.0%}]超限(≤{exp_pos_max:.0%})")
    risks = dec.get("风险预案")
    if expect_risk and not risks:
        errs.append("应至少1条风险预案")
    if not dec.get("决策依据"):
        errs.append("缺决策依据")
    return errs


def run_drill(date: str | None = None, include_extra: bool = True) -> dict:
    """运行全部场景。include_extra=True 时合并 knowledge-base 扩展场景库(21个)。"""
    date = date or datetime.now(CST).strftime("%Y-%m-%d")
    scenes = list(SCENES)
    if include_extra:
        try:
            from scenario_library import EXTRA_SCENES  # noqa: PLC0415
            scenes += EXTRA_SCENES
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️ 扩展场景库加载失败(仅跑内置): {str(e)[:80]}")
    results = {}
    all_ok = True
    for sc in scenes:
        name = sc["name"]
        errs = _check_all(sc["blocks"], sc.get("exp_posture"), sc.get("exp_pos_max"),
                          sc.get("exp_posture_banned") or (), sc.get("expect_risk", False))
        ok = not errs
        all_ok = all_ok and ok
        results[name] = {"ok": ok, "errors": errs[:5], "scene_keys": len(sc["blocks"]),
                         "source": sc.get("source", "内置")}
        print(f"  [{'✅' if ok else '❌'}] {name}: {('; '.join(errs[:3])) if errs else '安全'} ({sc.get('source','')[:24]})")
    report = {"date": date, "scenes": len(scenes), "all_ok": all_ok,
              "results": results, "checked_at": datetime.now(CST).isoformat(timespec="seconds")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"extreme_{date}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def render_md(report: dict) -> str:
    L = [f"# 🌪️ 极端行情压力测试（{report['date']}）",
         f"- 场景 {report['scenes']} 个 · 全过 **{'✅' if report['all_ok'] else '❌'}**",
         ""]
    for name, r in report["results"].items():
        mark = "✅" if r["ok"] else "❌"
        src = r.get("source", "内置")
        L.append(f"### {mark} {name}（{r['scene_keys']} 块 · 源:{src[:40]}）")
        if r["errors"]:
            for e in r["errors"]:
                L.append(f"- ⚠️ {e}")
        else:
            L.append("- 决策安全：定调保守、仓位受控、无异常")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=None)
    args = ap.parse_args()
    if args.scene:
        for sc in SCENES:
            if sc["name"] == args.scene:
                errs = _check_all(sc["blocks"], sc.get("exp_posture"), sc.get("exp_pos_max"),
                                  sc.get("exp_posture_banned") or (), sc.get("expect_risk", False))
                print(json.dumps({"name": sc["name"], "ok": not errs, "errors": errs[:5]},
                                 ensure_ascii=False, indent=1))
                sys.exit(0 if not errs else 1)
        print("场景未找到:", args.scene)
        sys.exit(2)
    r = run_drill()
    print(render_md(r))
    sys.exit(0 if r["all_ok"] else 1)