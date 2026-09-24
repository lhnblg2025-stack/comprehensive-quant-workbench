#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""引擎→融合适配器（2026-08-22 融合升级 —— 修复"引擎→决策断链"）

背景: collect_engines 把 87 引擎账本采集进 blocks["engines"]，但只在研报做展示矩阵，
make_decision 从不读取 → risk_system/meta_reviewer/decision_card/multi_agent 的结论
全部不进决策。本适配器把高价值引擎信号翻译成与融合决策兼容的"引擎共识"维度。

信号源（每个均带超时兜底、不炸链；账本非"真产"的直接跳过）:
- risk_system.market_temperature  → 风险温度 (43=低迷/吸筹)
- meta_reviewer.scan_reports      → 元审查 stance (target/neutral/negative + stale)
- decision_card.build_card        → 行动卡 stance/仓位区间
- multi_agent.arbitrate           → 专家共识 (与 model_verdict 一致性)
- calibration.run / predictions   → 校准信任度 (verified 样本/命中率 → 闭环可信)
- trend_system.detect             → 标的趋势行数（辅助）

用法:
  python3 scripts/engine_fusion.py                    # 直接读 blocks(合成) 演示
  python3 scripts/engine_fusion.py --date 2026-08-21  # 指定日期
"""
from __future__ import annotations

import json
import logging
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quant_system"))

logger = logging.getLogger("engine_fusion")

MIN_OK_ENGINES = 3   # 少于3个可用引擎 → 共识不可用
CALL_TIMEOUT = 18    # 单引擎调用超时(s)

# 账本状态白名单
_TRUE_STATUS = {"真产"}


def _num(v, d=0.0):
    try:
        v = float(v)
        return v if v == v else d
    except (TypeError, ValueError):
        return d


def _call_engine(engine: str, entry: str, kwargs: dict | None = None,
                 timeout: int = CALL_TIMEOUT) -> dict:
    """daemon 线程 + join(timeout) 调用引擎入口，防单源挂死全链。"""
    box = {}

    def _run():
        try:
            import importlib
            m = importlib.import_module(f"quant_system.analysis_core.{engine}")
            if "." in entry:
                cls_name, meth = entry.split(".", 1)
                inst = getattr(m, cls_name)()
                f = getattr(inst, meth)
            else:
                f = getattr(m, entry)
            box["v"] = f(**(kwargs or {}))
        except Exception as e:  # noqa: BLE001
            box["e"] = f"{type(e).__name__}: {str(e)[:100]}"

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout)
    if "e" in box:
        return {"ok": False, "note": box["e"]}
    if "v" not in box:
        return {"ok": False, "note": f"超时>{timeout}s"}
    return {"ok": True, "value": box["v"]}


# ── 各引擎信号提取（输出 0-100 子分 + note + ok）──────────────
def _risk_signal(v) -> dict:
    # W2.5 修复：引擎内部 ok=False（数据缺失）不得当有效中性 50 参与融合
    if isinstance(v, dict) and v.get("ok") is False:
        return {"score": 50, "note": f"风险引擎内部失败: {v.get('note') or v.get('error') or '无数据'}",
                "ok": False}
    temp = _num(v.get("temperature"))
    stage = str(v.get("stage") or "?")
    s = 50 + (temp - 50) * 0.8          # 43→44.4, 60→58, 30→34
    note = f"风险温度{temp:.0f}({stage})"
    return {"score": round(max(5, min(95, s)), 1), "note": note, "ok": True}


def _meta_signal(v) -> dict:
    cons = v.get("conclusions") or []
    if not cons:
        return {"score": 50, "note": "元审查无结论", "ok": True}
    stances = [str(c.get("stance") or "").lower() for c in cons]
    neg = sum(1 for s in stances if s in ("negative", "bearish", "看空"))
    pos = sum(1 for s in stances if s in ("positive", "target", "看多", "乐观"))
    stale = sum(1 for c in cons if c.get("stale"))
    s = 50 + (pos - neg) * 6 - (3 if stale else 0)
    note = f"元审查 {pos}多/{neg}空/{len(cons)}篇" + (" ⚠️含陈旧" if stale else "")
    return {"score": round(max(5, min(95, s)), 1), "note": note, "ok": True}


def _card_signal(v) -> dict:
    stance = str(v.get("stance") or "")
    pr = str(v.get("position_range") or "")
    s = 50
    if any(k in stance for k in ("进攻", "积极", "加仓")):
        s += 10
    elif any(k in stance for k in ("防守", "观望", "轻仓", "减仓")):
        s -= 8
    # 仓位区间上限解析（如 "10-20%"）
    up = None
    if "-" in pr:
        try:
            up = _num(pr.split("-")[1].replace("%", "").strip())
        except Exception:  # noqa: BLE001
            up = None
    if up is not None:
        s += (up - 40) * 0.2            # 上限20%→-4, 上限60%→+4
    note = f"行动卡 {stance}（仓位{pr}）"
    return {"score": round(max(5, min(95, s)), 1), "note": note, "ok": True}


def _expert_signal(v) -> dict:
    cons = str(v.get("consensus") or "")
    conf = _num(v.get("confidence"))
    votes = v.get("votes") or v.get("agree_n") or 0
    s = 50
    if any(k in cons for k in ("多", "乐观", "进攻", "主升")):
        s += 8
    elif any(k in cons for k in ("空", "悲观", "退潮", "防守")):
        s -= 8
    s += (conf - 0.5) * 20               # 置信 0.5→0, 0.8→+6, 0.2→-6
    note = f"专家共识 {cons}（置信{conf}）"
    return {"score": round(max(5, min(95, s)), 1), "note": note, "ok": True}


def _trust_signal(v) -> dict:
    """calibration/预测闭环信任度: 有已验样本且命中率≥0.5 才给正信。"""
    verified = _num(v.get("verified"))
    hit = _num(v.get("overall_hit"), 0.5)
    if verified >= 5 and hit >= 0.5:
        s, note = 55 + (hit - 0.5) * 40, f"校准闭环✓ {int(verified)}条 命中{hit:.0%}"
    elif verified >= 1:
        s, note = 45, f"校准样本少({int(verified)}条) 命中{hit:.0%}"
    else:
        s, note = 40, f"校准未闭环({int(verified)}条已验证)"
    return {"score": s, "note": note, "ok": verified >= 1}


def _expert_from_artifact() -> dict | None:
    """multi_agent 产物优先: generated/multi_agent_*.json 最新且≤3天 → 直接解析。"""
    import json as _json
    import time as _time
    try:
        files = sorted((ROOT / "generated").glob("multi_agent_*.json"))
        if not files:
            return None
        p = files[-1]
        if _time.time() - p.stat().st_mtime > 3 * 86400:
            return None  # 陈旧产物不作数
        d = _json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(d, dict):
            return None
        cons = d.get("consensus") or d.get("verdict", {}).get("consensus") or ""
        conf = d.get("confidence") or d.get("verdict", {}).get("confidence")
        if cons:
            return {"ok": True, "value": {"consensus": cons, "confidence": conf or 0.5},
                    "note": f"产物{p.name}"}
    except Exception:  # noqa: BLE001
        return None
    return None


def engine_fused_signals(blocks: dict | None = None, date: str | None = None) -> dict:
    """主适配器 → {date, dimensions, score, consensus, trust, note, ok_n}。"""
    ledger = {}
    if isinstance(blocks, dict):
        eb = blocks.get("engines")
        if hasattr(eb, "value"):
            raw = eb.value
        else:
            raw = eb
        if isinstance(raw, dict):
            data = raw.get("data")
            if isinstance(data, dict) and isinstance(data.get("engines"), dict):
                ledger = {k: v for k, v in data["engines"].items()
                          if isinstance(v, dict) and v.get("status") in _TRUE_STATUS}

    dims: dict = {}
    # 1) risk_system
    _r = _call_engine("risk_system", "market_temperature")
    if _r["ok"]:
        dims["risk"] = _risk_signal(_r["value"])
    else:
        dims["risk"] = {"score": 50, "note": f"风险引擎{_r['note']}", "ok": False}
    # 2) meta_reviewer
    _m = _call_engine("meta_reviewer", "scan_reports")
    if _m["ok"]:
        dims["meta"] = _meta_signal(_m["value"])
    else:
        dims["meta"] = {"score": 50, "note": f"元审查{_m['note']}", "ok": False}
    # 3) decision_card
    _c = _call_engine("decision_card", "build_card")
    if _c["ok"]:
        dims["card"] = _card_signal(_c["value"])
    else:
        dims["card"] = {"score": 50, "note": f"行动卡{_c['note']}", "ok": False}
    # 4) multi_agent —— 产物优先(新鲜≤3天), 避免每次链跑都等 18s 超时（2026-08-22）
    _x = _expert_from_artifact()
    if _x is None:
        _x = _call_engine("multi_agent", "arbitrate", timeout=10)
    if _x["ok"]:
        dims["expert"] = _expert_signal(_x["value"])
    else:
        dims["expert"] = {"score": 50, "note": f"专家层{_x['note']}", "ok": False}
    # 5) 校准信任度（闭环保真）
    try:
        from quant_system.analysis_core import calibration, predictions  # noqa: PLC0415
        cr = calibration.run()
        if not cr.get("verified"):
            cr = predictions.report()
        dims["trust"] = _trust_signal(cr)
    except Exception as e:  # noqa: BLE001
        dims["trust"] = {"score": 40, "note": f"校准不可用 {str(e)[:50]}", "ok": False}

    ok_n = sum(1 for d in dims.values() if d["ok"])
    if ok_n >= MIN_OK_ENGINES:
        # W2.5 修复：只对 ok 维度加权平均，失败引擎(中性50)不再稀释共识分
        ok_scores = [d["score"] for d in dims.values() if d["ok"]]
        score = round(sum(ok_scores) / len(ok_scores), 1)
    else:
        score = 50.0
    notes = "；".join(f"{k}:{d['note']}" for k, d in dims.items() if d.get("note"))
    consensus = "共振" if (ok_n >= MIN_OK_ENGINES and score >= 58) else (
        "分歧" if (ok_n >= MIN_OK_ENGINES and score <= 42) else (
            "样本不足" if ok_n < MIN_OK_ENGINES else "中性"))
    trust = round(sum(1 for d in dims.values() if d["ok"]) / max(1, len(dims)), 2)
    return {
        "date": date,
        "dimensions": dims,
        "score": score,
        "consensus": consensus,
        "trust": trust,
        "ok_n": ok_n,
        "note": f"引擎共识[{consensus}] {notes}"[:400],
    }


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    date = None
    args = sys.argv[1:]
    if "--date" in args:
        date = args[args.index("--date") + 1]
    r = engine_fused_signals(date=date)
    print(json.dumps(r, ensure_ascii=False, indent=1))