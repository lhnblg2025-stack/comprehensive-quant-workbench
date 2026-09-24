"""
hypothesis_generator — 生成式策略假设引擎（V11 短线 M?）

目标: 每日生成 3 条"反共识假设"，呈现在作战地图顶部作为选择题，
打破"情绪好→追高、情绪差→空仓"的思维定式。

双通道:
  1. 规则模板通道（默认，零网络零 LLM）: 预置假设模板库 A/B/C/D，
     用当日真实本地数据（zt_daily_stats / fund_forces / theme_cycle /
     battle_map）实例化；每条假设的 evidence 必须含具体数字，禁止编造。
  2. LLM 通道（可选）: 环境变量 HYPOTHESIS_LLM=1 时启用，用 openai
     兼容接口（HYPOTHESIS_LLM_BASE / HYPOTHESIS_LLM_KEY，默认不启用）
     生成假设；失败自动回退模板通道并标注 source=template。
     本模块 import 层无 requests/akshare 等网络依赖。

用法:
  python3 -m quant_system.analysis_core.hypothesis_generator [--date 2026-08-07]
"""

from __future__ import annotations
import logging

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.config import MARKET_DIR, ZT_DAILY_STATS  # noqa: E402

FUND_FORCES = MARKET_DIR / "fund_forces.parquet"
THEME_CYCLE = MARKET_DIR / "theme_cycle.parquet"
GENERATED_DIR = ROOT / "generated"
BATTLE_MAP_TPL = "battle_map_{date}.json"
OUT_TPL = "hypotheses_{date}.json"

HIST_WINDOW = 120        # 模板A历史回看窗口（交易日）
WINDOW_3D = 3            # 连板高度回落观察窗口
ZT_SURGE = 5             # 模板B"大涨日"单日涨停家数阈值
GAP_TP = 3.0             # 模板B抢跑止盈高开阈值(%)
ZB_COMBO_BAND = 0.05     # 模板D炸板率匹配带宽(±5pp)

# 模板C: 情绪强弱（正向=追高情绪，负向=防守情绪）
POSITIVE_STAGES = {"修复", "发酵", "高潮"}
NEGATIVE_STAGES = {"冰点", "退潮", "分歧"}


# ── 数据装载 ────────────────────────────────────────────────
def _load_stats(target: pd.Timestamp) -> tuple[pd.DataFrame, pd.Series | None]:
    """zt_daily_stats: 返回 (<=target 的全量df, 目标日行)。无数据→(None, None)。"""
    if not ZT_DAILY_STATS.exists():
        return None, None
    df = pd.read_parquet(ZT_DAILY_STATS)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= target].sort_values("date").reset_index(drop=True)
    if df.empty:
        return None, None
    return df, df.iloc[-1]


def _load_fund(target: pd.Timestamp) -> pd.DataFrame:
    if not FUND_FORCES.exists():
        return pd.DataFrame()
    df = pd.read_parquet(FUND_FORCES)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= target].sort_values("date").reset_index(drop=True)
    return df


def _load_theme(target: pd.Timestamp) -> pd.DataFrame:
    if not THEME_CYCLE.exists():
        return pd.DataFrame()
    df = pd.read_parquet(THEME_CYCLE)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= target].sort_values("date").reset_index(drop=True)
    return df


def _load_battle_map(target: pd.Timestamp, requested: str | None) -> dict:
    for d in (target.strftime("%Y-%m-%d"), requested or ""):
        if not d:
            continue
        p = GENERATED_DIR / BATTLE_MAP_TPL.format(date=d)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                logging.getLogger(__name__).error(f"[hypothesis_generator] 操作失败: {e}", exc_info=True)
    return {}


def _classify_stage(row: pd.Series) -> str:
    """基于 zt_daily_stats 的轻量情绪阶段（battle_map 缺失时兜底）。"""
    zt, mb, zb, prem = row.get("zt_cnt"), row.get("max_board"), row.get("zb_rate"), row.get("premium")
    if zt is None or pd.isna(zt):
        return "未知"
    if zt >= 95 and mb is not None and not pd.isna(mb) and mb >= 5:
        return "高潮"
    if zb is not None and not pd.isna(zb) and zb >= 0.38:
        return "分歧"
    if mb is not None and not pd.isna(mb) and mb <= 2 and zt <= 45:
        return "退潮"
    if zt <= 35:
        return "冰点"
    if zt >= 55 and mb is not None and not pd.isna(mb) and mb >= 3:
        return "发酵"
    return "修复"


def _fmt(x) -> float:
    return float(x) if x is not None and not pd.isna(x) else np.nan


# ── 模板A: 炸板率温和 + 连板高度回落 ─────────────────────────
def _template_a(df: pd.DataFrame) -> dict | None:
    row = df.iloc[-1]
    zb, mb = _fmt(row.get("zb_rate")), _fmt(row.get("max_board"))
    if pd.isna(zb) or pd.isna(mb) or zb >= 0.40:
        return None
    boards = df["max_board"].astype(float).tolist()

    # 模式1: 连续{N}日严格下降
    n, i = 0, len(df) - 1
    while i - 1 >= 0 and boards[i] < boards[i - 1]:
        n += 1
        i -= 1
    drop1 = boards[-1] - boards[-2] if len(boards) >= 2 else 0.0
    if n >= 2:
        hi, lo, mode, pat_desc, cond = boards[i], boards[-1], f"连续{n}日下降", "最高板较前日下降", None
    elif drop1 <= -2.0:
        # 模式2: 单日崩塌式回落（≥2板）
        hi, lo = boards[-2], boards[-1]
        mode = f"单日从{hi:.0f}板崩塌至{lo:.0f}板"
        pat_desc = "最高板单日崩塌≥2板"
        cond = (df["zb_rate"] < 0.40) & (df["max_board"].diff() <= -2)
    else:
        return None
    if lo >= hi:
        return None

    # 历史类似时期: 窗口内与当日同型回落（连降N日 / 单日崩塌≥2板），且炸板率<40%
    if cond is None:
        cond = (df["zb_rate"] < 0.40) & (df["max_board"].diff() < 0)
    win = df.iloc[-HIST_WINDOW:-1]  # 不含当日
    hist = win[cond.loc[win.index].fillna(False).astype(bool)]
    m = len(hist)
    if m < 3:
        return None
    # 打板胜率 = 类似时期次日 1进2晋级率 环比上升占比（样本不足则用涨停家数上升占比）
    nxt = hist["jr1"].shift(-1) if "jr1" in hist else None
    up = (nxt > hist["jr1"]) if nxt is not None else None
    n_ok = int(up.sum()) if up is not None else 0
    if up is None or int(up.notna().sum()) < 5:
        up = hist["zt_cnt"].shift(-1) > hist["zt_cnt"]
        n_ok = int(up.sum())
    rate = n_ok / m * 100
    coef = 1.0 if rate >= 55 else 0.8 if rate >= 45 else 0.6 if rate >= 35 else 0.4

    hypothesis = (f"炸板率虽<40%({zb * 100:.1f}%)但连板高度{mode}"
                  f"(最高板{hi:.0f}板→{lo:.0f}板)，历史类似时期(近{HIST_WINDOW}日出现{m}次)"
                  f"打板胜率{rate:.0f}%")
    evidence = (f"当日炸板率{zb * 100:.1f}%(<40%)、最高板{mb:.0f}板({mode})；"
                f"近{HIST_WINDOW}个交易日出现{m}次同型(炸板率<40%且{pat_desc})，"
                f"其中{n_ok}次次日晋级率/涨停家数上升，打板胜率{rate:.0f}%")
    conf = round(min(0.80, 0.40 + 0.02 * m + 0.20 * abs(rate - 50) / 50), 2)
    return {"template": "A", "hypothesis": hypothesis,
            "evidence": evidence,
            "action": f"建议打板组仓位系数×{coef}（打板胜率{rate:.0f}%，仓位系数{coef}）",
            "confidence": max(0.30, conf)}


# ── 模板B: 主线板块连续大涨 → 高开抢跑止盈 ──────────────────
def _template_b(tc: pd.DataFrame) -> dict | None:
    if tc.empty:
        return None
    best = None  # (concept, streak, sum_zt, max_mb, board_name, role)
    for name, g in tc.groupby("concept"):
        g = g.sort_values("date").reset_index(drop=True)
        s, i = 0, len(g) - 1
        while i >= 0 and g["zt_cnt"].iloc[i] >= ZT_SURGE:
            s += 1
            i -= 1
        if s < 2:
            continue
        seg = g.iloc[-s:]
        key = (s, int(seg["zt_cnt"].sum()), int(seg["max_board"].max()))
        if best is None or key > best[0]:
            best = (key, name, seg["board_name"].iloc[-1], seg["role"].iloc[-1])
    if best is None:
        return None
    (n, sum_zt, max_mb), _concept, bname, role = best
    hypothesis = (f"板块{bname}已连续大涨{n}日(累计涨停{sum_zt}只/最高{max_mb}板)，"
                  f"若明日高开>{GAP_TP:.0f}%建议抢跑止盈")
    evidence = (f"theme_cycle: 板块{bname}({role})截至当日已连续{n}个交易日单日涨停≥{ZT_SURGE}只，"
                f"累计涨停{sum_zt}只、区间最高{max_mb}板，处于主线位置")
    action = f"若明日高开>{GAP_TP:.0f}%建议抢跑止盈（竞价冲高先减半仓落袋，回踩再低吸）"
    conf = round(min(0.85, 0.45 + 0.02 * n), 2)
    return {"template": "B", "hypothesis": hypothesis, "evidence": evidence,
            "action": action, "confidence": conf}


# ── 模板C: 情绪-资金合力背离 ────────────────────────────────
def _template_c(df: pd.DataFrame, fund: pd.DataFrame, bm: dict) -> dict | None:
    if fund.empty:
        return None
    row = df.iloc[-1]
    stage = bm.get("emotion_stage") or _classify_stage(row)
    frow = fund.iloc[-1]
    try:
        signs = json.loads(frow["signs"]) if isinstance(frow.get("signs"), str) else frow.get("signs") or {}
    except Exception:
        signs = {}
    youzi = signs.get("游资")
    force_idx = _fmt(frow.get("force_index"))
    if youzi is None or pd.isna(force_idx):
        return None

    # 正向情绪+资金流出 / 负向情绪+资金流入 = 背离
    if stage in POSITIVE_STAGES and youzi < 0:
        kind, direction = "pos", "情绪强但游资净流出"
    elif stage in NEGATIVE_STAGES and youzi > 0:
        kind, direction = "neg", "情绪弱但游资净流入"
    else:
        return None

    # 历史背离样本: 用同一阶段口径 + 游资方向 判定次日
    fund2 = fund.copy()
    fund2["yousign"] = fund2["signs"].map(lambda s: json.loads(s).get("游资") if isinstance(s, str) else (s or {}).get("游资"))
    z = df.copy()
    z["stage"] = z.apply(_classify_stage, axis=1)
    m = pd.merge_asof(z.sort_values("date"), fund2[["date", "yousign", "force_index"]].sort_values("date"),
                      on="date", direction="backward")
    if kind == "pos":
        hist = m[m["stage"].isin(POSITIVE_STAGES) & (m["yousign"] < 0)]
    else:
        hist = m[m["stage"].isin(NEGATIVE_STAGES) & (m["yousign"] > 0)]
    hist = hist.iloc[:-1]  # 不含当日（当日无次日数据）
    if len(hist) < 3:
        return None
    nxt = hist["zt_cnt"].shift(-1)
    med = float(nxt.median())
    pct = float((nxt > hist["zt_cnt"]).mean()) * 100

    force_word = "强" if force_idx >= 70 else ("中" if force_idx >= 50 else "弱")
    outcome = f"次日涨停家数中位数{med:.0f}家、环比上升概率{pct:.0f}%"
    if kind == "pos":
        action = "次日竞价不追高，打板组仓位系数×0.5，等游资回流确认后再加"
    else:
        action = "情绪弱但资金进场，次日可对主线前排轻仓试错（低吸优先，仓位≤3成）"
    hypothesis = (f"情绪{stage}但资金合力背离(游资{direction}、合力{force_word})，"
                  f"历史上背离后{outcome}")
    evidence = (f"当日情绪阶段{stage}、游资净额{youzi}（{'净流出' if youzi < 0 else '净流入'}）、"
                f"资金合力指数{force_idx:.0f}({force_word})；历史上同类背离出现{len(hist)}次，"
                f"{outcome}")
    conf = round(min(0.90, 0.40 + 0.02 * len(hist) + 0.40 * abs(pct - 50) / 50), 2)
    return {"template": "C", "hypothesis": hypothesis, "evidence": evidence,
            "action": action, "confidence": max(0.30, conf)}


# ── 模板D: 最高板+炸板率组合的历史次日分布 ──────────────────
def _template_d(df: pd.DataFrame) -> dict | None:
    row = df.iloc[-1]
    mb, zb = _fmt(row.get("max_board")), _fmt(row.get("zb_rate"))
    if pd.isna(mb) or pd.isna(zb):
        return None
    hist = df[(df["max_board"] == mb) & (df["zb_rate"] - zb).abs() <= ZB_COMBO_BAND]
    hist = hist.iloc[:-1]  # 不含当日
    n = len(hist)
    if n < 3:
        return None
    nxt = hist["zt_cnt"].shift(-1)
    med = float(nxt.median())
    up_pct = float((nxt > hist["zt_cnt"]).mean()) * 100
    pos = 0.8 if med >= 80 else 0.7 if med >= 60 else 0.6 if med >= 45 else 0.4 if med >= 30 else 0.3
    hypothesis = (f"最高板{mb:.0f}板+炸板率{zb * 100:.1f}%组合历史上{n}次出现，"
                  f"次日涨停家数中位数{med:.0f}家 → 仓位{pos:.0%}")
    evidence = (f"当日最高板{mb:.0f}板、炸板率{zb * 100:.1f}%；全历史同组合(炸板率±5pp)"
                f"出现{n}次，次日涨停家数中位数{med:.0f}家、环比上升概率{up_pct:.0f}%")
    action = f"次日总仓位系数建议{pos:.0%}（若当日情绪修复未确认，实际执行再×0.8）"
    conf = round(min(0.85, 0.40 + 0.02 * n), 2)
    return {"template": "D", "hypothesis": hypothesis, "evidence": evidence,
            "action": action, "confidence": conf}


# ── 模板通道编排 ────────────────────────────────────────────
def _template_generate(snapshot: dict, k: int) -> list[dict]:
    df = snapshot["stats_df"]
    items: list[dict] = []
    builders = [
        lambda: _template_a(df),
        lambda: _template_b(snapshot["theme_df"]),
        lambda: _template_c(df, snapshot["fund_df"], snapshot["battle_map"]),
        lambda: _template_d(df),
    ]
    for build in builders:
        if len(items) >= k:
            break
        try:
            item = build()
        except Exception:
            item = None
        if item:
            item["source"] = "template"
            items.append(item)
    return items[:k]


# ── LLM 通道（默认关闭，失败回退模板）───────────────────────
def _llm_enabled() -> bool:
    return os.environ.get("HYPOTHESIS_LLM", "0") == "1"


def _llm_generate(snapshot: dict, k: int) -> list[dict] | None:
    base = os.environ.get("HYPOTHESIS_LLM_BASE", "").strip()
    key = os.environ.get("HYPOTHESIS_LLM_KEY", "").strip()
    model = os.environ.get("HYPOTHESIS_LLM_MODEL", "gpt-4o-mini").strip()
    if not base or not key:
        return None
    prompt = (
        "你是A股短线策略研究员。基于以下当日数据快照，生成"
        f"{k}条'反共识假设'（挑战常规结论），严格输出JSON数组，"
        "每项含 hypothesis/evidence/action/confidence 四个字段，"
        "evidence必须引用快照中的具体数字，action必须可执行(具体到仓位系数)。\n"
        + json.dumps(snapshot["llm_snapshot"], ensure_ascii=False, default=str)
    )
    payload = {"model": model, "messages": [
        {"role": "system", "content": "你只输出合法JSON数组，不要输出其他文字。"},
        {"role": "user", "content": prompt},
    ], "temperature": 0.7}
    try:
        import urllib.request  # 延迟导入：默认通道零网络依赖
        req = urllib.request.Request(
            base.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            method="POST")
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
        text = data["choices"][0]["message"]["content"]
        start, end = text.find("["), text.rfind("]")
        if start < 0 or end <= start:
            return None
        parsed = json.loads(text[start:end + 1])
    except Exception:
        return None
    items = []
    for it in parsed[:k]:
        if not all(it.get(f) for f in ("hypothesis", "evidence", "action")):
            continue
        items.append({"hypothesis": it["hypothesis"], "evidence": it["evidence"],
                      "action": it["action"],
                      "confidence": float(it.get("confidence", 0.5)),
                      "source": "llm"})
    return items or None


# ── 主入口 ──────────────────────────────────────────────────
def _build_snapshot(date: str | None) -> tuple[dict | None, pd.Timestamp | None]:
    if not ZT_DAILY_STATS.exists():
        return None, None
    all_df = pd.read_parquet(ZT_DAILY_STATS)
    all_df["date"] = pd.to_datetime(all_df["date"])
    if all_df.empty:
        return None, None
    target = pd.Timestamp(date) if date else all_df["date"].max()
    if target > all_df["date"].max():  # 不允许前视/伪当日
        target = all_df["date"].max()
    df, row = _load_stats(target)
    if df is None or row is None:
        return None, None
    fund = _load_fund(target)
    theme = _load_theme(target)
    bm = _load_battle_map(target, date)
    llm_snapshot = {
        "date": str(target.date()),
        "emotion_stage": bm.get("emotion_stage") or _classify_stage(row),
        "zt_stats": {c: (None if pd.isna(row[c]) else float(row[c]))
                     for c in row.index if c not in ("date", "ladder_json")},
        "fund_force": (None if fund.empty else {
            "date": str(fund.iloc[-1]["date"].date()), "force_index": _fmt(fund.iloc[-1].get("force_index")),
            "signs": json.loads(fund.iloc[-1]["signs"]) if isinstance(fund.iloc[-1].get("signs"), str) else fund.iloc[-1].get("signs")}),
        "battle_map": {k: bm.get(k) for k in ("regime", "position_range", "recommended", "confidence")
                       if k in bm},
    }
    return {"date": str(target.date()), "stats_df": df, "fund_df": fund,
            "theme_df": theme, "battle_map": bm, "llm_snapshot": llm_snapshot}, target


def generate(date: str | None = None, k: int = 3) -> list[dict]:
    """生成当日反共识假设。

    返回 [{hypothesis, evidence, action, confidence, source}]；
    无数据/数据不足 → 空列表；结果存 generated/hypotheses_{date}.json。
    """
    if k < 1:
        k = 3
    snapshot, target = _build_snapshot(date)
    if snapshot is None or target is None:
        return []

    if _llm_enabled():
        items = _llm_generate(snapshot, k) or []
        if items:
            out = items[:k]
        else:
            out = _template_generate(snapshot, k)  # LLM失败自动回退模板
    else:
        out = _template_generate(snapshot, k)

    payload = {"date": snapshot["date"], "k": len(out),
               "channel": "llm" if out and out[0]["source"] == "llm" else "template",
               "hypotheses": out}
    try:
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        (GENERATED_DIR / OUT_TPL.format(date=snapshot["date"])).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logging.getLogger(__name__).error(f"[hypothesis_generator] 操作失败: {e}", exc_info=True)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="生成式策略假设引擎")
    ap.add_argument("--date", default=None, help="目标日期(YYYY-MM-DD)，默认取本地数据最新日")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()
    result = generate(args.date, args.k)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    print(f"\n共 {len(result)} 条假设 → generated/hypotheses_{args.date or ''}.json")
