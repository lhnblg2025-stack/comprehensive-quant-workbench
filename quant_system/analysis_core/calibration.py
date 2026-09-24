"""
calibration — 预测校准报告（V11 进化闭环，周更）

读 predictions.parquet → 分模块胜率 / 方向命中 / Brier / 分情绪阶段窗口
输出: generated/calibration_report.md + json

用法:
  python3 -m quant_system.analysis_core.calibration
"""

from __future__ import annotations
import logging

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.config import PREDICTIONS  # noqa: E402


def get_module_weights(date: str | None = None, window_days: int = 30) -> dict:
    """预测记录库 → 分模块权重（决策层反哺，battle_map 调用）。

    规则: 近 window_days 天已验证预测的模块，命中率映射为权重 [0.4, 1.2];
    样本 <5 的模块给默认 1.0；无验证样本的模块 0.9（未证明有效）。
    另叠加用户反馈（feedback.jsonl）: dislike 占比 ≥50% 的方向额外降权 0.85。
    """
    if not PREDICTIONS.exists():
        return {}
    df = pd.read_parquet(PREDICTIONS)
    if df.empty:
        return {}
    df["created_at"] = pd.to_datetime(df["created_at"])
    if df["created_at"].dt.tz is not None:
        cutoff = pd.Timestamp.now(tz=df["created_at"].dt.tz)
        if date:
            cutoff = pd.Timestamp(date).tz_localize(df["created_at"].dt.tz)
    else:
        cutoff = pd.Timestamp.now()
        if date:
            cutoff = pd.Timestamp(date)
    recent = df[df["created_at"] >= cutoff - pd.Timedelta(days=window_days)]
    done = recent[recent["status"].isin(["correct", "wrong"])]
    weights: dict[str, float] = {}
    if done.empty:
        return {}
    for m, g in done.groupby("source_module"):
        n = len(g)
        hit = float((g["status"] == "correct").mean())
        if n >= 5:
            w = 0.4 + hit * 0.8          # 命中率 0→0.4, 1→1.2
        else:
            w = 1.0 if n >= 3 else 0.9   # 样本不足: 微降权
        weights[m] = round(max(0.4, min(1.2, w)), 2)
    # 用户反馈降权: 近 window_days 天 dislike 占比 ≥50% 的方向
    fb_file = ROOT / "generated" / "feedback.jsonl"
    if fb_file.exists():
        try:
            fbs = [json.loads(l) for l in fb_file.read_text(encoding="utf-8").strip().splitlines() if l.strip()]
            # 按时间窗口过滤
            cutoff_ts = (pd.Timestamp.now() - pd.Timedelta(days=window_days)).isoformat()
            fbs = [f for f in fbs if f.get("direction") and str(f.get("ts", "")) >= cutoff_ts[:10]]
            if fbs:
                from collections import Counter
                cnt = Counter(f["direction"] for f in fbs)
                dis = Counter(f["direction"] for f in fbs if f.get("vote") == "dislike")
                # 方向 → 模块映射（供消费方读取）
                DIR_MODULE = {"emotion": "emotion_cycle", "scenario": "scenario",
                              "打板": "ladder", "共振": "resonance", "主线": "resonance"}
                for d, n in cnt.items():
                    if n >= 3 and dis.get(d, 0) / n >= 0.5:
                        mod = next((m for k, m in DIR_MODULE.items() if k in d), None)
                        if mod and mod in weights:
                            weights[mod] = round(weights[mod] * 0.85, 2)
                        weights.setdefault(f"user::{d}", 1.0)
                        weights[f"user::{d}"] = 0.85
        except Exception as e:
            logging.getLogger(__name__).error(f"[calibration] 操作失败: {e}", exc_info=True)
    return weights


def run() -> dict:
    if not PREDICTIONS.exists():
        return {"total": 0, "note": "预测记录库为空，等待积累"}
    df = pd.read_parquet(PREDICTIONS)
    done = df[df["status"].isin(["correct", "wrong"])]
    if done.empty:
        return {"total": len(df), "pending": int((df["status"] == "pending").sum()),
                "note": "暂无已验证预测"}

    out = {"total": len(df), "verified": len(done),
           "overall_hit": round(float((done["status"] == "correct").mean()), 3)}

    # 分模块
    mods = []
    for m, g in done.groupby("source_module"):
        mods.append({"module": m, "n": len(g),
                     "hit": round(float((g["status"] == "correct").mean()), 3)})
    out["by_module"] = mods

    # Brier（有概率的预测）
    prob = done[done["probability"].notna()]
    if len(prob):
        p = prob["probability"].astype(float)
        y = (prob["status"] == "correct").astype(int)
        out["brier"] = round(float(((p - y) ** 2).mean()), 4)

    # 低胜率模块预警（决策层降权依据）
    out["degrade_warning"] = [m for m in mods if m["n"] >= 5 and m["hit"] < 0.5]
    return out


def report_markdown() -> str:
    r = run()
    if r.get("total") == 0:
        return "# 📊 V11 预测校准报告\n\n预测记录库为空，等每日推演积累后自动出报告。"
    lines = [
        "# 📊 V11 预测校准报告",
        "",
        f"- 总预测: {r['total']} | 已验证: {r.get('verified', 0)} | 总命中率: {r.get('overall_hit')}",
    ]
    if "brier" in r:
        lines.append(f"- Brier 得分: {r['brier']}（越低越好）")
    lines += ["", "## 分模块胜率"]
    for m in r.get("by_module", []):
        warn = " ⚠️ 低于50%→降权" if m["hit"] < 0.5 and m["n"] >= 5 else ""
        lines.append(f"- {m['module']}: {m['hit']}（n={m['n']}）{warn}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="预测校准报告")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if args.json:
        print(json.dumps(run(), ensure_ascii=False, indent=2))
    else:
        md = report_markdown()
        out = ROOT / "generated" / "calibration_report.md"
        out.write_text(md, encoding="utf-8")
        print(md)
        print(f"\n已保存: {out}")
