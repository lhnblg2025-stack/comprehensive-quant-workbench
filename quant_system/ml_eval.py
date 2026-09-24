# -*- coding: utf-8 -*-
"""
ml_eval.py — ML 模型评估（验证层 · 模块 4/5）

问题：预测层 20 个模块没有正规评估（Accuracy/AUC/F1/Brier/Calibration）。
本模块提供统一评估接口：

    evaluate_classifier(y_true, y_prob) -> dict  # 全指标
    calibration_curve(y_true, y_prob, n_bins)    # 分桶命中率
    evaluate_history(pred_history_csv)           # 从预测历史文件评估

用于 next_day / ensemble / stacking 等所有概率预测器的统一验收。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parents[1]  # V11 审计修复(P0-A): quant_v6/validate 迁移时层级差一级, parents[2]=~主目录 / "generated" / "validate_report"


def evaluate_classifier(y_true, y_prob, name: str = "model") -> dict:
    """完整评估：Accuracy/Precision/Recall/F1/AUC/Brier/LogLoss/校准。"""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_prob))
    y_true, y_prob = y_true[mask], y_prob[mask]
    if len(y_true) < 20 or len(np.unique(y_true)) < 2:
        return {"name": name, "error": "样本不足或标签单一"}

    pred = (y_prob >= 0.5).astype(int)
    acc = float((pred == y_true).mean())

    tp = float(((pred == 1) & (y_true == 1)).sum())
    fp = float(((pred == 1) & (y_true == 0)).sum())
    fn = float(((pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    # AUC
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float("nan")

    # Brier
    brier = float(((y_prob - y_true) ** 2).mean())

    # LogLoss
    # V11 审计修复（Medium）: 原实现 y_prob=0/1 时 log(eps) 被单点拉到 -27.6 污染指标。
    # 修正: 概率先 clip 到 [eps, 1-eps] 再算 logloss。
    eps = 1e-12
    y_prob_c = np.clip(np.asarray(y_prob, dtype=float), eps, 1.0 - eps)
    logloss = float(-(np.asarray(y_true, dtype=float) * np.log(y_prob_c)
                      + (1 - np.asarray(y_true, dtype=float)) * np.log(1 - y_prob_c)).mean())

    # 校准（分桶命中率）
    cal = calibration_curve(y_true, y_prob, n_bins=10)

    return {
        "name": name, "n": int(len(y_true)),
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auc": round(auc, 4) if auc == auc else None,
        "brier": round(brier, 4),
        "logloss": round(logloss, 4),
        "calibration": cal,
        "base_rate": round(float(y_true.mean()), 4),
    }


def calibration_curve(y_true, y_prob, n_bins: int = 10) -> list[dict]:
    """分桶校准：把概率分成 n_bins 桶，每桶实际命中率 vs 预测概率。"""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_prob))
    y_true, y_prob = y_true[mask], y_prob[mask]
    if len(y_true) < 50:
        return []
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if sel.sum() < 5:
            continue
        out.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "pred_mean": round(float(y_prob[sel].mean()), 3),
            "actual": round(float(y_true[sel].mean()), 3),
            "n": int(sel.sum()),
        })
    return out


def evaluate_history(csv_path: str | Path) -> dict:
    """从预测历史 CSV（列: prob/actual 或 p_up/label）评估。"""
    df = pd.read_csv(csv_path)
    prob_col = next((c for c in ["prob", "p_up", "probability", "pred"] if c in df.columns), None)
    label_col = next((c for c in ["actual", "label", "y", "y_true"] if c in df.columns), None)
    if not prob_col or not label_col:
        return {"error": f"找不到概率列/标签列，可用: {list(df.columns)}"}
    return evaluate_classifier(df[label_col], df[prob_col], name=Path(csv_path).stem)


def _demo() -> dict:
    """演示：构造一个校准良好 + 一个过拟合的模型对比。"""
    rng = np.random.default_rng(1)
    n = 2000
    # 真实概率
    p = rng.beta(2, 2, n)
    y = (rng.random(n) < p).astype(int)
    # 模型1：校准良好（加小噪声）
    prob1 = np.clip(p + rng.normal(0, 0.05, n), 0.01, 0.99)
    # 模型2：过度自信（推向 0/1）
    prob2 = np.clip(np.where(p > 0.5, 0.9 + rng.random(n) * 0.1, 0.1 - rng.random(n) * 0.1), 0.01, 0.99)
    return {"well_calibrated": evaluate_classifier(y, prob1, "校准良好"),
            "overconfident": evaluate_classifier(y, prob2, "过度自信")}


def ml_eval_cli() -> None:
    ap = argparse.ArgumentParser(description="ML 模型评估")
    ap.add_argument("--csv", default="", help="预测历史 CSV（列 prob/actual）")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.csv:
        res = evaluate_history(args.csv)
    else:
        res = _demo()

    lines = ["# ML 模型评估报告", ""]
    if isinstance(res, dict) and "error" in res:
        lines.append(f"错误: {res['error']}")
    elif isinstance(res, dict) and "name" in res:
        lines.append(_metrics_to_md(res))
    else:
        for name, m in res.items():
            lines.append(f"## {name}")
            lines.append(_metrics_to_md(m))
            lines.append("")
    md = "\n".join(lines)
    out = Path(args.out) / "ml_eval_report.md"
    out.write_text(md, encoding="utf-8")
    print(f"✓ 已输出 {out}")
    print(md[:800])


def _metrics_to_md(m: dict) -> str:
    if "error" in m:
        return f"错误: {m['error']}"
    lines = [
        f"- 样本数: {m['n']}  基准率: {m['base_rate']}",
        f"- Accuracy: {m['accuracy']}  Precision: {m['precision']}  Recall: {m['recall']}  F1: {m['f1']}",
        f"- AUC: {m['auc']}  Brier: {m['brier']}  LogLoss: {m['logloss']}",
        "",
        "### 校准曲线",
        "| 概率区间 | 预测均值 | 实际命中 | 样本 |",
        "|---|---|---|---|",
    ]
    for c in m.get("calibration", []):
        lines.append(f"| {c['bin']} | {c['pred_mean']} | {c['actual']} | {c['n']} |")
    return "\n".join(lines)


if __name__ == "__main__":
    ml_eval_cli()
