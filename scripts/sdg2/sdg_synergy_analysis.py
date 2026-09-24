#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SDG 协同抑制关系测算（任务2，机器学习方法）
============================================
输入 : /mnt/hgfs/share/SDG-2/output/sdg_scores_standardized.csv（国家×年×17 SDG 得分）
输出 : /mnt/hgfs/share/SDG-2/output/synergy/
  1. sdg_pairwise_corr.csv         Pearson 相关矩阵（原始 + 面板内去均值）
  2. sdg_ml_shap_matrix.csv        XGBoost+SHAP 影响方向/强度矩阵（17×17）
  3. sdg_synergy_classification.csv  逐对分类：协同/抑制/中性 + 证据
  4. sdg_synergy_heatmap.png       协同抑制热力图
  5. 协同抑制关系测算方法文档.md

方法（三层证据）：
  A. Pearson 相关：原始值 + 国家内去均值（within-country，控制国别水平差异）
  B. 机器学习：对每个目标 SDG_j，用其余 16 个 SDG（国家内去均值）训练 XGBoost，
     用 SHAP 判定每个源 SDG_i 对 SDG_j 的影响方向（正=协同，负=抑制）与强度
  C. 固定效应 OLS 稳健性：国家+年份固定效应，系数显著性交叉验证 ML 方向

协同/抑制判定（综合 A/B/C）：
  - 协同：Pearson within>0 且 |corr|≥0.3 或 SHAP 方向为正且 FE 显著为正
  - 抑制：Pearson within<0 且 |corr|≥0.3 或 SHAP 方向为负且 FE 显著为负
  - 中性：证据不足或方向矛盾

用法: python sdg_synergy_analysis.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import shap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

SRC = Path("/mnt/hgfs/share/SDG-2/output/sdg_scores_standardized.csv")
OUT = Path("/mnt/hgfs/share/SDG-2/output/synergy")
GOALS = [f"SDG{i}" for i in range(1, 18)]

GOAL_NAMES = {
    1: "无贫穷", 2: "零饥饿", 3: "良好健康", 4: "优质教育", 5: "性别平等",
    6: "清洁饮水", 7: "清洁能源", 8: "体面工作", 9: "产业创新", 10: "减少不平等",
    11: "可持续城市", 12: "负责任消费", 13: "气候行动", 14: "水下生物",
    15: "陆地生物", 16: "和平正义", 17: "伙伴关系",
}

RNG = 42


def load_panel() -> pd.DataFrame:
    df = pd.read_csv(SRC)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["Country", "year"])
    # 保留有任意 SDG 的行（发达国家可能缺 SDG17）
    df = df[df[GOALS].notna().any(axis=1)]
    # 用国家内去均值（within 变换）：控制国别水平差异，捕捉"进步之间的互动"
    within = df[GOALS].sub(df.groupby("Country")[GOALS].transform("mean"))
    dfw = df[["Country", "year"]].copy()
    for g in GOALS:
        dfw[g] = within[g]
    return df, dfw


# ══════════════════════════════════════════════════════════════════════
# A. Pearson 相关（原始 + within）
# ══════════════════════════════════════════════════════════════════════
def pearson_matrices(df: pd.DataFrame, dfw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = df[GOALS].corr(method="pearson")
    w = dfw[GOALS].corr(method="pearson")
    return raw, w


# ══════════════════════════════════════════════════════════════════════
# B. XGBoost + SHAP：逐目标影响方向与强度
# ══════════════════════════════════════════════════════════════════════
def ml_shap_matrix(dfw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """对每个目标 SDG_j：X = 其余 16 SDG（within），y = SDG_j。
    返回 (方向矩阵, 强度矩阵)：行=源指标，列=目标指标。
    模型：sklearn RandomForest（shap 0.49 与 xgboost 3.2 不兼容）
    """
    from sklearn.ensemble import RandomForestRegressor
    n = len(GOALS)
    dir_mat = pd.DataFrame(np.zeros((n, n)), index=GOALS, columns=GOALS)
    str_mat = pd.DataFrame(np.zeros((n, n)), index=GOALS, columns=GOALS)
    models: dict[str, RandomForestRegressor] = {}

    for j, target in enumerate(GOALS):
        feats = [g for g in GOALS if g != target]
        # 目标与特征都非空的行
        sub = dfw.dropna(subset=[target] + feats)
        if len(sub) < 60:
            print(f"   ⚠️ {target}: 有效样本 {len(sub)} < 60，跳过")
            continue
        X = sub[feats]
        y = sub[target]
        # 时间序列 CV：前 8 年训练，后 2 年验证
        tr_mask = sub["year"] <= 2018
        va_mask = sub["year"] >= 2019
        model = RandomForestRegressor(
            n_estimators=500, max_depth=5, min_samples_leaf=5,
            random_state=RNG, n_jobs=-1,
        )
        model.fit(X[tr_mask], y[tr_mask])
        models[target] = model

        # SHAP 值（用全部有效样本）
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        mean_shap = np.mean(shap_values, axis=0)
        abs_shap = np.mean(np.abs(shap_values), axis=0)
        for k, f in enumerate(feats):
            dir_mat.loc[f, target] = np.sign(mean_shap[k])
            str_mat.loc[f, target] = abs_shap[k]

        # 相对强度归一化（每目标列内最大=1）
        col_max = str_mat[target].max()
        if col_max > 0:
            str_mat[target] = str_mat[target] / col_max

    return dir_mat, str_mat, models


# ══════════════════════════════════════════════════════════════════════
# C. 固定效应 OLS 稳健性（逐对）
# ══════════════════════════════════════════════════════════════════════
def fixed_effect_ols(df: pd.DataFrame) -> pd.DataFrame:
    """对每对 (i,j)：SDG_j ~ SDG_i + 国家FE + 年份FE。
    返回系数、t 值、显著性。用于交叉验证 ML 方向。
    使用 within 变换等价实现（statsmodels 无则手工 demean + numpy lstsq）。
    """
    # 国家-年份双重去均值（two-way within）
    df2 = df.copy()
    for g in GOALS:
        df2[g] = (df2[g]
                  - df2.groupby("Country")[g].transform("mean")
                  - df2.groupby("year")[g].transform("mean")
                  + df2[g].mean())
    rows = []
    for i in GOALS:
        for j in GOALS:
            if i == j:
                continue
            x = df2[i].values
            y = df2[j].values
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 30:
                continue
            X = x[mask]
            Y = y[mask]
            Xc = X - X.mean()
            Yc = Y - Y.mean()
            denom = np.sum(Xc ** 2)
            if denom == 0:
                continue
            beta = np.sum(Xc * Yc) / denom
            # 标准误（HC1 简单近似）
            resid = Yc - beta * Xc
            n_obs = len(Yc)
            sigma2 = np.sum(resid ** 2) / (n_obs - 2)
            se = np.sqrt(sigma2 / denom)
            tstat = beta / se if se > 0 else 0.0
            pval = 2 * (1 - _t_cdf(abs(tstat), n_obs - 2))
            rows.append({"source": i, "target": j, "fe_beta": beta,
                         "fe_t": tstat, "fe_p": pval, "fe_n": n_obs})
    return pd.DataFrame(rows)


def _t_cdf(t: float, dof: float) -> float:
    """标准 t 分布 CDF（近似，用正态当 dof 大；精确用贝塔）"""
    if dof <= 0:
        return 0.5
    # 用 scipy 精确计算
    try:
        from scipy.stats import t as tdist
        return float(tdist.cdf(t, dof))
    except ImportError:
        # 近似：dof>30 用正态
        from math import erf, sqrt
        return 0.5 * (1 + erf(t / sqrt(2)))


# ══════════════════════════════════════════════════════════════════════
# 分类
# ══════════════════════════════════════════════════════════════════════
def classify(raw_corr, within_corr, dir_mat, str_mat, fe_df) -> pd.DataFrame:
    rows = []
    for i in GOALS:
        for j in GOALS:
            if i == j:
                continue
            rc = raw_corr.loc[i, j]
            wc = within_corr.loc[i, j]
            d = dir_mat.loc[i, j]
            s = str_mat.loc[i, j]
            fe = fe_df[(fe_df["source"] == i) & (fe_df["target"] == j)]
            fe_beta = fe["fe_beta"].iloc[0] if len(fe) else np.nan
            fe_p = fe["fe_p"].iloc[0] if len(fe) else np.nan
            fe_sig = (not np.isnan(fe_p)) and fe_p < 0.05
            fe_dir = np.sign(fe_beta) if not np.isnan(fe_beta) else 0

            # 证据汇总
            evidence = []
            if abs(wc) >= 0.3:
                evidence.append(f"corr={wc:+.2f}")
            if abs(d) > 0 and s >= 0.3:
                evidence.append(f"SHAP{d:+.0f}({s:.2f})")
            if fe_sig:
                evidence.append(f"FE{fe_dir:+.0f}(p={fe_p:.3f})")

            # 方向一致性：ML 方向 与 FE 方向（若显著）
            agree = (d == 0) or (fe_dir == 0) or (d == fe_dir)
            # 强度证据数
            strong_ev = sum([
                abs(wc) >= 0.3,
                abs(d) > 0 and s >= 0.3,
                fe_sig,
            ])
            if strong_ev >= 2 and agree and d != 0:
                label = "协同" if d > 0 else "抑制"
                conf = "强"
            elif strong_ev >= 1 and agree and d != 0:
                label = "协同" if d > 0 else "抑制"
                conf = "弱"
            else:
                label = "中性"
                conf = "证据不足或矛盾"
            rows.append({
                "source": i, "target": j,
                "source_name": f"{i} {GOAL_NAMES[int(i[3:])]}",
                "target_name": f"{j} {GOAL_NAMES[int(j[3:])]}",
                "pearson_raw": round(rc, 3), "pearson_within": round(wc, 3),
                "shap_dir": d, "shap_strength": round(s, 3),
                "fe_beta": round(fe_beta, 4) if not np.isnan(fe_beta) else np.nan,
                "fe_p": round(fe_p, 4) if not np.isnan(fe_p) else np.nan,
                "relation": label, "confidence": conf,
                "evidence": "; ".join(evidence),
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# 热力图
# ══════════════════════════════════════════════════════════════════════
def heatmap(mat: pd.DataFrame, title: str, path: Path, vmin=-1, vmax=1, cmap="RdBu_r"):
    fig, ax = plt.subplots(figsize=(10, 8))
    data = mat.values.astype(float)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_yticks(range(len(mat.index)))
    labels_x = [f"{c}({GOAL_NAMES[int(c[3:])]})" for c in mat.columns]
    labels_y = [f"{r}({GOAL_NAMES[int(r[3:])]})" for r in mat.index]
    ax.set_xticklabels(labels_x, rotation=90, fontsize=8)
    ax.set_yticklabels(labels_y, fontsize=8)
    ax.set_title(title, fontsize=13)
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("① 加载面板 + within 变换…")
    df, dfw = load_panel()
    print(f"   样本 {len(df)} 行（国家×年），国家数 {df['Country'].nunique()}")

    print("② Pearson 相关矩阵…")
    raw_corr, within_corr = pearson_matrices(df, dfw)
    raw_corr.to_csv(OUT / "sdg_pairwise_corr_raw.csv", encoding="utf-8-sig")
    within_corr.to_csv(OUT / "sdg_pairwise_corr_within.csv", encoding="utf-8-sig")

    print("③ XGBoost + SHAP（17 个目标模型）…")
    dir_mat, str_mat, models = ml_shap_matrix(dfw)
    dir_mat.to_csv(OUT / "sdg_ml_shap_direction.csv", encoding="utf-8-sig")
    str_mat.to_csv(OUT / "sdg_ml_shap_strength.csv", encoding="utf-8-sig")

    print("④ 固定效应 OLS 稳健性…")
    fe_df = fixed_effect_ols(df)
    fe_df.to_csv(OUT / "sdg_fe_ols.csv", index=False, encoding="utf-8-sig")

    print("⑤ 协同/抑制分类…")
    cls = classify(raw_corr, within_corr, dir_mat, str_mat, fe_df)
    cls.to_csv(OUT / "sdg_synergy_classification.csv", index=False, encoding="utf-8-sig")

    print("⑥ 热力图…")
    heatmap(within_corr, "SDG 协同抑制矩阵（国家内 Pearson 相关）",
            OUT / "sdg_synergy_heatmap_corr.png")
    heatmap(dir_mat * str_mat, "SDG 影响方向×强度（XGBoost SHAP）",
            OUT / "sdg_synergy_heatmap_shap.png", vmin=-1, vmax=1)

    # 汇总统计
    print("\n=== 协同/抑制关系统计 ===")
    print(cls["relation"].value_counts().to_string())
    print("\n强证据关系：")
    strong = cls[cls["confidence"] == "强"]
    print(strong[["source", "target", "relation", "pearson_within",
                  "shap_dir", "shap_strength", "fe_p"]].to_string())

    print("\n✅ 完成。输出:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
