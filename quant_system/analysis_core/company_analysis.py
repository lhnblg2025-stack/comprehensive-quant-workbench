"""
company_analysis — 中长线引擎 N4：公司八维档案（V11 Phase C）

数据源: 同花顺财务摘要(stock_financial_abstract_new_ths, 1200行/股) + baostock(估值降级)

八维（v1 全自动维度; LLM 维度留接口）:
  1. 商业模式     —— 模板+数据摘要（LLM 增强留接口）
  2. 护城河       —— 毛利率/ROE 稳定性评分
  3. 财务三维     —— 利润质量/资产负债/现金流
  4. 管理层       —— 模板（LLM 增强留接口）
  5. 成长驱动     —— 营收/净利增速拆解
  6. 业绩拐点     —— 同比序列二阶导
  7. 产业链定价权 —— 毛利率趋势/波动
  8. 风险清单     —— 负债率/现金流/增速转负

用法:
  python3 -m quant_system.analysis_core.company_analysis 000001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


# ── 数据获取 ────────────────────────────────────────────
def fetch_financial_ths(code: str) -> pd.DataFrame:
    """同花顺财务摘要长表 → 宽表（行=报告期，列=指标）。"""
    import akshare as ak
    raw = ak.stock_financial_abstract_new_ths(symbol=code)
    if raw is None or raw.empty:
        return pd.DataFrame()
    # 只保留最近 12 期
    raw = raw[raw["report_date"].notna()].copy()
    raw["report_date"] = pd.to_datetime(raw["report_date"])
    raw = raw.sort_values("report_date").tail(12)
    df = raw.pivot_table(index="report_date", columns="metric_name",
                         values="value", aggfunc="first")
    return df.reset_index()


KEY_METRICS = {
    "rev_yoy": "calculate_operating_income_total_yoy_growth_ratio",  # 营收同比
    "np_yoy": "calculate_parent_holder_net_profit_yoy_growth_ratio",  # 归母净利同比
    "np_deduct_yoy": "deduct_net_profit_yoy_growth_ratio",            # 扣非净利同比
    "roe": "index_weighted_avg_roe",                                   # 加权ROE
    "roe_full": "index_full_diluted_roe",
    "gross_margin": "sale_gross_margin",                              # 销售毛利率
    "debt_ratio": "assets_debt_ratio",                                # 资产负债率
    "np": "parent_holder_net_profit",                                  # 归母净利
    "np_deduct": "index_deduct_holder_net_profit",                     # 扣非归母净利
    "ocf_per_share": "index_per_operating_cash_flow_net",              # 每股经营现金流
}


def _safe(df: pd.DataFrame, key: str, default: float = np.nan) -> pd.Series:
    name = KEY_METRICS.get(key)
    if name and name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(default, index=df.index)


def compute_dims(code: str) -> dict:
    df = fetch_financial_ths(code)
    if df.empty:
        return {"code": code, "error": "同花顺财务摘要为空"}

    rev_yoy = _safe(df, "rev_yoy")
    np_yoy = _safe(df, "np_yoy")
    roe = _safe(df, "roe")
    gm = _safe(df, "gross_margin")
    debt = _safe(df, "debt_ratio")
    np_val = _safe(df, "np")
    np_deduct = _safe(df, "np_deduct")
    ocf_ps = _safe(df, "ocf_per_share")

    def last2(s: pd.Series) -> tuple[float, float]:
        v = s.dropna()
        if len(v) >= 2:
            return float(v.iloc[-1]), float(v.iloc[-2])
        if len(v) == 1:
            return float(v.iloc[-1]), np.nan
        return np.nan, np.nan

    # 财务三维
    # 利润质量: 扣非/归母 比值（1 附近=利润实在）；现金流用每股经营现金流趋势替代
    pq = (np_deduct / np_val.replace(0, np.nan)).dropna()
    profit_quality = float(pq.iloc[-1]) if len(pq) else np.nan
    # 2026-08-10 审计: 标签与指标不符（标注"净利/经营现金流"实为"扣非/归母"）→ 统一口径注释
    profit_quality_label = "利润质量(扣非/归母)"
    ocf_trend = float(ocf_ps.dropna().tail(4).mean()) if len(ocf_ps.dropna()) else np.nan

    # 业绩拐点: 同比增速二阶导（最近4期一阶差分趋势）
    def second_deriv(s: pd.Series) -> float:
        v = s.dropna().tail(4)
        if len(v) >= 3:
            d1 = np.diff(v.values)
            return float(np.mean(np.diff(d1))) if len(d1) >= 2 else np.nan
        return np.nan

    # 定价权: 毛利率最近4期均值与波动（同花顺口径=百分比数值，如 89.76）
    gm4 = gm.dropna().tail(4)
    pricing_power = "强" if len(gm4) and gm4.mean() > 30 and gm4.std() < 5 else \
                    ("中" if len(gm4) and gm4.mean() > 15 else "弱")

    # 护城河: ROE+毛利率稳定性（2026-08-10 审计: 同花顺毛利率为百分比数值，
    # 原用 0.4/0.03 小数阈值导致"水平分恒真、稳定性分恒假"，评分失真）
    roe4 = roe.dropna().tail(4)
    moat_score = 0
    if len(roe4) and roe4.mean() > 15: moat_score += 2
    elif len(roe4) and roe4.mean() > 10: moat_score += 1
    if len(gm4) and gm4.mean() > 40: moat_score += 1
    if len(gm4) and gm4.std() < 3: moat_score += 1

    # 风险清单（同花顺口径=百分比数值）
    risks = []
    if len(debt.dropna()) and debt.dropna().iloc[-1] > 70: risks.append("资产负债率偏高")
    if not np.isnan(profit_quality) and profit_quality < 0.6: risks.append("利润含金量低(扣非/归母偏低)")
    if not np.isnan(last2(np_yoy)[0]) and last2(np_yoy)[0] < 0: risks.append("净利同比转负")
    if not np.isnan(last2(rev_yoy)[0]) and last2(rev_yoy)[0] < 0: risks.append("营收同比转负")
    if not np.isnan(ocf_trend) and ocf_trend < 0: risks.append("每股经营现金流为负")

    return {
        "code": code,
        "report_date": str(df["report_date"].iloc[-1].date()),
        "growth": {"rev_yoy": round(last2(rev_yoy)[0], 2) if not np.isnan(last2(rev_yoy)[0]) else None,
                   "np_yoy": round(last2(np_yoy)[0], 2) if not np.isnan(last2(np_yoy)[0]) else None},
        "dim3_finance": {
            "profit_quality": round(profit_quality, 2) if not np.isnan(profit_quality) else None,
            "roe": round(last2(roe)[0], 2) if not np.isnan(last2(roe)[0]) else None,
            "debt_ratio": round(last2(debt)[0], 2) if not np.isnan(last2(debt)[0]) else None,
            "ocf_per_share": round(ocf_trend, 3) if not np.isnan(ocf_trend) else None,
        },
        "dim5_growth_driver": {
            "rev_yoy_trend": round(second_deriv(rev_yoy), 3) if not np.isnan(second_deriv(rev_yoy)) else None,
            "np_yoy_trend": round(second_deriv(np_yoy), 3) if not np.isnan(second_deriv(np_yoy)) else None,
        },
        "dim6_turning_point": {
            "rev_accel": round(second_deriv(rev_yoy), 3) if not np.isnan(second_deriv(rev_yoy)) else None,
            "np_accel": round(second_deriv(np_yoy), 3) if not np.isnan(second_deriv(np_yoy)) else None,
        },
        "dim7_pricing_power": pricing_power,
        "dim2_moat": {"score": moat_score, "level": "强" if moat_score >= 3 else ("中" if moat_score >= 2 else "弱")},
        "dim8_risks": risks,
        "gross_margin": round(gm4.mean(), 2) if len(gm4) else None,
    }


def profile_markdown(code: str) -> str:
    d = compute_dims(code)
    if "error" in d:
        return f"# {code} 八维档案\n\n数据不可用: {d['error']}"
    g = d["growth"]
    f = d["dim3_finance"]
    lines = [
        f"# {code} 中长线八维档案（{d['report_date']}）",
        "",
        "## 1. 商业模式",
        f"- 行业/业务摘要（自动生成，LLM 增强待接）: 营收同比{g['rev_yoy']}%，净利同比{g['np_yoy']}%",
        "",
        "## 2. 护城河",
        f"- 评分 {d['dim2_moat']['score']}/4 = **{d['dim2_moat']['level']}**（ROE+毛利率水平与稳定性）",
        "",
        "## 3. 财务三维",
        f"- {f.get('profit_quality_label', '利润质量(扣非/归母)')}: {f['profit_quality']} | ROE: {f['roe']}% | 资产负债率: {f['debt_ratio']}%",
        "",
        "## 4. 管理层",
        "- 模板占位（LLM 增强待接）",
        "",
        "## 5. 成长驱动",
        f"- 营收增速二阶导: {d['dim5_growth_driver']['rev_yoy_trend']} | 净利增速二阶导: {d['dim5_growth_driver']['np_yoy_trend']}",
        "",
        "## 6. 业绩拐点",
        f"- 营收加速: {d['dim6_turning_point']['rev_accel']} | 净利加速: {d['dim6_turning_point']['np_accel']}（正=加速）",
        "",
        "## 7. 产业链定价权",
        f"- **{d['dim7_pricing_power']}**（毛利率 {d['gross_margin']}）",
        "",
        "## 8. 风险清单",
        *([f"- {r}" for r in d["dim8_risks"]] or ["- 暂无显著风险"]),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="公司八维档案")
    ap.add_argument("code")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if args.json:
        print(json.dumps(compute_dims(args.code), ensure_ascii=False, indent=2))
    else:
        print(profile_markdown(args.code))
