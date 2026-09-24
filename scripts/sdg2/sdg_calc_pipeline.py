#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SDG 计算管线（2011-2020 面板，193 国）
========================================
输入 : /mnt/hgfs/share/SDG-2/SDG20260805整合版(1).xlsx
输出 : /mnt/hgfs/share/SDG-2/output/
  1. sdg_panel_imputed.csv        插补后的完整面板（长表：国家×年×指标×得分×插补标记）
  2. sdg_scores_by_goal.csv       每个 SDG 得分（国家×年×17目标）
  3. sdg_total_score.csv          综合得分（国家×年）
  4. sdg_scores_standardized.csv  宽表：国家×年×17个SDG列
  5. data_quality_report.csv      每指标缺失率/插补率/方向/边界
  6. imputation_records.csv       逐国×指标的插补明细
  7. 数据信息与插补方法文档.md    方法说明文档

方法（参考 UN SDG Index / Sachs et al.）：
  - 方向判定：正向指标越大越好，负向指标越小越好（人工规则表）
  - 插补：文本注记→NaN → 国家内时间线性插值 → 区域×收入组中位数 → 区域中位数 → 全球中位数
         （0/1 指标用众数，不线性插值）
  - 标准化：min-max 到 0-100；百分比指标用 0-100 天然边界，其余用 5-95 分位数边界；
            负向指标反转；0/1 指标 0→0, 1→100
  - SDG 得分：该 SDG 下所有指标标准化分的等权平均
  - 特殊 SDG14：内陆国（Land or not=1）海洋指标标记 not_applicable，不参与平均
  - 特殊 SDG15：数据已含山地/无山国区域口径，不额外剔除；保持原值
  - 综合得分：17 个 SDG 得分的简单平均（对缺失的 SDG 用可用 SDG 平均）

用法: python sdg_calc_pipeline.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("/mnt/hgfs/share/SDG-2/SDG20260805整合版(1).xlsx")
OUT = Path("/mnt/hgfs/share/SDG-2/output")

META_COLS = ["Country", "year", "CountryCode", "GeoAreaCode", "Region", "Income"]
FLAG_COLS = {"Land or not", "Developing country or not", "Donee or not",
             "Unnamed: 10", "Unnamed: 11"}
YEARS = list(range(2011, 2021))

# ══════════════════════════════════════════════════════════════════════
# 1. 指标方向表（+1 = 正向越大越好；-1 = 负向越小越好）
#    依据：指标官方描述语义 + SDG Index 惯例
# ══════════════════════════════════════════════════════════════════════
DIRECTION: dict[str, int] = {
    # ── SDG1 ──
    "1.1.1 SI_POV_DAY1": -1, "1.2.1 SI_POV_NAHC": -1,
    "1.4.1SP_ACS_BSRVH2OALLAREA": +1, "1.5.1 VC_DSR_AFFCT": -1,
    "1.a.1DC_ODA_POVLG": +1, "1.a.2SD_XPD_ESED": +1,
    "1.3.1 SI_COV_BENFTS": +1, "1.3.1 SI_COV_UEMP": +1,
    # ── SDG2 ──
    "2.1.1 SN_ITK_DEFC": -1, "2.2.1 SH_STA_STNT": -1,
    "2.2.2 SN_STA_OVWGT": -1, "2.2.3 SH_STA_ANEM": -1,
    "2.5.2ER_NOEX_LBREDN": +1, "2.a.1AG_PRD_AGVAS": +1,
    "2.a.1AG_PRD_ORTIND": +1, "2.c.1AG_FPA_CFPI": -1,
    # ── SDG3 ──
    "3.1.1SH_STA_MORTFEMALE": -1, "3.2.1SH_DYN_IMRT": -1,
    "3.3.1SH_HIV_INCDALLAGEBOTHSEX": -1, "3.3.2SH_TBS_INCD": -1,
    "3.4.1SH_DTH_NCOM": -1, "3.5.2SH_ALC_CONSPT": -1,
    "3.7.2SP_DYN_ADKL": -1, "3.9.1SH_STA_ASAIRP": -1,
    "3.b.1SH_ACS_DTP3": +1, "3.c.1SH_MED_DEN": +1, "3.d.1SH_IHR_CAPS": +1,
    # ── SDG4 ──
    "4.1.2SE_TOT_CPLRALLAREABOTHSEX": +1, "4.2.2SE_PRE_PARTNBOTHSEX": +1,
    "4.3.1SE_ADT_EDUCTRN": +1, "4.4.1SE_ADT_ACTS": +1,
    "4.5.1SE_AGP_CPRAALLAREA": +1, "4.6.1SE_ADT_FUNS": +1,
    "4.a.1SE_ACS_CMPTR": +1, "4.b.1DC_TOF_SCHIPSL": +1, "4.c.1SE_TRA_GRDL": +1,
    # ── SDG5 ──
    "5.3.1SP_DYN_MRBF18": -1, "5.a.1SP_GNP_WNOWNS": +1,
    "5.5.1SG_GEN_PARLFEMALE": +1, "5.b.1IT_MOB_OWN": +1,
    # ── SDG6 ──
    "6.1.1SH_H2O_SAFE": +1, "6.2.1SH_SAN_DEFECT": -1,
    "6.3.1EN_WWT_WWDS": +1, "6.4.1ER_H2O_WUEYST": +1,
    "6.4.2ER_H2O_STRESS": -1, "6.5.1ER_H2O_IWRMD": +1,
    "6.6.1EN_LKRV_PWAC": +1, "6.a.1DC_TOF_WASHL": +1,
    # ── SDG7 ──
    "7.1.1EG_ACS_ELECALLAREA": +1, "7.1.2EG_EGY_CLEAN": +1,
    "7.2.1EG_FEC_RNEW": +1, "7.3.1EG_EGY_PRIM": -1,
    "7.a.1EG_IFF_RANDN": +1, "7.b.1EG_EGY_RNEW": +1,
    # ── SDG8 ──
    "8.1.1NY_GDP_PCAP": +1, "8.10.1FB_CBK_BRCH15+": +1,
    "8.2.1SL_EMP_PCAP": +1, "8.4.2EN_MAT_DOMCMPG": -1,
    "8.5.2SL_TLF_UEM15+BOTHSEX": -1, "8.6.1_SL_TLF_NEET_19ICLS": -1,
    "8.7.1_SL_TLF_CHLDEA": -1, "8.9.1_ST_GDP_ZS": +1,
    "8.a.1DC_TOF_TRDDBML": +1, "8.b.1_SL_CPA_YEMP": +1,
    "8.3.1SL_ISV_IFEM": -1,
    # ── SDG9 ──
    "9.1.2IS_RDP_PORFVOL": +1, "9.2.1NV_IND_MANFISIC4_C": +1,
    "9.3.1_NV_IND_SSIS": +1, "9.4.1EN_ATM_CO2TOTAL": -1,
    "9.5.1_GB_XPD_RSDV": +1, "9.a.1DC_TOF_INFRAL": +1,
    "9.b.1NV_IND_TECHISIC4_C": +1, "9.c.1IT_MOB_3GNTWK": +1,
    # ── SDG10 ──
    "10.2.1_SI_POV_50MI": -1, "10.4.1SL_EMP_GTOTL": +1,
    "10.5.1FI_FSI_FSANL": -1, "10.6.1SG_INT_VRTDEVUNGA": +1,
    "10.7.4SM_POP_REFG_OR": -1, "10.a.1_TM_TRF_ZERO": +1,
    "10.b.1DC_TRF_TOTL": +1,
    # ── SDG11 ──
    "11.1.1_EN_LND_SLUM": -1, "11.2.1_SP_TRN_PUBL": +1,
    "11.3.1_EN_LND_CNSPOP": -1, "11.4.1_GB_XPD_CULNAT_PBPV": +1,
    "11.5.3_VC_DSR_BSDN": -1, "11.6.2EN_ATM_PM25ALLAREA": -1,
    # ── SDG12 ──
    "12.3.1_AG_FOOD_WST_PC": -1, "12.4.2_EN_EWT_GENV": -1,
    "12.5.1_EN_EWT_RCYV": +1, "12.5.1_EN_MWT_RCYV": +1,
    "12.b.1ST_EEV_STDACCT": +1, "12.c.1ER_FFS_CMPT_GDP": -1,
    # ── SDG13 ──
    "13.1.1VC_DSR_MTMP": -1, "13.2.2 替代数据": -1,
    "13.2.1NDC替代": +1, "13.a.1 mitigation_constant price": +1,
    # ── SDG14 ──
    "14.1.1EN_MAR_BEALIT_OV": -1, "14.1.1EN_MAR_CHLDEV": -1,
    "14.5.1ER_MRN_MPA": +1, "14.a.1_ER_RDE_OSEX": +1,
    "14.3.1ER_OAW_MNACD": +1,
    # ── SDG15 ──
    "15.1.1AG_LND_FRSTN": +1, "15.2.1AG_LND_FRSTCERT": +1,
    "15.3.1_AG_LND_DGRD": -1, "15.4.1ER_PTD_MTN": +1,
    "15.4.2_ER_MTN_GRNCVI": +1, "15.5.1ER_RSK_LST": +1,
    "15.6.1ER_CBD_SMTA": +1, "15.8.1_ER_IAS_LEGIS": +1,
    "15.a.1DC_ODA_BDVL": +1,
    # ── SDG16 ──
    "16.1.1VC_IHR_PSRCBOTHSEX": -1, "16.2.2_VC_HTF_DETV": -1,
    "16.3.2_VC_PRS_UNSNT": -1, "16.6.1GF_XPD_GBPC": +1,
    "16.7.1_SG_DMK_JDC": +1, "16.a.1_SG_NHR_CMPLNC": +1,
    "16.5 score": +1, "16.9.1 SG_REG_BRTH": +1, "16.10 score": +1,
    # ── SDG17 ──
    "17.1.1GR_G14_GDP": +1, "17.11.1TX_EXP_GBMRCH": +1,
    "17.11.1TX_IMP_GBMRCH": +1, "17.12.1TM_TAX_DMFNALP": -1,
    "17.13.1BX_KLT_DINV_WD_GD_ZS": +1, "17.17.1GF_COM_PPPI_KD": +1,
    "17.19.1SG_STT_CAPTY": +1, "17.2.1DC_ODA_TOTG": +1,
    "17.3.1GF_FRN_FDI": +1, "17.3.2BX_TRF_PWKR": +1,
    "17.6.1IT_NET_BBNDANYS": +1, "17.7.1DC_ENVTECH_TT": +1,
    "17.8.1IT_USE_ii99BOTHSEX": +1, "17.9.1DC_FTA_TOTAL": +1,
}

# 0/1 型指标（16.a.1 为三级 0/1/2）
BINARY_INDICATORS = {
    "13.2.1NDC替代",
    "8.b.1_SL_CPA_YEMP",
    "16.a.1_SG_NHR_CMPLNC",
}

# SDG14 中海洋专属指标（内陆国不适用）
SDG14_MARINE = {
    "14.1.1EN_MAR_BEALIT_OV",
    "14.1.1EN_MAR_CHLDEV",
    "14.5.1ER_MRN_MPA",
    "14.3.1ER_OAW_MNACD",
}

# 百分比指标用 0-100 天然边界
PERCENT_BOUNDED = {
    "1.1.1 SI_POV_DAY1", "1.2.1 SI_POV_NAHC", "1.4.1SP_ACS_BSRVH2OALLAREA",
    "1.3.1 SI_COV_BENFTS", "1.3.1 SI_COV_UEMP", "2.1.1 SN_ITK_DEFC",
    "2.2.1 SH_STA_STNT", "2.2.2 SN_STA_OVWGT", "2.2.3 SH_STA_ANEM",
    "2.a.1AG_PRD_AGVAS", "3.1.1SH_STA_MORTFEMALE", "3.4.1SH_DTH_NCOM",
    "3.5.2SH_ALC_CONSPT", "3.9.1SH_STA_ASAIRP", "3.b.1SH_ACS_DTP3",
    "3.c.1SH_MED_DEN", "3.d.1SH_IHR_CAPS", "4.1.2SE_TOT_CPLRALLAREABOTHSEX",
    "4.2.2SE_PRE_PARTNBOTHSEX", "4.3.1SE_ADT_EDUCTRN", "4.4.1SE_ADT_ACTS",
    "4.6.1SE_ADT_FUNS", "4.a.1SE_ACS_CMPTR", "4.c.1SE_TRA_GRDL",
    "5.3.1SP_DYN_MRBF18", "5.a.1SP_GNP_WNOWNS", "5.5.1SG_GEN_PARLFEMALE",
    "5.b.1IT_MOB_OWN", "6.1.1SH_H2O_SAFE", "6.2.1SH_SAN_DEFECT",
    "6.3.1EN_WWT_WWDS", "6.5.1ER_H2O_IWRMD", "6.6.1EN_LKRV_PWAC",
    "7.1.1EG_ACS_ELECALLAREA", "7.1.2EG_EGY_CLEAN", "7.2.1EG_FEC_RNEW",
    "8.9.1_ST_GDP_ZS", "8.3.1SL_ISV_IFEM", "9.2.1NV_IND_MANFISIC4_C",
    "9.3.1_NV_IND_SSIS", "9.5.1_GB_XPD_RSDV", "9.b.1NV_IND_TECHISIC4_C",
    "9.c.1IT_MOB_3GNTWK", "10.2.1_SI_POV_50MI", "10.4.1SL_EMP_GTOTL",
    "10.5.1FI_FSI_FSANL", "10.6.1SG_INT_VRTDEVUNGA", "10.a.1_TM_TRF_ZERO",
    "11.1.1_EN_LND_SLUM", "11.2.1_SP_TRN_PUBL", "11.6.2EN_ATM_PM25ALLAREA",
    "12.b.1ST_EEV_STDACCT", "12.c.1ER_FFS_CMPT_GDP", "15.1.1AG_LND_FRSTN",
    "15.2.1AG_LND_FRSTCERT", "15.3.1_AG_LND_DGRD", "15.4.2_ER_MTN_GRNCVI",
    "15.5.1ER_RSK_LST", "15.8.1_ER_IAS_LEGIS", "16.1.1VC_IHR_PSRCBOTHSEX",
    "16.3.2_VC_PRS_UNSNT", "16.6.1GF_XPD_GBPC", "16.9.1 SG_REG_BRTH",
    "17.1.1GR_G14_GDP", "17.13.1BX_KLT_DINV_WD_GD_ZS", "17.3.2BX_TRF_PWKR",
    "17.6.1IT_NET_BBNDANYS", "17.8.1IT_USE_ii99BOTHSEX", "17.12.1TM_TAX_DMFNALP",
    "10.7.4SM_POP_REFG_OR",
}


# ══════════════════════════════════════════════════════════════════════
# 2. 读取与清洗
# ══════════════════════════════════════════════════════════════════════
def load_sheets() -> dict[str, pd.DataFrame]:
    wb = pd.ExcelFile(SRC)
    return {sn: pd.read_excel(SRC, sheet_name=sn)
            for sn in wb.sheet_names if sn.startswith("SDG")}


def clean_col(s: pd.Series) -> pd.Series:
    """文本注记→NaN；非数值→NaN"""
    s = s.astype(str).str.strip()
    s = s.replace({"nan": np.nan, "None": np.nan, "": np.nan})
    mask_note = s.str.contains("注：|注:|只有|仅|数据来源", na=False)
    s = s.mask(mask_note, np.nan)
    return pd.to_numeric(s, errors="coerce")


def build_panel(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """清洗全部 SDG 表 → 长表（country, year, goal, indicator, value, 标记列）"""
    rows = []
    for goal, df in sheets.items():
        gnum = int(goal.replace("SDG", ""))
        flags = {fc: df[fc] for fc in FLAG_COLS if fc in df.columns}
        for col in df.columns:
            if col in META_COLS or col in FLAG_COLS:
                continue
            rows.append(pd.DataFrame({
                "Country": df["Country"],
                "year": pd.to_numeric(df["year"], errors="coerce"),
                "goal": gnum,
                "indicator": col,
                "value": clean_col(df[col]),
                "landlocked": flags.get("Land or not", np.nan),
                "developing": flags.get("Developing country or not", np.nan),
                "donee": flags.get("Donee or not", np.nan),
            }))
    panel = pd.concat(rows, ignore_index=True)
    panel = panel.dropna(subset=["Country"])
    panel = panel[panel["year"].isin(YEARS)]
    return panel


# ══════════════════════════════════════════════════════════════════════
# 3. 插补（长表直接操作）
# ══════════════════════════════════════════════════════════════════════
def impute_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """逐 国家×指标 插补。返回 (插补后长表, 插补记录表)。
    顺序：1) 内陆国海洋指标→not_applicable
          2) 国家内时间线性插值（连续）/ 众数（0-1）
          3) 区域×收入组中位数 → 区域中位数 → 全球中位数
    """
    recs: list[dict] = []
    panel = panel.sort_values(["Country", "indicator", "year"]).copy()
    panel["impute_flag"] = "orig"

    # ── 阶段 1：内陆国海洋指标 ──
    mask_na = panel["indicator"].isin(SDG14_MARINE) & (panel["landlocked"] == 1) & panel["value"].isna()
    panel.loc[mask_na, "impute_flag"] = "not_applicable"
    # 有值的也标记（内陆国海洋指标即使有值也剔除？统一按不适用处理）
    mask_all = panel["indicator"].isin(SDG14_MARINE) & (panel["landlocked"] == 1)
    panel.loc[mask_all, "impute_flag"] = "not_applicable"

    # ── 阶段 2：国家内时间序列填充 ──
    for (country, ind), grp in panel.groupby(["Country", "indicator"]):
        if grp["impute_flag"].eq("not_applicable").all():
            continue
        is_binary = ind in BINARY_INDICATORS
        orig_vals = grp["value"].copy()
        new_vals = orig_vals.copy()
        if is_binary:
            # 0/1：前后填充最近值
            filled = new_vals.ffill().bfill()
            n = int((filled.notna() & new_vals.isna()).sum())
            new_vals = filled
        else:
            # 连续：线性插值（双向）
            interp = new_vals.interpolate(method="linear", limit_direction="both")
            n = int((interp.notna() & new_vals.isna()).sum())
            new_vals = interp
        filled_mask = new_vals.notna() & orig_vals.isna()
        panel.loc[grp.index[filled_mask.values], "value"] = new_vals[filled_mask]
        panel.loc[grp.index[filled_mask.values], "impute_flag"] = (
            "binary_ffill" if is_binary else "country_linear_interp")
        if n > 0:
            recs.append({"country": country, "indicator": ind,
                         "method": "binary_ffill" if is_binary else "country_linear_interp",
                         "n_imputed": n})

    # ── 阶段 3：区域×收入组 → 区域 → 全球 中位数 ──
    # 只处理仍缺失且非 not_applicable 的行
    meta = _get_country_meta()
    panel = panel.merge(meta, on="Country", how="left", suffixes=("", "_meta"))

    na_pool = panel[(panel["value"].isna()) & (panel["impute_flag"] != "not_applicable")]
    for ind, grp in na_pool.groupby("indicator"):
        is_binary = ind in BINARY_INDICATORS
        agg = "median" if not is_binary else lambda x: x.mode().iloc[0] if len(x.mode()) else np.nan
        # 3a 区域×收入组
        for (reg, inc), g2 in grp.groupby(["Region", "Income"]):
            if pd.isna(reg) or pd.isna(inc):
                continue
            ref = panel[(panel["indicator"] == ind) & (panel["Region"] == reg)
                        & (panel["Income"] == inc) & panel["value"].notna()]
            if ref.empty:
                continue
            fill_v = ref["value"].median() if not is_binary else ref["value"].mode().iloc[0]
            if pd.notna(fill_v):
                idx = g2.index
                panel.loc[idx, "value"] = fill_v
                panel.loc[idx, "impute_flag"] = "region_income_median"
                recs.append({"country": g2["Country"].iloc[0], "indicator": ind,
                             "method": "region_income_median", "n_imputed": len(idx)})
        # 3b 区域（动态：本组已被 3a 填掉的不再处理）
        for reg, g2 in grp[panel.loc[grp.index, "value"].isna()].groupby("Region"):
            if pd.isna(reg):
                continue
            ref = panel[(panel["indicator"] == ind) & (panel["Region"] == reg) & panel["value"].notna()]
            if ref.empty:
                continue
            fill_v = ref["value"].median() if not is_binary else ref["value"].mode().iloc[0]
            if pd.notna(fill_v):
                idx = g2.index
                panel.loc[idx, "value"] = fill_v
                panel.loc[idx, "impute_flag"] = "region_median"
                recs.append({"country": g2["Country"].iloc[0], "indicator": ind,
                             "method": "region_median", "n_imputed": len(idx)})
        # 3c 全球
        still_missing = panel[(panel["indicator"] == ind) & panel["value"].isna()
                              & (panel["impute_flag"] != "not_applicable")]
        if not still_missing.empty:
            ref = panel[(panel["indicator"] == ind) & panel["value"].notna()]
            if not ref.empty:
                fill_v = ref["value"].median() if not is_binary else ref["value"].mode().iloc[0]
                if pd.notna(fill_v):
                    panel.loc[still_missing.index, "value"] = fill_v
                    panel.loc[still_missing.index, "impute_flag"] = "global_median"
                    recs.append({"country": still_missing["Country"].iloc[0], "indicator": ind,
                                 "method": "global_median", "n_imputed": len(still_missing)})

    rec_df = pd.DataFrame(recs)
    return panel, rec_df


def _get_country_meta() -> pd.DataFrame:
    df = pd.read_excel(SRC, sheet_name="SDG1")
    return df[["Country", "Region", "Income"]].drop_duplicates(subset=["Country"])


# ══════════════════════════════════════════════════════════════════════
# 4. 标准化（min-max → 0-100）
# ══════════════════════════════════════════════════════════════════════
def standardize(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = panel.copy()
    bounds: dict[str, dict] = {}
    for ind, grp in out.groupby("indicator"):
        d = DIRECTION.get(ind, 1)
        v = grp["value"]
        if ind in BINARY_INDICATORS:
            if ind == "16.a.1_SG_NHR_CMPLNC":
                score = v * 50.0  # 0/1/2 → 0/50/100
            else:
                score = v * 100.0
            out.loc[grp.index, "score"] = score
            bounds[ind] = {"type": "binary", "direction": d}
            continue
        if ind in PERCENT_BOUNDED:
            lo, hi = 0.0, 100.0
            btype = "percent_0_100"
        else:
            lo = v.quantile(0.05)
            hi = v.quantile(0.95)
            btype = "p5_p95"
            if lo == hi:
                lo, hi = v.min(), v.max()
        if hi <= lo:
            out.loc[grp.index, "score"] = 50.0
        elif d == -1:
            out.loc[grp.index, "score"] = ((hi - v) / (hi - lo) * 100.0).clip(0, 100)
        else:
            out.loc[grp.index, "score"] = ((v - lo) / (hi - lo) * 100.0).clip(0, 100)
        bounds[ind] = {"type": btype, "direction": d, "lo": float(lo), "hi": float(hi)}
    return out, bounds


# ══════════════════════════════════════════════════════════════════════
# 5. SDG 得分
# ══════════════════════════════════════════════════════════════════════
def goal_scores(panel: pd.DataFrame) -> pd.DataFrame:
    """国家×年份×SDG 得分 = 该 SDG 指标得分等权平均（剔除 not_applicable）。"""
    p = panel[panel["impute_flag"] != "not_applicable"]
    g = (p.groupby(["Country", "year", "goal"])["score"].mean()
         .reset_index().rename(columns={"score": "goal_score"}))
    return g


def total_scores(g: pd.DataFrame) -> pd.DataFrame:
    t = (g.groupby(["Country", "year"])["goal_score"]
         .agg(["mean", "count"]).reset_index())
    t.columns = ["Country", "year", "total_score", "n_goals"]
    return t


# ══════════════════════════════════════════════════════════════════════
# 6. 主流程
# ══════════════════════════════════════════════════════════════════════
def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("① 读取与清洗…")
    sheets = load_sheets()
    panel = build_panel(sheets)
    print(f"   面板行数 {len(panel)}，指标数 {panel['indicator'].nunique()}，"
          f"国家数 {panel['Country'].nunique()}")

    print("② 插补…")
    panel_imp, imp_rec = impute_panel(panel)
    n_na = int(panel_imp["value"].isna().sum())
    print(f"   插补后剩余缺失 {n_na}（应仅为无参照的极端情况）")
    imp_rec.to_csv(OUT / "imputation_records.csv", index=False, encoding="utf-8-sig")

    print("③ 标准化…")
    panel_sc, bounds = standardize(panel_imp)
    panel_sc.to_csv(OUT / "sdg_panel_imputed.csv", index=False, encoding="utf-8-sig")

    print("④ SDG 得分…")
    g = goal_scores(panel_sc)
    g.to_csv(OUT / "sdg_scores_by_goal.csv", index=False, encoding="utf-8-sig")
    t = total_scores(g)
    t.to_csv(OUT / "sdg_total_score.csv", index=False, encoding="utf-8-sig")

    wide = g.pivot_table(index=["Country", "year"], columns="goal",
                         values="goal_score").reset_index()
    wide.columns = [f"SDG{int(c)}" if isinstance(c, (int, float)) and not pd.isna(c) else c
                    for c in wide.columns]
    wide.to_csv(OUT / "sdg_scores_standardized.csv", index=False, encoding="utf-8-sig")

    q = panel_imp.groupby("indicator").agg(
        goal=("goal", "first"),
        n_total=("value", "size"),
        n_orig=("value", lambda s: (panel_imp.loc[s.index, "impute_flag"] == "orig").sum()),
        n_imputed=("value", lambda s: panel_imp.loc[s.index, "impute_flag"].isin(
            ["country_linear_interp", "binary_ffill", "region_income_median",
             "region_median", "global_median"]).sum()),
        n_not_applicable=("impute_flag", lambda s: (s == "not_applicable").sum()),
    ).reset_index()
    q["direction"] = q["indicator"].map(DIRECTION)
    q["bound"] = q["indicator"].map(lambda i: bounds.get(i, {}).get("type", ""))
    q.to_csv(OUT / "data_quality_report.csv", index=False, encoding="utf-8-sig")

    with open(OUT / "bounds.json", "w", encoding="utf-8") as f:
        json.dump(bounds, f, ensure_ascii=False, indent=2, default=str)

    print("✅ 完成。输出目录:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
