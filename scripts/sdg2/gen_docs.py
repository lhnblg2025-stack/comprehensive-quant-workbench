#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成《数据信息与插补方法文档.md》+ 完整指标清单（含描述/方向/边界/插补率）"""
import pandas as pd
from pathlib import Path

OUT = Path("/mnt/hgfs/share/SDG-2/output")
SRC = Path("/mnt/hgfs/share/SDG-2/20260805 indicator new(1).xlsx")

# 读取指标描述映射
cb = pd.read_excel(SRC, sheet_name='TotalCodebook')
desc_map = {}
for _, r in cb.iterrows():
    if pd.notna(r['SeriesCode']):
        desc_map[str(r['SeriesCode']).strip()] = str(r['SeriesDescription'])

# 数据列 → series 近似（从 pipeline 的 DIRECTION 键提取前缀系列码）
import re

def series_of(col: str) -> str:
    m = re.search(r'([A-Z]{2,}_[A-Z0-9_]+)', col.replace(' ', ''))
    return m.group(1) if m else col

q = pd.read_csv(OUT / 'data_quality_report.csv')
q['series'] = q['indicator'].apply(series_of)
q['description'] = q['series'].apply(lambda s: desc_map.get(s, ''))

goal_names = {
    1: '无贫穷', 2: '零饥饿', 3: '良好健康与福祉', 4: '优质教育', 5: '性别平等',
    6: '清洁饮水和卫生设施', 7: '经济适用的清洁能源', 8: '体面工作和经济增长',
    9: '产业、创新和基础设施', 10: '减少不平等', 11: '可持续城市和社区',
    12: '负责任消费和生产', 13: '气候行动', 14: '水下生物', 15: '陆地生物',
    16: '和平、正义与强大机构', 17: '促进目标实现的伙伴关系',
}
dir_names = {1: '正向↑', -1: '负向↓'}
q['方向'] = q['direction'].map(dir_names)
q['缺失率'] = (q['n_imputed'] / (q['n_total'] - q['n_not_applicable'])).round(3)

q_sorted = q.sort_values(['goal', 'indicator'])
q_sorted.to_csv(OUT / 'indicator_catalog.csv', index=False, encoding='utf-8-sig')

# ── 生成 Markdown 文档 ──
lines = []
lines.append("# 数据信息与插补方法文档")
lines.append("")
lines.append("> 生成时间：2026-08-20　|　数据源：`SDG20260805整合版(1).xlsx`（共享文件夹 SDG-2）")
lines.append("> 处理脚本：`workspace/scripts/sdg2/sdg_calc_pipeline.py`　|　输出：`/mnt/hgfs/share/SDG-2/output/`")
lines.append("")
lines.append("## 一、数据基本信息")
lines.append("")
lines.append("| 项 | 内容 |")
lines.append("|---|---|")
lines.append("| 面板结构 | 193 个国家/地区 × 2011–2020 年（10 年），平衡面板 |")
lines.append("| 目标范围 | SDG 1–17 全部 17 个目标 |")
lines.append("| 指标数 | 133 个（每 SDG 4–14 个不等） |")
lines.append("| 原始行数 | 248,430 条（国家×年×指标） |")
lines.append("| 元数据 | Country / year / CountryCode / GeoAreaCode / Region / Income |")
lines.append("| 特殊标记列 | SDG14 `Land or not`（42 个内陆国）；SDG17 `Developing country or not` / `Donee or not` |")
lines.append("| SDG17 范围 | 仅发展中国家（134 国 × 10 年 = 1,340 行），非 193 国全样本 |")
lines.append("")
lines.append("### 各 SDG 指标数量")
lines.append("")
lines.append("| SDG | 目标名称 | 指标数 |")
lines.append("|---|---|---|")
for g in range(1, 18):
    n = q_sorted[q_sorted['goal'] == g].shape[0]
    lines.append(f"| {g} | {goal_names[g]} | {n} |")
lines.append("")
lines.append("## 二、数据补完（缺失处理）方法")
lines.append("")
lines.append("### 2.1 总体缺失情况")
lines.append("")
lines.append(f"- 全样本原始缺失率：**{(q_sorted['n_imputed'].sum() / q_sorted['n_total'].sum() * 100):.1f}%**")
lines.append(f"- 插补单元总数：**{q_sorted['n_imputed'].sum():,}** 个（国家×年×指标）")
lines.append(f"- 标记为“不适用”的单元：**{q_sorted['n_not_applicable'].sum():,}** 个（全部为内陆国海洋指标，见 §2.4）")
lines.append(f"- 插补后仍缺失：**0** 个")
lines.append("")
lines.append("### 2.2 缺失率分布（按指标）")
lines.append("")
lines.append("| 缺失率区间 | 指标数 | 说明 |")
lines.append("|---|---|---|")
bins = [(0, 0.01, '≤1%（几乎完整）'), (0.01, 0.1, '1%–10%'), (0.1, 0.3, '10%–30%'),
        (0.3, 0.5, '30%–50%'), (0.5, 1.0, '>50%（严重缺失）')]
for lo, hi, label in bins:
    n = q_sorted[(q_sorted['缺失率'] >= lo) & (q_sorted['缺失率'] < hi)].shape[0]
    lines.append(f"| {label} | {n} | |")
lines.append("")
lines.append("**严重缺失（缺失率>50%）指标清单：**")
lines.append("")
lines.append("| 指标 | SDG | 缺失率 | 说明 |")
lines.append("|---|---|---|---|")
for _, r in q_sorted[q_sorted['缺失率'] > 0.5].iterrows():
    lines.append(f"| `{r['indicator']}` | {r['goal']} | {r['缺失率']:.0%} | {r['description'][:60]} |")
lines.append("")
lines.append("### 2.3 插补层级（由精确到粗略）")
lines.append("")
lines.append("| 层级 | 方法 | 适用对象 | 说明 |")
lines.append("|---|---|---|---|")
lines.append("| L0 | 文本注记清洗 | 全部 | “注：只有2015以后数据”等文本 → NaN；非数值 → NaN |")
lines.append("| L1 | 国家内时间线性插值 | 连续指标 | 同一国家 2011–2020 年内插值（`limit_direction=both` 含端点外推） |")
lines.append("| L1b | 国家内前后填充（ffill/bfill） | 0/1 型指标 | 0/1 指标不做线性插值，取最近非空值 |")
lines.append("| L2 | 区域×收入组中位数 | 连续指标 | 同 Region × 同 Income 组内中位数（0/1 取众数） |")
lines.append("| L3 | 区域中位数 | 连续指标 | 同 Region 内中位数（0/1 取众数） |")
lines.append("| L4 | 全球中位数 | 连续指标 | 全样本中位数兜底（0/1 取众数） |")
lines.append("| NA | 不适用标记 | 内陆国海洋指标 | 见 §2.4，不参与插补与得分 |")
lines.append("")
lines.append("### 2.4 特殊处理：SDG14 内陆国")
lines.append("")
lines.append("- 识别：`Land or not == 1`（42 个内陆国：阿富汗、玻利维亚、捷克等，含全部 420 个 国家×年 单元）")
lines.append("- 处理：内陆国的海洋专属指标——`14.1.1`（海滩垃圾）、`14.1.1`（叶绿素偏差）、`14.5.1`（海洋保护区）、`14.3.1`（海洋酸度）——**标记为 not_applicable，不插补、不参与 SDG14 得分平均**")
lines.append("- 依据：UN SDG Index（Sachs et al.）惯例——内陆国无海洋管辖，海洋指标不应以插补值填充")
lines.append("- 影响：内陆国 SDG14 得分仅基于 `14.a.1`（海洋科研支出占比）等非海洋指标，得分可比性需注意（见 §4）")
lines.append("")
lines.append("### 2.5 特殊处理：SDG15")
lines.append("")
lines.append("- SDG15 全部 9 个指标原始缺失率为 0（数据已含山地/无山国的区域口径值，如马尔代夫、新加坡的 15.4.1 山地指标已填区域代表值）")
lines.append("- 无需额外插补；`15.4.1`/`15.4.2` 山地指标对无山国家保留原值（区域口径），不做剔除")
lines.append("- `15.5.1` 红色名录指数为 0–1 型，按正向指标 min-max 标准化处理")
lines.append("")
lines.append("## 三、SDG 得分计算方法")
lines.append("")
lines.append("### 3.1 标准化（0–100 分制）")
lines.append("")
lines.append("| 指标类型 | 边界 | 公式（正向） | 公式（负向） |")
lines.append("|---|---|---|---|")
lines.append("| 百分比指标（占比/覆盖率/比率） | 0–100 天然边界 | (x−0)/(100−0)×100 | (100−x)/(100−0)×100 |")
lines.append("| 其他指标（绝对值/指数/美元等） | 数据 5%–95% 分位数 | (x−P5)/(P95−P5)×100 | (P95−x)/(P95−P5)×100 |")
lines.append("| 0/1 型指标 | 二值 | 0→0，1→100 | — |")
lines.append("| 三级合规指标（16.a.1） | 0/1/2 | 0→0，1→50，2→100 | — |")
lines.append("")
lines.append("- 负向指标（越小越好）已反转：得分越高 = 表现越好（方向表见 `indicator_catalog.csv`）")
lines.append("- 标准化后截断至 [0, 100]")
lines.append("")
lines.append("### 3.2 0/1 指标处理")
lines.append("")
lines.append("| 指标 | 含义 | 处理 |")
lines.append("|---|---|---|")
lines.append("| `13.2.1NDC替代` | 是否提交国家自主贡献（NDC） | 0→0 分，1→100 分（正向） |")
lines.append("| `8.b.1_SL_CPA_YEMP` | 是否有青年就业战略 | 0→0 分，1→100 分（正向） |")
lines.append("| `16.a.1_SG_NHR_CMPLNC` | 国家人权机构合规（0/1/2） | 0→0，1→50，2→100（正向） |")
lines.append("| `16.5 score` / `16.10 score` | 反腐/信息自由替代分（已 0–100） | 直接使用（正向） |")
lines.append("| `13.2.2 替代数据` | GHG 排放替代（万吨 CO₂e） | 负向标准化 |")
lines.append("")
lines.append("### 3.3 分目标得分")
lines.append("")
lines.append("$$ \\text{SDG}_g\\text{ 得分}_{c,t} = \\frac{1}{N_{c,t,g}} \\sum_{i \\in g} \\text{score}_{i,c,t} $$")
lines.append("")
lines.append("- 国家 c 在年份 t 的 SDG g 得分 = 该 SDG 下全部指标标准化得分的**等权平均**")
lines.append("- not_applicable 指标剔除在分母之外")
lines.append("")
lines.append("### 3.4 综合得分")
lines.append("")
lines.append("$$ \\text{总分}_{c,t} = \\frac{1}{K_{c,t}} \\sum_{g=1}^{17} \\text{SDG}_g\\text{ 得分}_{c,t} $$")
lines.append("")
lines.append("- 17 个 SDG 得分的简单平均（等权）")
lines.append("- 某国某年缺失的 SDG（如无 SDG17 数据的发展中国家样本以外的年份）用可用 SDG 数 K 平均，并记录 `n_goals`")
lines.append("")
lines.append("## 四、数据质量说明与使用注意事项")
lines.append("")
lines.append("1. **高缺失插补指标谨慎解读**：`17.2.1`（ODA 占 GNI，缺失 99%）、`14.3.1`（海洋酸度，缺失 90%）、`1.3.1` 两个社保指标（缺失 77–86%）、`16.9.1`（出生登记，缺失 82%）主要由区域/全球中位数填充，插补值信息量有限，建议在稳健性检验中剔除或降权。")
lines.append("2. **SDG14 内陆国可比性**：内陆国 SDG14 仅基于 1 个非海洋指标（14.a.1），与沿海国（5 个指标）口径不同，解释时注意区分；如做跨国比较建议对内陆国 SDG14 单独标记。")
lines.append("3. **SDG17 样本**：仅 134 个发展中国家，且 17.2.1（援助方口径）对发展中国家几乎无数据，主要靠其余 13 个指标。")
lines.append("4. **替代指标**：`13.2.2`、`13.a.1`、`16.5 score`、`16.10 score` 为替代/代理数据，非官方原始口径，引用时需注明。")
lines.append("5. **方向判定**：基于指标官方描述语义人工判定（133 个指标全部有方向），详见 `indicator_catalog.csv`，如需调整可修改脚本 `DIRECTION` 表后重跑。")
lines.append("")
lines.append("## 五、输出文件清单")
lines.append("")
lines.append("| 文件 | 内容 |")
lines.append("|---|---|")
lines.append("| `sdg_panel_imputed.csv` | 插补后完整长表：Country, year, goal, indicator, value, score, impute_flag, landlocked, developing, donee, Region, Income |")
lines.append("| `sdg_scores_by_goal.csv` | 国家×年×SDG 得分（长表，含 goal_score） |")
lines.append("| `sdg_scores_standardized.csv` | 宽表：Country, year, SDG1…SDG17 |")
lines.append("| `sdg_total_score.csv` | 国家×年 综合得分 + n_goals |")
lines.append("| `data_quality_report.csv` | 每指标：原始数/插补数/不适用数/方向/边界类型 |")
lines.append("| `indicator_catalog.csv` | 完整指标清单（含描述、方向、缺失率） |")
lines.append("| `imputation_records.csv` | 逐国×指标的插补明细（方法、数量） |")
lines.append("| `bounds.json` | 各指标标准化边界（lo/hi） |")
lines.append("")
lines.append("## 六、可复现性")
lines.append("")
lines.append("```bash")
lines.append("python workspace/scripts/sdg2/sdg_calc_pipeline.py")
lines.append("```")
lines.append("")
lines.append("脚本读取共享文件夹原 Excel → 全流程重跑 → 覆盖 output/ 下所有文件。")
lines.append("")

doc = "\n".join(lines)
(OUT / "数据信息与插补方法文档.md").write_text(doc, encoding="utf-8")
print("✅ 文档已生成:", OUT / "数据信息与插补方法文档.md")
print("   指标清单:", OUT / "indicator_catalog.csv")
