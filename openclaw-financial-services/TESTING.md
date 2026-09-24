# 🧪 测试任务：Financial Services Skills for OpenClaw

8 个测试任务，覆盖所有模块，从简单到复杂。安装 skills 后在 OpenClaw 中直接使用以下 prompt 进行测试。

> **前置条件**：确保 skills 已复制到 `${OPENCLAW_SKILLS_DIR:-$HOME/.local/share/openclaw/skills}/` 并重启了 Gateway。

---

## Test 1: 可比公司分析

**模块**: Financial Analysis | **难度**: ⭐⭐ | **Skills**: `fsi-comps-analysis`, `fsi-fa-cmd-comps`

**Prompt**:
```
帮我做一个英伟达（NVDA）的可比公司分析，和 AMD、博通、高通做对比，重点看估值是否偏高。
```

**验证要点**:
- [ ] 自动选择了 EV/Revenue、EV/EBITDA、P/E 等核心估值倍数
- [ ] 包含统计汇总（Median, 75th, 25th percentile）
- [ ] 生成 Excel 文件而非纯文本
- [ ] 有方法论说明和数据来源引注

---

## Test 2: DCF 估值模型

**模块**: Financial Analysis | **难度**: ⭐⭐⭐ | **Skill**: `fsi-fa-dcf-model`

**Prompt**:
```
用 DCF 模型估算特斯拉的内在价值，假设 WACC 是 10%，永续增长率 3%，预测 5 年。
```

**验证要点**:
- [ ] 构建了完整的自由现金流预测
- [ ] 有终值计算（Gordon Growth 或 Exit Multiple）
- [ ] 包含敏感性分析表（WACC vs Growth Rate）
- [ ] 使用 Excel 公式引用而非硬编码

---

## Test 3: 财报分析

**模块**: Equity Research | **难度**: ⭐⭐ | **Skill**: `fsi-er-earnings-analysis`

**Prompt**:
```
分析 Apple 最新一个季度的财报表现，写一份股票研究报告。
```

**验证要点**:
- [ ] 搜索了最新数据而非依赖训练数据
- [ ] 包含 beat/miss 分析（实际 vs 预期）
- [ ] 生成 Word 文档（8-12 页格式）
- [ ] 图表数量在 8-12 个范围内
- [ ] 有完整的数据来源引用和超链接

---

## Test 4: CIM 起草

**模块**: Investment Banking | **难度**: ⭐⭐⭐⭐ | **Skill**: `fsi-ib-cim-builder`

**Prompt**:
```
假设我们在帮一家年收入 5000 万美元的 SaaS 公司做卖方 M&A，请帮我起草 CIM 的框架和投资亮点部分。
公司主要做企业级数据分析平台，ARR 增长 40%，毛利率 75%，有 200 个企业客户。
```

**验证要点**:
- [ ] 按标准 CIM 结构（Executive Summary → Company Overview → Industry → Growth → Financials）
- [ ] 投资亮点抓住增长 + 利润率 + 防御性三个核心
- [ ] 语气专业、fact-based（不夸张）
- [ ] 建议了 40-60 页的完整 CIM 结构

---

## Test 5: 项目筛选

**模块**: Private Equity | **难度**: ⭐⭐ | **Skill**: `fsi-pe-deal-screening`

**Prompt**:
```
我收到一份 CIM，这是一家做宠物医疗连锁的公司，年收入 2 亿元，EBITDA 3000 万元，
目前有 50 家门店，卖方预期估值 10x EBITDA。帮我快速评估一下这个项目是否值得深入看。
```

**验证要点**:
- [ ] 评估了估值是否合理（10x EBITDA vs 行业基准）
- [ ] 识别了关键风险（集中度、人才依赖、监管等）
- [ ] 给出了明确的 pass/pursue 建议
- [ ] 建议了下一步尽调重点

---

## Test 6: 客户会议准备

**模块**: Wealth Management | **难度**: ⭐⭐ | **Skill**: `fsi-wm-client-review-prep`

**Prompt**:
```
我明天要和一个高净值客户做季度回顾，他的投资组合大约 1000 万美元，
主要是 60% 美股 + 20% 债券 + 10% 另类 + 10% 现金。帮我准备一下会议材料。
```

**验证要点**:
- [ ] 包含投资组合绩效摘要
- [ ] 分析了资产配置偏移
- [ ] 准备了谈话要点和建议
- [ ] 考虑了当前市场环境

---

## Test 7: 并购模型

**模块**: Investment Banking | **难度**: ⭐⭐⭐⭐ | **Skill**: `fsi-ib-merger-model`

**Prompt**:
```
假设微软以 500 亿美元收购 Palantir，50% 现金 + 50% 股票，帮我做一个增厚/稀释分析。
```

**验证要点**:
- [ ] 计算了 pro forma EPS
- [ ] 分析了交易对收购方 EPS 的增厚/稀释影响
- [ ] 包含协同效应假设
- [ ] 生成了 Excel 模型

---

## Test 8: 综合端到端工作流（跨模块）

**模块**: 跨模块 | **难度**: ⭐⭐⭐⭐⭐ | **Skills**: 多个联动

**Prompt**:
```
我们在考虑投资一家做 AI 客服 SaaS 的公司。请帮我完成以下工作：
1) 找 4-5 个可比公司做 comps 分析
2) 做一个简单的 DCF 估值
3) 用私募股权的视角列出关键尽调问题
```

**验证要点**:
- [ ] 依次触发了 comps → DCF → DD checklist 三个 skill
- [ ] 各环节的输出前后一致（同样的可比公司、同样的假设）
- [ ] 生成了多个交付物（Excel + checklist）
- [ ] 整体工作流连贯

---

## 📋 测试建议

| 优先级 | 测试 | 原因 |
|:------:|------|------|
| 1 | Test 1 | 最基础，确认 skill 能正常加载和触发 |
| 2 | Test 5 | 最简单的端到端测试，输入少，输出明确 |
| 3 | Test 3 | 测试实时数据获取能力 |
| 4 | Test 8 | 最接近真实使用场景的多 skill 联动测试 |

> **注意**：如果没有配置 MCP 数据源，OpenClaw 会通过 web search 或要求你手动提供数据来完成分析。
> 核心 skill 逻辑（工作流、格式、检查清单）不依赖 MCP，仍然可以正常工作。
