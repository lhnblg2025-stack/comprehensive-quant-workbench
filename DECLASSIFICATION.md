# 脱密说明 · Declassification Notice

本仓库是 A股量化研究工作台的**公开脱密版本**。本文档说明发布范围、排除方法、
被移除的内容类别，以及**残留风险**。请与 [`README.md`](README.md)、
[`docs/RELEASE_SCOPE.md`](docs/RELEASE_SCOPE.md) 一起阅读。

---

## 1. 发布原则

1. **保留 Git 历史**：本仓库沿用原有公开提交历史，脱密后的完整工作台在其之上追加提交。
2. **公开工作台框架**：后端、前端、平台门面、脚本、部署资产、技能包与 RAG 检索代码。
3. **只公开三个策略**：`rsi_reversal`、`low_volatility`、`momentum_12_1`。
   保留策略引擎与准入框架，移除具体私有策略实现与私有研究活动配置。
4. **排除四类内容**：凭据、运行数据、个人信息、敏感恢复目录。
5. **宁缺毋滥**：无法确认版权的第三方资料一律不发布。

---

## 2. 排除项清单

### 2.1 凭据与密钥

| 内容 | 处理 |
|---|---|
| 真实 API key（多个供应商，`sk-` 前缀，存在 `deploy/ai-coders/keys/`） | **整目录移除** |
| 个人 LLM 中转/代理配置与域名 | 移除；代码中的域名替换为 `your-llm-relay.example.com` |
| 云主机登录账号与公网 IP | 替换为 `user@your-cloud-host.example.com` / 占位域名 |
| 微信账号 ID / 聊天 openid | 改为环境变量注入，仓库内留空 |
| `.env` / `.pem` / `.key` / cookie 文件 | 不纳入；`.gitignore` 阻止 |
| `config/.env.secrets.example` | 仅保留 `CHANGE_ME` / 空值占位符 |

### 2.2 运行数据与产物

`data_warehouse/`、`data_cache/`、`generated/`、`logs/`、`*.parquet`、`*.db`、`*.sqlite*`、
`*.pkl`、`*.log`、`__pycache__/`、`.coverage`、`scripts/ccrd_output/`、覆盖率与审计转储。

### 2.3 个人信息

个人绝对路径（`/home/<user>/...`、`C:\Users\<user>\...`）→ 仓库相对路径或环境变量
（`QUANT_WORKSPACE`、`QUANT_DESKTOP`、`QUANT_ROOT`）。
自选股与持仓状态（`config/watchlist.json`、`config/position_map.json`、
`config/stock_pool.json`、`config/trade_orders.lock`）→ 移除。

### 2.4 敏感恢复目录

`recovered_cloud*`、`recovered_knowledge`、`recovered_external`、`recovered_git_history`、
`private_recovery/`、`recovery_docs/`、`项目文档/`、`研究报告/`、`competition_docs/`，
以及**嵌套的私有 Git 历史** `quant_system/.git/`。

### 2.5 私有研究资产（"只公开三个策略"）

移除的具体策略实现与研究配置：

- `quant_system/strategy_library.py`（约 30 个内置策略实现）
- `quant_system/advanced_strategies.py`、`alpha_mining.py`、`expanded_signals.py`
- `quant_system/strategy_selection.py`、`strategy_research_runner.py`
- `quant_system/strategy_matrix.py`、`run_pit_strategy_matrix.py`
- `quant_system/configs/research/` 下的私有研究活动配置（campaign / matrix / OOS 基线等）
- `scripts/` 下的策略筛选、矩阵、准入、OOS 与稳定性研究脚本
- `quant_system/design/` 内部设计、审计与代理配置文档
- 与上述模块绑定的测试
- 私有因子清单：`strategy_registry.py` 仅保留三个公开策略的信号，
  其余家族仅以"本版未公开具体信号"列出

保留的框架：`strategy_engine.py`（信号融合/资金分配）、`strategy_matrix_backtest.py`
（A股执行会计）、`strategy_data_gate.py` / `strategy_admission.py` /
`strategy_promotion_gate.py`（准入与数据门禁）、`strategy_registry.py`（家族目录）、
`factor_backtest_runner.py`（通用因子回测）。

### 2.6 知识库与技能内容

- `quant_web/static/knowledge_base.html`：原先内嵌约 287 项个人技能导出
  （含大量第三方书籍摘要与个人工作区绝对路径）。公开版**替换为纯界面壳**，
  内嵌知识库为空，需使用者用自有资料在本地重建。
- 个人知识库导出、书籍摘要、向量索引与缓存：不纳入。
- `openclaw-financial-services/skills/` 技能包按 `SKILL.md` 规范保留。

---

## 3. 方法与工具

| 步骤 | 命令 / 方式 | 作用 |
|---|---|---|
| 索引隔离 | `.git/info/exclude` | 从索引层阻止凭据、恢复目录、运行数据进入提交 |
| 密钥/PII 扫描 | `python3 scripts/check_release.py` | 私钥块、主流 token、硬编码凭据、本机路径 |
| 编译检查 | `make lint-config` | 批量编译，捕捉被清理代码导致的语法/导入错误 |
| 依赖检查 | 静态扫描已删除模块的引用 | 修断链，避免公开版导入失败 |
| 测试 | `make contract` | 全量非集成测试 |
| 人工审阅 | 逐类核对 | 自动规则不足以判定敏感性 |

---

## 4. 已知限制与残留风险

1. **数据源名称与接口路径仍在代码中**（如东方财富、akshare、Tushare 等）。
   这是功能所需，不等于获得再分发授权；使用者须自行遵守供应商条款。
2. **系统设计信息保留**：代码包含通用量化系统架构、指标与风控设计。
   这部分是公开工作台的核心价值，无法在不破坏可用性的前提下移除。
3. **基准与成本口径为研究口径**：三个策略的历史数字来自固定研究代理样本，
   **不是实盘业绩**，且均未通过正式准入（`DATA_BLOCKED / research_only`）。
4. **自动扫描不能证明不存在所有形式的敏感信息**。若发现遗漏，请提 issue，
   我们会尽快处理并视情况轮换相关凭据。
5. **凭据轮换建议**：任何曾经存在于工作区、但未进入本仓库的密钥，
   若怀疑泄露仍建议立即轮换。

---

## 5. 使用要求

- 本仓库代码以 [MIT](LICENSE) 发布；第三方数据、技能内容与引用资料的版权归各自权利人。
- 不得将本仓库用于违反证券监管、数据供应商条款或第三方版权的用途。
- 研究结果仅供参考，不构成投资建议。
