# 发布范围 · Release Scope

本仓库是 A股量化研究工作台的公开脱密版本。脱密原则、排除清单与残留风险的完整说明见
根目录 [`DECLASSIFICATION.md`](../DECLASSIFICATION.md)；本文只记录**范围边界**。

## 公开内容

- **研究核心**：`quant_system/`（因子、回测、指标、风控、研究协议、数据门禁、准入框架）
- **平台门面**：`quant_platform/`
- **后端与前端**：`quant_web/`（标准库 HTTP 服务、API 处理器、多页面前端）
- **脚本**：`scripts/`（数据、报告、审计、运维）
- **配置模板**：`config/`（仅占位符与环境变量引用）
- **部署**：`deploy/`（systemd、Nginx 模板、安装与验收脚本）
- **技能包**：`openclaw-financial-services/skills/`
- **RAG 检索代码**：`quant_system/analysis_core/knowledge_rag.py`
- **测试**：`tests/`、`quant_system/tests/`、`quant_web/tests/`、`scripts/tests/`
- **公开策略**：仅 `rsi_reversal`、`low_volatility`、`momentum_12_1`

## 明确未纳入

| 类别 | 说明 |
|---|---|
| 凭据 | API key、token、cookie、密码、SSH/云私钥、`.env`、`.pem`、`.key` |
| 运行数据 | Parquet 仓库、数据库、缓存、日志、回测明细与审计转储 |
| 个人信息 | 账户、持仓、自选股、订单记录、聊天账号标识、个人绝对路径 |
| 恢复目录 | 本地恢复/云同步/历史归档目录、嵌套私有 Git 历史 |
| 私有策略 | 具体策略实现与参数、私有因子清单、私有研究活动配置 |
| 知识库 | 个人知识库导出、第三方书籍摘要、向量索引与缓存 |
| 模型 | 训练权重与大型中间产物 |

## 三个公开策略的定位

三者均以固定 300 标的研究代理样本运行，数据门禁为 `DATA_BLOCKED / research_only`，
**没有通过正式准入**。公开的是可复现的研究模板与其研究代理结果摘要
（`research/strategy_results.csv`），不是实盘策略，也不构成投资建议。

## 检查方法

1. `.git/info/exclude` 从索引层隔离凭据、恢复目录与运行数据；
2. `python3 scripts/check_release.py` 扫描已跟踪文件中的私钥块、主流 token、
   硬编码凭据与本机路径；
3. `make lint-config` 编译全部 Python，捕捉清理导致的语法/导入错误；
4. 静态扫描已删除模块的引用，修复断链；
5. `make contract` 运行全量非集成测试；
6. 人工逐类审阅发布文本——自动规则**不能**证明不存在所有形式的敏感信息，
   尤其不能单凭文件名判断策略敏感性。

## 本次修复

公开脱密过程中修复了两处会影响公开版可用性的问题：

- `quant_system/tests/test_factor_backtest_runner.py` 使用顶层导入，
  在 `--import-mode=importlib` 下无法收集；改为包内绝对导入。
- `quant_system/factor_backtest_runner._validate_parameters` 把 OOS 比例硬绑定到
  5:3 研究协议（`research_protocol_requires_5_3_train_test_split`），
  使通用因子回测器无法接受其它合法切分；已放宽为 `0 < oos_ratio < 1`，
  协议常量仍作为默认值保留。
