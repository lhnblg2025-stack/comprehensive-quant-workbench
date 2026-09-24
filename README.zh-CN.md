# comprehensive-quant-workbench｜A 股量化研究工作台

> **面向研究团队的可复核量化工作台：数据门禁、因子研究、策略回测、风险分析、浏览器工作台、技能与 RAG 适配器。**

[English](README.md) · **简体中文** · [产品与合规说明](FULL_PUBLIC_RELEASE_README.md) · [脱密发布说明](DECLASSIFICATION_NOTICE_FULL.md)

[![CI](https://github.com/lhnblg2025-stack/comprehensive-quant-workbench/actions/workflows/ci.yml/badge.svg)](https://github.com/lhnblg2025-stack/comprehensive-quant-workbench/actions/workflows/ci.yml)
[![公开版](https://img.shields.io/badge/版本-脱密公开版-2f855a.svg)](DECLASSIFICATION_NOTICE_FULL.md)

## 项目是什么

`comprehensive-quant-workbench` 是一个研究型产品与工程框架，用来把用户授权的数据转换为可检查、可复现、可审计的研究结果。它覆盖从数据质量检查、因子与市场状态分析，到策略信号、组合模拟、风险诊断、报告生成和浏览器复核的完整链路。

它适合：

- 量化研究人员搭建和比较研究假设；
- 数据科学家验证因子、信号和组合构建方法；
- 工程团队开发带有数据契约、审计台账和发布门禁的研究服务；
- 教学、代码审阅和可复现实验。

它不是投资顾问、资产管理、证券经纪或自动交易服务，也不提供收益承诺。

## 产品能力

| 模块 | 能力 |
| --- | --- |
| 数据门禁 | schema、日期、缺失值、重复值、新鲜度和质量检查 |
| 研究引擎 | 因子、市场状态、信号、组合构建和研究管线 |
| 审计回测 | 成本、滑点、换手、流动性、涨跌停、停牌、交易台账和回撤 |
| 浏览器工作台 | 研究总览、策略实验、回测控制台、组合、风险和运营页面 |
| 策略框架 | 注册表、策略接口、准入规则、政策层和可扩展引擎 |
| 技能与 RAG | 金融研究工作流技能、可插拔检索适配器和用户自建索引 |
| 部署 | 本地服务、systemd、定时任务、Nginx 和环境变量模板 |

## 系统结构

```text
用户授权数据
    ↓
数据门禁与质量检查
    ↓
因子 / 市场状态 / 信号研究
    ↓
策略注册表与准入策略
    ↓
审计型组合回测
    ↓
JSON / CSV / 报告工件
    ↓
quant_web 浏览器工作台
```

主要目录：

- `quant_system/`：研究、因子、信号、组合、风险与执行抽象；
- `quant_web/`：Python Web 服务、路由、响应契约和前端页面；
- `quant_platform/`：平台审计和集成辅助模块；
- `scripts/`：数据准备、报告、运维和公开发布扫描器；
- `skills/`、`openclaw-financial-services/`：可复用研究技能；
- `rag/`：RAG 接口和适配边界；
- `deploy/`：服务和部署模板；
- `tests/`：引擎、契约、前端和回归测试。

## 公开策略

公开版本只提供三个策略键：

| 策略键 | 信号列 | 研究含义 |
| --- | --- | --- |
| `rsi_reversal` | `rsi_rev_14` | RSI(14) 均值回归信号 |
| `low_volatility` | `low_vol_60` | 60 个交易日低波动信号 |
| `momentum_12_1` | `mom_12_1` | 跳过最近一个月的 12—1 月动量信号 |

通用策略引擎、数据契约、组合模拟和准入框架仍然保留；具体私有策略实现、私有参数和私有研究结果已从公开树移除。

## 快速开始

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q tests quant_system/tests quant_web/tests scripts/tests -m 'not integration'
bash start_quant.sh
```

启动后访问终端输出的本地地址。运行发布检查：

```bash
python3 -m compileall -q quant_system quant_platform quant_web scripts tests
python3 scripts/check_release.py
```

## 合规与责任边界

本项目仅用于软件开发、量化研究、教学和内部验证。策略、信号、图表和回测结果不构成投资建议、证券推荐、收益承诺或未来表现预测。

使用者必须确认数据、新闻、研究报告、书籍和 RAG 来源具有合法授权，并遵守隐私、信息安全、第三方许可、适当性、模型风险和记录留存要求。不得提交客户信息、账户状态、交易凭据、生产日志或私有研究资料。

真实交易或受监管业务必须经过法律、合规、信息安全和风险审批，并配置访问控制、交易限额、熔断、审计留痕、人工复核和回滚机制。回测可能受到前视偏差、幸存者偏差、数据修订、滑点、流动性、过拟合和市场制度变化影响。

完整产品介绍与合规说明：

- [产品与合规说明](FULL_PUBLIC_RELEASE_README.md)
- [脱密发布说明](DECLASSIFICATION_NOTICE_FULL.md)
- [发布范围](docs/RELEASE_SCOPE.md)
- [部署说明](deploy/README.md)
- [RAG 公开版说明](RAG_PUBLIC_RELEASE.md)

## 许可证与责任

使用前请审阅仓库及依赖项许可证。项目维护者不对使用者的数据、部署、交易决策、监管义务或第三方服务承担责任。
