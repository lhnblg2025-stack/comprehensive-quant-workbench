# A股量化研究工作台 · Comprehensive Quant Workbench

一套可自托管的 A 股量化**研究工作台**：Python 研究核心库 + 自建 Web 前端 + 运维/部署脚本。

> ⚠️ **本项目仅用于研究与教育。** 系统不连接券商、不自动下单，所有信号都需要人工确认。
> 历史结果不代表未来收益，也不构成投资建议。

本仓库是经过**脱密处理的公开版本**，只包含工作台框架与**三个**筛选后公开的策略。
发布范围、排除项与已知限制见 [`DECLASSIFICATION.md`](DECLASSIFICATION.md)。

---

## 目录

- [特性概览](#特性概览)
- [公开策略（仅三个）](#公开策略仅三个)
- [仓库结构](#仓库结构)
- [安装](#安装)
- [快速开始](#快速开始)
- [Web 工作台](#web-工作台)
- [策略实验室](#策略实验室)
- [知识库与 RAG](#知识库与-rag)
- [部署](#部署)
- [测试与发布检查](#测试与发布检查)
- [数据源与凭据](#数据源与凭据)
- [发布范围与排除项](#发布范围与排除项)
- [许可](#许可)

---

## 特性概览

| 层 | 模块 | 说明 |
|---|---|---|
| 研究核心 | `quant_system/` | 因子、回测、指标、风控、研究协议、数据门禁 |
| 平台门面 | `quant_platform/` | 数据层 / 分析层 / 服务层 / 运行层统一入口 |
| 后端服务 | `quant_web/` | 零依赖 `http.server` 实现的 JSON API + 静态页面服务 |
| 前端 | `quant_web/static/` | 多页面工作站（回测台、策略实验室、盘面、复盘、知识库壳） |
| 脚本 | `scripts/` | 数据、报告、审计、运维脚本 |
| 配置 | `config/` | 数据源、策略门禁、定时任务与投递配置（**仅占位符**） |
| 部署 | `deploy/` | systemd 单元、Nginx 模板、安装/验收脚本 |
| 技能 | `openclaw-financial-services/skills/` | 金融分析技能包（SKILL.md 规范） |
| 测试 | `tests/`, `quant_system/tests/`, `quant_web/tests/`, `scripts/tests/` | 契约、回归与数据门禁测试 |

设计要点：

- **零重型 Web 依赖**：后端基于标准库 `http.server`，便于内网自托管与审计。
- **显式成本模型**：回测包含佣金、印花税、过户费与滑点，不使用零成本乐观成交。
- **T+1 语义**：持仓次日生效，避免"当日买当日卖获利"的前视偏差。
- **数据门禁优先**：缺少权威交易状态 / 生命周期 / 容量数据时保持 `DATA_BLOCKED`，不伪造成功。
- **合成数据可离线运行**：`quant_platform.demo` 可在禁网环境下跑通完整回测。

---

## 公开策略（仅三个）

本公开版**只发布以下三个策略**。三者都以"研究代理样本"运行，数据门禁为
`DATA_BLOCKED / research_only`，**均未通过正式准入**。

| ID | 名称 | 逻辑 | 参数 |
|---|---|---|---|
| `rsi_reversal` | RSI 反转 | 选 14 日 RSI 较低的超卖标的 | `top_n=20`, 每 10 个交易日再平衡 |
| `low_volatility` | 低波动 | 选 60 日波动率较低的标的 | `top_n=20`, 每 20 个交易日再平衡 |
| `momentum_12_1` | 12-1 月动量 | 跳过最近一个月的经典横截面动量 | `top_n=15`, 每 63 个交易日再平衡 |

策略模板实现在 [`quant_web/handlers/strategy_templates.py`](quant_web/handlers/strategy_templates.py)，
策略家族目录在 [`quant_system/strategy_registry.py`](quant_system/strategy_registry.py)。

### 研究代理结果摘要

样本区间 **2021-08-31 – 2026-08-31**，固定 300 标的研究代理样本，按工作台记录的交易成本与执行规则计算：

| 策略 | 总收益 | 年化 | 基准年化 | 超额年化 | Sharpe | 最大回撤 |
|---|---:|---:|---:|---:|---:|---:|
| `rsi_reversal` | 50.03% | 15.19% | 5.52% | +9.16% | 0.61 | -29.85% |
| `momentum_12_1` | 31.92% | 10.14% | 5.52% | +4.38% | 0.41 | -33.77% |
| `low_volatility` | 24.08% | 7.81% | 5.52% | +2.17% | 0.41 | -13.38% |

数据来源：[`research/strategy_results.csv`](research/strategy_results.csv)。

> **必须同时阅读的限制**：样本为固定研究代理池，缺少权威历史交易状态与生命周期数据，
> 存在幸存者偏差与代理字段误差；基准为研究口径，非实盘可比。上述数字**不是实盘业绩**，
> 也不构成任何收益预期。

---

## 仓库结构

```
.
├── quant_system/              # 研究核心库（版本见 __init__.py）
│   ├── analysis_core/         # 分析/决策/融合主链路（含 knowledge_rag.py）
│   ├── factor_system/         # 因子体系
│   ├── ic_factors/            # 因子注册表与 IC
│   ├── market_forecast/       # ML / 预测
│   ├── portfolio_diagnostics/ # 组合诊断
│   ├── configs/               # 引擎配置（私有研究活动配置未纳入）
│   └── tests/
├── quant_platform/            # 平台门面：data / analysis / runtime
├── quant_web/                 # Web 后端 + 前端
│   ├── server.py              # 标准库 HTTP 服务与 API 路由
│   ├── handlers/              # 版本化 API 与策略实验室
│   ├── static/                # 前端页面、样式与脚本
│   └── tests/
├── scripts/                   # 数据 / 报告 / 审计 / 运维脚本
├── config/                    # 配置（仅占位符，无真实凭据）
├── deploy/                    # systemd / nginx / 安装与验收脚本
├── openclaw-financial-services/skills/   # 金融技能包
├── docs/RELEASE_SCOPE.md      # 发布范围
├── research/                  # 公开策略结果摘要
├── reports/                   # 报告说明
├── tests/                     # 顶层契约与回归测试
├── DECLASSIFICATION.md        # 脱密说明（必读）
├── Makefile                   # lint / fast / contract / ci
├── pytest.ini                 # 测试发现与标记
└── conftest.py                # 缺少运行资产时自动跳过相关测试
```

---

## 安装

需要 **Python 3.10+**。

```bash
git clone https://github.com/lhnblg2025-stack/comprehensive-quant-workbench.git
cd comprehensive-quant-workbench

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock.txt      # 完整运行依赖
# 或仅跑合成数据演示：
pip install -r requirements.txt
```

依赖分三层：

| 文件 | 用途 |
|---|---|
| `requirements.txt` | 最小集合（numpy / pandas / scipy），可跑合成演示与核心测试 |
| `requirements.lock.txt` | 完整运行依赖（含 akshare、pyarrow、matplotlib、plotly 等） |
| `requirements-dev.txt` | 开发工具（pytest、ruff、mypy） |

---

## 快速开始

### 1. 离线合成数据演示（不需要任何数据或网络）

```bash
python3 -c "from quant_platform.demo import run; print(run()['result']['status'])"
```

### 2. 核心回测 CLI

```bash
python3 -m quant_system.cli scan     --symbols 002714 601899 --start 20240101
python3 -m quant_system.cli backtest --symbols 002714 601899 --start 20240101
```

### 3. 因子回测（研究协议 5:3 切分）

```bash
python3 -m quant_system.factor_backtest_runner \
  --data-dir data_warehouse/feature_store \
  --output-dir generated/factor_backtest \
  --factors momentum_12_1 --quantile 0.2
```

### 4. 市场数据源自检

```bash
PYTHONPATH=. python3 scripts/check_market_data_providers.py
```

行情读取统一走 `quant_system.market_data_providers`：`cloud_eastmoney`（东方财富历史 K 线）、
`local_akshare`（本机 akshare 补缺）、`yahoo_chart`（可覆盖标的的回退）。

### 5. 启动 Web 工作台

```bash
bash start_quant.sh            # 默认 http://127.0.0.1:8600
```

---

## Web 工作台

后端由 `quant_web/server.py` 提供，零第三方 Web 框架依赖。主要页面：

| 页面 | 路径 | 用途 |
|---|---|---|
| 总览 | `/index.html` | 主入口与导航 |
| 回测台 | `/backtest_console.html` | 因子/信号回放、成本与阻断审计 |
| 策略实验室 | `/strategy_lab.html` | 编辑/运行策略模板，产出回测与年度表现 |
| 盘面 | `/base_panorama.html` | 市场全景 |
| 复盘 | `/review_dashboard.html` | 盘后复盘 |
| 研究 | `/research_dashboard.html` | 研究进度与图表 |
| 知识库 | `/knowledge_base.html` | 检索界面壳（数据需本地构建，见下） |

服务默认仅绑定本机回环地址。公网部署时请设置 `QUANT_WEB_API_KEY` 并由 Nginx 终止 HTTPS，
不要直接暴露后端端口。

---

## 策略实验室

策略以**受控源码字符串**形式提交，服务端会校验后再执行：

- 必须定义 `build_targets(panel, params)`；
- 禁止 `import os` 等非白名单导入；
- 禁止 `eval` / `exec` 等动态执行。

```python
# build_targets 契约
def build_targets(panel, params):
    # panel: DataFrame[code, date, raw_close, raw_high, volume, amount, ...]
    # 返回: {Timestamp: {code: weight}}
    ...
```

公开版内置上述三个模板。私有策略实现、私有参数与私有研究活动配置**不在本仓库内**。

---

## 知识库与 RAG

- 检索/融合逻辑：`quant_system/analysis_core/knowledge_rag.py`
- 前端检索界面：`quant_web/static/knowledge_base.html`（**壳**，向量数据不含在本仓库）

公开版**不包含**任何个人知识库导出、书籍摘要或第三方版权资料。请在本地按自己的资料重建索引：

```bash
# 示例：将自有 markdown 资料放入 skills/ 后重建本地索引
python3 -c "from quant_system.analysis_core import knowledge_rag as k; print(k.__doc__)"
```

技能包采用 `SKILL.md` 规范，示例见 `openclaw-financial-services/skills/`。

---

## 部署

`deploy/` 内含腾讯云 CVM / 通用 Linux 的常驻部署资产：

| 文件 | 用途 |
|---|---|
| `install_tencent_cloud.sh` | 一键安装（系统用户、venv、systemd、timer） |
| `quant-web.service` / `quant-web-vultr.service` | Web 服务单元 |
| `quant-after-close.timer` / `.service` | 盘后流水线定时任务 |
| `nginx-quant.conf.example` | Nginx HTTPS 反向代理模板 |
| `quant.env.example` | 环境变量模板（**仅占位符**） |
| `verify_unattended.sh` | 无人值守验收 |
| `README.md` | 部署步骤与凭据约定 |

真实密钥只写入 `/etc/quant/quant.env`（权限 `0600`），**不得提交到仓库**。

---

## 测试与发布检查

```bash
make fast        # 快速契约测试
make contract    # 全量非集成测试
make lint-config # 编译检查
make ci          # lint-config + fast + contract
```

当前基线（干净克隆，无运行数据）：

- **1832 项测试可收集；1762 通过、70 跳过、0 失败**
- 在有完整 `data_warehouse/` 与 `generated/` 的机器上：1830 通过、2 跳过、0 失败

需要运行资产（Parquet 数据仓库、生成产物）的测试由 `conftest.py` 自动跳过，
并给出 `runtime asset missing: <path>` 的明确原因，而不是让构建变红。
新增依赖运行资产的测试时，请在 `conftest.py` 的 `ASSET_TESTS` 中登记对应路径。

发布前做一次脱密扫描：

```bash
python3 scripts/check_release.py     # 检查已跟踪文件中的密钥/PII/异常文件
```

> 自动规则不能证明不存在所有形式的敏感信息。`scripts/check_release.py` 只是辅助门禁，
> 仍需人工审阅。详见 [`DECLASSIFICATION.md`](DECLASSIFICATION.md)。

---

## 数据源与凭据

所有凭据均通过**环境变量**注入，仓库内只有占位符：

```bash
# config/.env.secrets.example -> 复制为本地密钥文件（已被 .gitignore 忽略）
TUSHARE_API_KEY=CHANGE_ME
FINNHUB_API_KEY=CHANGE_ME
YOUDAO_API_KEY=CHANGE_ME
WECHAT_ACCOUNT=
QUANT_WEB_API_KEY=            # 公网部署必设随机值
```

配置文件用 `env:NAME` 语法引用环境变量，例如 `config/private_data_sources.json`。
缺少可选凭据时，对应数据源应显示为不可用，而不是伪造成功。

使用数据源前请自行确认供应商许可与合规要求。

---

## 发布范围与排除项

本公开版**刻意排除**：

- **凭据**：API key、token、cookie、密码、SSH/云私钥、`.env`、`.pem`、`.key`；
- **运行数据**：Parquet/行情财务数据仓、`.db`/`.sqlite*`、缓存、日志、回测明细产物；
- **个人信息**：账户/持仓/自选股、订单与交易记录、聊天账号标识、个人绝对路径；
- **敏感恢复目录**：本地恢复/云同步/历史归档目录与嵌套私有 Git 历史；
- **私有研究资产**：具体策略实现与参数、私有因子清单、研究活动配置、模型权重、
  个人知识库导出与第三方书籍摘要。

完整清单、方法与残留风险见 [`DECLASSIFICATION.md`](DECLASSIFICATION.md) 与
[`docs/RELEASE_SCOPE.md`](docs/RELEASE_SCOPE.md)。

---

## 许可

代码以 [MIT](LICENSE) 发布。第三方数据、技能包内容与引用资料的版权归各自权利人所有；
使用前请自行确认许可。研究报告（如有）单独标注许可。
