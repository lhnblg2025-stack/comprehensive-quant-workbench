# quant_web · 量化研究工作台 Web UI

A股全栈量化研究工作台 · 自建 Web UI + Python 后端。版本跟随 `quant_system/__init__.py`。

> 本项目仅用于研究与教育，**不自动下单**，所有交易为研究/纸面/模拟。

---

## 启动

```bash
# 一键启动（推荐，默认绑定 127.0.0.1:8600）
bash start_quant.sh

# 手动启动
python3 quant_web/server.py 8600
# 浏览器打开 http://127.0.0.1:8600
```

公网部署时必须设置随机的 `QUANT_WEB_API_KEY`，并由 Nginx 终止 HTTPS；
不要直接把后端端口暴露到公网。

## 架构

```
quant_web/
├── server.py              # 标准库 http.server 后端与 /api 路由
├── stock_analysis.py      # 个股分析/指标聚合
├── decision_contract.py   # 决策输出契约
├── handlers/              # 版本化 API、回测控制台、策略实验室
├── static/                # 前端页面、样式与脚本
└── tests/                 # 前端/API 契约测试
```

前端主要页面：

| 页面 | 文件 | 用途 |
|---|---|---|
| 总览 | `static/index.html` | 主入口与导航 |
| 回测台 | `static/backtest_console.html` | 因子/信号回放与执行审计 |
| 策略实验室 | `static/strategy_lab.html` | 运行公开策略模板 |
| 盘面 | `static/base_panorama.html` | 市场全景 |
| 复盘 | `static/review_dashboard.html` | 盘后复盘 |
| 研究 | `static/research_dashboard.html` | 研究进度与图表 |
| 知识库 | `static/knowledge_base.html` | 检索界面壳（索引需本地构建） |

## 主要 API 域

- 市场总览/行情：`/api/market`、`/api/global_quotes`、`/api/indices`、`/api/realtime`
- K线/个股：`/api/history`、`/api/stock_profile`
- 因子/IC/ML：`/api/factor_library`、`/api/factor_ic`、`/api/factor_combine`、`/api/ml_signal`
- 回测/组合/风控：`/api/backtest*`、`/api/portfolio*`、`/api/optimize`、`/api/portfolio_risk`
- 决策链：`/api/decision_chain`、`/api/v11/battle`、`/api/v11/daily`
- 策略实验室：`/api/strategy-lab/*`（仅三个公开模板）
- RAG/知识：`/api/rag_search`
- 系统/报告：`/api/reports`、`/api/data_health`、`/api/tasks_status`、`/api/system/status`

## 策略实验室

策略以受控源码字符串提交，服务端校验后执行：

- 必须定义 `build_targets(panel, params)`；
- 禁止非白名单导入（如 `import os`）；
- 禁止 `eval` / `exec`。

公开版内置 `rsi_reversal`、`low_volatility`、`momentum_12_1` 三个模板，
见 `handlers/strategy_templates.py`。

## 页面版本注入

页面版本号由服务端注入 `{{VERSION}}`，跟随 `quant_system.__version__`。

## 注意

- 系统不连接券商、不自动下单，所有信号需要人工确认。
- 缺少数据或凭据时，对应数据源应显示为不可用，而不是伪造成功。
