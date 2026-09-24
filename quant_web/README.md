# 🐚 牧云天枢 · 量化系统 Web UI

> 当前系统版本跟随 `quant_system/__init__.py`：**V12.3.0**
> 本 README 于 2026-08-16 整理，替代旧 V6.0 说明。

代号：**天枢**（北斗七星之首，喻智慧与方向）

A股全栈量化工作站 · 自建 Web UI + Python 后端。

---

## 启动

```bash
# 一键启动（推荐）
bash ${PROJECT_ROOT}/start_quant.sh

# 手动启动
cd /path/to/comprehensive-quant-workbench
python3 quant_web/server.py 8600
# 浏览器打开 http://127.0.0.1:8600
```

## 当前架构

```
quant_web/
├── server.py              # HTTP 后端（V12.x，大量 /api 端点）
├── stock_analysis.py      # 个股分析/指标聚合
├── AUDIT_FINDINGS.md      # 前端-后端契约审计发现
├── handlers/              # 部分版本化 API（v11/risk 等）
├── static/
│   ├── index.html         # 主页面
│   ├── app.js             # 前端主逻辑
│   ├── trader.js          # 交易/持仓页面逻辑
│   ├── research_dashboard.html
│   ├── task-dashboard.js
│   ├── styles.css
│   └── ...
└── tests/                 # 前端/API 契约测试
```

## 主要 API 域

- 市场总览/行情：`/api/market`、`/api/global_quotes`、`/api/indices`、`/api/realtime`
- K线/个股：`/api/history`、`/api/v12/kline`、`/api/stock_profile`、`/api/stock_lens`
- 因子/IC/ML：`/api/factor_library`、`/api/factor_ic`、`/api/factor_combine`、`/api/ml_signal`
- 回测/组合/风控：`/api/backtest*`、`/api/portfolio*`、`/api/optimize`、`/api/portfolio_risk`
- V11/V12 决策链：`/api/decision_chain`、`/api/v11/battle`、`/api/v11/daily`
- RAG/知识：`/api/rag_search`
- 研报工作流：`/api/research_flow`（研报NLP/OCR + 产业链/龙头融合）
- 系统/报告：`/api/reports`、`/api/data_health`、`/api/tasks_status`、`/api/system/status`

## 数据源

| 数据 | 来源 | 状态 |
|------|------|------|
| A股日线 | akshare / 本地 Parquet / 腾讯 | ✅ |
| A股实时行情 | 新浪/腾讯 | ✅ |
| 板块/行业 | 东方财富 / THS / 申万 | ✅（有降级链） |
| 分钟K线 | 东方财富 / 腾讯 | ⚠️ 偶发502 |
| 港股/美股 | 新浪 | ✅ |
| 融资融券/北向 | akshare | ✅ |
| 市场温度/情绪 | 本地融合/云快照 | ✅ |

## 审计状态（2026-08-15）

- 已做前端-后端契约复审，修复多 P1 契约断裂（factor_ic/factor_combine/news_sentiment/reports/global_quotes 等）。
- 已知 P2/设计改造项见 `AUDIT_FINDINGS.md`。
- 页面版本号由服务端注入 `{{VERSION}}`，跟随 `quant_system.__version__`。

## 注意

- 当前系统**不自动下单**，所有交易为研究/纸面/模拟。
- 运行中的 quant_web 被云端同步 Move 时可能卡住；同步脚本已做有界 Move + SKIPPED 处理。

