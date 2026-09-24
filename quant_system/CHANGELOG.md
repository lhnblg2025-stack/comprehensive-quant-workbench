# CHANGELOG

> 版本号以 `quant_system/__init__.py` 的 `__version__` 为准；本文件记录各版本变更。
> 命名规范见 `VERSIONING.md`（模块/文件/任务/产物名与版本解耦）。

## V13.0.0（2026-08-20）

### 新增：研报工作流（research_flow）
- 新增 `quant_system/analysis_core/research_flow.py`：研报 NLP + 图片 OCR + 产业链/龙头融合
  - 研报接入：本地 `data_warehouse/research_reports/` + 有道云笔记 MCP（`ingest_youdao_notes` / `ingest_text`）
  - 研报 NLP：提取股票代码、公司、概念、评级、目标价、产业链层（上/中/下游）、龙头标记、催化事件
  - 图片文字识别：复用 `ocr_util`（tesseract），处理研报截图/新闻图片
  - 融合：研报概念与 `chain_map`（产业链温度）+ `leader_follower`（短线龙头扩散）求交集
  - 落盘 `generated/research_flow_{date}.json`
- `pipeline.py`：新增 step 11.5 研报工作流，输出研报/概念/链命中/龙头命中计数
- `battle_map.py`：研报链命中并入产业链联动，`bm["research"]` 暴露研报摘要
- 新增 `tests/test_research_flow.py`（4 用例）

### 全面审计修改（承接 2026-08-16 系列）
- ML 分组时序 CV、feature_store 样本外回填、回测缺失行情不填 0、配置环境变量化等
  （详见 git 历史 8ade5ee/654737b/8da480d）

## V12.3.0（2026-08-14）

### 命名解耦（本版本重点）
- 分析目录 `analysis_v11` → `analysis_core`（与版本解耦，89 处 import / 云端任务参数全量替换）
- 模块文件：`data_sources_v11.py` → `data_sources.py`；`daily_report_v11.py` → `daily_report.py`
- 产物：`short_term_daily_v11.md` → `short_term_daily.md`；`broker_profiles_v11.*` → `broker_profiles.*`；`v11_daily_long.png` → `daily_long.png`
- 云端任务：`quant_v11_weekly` → `quant_weekly`
- 前端：`V6_LOAD_*`/`V7_LOAD_*` → `LOAD_*`；`v6SwitchSub/v6StopMonitor/startV5AutoRefresh` → 中性名；
  `v12_trader.html`/`v12-trader.js` → `trader.html`/`trader.js`；`v11_dashboard.html` → `research_dashboard.html`
- 界面版本显示：index.html 写死的 "V11.0" → `{{VERSION}}` 服务端注入（跟随 `__version__`）
- 新增 `VERSIONING.md`：版本与模块解耦规范（升级只改 `__version__` + CHANGELOG）

### 功能
- 前端 🧭 决策链+RAG 页：盘中五视角 / 盘后龙虎榜+低估池 / 作战地图 + 知识库检索
- 后端 `/api/decision_chain`（mode=intraday|after_close|battle_map）+ `/api/rag_search`
- API 契约测试 +5（决策链 3 + RAG 2）

### 修复
- 云端 quant_web 重启后"连接被强制关闭"：ssh 前台启动残留进程 stderr 管道已死 →
  `start_quant_web.bat` 增加 `>> logs\web_server.log 2>&1` 重定向 + schtasks 分离启动

## V12.2（历史）
- （详见 git 历史 / 审计报告 V12.3_20260814）
