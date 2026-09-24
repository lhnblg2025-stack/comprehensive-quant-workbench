# VERSIONING — 版本与模块解耦规范（V12.3.0 起强制）

> 核心原则：**系统版本号只存在于两处 —— `quant_system/__init__.py` 的 `__version__`
> 与 `CHANGELOG.md`。模块名、目录名、文件名、定时任务名、生成产物名一律不带版本号。**
> 版本升级只改 `__version__` + CHANGELOG，其余代码零改动。

## 1. 为什么

- 2026-08-14 前：系统升级到 V12.3.0，但分析代码目录仍叫 `analysis_v11`，前端页面
  显示 `V11.0`，云端任务叫 `quant_v11_weekly` —— 版本号散落各处，升级时极易漏改，
  造成"界面版本 ≠ 实际版本"的假象。
- 解耦后：目录统一为 `analysis_core`（与版本无关的中性名），界面版本由服务端
  注入 `{{VERSION}}`（跟随 `__version__`），升级只动一处。

## 2. 现行版本

- 当前系统版本：**V12.3.0**（`quant_system/__init__.py` `__version__`）
- 分析代码载体：`quant_system/analysis_core/`（承载 V12.x 全部分析逻辑，与版本解耦）

## 3. 升级版本时必做清单

1. `quant_system/__init__.py`：`__version__ = "新版本号"`
2. `quant_system/CHANGELOG.md`：追加新版本条目（日期 + 变更摘要）
3. `quant_system/VERSIONING.md`：更新第 2 节"现行版本" + 第 5 节对照表
4. 前端页面版本显示：**无需改动**（`server.py _serve_index_html` 自动注入 `{{VERSION}}`）
5. 涉及 API 契约变更：`quant_web/handlers/` 与前端调用同步改；旧 API 保留兼容层
6. 涉及云端任务参数：任务名保持中性（不带版本号），只改内部 `-m` 模块路径（如有）

## 4. 命名规范（新代码强制）

| 对象 | 规范 | 示例 |
|------|------|------|
| 分析模块目录 | 中性名 | `analysis_core`（禁止 analysis_v12/v13） |
| 模块文件名 | 中性名，不带版本 | `daily_report.py`（原 daily_report_v11.py） |
| 生成产物 | 中性名 | `short_term_daily.md` / `broker_profiles.json` |
| 云端定时任务 | 中性名 | `quant_weekly`（原 quant_v11_weekly） |
| 前端 JS 函数 | 中性前缀 | `LOAD_STOCK_kline` / `switchSubView`（原 V6_LOAD_/v6SwitchSub） |
| 前端页面文件 | 中性名 | `trader.html` / `research_dashboard.html` |
| API 路径版本段 | **允许**（接口契约） | `/api/v1/health`、`/api/v11/daily`（升级加新版本段，保留旧段兼容） |
| 历史变更注释 | 允许保留 | `# V11 (2026-08-07): ...` 属变更记录，可留 |

## 5. 版本 ↔ 代码载体对照（历史与现行）

| 系统版本 | 代码载体 | 说明 |
|---------|---------|------|
| V11.x | `analysis_v11/` | 2026-08-14 已 git mv 为 `analysis_core`（历史快照见 git 历史） |
| V12.x（现行） | `analysis_core/` | 与版本解耦，后续版本沿用 |
| 历史冻结前端 | `v3-system.js` / `v5-init.js` / `viz9.js` / `ui9.js` / `v4_dashboard.html` | 对应旧页面，已冻结不再迭代，保留原名不破坏历史关联 |

## 6. 改名前后的对应关系（迁移索引）

| 旧名 | 新名 | 位置 |
|------|------|------|
| `analysis_v11` | `analysis_core` | 目录 / import / 云端任务 `-m` 参数 |
| `data_sources_v11.py` | `data_sources.py` | 模块文件 |
| `daily_report_v11.py` | `daily_report.py` | 模块文件 |
| `test_data_sources_v11.py` | `test_data_sources.py` | 测试文件 |
| `short_term_daily_v11.md` | `short_term_daily.md` | generated 产物 |
| `broker_profiles_v11.{json,parquet}` | `broker_profiles.{json,parquet}` | generated 产物 |
| `v11_daily_long.png` | `daily_long.png` | generated 产物 |
| `quant_v11_weekly` | `quant_weekly` | 云端 schtask |
| `V6_LOAD_*` / `V7_LOAD_*` | `LOAD_*` | app.js 函数 |
| `v6SwitchSub` / `v6StopMonitor` / `startV5AutoRefresh` | `switchSubView` / `stopMonitor` / `startAutoRefresh` | app.js 函数 |
| `v12_trader.html` / `v12-trader.js` | `trader.html` / `trader.js` | 前端页面 |
| `v11_dashboard.html` | `research_dashboard.html` | 前端页面 |
| index.html 写死 "V11.0" | `{{VERSION}}`（服务端注入） | 界面版本显示 |
