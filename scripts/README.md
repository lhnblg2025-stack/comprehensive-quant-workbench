# scripts/ 目录分类清单

> 骨架收口 P1-2（2026-08-07）：只分类标注，不物理移动（避免破坏其他项目引用）。
> 量化平台核心脚本见 `quant_platform/` 统一入口。

## 一、量化平台核心（常驻任务，勿动）

### 数据更新
| 脚本 | 用途 | 调度 |
|---|---|---|
| update_kline_tencent.py | K线增量（腾讯+东财兜底，断点续传） | 盘后 17:05 |
| update_valuation_baostock.py | 估值增量（baostock） | 盘后 17:05 |
| after_close_update.sh | 盘后并行启动 K线+估值 | cron 17:05 |
| realtime_snapshot.py | 盘中5分钟实时快照 | cron 每5分钟 |
| update_kline_incremental.py | 旧版K线增量（保留参考） | - |
| sync_gaps_local.py / sync_margin_sh.py / sync_retry_results.py | 补缺口/同步两融 | 手动 |
| pull_incremental.py / pull_from_server.py / fetch_all_2020.py | 服务器抓取搬运 | 服务器侧 |

### 日报/周报
| 脚本 | 用途 |
|---|---|
| a_share_daily_report.py | A股日报（主） |
| build_html_daily_report.py | HTML日报生成器 |
| daily_report_runner.py / v61_daily_push.py | 日报 runner |
| generate_*_weekly.py | 宏观/红利/黄金/生猪等周报 |
| report_common.py | D7公共口径：pct/status_line（cpi_ppi/social 收敛源）；a_share_daily_report 硬规则为真源，不在此复制 |
| weekly_report.py / run_cron_report_task.py | 周报聚合 |
| run_all.py | 全量报告运行 |

### 分析/预警
| 脚本 | 用途 |
|---|---|
| factor_extreme_alert.py | 多因子极端预警引擎（44因子） |
| quant_scan_alert.py | 盘中扫描+飞书推送 |
| market_monitor.py | 盘中监控（409只主板） |
| risk_monitor.py / opportunity_monitor.py / watch_stocks.py | 风险/机会/自选监控 |
| review_tool.py | 复盘工具 |
| stock_screener.py | 选股器 |
| daily_watchlist_scan.py | 自选扫描 |

### 基础设施
| 脚本 | 用途 |
|---|---|
| report_delivery.py | 飞书/微信/QQ投递 |
| report_paths.py | 桌面路径规则（唯一） |
| data_sources.py | 数据源封装 |
| report_data_base.py | 报告数据基类 |
| feishu_sender.py / youdao_verify.py | 投递验证 |
| dashboard.py / system_manager.py / desktop_audit.py | 系统管理 |
| normalize_valuation.py | 估值schema规范化（收口） |
| build_factor_library.py | 因子库生成 |

## 二、其他项目脚本（非量化平台，勿误删）

- **SDG13 气候面板**：apply_sdg13_* / build_sdg13_* / fix_sdg13_* / merge_sdg13_* / search_sdg13_*
- **耳机演示**：create_headphone_presentation*.py
- **CCRD**：ccrd_scraper.py / merge_ccrd_docx.py
- **其他一次性**：check_macro_data_sources.py / gold_data_extender.py / industry_alt_data.py / report_data_source_probe.py / skills_analysis.py
- **已退役（2026-08-08 移入 垃圾文件夹/脚本-legacy/）**：quant_v6_full_verify.py / v61_daily_push.py / warehouse_next_batch.sh / remote_start_verify.sh / setup_domestic_cloud.sh / mutation_test.py（mutation 工具待重指向 quant_system）

## 三、收口后新代码规范

1. 新功能先写 `quant_platform/` 统一入口，再落 scripts/
2. 数据读取一律走 `quant_platform.data`（→ DataStore 门面）
3. 分析调用一律走 `quant_platform.analysis`
4. 常驻任务必须有 `is_trading_time()` 守卫 + CpuThrottle + os.nice(10)
5. 日志统一 /tmp/quant_logs/ 或 logs/
