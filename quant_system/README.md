# quant_system 核心库

> 当前版本见 `quant_system/__init__.py`。
> 发布范围与排除项见仓库根目录 `DECLASSIFICATION.md`。

本目录是 A股量化研究工作台的核心 Python 库：多因子研究、回测、ML 预测、风控、
分析决策与 Web 后端支撑。

## Quick Start

```bash
# 基础扫描/回测（兼容旧 CLI）
python3 -m quant_system.cli scan --symbols 002714 601899 159792 --start 20240101
python3 -m quant_system.cli backtest --symbols 002714 601899 --start 20240101

# 本地闭环：候选表现库 + 基准/因子诊断 + 次日纸面预案
# --data 可指向一个 parquet/csv 面板或包含快照的目录
python3 -m quant_system.cli closed-loop \\
  --data ../data_warehouse/feature_store \\
  --output-dir generated/closed_loop \\
  --factors momentum_12_1 --directions momentum_12_1:1 --top-n 10

# 因子回测（通用研究协议；本版公开的策略信号见 strategy_registry.py）
python3 -m quant_system.factor_backtest_runner \\
  --data-dir ../data_warehouse/feature_store \\
  --output-dir generated/factor_backtest \\
  --factors momentum_12_1 --quantile 0.2

# 当前主分析链路
python3 -m quant_system.analysis_core.pipeline --daily
python3 -m quant_system.analysis_core.fusion --today
python3 -m quant_system.analysis_core.data_sources --cloud-status
```

## 目录

- `analysis_core/`：V12.x 主分析/作战系统（85 模块）
- `ic_factors/`：因子注册表、IC、GTJA 因子
- `factor_system/`、`deep_factors/`：因子体系
- `portfolio_diagnostics/`、`cross_market/`：持仓诊断与跨市场验证
- `market_forecast/`、`models/`：ML/预测相关
- `tests/`：核心测试

## 关键约束

- 不连接券商、不自动下单。
- 所有信号默认需要人工确认。
- 回测包含手续费和滑点参数，但仍不代表未来收益。
- 版本号只维护在 `__init__.py` + `CHANGELOG.md`，模块/目录/产物不带版本号（见 `VERSIONING.md`）。

