#!/usr/bin/env python3
"""quant_platform — 平台统一入口（骨架收口层 V1）

对外唯一门面：所有调用方（前端 / cron / 脚本）只 import 本包，
背后自动路由到 quant_system（主）或 quant_v6（V8 体系）。

四层骨架：
  数据层  platform.data        → data_warehouse 唯一读取（DataStore 门面）
  分析层  platform.analysis    → 因子/回测/风险/预警统一入口
  服务层  platform.api         → 路由注册表（供 quant_web/server.py 使用）
  运行层  platform.runtime     → 任务状态机 / 审计 / 监控

用法:
  from quant_platform import data, analysis
  df = data.kline("002714", days=250)
  factors = analysis.factor_scan("002714")
"""
from __future__ import annotations

from pathlib import Path

PLATFORM_ROOT = Path(__file__).resolve().parent
WORKSPACE = PLATFORM_ROOT.parent

# 数据仓库根
WAREHOUSE = WORKSPACE / "data_warehouse"

__all__ = ["WAREHOUSE", "data", "analysis", "api", "runtime"]
