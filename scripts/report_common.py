#!/usr/bin/env python3
"""report_common — 报告域公共格式化/状态函数（D7 保守收敛）。

D7 收敛说明:
- pct(): 并集超集。来源 generate_cpi_ppi_report.pct（仅 float）与
  generate_social_finance_m2_report.pct（float|int|None）；对非 None 数值输出
  逐字节一致（f"{float(x):.1f}%"），None → "未取得" 保留 social 原分支。
- status_line(): generate_cpi_ppi_report.status_line 与
  generate_social_finance_m2_report.status_line 为逐字节相同实现，收敛于此。

硬规则声明: a_share_daily_report.py 的固定口径（热股等权/全A等权/去ST/融资TMT/
量能/风格）是唯一真源，本模块不复制、不覆盖这些口径，禁止在此新增同名替代。
"""
from __future__ import annotations

from typing import Any


def pct(x: Any) -> str:
    """百分比格式化：非 None 数值输出 f"{float(x):.1f}%"，None 输出 "未取得"。"""
    return "未取得" if x is None else f"{float(x):.1f}%"


def status_line(result: Any) -> str:
    """SourceResult 状态行：成功/失败 + 来源 + 说明（与 macro 系列原实现一致）。"""
    state = "成功" if result.ok else "失败"
    return f"{result.name}：{state}；来源：{result.source or '无'}；说明：{result.note}"
