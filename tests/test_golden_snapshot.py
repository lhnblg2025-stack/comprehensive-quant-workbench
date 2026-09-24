"""
test_golden_snapshot.py — 黄金快照测试（V11 项6）

冻结标杆交易日 2026-08-06 的日报关键指标 + IC 报告头部，
任何代码修改若导致核心输出数值漂移（>容差），测试失败，需人工确认。

用途：防止"有输出但错误"——审计只查崩溃，黄金快照查数值合理性。
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DESKTOP_DAILY = Path(os.environ.get("QUANT_DESKTOP", str(Path.home() / "Desktop"))) / "每日任务" / "A股量化日报"
IC_REPORT = ROOT / "generated" / "ic_report" / "FACTOR_IC_REPORT_VECTORIZED.csv"

# ── 黄金快照：2026-08-06 日报关键指标 ──
# 基准来源：云端恢复的原始日报（SHA256:
# 9e5707cd69a6e06b60b7b7459baa452d8ab0d9f94ce97baa8ddcf94a6565190a）。
# 旧基准来自另一版生成结果，已与可审计恢复文件对齐。
GOLDEN_DAILY = {
    # 赚钱效应（热股等权/全A等权/上涨占比）
    "hot_equal_ret": 2.61,   # 热股等权涨跌幅 %
    "all_equal_ret": 0.55,   # 全A等权涨跌幅 %
    "up_ratio": 50.4,        # 过滤后上涨占比 %
    # 宽度
    "up_count": 2519,
    "down_count": 2342,
    "limit_up": 80,          # 近似涨停
    "limit_down": 3,         # 近似跌停
    # 量能与融资
    "amount_yi": 25153.7,    # 过滤后成交额 亿
    "margin_total": 25962.6, # 全A融资 亿
    "margin_tmt": 8696.4,    # TMT融资 亿
}

TOL = 0.05  # 相对容差 5%


def _extract_daily(path: Path) -> dict[str, float]:
    """从日报 markdown 提取关键数值。"""
    txt = path.read_text(encoding="utf-8")
    out: dict[str, float] = {}
    m = re.search(r"热股等权\s*([-\d.]+)%", txt)
    if m:
        out["hot_equal_ret"] = float(m.group(1))
    m = re.search(r"全A等权\s*([-\d.]+)%", txt)
    if m:
        out["all_equal_ret"] = float(m.group(1))
    m = re.search(r"上涨占比\s*([\d.]+)%", txt)
    if m:
        out["up_ratio"] = float(m.group(1))
    m = re.search(r"上涨\s*(\d+)、下跌\s*(\d+)", txt)
    if m:
        out["up_count"] = float(m.group(1))
        out["down_count"] = float(m.group(2))
    m = re.search(r"近似涨停\s*(\d+)、近似跌停\s*(\d+)", txt)
    if m:
        out["limit_up"] = float(m.group(1))
        out["limit_down"] = float(m.group(2))
    m = re.search(r"成交额约\s*([\d.]+)\s*亿元", txt)
    if m:
        out["amount_yi"] = float(m.group(1))
    m = re.search(r"全A融资\s*([\d.]+)亿元", txt)
    if m:
        out["margin_total"] = float(m.group(1))
    m = re.search(r"TMT融资\s*([\d.]+)亿元", txt)
    if m:
        out["margin_tmt"] = float(m.group(1))
    return out


def test_golden_daily_20260806():
    """黄金快照：2026-08-06 日报关键指标不得漂移。"""
    p = DESKTOP_DAILY / "2026-08-06-A股量化日报.md"
    if not p.exists():
        pytest.skip("黄金日报文件不存在，跳过")
    got = _extract_daily(p)
    for key, golden in GOLDEN_DAILY.items():
        if key not in got:
            pytest.fail(f"日报缺少关键指标 {key}（可能日报结构已改）")
        ratio = abs(got[key] - golden) / max(abs(golden), 1e-9)
        assert ratio <= TOL, (
            f"黄金指标漂移 {key}: 冻结={golden} 当前={got[key]} "
            f"(偏差 {ratio*100:.1f}% > {TOL*100:.0f}%)——代码改动需人工确认"
        )


def test_golden_ic_report_top3():
    """黄金快照：IC 报告头部 3 个因子不得漂移。"""
    if not IC_REPORT.exists():
        pytest.skip("IC 报告不存在（IC 未重跑），跳过")
    import pandas as pd
    df = pd.read_csv(IC_REPORT)
    assert len(df) > 0, "IC 报告为空"
    # 冻结：前 3 个因子名称 + IC 值（2026-08-06 快照）
    golden = {
        "amount_liquidity": -0.0944,
        "atr_ratio": -0.093,
        "tech_atr_pct": -0.093,
    }
    top = df.head(3)
    for _, row in top.iterrows():
        name = row["factor"]
        if name in golden:
            ratio = abs(float(row["ic_mean"]) - golden[name]) / max(abs(golden[name]), 1e-9)
            assert ratio <= TOL, (
                f"IC 漂移 {name}: 冻结={golden[name]} 当前={row['ic_mean']}"
            )


def test_golden_registry_count():
    """黄金快照：registry 因子注册规模（ensure_loaded 生产口径）不得大幅漂移。

    2026-08-14 更新: GTJA 国泰君安 Alpha 因子移植 18 个 + import_from_zoo 全量
    同步 → 冻结值 190 → 252（生产口径 ensure_loaded 实测, 人工确认）。
    """
    import sys
    sys.path.insert(0, str(ROOT))
    from quant_system.ic_factors import registry
    # 强制 autodiscover: ensure_loaded 只在 _REGISTRY 空时扫描, 测试顺序下
    # 其他模块(如 market_level_factors)提前注册会跳过全量, 故显式扫描
    registry.autodiscover()
    registry.import_from_zoo()
    registry.ensure_loaded()  # autodiscover + import_from_zoo + 方向覆盖(幂等)
    n = len(registry.list_factors(active_only=False))
    # 冻结 299（2026-08-15 V12.3: 市场级 29 + K线横截面 18 因子已全部注册进 registry,
    # 合计 = 原 281 注册因子 + 18 个 kline_extra 因子补齐注册）；
    # 允许 ±12（新增/停用因子需人工确认）
    assert abs(n - 299) <= 12, (
        f"注册表因子数漂移: 冻结=299 当前={n}——新增/停用因子需人工确认"
    )
