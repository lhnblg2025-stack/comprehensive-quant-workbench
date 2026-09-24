#!/usr/bin/env python3
"""factor_health_monitor.py — P1-2 因子拥挤度 / 滚动 IC 失效监控（每日）

审计_因子层.md P1-2：现有失效监控缺 滚动 IC+t 值、因子收益波动率（拥挤）、
同类因子相关均值、多空组合换手率。本脚本每日对每个因子输出这四项指标，
落盘到 data_cache/factor_health/，并在 --alert-threshold 超出时返回/打印告警，
方便接入 health_guardian。

用法:
  python3 scripts/factor_health_monitor.py                    # 全量来源 kline 面板
  python3 scripts/factor_health_monitor.py --codes 000001,600000
  python3 scripts/factor_health_monitor.py --alert-threshold ic_tvalue:1.0,ls_turnover:0.6

产出:
  data_cache/factor_health/daily/<date>.parquet   — 每因子一行（滚动IC/t/波动/相关/换手/告警）
  data_cache/factor_health/latest.parquet         — 最新快照
  data_cache/factor_health/alerts_<date>.csv      — 超标告警明细（若有）

诚实原则：指标输入缺失时输出 NaN，绝不伪造；面板/收益/IC 缺失在日志中如实说明。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quant_system.market_forecast._support.common.logger import get_logger

log = get_logger("qv6.factor_health_monitor")

_ROOT = Path(__file__).resolve().parent
WORKSPACE = _ROOT.parent


def _default_alerts() -> dict:
    return {"ic_tvalue": 1.0, "factor_ret_vol": 0.05,
            "peer_corr_mean": 0.7, "ls_turnover": 0.5}


def parse_alert_thresholds(text: str | None) -> dict:
    """--alert-threshold 解析：'ic_tvalue:1.0,ls_turnover:0.6' → dict。"""
    out = {}
    if not text:
        return out
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            k, v = part.split(":", 1)
            out[k.strip()] = float(v)
    return out


def load_panels(codes: list[str] | None, max_stocks: int = 0) -> tuple[dict, pd.DataFrame]:
    """构建因子面板 + 公共市场收益序列（复用 ic_vectorized 面板构建）。"""
    sys.path.insert(0, str(WORKSPACE))
    import scripts.ic_vectorized as icv

    kl = icv._load_from_warehouse(codes)
    if not kl:
        return {}, pd.DataFrame()
    if max_stocks:
        import random as _rng
        _rng.seed(42)
        kl = {c: kl[c] for c in sorted(kl)[:max_stocks]}
    common, common_codes = icv._prepare_common(kl)
    if not common_codes:
        return {}, pd.DataFrame()
    panels: dict = {}
    panels.update(icv.build_tech_panels(kl, common=common, codes=common_codes,
                                        spill_dir=None, sample_dates=None))
    panels.update(icv.build_zoo_panels(kl, common=common, codes=common_codes,
                                       spill_dir=None, sample_dates=None))
    # 市场收益代理：样本池等权（仅监控用；警告见审计 P2-3）
    fwd = icv.build_forward_returns(kl, 1)
    return panels, fwd


def run(panels: dict[str, pd.DataFrame],
        ret_wide: pd.DataFrame,
        alert_override: dict | None = None) -> tuple[pd.DataFrame, list[str], list[dict]]:
    """执行监控，返回 (报告, 告警因子列表, 告警明细)。"""
    import scripts.ic_vectorized as icv
    from quant_system.ic_factors import registry as reg
    try:
        reg.autodiscover()
        reg.import_from_zoo()
    except Exception as e:  # noqa: BLE001
        log.warning(f"因子注册表加载失败（category 映射退化）: {e}")

    category_map = {}
    for name in panels:
        m = reg.get_factor(name)
        if m is not None:
            category_map[name] = m.category

    # 逐因子滚动 IC：用每日横截面 IC（前瞻 1 日）最贴切监控语义
    ic_map: dict[str, pd.Series] = {}
    log.info("[P1-2] 计算各因子逐日 IC（前瞻1日）…")
    for name, panel in panels.items():
        is_market = str(name).startswith("mkt_")
        if is_market:
            continue  # 市场级时序因子不做横截面 IC（监控单独口径，此处跳过）
        try:
            fwd = ret_wide.reindex(panel.index)
            min_n = max(8, int(panel.shape[1] * 0.3))
            ic = icv.compute_ic_fast(panel, fwd, min_n=min_n)
            if not ic.empty:
                ic_map[name] = ic
            panels[name] = panel
        except Exception as e:  # noqa: BLE001
            log.warning(f"[P1-2] 因子 {name} IC 计算失败: {e}")

    from quant_system.ic_factors.monitor import monitor_panel
    report, alerted = monitor_panel(
        panels, ic_map=ic_map, ret_wide=ret_wide, category_map=category_map)

    # 阈值告警明细
    thresholds = {**_default_alerts(), **(alert_override or {})}
    details = []
    for _, r in report.iterrows():
        reasons = []
        for key, thr in thresholds.items():
            val = r.get(key, np.nan)
            if pd.notna(val) and abs(val) > thr:
                reasons.append(f"{key}={val:.3f}>{thr}")
        if reasons and len(reasons) > 0:
            details.append({"factor": r["factor"], "category": r["category"],
                            "reasons": ";".join(reasons),
                            "alerts": r.get("alerts", "")})
    # 去重合并（与 monitor_panel 的 status=alert 对齐）
    fine = []
    for d in details:
        if d not in fine:
            fine.append(d)
    details = fine
    return report, alerted, details


def main() -> int:
    ap = argparse.ArgumentParser(description="因子拥挤度 / 滚动 IC 失效监控（每日）")
    ap.add_argument("--codes", default="", help="逗号分隔股票（空=全量）")
    ap.add_argument("--max-stocks", type=int, default=0,
                    help="只取前 N 只股票（调试用）")
    ap.add_argument("--alert-threshold", default="",
                    help="个性化阈值: ic_tvalue:1.0,ls_turnover:0.6 等")
    ap.add_argument("--out-dir", default=str(WORKSPACE / "data_cache" / "factor_health"),
                    help="输出目录")
    args = ap.parse_args()

    codes = [c.strip() for c in args.codes.split(",") if c.strip()] or None
    panels, ret_wide = load_panels(codes, args.max_stocks)
    if not panels:
        log.error("[P1-2] 无面板产出，监控未执行（面板输入缺失）")
        return 2
    if ret_wide.empty:
        log.error("[P1-2] 无市场收益序列，监控未执行（需 kline 收益）")
        return 2

    alert_override = parse_alert_thresholds(args.alert_threshold) or None
    report, alerted, details = run(panels, ret_wide, alert_override)

    out_dir = Path(args.out_dir)
    daily_dir = out_dir / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.now().strftime("%Y%m%d")

    # 落盘 parquet（稳定 schema，方便 health_guardian / 下游消费）
    report.to_parquet(daily_dir / f"{today}.parquet")
    report.to_parquet(out_dir / "latest.parquet", index=False)

    # 告警明细 CSV
    alert_df = pd.DataFrame(details)
    if not alert_df.empty:
        alert_df.to_csv(out_dir / f"alerts_{today}.csv", index=False,
                        encoding="utf-8-sig")

    print("\n=== 因子健康监控（P1-2）===")
    print(report.sort_values("status").to_string(index=False))
    print(f"\n告警因子 {len(alerted)} 个，明细已写入 {out_dir / ('alerts_' + today + '.csv')}")
    print(f"报告已写入 {daily_dir / (today + '.parquet')} 与 {out_dir / 'latest.parquet'}")

    # 超额告警时返回非零，便于 health_guardian / cron 捕获
    return 1 if details else 0


if __name__ == "__main__":
    sys.exit(main())
