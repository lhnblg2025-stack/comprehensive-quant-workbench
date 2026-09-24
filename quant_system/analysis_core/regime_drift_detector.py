"""
regime_drift_detector — 概念漂移检测（V11 月更, 市场规则变了阈值要跟着变）

A股制度在变（注册制/量化监管/涨跌停规则），固定阈值会失真。
每月 1 号对比 近3个月 vs 近1年 的关键统计量:
  日均涨停数 / 平均连板高度 / 封板率中位数 / 炸板率中位数 / 涨停溢价均值

漂移规则（阈值自动调整建议）:
  日均涨停数 下降 >30% → 情绪周期 zt_cnt 阈值整体下移 20%
  平均连板高度 下降 → 空间龙头判定阈值下调 1 板
  封板率 上升 >10pp   → 打板胜率环境改善, 打板策略权重上调
  炸板率 上升 >10pp   → 炸板清仓阈值从 40% 上调至 45%

输出: generated/regime_drift_{date}.json（含阈值调整建议，供 emotion_cycle 手动采纳）

用法:
  python3 -m quant_system.analysis_core.regime_drift_detector [--date 2026-08-01]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.config import ZT_DAILY_STATS  # noqa: E402

CST = timezone(timedelta(hours=8))


def detect(date: str | None = None) -> dict:
    date = date or datetime.now(CST).date().isoformat()
    df = pd.read_parquet(ZT_DAILY_STATS)
    # 列存在性校验（schema 演进防护）
    required = {"date", "zt_cnt", "max_board"}
    if not required.issubset(df.columns):
        return {"date": date, "status": "降级", "note": f"缺列: {required - set(df.columns)}"}
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= pd.Timestamp(date)].sort_values("date")
    if len(df) < 200:
        return {"date": date, "status": "样本不足", "note": f"仅 {len(df)} 交易日"}

    end = df["date"].max()
    recent = df[df["date"] >= end - pd.Timedelta(days=90)]
    year = df[df["date"] >= end - pd.Timedelta(days=365)]

    def _med(s: pd.Series):
        return float(s.median()) if len(s) and s.notna().any() else None

    stats = {
        "recent_zt_median": _med(recent["zt_cnt"]),
        "year_zt_median": _med(year["zt_cnt"]),
        "recent_max_board_median": _med(recent["max_board"]),
        "year_max_board_median": _med(year["max_board"]),
        "recent_seal_rate_median": _med(1 - recent["zb_rate"]) if "zb_rate" in recent else None,
        "year_seal_rate_median": _med(1 - year["zb_rate"]) if "zb_rate" in year else None,
        "recent_premium_mean": float(recent["premium"].mean()) if recent["premium"].notna().any() else None,
        "year_premium_mean": float(year["premium"].mean()) if year["premium"].notna().any() else None,
    }

    suggestions: list[str] = []
    drift = {}
    zt_r, zt_y = stats["recent_zt_median"], stats["year_zt_median"]
    if zt_r is not None and zt_y is not None and zt_y > 0:
        chg = (zt_r - zt_y) / zt_y
        drift["zt_cnt_change"] = round(chg, 2)
        if chg < -0.3:
            suggestions.append("日均涨停数下降>30% → 情绪周期 zt_cnt 阈值整体下移 20%")
    mb_r, mb_y = stats["recent_max_board_median"], stats["year_max_board_median"]
    if mb_r is not None and mb_y is not None and mb_y - mb_r >= 1:
        drift["max_board_change"] = mb_r - mb_y
        suggestions.append("平均连板高度下降 → 空间龙头判定阈值下调 1 板")
    sr_r, sr_y = stats["recent_seal_rate_median"], stats["year_seal_rate_median"]
    if sr_r is not None and sr_y is not None:
        d = sr_r - sr_y
        drift["seal_rate_change"] = round(d, 3)
        if d > 0.10:
            suggestions.append("封板率上升>10pp → 打板策略权重上调")
        elif d < -0.10:
            suggestions.append("封板率下降>10pp → 炸板清仓阈值 40%→45%, 打板权重下调")

    res = {
        "date": date, "status": "ok",
        "window_recent": str(recent["date"].min().date()), "window_year": str(year["date"].min().date()),
        "stats": stats, "drift": drift, "suggestions": suggestions,
    }
    out = ROOT / "generated" / f"regime_drift_{date}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="概念漂移检测")
    ap.add_argument("--date", default=None)
    args = ap.parse_args()
    r = detect(args.date)
    print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
