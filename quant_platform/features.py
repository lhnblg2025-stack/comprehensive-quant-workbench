"""
features.py — 截面特征统一入口（数据资产化 V2-3，2026-08-07）

主链路闭环：feature_store → 因子生产 → IC/分位收益 → 组合构建 → paper 净值 → 复盘归因 → 报告

职责：
  1. 读取 data_warehouse/feature_store/{date}.parquet 截面（经 DataStore 门面）
  2. 多日截面拼成 (date, symbol) MultiIndex 面板
  3. 计算前向收益（N 日），供 FactorEvaluator 做 IC/分层/衰减评估
  4. 统一的截面读取入口：因子/策略/报告/前端全部走这里，禁止散落直读

用法:
    from quant_platform.features import load_panel, forward_returns, list_feature_dates
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def list_feature_dates() -> list[str]:
    """feature_store 已有截面日期（升序 YYYYMMDD）。"""
    fs = ROOT / "data_warehouse" / "feature_store"
    if not fs.exists():
        return []
    return sorted(p.stem for p in fs.glob("*.parquet") if p.stem.isdigit() and len(p.stem) == 8)


def load_cross_section(date: str) -> pd.DataFrame:
    """读取单日截面（经 DataStore 门面）。date 支持 YYYYMMDD / YYYY-MM-DD。"""
    from quant_system.data_store import DataStore
    return DataStore().get_dataset("feature_store", date=date.replace("-", ""))


def load_panel(start: str | None = None, end: str | None = None,
               features: list[str] | None = None) -> pd.DataFrame:
    """多日截面 → (date, symbol) MultiIndex 面板。

    Args:
        start/end: 日期区间（YYYYMMDD）。
        features:  只取这些特征列（默认全取）。
    Returns:
        MultiIndex (date, symbol) × 特征列的 DataFrame。
    """
    dates = list_feature_dates()
    if not dates:
        return pd.DataFrame()
    if start:
        dates = [d for d in dates if d >= start.replace("-", "")]
    if end:
        dates = [d for d in dates if d <= end.replace("-", "")]
    frames = []
    for d in dates:
        try:
            df = load_cross_section(d)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        if "code" not in df.columns:
            continue
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["code"] = df["code"].astype(str).str.zfill(6)
        df = df.dropna(subset=["date", "code"])
        if features:
            keep = ["code", "date"] + [f for f in features if f in df.columns]
            df = df[keep]
        df = df.set_index(["date", "code"])
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()


def forward_returns(panel: pd.DataFrame, horizon: int = 5,
                    close_col: str = "close") -> pd.Series:
    """面板前向收益（按 symbol 组内滚动 horizon 日）。

    Returns:
        Series: MultiIndex (date, symbol) 对齐 panel 的 close 前向收益。
    """
    if panel.empty or close_col not in panel.columns:
        return pd.Series(dtype=float, index=panel.index)
    sym_level = "symbol" if "symbol" in panel.index.names else ("code" if "code" in panel.index.names else None)
    if sym_level is None:
        return pd.Series(dtype=float, index=panel.index)
    closes = panel[close_col].unstack(sym_level)
    fwd = closes.shift(-horizon) / closes - 1
    return fwd.stack().reindex(panel.index)


def evaluate_feature(panel: pd.DataFrame, feature: str,
                     horizon: int = 5) -> dict:
    """单特征全维度评估：IC 序列 / ICIR / 分层收益 / 衰减。

    主链路核心：feature_store 截面 → IC 评估（复用 FactorEvaluator）。
    """
    from quant_system.factor_system.evaluation import FactorEvaluator

    if feature not in panel.columns:
        return {"feature": feature, "error": f"特征 {feature} 不在面板中"}
    alpha = panel[feature]
    fwd = forward_returns(panel, horizon=horizon)
    # 面板格式校验（索引层级名可能是 code/symbol）
    sym_level = "symbol" if "symbol" in alpha.index.names else ("code" if "code" in alpha.index.names else None)
    if not isinstance(alpha.index, pd.MultiIndex) or sym_level is None:
        return {"feature": feature, "error": "面板索引需 (date, symbol) MultiIndex"}
    fwd_mi = fwd
    if not isinstance(fwd_mi.index, pd.MultiIndex):
        fwd_mi = fwd.unstack(sym_level).stack()
    ic_series = FactorEvaluator.ic_series(alpha, fwd_mi)
    qr = FactorEvaluator.quantile_returns(alpha, fwd_mi, n_buckets=10)
    decay = FactorEvaluator.factor_decay(alpha, fwd_mi, max_lag=10)
    return {
        "feature": feature,
        "horizon": horizon,
        "rank_ic_mean": round(float(ic_series.mean()), 4) if len(ic_series) else None,
        "rank_ic_std": round(float(ic_series.std(ddof=1)), 4) if len(ic_series) > 1 else None,
        "rank_icir": round(float(FactorEvaluator.icir(ic_series)), 4) if len(ic_series) >= 5 else None,
        "positive_pct": round(float((ic_series > 0).mean()), 4) if len(ic_series) else None,
        "n_periods": int(len(ic_series)),
        "spread_return": round(float(FactorEvaluator.spread_return(alpha, fwd_mi)), 4),
        "quantile_returns": {int(k): round(float(v), 4) for k, v in qr.items() if pd.notna(v)},
        "decay_ic": {int(k): round(float(v), 4) for k, v in decay.items() if pd.notna(v)},
    }


def build_panel_and_evaluate(features: list[str] | None = None,
                             horizon: int = 5) -> dict:
    """一键：读全截面 → 建面板 → 批量评估特征（V2-3 主链路标准入口）。"""
    panel = load_panel()
    if panel.empty:
        return {"ok": False, "error": "feature_store 无截面数据"}
    cols = features or [c for c in panel.columns if c not in ("close", "pct_chg")]
    results = {}
    for f in cols:
        try:
            results[f] = evaluate_feature(panel, f, horizon=horizon)
        except Exception as e:  # noqa: BLE001
            results[f] = {"feature": f, "error": str(e)[:120]}
    return {
        "ok": True,
        "n_dates": len(panel.index.get_level_values(0).unique()),
        "n_symbols": len(panel.index.get_level_values(1).unique()),
        "n_features": len(cols),
        "results": results,
    }


if __name__ == "__main__":
    import json

    horizon = 5
    if len(sys.argv) > 1:
        horizon = int(sys.argv[1])
    res = build_panel_and_evaluate(horizon=horizon)
    print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
