"""
quant_platform.cb_strategy — 可转债双低策略（V3 融合层）

来源：quant_v6/strategy/cb_double_low_v7.py（V7.0 双模策略，quant_v6 退役后融合到主栈）
设计：纯函数式实现（无 quant_v6 依赖），数据源走 DataStore cb_spot（本地 320 只可转债）。

双模（V7.0 P2 修复）:
  - premium 模式（默认）: 双低 = 转债价格 + 转股溢价率，低=好
  - ytm 模式: 用到期收益率（YTM）排序替代溢价率，高 YTM（深度折价债）优先
  - auto 模式（默认）: 样本池中位溢价率 > 80% 时自动切到 ytm 模式

筛选规则:
  1. 剔除双低缺失/价格异常（>130 元强赎区、<70 元高风险债）
  2. 剔除成交额过低（流动性不足，默认 <500 万）
  3. premium 模式额外剔除溢价率 > 上限（默认 80%）；ytm 模式不按溢价率排除
  4. 行业/正股分散可选（按名称前缀粗分）

输出: {"picks": [...], "mode": ..., "thermometer": ..., "note": ...}
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# 确保 ROOT 在 sys.path（脚本直接运行时也可 import quant_system）
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 到期收益率列名候选（akshare 不同版本）
YTM_COL_CANDIDATES = ["ytm_rt", "ytm", "到期收益率", "ytm_value", "ytm_rate", "ytm_rt_2"]


def _find_ytm_col(df: pd.DataFrame) -> Optional[str]:
    for c in YTM_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def _norm_cb_df(df: pd.DataFrame) -> pd.DataFrame:
    """归一化列名：兼容 DataStore cb_spot（symbol/name/trade/changepercent/premium_rt/double_low）与 akshare 中文列。"""
    d = df.copy()
    rename = {
        "symbol": "code", "名称": "name", "转债代码": "code",
        "转债名称": "name", "最新价": "trade", "现价": "trade",
        "涨跌幅": "changepercent", "转股溢价率": "premium_rt",
        "成交额": "amount", "到期收益率": "ytm_rt",
    }
    # 只映射目标列不存在的列（避免 symbol→code 与已有 code 列冲突产生重复列）
    rename = {k: v for k, v in rename.items() if k in d.columns and v not in d.columns}
    d = d.rename(columns=rename)
    # code 归一化：优先已有 code 列（本地 cb_spot 已有纯数字 code），symbol 带前缀仅 fallback
    if "code" in d.columns:
        d["code"] = d["code"].astype(str).str.replace(r"^(sh|sz|bj)", "", regex=True)
    elif "symbol" in d.columns:
        d["code"] = d["symbol"].astype(str).str.replace(r"^(sh|sz|bj)", "", regex=True)
    for col in ["trade", "changepercent", "premium_rt", "amount", "ytm_rt", "double_low"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    # 剔除无效行（trade=0 / double_low=0，北交定转数据未开盘）
    if "trade" in d.columns:
        d = d[d["trade"] > 0]
    # double_low = 价格 + 溢价率（缺失时合成）
    if "double_low" not in d.columns and "trade" in d.columns and "premium_rt" in d.columns:
        d["double_low"] = d["trade"] + d["premium_rt"]
    elif "double_low" in d.columns:
        # double_low 缺失/为 0 时用 trade 兜底
        d.loc[d["double_low"].isna() | (d["double_low"] <= 0), "double_low"] = d["trade"]
    if "name" not in d.columns:
        d["name"] = d.get("code", "")
    return d


def _filter(d: pd.DataFrame, min_price: float = 70.0, max_price: float = 130.0,
            min_amount: float = 500.0, premium_cap: bool = True,
            max_premium: float = 80.0) -> pd.DataFrame:
    """基础筛选；premium_cap=True 时额外剔除高溢价债（仅低溢价模式）。"""
    if "double_low" not in d.columns:
        d["double_low"] = pd.to_numeric(d.get("trade", 0), errors="coerce")
    d = d.dropna(subset=["double_low"])
    # 价格区间（double_low 列可能是价格，用 trade 判断更准）
    if "trade" in d.columns:
        d = d[(d["trade"] >= min_price) & (d["trade"] <= max_price)]
    if "amount" in d.columns:
        d = d[d["amount"] >= min_amount]
    if premium_cap and "premium_rt" in d.columns:
        d = d[d["premium_rt"] <= max_premium]
    return d


def _decide_mode(universe: pd.DataFrame, mode: str = "auto",
                 ytm_threshold: float = 80.0) -> str:
    if mode not in ("auto", "premium", "ytm"):
        mode = "auto"
    if mode == "auto":
        if "premium_rt" in universe.columns:
            med = pd.to_numeric(universe["premium_rt"], errors="coerce").median()
            mode = "ytm" if (pd.notna(med) and med > ytm_threshold) else "premium"
        else:
            mode = "premium"
    if mode == "ytm" and _find_ytm_col(universe) is None:
        mode = "premium"
    return mode


def cb_double_low(
    df: pd.DataFrame | None = None,
    top_n: int = 10,
    mode: str = "auto",
    min_price: float = 70.0,
    max_price: float = 130.0,
    min_amount: float = 500.0,
    max_premium: float = 80.0,
    ytm_threshold: float = 80.0,
    thermometer: float = 60.0,
) -> dict[str, Any]:
    """可转债双低策略主入口（V3 融合，quant_v6 退役后主栈原生实现）。

    Args:
        df: cb_spot DataFrame（缺省时读 DataStore 本地 cb_spot）
        top_n: 选债数量
        mode: auto/premium/ytm
        thermometer: 0-100 情绪温度计，控制仓位（clip 0.2-1.0）

    Returns:
        {"picks": [...], "mode": ..., "position": ..., "note": ...}
    """
    try:
        if df is None:
            from quant_system.data_store import DataStore
            df = DataStore().get_dataset("cb_spot")
        if df is None or len(df) == 0:
            return {"picks": [], "mode": mode, "position": 0.0,
                    "note": "无转债数据（DataStore cb_spot 为空）"}
        d = _norm_cb_df(df)
        # 先跑基础筛选（不含溢价率上限），用于模式决策
        universe = _filter(d, min_price, max_price, min_amount,
                           premium_cap=False, max_premium=max_premium)
        if universe.empty:
            return {"picks": [], "mode": mode, "position": 0.0, "note": "筛选后无转债"}
        decided = _decide_mode(universe, mode, ytm_threshold)

        if decided == "ytm":
            ytm_col = _find_ytm_col(universe)
            dd = universe.copy()
            dd["ytm"] = pd.to_numeric(dd[ytm_col], errors="coerce")
            dd = dd.dropna(subset=["ytm"])
            if dd.empty:
                return {"picks": [], "mode": decided, "position": 0.0,
                        "note": "YTM 模式下无有效到期收益率"}
            dd = dd.sort_values("ytm", ascending=False).head(top_n)
            score_col = "ytm"
        else:
            dd = _filter(d, min_price, max_price, min_amount,
                         premium_cap=True, max_premium=max_premium)
            if dd.empty:
                return {"picks": [], "mode": decided, "position": 0.0,
                        "note": "筛选后无转债"}
            dd = dd.sort_values("double_low").head(top_n)
            score_col = "double_low"

        dd = dd.drop_duplicates(subset=["code"], keep="first")
        position = float(np.clip(thermometer / 100.0, 0.2, 1.0))

        picks: list[dict[str, Any]] = []
        for _, r in dd.iterrows():
            code = str(r.get("code", ""))
            ytm_raw = r.get("ytm", np.nan)
            ytm = round(float(ytm_raw), 2) if pd.notna(ytm_raw) else None
            prem = r.get("premium_rt", np.nan)
            picks.append({
                "code": code,
                "name": str(r.get("name", "")),
                "score": round(float(r[score_col]), 2),
                "trade": round(float(r.get("trade", np.nan)), 2) if pd.notna(r.get("trade")) else None,
                "premium": round(float(prem), 2) if pd.notna(prem) else None,
                "ytm": ytm,
                "mode": decided,
            })
        return {"picks": picks, "mode": decided, "position": round(position, 2),
                "top_n": len(picks),
                "note": f"{decided} 模式，双低选债 {len(picks)} 只，仓位 {position:.0%}"}
    except Exception as e:
        return {"picks": [], "mode": mode, "position": 0.0, "note": f"计算失败: {e}"}


if __name__ == "__main__":
    import json
    r = cb_double_low()
    print("=== 可转债双低策略（本地 cb_spot）===")
    print(f"模式: {r.get('mode')} | 仓位: {r.get('position')} | 说明: {r.get('note')}")
    for p in r.get("picks", [])[:8]:
        print(f"  {p['code']} {p['name']} 双低={p['score']} 价={p['trade']} 溢价={p['premium']} YTM={p['ytm']}")
