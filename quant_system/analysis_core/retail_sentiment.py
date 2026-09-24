"""retail_sentiment — 散户情绪代理指标（V11）

微博/B站社交被风控后，用 A 股特有数据代理"散户情绪"，替代社交情绪维度：
  - 融资余额变化率（散户杠杆风向标）: margin_sentiment()
  - 股东户数变化（户数激增=筹码分散=机构出货；户数减少=筹码集中）: holder_sentiment()
  - 低价股占比（<5 元占比急剧下降=增量资金入场）: cheap_stock_ratio()
  - 三指标加权合成散户情绪分 0-100: composite()

数据源（全部本地，禁止网络）:
  data_warehouse/market/margin_detail_sh|sz/*.parquet   两融个股明细（按交易日分文件）
  data_warehouse/market/gdhs_all.parquet               股东户数（最新一期快照）
  data_warehouse/kline/*.parquet                       全市场日线（抽样 list(glob)[::50]）

日期对齐: 每个维度用各自数据集的最新日期，不强制对齐。
缺失容错: 某维度数据缺失/异常时跳过并重分配权重，composite 不崩溃。

用法:
  python3 -m quant_system.analysis_core.retail_sentiment
"""

from __future__ import annotations
import logging

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.config import MARKET_DIR  # noqa: E402

GENERATED_DIR = ROOT / "generated"

MARGIN_SH_DIR = MARKET_DIR / "margin_detail_sh"
MARGIN_SZ_DIR = MARKET_DIR / "margin_detail_sz"
GDHS_ALL = MARKET_DIR / "gdhs_all.parquet"
KLINE_DIR = MARKET_DIR.parent / "kline"

# 全A 股票代码: 沪市 60/68，深市 00/30（剔除 5x 基金、1x 债券等非个股标的）
A_SHARE_CODE = r"^[036]\d{5}$"

# 两融明细列名（沪市带日期列，深市不带，日期取文件名）
SH_DATE_COL = "信用交易日期"
SH_CODE_COL = "标的证券代码"
SH_BAL_COL = "融资余额"
SZ_CODE_COL = "证券代码"
SZ_BAL_COL = "融资余额"

# 股东户数列名
GDHS_CUR_COL = "股东户数-本次"
GDHS_PREV_COL = "股东户数-上次"
GDHS_CHG_COL = "股东户数-增减"
GDHS_CHG_PCT_COL = "股东户数-增减比例"
GDHS_END_CUR_COL = "股东户数统计截止日-本次"
GDHS_END_PREV_COL = "股东户数统计截止日-上次"
GDHS_ANN_COL = "公告日期"

# 低价股阈值（元）
CHEAP_PRICE = 5.0


def _read_margin_series(market_dir: Path, days: int, buf: int = 20) -> dict[str, float]:
    """读某市场近 days 交易日融资余额序列 {YYYYMMDD: 余额}，跳过空文件。"""
    files = sorted(market_dir.glob("*.parquet"))
    if not files:
        return {}
    series: dict[str, float] = {}
    for p in files[-(days + buf):]:
        try:
            df = pd.read_parquet(p)
        except Exception as e:
            logging.getLogger(__name__).error(f"[retail_sentiment] 操作失败: {e}", exc_info=True)
            continue
        if df is None or df.empty or SH_BAL_COL not in df:
            continue
        code_col = SH_CODE_COL if SH_CODE_COL in df else SZ_CODE_COL
        if code_col not in df:
            continue
        # 仅保留 A 股股票，剔除 ETF/基金/债券
        codes = df[code_col].astype(str)
        stock = df[codes.str.match(A_SHARE_CODE, na=False)]
        if stock.empty:
            continue
        bal = float(pd.to_numeric(stock[SH_BAL_COL], errors="coerce").sum())
        if not np.isfinite(bal) or bal <= 0:
            continue
        # 日期: 沪市文件内带日期列，深市取文件名 YYYYMMDD
        if SH_DATE_COL in df:
            d = str(df[SH_DATE_COL].iloc[0])
            if not (len(d) == 8 and d.isdigit()):
                d = p.stem
        else:
            d = p.stem
        series[d] = bal
    return series


def margin_sentiment(days: int = 30) -> dict:
    """全A融资余额近 days 日序列 + 5日变化率 + 信号。

    沪深两所按日期对齐求和（任一市场缺失当日则跳过该日，避免低估）。
    """
    result: dict = {"available": False, "error": None}
    try:
        sh = _read_margin_series(MARGIN_SH_DIR, days)
        sz = _read_margin_series(MARGIN_SZ_DIR, days)
        common = sorted(set(sh) & set(sz))
        if len(common) < 2:
            result["error"] = "沪深两融明细不足两个交易日"
            return result
        common = common[-days:]
        balances = [sh[d] + sz[d] for d in common]
        latest = balances[-1]
        base_idx = max(0, len(balances) - 6)
        base = balances[base_idx]
        chg_5d_pct = (latest / base - 1.0) * 100.0 if base > 0 else None

        # 加杠杆=乐观 / 降杠杆=谨慎（微小波动视为中性，防噪声抖动）
        if chg_5d_pct is None:
            signal = "数据不足"
        elif chg_5d_pct >= 0.3:
            signal = "乐观(加杠杆)"
        elif chg_5d_pct <= -0.3:
            signal = "谨慎(降杠杆)"
        else:
            signal = "中性(杠杆平稳)"

        result.update({
            "available": True,
            "date": common[-1],
            "latest_balance": latest,
            "latest_balance_yi": round(latest / 1e8, 2),
            "chg_5d_pct": round(chg_5d_pct, 2) if chg_5d_pct is not None else None,
            "signal": signal,
            "series": [{"date": d, "balance": round(sh[d] + sz[d], 2)} for d in common],
            "sh_latest_date": common[-1] if sh else None,
            "sz_latest_date": common[-1] if sz else None,
        })
    except Exception as e:  # noqa: BLE001 — 容错设计，失败不阻塞 composite
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def holder_sentiment() -> dict:
    """最近两期股东户数变化分布 → 增加/减少家数、中位数变化率 → 信号。"""
    result: dict = {"available": False, "error": None}
    try:
        if not GDHS_ALL.exists():
            result["error"] = "gdhs_all.parquet 不存在"
            return result
        g = pd.read_parquet(GDHS_ALL)
        chg = pd.to_numeric(g[GDHS_CHG_COL], errors="coerce").dropna()
        chg_pct = pd.to_numeric(g[GDHS_CHG_PCT_COL], errors="coerce").dropna()
        if chg.empty:
            result["error"] = "股东户数增减列为空"
            return result

        inc = int((chg > 0).sum())
        dec = int((chg < 0).sum())
        flat = int((chg == 0).sum())
        median_pct = float(chg_pct.median()) if not chg_pct.empty else None
        inc_ratio = inc / (inc + dec) if (inc + dec) > 0 else None

        # 户数普增=筹码分散=谨慎；户数普减=筹码集中=偏乐观
        if inc_ratio is None:
            signal = "数据不足"
        elif inc_ratio >= 0.55:
            signal = "谨慎(户数普增,筹码分散)"
        elif inc_ratio <= 0.45:
            signal = "乐观(户数普减,筹码集中)"
        else:
            signal = "中性(户数变化均衡)"

        end_cur = g[GDHS_END_CUR_COL].dropna().astype(str)
        ann = g[GDHS_ANN_COL].dropna().astype(str)
        result.update({
            "available": True,
            "date": str(pd.to_datetime(end_cur, errors="coerce").max().date()) if not end_cur.empty else None,
            "as_of": str(end_cur.iloc[0]) if not end_cur.empty else None,
            "latest_announce_date": str(ann.max()) if not ann.empty else None,
            "total": int(len(g)),
            "increase_count": inc,
            "decrease_count": dec,
            "flat_count": flat,
            "increase_ratio": round(inc_ratio, 4) if inc_ratio is not None else None,
            "median_chg_pct": round(median_pct, 4) if median_pct is not None else None,
            "signal": signal,
        })
    except Exception as e:  # noqa: BLE001
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def cheap_stock_ratio(days: int = 20) -> dict:
    """每日 <5 元股票占比序列（kline 抽样，list(glob)[::50]）+ 5日变化 → 信号。"""
    result: dict = {"available": False, "error": None}
    try:
        files = sorted(KLINE_DIR.glob("*.parquet"))[::50]  # 抽样，避免遍历全部
        frames = []
        for p in files:
            try:
                frames.append(pd.read_parquet(p, columns=["date", "close"]))
            except Exception as e:
                logging.getLogger(__name__).error(f"[retail_sentiment] 操作失败: {e}", exc_info=True)
                continue
        if not frames:
            result["error"] = "kline 抽样文件全部读取失败"
            return result
        df = pd.concat(frames, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["date", "close"])

        grp = df.groupby("date")["close"]
        daily = pd.DataFrame({"count": grp.count(), "cheap": grp.apply(lambda s: float((s < CHEAP_PRICE).sum()))})
        daily["ratio"] = daily["cheap"] / daily["count"] * 100.0
        daily = daily.sort_index()
        if len(daily) < 2:
            result["error"] = "kline 抽样有效交易日不足"
            return result

        seq = daily.tail(days)
        latest_ratio = float(seq["ratio"].iloc[-1])
        base_idx = max(0, len(seq) - 6)
        base_ratio = float(seq["ratio"].iloc[base_idx])
        chg_5d_pp = latest_ratio - base_ratio

        # 占比骤降=增量资金入场=乐观；占比骤升=存量博弈/避险=谨慎
        if chg_5d_pp <= -2.0:
            signal = "乐观(低价股占比骤降,增量入场)"
        elif chg_5d_pp >= 2.0:
            signal = "谨慎(低价股占比骤升)"
        else:
            signal = "中性(低价股占比平稳)"

        result.update({
            "available": True,
            "date": seq.index[-1].strftime("%Y%m%d"),
            "latest_ratio_pct": round(latest_ratio, 2),
            "chg_5d_pp": round(chg_5d_pp, 2),
            "signal": signal,
            "series": [{"date": d.strftime("%Y%m%d"), "ratio_pct": round(r, 2)}
                       for d, r in zip(seq.index, seq["ratio"])],
            "sampled_stocks": int(df["date"].max() is not None and len(df[df["date"] == seq.index[-1]]["close"])),
        })
    except Exception as e:  # noqa: BLE001
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def _margin_score(res: dict) -> float | None:
    """融资5日变化率 → 0-100 乐观分（±5% 打满/打空）。"""
    c = res.get("chg_5d_pct")
    if c is None:
        return None
    return float(np.clip(50.0 + c * 10.0, 0.0, 100.0))


def _holder_score(res: dict) -> float | None:
    """户数增加占比 → 0-100 乐观分（增加占比越高=筹码越分散=分越低）。"""
    r = res.get("increase_ratio")
    if r is None:
        return None
    return float(np.clip((1.0 - r) * 100.0, 0.0, 100.0))


def _cheap_score(res: dict) -> float | None:
    """低价股占比5日变化(pp) → 0-100 乐观分（骤降=分高，±5pp 打满/打空）。"""
    c = res.get("chg_5d_pp")
    if c is None:
        return None
    return float(np.clip(50.0 - c * 10.0, 0.0, 100.0))


def _label(score: float) -> str:
    if score >= 75:
        return "亢奋"
    if score >= 60:
        return "偏热"
    if score >= 45:
        return "中性"
    if score >= 30:
        return "低迷"
    return "冰点"


def composite() -> dict:
    """三指标加权合成散户情绪分 0-100（融资40%+户数30%+低价股30%）。

    某维度缺失/异常时跳过并重分配权重；全部缺失时返回"数据不足"而不崩溃。
    结果存 generated/retail_sentiment_{date}.json。
    """
    margin_res = margin_sentiment()
    holder_res = holder_sentiment()
    cheap_res = cheap_stock_ratio()

    dims: list[tuple[str, dict, float | None, callable]] = [
        ("margin", margin_res, 0.40, _margin_score),
        ("holder", holder_res, 0.30, _holder_score),
        ("cheap", cheap_res, 0.30, _cheap_score),
    ]

    used: list[dict] = []
    total_w = 0.0
    for name, res, w, scorer in dims:
        if not res.get("available"):
            continue
        score = scorer(res)
        if score is None:
            continue
        used.append({"dimension": name, "weight": w, "score": round(score, 1),
                     "date": res.get("date"), "signal": res.get("signal")})
        total_w += w

    dates = [u["date"] for u in used if u.get("date")]
    date = max(dates) if dates else datetime.now().strftime("%Y%m%d")

    if not used or total_w <= 0:
        out = {"date": date, "available": False, "score": None, "label": "数据不足",
               "dimensions": [], "note": "三个数据源均不可用"}
    else:
        score = round(sum(u["score"] * u["weight"] for u in used) / total_w)
        out = {"date": date, "available": True, "score": int(score), "label": _label(score),
               "dimensions": used,
               "weights_note": "缺失维度已跳过并按剩余权重重新归一",
               "margin": margin_res, "holder": holder_res, "cheap": cheap_res}

    # 落盘 generated/retail_sentiment_{date}.json
    try:
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        path = GENERATED_DIR / f"retail_sentiment_{date}.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
        out["saved_to"] = str(path)
    except Exception as e:  # noqa: BLE001
        out["save_error"] = f"{type(e).__name__}: {e}"
    return out


if __name__ == "__main__":
    res = composite()
    print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
