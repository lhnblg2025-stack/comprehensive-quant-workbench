"""
market_microstructure — 涨跌停微观结构（V11 短线 M?）

从涨停/炸板池提取 A 股特有的微观结构信号，输出三个维度:

  1. 封单强度 seal_strength: 封单比 = seal_fund / float_mv
     强(≥5%) / 中(≥2%) / 弱(<2%)
  2. 洗盘 vs 出货 washout_vs_distribution: 炸板后回封且炸板≤2次 = 洗盘；
     多次炸板(>2次)或炸板未回封 = 出货
  3. 封板时段溢价 seal_time_premium: 首封时段 × 次日溢价(zt_pool_history.next_pct)
     时段: 早盘<10:00 / 上午10:00-11:30 / 午后13:00-14:30 / 尾盘>14:30
     结论: 早盘封板溢价显著高于尾盘

数据源（全部本地，禁止网络）:
  data_warehouse/market/zt_pool_em_daily.parquet  东财三池精确日更（封单金额/首封时间/炸板次数）
  data_warehouse/market/zt_pool_history.parquet   K线重建全量（next_pct 次日涨幅）

输出: generated/microstructure_{date}.json
集成: battle_map 消费 run_today().distribution_risks（风险清单）与 seal_top（观察栏）

用法:
  python3 -m quant_system.analysis_core.market_microstructure --today
  python3 -m quant_system.analysis_core.market_microstructure --premium 60
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.config import ZT_EM_DAILY, ZT_HISTORY  # noqa: E402

SEAL_LEVELS = [(0.05, "强"), (0.02, "中")]
TIME_BUCKETS = ["早盘(<10:00)", "上午(10:00-11:30)", "午后(13:00-14:30)", "尾盘(>14:30)", "未知"]


def _json_safe(df: pd.DataFrame) -> list[dict]:
    """DataFrame → JSON 安全记录列表（NaN/None 统一转 null）。"""
    if df is None or df.empty:
        return []
    return json.loads(df.where(pd.notna(df), None).to_json(orient="records", force_ascii=False))


def _time_bucket(t) -> str:
    """HHMMSS 字符串 → 封板时段；None/非 6 位数字归入"未知"（字典序即时间序）。"""
    if t is None or pd.isna(t):
        return "未知"
    s = str(t).strip()
    if len(s) != 6 or not s.isdigit():
        return "未知"
    if s < "100000":
        return "早盘(<10:00)"
    if s < "113000":
        return "上午(10:00-11:30)"
    if s < "130000":
        return "未知"  # 午间休市时间戳异常
    if s <= "143000":
        return "午后(13:00-14:30)"
    return "尾盘(>14:30)"


def _load_day(date: str | None) -> tuple[pd.DataFrame, str | None]:
    """读取 zt_pool_em_daily 指定交易日（默认最新）的涨停+炸板行。"""
    if not ZT_EM_DAILY.exists():
        print("[market_microstructure] 告警: 数据缺失 zt_pool_em_daily.parquet")
        return pd.DataFrame(), None
    df = pd.read_parquet(ZT_EM_DAILY)
    if df.empty:
        print("[market_microstructure] 告警: zt_pool_em_daily.parquet 为空")
        return pd.DataFrame(), None
    day = pd.Timestamp(date) if date else df["date"].max()
    day_str = str(pd.Timestamp(day).date())
    if {"is_zt", "is_zb"}.issubset(df.columns):
        sub = df[(df["date"] == day) & (df["is_zt"] | df["is_zb"])]
    else:
        sub = df[df["date"] == day]
    if sub.empty:
        print(f"[market_microstructure] 告警: {day_str} 无涨停/炸板数据")
        return pd.DataFrame(), day_str
    return sub, day_str


def seal_strength(df_day: pd.DataFrame) -> pd.DataFrame:
    """封单强度: 封单比 = seal_fund/float_mv → 强≥0.05/中≥0.02/弱<0.02。

    float_mv 为 0/NaN 时 seal_ratio 置 None 并跳过该股（防除零，
    seal_top 排名时 dropna 剔除）。
    """
    cols = ["code", "name", "seal_ratio", "seal_level"]
    if df_day.empty or not {"seal_fund", "float_mv"}.issubset(df_day.columns):
        print("[market_microstructure] 告警: 封单强度输入缺失 seal_fund/float_mv")
        return pd.DataFrame(columns=cols)
    rows = []
    for _, r in df_day.iterrows():
        sf, fm = r.get("seal_fund"), r.get("float_mv")
        if pd.isna(sf) or pd.isna(fm) or fm == 0:
            rows.append({"code": r.get("code"), "name": r.get("name"),
                         "seal_ratio": None, "seal_level": None})
            continue
        ratio = float(sf) / float(fm)
        level = "强" if ratio >= SEAL_LEVELS[0][0] else ("中" if ratio >= SEAL_LEVELS[1][0] else "弱")
        rows.append({"code": r.get("code"), "name": r.get("name"),
                     "seal_ratio": round(ratio, 4), "seal_level": level})
    return pd.DataFrame(rows, columns=cols)


def washout_vs_distribution(df_day: pd.DataFrame) -> pd.DataFrame:
    """炸板行为判别: 洗盘(炸板后回封且 zb_times≤2, signal=0) / 出货(多次炸板或未回封, signal=1)。

    final_sealed = 最终是否封板（涨停池行）；未炸板(回封且 zb_times=0) 标记"无炸板" signal=None。
    """
    cols = ["code", "name", "zb_times", "final_sealed", "judge", "signal"]
    if df_day.empty or not {"is_zt", "zb_times"}.issubset(df_day.columns):
        print("[market_microstructure] 告警: 炸板判别输入缺失 is_zt/zb_times")
        return pd.DataFrame(columns=cols)
    rows = []
    for _, r in df_day.iterrows():
        zb = r.get("zb_times")
        zb = 0 if pd.isna(zb) else int(zb)
        final_sealed = bool(r.get("is_zt", False))
        if not final_sealed:
            judge, signal = "出货", 1
        elif zb == 0:
            judge, signal = "无炸板", None
        elif zb <= 2:
            judge, signal = "洗盘", 0
        else:
            judge, signal = "出货", 1
        rows.append({"code": r.get("code"), "name": r.get("name"),
                     "zb_times": zb, "final_sealed": final_sealed,
                     "judge": judge, "signal": signal})
    out = pd.DataFrame(rows, columns=cols)
    out["zb_times"] = out["zb_times"].astype("Int64")
    out["signal"] = out["signal"].astype("Int64")
    return out


def seal_time_premium(days: int = 60) -> pd.DataFrame:
    """封板时段 × 次日溢价: em 池首封时间分桶 × zt_pool_history.next_pct。

    结论: 早盘封板溢价显著高于尾盘（越早封板次日溢价越高）。
    """
    cols = ["time_bucket", "count", "avg_next_pct", "win_rate"]
    if days <= 0:
        return pd.DataFrame(columns=cols)
    if not ZT_EM_DAILY.exists() or not ZT_HISTORY.exists():
        print("[market_microstructure] 告警: 溢价统计数据缺失 zt_pool_em_daily/zt_pool_history")
        return pd.DataFrame(columns=cols)
    em = pd.read_parquet(ZT_EM_DAILY)
    hist = pd.read_parquet(ZT_HISTORY)
    if em.empty or hist.empty or "first_seal_time" not in em.columns:
        print("[market_microstructure] 告警: 溢价统计数据为空或缺 first_seal_time")
        return pd.DataFrame(columns=cols)
    em = em[em["is_zt"]].copy()
    if em.empty:
        return pd.DataFrame(columns=cols)
    dates = sorted(em["date"].unique())[-days:]
    m = em[em["date"].isin(dates)].merge(
        hist[["date", "code", "next_pct"]], on=["date", "code"], how="left")
    if m.empty:
        return pd.DataFrame(columns=cols)
    m["time_bucket"] = m["first_seal_time"].map(_time_bucket)
    g = []
    for b in TIME_BUCKETS:
        sub = m[m["time_bucket"] == b]
        if sub.empty:
            continue
        nxt = sub["next_pct"].dropna()
        g.append({
            "time_bucket": b,
            "count": int(len(sub)),
            "avg_next_pct": round(float(nxt.mean()), 3) if len(nxt) else None,
            "win_rate": round(float((nxt > 0).mean()), 3) if len(nxt) else None,
        })
    return pd.DataFrame(g, columns=cols)


def run_today(date: str | None = None) -> dict:
    """当日微观结构汇总并落盘 generated/microstructure_{date}.json。"""
    day, date_str = _load_day(date)
    prem = seal_time_premium(60)
    result = {
        "date": date_str,
        "seal_top": [],
        "distribution_risks": [],
        "time_premium_summary": _json_safe(prem),
    }
    if day.empty:
        print("[market_microstructure] 告警: 当日无数据，返回空汇总")
        return result
    zt = day[day["is_zt"]] if "is_zt" in day.columns else day
    wash = washout_vs_distribution(day)
    seal = seal_strength(zt)
    if not seal.empty:
        top = seal.dropna(subset=["seal_ratio"]).sort_values("seal_ratio", ascending=False).head(10)
        result["seal_top"] = _json_safe(top[["code", "name", "seal_ratio", "seal_level"]])
    if not wash.empty:
        risks = wash[wash["signal"] == 1]
        if not risks.empty:
            result["distribution_risks"] = _json_safe(
                risks[["code", "name", "zb_times", "final_sealed"]].sort_values(
                    "zb_times", ascending=False))
    if date_str:
        out = ROOT / "generated" / f"microstructure_{date_str}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="涨跌停微观结构（封单强度/洗盘vs出货/封板时段溢价）")
    ap.add_argument("--today", action="store_true", help="输出当日汇总并落盘 JSON")
    ap.add_argument("--date", default=None, help="指定交易日 YYYY-MM-DD（默认最新）")
    ap.add_argument("--premium", type=int, nargs="?", const=60, metavar="DAYS",
                    help="封板时段溢价统计（默认最近 60 个交易日）")
    args = ap.parse_args()
    if args.premium:
        print(seal_time_premium(args.premium).to_string(index=False))
    else:
        print(json.dumps(run_today(args.date), ensure_ascii=False, indent=2))
