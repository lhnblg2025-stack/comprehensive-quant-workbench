"""
calendar_effects — A股日历效应规则库（V11）

把 A 股日历效应编码成规则，在特定日期自动调整仓位系数与因子权重，输出到作战地图。

所有候选效应均用 zt_pool_history 全量数据（2020→今，真实交易日）统计验证
（groupby + mean），禁止拍脑袋给结论；无网络调用；农历简化为公历日期近似
（春节按 1月底-2月中窗口，用 pandas 日期区间）。

效应列表:
  1. spring_festival  春节行情:  春节窗口(1/25-2/20) 涨停家数/上证指数 年胜率
  2. lianghui         两会窗口:  3月前两周 涨停家数均值 vs 全年均值
  3. earnings_april   业绩真空期: 4月中下旬(年报密集) 涨停家数/风格(大市值占比)
  4. earnings_oct     业绩真空期: 10月(三季报) 涨停家数/风格(大市值占比)
  5. year_end         年底排名战: 12月 机构重仓代理(大市值涨停股)占比与次日表现
  6. month_01..12     月度效应:   各月份涨停家数均值/中位数/胜率
  7. weekday_0..4     周一/周五效应: 星期几涨停家数分布

判定规则（verdict）:
  样本数(年) >= 3，否则 verdict=样本不足
  显著性: |差值| >= 20%（效应期均值 vs 基准均值 的相对差）
  胜率 >= 60% 且 显著性成立 且 方向符合假设 → verdict=支持（否则不支持）
  方向: 春节/两会/年底排名战 假设为正向；业绩真空期/月度/星期几 为描述性(±皆可)
  active_rules 只返回 verdict=支持 且 当日命中窗口 的效应
  仓位系数 multiplier = clamp(1 + 差值, 0.8, 1.2)

数据源:
  data_warehouse/market/zt_pool_history.parquet  涨停长表（is_zt 按日求和=涨停家数）
  data_warehouse/market/index_daily.parquet      上证指数日线（春节行情收益胜率用）
  float_mv 单位=亿元（>=100亿 视为大市值/机构重仓代理）

输出: generated/calendar_effects_{date}.json

用法:
  python3 -m quant_system.analysis_core.calendar_effects --verify   # 全效应统计验证
  python3 -m quant_system.analysis_core.calendar_effects --today    # 当日规则+落盘
  python3 -m quant_system.analysis_core.calendar_effects --date 2026-08-07
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.config import MARKET_DIR, ZT_HISTORY  # noqa: E402

CST = timezone(timedelta(hours=8))
INDEX_DAILY = MARKET_DIR / "index_daily.parquet"
OUT_DIR = ROOT / "generated"

MIN_YEARS = 3            # 最少样本年数
WIN_RATE_TH = 0.60       # 胜率阈值
SIGNIFICANT_DIFF = 0.20  # 显著性: |差值| >= 20%
MULT_LO, MULT_HI = 0.8, 1.2
BIG_CAP_MV_YI = 100.0    # 大市值代理: 流通市值 >= 100 亿元

# 公历近似窗口 (month1, day1, month2, day2)
SPRING_FEST_WIN = (1, 25, 2, 20)  # 春节: 1月底-2月中
LIANGHUI_WIN = (3, 1, 3, 14)      # 两会: 3月前两周
EARN_APR_WIN = (4, 11, 4, 30)     # 年报密集: 4月中下旬
EARN_OCT_WIN = (10, 1, 10, 31)    # 三季报: 10月
YEAR_END_MONTH = 12

WEEKDAY_NAMES = ["周一", "周二", "周三", "周四", "周五"]


# ────────────────────────────────────────────────────────────
# 数据加载（进程内缓存）
# ────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_daily() -> pd.DataFrame:
    """涨停家数日度序列（真实交易日，取自 zt_pool_history 的 date 列）。"""
    zt = pd.read_parquet(ZT_HISTORY)
    zt["date"] = pd.to_datetime(zt["date"]).dt.normalize()
    daily = zt.groupby("date", as_index=False)["is_zt"].agg(zt_cnt="sum")
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    daily["day"] = daily["date"].dt.day
    daily["weekday"] = daily["date"].dt.weekday
    return daily


@lru_cache(maxsize=1)
def _load_index() -> pd.DataFrame:
    """上证指数日线 + 日收益率。"""
    idx = pd.read_parquet(INDEX_DAILY)
    idx["date"] = pd.to_datetime(idx["date"]).dt.normalize()
    idx = idx.sort_values("date").reset_index(drop=True)
    idx["ret"] = idx["close"].pct_change()
    idx["year"] = idx["date"].dt.year
    return idx


@lru_cache(maxsize=1)
def _load_zt_style() -> pd.DataFrame:
    """涨停股日度风格: 大市值占比 big_share、大市值涨停股次日表现 big_next。"""
    zt = pd.read_parquet(ZT_HISTORY)
    zt["date"] = pd.to_datetime(zt["date"]).dt.normalize()
    zt = zt[zt["is_zt"]].copy()
    zt["big"] = zt["float_mv"].fillna(0.0) >= BIG_CAP_MV_YI
    share = zt.groupby("date")["big"].mean().rename("big_share")
    big_next = zt.loc[zt["big"]].groupby("date")["next_pct"].mean().rename("big_next")
    style = pd.concat([share, big_next], axis=1).reset_index()
    style["year"] = style["date"].dt.year
    return style


# ────────────────────────────────────────────────────────────
# 统计工具
# ────────────────────────────────────────────────────────────
def _between_mask(daily: pd.DataFrame, m1: int, d1: int, m2: int, d2: int) -> pd.Series:
    return ((daily["month"] == m1) & (daily["day"] >= d1)) | (
        (daily["month"] == m2) & (daily["day"] <= d2)
    )


def _window_vs_rest(daily: pd.DataFrame, mask: pd.Series) -> tuple[pd.Series, pd.Series, int]:
    """窗口日均涨停 vs 当年非窗口日均涨停（按年）。"""
    df = daily.copy()
    df["_win"] = mask
    win = df.loc[df["_win"]].groupby("year")["zt_cnt"].mean()
    base = df.loc[~df["_win"]].groupby("year")["zt_cnt"].mean()
    return win, base, int(df["_win"].sum())


def _window_vs_year(daily: pd.DataFrame, mask: pd.Series) -> tuple[pd.Series, pd.Series, int]:
    """窗口日均涨停 vs 全年均值（按年，全年均值含窗口，贴近 spec 口径）。"""
    df = daily.copy()
    df["_win"] = mask
    win = df.loc[df["_win"]].groupby("year")["zt_cnt"].mean()
    year = df.groupby("year")["zt_cnt"].mean()
    return win, year, int(df["_win"].sum())


def _year_win_rate(win: pd.Series, base: pd.Series) -> float:
    """按年胜率: 窗口年均值 > 基准年均值的年份占比。"""
    y = pd.DataFrame({"w": win, "b": base}).dropna()
    return float((y["w"] > y["b"]).mean()) if len(y) else float("nan")


def _judge(n_years: int, win_rate: float, diff: float, direction: float | None) -> str:
    if n_years < MIN_YEARS:
        return "样本不足"
    if not np.isfinite(diff) or not np.isfinite(win_rate):
        return "样本不足"
    significant = abs(diff) >= SIGNIFICANT_DIFF
    dir_ok = direction is None or (diff * direction) >= 0.0
    return "支持" if (win_rate >= WIN_RATE_TH and significant and dir_ok) else "不支持"


def _multiplier(diff: float) -> float:
    if not np.isfinite(diff):
        return 1.0
    return round(float(np.clip(1.0 + diff, MULT_LO, MULT_HI)), 2)


def _base_dict(name: str, win: pd.Series, base: pd.Series, n_days: int,
               win_rate: float, direction: float | None, note: str = "",
               eff: float | None = None, b: float | None = None) -> dict:
    """拼装统一输出结构。win/base 为按年均值 Series（用于胜率/样本数），
    eff/b 为池化均值（用于 效应期均值/基准均值/差值），缺省时退化为按年均值。"""
    y = pd.DataFrame({"w": win, "b": base}).dropna()
    n_years = len(y)
    eff = float(eff) if eff is not None else (float(y["w"].mean()) if len(y) else float("nan"))
    b = float(b) if b is not None else (float(y["b"].mean()) if len(y) else float("nan"))
    diff = (eff - b) / b if b else float("nan")
    verdict = _judge(n_years, win_rate, diff, direction)
    sig = bool(np.isfinite(diff) and abs(diff) >= SIGNIFICANT_DIFF)
    return {
        "效应名": name,
        "样本数": n_years,
        "样本天数": n_days,
        "效应期均值": round(eff, 2),
        "基准均值": round(b, 2),
        "差值": None if not np.isfinite(diff) else round(float(diff), 4),
        "显著性": sig,
        "胜率": None if not np.isfinite(win_rate) else round(float(win_rate), 4),
        "verdict": verdict,
        "multiplier": _multiplier(diff),
        "note": note,
        "reason": "",
    }


# ────────────────────────────────────────────────────────────
# 核心: 全效应统计验证（真实统计，groupby+mean）
# ────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def verify_effects() -> dict:
    daily = _load_daily()
    idx = _load_index()
    style = _load_zt_style()
    out: dict[str, dict] = {}

    # 1. 春节行情: 窗口涨停家数/上证指数 年胜率（方向=正向）
    mask = _between_mask(daily, *SPRING_FEST_WIN)
    win, base, n_days = _window_vs_rest(daily, mask)
    win_rate = _year_win_rate(win, base)
    eff_pool = float(daily.loc[mask, "zt_cnt"].mean())
    base_pool = float(daily.loc[~mask, "zt_cnt"].mean())
    diff = (eff_pool - base_pool) / base_pool if base_pool else float("nan")
    win_dates = set(daily.loc[mask, "date"])
    ii = idx[idx["year"].isin(win.index)].copy()
    ii["_win"] = ii["date"].isin(win_dates)
    win_r = ii.loc[ii["_win"]].groupby("year")["ret"].mean()
    base_r = ii.loc[~ii["_win"]].groupby("year")["ret"].mean()
    idx_win_rate = _year_win_rate(win_r, base_r)
    idx_eff = float(ii.loc[ii["_win"], "ret"].mean()) if ii["_win"].any() else float("nan")
    idx_base = float(ii.loc[~ii["_win"], "ret"].mean()) if (~ii["_win"]).any() else float("nan")
    idx_diff = (idx_eff - idx_base) / idx_base if idx_base else float("nan")
    d = _base_dict("春节行情", win, base, n_days, win_rate, 1.0, eff=eff_pool, b=base_pool)
    d["上证胜率"] = None if not np.isfinite(idx_win_rate) else round(float(idx_win_rate), 4)
    d["上证日均收益_效应期"] = None if not np.isfinite(idx_eff) else round(float(idx_eff), 5)
    d["上证日均收益_基准"] = None if not np.isfinite(idx_base) else round(float(idx_base), 5)
    d["reason"] = (
        f"春节窗口({SPRING_FEST_WIN[0]}/{SPRING_FEST_WIN[1]}-{SPRING_FEST_WIN[2]}/{SPRING_FEST_WIN[3]})"
        f"涨停日均 {eff_pool:.1f} vs 基准 {base_pool:.1f}（{diff:+.0%}），年胜率 {win_rate:.0%}；"
        f"上证日均收益 {idx_eff:.2%} vs {idx_base:.2%}，年胜率 {idx_win_rate:.0%}"
    )
    out["spring_festival"] = d

    # 2. 两会窗口: 3月前两周 涨停均值 vs 全年均值（方向=正向）
    mask = _between_mask(daily, *LIANGHUI_WIN)
    win, base, n_days = _window_vs_year(daily, mask)
    win_rate = _year_win_rate(win, base)
    eff_pool = float(daily.loc[mask, "zt_cnt"].mean())
    base_pool = float(daily["zt_cnt"].mean())
    diff = (eff_pool - base_pool) / base_pool if base_pool else float("nan")
    d = _base_dict("两会窗口", win, base, n_days, win_rate, 1.0, eff=eff_pool, b=base_pool)
    d["reason"] = (
        f"3月前两周涨停日均 {eff_pool:.1f} vs 全年均值 {base_pool:.1f}（{diff:+.0%}），"
        f"年胜率 {win_rate:.0%}"
    )
    out["lianghui"] = d

    # 3. 业绩真空期: 4月中下旬 / 10月（涨停家数 + 风格，方向=描述性）
    for key, name, w in (
        ("earnings_april", "业绩真空期-4月年报密集", EARN_APR_WIN),
        ("earnings_oct", "业绩真空期-10月三季报", EARN_OCT_WIN),
    ):
        mask = _between_mask(daily, *w)
        win, base, n_days = _window_vs_rest(daily, mask)
        win_rate = _year_win_rate(win, base)
        eff_pool = float(daily.loc[mask, "zt_cnt"].mean())
        base_pool = float(daily.loc[~mask, "zt_cnt"].mean())
        diff = (eff_pool - base_pool) / base_pool if base_pool else float("nan")
        win_dates = set(daily.loc[mask, "date"])
        st = style.copy()
        st["_win"] = st["date"].isin(win_dates)
        share_eff = float(st.loc[st["_win"], "big_share"].mean()) if st["_win"].any() else float("nan")
        share_base = float(st.loc[~st["_win"], "big_share"].mean()) if (~st["_win"]).any() else float("nan")
        share_diff = (share_eff - share_base) / share_base if share_base else float("nan")
        next_eff = float(st.loc[st["_win"], "big_next"].mean()) if st["_win"].any() else float("nan")
        next_base = float(st.loc[~st["_win"], "big_next"].mean()) if (~st["_win"]).any() else float("nan")
        d = _base_dict(name, win, base, n_days, win_rate, None, eff=eff_pool, b=base_pool)
        d["大市值占比_效应期"] = None if not np.isfinite(share_eff) else round(float(share_eff), 4)
        d["大市值占比_基准"] = None if not np.isfinite(share_base) else round(float(share_base), 4)
        d["大市值次日_效应期"] = None if not np.isfinite(next_eff) else round(float(next_eff), 4)
        d["大市值次日_基准"] = None if not np.isfinite(next_base) else round(float(next_base), 4)
        d["reason"] = (
            f"窗口({w[0]}/{w[1]}-{w[2]}/{w[3]})涨停日均 {eff_pool:.1f} vs 基准 {base_pool:.1f}"
            f"（{diff:+.0%}），年胜率 {win_rate:.0%}；"
            f"大市值占比 {share_eff:.1%} vs {share_base:.1%}（{share_diff:+.0%}），"
            f"大市值涨停次日 {next_eff:.2%} vs {next_base:.2%}"
        )
        out[key] = d

    # 4. 年底排名战: 12月 大市值(机构重仓代理)涨停股占比/次日表现（方向=正向）
    mask = daily["month"] == YEAR_END_MONTH
    win, base, n_days = _window_vs_rest(daily, mask)
    win_dates = set(daily.loc[mask, "date"])
    st = style.copy()
    st["_win"] = st["date"].isin(win_dates)
    share_win = st.loc[st["_win"]].groupby("year")["big_share"].mean()
    share_base = st.loc[~st["_win"]].groupby("year")["big_share"].mean()
    win_rate = _year_win_rate(share_win, share_base)
    share_eff = float(st.loc[st["_win"], "big_share"].mean())
    share_base_v = float(st.loc[~st["_win"], "big_share"].mean())
    diff = (share_eff - share_base_v) / share_base_v if share_base_v else float("nan")
    next_eff = float(st.loc[st["_win"], "big_next"].mean())
    next_base = float(st.loc[~st["_win"], "big_next"].mean())
    d = _base_dict("年底排名战", share_win, share_base, n_days, win_rate, 1.0,
                   note="主指标=大市值涨停股占比", eff=share_eff, b=share_base_v)
    d["大市值占比_效应期"] = round(share_eff, 4)
    d["大市值占比_基准"] = round(share_base_v, 4)
    d["大市值次日_效应期"] = round(next_eff, 4) if np.isfinite(next_eff) else None
    d["大市值次日_基准"] = round(next_base, 4) if np.isfinite(next_base) else None
    d["涨停日均_12月"] = round(float(daily.loc[mask, "zt_cnt"].mean()), 2)
    d["涨停日均_基准"] = round(float(daily.loc[~mask, "zt_cnt"].mean()), 2)
    d["reason"] = (
        f"12月大市值涨停占比 {share_eff:.1%} vs 基准 {share_base_v:.1%}（{diff:+.0%}），"
        f"年胜率 {win_rate:.0%}；大市值涨停次日 {next_eff:.2%} vs {next_base:.2%}"
    )
    out["year_end"] = d

    # 5. 月度效应: 各月 涨停均值/中位数/胜率（vs 全年均值，方向=描述性）
    for m in range(1, 13):
        mask = daily["month"] == m
        win, base, n_days = _window_vs_year(daily, mask)
        win_rate = _year_win_rate(win, base)
        eff_pool = float(daily.loc[mask, "zt_cnt"].mean())
        base_pool = float(daily["zt_cnt"].mean())
        diff = (eff_pool - base_pool) / base_pool if base_pool else float("nan")
        d = _base_dict(f"{m}月效应", win, base, n_days, win_rate, None, eff=eff_pool, b=base_pool)
        d["中位数"] = float(daily.loc[mask, "zt_cnt"].median())
        d["reason"] = (
            f"{m}月涨停日均 {eff_pool:.1f} vs 全年均值 {base_pool:.1f}（{diff:+.0%}），"
            f"中位数 {d['中位数']:.0f}，年胜率 {win_rate:.0%}"
        )
        out[f"month_{m:02d}"] = d

    # 6. 周一/周五效应: 星期几 涨停分布（vs 全样本日均，方向=描述性）
    overall_mean = float(daily["zt_cnt"].mean())
    overall_median = float(daily["zt_cnt"].median())
    for wd in range(5):
        mask = daily["weekday"] == wd
        n_days = int(mask.sum())
        n_years = int(daily.loc[mask, "year"].nunique())
        eff = float(daily.loc[mask, "zt_cnt"].mean())
        diff = (eff - overall_mean) / overall_mean if overall_mean else float("nan")
        win_rate = float((daily.loc[mask, "zt_cnt"] > overall_mean).mean())
        verdict = _judge(n_years, win_rate, diff, direction=None)
        sig = bool(np.isfinite(diff) and abs(diff) >= SIGNIFICANT_DIFF)
        d = {
            "效应名": f"{WEEKDAY_NAMES[wd]}效应",
            "样本数": n_years,
            "样本天数": n_days,
            "效应期均值": round(eff, 2),
            "基准均值": round(overall_mean, 2),
            "中位数": float(daily.loc[mask, "zt_cnt"].median()),
            "差值": None if not np.isfinite(diff) else round(float(diff), 4),
            "显著性": sig,
            "胜率": round(float(win_rate), 4),
            "verdict": verdict,
            "multiplier": _multiplier(diff),
            "note": f"全样本日均 {overall_mean:.1f} / 中位 {overall_median:.0f}",
            "reason": (
                f"{WEEKDAY_NAMES[wd]}涨停日均 {eff:.1f} vs 全样本日均 {overall_mean:.1f}"
                f"（{diff:+.0%}），中位数 {d['中位数']:.0f}，超过日均的胜率 {win_rate:.0%}"
            ),
        }
        out[f"weekday_{wd}"] = d

    return out


# ────────────────────────────────────────────────────────────
# 当日规则 + 落盘
# ────────────────────────────────────────────────────────────
def _hit(key: str, target: pd.Timestamp) -> bool:
    m, dd, wd = target.month, target.day, target.weekday()
    if key == "spring_festival":
        return (m == SPRING_FEST_WIN[0] and dd >= SPRING_FEST_WIN[1]) or (
            m == SPRING_FEST_WIN[2] and dd <= SPRING_FEST_WIN[3])
    if key == "lianghui":
        return m == LIANGHUI_WIN[0] and dd <= LIANGHUI_WIN[3]
    if key == "earnings_april":
        return m == EARN_APR_WIN[0] and dd >= EARN_APR_WIN[1]
    if key == "earnings_oct":
        return m == EARN_OCT_WIN[0]
    if key == "year_end":
        return m == YEAR_END_MONTH
    if key.startswith("month_"):
        return m == int(key.split("_")[1])
    if key.startswith("weekday_"):
        return wd == int(key.split("_")[1])
    return False


def active_rules(date: str | None = None) -> list[dict]:
    """当日命中的已验证效应。只返回 verdict=支持 的效应。"""
    target = pd.Timestamp(date) if date else pd.Timestamp.now(CST).normalize()
    rules = []
    for key, v in verify_effects().items():
        if v["verdict"] != "支持":
            continue
        if _hit(key, target):
            rules.append({
                "effect": v["效应名"],
                "key": key,
                "multiplier": v["multiplier"],
                "reason": v["reason"],
            })
    return rules


def run_today() -> dict:
    today = pd.Timestamp.now(CST).normalize()
    date_s = str(today.date())
    verify = verify_effects()
    out = {
        "date": date_s,
        "active_rules": active_rules(date_s),
        "verified": {k: v["verdict"] for k, v in verify.items()},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"calendar_effects_{date_s}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[calendar_effects] 已输出 {p}")
    return out


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="A股日历效应规则库（统计验证+当日规则）")
    ap.add_argument("--verify", action="store_true", help="全效应统计验证")
    ap.add_argument("--today", action="store_true", help="生成当日规则并落盘")
    ap.add_argument("--date", default=None, help="指定日期(YYYY-MM-DD)的命中规则")
    args = ap.parse_args()

    if args.verify or not (args.today or args.date):
        r = verify_effects()
        print(f"{'key':<18}{'效应':<20}{'verdict':<6}{'样本':<8}{'胜率':<8}{'差值':<8}{'mult'}")
        for k, v in r.items():
            diff = f"{v['差值']:+.0%}" if isinstance(v["差值"], (int, float)) else "  -"
            wr = f"{v['胜率']:.0%}" if isinstance(v["胜率"], (int, float)) else "  -"
            print(f"{k:<18}{v['效应名']:<20}{v['verdict']:<6}"
                  f"{v['样本数']}年/{v['样本天数']}天{wr:>8}{diff:>8}{v['multiplier']:>7.2f}")
        print("\n支持效应 reason:")
        for k, v in r.items():
            if v["verdict"] == "支持":
                print(f"  [{k}] {v['reason']}")

    if args.today:
        print(json.dumps(run_today(), ensure_ascii=False, indent=2))
    elif args.date:
        print(json.dumps({"date": args.date, "active_rules": active_rules(args.date)},
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
