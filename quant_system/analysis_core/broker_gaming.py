"""
broker_gaming — 游资席位博弈层（V11 短线 M6）

三层能力:
  1. 席位画像升级（滚动）: 活跃度/风格标签/偏好市值/近20·60日胜率/风格漂移
     - 数据: lhb_hyyyb_em.parquet(席位日聚合) × lhb_20*.parquet(个股明细含上榜后5日)
  2. 个股席位博弈: 买榜前5 vs 卖榜前5 的席位类型结构判定
     - 数据: lhb_stock_detail_daily.parquet（每日增量累积, stock_lhb_stock_detail_em）
  3. 三日榜追溯: 前2日买榜席位今日是否在卖榜 → 锁仓/换手/出货

用法:
  python3 -m quant_system.analysis_core.broker_gaming --profiles        # 全量画像
  python3 -m quant_system.analysis_core.broker_gaming --fetch-detail 10 # 回补近N交易日个股席位明细
  python3 -m quant_system.analysis_core.broker_gaming --gaming 20260807 000603
"""

from __future__ import annotations
import logging

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.config import MARKET_DIR  # noqa: E402
from quant_system.analysis_core.zt_pool_history import ensure_name_map  # noqa: E402
from quant_system.cpu_throttle import CpuThrottle  # noqa: E402

CST = timezone(timedelta(hours=8))
throttle = CpuThrottle()

LHB_BROKER = MARKET_DIR / "lhb_hyyyb_em.parquet"
LHB_DETAIL_DAILY = MARKET_DIR / "lhb_stock_detail_daily.parquet"
PROFILES_OUT = ROOT / "generated" / "broker_profiles.parquet"
PROFILES_JSON = ROOT / "generated" / "broker_profiles.json"

# ── 席位类型规则（可扩展）──────────────────────────────
SEAT_TYPE_RULES: list[tuple[tuple[str, ...], str]] = [
    (("机构专用",), "机构"),
    (("股通专用",), "北向"),
    (("量化",), "量化"),
    (("瑞银证券", "摩根大通", "高盛", "美林", "花旗"), "外资"),
]
KNOWN_YOUZI = ()  # 可维护知名游资名单，命中优先判定为游资


def classify_seat(name: str) -> str:
    """营业部名称 → 席位类型（机构/北向/量化/外资/游资/其他）。"""
    for kws, t in SEAT_TYPE_RULES:
        if any(k in name for k in kws):
            return t
    if any(k in name for k in KNOWN_YOUZI):
        return "游资"
    if "证券" in name or "营业部" in name:
        return "游资"
    return "其他"


def _load_lhb_detail_all() -> pd.DataFrame:
    """合并全部季度龙虎榜个股明细。"""
    files = sorted(MARKET_DIR.glob("lhb_20*.parquet"))
    frames = []
    for f in files:
        try:
            frames.append(pd.read_parquet(f))
        except Exception as e:
            logging.getLogger(__name__).error(f"[broker_gaming] 操作失败: {e}", exc_info=True)
            continue
    if not frames:
        raise FileNotFoundError("未找到 lhb_20*.parquet 明细文件")
    df = pd.concat(frames, ignore_index=True)
    df["代码"] = df["代码"].astype(str).str.zfill(6)
    df["上榜日"] = pd.to_datetime(df["上榜日"])
    return df


# ────────────────────────────────────────────────────────────
# 1. 席位画像（滚动刷新）
# ────────────────────────────────────────────────────────────
def build_profiles(verbose: bool = True) -> pd.DataFrame:
    """全量/滚动刷新席位画像 → generated/broker_profiles.parquet。"""
    broker = pd.read_parquet(LHB_BROKER)
    broker["上榜日"] = pd.to_datetime(broker["上榜日"])
    detail = _load_lhb_detail_all()
    nm = ensure_name_map()
    name2code = dict(zip(nm["name"], nm["code"]))

    # 名字 → 代码（买入股票字段是空格分隔的名称串）
    def names_to_codes(s: str) -> list[str]:
        if pd.isna(s):
            return []
        return [name2code.get(n) for n in str(s).split() if name2code.get(n)]

    broker["buy_codes"] = broker["买入股票"].map(names_to_codes)
    detail_fwd = detail[["代码", "上榜日", "上榜后5日", "流通市值", "换手率", "涨跌幅"]]
    # 轻量索引: (code, date) → tuple（避免 dict of Series 撑爆内存）
    arr = detail_fwd.to_records(index=False)
    idx = {(r["代码"], r["上榜日"]): (r["上榜后5日"], r["流通市值"], r["换手率"], r["涨跌幅"]) for r in arr}
    del detail_fwd, arr

    now = datetime.now(CST).replace(tzinfo=None)
    broker["days_ago"] = (now - broker["上榜日"]).dt.days
    broker["net_buy_flag"] = broker["总买卖净额"] > 0

    def _feat(g: pd.DataFrame) -> pd.DataFrame:
        feat = []
        for _, r in g.loc[g["net_buy_flag"]].head(60).iterrows():
            for c in r["buy_codes"][:8]:
                hit = idx.get((c, r["上榜日"]))
                if hit is not None:
                    feat.append(hit)
        if not feat:
            return pd.DataFrame(columns=["fwd5", "mv", "tr", "pct"])
        return pd.DataFrame(feat, columns=["fwd5", "mv", "tr", "pct"])

    rows = []
    for bname, g in broker.groupby("营业部名称"):
        g20 = g[g["days_ago"] <= 20]
        g60 = g[g["days_ago"] <= 60]
        g120 = g[g["days_ago"] <= 120]
        g5 = g[g["days_ago"] <= 5]

        feat20 = _feat(g20)
        feat60 = _feat(g60)

        # 胜率（净买入股上榜后5日涨幅>0）
        def winrate(sub: pd.DataFrame) -> float | None:
            if sub.empty:
                return None
            fwd = sub["fwd5"].dropna()
            return float((fwd > 0).mean()) if len(fwd) >= 5 else None

        avg_mv = float(feat60["mv"].mean()) if len(feat60) else np.nan
        avg_tr = float(feat60["tr"].mean()) if len(feat60) else np.nan
        avg_pct = float(feat60["pct"].mean()) if len(feat60) else np.nan

        # 风格标签
        if not np.isnan(avg_tr):
            if avg_tr > 12 and (np.isnan(avg_mv) or avg_mv < 150):
                style = "打板型"
            elif avg_mv > 250 and avg_tr < 8:
                style = "机构型"
            elif avg_pct < 2.5:
                style = "低吸型"
            elif 3 <= avg_pct <= 10:
                style = "趋势型"
            else:
                style = "混合型"
        else:
            style = "未知"

        # 风格漂移: 近20日 vs 近120日 活跃度/净买占比
        drift = ""
        if len(g20) >= 3 and len(g120) >= 10:
            net20 = float(g20["总买卖净额"].sum())
            net120 = float(g120["总买卖净额"].sum())
            if net120 != 0 and abs(net20 / net120) > 1.2:
                drift = "加仓加速" if net20 * net120 > 0 else "反向操作"

        rows.append({
            "seat": bname,
            "seat_code": g["营业部代码"].iloc[-1] if "营业部代码" in g.columns else "",
            "seat_type": classify_seat(bname),
            "total_cnt": len(g),
            "cnt_20d": len(g20),
            "cnt_60d": len(g60),
            "act_trend": round(len(g5) / max(len(g60) / 12, 1e-9), 2),  # 近5日 vs 60日均频
            "net_buy_20d": round(float(g20["总买卖净额"].sum()), 0),
            "net_buy_all": round(float(g["总买卖净额"].sum()), 0),
            "style": style,
            "avg_float_mv": round(avg_mv, 1) if not np.isnan(avg_mv) else None,
            "avg_turnover": round(avg_tr, 2) if not np.isnan(avg_tr) else None,
            "winrate_20d": winrate(feat20),
            "winrate_60d": winrate(feat60),
            "samples_60d": len(feat60),
            "drift_flag": drift,
            "last_active": str(g["上榜日"].max().date()),
        })

    out = pd.DataFrame(rows).sort_values("cnt_60d", ascending=False).reset_index(drop=True)
    PROFILES_OUT.parent.mkdir(exist_ok=True)
    out.to_parquet(PROFILES_OUT, index=False)
    out.to_json(PROFILES_JSON, orient="records", force_ascii=False, indent=1)
    if verbose:
        print(f"[profiles] {len(out)} 席位已画像 → {PROFILES_OUT}")
    return out


# ────────────────────────────────────────────────────────────
# 2. 个股席位博弈（每日增量累积）
# ────────────────────────────────────────────────────────────
def fetch_stock_detail(date_str: str, top_n: int = 30) -> int:
    """抓某交易日龙虎榜个股的买5/卖5 席位明细，滚动累积。"""
    import akshare as ak
    d = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    lst = ak.stock_lhb_detail_em(start_date=d, end_date=d)
    if lst is None or lst.empty:
        return 0
    lst["代码"] = lst["代码"].astype(str).str.zfill(6)
    lst = lst.sort_values("龙虎榜净买额", ascending=False)
    top = lst.drop_duplicates("代码").head(top_n)

    frames = []
    for _, r in top.iterrows():
        try:
            # 2026-08-10 审计(Critical): 接口默认 flag="卖出" 只返回卖榜，
            # 必须分别抓 买入/卖出 两榜合并，否则买榜前5从未入库，
            # 个股席位博弈与三日榜追溯整体失真。
            dd_buy = ak.stock_lhb_stock_detail_em(symbol=r["代码"], date=date_str, flag="买入")
            dd_sell = ak.stock_lhb_stock_detail_em(symbol=r["代码"], date=date_str, flag="卖出")
            dd = pd.concat([dd_buy, dd_sell], ignore_index=True) if (dd_buy is not None and not dd_buy.empty) else dd_sell
        except Exception as e:
            logging.getLogger(__name__).error(f"[broker_gaming] 操作失败: {e}", exc_info=True)
            continue
        if dd is None or dd.empty:
            continue
        dd = dd.copy()
        dd["date"] = pd.Timestamp(d)
        dd["code"] = r["代码"]
        dd["name"] = r["名称"]
        dd = dd.rename(columns={
            "交易营业部名称": "seat", "买入金额": "buy_amt",
            "卖出金额": "sell_amt", "净额": "net_amt", "类型": "reason"})
        keep = [c for c in ["date", "code", "name", "seat", "buy_amt", "sell_amt",
                            "net_amt", "reason"] if c in dd.columns]
        dd["seat_type"] = dd["seat"].map(classify_seat)
        frames.append(dd[keep])
        throttle.sleep(default_ms=400)

    if not frames:
        return 0
    new = pd.concat(frames, ignore_index=True)
    if LHB_DETAIL_DAILY.exists():
        old = pd.read_parquet(LHB_DETAIL_DAILY)
        new = pd.concat([old, new], ignore_index=True)
    # 2026-08-10 审计(Major): 同一席位同日买卖两榜会出现两行，
    # keep=first 会丢数据 → 改为按 (date,code,seat) 聚合求和
    if not new.empty:
        agg = new.groupby(["date", "code", "name", "seat", "seat_type", "reason"], as_index=False).agg(
            buy_amt=("buy_amt", "sum"), sell_amt=("sell_amt", "sum"), net_amt=("net_amt", "sum"))
        new = agg
    new = new.drop_duplicates(subset=["date", "code", "seat"]).reset_index(drop=True)
    new.to_parquet(LHB_DETAIL_DAILY, index=False)
    print(f"[detail] {date_str}: {len(new)} 条席位记录（累计）")
    return len(new)


def fetch_detail_window(days: int = 10) -> int:
    """回补近 N 个交易日个股席位明细（滚动去重）。"""
    import akshare as ak
    cal = ak.tool_trade_date_hist_sina()
    cal = pd.to_datetime(cal["trade_date"])
    today = pd.Timestamp(datetime.now(CST).date())
    cal = cal[(cal <= today) & (cal >= today - pd.Timedelta(days=days * 2 + 5))]
    total = 0
    for d in cal:
        total += fetch_stock_detail(d.strftime("%Y%m%d"))
        throttle.sleep(default_ms=500)
    return total


# ────────────────────────────────────────────────────────────
# 3. 个股博弈结构 + 三日榜追溯
# ────────────────────────────────────────────────────────────
def stock_gaming(date_str: str, code: str) -> dict:
    if not LHB_DETAIL_DAILY.exists():
        raise FileNotFoundError("先运行 --fetch-detail 累积个股席位明细")
    df = pd.read_parquet(LHB_DETAIL_DAILY)
    d = pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}")
    day = df[(df["date"] == d) & (df["code"] == code)]
    if day.empty:
        return {"code": code, "date": date_str, "note": "无席位数据"}

    buy5 = day[day["buy_amt"] > 0].sort_values("buy_amt", ascending=False).head(5)
    sell5 = day[day["sell_amt"] > 0].sort_values("sell_amt", ascending=False).head(5)
    buy_types = Counter(buy5["seat_type"])
    sell_types = Counter(sell5["seat_type"])

    # 结构判定
    if buy_types["游资"] >= 3 and sell_types["量化"] >= 1:
        structure = "游资扎堆买 + 量化在卖 → 游资接量化的货（偏多）"
    elif buy_types["机构"] >= 3 and sell_types["游资"] >= 2:
        structure = "机构扎堆买 + 游资在卖 → 机构接游资的货（中长线认可/短线分歧）"
    elif buy_types["游资"] >= 3:
        structure = "游资扎堆买（情绪票，短线博弈）"
    elif buy_types["机构"] >= 3:
        structure = "机构扎堆买（机构票，趋势为主）"
    elif buy_types["量化"] >= 2:
        structure = "量化主导（做T/套利，参考价值低）"
    else:
        structure = "混合结构"

    # 三日榜追溯（前2个交易日买榜席位是否今日卖出）
    trace = []
    prev_days = sorted(df[df["code"] == code]["date"].unique())
    prev_days = [x for x in prev_days if x < d][-2:]
    prev_buyers: set[str] = set()
    for pd_ in prev_days:
        p = df[(df["date"] == pd_) & (df["code"] == code) & (df["buy_amt"] > 0)]
        prev_buyers |= set(p["seat"])
    today_sellers = set(sell5["seat"])
    dumping = prev_buyers & today_sellers
    # 2026-08-10 审计(Major): 无历史买榜数据时 lock_rate=1.0 属"有输出但错误"，
    # 应显式标注无数据而非给出最强肯定结论
    if not prev_buyers:
        lock_rate = None
        lock_verdict = "无历史买榜数据"
    else:
        lock_rate = 1 - len(dumping) / len(prev_buyers)
        lock_verdict = "锁仓健康" if lock_rate >= 0.78 else ("换手" if lock_rate >= 0.5 else "筹码松动/出货")

    return {
        "code": code, "date": date_str,
        "buy5": buy5[["seat", "seat_type", "buy_amt"]].to_dict("records"),
        "sell5": sell5[["seat", "seat_type", "sell_amt"]].to_dict("records"),
        "buy_types": dict(buy_types), "sell_types": dict(sell_types),
        "structure": structure,
        "prev_buyers": sorted(prev_buyers)[:10],
        "dumping_seats": sorted(dumping),
        "lock_rate": round(lock_rate, 2) if lock_rate is not None else None,
        "lock_verdict": lock_verdict,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="游资席位博弈")
    ap.add_argument("--profiles", action="store_true")
    ap.add_argument("--fetch-detail", type=int, default=0)
    ap.add_argument("--gaming", nargs=2, metavar=("DATE", "CODE"))
    args = ap.parse_args()

    if args.profiles:
        build_profiles()
    if args.fetch_detail:
        fetch_detail_window(days=args.fetch_detail)
    if args.gaming:
        print(json.dumps(stock_gaming(args.gaming[0], args.gaming[1]), ensure_ascii=False, indent=2))
    if not (args.profiles or args.fetch_detail or args.gaming):
        ap.print_help()
