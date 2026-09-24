"""
broker_profile_deep — 席位行为画像深化（V11）

在 broker_gaming.build_profiles() 的静态画像（style / winrate_20d / winrate_60d 等）之上，
把席位升级为"行为预测模型":

  style_breakdown(seat)  → 操作风格分布 / 持仓周期 / 板块偏好 / 胜率
  interpret(seat, date)  → 当日净买净卖 + 一句话行为解读
  hot_seats(date, n)     → 当日最活跃席位 + 风格标签

数据（全部本地，禁止网络）:
  lhb_hyyyb_em.parquet       席位日聚合（营业部名称/上榜日/总买卖净额/买入股票）
  lhb_20*.parquet            个股明细（代码/上榜日/涨跌幅/龙虎榜净买额/上榜后1·2·5·10日）
  zt_pool_em_daily.parquet   EM涨停池（board_count/industry）
  generated/broker_profiles.parquet  已有画像（winrate_20d/60d、style）

用法:
  python3 -m quant_system.analysis_core.broker_profile_deep --style "某营业部"
  python3 -m quant_system.analysis_core.broker_profile_deep --interpret "某营业部" 20260807
  python3 -m quant_system.analysis_core.broker_profile_deep --hot 20260807 5
"""

from __future__ import annotations
import logging

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.config import MARKET_DIR, ZT_EM_DAILY  # noqa: E402
from quant_system.analysis_core.zt_pool_history import ensure_name_map  # noqa: E402

CST = timezone(timedelta(hours=8))
WINDOW_DAYS = 120  # 风格统计窗口（自然日）
_CACHE_VERSION = "1"  # 缓存键版本，代码变更后自动失效旧缓存

LHB_BROKER = MARKET_DIR / "lhb_hyyyb_em.parquet"
LHB_DETAIL_DAILY = MARKET_DIR / "lhb_stock_detail_daily.parquet"
PROFILES_OUT = ROOT / "generated" / "broker_profiles.parquet"
DEEP_DIR = ROOT / "generated"


# ── 数据加载（进程内缓存）──────────────────────────────
@lru_cache(maxsize=1)
def _load_broker() -> pd.DataFrame:
    """席位日聚合表。"""
    df = pd.read_parquet(LHB_BROKER)
    df["上榜日"] = pd.to_datetime(df["上榜日"])
    for c in ("买入总金额", "卖出总金额", "总买卖净额"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@lru_cache(maxsize=1)
def _load_detail() -> pd.DataFrame:
    """合并全部季度龙虎榜个股明细（按 代码+上榜日 去重，保留首条原因行）。"""
    frames = []
    for f in sorted(MARKET_DIR.glob("lhb_20*.parquet")):
        try:
            frames.append(pd.read_parquet(f))
        except Exception as e:
            logging.getLogger(__name__).error(f"[broker_profile_deep] 操作失败: {e}", exc_info=True)
            continue
    if not frames:
        raise FileNotFoundError("未找到 lhb_20*.parquet 明细文件")
    df = pd.concat(frames, ignore_index=True)
    df["代码"] = df["代码"].astype(str).str.zfill(6)
    df["上榜日"] = pd.to_datetime(df["上榜日"])
    df = df.drop_duplicates(subset=["代码", "上榜日"], keep="first").reset_index(drop=True)
    return df


@lru_cache(maxsize=1)
def _load_zt_em() -> pd.DataFrame:
    """EM 涨停/炸板/跌停池（board_count / industry）。"""
    df = pd.read_parquet(ZT_EM_DAILY)
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "code", "board_count", "industry"]].drop_duplicates(subset=["date", "code"])


@lru_cache(maxsize=1)
def _load_profiles() -> pd.DataFrame:
    """broker_gaming 生成的静态画像（只读引用）。"""
    if PROFILES_OUT.exists():
        return pd.read_parquet(PROFILES_OUT)
    return pd.DataFrame()


@lru_cache(maxsize=1)
def _detail_index() -> dict[tuple[str, object], tuple]:
    """(代码, 上榜日.date) → (涨跌幅, 龙虎榜净买额, 上榜后1/2/5/10日)。"""
    df = _load_detail()
    need = ["代码", "上榜日", "涨跌幅", "龙虎榜净买额", "上榜后1日", "上榜后2日", "上榜后5日", "上榜后10日"]
    df = df[[c for c in need if c in df.columns]]
    idx: dict[tuple[str, object], tuple] = {}
    for r in df.to_records(index=False):
        key = (str(r["代码"]), pd.Timestamp(r["上榜日"]).date())
        idx[key] = (
            _f(r["涨跌幅"]), _f(r["龙虎榜净买额"]),
            _f(r["上榜后1日"]), _f(r["上榜后2日"]),
            _f(r["上榜后5日"]), _f(r["上榜后10日"]),
        )
    return idx


@lru_cache(maxsize=1)
def _zt_index() -> dict[tuple[str, object], tuple]:
    """(代码, date) → (board_count, industry)。"""
    df = _load_zt_em()
    idx: dict[tuple[str, object], tuple] = {}
    for r in df.to_records(index=False):
        idx[(str(r["code"]), pd.Timestamp(r["date"]).date())] = (int(r["board_count"]) if not pd.isna(r["board_count"]) else 0, r["industry"])
    return idx


@lru_cache(maxsize=1)
def _name2code() -> dict[str, str]:
    nm = ensure_name_map()
    return dict(zip(nm["name"], nm["code"]))


# ── 基础工具 ─────────────────────────────────────────────
def _f(x: object) -> float | None:
    """安全转 float，NaN → None。"""
    try:
        v = float(x)
        return round(v, 4) if not np.isnan(v) else None
    except (TypeError, ValueError):
        return None


def _today_str() -> str:
    return datetime.now(CST).strftime("%Y-%m-%d")


def _parse_date(date: str) -> pd.Timestamp:
    d = str(date).replace("-", "")
    return pd.Timestamp(f"{d[:4]}-{d[4:6]}-{d[6:8]}")


def _zt_pct_threshold(code: str) -> float:
    """按代码前缀给出涨停判定阈值（百分点）。"""
    if code[:2] in ("30", "68"):
        return 19.5
    if code[0] in ("8", "4", "9"):
        return 29.5
    return 9.7


def _classify_style(net: float | None, pct: float | None, board_count: int, is_zt: bool, net_th: float) -> str:
    """单笔操作风格: 接力 > 首板打板 > 趋势 > 低吸。"""
    if board_count >= 2:
        return "接力"
    if is_zt and board_count == 1:
        return "首板打板"
    if net is None:
        return "低吸"
    if pct is not None and pct < 0:
        return "低吸"  # 跌买
    return "趋势" if net >= net_th else "低吸"


def _seat_window(seat: str, days: int = WINDOW_DAYS) -> pd.DataFrame:
    """该席位近 N 自然日席位日聚合记录。"""
    broker = _load_broker()
    cutoff = pd.Timestamp(datetime.now(CST).replace(tzinfo=None) - timedelta(days=days))
    g = broker[(broker["营业部名称"] == seat) & (broker["上榜日"] >= cutoff)]
    return g


def _seat_ops(seat: str, days: int = WINDOW_DAYS) -> list[dict]:
    """展开席位近窗口内每笔(上榜股)操作，附风格/持仓周期/板块信息。"""
    g = _seat_window(seat, days)
    if g.empty:
        return []
    di, zi = _detail_index(), _zt_index()
    n2c = _name2code()
    ops: list[dict] = []
    for _, r in g.iterrows():
        d = r["上榜日"].date()
        names = [] if pd.isna(r["买入股票"]) else str(r["买入股票"]).split()
        for n in names[:8]:
            c = n2c.get(n)
            if not c:
                continue
            hit = di.get((c, d))
            if hit is None:
                continue
            pct, net, f1, f2, f5, f10 = hit
            board_count, industry = zi.get((c, d), (0, None))
            is_zt = board_count >= 1 or (pct is not None and pct >= _zt_pct_threshold(c))
            ops.append({
                "code": c, "name": n, "date": d,
                "pct": pct, "net": net,
                "fwd": {"1": f1, "2": f2, "5": f5, "10": f10},
                "board_count": int(board_count) if board_count else 0,
                "is_zt": bool(is_zt),
                "industry": industry,
            })
    return ops


# ── 缓存（seat+date 哈希，避免重复计算）────────────────
def _cache_path(seat: str, date: str = "") -> Path:
    h = hashlib.md5(f"{seat}|{date}|{_CACHE_VERSION}".encode("utf-8")).hexdigest()[:16]
    return DEEP_DIR / f"broker_deep_{h}.json"


def _cache_get(path: Path) -> dict | None:
    try:
        if path.exists():
            with open(path, encoding="utf-8") as fp:
                return json.load(fp)
    except Exception as e:
        logging.getLogger(__name__).error(f"[broker_profile_deep] 操作失败: {e}", exc_info=True)
    return None


def _cache_set(path: Path, obj: dict) -> None:
    try:
        DEEP_DIR.mkdir(exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(obj, fp, ensure_ascii=False, indent=1)
    except Exception as e:
        logging.getLogger(__name__).error(f"[broker_profile_deep] 操作失败: {e}", exc_info=True)


# ─────────────────────────────────────────────────────────
# 1. 席位行为画像
# ─────────────────────────────────────────────────────────
def style_breakdown(seat: str) -> dict:
    """席位历史行为分布: 操作风格 / 持仓周期 / 板块偏好 / 胜率。

    近 {WINDOW_DAYS} 自然日窗口; 无历史样本时返回 note 不崩溃。
    """
    cache = _cache_get(_cache_path(seat, _today_str()))
    if cache is not None:
        return cache

    ops = _seat_ops(seat)
    if not ops:
        out = {"seat": seat, "note": "无历史样本", "window_days": WINDOW_DAYS}
        _cache_set(_cache_path(seat, _today_str()), out)
        return out

    # 净买大/小的分界: 该席位自身窗口内单股净买中位数
    nets = [o["net"] for o in ops if o["net"] is not None]
    net_th = float(np.median(nets)) if nets else 0.0

    styles: Counter[str] = Counter()
    for o in ops:
        styles[_classify_style(o["net"], o["pct"], o["board_count"], o["is_zt"], net_th)] += 1
    total = len(ops)
    style_dist = {k: styles[k] for k in sorted(styles, key=lambda x: -styles[x])}
    style_pct = {k: round(v / total * 100, 1) for k, v in style_dist.items()}

    # 持仓周期: 上榜后 1/2/5/10 日涨跌幅均值
    fwd_keys = ["1", "2", "5", "10"]
    holding: dict = {}
    for k in fwd_keys:
        vals = [o["fwd"][k] for o in ops if o["fwd"].get(k) is not None]
        holding[f"fwd_{k}d"] = round(float(np.mean(vals)), 4) if vals else None
        holding[f"fwd_{k}d_n"] = len(vals)
    holding["n"] = total

    # 板块偏好: EM 池 industry top3
    ind_cnt = Counter(o["industry"] for o in ops if o["industry"])
    industry_top3 = [{"industry": k, "count": v} for k, v in ind_cnt.most_common(3)]

    # 胜率: 直接引用已有画像
    prof = _load_profiles()
    prow = prof[prof["seat"] == seat]
    winrate = {
        "winrate_20d": _f(prow["winrate_20d"].iloc[0]) if len(prow) else None,
        "winrate_60d": _f(prow["winrate_60d"].iloc[0]) if len(prow) else None,
        "samples_60d": int(prow["samples_60d"].iloc[0]) if len(prow) and not pd.isna(prow["samples_60d"].iloc[0]) else None,
        "style_profile": str(prow["style"].iloc[0]) if len(prow) else None,
        "last_active": str(prow["last_active"].iloc[0]) if len(prow) else None,
    }

    out = {
        "seat": seat,
        "note": None,
        "window_days": WINDOW_DAYS,
        "window_start": str((datetime.now(CST).replace(tzinfo=None) - timedelta(days=WINDOW_DAYS)).date()),
        "window_end": _today_str(),
        "ops_n": total,
        "style_dist": style_dist,
        "style_pct": style_pct,
        "dominant_style": max(style_dist, key=style_dist.get) if style_dist else None,
        "holding_period": holding,
        "industry_top3": industry_top3,
        "winrate": winrate,
        "net_threshold": round(net_th, 2),
    }
    _cache_set(_cache_path(seat, _today_str()), out)
    return out


# ─────────────────────────────────────────────────────────
# 2. 当日行为解读
# ─────────────────────────────────────────────────────────
def interpret(seat: str, date: str) -> dict:
    """对"该席位今日出现在某股"给出解读: 当日净买/净卖 + 一句话结论。"""
    cache = _cache_get(_cache_path(seat, str(date)))
    if cache is not None:
        return cache

    d = _parse_date(date)
    broker = _load_broker()
    rows = broker[(broker["营业部名称"] == seat) & (broker["上榜日"] == d)]
    if rows.empty:
        out = {"seat": seat, "date": str(d.date()), "note": "该日无该席位记录"}
        _cache_set(_cache_path(seat, str(date)), out)
        return out

    r = rows.iloc[0]
    buy_amt = _f(r["买入总金额"]) or 0.0
    sell_amt = _f(r["卖出总金额"]) or 0.0
    net = _f(r["总买卖净额"]) or 0.0

    names = [] if pd.isna(r["买入股票"]) else str(r["买入股票"]).split()
    n2c, di = _name2code(), _detail_index()
    stocks = []
    for n in names[:8]:
        c = n2c.get(n)
        item = {"name": n, "code": c}
        if c:
            hit = di.get((c, d.date()))
            if hit is not None:
                item["net"] = hit[1]
                item["pct"] = hit[0]
        stocks.append(item)

    sd = style_breakdown(seat)
    if sd.get("note"):
        conclusion = f"{seat} 历史样本不足（近{WINDOW_DAYS}日无有效上榜股），今日{'净买' if net >= 0 else '净卖'} {abs(net):,.0f}，参考价值有限"
    else:
        dominant = sd["dominant_style"]
        wr = sd["winrate"].get("winrate_60d")
        fwd5 = sd["holding_period"].get("fwd_5d")
        style_txt = {
            "首板打板": "惯用低位首板打板",
            "接力": "惯用高位接力（2板+）",
            "趋势": "惯用趋势股波段",
            "低吸": "惯用低吸/跌买",
        }.get(dominant, "操作风格混合")

        buy_names = [s["name"] for s in stocks]
        if net < 0:
            match = "今日净卖出，倾向兑现离场"
        elif dominant in ("首板打板", "接力") and buy_names:
            match = f"今日买{'/'.join(buy_names[:3])}=符合惯用打法"
        else:
            match = f"今日买{'/'.join(buy_names[:3])}" if buy_names else "今日无买入上榜"

        wr_txt = f"历史60日胜率{wr * 100:.0f}%" if wr is not None else "历史胜率样本不足"
        fwd_txt = f"上榜后5日均值{fwd5:+.2f}%" if fwd5 is not None else "上榜后走势样本不足"
        conclusion = f"{seat} {style_txt}，{match}；{wr_txt}，{fwd_txt}，预示1-5日兑现节奏"

    out = {
        "seat": seat,
        "date": str(d.date()),
        "net_buy": round(net, 2),
        "buy_amt": round(buy_amt, 2),
        "sell_amt": round(sell_amt, 2),
        "buy_stock_cnt": int(r["买入个股数"]) if not pd.isna(r["买入个股数"]) else len(stocks),
        "sell_stock_cnt": int(r["卖出个股数"]) if not pd.isna(r["卖出个股数"]) else None,
        "stocks": stocks,
        "style_breakdown": sd,
        "conclusion": conclusion,
    }
    _cache_set(_cache_path(seat, str(date)), out)
    return out


# ─────────────────────────────────────────────────────────
# 3. 当日最活跃席位
# ─────────────────────────────────────────────────────────
def hot_seats(date: str | None = None, n: int = 10) -> pd.DataFrame:
    """当日最活跃席位（按成交额），附风格标签; 日期无数据时取最近日期。"""
    broker = _load_broker()
    if date is None:
        d = broker["上榜日"].max()
    else:
        d = _parse_date(date)
        if broker[broker["上榜日"] == d].empty:
            d = broker["上榜日"].max()  # 取 lhb 最近日期

    day = broker[broker["上榜日"] == d].copy()
    day["turnover"] = day["买入总金额"].fillna(0) + day["卖出总金额"].fillna(0)
    day = day.sort_values("turnover", ascending=False).head(n)

    prof = _load_profiles()
    if not prof.empty:
        prof_map = prof.set_index("seat")
        day["style"] = day["营业部名称"].map(lambda s: prof_map.loc[s, "style"] if s in prof_map.index else None)
        day["winrate_60d"] = day["营业部名称"].map(lambda s: prof_map.loc[s, "winrate_60d"] if s in prof_map.index else None)
        day["samples_60d"] = day["营业部名称"].map(lambda s: prof_map.loc[s, "samples_60d"] if s in prof_map.index else None)
        day["seat_type"] = day["营业部名称"].map(lambda s: prof_map.loc[s, "seat_type"] if s in prof_map.index else None)
    else:
        day["style"] = None
        day["winrate_60d"] = None
        day["samples_60d"] = None
        from quant_system.analysis_core.broker_gaming import classify_seat  # noqa: E402
        day["seat_type"] = day["营业部名称"].map(classify_seat)

    out = pd.DataFrame({
        "seat": day["营业部名称"],
        "seat_type": day["seat_type"],
        "buy_amt": day["买入总金额"].round(0),
        "sell_amt": day["卖出总金额"].round(0),
        "net_amt": day["总买卖净额"].round(0),
        "turnover": day["turnover"].round(0),
        "style": day["style"],
        "winrate_60d": day["winrate_60d"],
        "samples_60d": day["samples_60d"],
    }).reset_index(drop=True)
    out.attrs["date"] = str(d.date())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="席位行为画像深化")
    ap.add_argument("--style", type=str)
    ap.add_argument("--interpret", nargs=2, metavar=("SEAT", "DATE"))
    ap.add_argument("--hot", nargs="*", metavar=("DATE", "N"))
    args = ap.parse_args()

    if args.style:
        print(json.dumps(style_breakdown(args.style), ensure_ascii=False, indent=1))
    if args.interpret:
        print(json.dumps(interpret(*args.interpret), ensure_ascii=False, indent=1))
    if args.hot is not None:
        date = args.hot[0] if args.hot else None
        n = int(args.hot[1]) if len(args.hot) > 1 else 10
        h = hot_seats(date, n)
        print(f"[hot] date={h.attrs.get('date')} rows={len(h)}")
        print(h.to_string())
    if not (args.style or args.interpret or args.hot is not None):
        ap.print_help()
