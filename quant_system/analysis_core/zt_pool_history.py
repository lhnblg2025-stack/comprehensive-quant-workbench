"""
zt_pool_history — 涨停池历史地基（V11 W1-1）

两条数据通路:
  1. K线重建（历史）: 遍历 data_warehouse/kline/*.parquet（5206 只，2020→今），
     按涨停规则（主板10%/创业板科创板20%/北交所30%）重建每日
     涨停/炸板(触板未封)/跌停 + 连板数 → zt_pool_history.parquet（长表）
  2. 东财三池（日更精确）: 涨停池/炸板池/跌停池，含封板资金/炸板次数/所属行业
     → zt_pool_em_daily.parquet

用法:
  python3 -m quant_system.analysis_core.zt_pool_history --reconstruct          # 全量K线重建
  python3 -m quant_system.analysis_core.zt_pool_history --fetch-em-window 30   # 回补近30交易日EM池
  python3 -m quant_system.analysis_core.zt_pool_history --update-today         # 每日盘后增量（cron）
  python3 -m quant_system.analysis_core.zt_pool_history --status               # 数据现状
"""

from __future__ import annotations
import logging

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quant_system"))

from quant_system.analysis_core.config import (  # noqa: E402
    
    NAME_MAP,
    ZT_EM_DAILY,
    ZT_HISTORY,
    
    LIMIT_RULES,
    DEFAULT_LIMIT,
    ZT_EPS,
    PCT_TOL,
)
from quant_system.cpu_throttle import CpuThrottle  # noqa: E402

CST = timezone(timedelta(hours=8))
# Canonical long table remains config.ZT_HISTORY; health/report consumers use
# this separate date-partition directory for exact daily freshness evidence.
ZT_HISTORY_DIR = ROOT / "data_warehouse" / "zt_history"
throttle = CpuThrottle()

# 股票代码白名单前缀（排除ETF/基金/指数）
STOCK_PREFIX = ("00", "30", "60", "68", "8", "4", "92")


def limit_ratio(code: str) -> float:
    """按代码前缀返回涨停幅度。"""
    for prefixes, ratio in LIMIT_RULES:
        if code.startswith(prefixes):
            return ratio
    return DEFAULT_LIMIT


def is_stock_code(code: str) -> bool:
    return code.startswith(STOCK_PREFIX)


# ────────────────────────────────────────────────────────────
# 1. 股票名称映射（含ST标记）
# ────────────────────────────────────────────────────────────
def ensure_name_map(force: bool = False) -> pd.DataFrame:
    """A股代码-名称-ST标记映射，缓存 parquet，月度刷新。

    优先网络刷新（东财，带重试）；失败时回退本地 quant_system/stock_name_map.json，
    保证重建管线不被网络故障阻塞。
    """
    if NAME_MAP.exists() and not force:
        age_days = (datetime.now(CST) - datetime.fromtimestamp(NAME_MAP.stat().st_mtime, CST)).days
        if age_days < 30:
            return pd.read_parquet(NAME_MAP)

    df = _fetch_name_map_remote()
    if df is None:
        df = _fetch_name_map_local()
    df.to_parquet(NAME_MAP, index=False)
    print(f"[namemap] 已缓存: {len(df)} 只", flush=True)
    return df


def _fetch_name_map_remote() -> pd.DataFrame | None:
    import akshare as ak
    last_err = None
    for attempt in range(1, 4):
        try:
            df = ak.stock_info_a_code_name()
            df = df.rename(columns={"code": "code", "name": "name"})
            df["code"] = df["code"].astype(str).str.zfill(6)
            df["is_st"] = df["name"].str.contains("ST", case=False, na=False)
            return df
        except Exception as e:
            last_err = e
            throttle.sleep(default_ms=3000 * attempt)
    print(f"[namemap] 远程失败，回退本地: {str(last_err)[:100]}", flush=True)
    return None


def _fetch_name_map_local() -> pd.DataFrame:
    """本地兜底: quant_system/stock_name_map.json（name→code，反转）。"""
    import json
    p = ROOT / "quant_system" / "stock_name_map.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    rows = [{"code": c, "name": n} for n, c in d.items()]
    df = pd.DataFrame(rows)
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["is_st"] = df["name"].str.contains("ST", case=False, na=False)
    print(f"[namemap] 本地兜底: {len(df)} 只", flush=True)
    return df


def get_st_set() -> set[str]:
    df = ensure_name_map()
    return set(df.loc[df["is_st"], "code"])


# ────────────────────────────────────────────────────────────
# 2. K线重建（历史地基）
# ────────────────────────────────────────────────────────────
def reconstruct_from_kline(start: str = "2020-01-01", progress_every: int = 500,
                           write: bool = True) -> pd.DataFrame:
    """遍历全部K线重建涨停/炸板/跌停长表。

    write=False 时只返回不落盘（供 incremental_kline 合并后统一写，
    避免覆盖清空历史——2026-08-10 审计发现的历史丢失 Critical）。
    """
    kline_dir = ROOT / "data_warehouse" / "kline"
    files = sorted(kline_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"kline 目录为空: {kline_dir}")

    st_set = get_st_set()
    start_ts = pd.Timestamp(start)
    # 连板计数缓冲: 增量重建时向前多取 8 天，保证窗口首日的连板数继承此前累计
    buffer_start = (start_ts - pd.Timedelta(days=8)).strftime("%Y-%m-%d")
    records: list[pd.DataFrame] = []
    total = len(files)

    for i, f in enumerate(files, 1):
        code = f.stem
        if not is_stock_code(code):
            continue
        try:
            try:
                df = pd.read_parquet(f, filters=[("date", ">=", pd.Timestamp(buffer_start))])
            except Exception:
                df = pd.read_parquet(f)
        except Exception as e:
            logging.getLogger(__name__).error(f"[zt_pool_history] 操作失败: {e}", exc_info=True)
            continue
        if df.empty:
            continue
        # 防御: 部分K线文件缺列/缺数据（不中断全量重建）
        for c in ["pct_chg", "turnover", "amount", "outstanding_share"]:
            if c not in df.columns:
                df[c] = np.nan
        # pct_chg 逐日补全（近期抓取器常缺此列，必须用 close 重算，否则涨停漏判）
        # 2026-08-21 审计: 必须先按 date 排序再 pct_change——原实现在排序前补全，
        # 增量抓取追加乱序时 pct_change 用错误相邻行，污染 is_zt 判定。
        df = df.sort_values("date").reset_index(drop=True)
        if df["pct_chg"].isna().any():
            df["pct_chg"] = df["pct_chg"].fillna(df["close"].pct_change() * 100)
        df["pct_chg"] = df["pct_chg"].fillna(0.0)
        if df.empty:
            continue

        is_st_code = code in st_set
        # ST 涨跌幅: 主板 ST 5%（创/科/北 不变）
        ratio = 0.05 if (is_st_code and not code.startswith(("30", "68", "8", "4", "92"))) else limit_ratio(code)
        prev_close = df["close"].shift(1)
        limit_price = (prev_close * (1 + ratio)).round(2)
        down_price = (prev_close * (1 - ratio)).round(2)

        pct = df["pct_chg"]
        is_zt = (df["close"] >= limit_price - ZT_EPS) & (pct >= ratio * 100 - PCT_TOL)
        is_dt = (df["close"] <= down_price + ZT_EPS) & (pct <= -(ratio * 100 - PCT_TOL))
        # 炸板: 摸板但未封 + 涨幅接近涨停（≥8%，防止上影毛刺误判）
        is_zb = (~is_zt) & (df["high"] >= limit_price - ZT_EPS) & (pct >= ratio * 100 - 2.0)

        # 连板数：今日涨停 → 连续涨停天数(含今日)；今日未涨停 → 截至昨日的连板数
        board = np.zeros(len(df), dtype=int)
        for j in range(len(df)):
            if is_zt.iloc[j]:
                board[j] = board[j - 1] + 1 if j > 0 and is_zt.iloc[j - 1] else 1
            else:
                board[j] = board[j - 1] if j > 0 else 0

        # 裁剪缓冲行（缓冲仅用于连板初始化）—— 全部转 numpy 掩码避免 pandas 索引错位
        keep = (df["date"] >= start_ts).values
        df = df[keep].reset_index(drop=True)
        board = board[keep]
        is_zt_a = is_zt.values[keep]
        is_zb_a = is_zb.values[keep]
        is_dt_a = is_dt.values[keep]
        if df.empty:
            continue

        sel = is_zt_a | is_zb_a | is_dt_a
        if not sel.any():
            continue

        sub = df.loc[sel].copy()
        sub["code"] = code
        sub["name"] = ""
        sub["is_zt"] = is_zt_a[sel]
        sub["is_zb"] = is_zb_a[sel]
        sub["is_dt"] = is_dt_a[sel]
        sub["is_st"] = is_st_code
        sub["board_count"] = board[sel]
        sub["float_mv"] = (sub["outstanding_share"] * sub["close"]) / 1e8  # 亿（近似）
        sub["next_pct"] = df["pct_chg"].shift(-1)[sel].values  # 下一交易日涨幅（算溢价/亏钱效应）
        sub = sub[["date", "code", "name", "close", "pct_chg", "turnover", "amount",
                   "float_mv", "board_count", "is_zt", "is_zb", "is_dt", "is_st", "next_pct"]]
        records.append(sub)

        if i % progress_every == 0:
            print(f"[reconstruct] {i}/{total} 代码处理中…", flush=True)

    out = pd.concat(records, ignore_index=True) if records else pd.DataFrame(
        columns=["date", "code", "name", "close", "pct_chg", "turnover", "amount",
                 "float_mv", "board_count", "is_zt", "is_zb", "is_dt", "is_st", "next_pct"])
    out = out.sort_values(["date", "code"]).reset_index(drop=True)
    out["source"] = "kl"
    if write:
        out.to_parquet(ZT_HISTORY, index=False)
        print(f"[reconstruct] 完成: {len(out)} 行 → {ZT_HISTORY}")
    return out


def incremental_kline(days: int = 10) -> pd.DataFrame:
    """滚动增量：只重算最近 days 个交易日（含缓冲），替换旧行后合并。

    每日 cron 调用，避免全量重建 5206 只K线。
    2026-08-10 审计修复: reconstruct(write=False) 不再覆盖文件，
    先读旧文件合并后统一原子写，杜绝历史被清空。
    """
    cut = (pd.Timestamp(datetime.now(CST).date()) - pd.Timedelta(days=days * 2 + 5)).strftime("%Y-%m-%d")
    tail = reconstruct_from_kline(start=cut, write=False)
    if ZT_HISTORY.exists():
        old = pd.read_parquet(ZT_HISTORY)
        old = old[old["date"] < pd.Timestamp(cut)]
        merged = pd.concat([old, tail], ignore_index=True)
        merged = merged.drop_duplicates(subset=["date", "code", "is_zt", "is_zb", "is_dt"])
        merged = merged.sort_values(["date", "code"]).reset_index(drop=True)
        # 原子写: 临时文件 + os.replace
        tmp = ZT_HISTORY.with_suffix(".parquet.tmp")
        merged.to_parquet(tmp, index=False)
        tmp.replace(ZT_HISTORY)
        print(f"[kline] 滚动增量完成: 历史 {len(old)} + 新增 {len(tail)} = {len(merged)} 行")
    return tail


# ────────────────────────────────────────────────────────────
# 3. 东财三池（精确日更）
# ────────────────────────────────────────────────────────────
_EM_POOLS = [
    ("stock_zt_pool_em", "zt"),
    ("stock_zt_pool_zbgc_em", "zb"),
    ("stock_zt_pool_dtgc_em", "dt"),
]


def fetch_em_pool(date_str: str, with_sleep: bool = True) -> pd.DataFrame | None:
    """抓取某交易日东财三池，归一化为长表。返回 None 表示该日无数据。"""
    import akshare as ak
    frames = []
    for fn_name, kind in _EM_POOLS:
        try:
            fn = getattr(ak, fn_name)
            df = fn(date=date_str)
        except Exception:
            df = None
        if df is None or df.empty:
            continue
        df = df.copy()
        df = df.rename(columns={
            "代码": "code", "名称": "name", "涨跌幅": "pct_chg", "最新价": "close",
            "成交额": "amount", "流通市值": "float_mv", "总市值": "total_mv",
            "换手率": "turnover", "封板资金": "seal_fund", "封单资金": "seal_fund",
            "首次封板时间": "first_seal_time", "最后封板时间": "last_seal_time",
            "炸板次数": "zb_times", "连板数": "board_count", "所属行业": "industry",
        })
        df["code"] = df["code"].astype(str).str.zfill(6)
        df["date"] = pd.Timestamp(date_str)
        df["is_zt"] = kind == "zt"
        df["is_zb"] = kind == "zb"
        df["is_dt"] = kind == "dt"
        df["is_st"] = df["name"].astype(str).str.contains("ST", case=False, na=False)
        keep = [c for c in ["date", "code", "name", "close", "pct_chg", "turnover",
                            "amount", "float_mv", "total_mv", "board_count", "seal_fund",
                            "first_seal_time", "last_seal_time", "zb_times", "industry",
                            "is_zt", "is_zb", "is_dt", "is_st"] if c in df.columns]
        frames.append(df[keep])
        if with_sleep:
            throttle.sleep(default_ms=800)

    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    out["board_count"] = out["board_count"].fillna(0).astype(int)
    out["source"] = "em"
    return out


def append_em_daily(df: pd.DataFrame) -> None:
    if ZT_EM_DAILY.exists():
        old = pd.read_parquet(ZT_EM_DAILY)
        df = pd.concat([old, df], ignore_index=True)
    df = df.drop_duplicates(subset=["date", "code", "is_zt", "is_zb", "is_dt"]).reset_index(drop=True)
    df.to_parquet(ZT_EM_DAILY, index=False)


def _trade_dates_upto_today(days: int) -> list[str]:
    """最近 days 个交易日（仅 ≤ 今天；Sina 日历含未来日期，必须过滤）。
    2026-08-10 审计: days 参数此前未使用导致窗口失效，现在真正截取最近 days 个。
    """
    import akshare as ak
    cal = ak.tool_trade_date_hist_sina()
    cal = pd.to_datetime(cal["trade_date"])
    today = pd.Timestamp(datetime.now(CST).date())
    cal = cal[cal <= today]
    return cal.dt.strftime("%Y%m%d").tolist()[-days:]


def fetch_em_window(days: int = 30) -> int:
    """从最新交易日向前回补 EM 池（连续 2 日无数据即停止）。滚动式：重复日期自动去重。"""
    recent = _trade_dates_upto_today(days)
    done = 0
    empty_run = 0
    for d in reversed(recent):
        if empty_run >= 2:
            break
        df = fetch_em_pool(d)
        if df is None or df.empty:
            empty_run += 1
            print(f"[em] {d} 无数据（连续空 {empty_run}）", flush=True)
            continue
        empty_run = 0
        _store_em_day(df, d)
        done += 1
        print(f"[em] {d} 已入库: {len(df)} 行", flush=True)
    print(f"[em] 回补完成: {done} 个交易日")
    return done


def _store_em_day(df: pd.DataFrame, day: str) -> None:
    """Atomically persist one EM pool day to canonical and history stores."""
    append_em_daily(df)
    ZT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    history = ZT_HISTORY_DIR / f"ztpool_{day}.parquet"
    temp = history.with_suffix(".tmp")
    df.to_parquet(temp, index=False)
    temp.replace(history)


def update_today() -> int:
    """每日盘后增量（cron）：抓当日 EM 三池（先校验是否交易日）。"""
    today = datetime.now(CST).strftime("%Y%m%d")
    if today not in _trade_dates_upto_today(5):
        print(f"[em] {today} 非交易日，跳过")
        return 0
    df = fetch_em_pool(today)
    if df is None or df.empty:
        print(f"[em] {today} 无数据（可能数据未更新）")
        return 0
    _store_em_day(df, today)
    print(f"[em] {today} 入库 {len(df)} 行 → ztpool_{today}.parquet")
    return len(df)


# ────────────────────────────────────────────────────────────
# 4. 状态
# ────────────────────────────────────────────────────────────
def status() -> None:
    def fmt(p: Path) -> str:
        if not p.exists():
            return "❌ 不存在"
        df = pd.read_parquet(p)
        return f"✅ {len(df):,} 行 / {df['date'].min().date()} ~ {df['date'].max().date()}"
    print("zt_pool_history.parquet :", fmt(ZT_HISTORY))
    print("zt_pool_em_daily.parquet:", fmt(ZT_EM_DAILY))
    print("stock_names.parquet     :", fmt(NAME_MAP))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="涨停池历史地基")
    ap.add_argument("--reconstruct", action="store_true", help="全量K线重建")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--update-kline", type=int, default=0, help="滚动增量：只重算最近 N 交易日")
    ap.add_argument("--fetch-em-window", type=int, default=0, help="回补 EM 池 N 交易日")
    ap.add_argument("--update-today", action="store_true", help="当日 EM 增量")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.reconstruct:
        reconstruct_from_kline(start=args.start)
    if args.update_kline:
        incremental_kline(days=args.update_kline)
    if args.fetch_em_window:
        fetch_em_window(days=args.fetch_em_window)
    if args.update_today:
        update_today()
    if args.status or not (args.reconstruct or args.update_kline or args.fetch_em_window or args.update_today):
        status()
