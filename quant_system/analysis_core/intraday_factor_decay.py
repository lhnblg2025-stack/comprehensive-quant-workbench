#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""intraday_factor_decay — 因子实时衰减热力图（V12.1 方案阶段2 · 域M）

A股日内风格漂移严重，等日终 IC 发现失效已晚。本模块消费
data_warehouse/realtime_snapshot/{YYYYMMDD}/ 下每 5 分钟全市场快照
（5203 只，含 price/pre_close/pct_chg/turnover 等列），对核心因子做
滚动 30 分钟截面 IC：

  数据源     : realtime_snapshot 最新快照目录（mock 用 tmp 构造同构 parquet）
  核心因子   : mom_5 动量 / rev_5 反转 / lowvol 低波 / turnover 换手
               （快照已有同名列则直接用；否则从 pct_chg/turnover 派生）
  滚动 IC    : 每 30 分钟窗口（快照 ts 分组），因子值 vs 下一窗口收益的
               截面 Spearman（默认）/ Pearson 相关
  下岗熔断   : 连续 3 个窗口 IC 为负且 |IC| 递增 → 状态 down（今日下岗，
               当日不回魂）
  权重再分配 : 下岗因子权重归 0；剩余因子按最新已结算 |IC| 比例归一
               （w=0 或 |IC| 全 0 时退化为等权）
  输出       : generated/factor_decay/YYYYMMDD/factor_decay_{date}.json|md
               每因子: 各窗口 IC 序列 / 状态 up|down / 当前权重 / 实时排名
  防前视     : IC[t] = corr(factor[t], ret[t→t+1])（t 窗口因子 vs t+1 窗口收益）；
               最后一个窗口视为盘中未收盘（open），不作为 IC 的收益端，
               IC 序列只到最新已收盘窗口（index = n-2）
  降级       : 无快照 → degraded=true 空结果，CLI 仍退出 0

用法:
  python3 -m quant_system.analysis_core.intraday_factor_decay --date 2026-08-11
  python3 -m quant_system.analysis_core.intraday_factor_decay            # 缺省最新
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent          # workspace 根
SNAPSHOT_DIR = ROOT / "data_warehouse" / "realtime_snapshot"  # 每5分钟全市场快照
DEFAULT_OUT_DIR = ROOT / "generated"                          # 输出根目录
CST = timezone(timedelta(hours=8))

FACTORS = ("mom_5", "rev_5", "lowvol", "turnover")  # 核心因子
WINDOW_MIN = 30        # 滚动窗口（分钟）
CONSEC_NEG = 3         # 连续 3 窗口为负 → 下岗
MIN_PAIRS = 3          # 截面相关最少有效对数

logger = logging.getLogger(__name__)


# ── 基础工具 ──────────────────────────────────────────────────────────────
def _now() -> str:
    return datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S")


def _r4(v):
    """None/非有限 → None，否则保留 4 位小数（JSON 友好）。"""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, 4) if np.isfinite(f) else None


def _parse_ts(ts) -> datetime | None:
    """快照 ts（YYYYMMDDHHMMSS，支持 int/datetime）→ datetime，解析失败返回 None。"""
    if isinstance(ts, datetime):
        return ts
    s = str(ts).strip()
    try:
        return datetime.strptime(s, "%Y%m%d%H%M%S")
    except ValueError:
        try:
            return pd.to_datetime(s)
        except Exception:
            return None


def _minutes_of(dt: datetime) -> int:
    return dt.hour * 60 + dt.minute


# ── 因子定义 ──────────────────────────────────────────────────────────────
def derive_factors(df: pd.DataFrame) -> pd.DataFrame:
    """从单份快照计算核心因子值（index=code）。

    优先使用快照已有同名因子列（mock 可直接控制）；否则从基础列派生:
      mom_5    = pct_chg             （5分钟涨跌幅作为动量代理）
      rev_5    = -pct_chg            （反转 = 动量反向）
      lowvol   = -abs(pct_chg)       （|涨跌幅| 越小 → 低波因子越高）
      turnover = turnover            （换手率列）
    """
    if "code" not in df.columns:
        return pd.DataFrame(index=df.index)
    codes = df["code"].astype(str)
    out = pd.DataFrame({"code": codes})
    out = out.set_index("code")

    def col_or(name, fallback):
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").to_numpy()
        return fallback

    pct = pd.to_numeric(df["pct_chg"], errors="coerce").to_numpy() if "pct_chg" in df.columns else np.full(len(df), np.nan)
    turn = pd.to_numeric(df["turnover"], errors="coerce").to_numpy() if "turnover" in df.columns else np.full(len(df), np.nan)

    out["mom_5"] = col_or("mom_5", pct)
    out["rev_5"] = col_or("rev_5", -pct)
    out["lowvol"] = col_or("lowvol", -np.abs(pct))
    out["turnover"] = col_or("turnover", turn)
    for name in FACTORS:
        out[name] = pd.to_numeric(out[name], errors="coerce")
    return out


def _window_mean_price(w: "Window") -> pd.Series:
    """窗口内个股均价（多快照取均值），index=code。"""
    s = w.df.groupby(w.df["code"].astype(str))["price"].mean()
    return s.astype(float)


# ── 窗口构建 ──────────────────────────────────────────────────────────────
@dataclass
class Window:
    index: int
    start: datetime
    end: datetime
    df: pd.DataFrame
    is_open: bool = False  # 盘中未收盘（最后一个窗口）


def build_windows(snapshots: list[pd.DataFrame]) -> list[Window]:
    """按快照 ts 每 30 分钟分组 → 排序窗口列表；最后一个窗口标记 is_open=True。

    同一 30 分钟桶内多份 5 分钟快照合并为一个窗口（因子取均值）。
    解析失败的 ts 行丢弃。
    """
    buckets: dict[tuple, list[pd.DataFrame]] = {}
    for df in snapshots:
        if df is None or "ts" not in df.columns:
            continue
        tmp = df.copy()
        dt = tmp["ts"].map(_parse_ts)
        tmp = tmp[dt.notna()].copy()
        if tmp.empty:
            continue
        tmp["_dt"] = tmp["ts"].map(_parse_ts)
        tmp["_bkt"] = tmp["_dt"].map(lambda d: (d.date(), _minutes_of(d) // WINDOW_MIN))
        for key, g in tmp.groupby("_bkt"):
            buckets.setdefault(key, []).append(g.drop(columns=["_bkt"]))
    keys = sorted(buckets, key=lambda k: (k[0], k[1]))
    windows = []
    for i, key in enumerate(keys):
        start = datetime.combine(key[0], datetime.min.time()) + timedelta(minutes=key[1] * WINDOW_MIN)
        end = start + timedelta(minutes=WINDOW_MIN)
        df = pd.concat(buckets[key], ignore_index=True).drop(columns=["_dt"])
        windows.append(Window(index=i, start=start, end=end, df=df))
    if windows:
        windows[-1].is_open = True
    return windows


def _spearman(a, b) -> float | None:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < MIN_PAIRS:
        return None
    ar = pd.Series(a[mask]).rank().to_numpy(dtype=float)
    br = pd.Series(b[mask]).rank().to_numpy(dtype=float)
    if float(np.std(ar)) == 0.0 or float(np.std(br)) == 0.0:
        return None
    return float(np.corrcoef(ar, br)[0, 1])


def _pearson(a, b) -> float | None:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < MIN_PAIRS:
        return None
    if float(np.std(a[mask])) == 0.0 or float(np.std(b[mask])) == 0.0:
        return None
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def rolling_ic(panel: list[pd.DataFrame], rets: dict[int, pd.Series], method: str = "spearman") -> pd.DataFrame:
    """滚动截面 IC：IC[t] = corr(factor[t], ret[t→t+1])。

    防前视：因子取 t 窗口，收益取 t→t+1 窗口；只算到最新已收盘窗口
    （t+1 ≤ n-2，最后一个窗口 open 不参与收益端）。
    """
    corr_fn = _spearman if method == "spearman" else _pearson
    n = len(panel)
    rows = []
    for t in range(max(0, n - 2)):  # t = 0..n-3
        f = panel[t]
        r = rets.get(t)
        if r is None:
            rows.append({name: None for name in FACTORS})
            continue
        common = f.index.intersection(r.index)
        if len(common) < MIN_PAIRS:
            rows.append({name: None for name in FACTORS})
            continue
        vals = {
            name: corr_fn(f.loc[common, name].to_numpy(dtype=float), r.loc[common].to_numpy(dtype=float))
            for name in FACTORS
        }
        rows.append(vals)
    return pd.DataFrame(rows, index=range(len(rows)))


def pad_ic(ic: pd.DataFrame, n_windows: int) -> dict[str, list[float | None]]:
    """IC 补齐到与窗口对齐的定长序列（未结算窗口为 None）。"""
    out: dict[str, list[float | None]] = {}
    for name in FACTORS:
        out[name] = [None] * n_windows
        for t, row in ic.iterrows():
            if t < n_windows:
                out[name][t] = None if row[name] is None else float(row[name])
    return out


# ── 下岗熔断 ──────────────────────────────────────────────────────────────
def decay_state(ic_full: dict[str, list], n_windows: int) -> dict[str, list[str]]:
    """每因子每窗口状态：决策只用到截至该窗口前已结算的 IC（ic[0..w-1]）。

    触发条件：连续 CONSEC_NEG 个有效 IC 均为负且 |IC| 严格递增 → 自窗口 w 起
    down（今日下岗，不回魂）。
    """
    out: dict[str, list[str]] = {}
    for name, ics in ic_full.items():
        status = ["up"] * n_windows
        triggered: int | None = None
        for w in range(1, n_windows):
            if triggered is not None:
                status[w] = "down"
                continue
            seq = [ics[k] for k in range(w) if ics[k] is not None]
            if len(seq) >= CONSEC_NEG:
                tail = seq[-CONSEC_NEG:]
                neg = all(v < 0.0 for v in tail)
                inc = all(abs(tail[i]) < abs(tail[i + 1]) for i in range(CONSEC_NEG - 1))
                if neg and inc:
                    status[w] = "down"
                    triggered = w
        out[name] = status
    return out


# ── 权重再分配 ────────────────────────────────────────────────────────────
def reallocate_weights(ic_full: dict[str, list], status: dict[str, list], n_windows: int) -> tuple[dict[str, list[float]], dict[str, list[int]]]:
    """每窗口权重/排名：下岗因子权重归 0，剩余按最新已结算 |IC| 比例归一。

    w=0 或 |IC| 全 0 → 剩余 up 因子等权；全下岗 → 全 0。
    排名 = 权重降序（并列按因子顺序），1 为最强。
    """
    names = list(ic_full)
    weights: dict[str, list[float]] = {name: [] for name in names}
    ranks: dict[str, list[int]] = {name: [] for name in names}
    for w in range(n_windows):
        if w == 0:
            base = {name: 0.0 for name in names}
        else:
            base = {}
            for name in names:
                icv = ic_full[name][w - 1]
                if status[name][w] == "down" or icv is None:
                    base[name] = 0.0
                else:
                    base[name] = abs(float(icv))
        total = sum(base.values())
        if total > 0:
            wt = {name: base[name] / total for name in names}
        else:
            up_names = [name for name in names if status[name][w] == "up"]
            if up_names:
                wt = {name: (1.0 / len(up_names) if name in up_names else 0.0) for name in names}
            else:
                wt = {name: 0.0 for name in names}
        order = sorted(names, key=lambda nm: (-wt[nm], names.index(nm)))
        rank_of = {nm: order.index(nm) + 1 for nm in names}
        for name in names:
            weights[name].append(wt[name])
            ranks[name].append(rank_of[name])
    return weights, ranks


# ── 数据加载 ──────────────────────────────────────────────────────────────
def _locate_snapshots(date: str | None, snapshot_dir: Path) -> tuple[str | None, list[Path]]:
    """定位 {snapshot_dir}/{YYYYMMDD}/*.parquet；date 缺省取最新日期目录。"""
    snap_dir = Path(snapshot_dir)
    if not snap_dir.is_dir():
        return date, []
    if date:
        d = date.replace("-", "")
        sub = snap_dir / d
        files = sorted(sub.glob("*.parquet")) if sub.is_dir() else []
        return date, files
    dirs = sorted([p for p in snap_dir.iterdir() if p.is_dir()], reverse=True)
    for d in dirs:
        files = sorted(d.glob("*.parquet"))
        if files:
            name = d.name
            if len(name) == 8 and name.isdigit():
                name = f"{name[:4]}-{name[4:6]}-{name[6:]}"
            return name, files
    return None, []


def _read_snapshot(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 —— 单文件损坏不阻断整体
        logger.warning("快照读取失败 %s: %s", path, exc)
        return None
    if "code" not in df.columns or "ts" not in df.columns:
        return None
    return df


# ── 组装与输出 ────────────────────────────────────────────────────────────
def _assemble_result(date_str: str, files: list[Path], windows: list[Window],
                     ic_full: dict[str, list], status: dict[str, list],
                     weights: dict[str, list], ranks: dict[str, list], method: str) -> dict:
    n = len(windows)
    factors = []
    for name in FACTORS:
        trig = next((w for w, s in enumerate(status[name]) if s == "down"), None)
        factors.append({
            "name": name,
            "ic": [_r4(v) for v in ic_full[name]],
            "status": status[name],
            "weight": [round(v, 6) for v in weights[name]],
            "rank": ranks[name],
            "trigger_window": trig,
            "current_weight": round(weights[name][-1], 6),
            "current_rank": ranks[name][-1],
        })
    return {
        "schema": "intraday_factor_decay/v1",
        "date": date_str,
        "generated_at": _now(),
        "degraded": False,
        "method": method,
        "window_min": WINDOW_MIN,
        "consec_neg": CONSEC_NEG,
        "n_windows": n,
        "latest_closed_window": n - 2,
        "snapshot_files": len(files),
        "windows": [
            {"index": w.index, "start": w.start.strftime("%H:%M"),
             "end": w.end.strftime("%H:%M"), "is_open": w.is_open}
            for w in windows
        ],
        "factors": factors,
    }


def _degraded_result(date_str: str | None, out_dir: Path | None, reason: str) -> dict:
    result = {
        "schema": "intraday_factor_decay/v1",
        "date": date_str,
        "generated_at": _now(),
        "degraded": True,
        "reason": reason,
        "method": "spearman",
        "window_min": WINDOW_MIN,
        "consec_neg": CONSEC_NEG,
        "n_windows": 0,
        "latest_closed_window": None,
        "snapshot_files": 0,
        "windows": [],
        "factors": [],
    }
    if out_dir is not None:
        write_outputs(result, out_dir)
    return result


def analyze(date: str | None = None, snapshot_dir: str | Path | None = None,
            out_dir: str | Path | None = None, method: str = "spearman") -> dict:
    """主流程：定位快照 → 分窗口 → 滚动 IC → 下岗熔断 → 权重再分配 → 输出。"""
    snap = Path(snapshot_dir) if snapshot_dir else SNAPSHOT_DIR
    date_str, files = _locate_snapshots(date, snap)
    if not files:
        return _degraded_result(date_str, Path(out_dir) if out_dir else None, "no_snapshot")

    snapshots = [_read_snapshot(f) for f in files]
    snapshots = [df for df in snapshots if df is not None]
    if len(snapshots) < 2:
        return _degraded_result(date_str, Path(out_dir) if out_dir else None, "insufficient_snapshot")

    windows = build_windows(snapshots)
    if len(windows) < 2:
        return _degraded_result(date_str, Path(out_dir) if out_dir else None, "insufficient_windows")

    panel = [derive_factors(w.df).groupby(level=0).mean() for w in windows]  # 桶内多快照按 code 取均值
    prices = [_window_mean_price(w) for w in windows]
    rets = {t: _next_window_return(prices, t) for t in range(len(windows) - 1)}
    ic = rolling_ic(panel, rets, method=method)
    ic_full = pad_ic(ic, len(windows))
    status = decay_state(ic_full, len(windows))
    weights, ranks = reallocate_weights(ic_full, status, len(windows))

    result = _assemble_result(date_str, files, windows, ic_full, status, weights, ranks, method)
    if out_dir is not None:
        write_outputs(result, Path(out_dir))
    return result


def _next_window_return(prices: list[pd.Series], t: int) -> pd.Series:
    """t→t+1 窗口收益（个股均价），index=code，仅两窗口共同代码。"""
    p_t, p_t1 = prices[t], prices[t + 1]
    common = p_t.index.intersection(p_t1.index)
    return p_t1.loc[common] / p_t.loc[common] - 1.0


def write_outputs(result: dict, out_dir: Path) -> tuple[Path, Path]:
    """写 generated/factor_decay/YYYYMMDD/factor_decay_{date}.json|md。"""
    out = Path(out_dir)
    day = result["date"].replace("-", "") if result["date"] else "unknown"
    sub = out / "factor_decay" / day
    sub.mkdir(parents=True, exist_ok=True)
    json_path = sub / f"factor_decay_{result['date']}.json"
    md_path = sub / f"factor_decay_{result['date']}.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_to_md(result), encoding="utf-8")
    return json_path, md_path


def _to_md(result: dict) -> str:
    lines = [f"# 因子实时衰减热力图 {result['date']}（V12.1 域M）", ""]
    if result.get("degraded"):
        lines.append(f"> 降级：{result.get('reason', '')}，无快照数据，结果为空。")
        return "\n".join(lines)
    lines += [
        f"- 方法：{result['method']} / 窗口 {result['window_min']} 分钟 / 连续 {result['consec_neg']} 负触发下岗",
        f"- 窗口数：{result['n_windows']}（最新窗口 open，不参与 IC 收益端，防前视）",
        f"- 生成：{result['generated_at']}",
        "",
        "## 滚动 IC 与状态（每窗口）",
        "",
    ]
    heads = ["因子"] + [f"{w['start']}-{w['end']}" for w in result["windows"]] + ["状态", "当前权重", "实时排名"]
    lines.append("| " + " | ".join(heads) + " |")
    lines.append("|" + "---|" * len(heads))
    for f in result["factors"]:
        cells = [f["name"]]
        for i in range(result["n_windows"]):
            ic = f["ic"][i]
            st = f["status"][i]
            cell = "—" if ic is None else f"{ic:+.2f}"
            cells.append(f"{cell}{'↓' if st == 'down' else ''}")
        cells += [f["status"][-1], f"{f['current_weight']:.2f}", str(f["current_rank"])]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "说明：↓=今日下岗（权重归0）；排名按当前权重降序。"]
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="因子实时衰减热力图（V12.1 域M）")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD，缺省取最新快照日期")
    ap.add_argument("--method", choices=["spearman", "pearson"], default="spearman")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="输出根目录（默认 workspace/generated）")
    ap.add_argument("--snapshot-dir", default=None, help="快照根目录（默认 data_warehouse/realtime_snapshot，测试用）")
    args = ap.parse_args(argv)
    try:
        result = analyze(date=args.date, snapshot_dir=args.snapshot_dir,
                         out_dir=args.out_dir, method=args.method)
    except Exception as exc:  # noqa: BLE001
        logger.exception("intraday_factor_decay 执行失败: %s", exc)
        return 1

    if result.get("degraded"):
        print(f"[factor_decay] {result.get('date')} 降级（{result.get('reason')}）：空结果")
        if args.out_dir:
            out = Path(args.out_dir) / "factor_decay" / (result["date"].replace("-", "") if result["date"] else "unknown")
            print(f"[factor_decay] 输出: {out}")
        return 0

    print(f"[factor_decay] {result['date']} 窗口数={result['n_windows']} "
          f"方法={result['method']} 快照文件={result['snapshot_files']}")
    for f in result["factors"]:
        print(f"  {f['name']:<9} 状态={f['status'][-1]:<4} 权重={f['current_weight']:.3f} "
              f"排名={f['current_rank']} 下岗窗口={f['trigger_window']}")
    out = Path(args.out_dir) / "factor_decay" / result["date"].replace("-", "")
    print(f"[factor_decay] 输出: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
