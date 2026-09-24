#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""watch_card — 自选股复盘卡片 + 指数观察（市场环境先行）

设计依据（skills）:
  - elder-technical-analysis: ATR(14) 波动通道（支撑=close-1.5×ATR、
    阻力=close+1.5×ATR，破位减仓/加仓提示）、新高/新低宽度思想。
  - intraday-trading-strategies: 影线/极值形态标记
    （长上影=(high-close)/close>3%；长下影=(close-low)/low>3%；
    十字星=(high-low)/close<1.5% 且 实体<0.5%）。
  - a-share-market-state-monitor: 复盘卡片提取五层看盘要素中的趋势/资金/风险层。
  - volume_profile（交易密集区）: 60日价格-量分布简化口径，POC / 密集区上沿/下沿 /
    当前价相对位置（样本 <20 日 → '样本不足'）。
  - fund_flow_divergence（资金流背离）: 东财 push2delay 个股资金流日线末行
    （f52 主力净额/元），与当日涨跌幅对比 → 量价背离/主力承接/正常；
    接口失败标注 unavailable（卡片显示 n/a）。口径：东财按单笔金额分桶估算，仅供参考。
  - RS 评级复用 rs_strength 的数据读取与基准（RS=个股收盘/沪深300收盘 同日期对齐，
    RS_MA20=RS 的 20 日均线，RS_slope=RS_MA20 最近5日线性回归斜率归一化 %/日，
    评级: 强=slope>0 且 RS>RS_MA20，中=满足其一，弱=均不满足）。
  - 新高判定沿用 breakout_watch 思想（当日收盘 > 此前 N 个交易日收盘最大值，
    需 ≥ N+1 个样本），独立实现不 import。
  - 口径说明: 影线/量比均为“收盘偏离近似”口径——长上影按收盘价偏离幅度
    (high-close)/close、长下影按 (close-low)/low、量比=当日量/前20日均量，
    以收盘价为基准的近似衡量，非盘中逐笔精确口径。

数据契约:
  data_warehouse/kline/{6位代码}.parquet  日K（列裁剪 date/open/high/low/close/volume）
  data_warehouse/market/index_daily.parquet  沪深300 日线（RS 基准）
  data_warehouse/market/index_daily_{指数名}.parquet  指数日线（红利指数/科创50/创业板指）
  data_warehouse/kline/513050.parquet  中概互联 ETF 日线（列裁剪 date/open/high/low/close/volume）

指数观察（先输出，市场环境）: 当日涨跌幅 / 5日涨跌幅 / MA20位置(上/下) /
  距年线MA250(%) / 60日新高；指数不输出 RS/密集区/资金流。

用法:
  python3 -m quant_system.analysis_core.watch_card --watch 600519,000001
  python3 -m quant_system.analysis_core.watch_card                 # 缺省读 config/watchlist.json
  python3 -m quant_system.analysis_core.watch_card --index 红利指数,科创50
  python3 -m quant_system.analysis_core.watch_card --watch 600519,000001 --date 2026-08-07
"""

from __future__ import annotations
import logging

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quant_system.analysis_core.common import load_names, read_kline_window  # noqa: E402
from quant_system.analysis_core.data_contract import format_pct_value  # noqa: E402
from quant_system.analysis_core.rs_strength import (
    compute_rs_metrics,
    load_benchmark,
)
from quant_system.analysis_core.volume_profile import calc_volume_profile, fmt_pos
from quant_system.analysis_core.fund_flow_divergence import check_divergence

ROOT = Path(__file__).resolve().parent.parent.parent  # workspace（与 analysis_core.config 同约定）
KLINE_DIR = ROOT / "data_warehouse" / "kline"
MARKET_DIR = ROOT / "data_warehouse" / "market"
# 报告输出目录：仓库根 generated/（本环境 workspace/generated 为只读）
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "generated"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "watchlist.json"
WATCHLIST_TEMPLATE = {"watch": ["600519", "000001"], "index": ["科创50", "创业板指"]}

# 指数观察: 名称 -> 文件（指数日线在 market/，中概互联走 kline/513050.parquet）
INDEX_MARKET_NAMES = {"红利指数", "科创50", "创业板指"}
INDEX_KLINE_ALIAS = {"中概互联": "513050"}

CODE_RE = re.compile(r"^\d{6}$")
LOOKBACK_DAYS = 400          # 日历回看窗口（≈275 交易日，覆盖 250日新高 + RS 60日 + ATR14/20日均量）
NEW_HIGH_DAYS = (60, 120, 250)
ATR_N = 14
VOL_N = 20                   # 量比分母：此前 20 日均量

WICK_UPPER = 0.03            # 长上影: (high-close)/close > 3%
WICK_LOWER = 0.03            # 长下影: (close-low)/low > 3%
DOJI_RANGE = 0.015           # 十字星: (high-low)/close < 1.5%
DOJI_BODY = 0.005            # 十字星: 实体 |close-open|/close < 0.5%
ATR_MULT = 1.5               # 通道: close ± 1.5×ATR14


# ────────────────────────────────────────────────────────────
# 自选股配置
# ────────────────────────────────────────────────────────────
def load_watchlist() -> tuple[list[str], list[str]]:
    """--watch/--index 缺省配置：读 config/watchlist.json，不存在则建模板。

    Returns: (个股代码列表, 指数名称列表)
    """
    if not CONFIG_PATH.exists():
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(WATCHLIST_TEMPLATE, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        print(f"[提示] 已创建自选股配置模板: {CONFIG_PATH}")
        return (list(WATCHLIST_TEMPLATE["watch"]), list(WATCHLIST_TEMPLATE["index"]))
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        watch = data.get("watch") or []
        codes = [str(c).strip().zfill(6) for c in watch if str(c).strip()]
        indexes = [str(i).strip() for i in (data.get("index") or []) if str(i).strip()]
        return codes, indexes
    except Exception as e:
        print(f"[错误] 读取 {CONFIG_PATH} 失败: {e}")
        sys.exit(1)


# ────────────────────────────────────────────────────────────
# 数据读取
# ────────────────────────────────────────────────────────────
def load_kline(code: str, target: pd.Timestamp, window_start: pd.Timestamp
               ) -> tuple[pd.DataFrame | None, str]:
    """自选股日K（date/open/high/low/close/volume），截断至 target。"""
    path = KLINE_DIR / f"{code}.parquet"
    if not path.exists():
        return None, "kline文件缺失"
    try:
        df = read_kline_window(path, ["date", "open", "high", "low", "close", "volume"],
                               pd.Timestamp(window_start))
    except Exception as e:
        return None, f"读取失败: {str(e)[:60]}"
    if df is None or df.empty:
        return None, "无数据"
    if not {"date", "close"}.issubset(df.columns):
        return None, "缺关键列"
    for c in ("open", "high", "low", "volume"):
        if c not in df.columns:
            df[c] = np.nan
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).drop_duplicates("date").sort_values("date")
    df = df[df["date"] <= target]
    if df.empty or not (df["date"] == target).any():
        return None, "目标日无交易"
    return df, ""


# ────────────────────────────────────────────────────────────
# 卡片指标
# ────────────────────────────────────────────────────────────
def _prior_volume_mean(vol: pd.Series) -> float:
    """前 20 日均量（量纲鲁棒）：kline volume 单位按股票/日期段不一致，
    从最近一日向前回溯累计同量纲天数（相邻日量比 >10x 视为量纲断裂停止），
    正常数据下即标准 20 日均量。"""
    prior = vol.iloc[:-1]
    n = len(prior)
    use = min(VOL_N, n)
    if use < 1:
        return np.nan
    vals = prior.iloc[-use:].astype(float).to_numpy()
    vt = float(vol.iloc[-1]) if np.isfinite(vol.iloc[-1]) else np.nan
    if not np.isfinite(vt):
        return float(np.mean(vals))
    acc = [float(vals[-1])]
    for v in reversed(vals[:-1]):
        v = float(v)
        if acc[-1] <= 0 or v <= 0:
            acc.append(v)
            continue
        r = max(v, acc[-1]) / min(v, acc[-1])
        if r > 10:
            break
        acc.append(v)
    return float(np.mean(acc))


def day_stats(df: pd.DataFrame) -> dict:
    """当日涨跌幅 / 收盘 / 量比（量/20日均量）。"""
    close = df["close"].astype(float)
    vol = df["volume"].astype(float)
    close_t = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) >= 2 else np.nan
    pct = (close_t / prev - 1.0) * 100.0 if np.isfinite(prev) and prev > 0 else np.nan
    vol_t = float(vol.iloc[-1]) if np.isfinite(vol.iloc[-1]) else np.nan
    ma20_vol = _prior_volume_mean(vol)
    vr = vol_t / ma20_vol if np.isfinite(vol_t) and np.isfinite(ma20_vol) and ma20_vol > 0 else np.nan
    return {
        "close": close_t,
        "pct_chg": pct,
        "volume_ratio": vr,
        "open": float(df["open"].astype(float).iloc[-1]),
        "high": float(df["high"].astype(float).iloc[-1]),
        "low": float(df["low"].astype(float).iloc[-1]),
    }


def wick_flags(o: float, h: float, l: float, c: float) -> list[str]:
    """影线/极值形态标记（intraday skill）。"""
    flags: list[str] = []
    if c > 0 and (h - c) / c > WICK_UPPER:
        flags.append("长上影")
    if l > 0 and (c - l) / l > WICK_LOWER:
        flags.append("长下影")
    if c > 0 and (h - l) / c < DOJI_RANGE and abs(c - o) / c < DOJI_BODY:
        flags.append("十字星")
    return flags


def atr14(df: pd.DataFrame) -> float:
    """Wilder ATR(14)（TR 首段 14 均值作种子（含第一根），后递归平滑）。"""
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    tr = tr.dropna()
    if len(tr) < ATR_N:
        return np.nan
    atr = float(tr.iloc[0:ATR_N].mean())
    for v in tr.iloc[ATR_N:].to_numpy():
        atr = (atr * (ATR_N - 1) + float(v)) / ATR_N
    return atr


def new_highs(close: pd.Series) -> dict[int, bool | None]:
    """60/120/250 日新高（收盘 > 此前 N 日收盘最大值；样本不足返回 None）。"""
    close = close.astype(float)
    n = len(close)
    close_t = float(close.iloc[-1])
    out: dict[int, bool | None] = {}
    for d in NEW_HIGH_DAYS:
        if n >= d + 1:
            out[d] = bool(close_t > float(close.iloc[-(d + 1):-1].max()))
        else:
            out[d] = None
    return out


# ────────────────────────────────────────────────────────────
# 卡片组装与输出
# ────────────────────────────────────────────────────────────
def build_card(code: str, target: pd.Timestamp, bench: pd.Series) -> dict:
    df, err = load_kline(code, target, target - pd.Timedelta(days=LOOKBACK_DAYS))
    if df is None:
        return {"code": code, "name": code, "missing": True, "error": err}
    ds = day_stats(df)
    close = df.set_index("date")["close"].astype(float)
    wicks = wick_flags(ds["open"], ds["high"], ds["low"], ds["close"])
    atr = atr14(df)
    if np.isfinite(atr):
        support = ds["close"] - ATR_MULT * atr
        resistance = ds["close"] + ATR_MULT * atr
        if ds["close"] < support:
            hint = "破支撑↓ 减仓提示"
        elif ds["close"] > resistance:
            hint = "破阻力↑ 加仓提示"
        else:
            hint = "区间内 观望"
        touch = []
        if ds["low"] <= support:
            touch.append("盘中触支撑")
        if ds["high"] >= resistance:
            touch.append("盘中触阻力")
        if touch:
            hint += "（" + "，".join(touch) + "）"
    else:
        support = resistance = np.nan
        hint = "ATR不足"
    nh = new_highs(close)
    rs = compute_rs_metrics(close, bench)
    volume_profile = calc_volume_profile(df)
    try:
        fund_flow = check_divergence(code, target)
    except Exception:
        fund_flow = {"code": code, "pct_chg": None, "main_net": None,
                     "signal": "unavailable", "note": "接口失败"}
    return {
        "code": code,
        "name": code,
        "date": target,
        "close": ds["close"],
        "pct_chg": ds["pct_chg"],
        "volume_ratio": ds["volume_ratio"],
        "wicks": wicks,
        "atr": atr,
        "support": support,
        "resistance": resistance,
        "stop_loss": ds["close"] - ATR_MULT * atr if np.isfinite(atr) else np.nan,
        "hint": hint,
        "new_highs": nh,
        "rs": rs,
        "volume_profile": volume_profile,
        "fund_flow": fund_flow,
        "missing": False,
        "error": "",
    }


def load_index(name: str, target: pd.Timestamp) -> tuple[pd.DataFrame | None, str]:
    """读取指数/ETF 日线（date/open/high/low/close/volume），截断至 target。

    - 红利指数/科创50/创业板指 -> data_warehouse/market/index_daily_{name}.parquet
    - 中概互联               -> data_warehouse/kline/513050.parquet
    """
    if name in INDEX_KLINE_ALIAS:
        path = KLINE_DIR / f"{INDEX_KLINE_ALIAS[name]}.parquet"
    elif name in INDEX_MARKET_NAMES:
        path = MARKET_DIR / f"index_daily_{name}.parquet"
    else:
        return None, f"未知指数: {name}"
    if not path.exists():
        return None, f"数据缺失（{path.name}）"
    try:
        df = pd.read_parquet(path, columns=["date", "open", "high", "low", "close", "volume"])
    except Exception as e:
        return None, f"读取失败: {str(e)[:60]}"
    if df is None or df.empty:
        return None, "无数据"
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).drop_duplicates("date").sort_values("date")
    df = df[df["date"] <= target]
    if df.empty or not (df["date"] == target).any():
        return None, "目标日无交易"
    return df, ""


def build_index_card(name: str, target: pd.Timestamp) -> dict:
    """指数观察卡片：当日涨跌/5日涨跌/MA20位置/距MA250(%)/60日新高（不输出RS/密集区/资金流）。"""
    df, err = load_index(name, target)
    if df is None:
        return {"name": name, "missing": True, "error": err}
    close = df.set_index("date")["close"].astype(float)
    close_t = float(close.iloc[-1])
    pct = (close_t / float(close.iloc[-2]) - 1.0) * 100.0 if len(close) >= 2 else np.nan
    pct5 = (close_t / float(close.iloc[-6]) - 1.0) * 100.0 if len(close) >= 6 else np.nan
    ma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else np.nan
    ma250 = float(close.rolling(250).mean().iloc[-1]) if len(close) >= 250 else np.nan
    if np.isfinite(ma20):
        ma20_pos = "上" if close_t > ma20 else ("下" if close_t < ma20 else "平")
    else:
        ma20_pos = "—"
    dist250 = (close_t / ma250 - 1.0) * 100.0 if np.isfinite(ma250) and ma250 > 0 else np.nan
    if len(close) >= 61:
        nh60 = "是" if close_t > float(close.iloc[-61:-1].max()) else "否"
    else:
        nh60 = "样本不足"
    return {
        "name": name,
        "date": target,
        "close": close_t,
        "pct_chg": pct,
        "pct_chg_5": pct5,
        "ma20_pos": ma20_pos,
        "ma250_dist": dist250,
        "nh60": nh60,
        "missing": False,
        "error": "",
    }


def _fmt_pct(v: float) -> str:
    return format_pct_value(v, unit="pct")


def _fmt_num(v: float, nd: int = 2) -> str:
    return f"{v:.{nd}f}" if np.isfinite(v) else "—"


def _fmt_nh(v: bool | None) -> str:
    if v is None:
        return "样本不足"
    return "是" if v else "否"


def _fmt_vp(vp: dict) -> str:
    """密集区行文本：POC / 上沿 / 下沿 / 当前价相对位置。"""
    if vp.get("poc") is None:
        note = vp.get("note") or "样本不足"
        return f"{note}（{vp.get('n', 0)}日）"
    return (f"POC {vp['poc']:.2f} | 上沿 {vp['upper']:.2f} | 下沿 {vp['lower']:.2f} | "
            f"位置 {fmt_pos(vp)}")


def _fmt_ff(ff: dict) -> str:
    """资金流行文本：主力净额（亿）/ 信号；接口失败显示 n/a。"""
    if ff.get("signal") == "unavailable" or ff.get("main_net") is None:
        return f"n/a（{ff.get('note') or '接口失败'}）"
    return f"主力净额 {ff['main_net'] / 1e8:+.2f}亿 | {ff['signal']}"


def _fmt_stop(card: dict) -> str:
    """ATR 止损位行：止损=现价-1.5×ATR14（挂单价），并列筹码支撑/阻力
    （volume_profile 数据可用时）。无数据项省略。"""
    parts: list[str] = []
    stop = card.get("stop_loss")
    if stop is not None and np.isfinite(stop):
        parts.append(f"止损 {stop:.2f}（1.5×ATR14）")
    vp = card.get("volume_profile") or {}
    if vp.get("poc") is not None and vp.get("lower") is not None and vp.get("upper") is not None:
        parts.append(f"支撑 {vp['lower']:.2f} 阻力 {vp['upper']:.2f}")
    return " | ".join(parts)


def render_index_text(ic: dict) -> list[str]:
    if ic.get("missing"):
        return [f"[指数 {ic['name']}] {ic['error']}"]
    return [
        f"[指数 {ic['name']}] {ic['date']:%Y-%m-%d}",
        f"  当日: 收盘 {ic['close']:.2f}（{_fmt_pct(ic['pct_chg'])}） | "
        f"5日 {_fmt_pct(ic['pct_chg_5'])}",
        f"  均线: MA20 {'上方' if ic['ma20_pos'] == '上' else ('下方' if ic['ma20_pos'] == '下' else '持平')} "
        f"| 距年线MA250 {_fmt_pct(ic['ma250_dist'])} | 60日新高 {ic['nh60']}",
    ]


def render_card_text(card: dict, names: dict[str, str]) -> list[str]:
    name = names.get(card["code"], card["code"])
    if card.get("missing"):
        return [f"[{card['code']} {name}] {card['error']}"]
    nh = card["new_highs"]
    rs = card["rs"]
    wick = "、".join(card["wicks"]) if card["wicks"] else "—"
    lines = [
        f"[{card['code']} {name}] {card['date']:%Y-%m-%d}",
        f"  当日: 收盘 {card['close']:.2f}（{_fmt_pct(card['pct_chg'])}） | "
        f"量比(量/20日均量) {_fmt_num(card['volume_ratio'])}",
        f"  影线: {wick}",
        f"  密集区: {_fmt_vp(card['volume_profile'])}",
        f"  大单背离: {_fmt_ff(card['fund_flow'])}",
        f"  ATR14 {_fmt_num(card['atr'])} | 支撑 {_fmt_num(card['support'])} | "
        f"阻力 {_fmt_num(card['resistance'])} | {card['hint']}",
    ]
    stop_line = _fmt_stop(card)
    if stop_line:
        lines.append(f"  {stop_line}")
    lines += [
        f"  新高: 60日 {_fmt_nh(nh[60])} | 120日 {_fmt_nh(nh[120])} | 250日 {_fmt_nh(nh[250])}",
        f"  RS {_fmt_num(rs['rs'], 4)} | MA20 {_fmt_num(rs['rs_ma20'], 4)} | "
        f"斜率 {_fmt_num(rs['rs_slope'], 3)}%/日 | 60日新高 {'是' if rs['rs_60d_high'] else '否'} "
        f"| 评级 {rs['rating']}",
    ]
    return lines


def render_markdown(index_cards: list[dict], cards: list[dict], target: pd.Timestamp,
                    names: dict[str, str], bench_note: str) -> str:
    lines = [
        f"# 自选股复盘卡片 — {target:%Y-%m-%d}",
        "",
        f"- 指数观察: {len(index_cards)} 个 | 个股: {len(cards)} 只 | RS 基准: {bench_note}",
        "- 数据: 全部基于日K（date/open/high/low/close/volume）",
        "",
    ]
    if index_cards:
        lines += ["## 指数观察（市场环境）", ""]
        for ic in index_cards:
            lines.append(f"### {ic['name']}")
            if ic.get("missing"):
                lines += ["", f"{ic['error']}", ""]
                continue
            lines += [
                "",
                "| 项目 | 值 |",
                "|---|---|",
                f"| 收盘 / 当日涨跌 | {ic['close']:.2f} / {_fmt_pct(ic['pct_chg'])} |",
                f"| 5日涨跌 | {_fmt_pct(ic['pct_chg_5'])} |",
                f"| MA20位置 | {ic['ma20_pos']} |",
                f"| 距年线MA250 | {_fmt_pct(ic['ma250_dist'])} |",
                f"| 60日新高 | {ic['nh60']} |",
                "",
            ]
        lines += ["---", ""]
    for card in cards:
        name = names.get(card["code"], card["code"])
        lines.append(f"## {card['code']} {name}")
        if card.get("missing"):
            lines += ["", f"{card['error']}", ""]
            continue
        nh = card["new_highs"]
        rs = card["rs"]
        wick = "、".join(card["wicks"]) if card["wicks"] else "—"
        stop_line = _fmt_stop(card)
        lines += [
            "",
            "| 项目 | 值 |",
            "|---|---|",
            f"| 收盘 / 当日涨跌 | {card['close']:.2f} / {_fmt_pct(card['pct_chg'])} |",
            f"| 量比（量/20日均量） | {_fmt_num(card['volume_ratio'])} |",
            f"| 影线标记 | {wick} |",
            f"| 密集区（POC/上沿/下沿/位置） | {_fmt_vp(card['volume_profile'])} |",
            f"| 大单背离（主力净额/信号） | {_fmt_ff(card['fund_flow'])} |",
            f"| ATR14 | {_fmt_num(card['atr'])} |",
            *([f"| 止损位（现价-{ATR_MULT}×ATR14 挂单价） | {stop_line} |"]
              if stop_line else []),
            f"| 支撑（close-1.5×ATR） | {_fmt_num(card['support'])} |",
            f"| 阻力（close+1.5×ATR） | {_fmt_num(card['resistance'])} |",
            f"| 通道提示 | {card['hint']} |",
            f"| 60日新高 | {_fmt_nh(nh[60])} |",
            f"| 120日新高 | {_fmt_nh(nh[120])} |",
            f"| 250日新高 | {_fmt_nh(nh[250])} |",
            f"| RS | {_fmt_num(rs['rs'], 4)} |",
            f"| RS_MA20 | {_fmt_num(rs['rs_ma20'], 4)} |",
            f"| RS_slope（%/日） | {format_pct_value(rs['rs_slope'], digits=3, unit='pct')} |",
            f"| RS 60日新高 | {'是' if rs['rs_60d_high'] else '否'} |",
            f"| RS 评级 | {rs['rating']} |",
            "",
        ]
    lines += ["---",
              "*方法: 长上影=(high-close)/close>3%; 长下影=(close-low)/low>3%; "
              "十字星=(high-low)/close<1.5%且实体<0.5%; ATR(14)=Wilder 平滑; "
              "新高=收盘>此前N日收盘最大值; RS=收盘/沪深300同日期对齐, "
              "RS_MA20 20日均线, RS_slope=MA20最近5日线性回归斜率归一化%/日; "
              "密集区=60日价格-量分布(POC/70%阈值, 0.5%格); "
              "资金流=东财按单笔金额分桶估算, 仅供参考; 指数观察="
              "当日/5日涨跌幅、MA20位置(上/下)、距年线MA250(%), 60日新高=收盘>此前60日收盘最大值。*"]
    return "\n".join(lines) + "\n"


def _latest_common_day(codes: list[str], bench: pd.Series | None) -> pd.Timestamp:
    """目标日：基准与自选股 kline 的最新共同交易日。"""
    bench_max = bench.index.max() if bench is not None else None
    watch_max = None
    for code in codes:
        p = KLINE_DIR / f"{code}.parquet"
        if not p.exists():
            continue
        try:
            d = pd.read_parquet(p, columns=["date"])
            m = pd.to_datetime(d["date"], errors="coerce").max()
            watch_max = m if watch_max is None else max(watch_max, m)
        except Exception as e:
            logging.getLogger(__name__).error(f"[watch_card] 操作失败: {e}", exc_info=True)
            continue
    cands = [x for x in (bench_max, watch_max) if x is not None]
    return pd.Timestamp(min(cands)).normalize() if cands else pd.Timestamp.now().normalize()


def main() -> None:
    ap = argparse.ArgumentParser(description="自选股复盘卡片 + 指数观察（当日/影线/ATR/新高/RS/密集区/资金流）")
    ap.add_argument("--watch", default=None,
                    help="自选股代码，逗号分隔，如 600519,000001（缺省读 config/watchlist.json）")
    ap.add_argument("--index", default=None,
                    help="指数观察，逗号分隔，如 红利指数,科创50（缺省读 config/watchlist.json index 字段）")
    ap.add_argument("--date", default=None, help="分析日期 YYYY-MM-DD（默认最新共同交易日）")
    ap.add_argument("--no-report", action="store_true", help="不写 generated/watch_card_{date}.md")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="报告输出目录（默认仓库 generated/）")
    args = ap.parse_args()

    cfg_codes, cfg_indexes = load_watchlist()
    codes = ([c.strip().zfill(6) for c in args.watch.split(",") if c.strip()]
             if args.watch else cfg_codes)
    indexes = ([i.strip() for i in args.index.split(",") if i.strip()]
               if args.index else cfg_indexes)
    bad = [c for c in codes if not CODE_RE.match(c)]
    if bad:
        print(f"[错误] 非法代码: {','.join(bad)}（需6位数字）")
        sys.exit(1)
    if not codes:
        print("[错误] --watch 不能为空")
        sys.exit(1)

    bench, bench_note = load_benchmark()
    target = pd.Timestamp(args.date).normalize() if args.date else _latest_common_day(codes, bench)
    if bench is None:
        print("[错误] RS 基准（沪深300 index_daily.parquet）不可用")
        sys.exit(1)

    names, _ = load_names()
    t0 = time.time()
    index_cards = [build_index_card(i, target) for i in indexes]
    cards = [build_card(c, target, bench) for c in codes]
    elapsed = time.time() - t0

    print(f"[自选股复盘卡片] {target.date()} | 指数 {len(indexes)} 个 | 个股 {len(codes)} 只 "
          f"| 基准: {bench_note} | 耗时 {elapsed:.1f}s")
    for ic in index_cards:
        print()
        print("\n".join(render_index_text(ic)))
    for card in cards:
        print()
        print("\n".join(render_card_text(card, names)))

    if not args.no_report:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"watch_card_{target:%Y-%m-%d}.md"
        out.write_text(render_markdown(index_cards, cards, target, names, bench_note), encoding="utf-8")
        print(f"\n已保存: {out}")


if __name__ == "__main__":
    main()
